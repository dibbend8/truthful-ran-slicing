"""Puffer ``video_size`` chunk tables and the live-TV video tenant (exact advance information).

Source rows (one per encoded 2.002-s chunk): ``time (ns GMT), format, channel, video_ts, size`` (size in bytes).
A Puffer date ``D`` covers ``[D 11:00 UTC, D+1 11:00 UTC)`` (object ``D T11_D+1 T11/video_size_....csv``).

Preparation (:func:`prepare_date`): one streaming pass over the stored gzip object (decompressed in memory only,
md5/sha256 of the stored bytes computed in the same pass), strict validation (exact header, integer fields,
timestamps inside the date, strictly increasing time per (channel, format) series), then a compact chunk table
with the four formats present on every channel (``FORMATS``) and all channels is written to
``data/puffer/<date>.parquet`` (+ ``<date>.json`` summary). Ported from ``positive_controls/puffer_acquisition``
(``_header/_parse_value/day_bounds_ns/bucket_object``) and ``queued_cohort.stream_windows`` without the receipt
layers.

Video tenant (:func:`video_tenant`): V viewers; viewer v has a channel, a fixed format and a phase
``phi_v in [0, 2.002 s)``. Chunk j of its series (encode-report time ``t_j``) is paced uniformly over
``[t_j + lead + phi_v, t_j + lead + phi_v + 2.002 s)``; the bytes released in ``[a, b)`` are
``size * clip(b - s, 0, P) // P - size * clip(a - s, 0, P) // P`` in integer ns (``queued_work.QueueJob.bytes_in``),
so per-ms bytes are integers and pacing conserves every byte. The truthful report for interval ``k`` (interval
``[k, k+1)`` s from the window start) is formed at ``k - report_lead`` from the chunks with
``t_j <= k - report_lead`` only: their bytes in ``[k, k+1)`` plus, when ``lead < 1 s + report_lead``, a
persistence forecast of the not-yet-reported chunks (next chunk at ``max(t_last + P, cutoff + 1 ns)`` with the last
reported size). With ``lead >= 1.05 s`` the report equals the next-interval ingress to the byte.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import functools
import gzip
import hashlib
import base64
import os
import pathlib
import re
import zlib

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .. import io as sio

NS = 1_000_000_000
MS = 1_000_000
PACING_NS = 2_002_000_000
VIDEO_TS_STEP = 180_180
TIME = "time (ns GMT)"
SCHEMA = (TIME, "format", "channel", "video_ts", "size")
CHANNELS = ("abc", "cbs", "cw", "fox", "nbc", "pbs", "univision")
#: Formats present on all seven channels, and the fixed viewer format mix (INTERFACES §8).
FORMATS = ("1280x720-20", "1280x720-24", "854x480-24", "640x360-24")
FORMAT_MIX = np.array([0.2, 0.4, 0.3, 0.1])
CACHE_DIR = sio.DATA_DIR / "puffer"
#: Directory for fresh TEST downloads (written only by puffer_fetch.py after the pre-registration freeze).
TEST_RAW_DIR = sio.DATA_DIR / "puffer_raw"

_PC = sio.ICC_DIR / "ICC_rebuild_2026-10-10" / "positive_controls"
#: Local raw sources (read-only). The pilot directory holds 2024-01-01 (see its request.json URL).
_LOCAL_DIRS = (_PC / "cohort_train_terminal_20261012", _PC / "cohort_development_20261012",
               _PC / "cohort_test_20261012", _PC / "reviewer_replication_20261014" / "sources",
               _PC / "reviewer_followup_20261014" / "sources")
_PILOT = ("2024-01-01", _PC / "puffer_pilot_raw_v3_20261012" / "video_size.csv.gz")

#: Fixed splits. TRAIN: the 15 local dates 2024-01-01..02-25; DEV: the 21 local dates 2024-02-28..05-06.
TRAIN_DATES = ("2024-01-01", "2024-01-08", "2024-01-14", "2024-01-15", "2024-01-21", "2024-01-22", "2024-01-28",
               "2024-01-29", "2024-02-04", "2024-02-05", "2024-02-11", "2024-02-12", "2024-02-18", "2024-02-19",
               "2024-02-25")
DEV_DATES = ("2024-02-28", "2024-03-02", "2024-03-04", "2024-03-06", "2024-03-09", "2024-03-10", "2024-03-13",
             "2024-03-16", "2024-03-18", "2024-03-20", "2024-03-23", "2024-03-24", "2024-03-27", "2024-03-30",
             "2024-04-01", "2024-04-03", "2024-04-06", "2024-04-07", "2024-04-15", "2024-04-21", "2024-05-06")


def test_dates(extended: bool = False) -> tuple[str, ...]:
    """Pre-registered fresh TEST dates: Sundays and Mondays 2024-09-01..2024-10-21 (16 dates).

    ``extended=True`` adds the power-triggered extension (Sundays and Mondays 2024-10-27..2024-11-18, 24 in total).
    """
    a, b = _dt.date(2024, 9, 1), (_dt.date(2024, 11, 18) if extended else _dt.date(2024, 10, 21))
    out = []
    d = a
    while d <= b:
        if d.weekday() in (6, 0):  # Sunday, Monday
            out.append(d.isoformat())
        d += _dt.timedelta(days=1)
    return tuple(out)


def effective_test_dates() -> tuple[str, ...]:
    """TEST dates after the fetch: the pre-registered list (extended if the fetch manifest says so) with the
    recorded next-calendar-day replacements applied. Before any fetch: ``test_dates()``."""
    manifest = TEST_RAW_DIR / "manifest.json"
    if not manifest.exists():
        return test_dates()
    m = sio.read_json(manifest)
    repl = m.get("replacements", {})
    return tuple(repl.get(d, d) for d in test_dates(bool(m.get("extended", False))))


def split_dates(split: str) -> tuple[str, ...]:
    """Dates of a split ('train'|'dev'|'test'; TEST = :func:`effective_test_dates`)."""
    if split == "train":
        return TRAIN_DATES
    if split == "dev":
        return DEV_DATES
    if split == "test":
        return effective_test_dates()
    raise ValueError("split must be 'train', 'dev' or 'test'")


def split_of_date(day: str) -> str | None:
    """Split of a Puffer date (TEST includes the 24-date extension and recorded replacements), or None."""
    if day in TRAIN_DATES:
        return "train"
    if day in DEV_DATES:
        return "dev"
    if day in test_dates(True) or day in effective_test_dates():
        return "test"
    return None


# ----------------------------------------------------------------------------- naming (ported)
def date_group(day: str) -> str:
    """Puffer object group of an explicit UTC date, e.g. '2024-01-01T11_2024-01-02T11'."""
    if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("explicit YYYY-MM-DD date required")
    first = _dt.date.fromisoformat(day)
    return f"{first.isoformat()}T11_{(first + _dt.timedelta(days=1)).isoformat()}T11"


def day_bounds_ns(day: str) -> tuple[int, int]:
    """[start, end) of a Puffer date in ns since the epoch (11:00 UTC to 11:00 UTC next day)."""
    date_group(day)
    days = (_dt.date.fromisoformat(day) - _dt.date(1970, 1, 1)).days
    start = (days * 86_400 + 11 * 3_600) * NS
    return start, start + 86_400 * NS


def bucket_object(day: str, measurement: str = "video_size") -> str:
    """Object name inside bucket ``puffer-data-release`` (naming convention; existence not asserted)."""
    group = date_group(day)
    return f"{group}/{measurement}_{group}.csv"


def local_source(day: str) -> pathlib.Path:
    """Path of the stored ``video_size.csv.gz`` of a date (local TRAIN/DEV files or fetched TEST files)."""
    if day == _PILOT[0]:
        return _PILOT[1]
    for d in _LOCAL_DIRS + (TEST_RAW_DIR,):
        p = d / day / "video_size.csv.gz"
        if p.exists():
            return p
    raise FileNotFoundError(f"no local video_size source for {day}")


# ----------------------------------------------------------------------------- one-pass preparation
class _HashingReader:
    """File wrapper hashing every stored byte read through it (md5 + sha256)."""

    def __init__(self, raw):
        self.raw, self.md5, self.sha = raw, hashlib.md5(), hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        b = self.raw.read(size)
        self.md5.update(b); self.sha.update(b)
        self.count += len(b)
        return b

    def readable(self):
        return True


def parse_video_size(source: pathlib.Path, day: str) -> tuple[pd.DataFrame, dict]:
    """Read and validate one stored ``video_size`` object in a single streaming pass.

    Returns the full validated table (all channels/formats; columns channel, format, time_ns, video_ts, size) and an
    info dict with row counts and the stored-byte md5 (base64) / sha256. Raises ValueError on any violation.
    """
    start, end = day_bounds_ns(day)
    with open(source, "rb") as raw:
        hr = _HashingReader(raw)
        gz = gzip.GzipFile(fileobj=hr, mode="rb")
        types = {TIME: pa.int64(), "format": pa.string(), "channel": pa.string(), "video_ts": pa.int64(),
                 "size": pa.int64()}
        tb = pacsv.read_csv(gz, read_options=pacsv.ReadOptions(use_threads=True, block_size=1 << 24),
                            convert_options=pacsv.ConvertOptions(column_types=types, strings_can_be_null=False))
        if gz.read(1):
            raise ValueError("bytes after gzip EOF")
        while hr.read(1 << 20):   # hash any stored trailer bytes
            pass
    if tuple(tb.column_names) != SCHEMA:
        raise ValueError(f"unexpected header {tb.column_names}; need exactly {SCHEMA}")
    if any(tb.column(c).null_count for c in SCHEMA):
        raise ValueError("missing values in video_size")
    d = pd.DataFrame({"channel": tb.column("channel").to_numpy(zero_copy_only=False),
                      "format": tb.column("format").to_numpy(zero_copy_only=False),
                      "time_ns": tb.column(TIME).to_numpy(), "video_ts": tb.column("video_ts").to_numpy(),
                      "size": tb.column("size").to_numpy()})
    if len(d) == 0:
        raise ValueError("empty video_size object")
    if (d.time_ns < start).any() or (d.time_ns >= end).any():
        raise ValueError("timestamp outside the Puffer date [11:00 UTC, +24 h)")
    if (d.video_ts < 0).any() or (d["size"] < 0).any():
        raise ValueError("negative video_ts or size")
    fmt_ok = d.format.drop_duplicates().map(lambda s: bool(re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*-[0-9]+", s)))
    if not fmt_ok.all():
        raise ValueError("format must be WxH-CRF")
    ch_ok = d.channel.drop_duplicates().map(lambda s: bool(s) and s == s.strip() and s.isprintable())
    if not ch_ok.all():
        raise ValueError("invalid channel name")
    step = d.groupby(["channel", "format"], sort=False).time_ns.diff()
    if (step.dropna() <= 0).any():
        raise ValueError("time duplicates or reverses within a (channel, format) series")
    info = dict(date=day, source=str(source), rows=len(d), series=int(d.groupby(["channel", "format"]).ngroups),
                stored_bytes=hr.count, md5_b64=base64.b64encode(hr.md5.digest()).decode(), sha256=hr.sha.hexdigest(),
                zero_size_rows=int((d["size"] == 0).sum()))
    return d, info


def _recorded_hashes(source: pathlib.Path) -> dict:
    """Hashes recorded next to a stored object: GCS ``md5Hash`` (object_metadata.json) and/or the download
    ``sha256`` (acquisition.json of the pilot object). Empty dict if none is recorded."""
    out = {}
    meta = source.parent / "object_metadata.json"
    if meta.exists() and sio.read_json(meta).get("md5Hash"):
        out["md5_b64"] = sio.read_json(meta)["md5Hash"]
    acq = source.parent / "acquisition.json"
    if acq.exists() and isinstance(sio.read_json(acq).get("download_sha256"), str):
        out["sha256"] = sio.read_json(acq)["download_sha256"]
    return out


class InvalidVideoSize(ValueError):
    """A stored ``video_size`` object cannot be decompressed or parsed, fails a validation of
    :func:`parse_video_size`, differs from its recorded hashes, or has no channel of :data:`CHANNELS` with all four
    :data:`FORMATS`. ``puffer_fetch.fetch_date`` maps it to ``CorruptObject`` (pre-registered corrupt date)."""


#: Exceptions that mean the stored bytes are unusable (gzip framing -> OSError/EOFError/zlib.error; CSV and value
#: errors -> pyarrow.ArrowInvalid (a ValueError) or ValueError). Disk and permission errors while WRITING the cache
#: are not in this list's scope: they are raised outside the parse step and propagate unchanged.
_CONTENT_ERRORS = (ValueError, OSError, EOFError, zlib.error, pa.ArrowException)


def cache_paths(day: str) -> tuple[pathlib.Path, pathlib.Path]:
    """``(parquet, json summary)`` of a prepared date. The parquet is written LAST, so it marks a complete cache."""
    return CACHE_DIR / f"{day}.parquet", CACHE_DIR / f"{day}.json"


def _parse_checked(source: pathlib.Path, day: str) -> tuple[pd.DataFrame, dict]:
    """:func:`parse_video_size` plus the recorded-hash check; every content failure becomes InvalidVideoSize."""
    try:
        d, info = parse_video_size(source, day)
    except InvalidVideoSize:
        raise
    except _CONTENT_ERRORS as exc:
        raise InvalidVideoSize(f"{day}: {source} cannot be decompressed, parsed or validated: "
                               f"{type(exc).__name__}: {exc}") from exc
    recorded = _recorded_hashes(source)
    for key, value in recorded.items():
        if info[key] != value:
            raise InvalidVideoSize(f"{day}: stored-object {key} {info[key]} != recorded {value}")
    info["verified_hashes"] = sorted(recorded)
    return d, info


def prepare_date(day: str, force: bool = False) -> pathlib.Path:
    """Write the compact chunk table ``data/puffer/<day>.parquet`` (4 common formats, all channels) once.

    Safe under concurrency (pre-freeze audit F2: on TEST, ~8 workers per date used to parse the same object at once
    and crash on the shared temp names): an exclusive ``fcntl.flock`` on ``data/puffer/<day>.lock`` serialises the
    writers of a date, the cache is re-checked inside the lock (the first writer builds it, the others return it),
    both files are written through process-unique temp names and ``os.replace``, and the summary json is written
    before the parquet, so an existing parquet always has its summary. Raises :class:`InvalidVideoSize` when the
    stored object is unusable (nothing is written then).
    """
    out, summary = cache_paths(day)
    if out.exists() and summary.exists() and not force:
        return out
    sio.confine(out).parent.mkdir(parents=True, exist_ok=True)
    with open(sio.confine(CACHE_DIR / f"{day}.lock"), "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if out.exists() and summary.exists() and not force:     # built by another process while we waited
                return out
            _build_date(day, out, summary)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return out


def _build_date(day: str, out: pathlib.Path, summary: pathlib.Path) -> None:
    sio.require_free_gb(3)
    source = local_source(day)
    d, info = _parse_checked(source, day)
    keep = d[d.format.isin(FORMATS)].copy()
    present = sorted(keep.channel.unique())
    complete = [c for c in present if set(keep.format[keep.channel == c]) == set(FORMATS)]
    if not set(complete) & set(CHANNELS):
        raise InvalidVideoSize(f"{day}: none of the channels {CHANNELS} has all four formats {FORMATS}")
    keep = keep[keep.channel.isin(complete)]
    if keep["size"].max() >= 2**31:
        raise OverflowError("chunk size exceeds int32")
    keep = keep.sort_values(["channel", "format", "time_ns"], kind="stable").reset_index(drop=True)
    keep["size"] = keep["size"].astype(np.int32)
    table = pa.Table.from_pandas(keep, preserve_index=False)
    table = table.cast(pa.schema([("channel", pa.dictionary(pa.int8(), pa.string())),
                                  ("format", pa.dictionary(pa.int8(), pa.string())),
                                  ("time_ns", pa.int64()), ("video_ts", pa.int64()), ("size", pa.int32())]))
    g = keep.groupby(["channel", "format"])
    rate = (g["size"].sum() * 8e-6 / 86_400.0)
    info.update(kept_rows=len(keep), channels=complete, channels_missing=[c for c in CHANNELS if c not in complete],
                extra_channels=[c for c in complete if c not in CHANNELS], formats=list(FORMATS),
                mean_mbps={f"{c}|{f}": float(v) for (c, f), v in rate.items()},
                chunks={f"{c}|{f}": int(v) for (c, f), v in g.size().items()})
    out.unlink(missing_ok=True)                  # a forced rebuild: no stale parquet next to the new summary
    sio.write_json(summary, info)
    tmp = sio.unique_tmp(out)
    try:
        pq.write_table(table, tmp, compression="zstd", use_dictionary=["channel", "format"],
                       column_encoding={"time_ns": "DELTA_BINARY_PACKED", "video_ts": "DELTA_BINARY_PACKED",
                                        "size": "PLAIN"})
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def remove_cache(day: str) -> None:
    """Delete a date's prepared cache (parquet first, then the summary); used when a fetched object is rejected."""
    out, summary = cache_paths(day)
    out.unlink(missing_ok=True)
    summary.unlink(missing_ok=True)
    load_chunks.cache_clear()


def date_info(day: str) -> dict:
    """Summary written by :func:`prepare_date` (prepares the date if needed)."""
    prepare_date(day)
    return sio.read_json(CACHE_DIR / f"{day}.json")


@functools.lru_cache(maxsize=6)
def load_chunks(day: str) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Chunk series of a date: {(channel, format): (time_ns int64 sorted, size_bytes int64)} (read-only arrays)."""
    t = pq.read_table(prepare_date(day)).to_pandas()
    out = {}
    for (c, f), g in t.groupby(["channel", "format"], sort=True, observed=True):
        tt = g.time_ns.to_numpy(np.int64).copy(); ss = g["size"].to_numpy(np.int64).copy()
        tt.flags.writeable = False; ss.flags.writeable = False
        out[(str(c), str(f))] = (tt, ss)
    return out


def channels_present(day: str) -> tuple[str, ...]:
    """Channels with all four formats on a date (sorted)."""
    return tuple(sorted({c for c, _ in load_chunks(day)}))


# ----------------------------------------------------------------------------- pacing (integer bytes)
@numba.njit(cache=False)
def _pace_ms(release_ns, size, pacing_ns, t0_ns, T_ms, out):
    for j in range(release_ns.shape[0]):
        s = release_ns[j] - t0_ns
        e = s + pacing_ns
        m0 = s // MS
        if m0 < 0:
            m0 = 0
        m1 = (e + MS - 1) // MS
        if m1 > T_ms:
            m1 = T_ms
        if m0 >= m1:
            continue
        sz = size[j]
        u = m0 * MS - s
        if u < 0:
            u = 0
        elif u > pacing_ns:
            u = pacing_ns
        prev = sz * u // pacing_ns
        for m in range(m0, m1):
            u = (m + 1) * MS - s
            if u < 0:
                u = 0
            elif u > pacing_ns:
                u = pacing_ns
            cur = sz * u // pacing_ns
            out[m] += cur - prev
            prev = cur


def pace_ms(release_ns: np.ndarray, size: np.ndarray, t0_ns: int, T_ms: int, pacing_ns: int = PACING_NS) -> np.ndarray:
    """Integer bytes per 1-ms bin of ``[t0_ns, t0_ns + T_ms ms)`` for chunks paced uniformly from ``release_ns``.

    Each chunk releases ``size * clip(t - release, 0, P) // P`` bytes by time t (ns); bins before t0 are dropped.
    """
    release_ns = np.ascontiguousarray(release_ns, np.int64)
    size = np.ascontiguousarray(size, np.int64)
    if release_ns.shape != size.shape or np.any(size < 0) or pacing_ns <= 0 or T_ms <= 0:
        raise ValueError("need aligned nonnegative sizes, positive pacing and horizon")
    if size.size and int(size.max()) * pacing_ns >= 2**62:
        raise OverflowError("size * pacing_ns overflows int64")
    out = np.zeros(T_ms, np.int64)
    _pace_ms(release_ns, size, np.int64(pacing_ns), np.int64(t0_ns), np.int64(T_ms), out)
    return out


def bytes_released(release_ns, size, a_ns, b_ns, pacing_ns: int = PACING_NS) -> np.ndarray:
    """Integer bytes of uniformly paced chunks released in ``[a_ns, b_ns)`` (broadcasting int64 arrays)."""
    release_ns, size = np.asarray(release_ns, np.int64), np.asarray(size, np.int64)

    def cum(t):
        return size * np.clip(np.asarray(t, np.int64) - release_ns, 0, pacing_ns) // pacing_ns

    return cum(b_ns) - cum(a_ns)


# ----------------------------------------------------------------------------- video tenant
def viewer_draws(n_viewers: int, channels: tuple[str, ...], streams: list[np.random.Generator]) -> list[dict]:
    """Channel (uniform over ``channels``), format (FORMAT_MIX) and phase (U[0, 2.002 s), integer ns) per viewer.

    ``streams[v]`` is viewer v's own generator, so the first V viewers do not depend on how many are drawn.
    """
    if len(streams) < n_viewers or not channels:
        raise ValueError("need one generator per viewer and at least one channel")
    cum = np.cumsum(FORMAT_MIX)
    out = []
    for v in range(n_viewers):
        u = streams[v].random(3)
        c = channels[min(int(u[0] * len(channels)), len(channels) - 1)]
        f = FORMATS[min(int(np.searchsorted(cum, u[1], side="right")), len(FORMATS) - 1)]
        out.append(dict(channel=c, format=f, phase_ns=int(u[2] * PACING_NS)))
    return out


def video_tenant(day: str, start_ns: int, K: int, viewers: list[dict], lead_ns: int,
                 report_lead_ns: int = 50 * MS, delta_ns: int = NS) -> dict:
    """Per-ms arrivals and truthful reports of one video tenant over intervals ``[start + k*Delta, +Delta)``.

    Returns dict with ``arrivals_bytes`` (K*1000,) int64 bytes per ms, ``interval_bytes`` (K,) int64 exact ingress,
    ``report_bytes`` (K,) int64 = bytes of interval k predicted at ``start + k*Delta - report_lead`` from reported
    chunks only, ``known_bytes`` (K,) the part from reported chunks, ``exact`` (bool: lead >= Delta + report_lead),
    and per-viewer chunk counts. Units: bytes; Delta = 1 s.
    """
    if delta_ns != NS:
        raise ValueError("the 1-ms arrival grid assumes Delta = 1 s")
    if lead_ns < 0 or report_lead_ns < 0:
        raise ValueError("lead and report lead must be nonnegative")
    chunks = load_chunks(day)
    d0, d1 = day_bounds_ns(day)
    end_ns = start_ns + K * delta_ns
    if start_ns - lead_ns - 2 * PACING_NS - report_lead_ns < d0 or end_ns > d1:
        raise ValueError("window (with pacing pre-roll) must lie inside the Puffer date")
    T_ms = K * 1000
    arrivals = np.zeros(T_ms, np.int64)
    known = np.zeros(K, np.int64)
    predicted = np.zeros(K, np.int64)
    a = start_ns + np.arange(K, dtype=np.int64) * delta_ns          # interval starts (ns)
    cutoff = a - report_lead_ns                                      # report formation times
    exact = lead_ns >= delta_ns + report_lead_ns
    used = []
    for v in viewers:
        key = (v["channel"], v["format"])
        if key not in chunks:
            raise KeyError(f"{day}: no series {key}")
        t, size = chunks[key]
        rel = t + lead_ns + v["phase_ns"]
        sel = (rel < end_ns) & (rel + PACING_NS > start_ns)
        tj, sj, rj = t[sel], size[sel], rel[sel]
        used.append(int(sel.sum()))
        arrivals += pace_ms(rj, sj, start_ns, T_ms)
        # known part: chunk j contributes to interval k iff t_j <= cutoff_k
        k0 = np.maximum((rj - start_ns) // delta_ns, 0)
        for off in range(4):                       # a 2.002-s pacing span touches at most 4 intervals
            k = k0 + off
            ok = k < K
            kk, tt, ss, rr = k[ok], tj[ok], sj[ok], rj[ok]
            c = bytes_released(rr, ss, a[kk], a[kk] + delta_ns)
            c = np.where(tt <= cutoff[kk], c, 0)
            np.add.at(known, kk, c)
        if not exact:
            # persistence forecast of chunks not yet reported at cutoff_k (same size as the last reported chunk)
            idx = np.searchsorted(t, cutoff, side="right") - 1
            has = idx >= 0
            t_last = np.where(has, t[np.maximum(idx, 0)], 0)
            s_last = np.where(has, size[np.maximum(idx, 0)], 0)
            t_next = np.maximum(t_last + PACING_NS, cutoff + 1)
            for m in range(3):
                r_hat = t_next + m * PACING_NS + lead_ns + v["phase_ns"]
                predicted += np.where(has, bytes_released(r_hat, s_last, a, a + delta_ns), 0)
    interval = arrivals.reshape(K, 1000).sum(1)
    report = known + predicted
    return dict(arrivals_bytes=arrivals, interval_bytes=interval, report_bytes=report, known_bytes=known,
                exact=bool(exact), chunks_used=used)


def viewer_window_mbps(windows, K: int, lead_ns: int, channels: tuple[str, ...] | None = None) -> float:
    """Expected mean rate (Mbit/s) of one viewer over a set of windows ``[(date, offset_s), ...]`` of K seconds:
    channel uniform over the group's channels present on the date, format by FORMAT_MIX; a series contributes the
    bytes of the chunks reported in ``[start - lead, start - lead + K)`` divided by K (the chunks paced into the
    window up to edge effects of one pacing span)."""
    vals = []
    for day, offset_s in windows:
        ch = load_chunks(day)
        d0, _ = day_bounds_ns(day)
        a = d0 + int(offset_s) * NS - lead_ns
        b = a + K * NS
        present = [c for c in channels_present(day) if channels is None or c in channels]
        if not present:
            raise ValueError(f"{day}: no channel of group {channels}")
        per_ch = []
        for c in present:
            r = 0.0
            for wgt, f in zip(FORMAT_MIX, FORMATS):
                t, size = ch[(c, f)]
                i0, i1 = np.searchsorted(t, [a, b])
                r += wgt * size[i0:i1].sum() * 8e-6 / K
            per_ch.append(r)
        vals.append(np.mean(per_ch))
    return float(np.mean(vals))


def viewer_mean_mbps(dates: tuple[str, ...], channels: tuple[str, ...] | None = None) -> float:
    """Expected mean rate (Mbit/s) of one viewer: channel uniform over the group's channels present on each date,
    format by FORMAT_MIX, averaged over ``dates`` (series rate = total bytes * 8e-6 / 86400 s)."""
    vals = []
    for d in dates:
        info = date_info(d)
        present = [c for c in info["channels"] if channels is None or c in channels]
        if not present:
            raise ValueError(f"{d}: no channel of group {channels}")
        per_ch = [sum(w * info["mean_mbps"][f"{c}|{f}"] for w, f in zip(FORMAT_MIX, FORMATS)) for c in present]
        vals.append(np.mean(per_ch))
    return float(np.mean(vals))
