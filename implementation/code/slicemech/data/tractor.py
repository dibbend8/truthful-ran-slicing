"""TRACTOR packet captures -> per-user 1-ms downlink byte series (URLLC-like and mMTC-like tenants).

Port of ``ICC_rebuild_2026-10-10/src/rebuild/data.read_tractor`` (L123-185) and Code B ``vslice/data.load_tractor``
(1-ms bins, idle trim), without the receipt layer.

* Files: ``ICC/data/raw/tractor/raw/{urll*,urllc*,mmtc*}.csv`` (columns ``App name,No.,Time,Source,Destination,
  Protocol,Length``). Downlink = rows with ``Source == 172.30.1.250``; each packet's ``Length`` (bytes) is added to
  the 1-ms bin ``floor((Time - t0) * 1000)`` where ``t0`` is the first timestamp of the file (any direction).
* Malformed rows (non-finite time, non-integer or negative length) raise ``ValueError``: missing packets are never
  silently read as silence.
* Idle trim: in the first campaign's URLLC file (``urll``), whole seconds belonging to all-zero runs longer than
  ``GAP_S`` = 5 s are removed. The TRACTOR README states that this campaign concatenated several capture periods and
  removed other-class traffic, leaving empty time; the URLLC files of every other campaign never show a zero run
  longer than 8 s. mMTC captures are naturally sparse (60-97 % zero seconds in every campaign), so their gaps are
  not artefacts and no mMTC file is trimmed.
* A *user* (``draw_user``) is a contiguous window of one trace at a uniformly random offset, the trace chosen with
  probability proportional to its length; if the window runs past the trace end it wraps circularly (flagged).
* Pool parts (round-1 review, finding 1; interleaved in round 2): ``pool(split, cls, part)`` restricts every trace of
  the pool to one of two disjoint, interleaved halves: the trace is cut into blocks of ``block_s`` = ``PART_BLOCK_S``
  = 2 s from its first millisecond, half ``'A'`` keeps the even blocks and half ``'B'`` the odd blocks, each
  concatenated in time order (stems ``<stem>#A`` / ``<stem>#B``). E5 draws the retraining windows of a date from one
  half and the evaluation windows from the other (``compose`` maps the E5 phase to a half per date), so the
  evaluation never replays a traffic-second the retraining saw.

  Why interleaved (round-2 review): the captures are not stationary. The round-1 time halves (first / second half of
  every capture) shifted the class loads systematically between the halves (TRAIN URLLC 0.72 / 1.28 of the pooled
  mean, DEV mMTC 0.06 / 1.94), and since every date draws from the same captures the shift had the same sign on
  every date. Alternating 2-s blocks sample the whole capture in both halves: on TRAIN and DEV every class mean of
  each half lies within 7 % of the pooled mean (:func:`part_balance`; ``experiments/diag_part_balance.py``). The
  block length is the smallest of {1, 2, 5, 10} s that passes that check on TRAIN and DEV (1 s fails DEV mMTC at
  +13.5 %). Costs: a user window crosses a splice every 2 s of half-time, and a half is half as long as its capture,
  so more user windows wrap (both are reported by :func:`part_balance`); the 1-s lag autocorrelation of URLLC
  traffic falls from 0.88 to 0.82 (TRAIN) and from 0.76 to about 0.70 (DEV).

Per-trace caches: ``data/tractor/<stem>.npz`` (int32 bytes per ms) + ``<stem>.json`` (info).
"""
from __future__ import annotations

import functools
import os
import pathlib

import numpy as np
import pandas as pd

from .. import io as sio

RAW_DIR = sio.ICC_DIR / "data" / "raw" / "tractor" / "raw"
CACHE_DIR = sio.DATA_DIR / "tractor"
DL_SOURCE = "172.30.1.250"
GAP_S = 5
#: Trace stems whose long idle runs are removed (see module docstring).
IDLE_TRIM_STEMS = frozenset({"urll"})

#: Fixed campaign splits (file stems) per class (INTERFACES §8).
SPLITS = {
    "train": {"urllc": ("urll", "urll_2", "urllc_3", "urll_11_18"),
              "mmtc": ("mmtc", "mmtc_2", "mmtc_3", "mmtc_11_18")},
    "dev": {"urllc": ("urllc_05_18", "urllc_06_12"),
            "mmtc": ("mmtc_05_18",)},
    "test": {"urllc": ("urllc_03_03", "urllc_04_10", "urllc_04_16"),
             "mmtc": ("mmtc_03_03", "mmtc_04_10")},
}
CLASSES = ("urllc", "mmtc")
#: Disjoint pool parts (interleaved halves of every trace: even / odd blocks); see the module docstring.
PARTS = ("A", "B")
#: Block length (s) of the interleaved pool parts.
PART_BLOCK_S = 2
#: Pre-registered part-balance tolerance: each class mean of each half within 10 % of the pooled mean (TRAIN, DEV).
PART_BALANCE_TOL = 0.10


def class_of_stem(stem: str) -> str:
    """'urllc' for urll*/urllc* stems, 'mmtc' for mmtc* stems; ValueError otherwise."""
    if stem.startswith("urll"):
        return "urllc"
    if stem.startswith("mmtc"):
        return "mmtc"
    raise ValueError(f"not a URLLC/mMTC TRACTOR trace: {stem}")


def split_of_stem(stem: str) -> str | None:
    """Split of a trace stem, or None when the stem is in no split."""
    for split, by_cls in SPLITS.items():
        if any(stem in stems for stems in by_cls.values()):
            return split
    return None


def trim_idle(bytes_ms: np.ndarray, gap_s: int = GAP_S) -> tuple[np.ndarray, int]:
    """Drop whole seconds inside all-zero runs longer than ``gap_s`` seconds (and the incomplete last second).

    Returns ``(trimmed per-ms array, removed seconds)``.
    """
    n = len(bytes_ms) // 1000
    sec = bytes_ms[: n * 1000].reshape(n, 1000)
    zero = sec.sum(1) == 0
    keep = np.ones(n, bool)
    i = 0
    while i < n:
        if zero[i]:
            j = i
            while j < n and zero[j]:
                j += 1
            if j - i > gap_s:
                keep[i:j] = False
            i = j
        else:
            i += 1
    return sec[keep].ravel(), int((~keep).sum())


def read_trace(path: pathlib.Path) -> tuple[np.ndarray, dict]:
    """Bin one TRACTOR CSV into downlink bytes per 1 ms (int64). Returns ``(bytes_ms, info)``."""
    path = pathlib.Path(path)
    stem = path.stem
    d = pd.read_csv(path, usecols=["Time", "Source", "Length"])
    t = pd.to_numeric(d["Time"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(t)):
        raise ValueError(f"{path.name}: {int((~np.isfinite(t)).sum())} rows with invalid timestamps")
    dl = (d["Source"] == DL_SOURCE).to_numpy()
    size = pd.to_numeric(d["Length"], errors="coerce").to_numpy(float)[dl]
    if not np.all(np.isfinite(size)) or np.any(size < 0) or np.any(size != np.floor(size)):
        raise ValueError(f"{path.name}: malformed downlink packet lengths")
    t0, t1 = t.min(), t.max()
    ms = np.floor((t[dl] - t0) * 1000.0).astype(np.int64)
    n_ms = int(np.floor((t1 - t0) * 1000.0)) + 1
    out = np.bincount(ms, weights=size, minlength=n_ms).astype(np.int64)
    if out.sum() != int(size.sum()):
        raise AssertionError("binning lost bytes")
    info = dict(stem=stem, cls=class_of_stem(stem), split=split_of_stem(stem), rows=len(d), dl_packets=int(dl.sum()),
                dl_bytes=int(size.sum()), elapsed_s=float(t1 - t0), origin_s=float(t0), trimmed_seconds=0)
    if stem in IDLE_TRIM_STEMS:
        out, removed = trim_idle(out)
        info["trimmed_seconds"] = removed
    info.update(length_ms=len(out), kept_bytes=int(out.sum()),
                mean_mbps=float(out.sum() * 8e-6 / (len(out) / 1000.0)),
                zero_second_fraction=float((out[: len(out) // 1000 * 1000].reshape(-1, 1000).sum(1) == 0).mean()))
    if out.sum() == 0:
        raise ValueError(f"{path.name}: no downlink traffic")
    return out, info


class Trace:
    """Per-ms downlink bytes of one capture: ``stem``, ``cls``, ``bytes_ms`` (int64), ``info``."""

    __slots__ = ("stem", "cls", "bytes_ms", "info")

    def __init__(self, stem, cls, bytes_ms, info):
        self.stem, self.cls, self.bytes_ms, self.info = stem, cls, bytes_ms, info

    def __len__(self):
        return len(self.bytes_ms)

    @property
    def mean_mbps(self) -> float:
        """Mean downlink rate (Mbit/s) over the whole (trimmed) trace."""
        return float(self.bytes_ms.sum() * 8e-6 / (len(self.bytes_ms) / 1000.0))


@functools.lru_cache(maxsize=None)
def load_trace(stem: str) -> Trace:
    """Load (building the cache on first use) the per-ms series of a trace stem."""
    cache = CACHE_DIR / f"{stem}.npz"
    meta = CACHE_DIR / f"{stem}.json"
    if cache.exists() and meta.exists():
        info = sio.read_json(meta)
        with np.load(cache) as z:
            b = z["bytes_ms"].astype(np.int64)
    else:
        src = RAW_DIR / f"{stem}.csv"
        b, info = read_trace(src)
        info["source_sha256"] = sio.file_sha256(src)
        if b.max() >= 2**31:
            raise OverflowError("per-ms bytes exceed int32 cache range")
        sio.confine(cache).parent.mkdir(parents=True, exist_ok=True)
        # pre-freeze audit F2: concurrent first users of a trace must never read a half-written cache, so both files
        # go through process-unique temp names and os.replace (the npz last: it marks a complete cache).
        sio.write_json(meta, info)
        tmp = sio.unique_tmp(cache)
        try:
            with open(tmp, "wb") as f:           # a file object: savez would append '.npz' to a temp *name*
                np.savez_compressed(f, bytes_ms=b.astype(np.int32))
            os.replace(tmp, cache)
        finally:
            tmp.unlink(missing_ok=True)
    b.flags.writeable = False
    return Trace(stem, class_of_stem(stem), b, info)


def capture_of(stem: str) -> str:
    """Capture (file stem) of a pool trace stem: ``'urllc_03_03#A'`` -> ``'urllc_03_03'``."""
    return str(stem).split("#", 1)[0]


def part_mask(length_ms: int, part: str, block_s: int = PART_BLOCK_S) -> np.ndarray:
    """Boolean mask over a trace's milliseconds: block ``j = floor(t_ms / (1000 block_s))``; 'A' keeps even j,
    'B' odd j."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    if isinstance(block_s, bool) or not isinstance(block_s, (int, np.integer)) or block_s < 1:
        raise ValueError(f"block_s must be an integer >= 1 s, got {block_s!r}")
    block = np.arange(int(length_ms)) // (1000 * int(block_s))
    return (block % 2) == PARTS.index(part)


@functools.lru_cache(maxsize=None)
def part_trace(stem: str, part: str, block_s: int = PART_BLOCK_S) -> Trace:
    """Interleaved half ``part`` of a trace (module docstring): the even ('A') or odd ('B') blocks of ``block_s``
    seconds, concatenated in time order, as a Trace with stem ``<stem>#<part>``. Raises if the trace has fewer than
    two blocks or the half carries no downlink traffic."""
    t = load_trace(stem)
    L = len(t)
    if L <= 1000 * int(block_s):
        raise ValueError(f"trace {stem} is too short to split into {block_s}-s blocks")
    sub = t.bytes_ms[part_mask(L, part, block_s)]
    if sub.sum() == 0:
        raise ValueError(f"part {part} of trace {stem} carries no downlink traffic")
    sub.flags.writeable = False
    info = dict(t.info, part=part, parent=stem, part_block_s=int(block_s), part_length_ms=int(len(sub)))
    return Trace(f"{stem}#{part}", t.cls, sub, info)


def pool(split: str, cls: str, part: str | None = None, block_s: int = PART_BLOCK_S) -> tuple[Trace, ...]:
    """The traces of class ``cls`` ('urllc'|'mmtc') in ``split``; with ``part`` ('A'|'B') the matching interleaved
    half of each trace (disjoint pools for the E5 retraining and evaluation windows)."""
    if split not in SPLITS or cls not in CLASSES:
        raise ValueError(f"unknown split/class {split}/{cls}")
    if part is None:
        return tuple(load_trace(s) for s in SPLITS[split][cls])
    return tuple(part_trace(s, part, int(block_s)) for s in SPLITS[split][cls])


def pool_stats(traces: tuple[Trace, ...], window_s: float) -> dict:
    """Per-user statistics of a pool under :func:`draw_user` (trace probability proportional to length, uniform
    offset): ``mean_mbps`` (expected user mean rate, Mbit/s) and ``wrap_prob`` (probability that a user window of
    ``window_s`` seconds wraps circularly, ``sum_i min(L_i, T) / sum_i L_i``)."""
    lens = np.array([len(t) for t in traces], float)
    T = 1000.0 * float(window_s)
    tot = float(sum(t.bytes_ms.sum() for t in traces))
    return dict(mean_mbps=tot * 8e-6 / (lens.sum() / 1000.0), wrap_prob=float(np.minimum(lens, T).sum() / lens.sum()),
                seconds=float(lens.sum() / 1000.0))


def part_balance(split: str, block_s: int = PART_BLOCK_S, window_s: float = 200.0, *, test_freeze=None) -> dict:
    """Pre-registered part-balance check (round-2 review) of one split's TRACTOR pools.

    For each class: the pooled per-user mean (Mbit/s), each half's mean and its ratio to the pooled mean, whether
    both ratios lie within ``1 +- PART_BALANCE_TOL``, and the probability that a user window of ``window_s`` seconds
    wraps (full pool and each half). TEST captures are refused unless ``test_freeze`` passes the freeze gate
    (``data.puffer_fetch.require_freeze_for_test_reads``): this is a TEST statistic.
    """
    if split == "test":
        if test_freeze is None:
            raise PermissionError("part_balance('test') is a TEST statistic: pass the frozen pre-registration")
        from .puffer_fetch import require_freeze_for_test_reads   # local import: no network module at load time
        require_freeze_for_test_reads(test_freeze)
    out = {}
    for cls in CLASSES:
        full = pool_stats(pool(split, cls), window_s)
        row = dict(pooled_mbps=full["mean_mbps"], wrap_prob=full["wrap_prob"], seconds=full["seconds"])
        for part in PARTS:
            st = pool_stats(pool(split, cls, part, block_s), window_s)
            row[f"{part}_mbps"] = st["mean_mbps"]
            row[f"{part}_ratio"] = st["mean_mbps"] / full["mean_mbps"]
            row[f"{part}_wrap_prob"] = st["wrap_prob"]
        row["balanced"] = bool(all(abs(row[f"{p}_ratio"] - 1.0) <= PART_BALANCE_TOL for p in PARTS))
        out[cls] = row
    return dict(split=split, block_s=int(block_s), window_s=float(window_s), tol=PART_BALANCE_TOL, classes=out,
                balanced=bool(all(r["balanced"] for r in out.values())))


def pool_mean_mbps(split: str, cls: str) -> float:
    """Expected per-user mean rate (Mbit/s) under ``draw_user``: total bytes / total duration of the pool."""
    ts = pool(split, cls)
    return float(sum(t.bytes_ms.sum() for t in ts) * 8e-6 / (sum(len(t) for t in ts) / 1000.0))


def draw_user(traces: tuple[Trace, ...], rng: np.random.Generator) -> tuple[int, int]:
    """Draw a (trace index, start offset in ms): trace with probability proportional to length, offset uniform."""
    w = np.array([len(t) for t in traces], float)
    i = int(rng.choice(len(traces), p=w / w.sum()))
    return i, int(rng.integers(len(traces[i])))


def user_slice(trace: Trace, offset_ms: int, length_ms: int) -> tuple[np.ndarray, bool]:
    """``length_ms`` bins of a trace from ``offset_ms``, wrapping circularly; returns ``(bytes_ms, wrapped)``."""
    L = len(trace)
    if not 0 <= offset_ms < L or length_ms <= 0:
        raise ValueError("offset must lie in the trace and length must be positive")
    if offset_ms + length_ms <= L:
        return trace.bytes_ms[offset_ms: offset_ms + length_ms], False
    return np.take(trace.bytes_ms, np.arange(offset_ms, offset_ms + length_ms), mode="wrap"), True


def users_sum(traces: tuple[Trace, ...], draws: list[tuple[int, int]], length_ms: int) -> tuple[np.ndarray, int]:
    """Sum of the drawn users' per-ms bytes (int64, length ``length_ms``); also returns the number wrapped."""
    total = np.zeros(length_ms, np.int64)
    wrapped = 0
    for i, off in draws:
        s, w = user_slice(traces[i], off, length_ms)
        total += s
        wrapped += w
    return total, wrapped
