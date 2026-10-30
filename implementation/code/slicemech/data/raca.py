"""Raca et al. 5G production dataset: 5G-mode CQI runs -> per-PRB rates (Mbit/s per PRB).

Port of ``ICC_rebuild_2026-10-10/src/rebuild/data.read_raca`` (L203-266) without the receipt layer.

Each CSV is a handset log with roughly one row per second. A *run* is a maximal sequence of consecutive seconds
of the same UTC date in which every row is a valid 5G-mode row with an integer CQI in 1..15. Rules (unchanged):

* all rows sharing a timestamp must be valid, otherwise the second breaks the run; if they are, the last row in
  source order is used (seconds whose duplicate rows disagree are counted);
* a gap of at most ``MAX_HOLD_S`` = 2 missing seconds is filled by holding the last sample (past-only hold); the
  held seconds carry their sample age (s) so consumers can see staleness; larger gaps, a date change or any invalid
  / non-5G row end the run;
* a non-monotone recording clock excludes the whole file.

Rates: ``phy.cqi_to_rate`` (TS 38.214 Table 5.2.2.1-2, 132 RE, BLER 0.1). Date splits are fixed (INTERFACES §8).

Pool parts (round-1 review, finding 1): ``pool(split, min_len, part='A'|'B')`` keeps the runs at even ('A') or odd
('B') positions of the admissible runs sorted by start time, so the E5 retraining windows and evaluation windows of
a date never share a CQI run (``compose`` maps the E5 phase to a half per date). Runs are not split in time: most
are barely longer than a window, and splicing would break the CQI autocorrelation that ``rate_hat`` relies on.
:func:`part_balance` reports the expected per-PRB rate of each half under :func:`draw_window` (round-2 review).
The prepared run table is cached in ``data/raca/runs.parquet``.
"""
from __future__ import annotations

import datetime as _dt
import functools
import pathlib

import numpy as np
import pandas as pd

from .. import io as sio
from ..phy import cqi_to_rate

RAW_DIR = sio.ICC_DIR / "data" / "raw" / "5g_prod" / "5G-production-dataset"
CACHE = sio.DATA_DIR / "raca" / "runs.parquet"
MAX_HOLD_S = 2

#: Inclusive UTC date ranges of the fixed Raca splits.
SPLITS = {
    "train": ("2019-11-20", "2019-12-06"),
    "dev": ("2019-12-14", "2020-01-06"),
    "test": ("2020-01-16", "2020-02-27"),
}


def split_of_date(day: str) -> str | None:
    """Split name ('train'|'dev'|'test') of a 'YYYY-MM-DD' date, or None if it lies in no split range."""
    for name, (a, b) in SPLITS.items():
        if a <= day <= b:
            return name
    return None


def read_runs(path: pathlib.Path) -> tuple[list[dict], dict]:
    """Parse one Raca CSV into 5G-mode CQI runs.

    Returns ``(runs, info)``; each run is ``dict(file, date, t0, cqi (int64, per s), age (int64 s, 0 = fresh))``
    where ``t0`` is the UTC epoch second of the first sample. ``info`` counts rows and exclusions.
    """
    path = pathlib.Path(path)
    rel = path.relative_to(RAW_DIR).as_posix() if path.is_relative_to(RAW_DIR) else path.name
    d = pd.read_csv(path, usecols=["Timestamp", "NetworkMode", "CQI"])
    ts = pd.to_datetime(d.Timestamp, format="%Y.%m.%d_%H.%M.%S", errors="coerce", utc=True)
    dates = ts.dt.strftime("%Y-%m-%d").fillna("").to_numpy()
    times = ts.to_numpy(dtype="datetime64[s]").astype(np.int64)
    cq = pd.to_numeric(d.CQI, errors="coerce").to_numpy(dtype=float)
    valid = (ts.notna().to_numpy() & (d.NetworkMode.to_numpy() == "5G")
             & np.isfinite(cq) & (cq >= 1) & (cq <= 15) & (cq == np.floor(cq)))
    info = dict(file=rel, rows=len(d), invalid_or_non5g_rows=int((~valid).sum()), held_seconds=0,
                duplicate_conflicting_seconds=0, excluded=None)
    good = times[ts.notna().to_numpy()]
    if (np.diff(good) < 0).any():
        info["excluded"] = "nonmonotone_recording_clock"
        return [], info
    runs, vals, clock, ages = [], [], [], []
    state = {"date": None}

    def flush():
        if vals:
            runs.append(dict(file=rel, date=state["date"], t0=int(clock[0]),
                             cqi=np.asarray(vals, np.int64), age=np.asarray(ages, np.int64)))
            vals.clear(); clock.clear(); ages.clear()

    k = 0
    while k < len(d):
        end = k + 1
        while end < len(d) and times[end] == times[k]:
            end += 1
        if not valid[k:end].all():
            flush()
            k = end
            continue
        if end - k > 1 and len(set(cq[k:end])) > 1:
            info["duplicate_conflicting_seconds"] += 1
        t, day = int(times[k]), str(dates[k])
        if vals:
            gap = t - clock[-1]
            if day != state["date"] or gap > MAX_HOLD_S + 1 or gap < 1:
                flush()
            else:
                last_q = vals[-1]
                for j, held in enumerate(range(clock[-1] + 1, t), start=1):
                    vals.append(last_q); clock.append(held); ages.append(j)
                    info["held_seconds"] += 1
        state["date"] = day
        vals.append(int(cq[end - 1])); clock.append(t); ages.append(0)
        k = end
    flush()
    info["runs"] = len(runs)
    return runs, info


def prepare(force: bool = False) -> pathlib.Path:
    """Parse every Raca CSV once and cache the run table (one row per second) at ``data/raca/runs.parquet``."""
    if CACHE.exists() and not force:
        return CACHE
    files = sorted(RAW_DIR.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no Raca CSV files under {RAW_DIR}")
    rows, infos = [], []
    run_id = 0
    for f in files:
        runs, info = read_runs(f)
        infos.append(info)
        for r in runs:
            L = len(r["cqi"])
            rows.append(pd.DataFrame(dict(run=np.full(L, run_id, np.int32), file=r["file"], date=r["date"],
                                          t=r["t0"] + np.arange(L, dtype=np.int64), cqi=r["cqi"].astype(np.int8),
                                          age=r["age"].astype(np.int16))))
            run_id += 1
    table = pd.concat(rows, ignore_index=True)
    sio.confine(CACHE).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(CACHE, index=False)
    sio.write_json(CACHE.with_suffix(".json"), dict(files=infos, runs=run_id, seconds=len(table),
                                                    source_sha256={i["file"]: sio.file_sha256(RAW_DIR / i["file"])
                                                                   for i in infos}))
    return CACHE


class Run:
    """One contiguous 5G-mode CQI run: ``cqi`` (int per s), ``age`` (s), ``rate`` (Mbit/s per PRB)."""

    __slots__ = ("run_id", "file", "date", "t0", "cqi", "age", "rate")

    def __init__(self, run_id, file, date, t0, cqi, age):
        self.run_id, self.file, self.date, self.t0 = int(run_id), str(file), str(date), int(t0)
        self.cqi = np.asarray(cqi, np.int64)
        self.age = np.asarray(age, np.int64)
        self.rate = cqi_to_rate(self.cqi)

    def __len__(self):
        return len(self.cqi)


@functools.lru_cache(maxsize=1)
def _all_runs() -> tuple[Run, ...]:
    t = pd.read_parquet(prepare())
    out = []
    for rid, g in t.groupby("run", sort=True):
        out.append(Run(rid, g.file.iloc[0], g.date.iloc[0], g.t.iloc[0], g.cqi.to_numpy(), g.age.to_numpy()))
    return tuple(out)


PARTS = ("A", "B")


def pool(split: str, min_len: int = 1, part: str | None = None) -> tuple[Run, ...]:
    """Runs whose date lies in ``split`` and whose length is at least ``min_len`` seconds; with ``part`` ('A'|'B')
    only the runs at even / odd positions of that list sorted by (start time, run id)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {tuple(SPLITS)}")
    runs = tuple(r for r in _all_runs() if split_of_date(r.date) == split and len(r) >= min_len)
    if part is None:
        return runs
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    ordered = sorted(runs, key=lambda r: (r.t0, r.run_id))
    return tuple(ordered[PARTS.index(part)::2])


def draw_window(runs: tuple[Run, ...], length: int, rng: np.random.Generator) -> tuple[Run, int]:
    """Draw a run with probability proportional to its number of admissible offsets, then a uniform offset.

    Returns ``(run, offset)``; the window is ``run.rate[offset: offset + length]`` (never wrapped).
    """
    runs = tuple(r for r in runs if len(r) >= length)
    if not runs:
        raise ValueError(f"no Raca run of at least {length} s")
    w = np.array([len(r) - length + 1 for r in runs], float)
    r = runs[int(rng.choice(len(runs), p=w / w.sum()))]
    return r, int(rng.integers(len(r) - length + 1))


def expected_window_rate(runs: tuple[Run, ...], length: int) -> float:
    """Expected per-PRB rate (Mbit/s per PRB) of a window of ``length`` seconds drawn by :func:`draw_window`."""
    runs = tuple(r for r in runs if len(r) >= length)
    if not runs:
        raise ValueError(f"no Raca run of at least {length} s")
    num = den = 0.0
    for r in runs:
        c = np.r_[0.0, np.cumsum(r.rate)]
        n_off = len(r) - length + 1
        num += float((c[length:length + n_off] - c[:n_off]).sum()) / length     # sum over offsets of window means
        den += n_off
    return num / den


def part_balance(split: str, length: int = 201, *, test_freeze=None) -> dict:
    """Expected window rate (Mbit/s per PRB) of the whole split and of each half, with the ratios to the whole and
    the run/second counts (round-2 review; tolerance ``tractor.PART_BALANCE_TOL``). TEST is refused unless
    ``test_freeze`` passes the freeze gate (``data.puffer_fetch.require_freeze_for_test_reads``)."""
    if split == "test":
        if test_freeze is None:
            raise PermissionError("raca.part_balance('test') is a TEST statistic: pass the frozen pre-registration")
        from .puffer_fetch import require_freeze_for_test_reads
        require_freeze_for_test_reads(test_freeze)
    from .tractor import PART_BALANCE_TOL
    whole = expected_window_rate(pool(split, length), length)
    out = dict(split=split, length=int(length), tol=PART_BALANCE_TOL, pooled_rate=whole)
    for part in PARTS:
        runs = pool(split, length, part)
        rate = expected_window_rate(runs, length)
        out[f"{part}_rate"], out[f"{part}_ratio"] = rate, rate / whole
        out[f"{part}_runs"], out[f"{part}_seconds"] = len(runs), int(sum(len(r) for r in runs))
    out["balanced"] = bool(all(abs(out[f"{p}_ratio"] - 1.0) <= PART_BALANCE_TOL for p in PARTS))
    return out


def mean_rate(split: str, min_len: int = 1) -> float:
    """Pooled mean per-PRB rate (Mbit/s per PRB) over all seconds of the split's runs of length >= ``min_len``."""
    rs = pool(split, min_len)
    if not rs:
        raise ValueError(f"no Raca runs in split {split}")
    return float(np.concatenate([r.rate for r in rs]).mean())


def date_list(split: str) -> list[str]:
    """All UTC dates of a split range (inclusive), for reporting."""
    a, b = (_dt.date.fromisoformat(x) for x in SPLITS[split])
    return [(a + _dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]
