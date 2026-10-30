"""E7: 5G-LENA open-loop quota replay (experiment_design.md §7 E7; plan phase H).

Question: do the service differences that the fluid model predicts between designs survive HARQ, AMC, RLC queues and
the proportional-fair MAC scheduler of 5G-LENA? The NR results are descriptive (H7a-H7c, no CIs).

Stages (``--stage``)
--------------------
calibrate  TRAIN label; reads no trace data. Short saturating runs (constant offered rate of ``CAL_OFFER_PER_PRB``
           Mbit/s per quota PRB, quotas 35/35/36 PRB, ``CAL_EPOCHS`` one-second epochs of which the first
           ``CAL_SKIP`` are discarded) for every ``--discRadius`` in ``RADII`` = {30, 60, 120, 200, 250, 300, 400} m and every ns-3 RngRun in
           ``CAL_RNG_RUNS`` (the RngRuns of the case windows). NR per-PRB rate = new-TB Mbit per used PRB-second,
           pooled over slices, epochs and RngRuns. The radius chosen is the one whose rate lies within +-20 % of the
           fluid model's TRAIN mean per-PRB rate (``Calibration.r_bar`` = r_bar_TRAIN = 0.437 Mbit/s per PRB) and is
           closest to it in log ratio (ties: the smaller radius); if none qualifies the stage fails and nothing is
           chosen. Result: ``results/raw/e7/nr_calibration.json`` (read by the later stages).
pilot      TRAIN only: 2 cases (the window at :data:`CASE_OFFSET_S` = 43260 s of the first two TRAIN dates, or
           ``--case-offset``; protected design, truthful).
           Gate: the quotas bind, i.e. in >= 50 % of the scored intervals some slice uses >= 90 % of its quota
           (``prb_used_mean / prb_quota`` of the KPI file).
cases      The 12 E7 cases: 4 windows (the window at :data:`CASE_OFFSET_S` = 43260 s = 23:01 UTC, the 5th TEST
           window of a date, of the first 4 dates of the split; see "Window choice"; ``--case-offset`` selects
           another pre-declared offset off TEST) x 3 arms
             protected_truthful      protected net, full-decision price, all tenants truthful;
             unconstrained_truthful  unconstrained net, fixed-priority price, all tenants truthful;
             unconstrained_br        unconstrained net, fixed-priority price, the class that gains most
                                     best-responds (ex post, every interval, against truthful others).
           The deviating class is fixed per window BEFORE any NR run by this rule: the tenant with the largest mean
           best-response gain (declared utility, credits per scored interval) among the fluid episodes of the
           unconstrained arm in which each tenant in turn best-responds to the fixed-priority price (ties: lowest
           tenant index). The protected design's best response equals truth, so no protected-manipulated case is
           replayed; this is checked offline on the same window (the fluid episodes with each tenant best-responding
           to the full-decision price must report the admitted type in every interval).
           ``--split test`` is the real E7: it refuses to start unless the pre-registration freeze passes
           (``data.puffer_fetch.require_test_read_gate``) and both checkpoints are listed with their SHA-256 in the
           freeze block. ``--split train`` and ``--split dev`` rehearse the same pipeline (untrained stand-in nets
           are allowed there, never on TEST).
report     Re-summarise the finished cases of a split (also run at the end of every stage).
--smoke    TRAIN window 2024-01-01 +60 s composed with K = 20 intervals (warm-up 5): the 8 fluid episodes of the
           cases stage (both nets; each tenant best-responding in turn, so 20 best-response intervals each) and ONE
           ns-3 run of 20 one-second epochs (protected arm, truthful) to confirm that the binary accepts the
           schedule and budget files. Its NR wall time is printed and stored.

Window choice (decided 2026-10-19; frozen as :data:`CASE_OFFSET_S`)
--------------------------------------------------------------------
The design's rule took each date's first window (offset 60 s = 11:01 UTC), which is the lightest window of the
day: TRAIN mean offered load 0.72 at offset 60 s against 0.90 over all windows (DEV 0.81 against 0.90; D2 fixes date
means, not windows). In a fluid pre-check the pilot window 2024-01-01 +60 s (offered load 0.61) binds in only 35-40 %
of the scored intervals, below the 50 % pilot gate, while offset 43260 s (23:01 UTC; the 5th of the 8 TEST
windows of a date; pre-declared on TRAIN, DEV and TEST; TRAIN mean load 0.91) binds in 100 % on both pilot dates.
The orchestrator chose 43260 s (``CASE_OFFSET_S``) for the pilot, the cases and TEST; ``--case-offset``
overrides it off TEST only.

One NR case
-----------
1. The fluid episode (``sim.episode.run_window``, measurement delay d = 2) gives the deployed integer quotas (K, 3).
2. ``nr.schedule.write_budget`` raises every quota below ``--minPrb`` = 4 to 4 by taking PRBs from the largest quota
   (every move logged); the SAME adjusted quotas feed every fluid comparison.
3. ``nr.run.run_open_loop`` (offline ``--port=0``; E7 flags ``--uePerSlice=4 --pdb=2002,150,300
   --rlcBuf=4194304,1048576,1048576 --mcsTable=1 --minPrb=4``; ``--discRadius`` from the calibration; ``--rngRun`` =
   1 + window index, common to the three arms of a window) replays the per-ms byte schedule of the window's true
   arrivals in aggregate mode. It checks that ns-3 applied exactly the written budget and records whether the PDCP
   SDU volume equals the schedule.
4. Fluid replays of the adjusted quotas, late-counted variant (no expiry; bits served after the delay budget are
   counted as late, because 5G-LENA delivers late packets instead of dropping them):
   ``fluid_raca`` with the window's true Raca per-PRB rates (the fluid model's own prediction);
   ``fluid_cal`` with the CALIBRATED NR per-PRB rate of each slice class, a constant taken from the calibration stage
   at the chosen radius (mean over its RngRuns), fixed before any case is run and independent of every case outcome
   (the out-of-sample "calibrated fluid" of H7c; round-3 review); and ``fluid_nr`` with the NR-measured per-interval
   per-PRB rate of the SAME run, an in-sample decomposition that isolates queueing and HARQ effects but is fitted to
   the outcome it is compared with, so it is never used as validation (its correlation with NR delivery is near 1 by
   construction). ``fluid_raca_expiry`` (the FIFO-with-expiry model of the other experiments) is kept for reference.
   NR-measured rates are NaN in intervals in which a slice used no PRB; they are filled from the slice's nearest
   earlier interval with a rate (then the nearest later one), and from the Raca rate if the slice never used a PRB;
   the filled entries are counted. Without a calibration file (smoke runs) ``fluid_cal`` is NaN.
5. One row per (interval, tenant) in ``table.parquet`` and a case summary (scored intervals).

Hypotheses (descriptive; computed in ``h7_summary``)
---------------------------------------------------
H7a  Victim effect = change in the honest (non-deviating) tenants' delivered-within-PDB Mbit per scored interval,
     unconstrained_br minus unconstrained_truthful. Holds when NR and fluid have the same NONZERO sign in >= 3/4
     windows AND the NR effect is within 2x of the fluid prediction in >= 3/4 windows (round-3 review: both criteria;
     two zero effects carry no sign information and do not count as agreeing).
H7b  The honest tenants receive more delivered Mbit under protected_truthful than under unconstrained_br.
H7c  Per-interval correlation between NR delivered Mbit and calibrated-fluid (``fluid_cal``, out of sample) served
     Mbit >= 0.8. The in-sample ``fluid_nr`` correlation is reported as a decomposition only.

TEST (round-3 review): ``--disc-radius`` is refused (the radius is the calibration's), ``--case-offset`` must equal
the frozen :data:`CASE_OFFSET_S` (pre-registration item: None = the design's first window, 60 s; or 43260), and the
two checkpoints must be the registry's default seed-:data:`CKPT_SEED` models (the default when no ``--ckpt-*`` is
given) and listed in the freeze block. Pre-freeze audit (F6, PV-8/F10): ``--dry-run`` is refused; the cases stage
requires ``results/raw/e7/nr_calibration.json`` to be listed in the freeze block (``registry.checkpoint_paths``
lists it), to have chosen :data:`E7_RADIUS_M` = 250 m, and to record the SHA-256 of the ns-3 binary and nr library
used now (``--binary`` is accepted only when it is that build, :func:`check_frozen_calibration`); every
``case.json`` records ``binary_sha256``.

Outputs
-------
``E7_NR_OUT`` (default ``/media/darpan/New Volume/ICC_rewrite_nr/e7``): ``calibration/``, ``<group>_<split>/
episodes/*.parquet|json`` and ``<group>_<split>/cases/<case>/`` (``schedule_run<r>.csv.gz``, ``budget_run<r>.csv``,
``budget_adjustments_run<r>.csv``, ``kpi_run<r>.csv``, ``ns3_run<r>.log``, ``cmd_run<r>.txt``, ``table.parquet``,
``case.json``, ``done.json``) and ``<group>_<split>/summary/``. No pcap or per-packet log is ever written: the
scenario writes none, and any file in a run directory other than the six expected ones is deleted and the case fails.
ns-3 runs in a staging directory inside the project tree (``nr/runs/e7/``; ``nr.run.run_open_loop`` refuses run
directories elsewhere), and its files are moved to ``E7_NR_OUT`` afterwards. Compact summaries (kB) are copied to
``results/raw/e7/``. Disk guards: >= 3 GB free on ``/`` (staging) and >= 5 GB on the NR volume before every run.

Resumable: a finished episode (parquet + json) or NR case (``done.json``) whose configuration digest matches is
skipped unless ``--force``; an interrupted case is rerun from scratch.

Orchestrator decisions (2026-10-19, binding; code/STATUS.md §0) that this driver uses
------------------------------------------------------------------------------------
D1  GBR fractions x 0.8 (g/mu = 0.40 video, 0.64 URLLC-like, 0.16 mMTC-like). They enter through the TRAIN
    calibration's contract (``data/calibration/n3_default.json``); a TRAIN-only decision.
D2  Per-date offered-load normalization: every window is composed with its date's counts
    (``Calibration.counts_for``; TEST counts exist only after the freeze). Each case reports its realized offered load
    (offered traffic over B x r_bar_TRAIN, scored intervals) and capacity load (``eval.headroom.window_load``).
D4  Harm is measured on the honest tenants' own outcomes: the H7a victim effect and H7b use the non-deviating
    tenants' delivered Mbit (NR) and on-time Mbit (fluid); tariff-weighted welfare is not used in E7.
D6  The nets replayed here are the PPO checkpoints, trained with the lookahead return (h = 3, ``PPOConfig.horizon=3``);
    E7 itself trains nothing.
D3, D5 and D7 do not concern E7.

How to run (from ``code/``, ``PYTHONDONTWRITEBYTECODE=1 /home/darpan/anaconda3/bin/python``)
-------------------------------------------------------------------------------------------
    experiments/e7_nr.py --smoke --workers 8
    experiments/e7_nr.py --split train --stage calibrate --workers 16
    experiments/e7_nr.py --split train --stage pilot --workers 2 [--ckpt-protected P]
    experiments/e7_nr.py --split test --list-cases                       # the 12 definitions; reads no data
    experiments/e7_nr.py --split test --stage cases --workers 6 --test-freeze ../prereg/PREREG.md
                                                    # only after the freeze; the seed-0 registry models by default
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if str(Path(__file__).resolve().parent) not in sys.path:                 # experiments/registry.py (TEST checkpoints)
    sys.path.insert(1, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from slicemech import io as sio  # noqa: E402
from slicemech.nr import run as nrrun  # noqa: E402
from slicemech.nr import schedule as nrs  # noqa: E402

# ------------------------------------------------------------------------------------------------------- constants
NR_VOLUME = Path("/media/darpan/New Volume")
NR_OUT_DEFAULT = Path(os.environ.get("E7_NR_OUT", str(NR_VOLUME / "ICC_rewrite_nr"))) / "e7"
SUMMARY_DEFAULT = sio.ROOT / "results" / "raw" / "e7"
STAGING_DEFAULT = sio.ROOT / "nr" / "runs" / "e7"
CAL_NAME = "n3_default"                  # n = 3 (video, urllc, mmtc): the NR scenario has exactly three slices
D = 2                                    # measurement delay (intervals), spec/params.yaml default
SPLITS = ("train", "dev", "test")
STAGES = ("calibrate", "pilot", "cases", "report")

# NR rate calibration (TRAIN)
RADII = (30, 60, 120, 200, 250, 300, 400)  # --discRadius candidates (m); 250-400 added 2026-10-20: with the AMC
                                           # v5.0 backport the NR rate at 200 m was 1.207 x r_bar_TRAIN (just outside +-20 %)
RATE_TOL = 0.20                          # NR per-PRB rate within +-20 % of r_bar_TRAIN
#: The radius the TRAIN calibration chose (``results/raw/e7/nr_calibration.json``, 2026-10-20, AMC v5.0 build:
#: 250 m). Frozen by the code hash: on TEST the frozen calibration file must name it (pre-freeze audit F6).
E7_RADIUS_M = 250
CAL_RNG_RUNS = (1, 2, 3, 4)              # the RngRuns of the four case windows (pilot windows use 1 and 2)
CAL_EPOCHS = 10
CAL_SKIP = 2                             # epochs discarded (start-up MCS before the first CQI)
CAL_QUOTAS = (35, 35, 36)                # tenant order (video, urllc, mmtc) == slice order; sum = B
CAL_OFFER_PER_PRB = 1.0                  # Mbit/s per quota PRB: above the 64QAM per-PRB goodput, so slices saturate
CAL_SATURATION_MIN = 0.9                 # mean quota use of a calibration run below this is flagged

# pilot gate (TRAIN)
BIND_FRAC = 0.9
BIND_SHARE = 0.5
N_PILOT_WINDOWS = 2

# cases
N_CASE_WINDOWS = 4
CASE_OFFSET_INDEX = 0                    # first pre-declared window of a date (offset 60 s on every split)
#: Pre-registration item (STATUS.md §8.3 item 1): the case window offset (s). None = the design's rule (first
#: window, offset 60 s); 43260 binds in 100 % of the TRAIN pilot intervals. Frozen by the code hash; on TEST
#: ``--case-offset`` must equal it.
#: Orchestrator decision 2026-10-19: offset 43260 s = 23:01 UTC (the 5th TEST window of a date); binds quotas in
#: 100 % of TRAIN pilot intervals vs 35-40 % at 60 s.
CASE_OFFSET_S: int | None = 43260
#: Checkpoint seed of the replayed nets (the registry's ``default/<kind>/seed<CKPT_SEED>/model.pt``; fixed on TEST).
CKPT_SEED = 0
ARMS = {"protected_truthful": ("protected", "full", False),
        "unconstrained_truthful": ("unconstrained", "fixed", False),
        "unconstrained_br": ("unconstrained", "fixed", True)}
KINDS = ("protected", "unconstrained")

# H7 thresholds (descriptive)
H7A_SIGN_SHARE = 0.75
H7A_RATIO = 2.0
H7C_CORR = 0.8

# smoke
SMOKE_WINDOW = ("2024-01-01", 60)
SMOKE_K = 20
SMOKE_WARMUP = 5

# NR run
NR_TIMEOUT_S = 6 * 3600.0
MIN_FREE_ROOT_GB = 3.0
MIN_FREE_OUT_GB = 5.0
FLUID_VARIANTS = (("fluid_raca", "raca", True), ("fluid_cal", "cal", True), ("fluid_nr", "nr", True),
                  ("fluid_raca_expiry", "raca", False))
#: The out-of-sample calibrated fluid of H7c (``fluid_nr`` is the in-sample decomposition).
H7C_VARIANT = "fluid_cal"
NR_COLUMNS = ("pdcp_sdu_mbit", "delivered_mbit", "delivered_pdb_mbit", "rlc_drop_mbit", "delay_mean_ms",
              "delay_p50_ms", "delay_p95_ms", "delay_p99_ms", "within_pdb_frac", "prb_quota", "prb_used_mean",
              "quota_used_frac", "mcs_mean", "rate_mbps_per_prb", "loss_prev")


# ------------------------------------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class Paths:
    """Output locations (passed to every worker, so spawned processes and tests agree on them)."""

    out: Path = NR_OUT_DEFAULT          # NR outputs (large): schedules, KPI CSVs, logs, tables
    staging: Path = STAGING_DEFAULT     # ns-3 run directories (inside the project tree), emptied after each run
    summary: Path = SUMMARY_DEFAULT     # compact copies inside the project tree
    min_free_root_gb: float = MIN_FREE_ROOT_GB
    min_free_out_gb: float = MIN_FREE_OUT_GB


@dataclass(frozen=True)
class Episode:
    """One fluid episode: window x net kind x tenant profile (``br_tenant`` = -1 for truthful)."""

    group: str                  # 'pilot' | 'cases' | 'smoke'
    split: str
    date: str
    offset: int
    kind: str                   # 'protected' | 'unconstrained'
    br_tenant: int = -1

    @property
    def wtag(self) -> str:
        return f"{self.date}_o{self.offset:05d}"

    @property
    def price(self) -> str:
        return "full" if self.kind == "protected" else "fixed"

    @property
    def tag(self) -> str:
        prof = "truthful" if self.br_tenant < 0 else f"br{self.br_tenant}"
        return f"{self.wtag}__{self.kind}__{prof}"


@dataclass(frozen=True)
class Case:
    """One NR case: window x arm, with the window's common RngRun."""

    group: str
    split: str
    date: str
    offset: int
    widx: int                   # window index within the group (RngRun = 1 + widx)
    arm: str

    @property
    def wtag(self) -> str:
        return f"{self.date}_o{self.offset:05d}"

    @property
    def tag(self) -> str:
        return f"{self.wtag}__{self.arm}"

    @property
    def rng_run(self) -> int:
        return 1 + self.widx

    def episode(self, br_tenant: int = -1) -> Episode:
        kind, _, br = ARMS[self.arm]
        if br and br_tenant < 0:
            raise ValueError(f"case {self.tag} needs the deviating tenant")
        return Episode(self.group, self.split, self.date, self.offset, kind, br_tenant if br else -1)


@dataclass(frozen=True)
class Ctx:
    """Run options shared by every job."""

    paths: Paths = field(default_factory=Paths)
    ckpts: tuple = ()                   # (('protected', path), ('unconstrained', path))
    allow_standin: bool = False         # untrained nets when no checkpoint is given (never on TEST)
    K: int | None = None                # composition length override (smoke)
    warmup: int | None = None
    test_freeze: str | None = None
    force: bool = False
    disc_radius: float | None = None
    cal_rates: tuple | None = None      # calibrated NR per-PRB rate per tenant (fluid_cal), from the calibration stage
    binary: str = str(nrrun.NS3_BIN)
    binary_sha: str = ""
    timeout_s: float = NR_TIMEOUT_S
    dry_run: bool = False

    def ckpt(self, kind: str) -> str | None:
        return dict(self.ckpts).get(kind)


# ------------------------------------------------------------------------------------------------------- helpers
def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (set, frozenset, tuple)):
        return list(x)
    raise TypeError(f"not JSON serialisable: {type(x)}")


def dump_json(path: Path, value) -> Path:
    """Atomic JSON write to any output path (the NR volume is outside the project tree, so ``sio.write_json``
    does not apply there)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=1, sort_keys=True, default=_jsonable)
    os.replace(tmp, path)
    return path


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_jsonable).encode()).hexdigest()[:16]


def _calibration_sha() -> str:
    from slicemech.data.calibrate import CALIB_DIR
    return sio.file_sha256(CALIB_DIR / f"{CAL_NAME}.json")


def disk_guard(paths: Paths) -> None:
    """Raise unless the NR volume is mounted (when used) and both file systems have the required free space."""
    out = Path(paths.out)
    if NR_VOLUME in (out, *out.parents) and not os.path.ismount(NR_VOLUME):
        raise RuntimeError(f"the NR output volume {NR_VOLUME} is not mounted")
    sio.require_free_gb(paths.min_free_root_gb, sio.ROOT)
    out.mkdir(parents=True, exist_ok=True)
    sio.require_free_gb(paths.min_free_out_gb, out)


def _write_summary_copy(paths: Paths, rel: str, value) -> None:
    """Compact copy inside the project tree (JSON dict or DataFrame -> CSV)."""
    p = sio.confine(Path(paths.summary) / rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, pd.DataFrame):
        value.to_csv(p, index=False)
    else:
        dump_json(p, value)


def _map(fn, jobs: list, workers: int) -> list:
    """Run ``fn`` over ``jobs``; serial in-process for ``workers <= 1`` (tests), else spawned processes. Each result
    is ``('ok', value)`` or ``('error', traceback)``, so one failure does not hide the others."""
    if not jobs:
        return []
    if workers <= 1 or len(jobs) == 1:
        return [_safe((fn, j)) for j in jobs]
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_safe, [(fn, j) for j in jobs]))


def _safe(arg):
    fn, job = arg
    try:
        return "ok", fn(job)
    except Exception:  # noqa: BLE001 -- reported by the caller
        return "error", traceback.format_exc()


def _unwrap(results: list, what: str) -> list:
    errors = [r[1] for r in results if r[0] == "error"]
    if errors:
        for e in errors:
            print(f"[e7] {what} failed:\n{e}", file=sys.stderr)
        raise RuntimeError(f"{len(errors)} of {len(results)} {what} job(s) failed (tracebacks above)")
    return [r[1] for r in results]


# ------------------------------------------------------------------------------------------------------- definitions
def case_windows(split: str, n: int = N_CASE_WINDOWS, offset: int | None = None) -> list[tuple[str, int]]:
    """(date, offset) of the first ``n`` dates of the split: the date's first pre-declared window (design default,
    offset 60 s) or the given ``offset``, which must be a pre-declared window offset of the split. Pure: reads the
    date list only (for TEST, the pre-registered Sundays/Mondays with any recorded replacements), no trace data."""
    from slicemech.data import compose, puffer
    dates = puffer.split_dates(split)
    if len(dates) < n:
        raise ValueError(f"split {split!r} has {len(dates)} dates, need {n}")
    offs = compose.window_offsets(split)
    if offset is None:
        offset = CASE_OFFSET_S if CASE_OFFSET_S is not None else offs[CASE_OFFSET_INDEX]
    if int(offset) not in offs:
        raise ValueError(f"offset {offset} s is not a pre-declared window of split {split!r} ({offs})")
    return [(d, int(offset)) for d in dates[:n]]


def define_cases(split: str, offset: int | None = None) -> list[Case]:
    """The 12 E7 cases of a split (4 windows x 3 arms), in a fixed order. Defining them reads no data."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return [Case("cases", split, d, o, j, arm) for j, (d, o) in enumerate(case_windows(split, offset=offset))
            for arm in ARMS]


def define_pilot(offset: int | None = None) -> list[Case]:
    """The 2 TRAIN pilot cases (protected design, truthful), at the same window offset as the cases."""
    return [Case("pilot", "train", d, o, j, "protected_truthful")
            for j, (d, o) in enumerate(case_windows("train", N_PILOT_WINDOWS, offset))]


def define_smoke() -> list[Case]:
    d, o = SMOKE_WINDOW
    return [Case("smoke", "train", d, o, 0, "protected_truthful")]


def episodes_for(cases: list[Case], with_br: bool) -> list[Episode]:
    """Fluid episodes a case list needs: the truthful run of every arm's net, plus (``with_br``) each tenant
    best-responding in turn under both nets (selection of the deviating class and the protected check)."""
    eps: list[Episode] = []
    for c in cases:
        for kind in (KINDS if with_br else (ARMS[c.arm][0],)):
            e = Episode(c.group, c.split, c.date, c.offset, kind)
            if e not in eps:
                eps.append(e)
            if with_br:
                for i in range(3):
                    b = replace(e, br_tenant=i)
                    if b not in eps:
                        eps.append(b)
    eps.sort(key=lambda e: (e.kind != "unconstrained" or e.br_tenant < 0, e.wtag, e.kind, e.br_tenant))  # slow first
    return eps


def require_split_allowed(split: str, stage: str, test_freeze=None, ckpts: dict | None = None) -> dict | None:
    """Gate before anything is read. TEST: the pre-registration freeze must pass and every checkpoint must be one of
    the frozen checkpoints; calibrate/pilot/smoke are TRAIN only."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if stage in ("calibrate", "pilot", "smoke") and split != "train":
        raise ValueError(f"stage {stage!r} runs on TRAIN only (got --split {split})")
    if split != "test":
        return None
    from slicemech.data import puffer_fetch
    rec = puffer_fetch.require_test_read_gate(test_freeze)        # PermissionError before the freeze
    if stage != "cases":
        return rec
    frozen = puffer_fetch.parse_freeze(Path(rec["path"]).read_text())["checkpoints"]
    need = dict(ckpts or {})
    for kind in KINDS:
        p = need.get(kind)
        if not p:
            raise PermissionError(f"TEST cases need the frozen {kind} checkpoint (--ckpt-{kind})")
        if sio.file_sha256(p) not in set(frozen.values()):
            raise PermissionError(f"checkpoint {p} is not among the frozen checkpoints of {rec['path']}")
        want = expected_ckpt(kind)
        if Path(p).resolve() != want.resolve():
            raise PermissionError(f"TEST replays the pre-registered {kind} net {want}, not {p} (round-3 review)")
    return rec


def expected_ckpt(kind: str) -> Path:
    """The pre-registered checkpoint of a net kind on TEST: the registry's default model of seed :data:`CKPT_SEED`."""
    import registry as R
    return R.model_dir("default", kind, CKPT_SEED) / R.MODEL_FILE


# ------------------------------------------------------------------------------------------------------- windows, nets
def compose(split: str, date: str, offset: int, ctx: Ctx):
    """(calibration, window) for one case window (TEST windows need the freeze; D2 per-date counts)."""
    from slicemech.data.calibrate import load_calibration
    cal = load_calibration(CAL_NAME)
    kw = {} if ctx.K is None else dict(K=int(ctx.K), warmup=int(ctx.warmup))
    w = cal.window(split, date, offset, test_freeze=ctx.test_freeze, **kw)
    if w.n != 3 or sorted(w.cls) != sorted(nrs.NR_SLICE_CLASS):
        raise ValueError(f"E7 needs n = 3 (video, urllc, mmtc); calibration {CAL_NAME} has {w.cls}")
    return cal, w


def controller(kind: str, ctx: Ctx, cal, split: str):
    """(net, operator forecaster, provenance) for a net kind: the checkpoint if given, else an untrained stand-in
    (rehearsals only; refused on TEST)."""
    from slicemech.control import nets
    path = ctx.ckpt(kind)
    if path:
        net = nets.load_net(path)
        src = dict(source="checkpoint", path=str(path), sha256=sio.file_sha256(path))
    elif ctx.allow_standin and split != "test":
        net = nets.make_net(kind, seed=0)
        src = dict(source="stand-in: untrained net, seed 0 (no checkpoint given)")
    else:
        raise ValueError(f"no {kind} checkpoint given (--ckpt-{kind}); stand-ins are allowed only on TRAIN/DEV")
    expected = {"protected": "ProtectedNet", "unconstrained": "UnconstrainedNet"}[kind]
    if type(net).__name__ != expected:
        raise ValueError(f"checkpoint for {kind} holds a {type(net).__name__}, expected {expected}")
    fc = getattr(net, "forecaster", None)
    if fc is None:
        fc = cal.operator_forecaster(D)
        src["forecaster"] = f"Calibration.operator_forecaster({D})"
    else:
        src["forecaster"] = "net.forecaster (from training)"
    return net, fc, src


def _ckpt_token(kind: str, ctx: Ctx) -> str:
    p = ctx.ckpt(kind)
    return sio.file_sha256(p) if p else "standin-seed0"


def offered_load(window, r_bar: float, B: int, scored_only: bool = False) -> float:
    """D2 realized offered load: summed measured demand over B x r_bar_TRAIN, averaged over ALL K intervals (the
    E0/E1 quantity, ``offered_load``) or over the scored intervals (``offered_load_scored``); round-3 review: one
    definition per name across drivers (``eval.headroom.offered_load``)."""
    from slicemech.config import Mechanism
    from slicemech.eval.headroom import offered_load as _ol
    return _ol(window, r_bar, Mechanism(B=int(B)), scored_only=scored_only)


# ------------------------------------------------------------------------------------------------------- episodes
def episode_digest(ep: Episode, ctx: Ctx) -> str:
    return _digest(dict(ep=asdict(ep), cal=_calibration_sha(), ckpt=_ckpt_token(ep.kind, ctx), K=ctx.K,
                        warmup=ctx.warmup, d=D))


def episode_paths(ep: Episode, paths: Paths) -> tuple[Path, Path]:
    base = Path(paths.out) / f"{ep.group}_{ep.split}" / "episodes"
    return base / f"{ep.tag}.parquet", base / f"{ep.tag}.json"


def run_episode(job) -> dict:
    """Play one fluid episode and store its table; returns the summary record (resumable)."""
    ep, ctx = job
    import torch
    torch.set_num_threads(1)
    from slicemech.config import Mechanism
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import BestResponder, Truthful

    pq, js = episode_paths(ep, ctx.paths)
    dig = episode_digest(ep, ctx)
    if js.exists() and pq.exists() and not ctx.force:
        rec = load_json(js)
        if rec.get("digest") == dig:
            return rec
    disk_guard(ctx.paths)
    cal, w = compose(ep.split, ep.date, ep.offset, ctx)
    net, fc, src = controller(ep.kind, ctx, cal, ep.split)
    pols = [Truthful() for _ in range(w.n)]
    if ep.br_tenant >= 0:
        pols[ep.br_tenant] = BestResponder(ep.price)
    t0 = time.perf_counter()
    df = E.run_window(w, net, E.PriceRule(ep.price), pols, D, Mechanism(), cal.shape, forecaster=fc,
                      price_columns="billed")
    secs = time.perf_counter() - t0
    df.attrs = {}
    pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq, index=False, compression="zstd")
    rec = dict(tag=ep.tag, episode=asdict(ep), window_id=w.id, K=int(w.K), warmup=int(w.warmup), digest=dig,
               controller=src, price=ep.price, seconds=secs)
    if ep.br_tenant >= 0:
        b = df[df["tenant"] == ep.br_tenant]
        sc = b[b["scored"]]
        rec.update(br_gain_mean=float(np.nanmean(sc["br_gain"])), br_gain_max=float(np.nanmax(sc["br_gain"])),
                   br_frac_misreport=float((sc["report"] != sc["theta_adm"]).mean()),
                   br_reports_equal_truth=bool((b["report"].to_numpy() == b["theta_adm"].to_numpy()).all()),
                   br_exact_share=float(sc["br_exact"].mean()) if "br_exact" in sc else float("nan"))
    dump_json(js, rec)
    return rec


def select_br_tenant(recs: list[dict]) -> int:
    """The deviating tenant of a window: largest mean best-response gain under the unconstrained net and the
    fixed-priority price (credits per scored interval); ties -> lowest tenant index."""
    gains = {}
    for r in recs:
        e = r["episode"]
        if e["kind"] == "unconstrained" and e["br_tenant"] >= 0:
            gains[int(e["br_tenant"])] = float(r["br_gain_mean"])
    if sorted(gains) != [0, 1, 2]:
        raise ValueError(f"need the unconstrained best-response episode of every tenant, got {sorted(gains)}")
    best = max(gains.values())
    return min(i for i, g in gains.items() if g == best)


def episode_quotas(ep: Episode, paths: Paths) -> tuple[np.ndarray, dict]:
    pq, js = episode_paths(ep, paths)
    df = pd.read_parquet(pq, columns=["k", "tenant", "quota"])
    q = df.pivot(index="k", columns="tenant", values="quota").sort_index().to_numpy(dtype=float)
    if not np.all(np.isfinite(q)) or np.any(q != np.rint(q)):
        raise ValueError(f"episode {ep.tag} has non-integer quotas")
    return q.astype(np.int64), load_json(js)


# ------------------------------------------------------------------------------------------------------- fluid vs NR
def fill_rates(r_nr: np.ndarray, r_fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN NR per-PRB rates (no PRB used in that interval): nearest earlier interval of the same slice, then
    nearest later one; a slice that never used a PRB takes ``r_fallback``. Returns (filled, mask of filled)."""
    r = np.array(r_nr, dtype=float)
    miss = ~np.isfinite(r) | (r <= 0)
    r[miss] = np.nan
    out = pd.DataFrame(r).ffill().bfill().to_numpy()
    none = ~np.isfinite(out)
    out[none] = np.asarray(r_fallback, dtype=float)[none]
    return out, miss


def fluid_table(window, q_episode: np.ndarray, q_adj: np.ndarray, kpi: pd.DataFrame, cal_rates=None) -> pd.DataFrame:
    """One row per (interval, tenant): NR KPIs next to the fluid replays of the adjusted quotas (module docstring,
    step 4). Volumes in Mbit per interval, rates in Mbit/s per PRB, delays in ms. ``cal_rates`` (n,) = the
    calibration stage's NR per-PRB rate per tenant (``fluid_cal``); None leaves the ``fluid_cal`` columns NaN."""
    from slicemech.sim import fluid
    K, n = q_adj.shape
    ct = window.contract
    A = fluid.cumulative_arrivals(np.asarray(window.arrivals_ms)[:K * fluid.SLOTS_PER_S])
    r_raca = np.asarray(window.rate_true, dtype=float)[:K]
    r_nr_raw = nrrun.kpi_matrix(kpi, "rate_mbps_per_prb")
    r_nr, filled = fill_rates(r_nr_raw, r_raca)
    r_cal = None if cal_rates is None else np.tile(np.asarray(cal_rates, dtype=float).reshape(1, n), (K, 1))
    cols = dict(k=np.repeat(np.arange(K), n), tenant=np.tile(np.arange(n), K), cls=np.tile(np.array(window.cls), K),
                scored=np.repeat(np.arange(K) >= int(window.warmup), n), quota_episode=q_episode[:K].ravel(),
                quota=q_adj.ravel(), adjusted=(q_episode[:K] != q_adj).ravel(), r_raca=r_raca.ravel(),
                r_nr=r_nr.ravel(), r_nr_filled=filled.ravel())
    for c in NR_COLUMNS:
        cols[f"nr_{c}"] = nrrun.kpi_matrix(kpi, c).ravel()
    for tag, rates, late in FLUID_VARIANTS:
        if rates == "cal" and r_cal is None:
            for c in ("arrivals", "on_time", "late", "dropped", "served", "backlog"):
                cols[f"{tag}_{c}"] = np.full(K * n, np.nan)
            continue
        r = {"raca": r_raca, "nr": r_nr, "cal": r_cal}[rates]
        out, _, _ = fluid.replay(A, q_adj.astype(float), r, ct.delay_ms, late_counted=late)
        cols[f"{tag}_arrivals"] = out.arrivals_Mbit.ravel()
        cols[f"{tag}_on_time"] = out.on_time_Mbit.ravel()
        cols[f"{tag}_late"] = out.late_Mbit.ravel()
        cols[f"{tag}_dropped"] = out.dropped_Mbit.ravel()
        cols[f"{tag}_served"] = out.served_Mbit.ravel()
        cols[f"{tag}_backlog"] = out.backlog_Mbit.ravel()
    return pd.DataFrame(cols)


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def binding(table: pd.DataFrame, frac: float = BIND_FRAC) -> dict:
    """Pilot gate: share of scored intervals in which some slice uses >= ``frac`` of its quota (NR KPIs)."""
    s = table[table["scored"]]
    u = s.pivot(index="k", columns="tenant", values="nr_quota_used_frac").sort_index().to_numpy(dtype=float)
    hit = np.nan_to_num(u, nan=0.0) >= frac
    share = float(hit.any(1).mean()) if len(hit) else float("nan")
    return dict(share_any=share, share_by_tenant=hit.mean(0).tolist(), frac=frac, required_share=BIND_SHARE,
                passes=bool(share >= BIND_SHARE), n_intervals=int(len(hit)))


def case_summary(table: pd.DataFrame) -> dict:
    """Per-tenant means over the scored intervals, correlations and the binding diagnostic."""
    s = table[table["scored"]]
    per = {}
    for i, g in s.groupby("tenant"):
        d = dict(cls=str(g["cls"].iloc[0]), quota=float(g["quota"].mean()), quota_episode=float(g["quota_episode"].mean()),
                 adjusted_share=float(g["adjusted"].mean()), r_raca=float(g["r_raca"].mean()),
                 r_nr=float(g["r_nr"].mean()), r_nr_filled_share=float(g["r_nr_filled"].mean()))
        for c in ("pdcp_sdu_mbit", "delivered_mbit", "delivered_pdb_mbit", "rlc_drop_mbit", "prb_used_mean",
                  "quota_used_frac", "mcs_mean", "rate_mbps_per_prb"):
            d[f"nr_{c}"] = float(np.nanmean(g[f"nr_{c}"])) if g[f"nr_{c}"].notna().any() else float("nan")
        for c in ("delay_p50_ms", "delay_p95_ms", "delay_p99_ms"):
            d[f"nr_{c}_median"] = float(np.nanmedian(g[f"nr_{c}"])) if g[f"nr_{c}"].notna().any() else float("nan")
        for tag, _, _ in FLUID_VARIANTS:
            for c in ("arrivals", "on_time", "late", "dropped", "served"):
                d[f"{tag}_{c}"] = float(g[f"{tag}_{c}"].mean())
            d[f"corr_delivered_{tag}"] = _corr(g["nr_delivered_mbit"], g[f"{tag}_served"])
            d[f"corr_pdb_{tag}"] = _corr(g["nr_delivered_pdb_mbit"], g[f"{tag}_on_time"])
        per[int(i)] = d
    pooled = {f"corr_delivered_{t}": _corr(s["nr_delivered_mbit"], s[f"{t}_served"]) for t, _, _ in FLUID_VARIANTS}
    pooled.update({f"corr_pdb_{t}": _corr(s["nr_delivered_pdb_mbit"], s[f"{t}_on_time"]) for t, _, _ in FLUID_VARIANTS})
    return dict(per_tenant=per, pooled=pooled, binding=binding(table), n_scored=int(s["k"].nunique()),
                adjusted_intervals=int(table.loc[table["adjusted"], "k"].nunique()))


# ------------------------------------------------------------------------------------------------------- NR runs
EXPECTED_FILES = ("schedule_run{r}.csv", "budget_run{r}.csv", "budget_adjustments_run{r}.csv", "kpi_run{r}.csv",
                  "ns3_run{r}.log", "cmd_run{r}.txt")


def _collect_run(stage: Path, dest: Path, rng_run: int) -> dict:
    """Move a finished run directory's files to ``dest`` (schedule gzipped); refuse unexpected files (pcap,
    per-packet traces): they are deleted and the run fails."""
    expected = {f.format(r=rng_run) for f in EXPECTED_FILES}
    present = {p.name for p in stage.iterdir()}
    unexpected = sorted(present - expected)
    if unexpected:
        for name in unexpected:
            (stage / name).unlink()
        raise RuntimeError(f"ns-3 wrote unexpected files {unexpected} in {stage} (deleted; no pcap or per-packet "
                           "logs are allowed)")
    dest.mkdir(parents=True, exist_ok=True)
    info = {}
    sched = stage / f"schedule_run{rng_run}.csv"
    if sched.exists():
        info["schedule_sha256"] = sio.file_sha256(sched)
        info["schedule_bytes"] = sched.stat().st_size
        with open(sched, "rb") as fi, gzip.open(dest / (sched.name + ".gz"), "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo)
    for name in sorted(present - {sched.name}):
        shutil.copyfile(stage / name, dest / name)
    shutil.rmtree(stage)
    return info


def case_digest(case: Case, ep_rec: dict, ctx: Ctx, disc_radius) -> str:
    return _digest(dict(case=asdict(case), episode=ep_rec["digest"], radius=disc_radius, rng=case.rng_run,
                        flags=nrrun.E7_FLAGS, binary=ctx.binary_sha, K=ep_rec["K"],
                        cal_rates=list(ctx.cal_rates) if ctx.cal_rates else None))


def case_dir(case: Case, paths: Paths) -> Path:
    return Path(paths.out) / f"{case.group}_{case.split}" / "cases" / case.tag


def run_case(job) -> dict:
    """One NR case (module docstring, steps 1-5); resumable via ``done.json``."""
    case, br_tenant, ctx, runner = job
    from slicemech.config import Mechanism
    from slicemech.eval.headroom import window_load
    mech = Mechanism()
    ep = case.episode(br_tenant)
    q_ep, ep_rec = episode_quotas(ep, ctx.paths)
    dest = case_dir(case, ctx.paths)
    dig = case_digest(case, ep_rec, ctx, ctx.disc_radius)
    if (dest / "done.json").exists() and not ctx.force and load_json(dest / "done.json").get("digest") == dig:
        return load_json(dest / "case.json")
    disk_guard(ctx.paths)
    shutil.rmtree(dest, ignore_errors=True)
    cal, w = compose(case.split, case.date, case.offset, ctx)
    if w.id != ep_rec["window_id"]:
        raise ValueError(f"window {w.id} differs from the episode's window {ep_rec['window_id']}")
    K = int(w.K)
    stage = sio.confine(Path(ctx.paths.staging) / f"{case.group}_{case.split}" / case.tag)
    shutil.rmtree(stage, ignore_errors=True)
    t0 = time.perf_counter()
    kpi, meta = nrrun.run_open_loop(w.arrivals_ms, q_ep[:K], w.cls, w.contract.delay_ms, stage,
                                    rng_run=case.rng_run, B=mech.B, disc_radius=ctx.disc_radius,
                                    binary=ctx.binary, runner=runner, timeout=ctx.timeout_s, dry_run=ctx.dry_run,
                                    **nrrun.E7_FLAGS)
    wall = time.perf_counter() - t0
    rec = dict(case=asdict(case), tag=case.tag, br_tenant=int(br_tenant) if ARMS[case.arm][2] else -1,
               window_id=w.id, K=K, warmup=int(w.warmup), rng_run=case.rng_run, disc_radius=ctx.disc_radius,
               episode=ep_rec["tag"], controller=ep_rec["controller"], digest=dig, cmd=meta["cmd"],
               ns3_wall_s=float(meta["wall_s"]), wall_s=wall, n_adjusted_intervals=meta["n_adjusted_intervals"],
               offered_load=offered_load(w, cal.r_bar, mech.B),
               offered_load_scored=offered_load(w, cal.r_bar, mech.B, scored_only=True),
               capacity_load=window_load(w, mech), cal_rates=list(ctx.cal_rates) if ctx.cal_rates else None,
               r_bar_train=float(cal.r_bar), dry_run=ctx.dry_run, binary_sha256=ctx.binary_sha)
    if ctx.dry_run:
        rec.update(_collect_run(stage, dest, case.rng_run))
        dump_json(dest / "case.json", rec)
        return rec
    rec["pdcp_matches_schedule"] = bool(meta["pdcp_matches_schedule"])
    table = fluid_table(w, q_ep, np.asarray(meta["quotas_adjusted"], dtype=np.int64), kpi, ctx.cal_rates)
    rec["summary"] = case_summary(table)
    rec.update(_collect_run(stage, dest, case.rng_run))
    table.to_parquet(dest / "table.parquet", index=False, compression="zstd")
    dump_json(dest / "case.json", rec)
    dump_json(dest / "done.json", dict(digest=dig, finished=time.strftime("%Y-%m-%d %H:%M:%S")))
    return rec


# ------------------------------------------------------------------------------------------------------- calibration
def calibration_arrivals(K: int = CAL_EPOCHS) -> tuple[np.ndarray, np.ndarray]:
    """Saturating constant arrivals (Mbit per ms, tenant order) and the equal quotas of a calibration run."""
    q = np.tile(np.array(CAL_QUOTAS, dtype=np.int64), (K, 1))
    a = np.tile(CAL_OFFER_PER_PRB * np.array(CAL_QUOTAS, dtype=float) / 1000.0, (K * 1000, 1)).astype(np.float32)
    return a, q


def run_calibration(job) -> dict:
    """One saturating run (radius, RngRun); returns pooled new-TB Mbit and used PRB-seconds (resumable)."""
    radius, rng_run, ctx, runner = job
    from slicemech.data.calibrate import load_calibration
    cal = load_calibration(CAL_NAME)
    cls = tuple(cal.classes)
    if cls != nrs.NR_SLICE_CLASS:
        raise ValueError(f"calibration expects tenant order {nrs.NR_SLICE_CLASS}, got {cls}")
    dest = Path(ctx.paths.out) / "calibration" / f"r{radius:03d}_run{rng_run}"
    dig = _digest(dict(radius=radius, rng=rng_run, K=CAL_EPOCHS, q=CAL_QUOTAS, offer=CAL_OFFER_PER_PRB,
                       flags=nrrun.E7_FLAGS, binary=ctx.binary_sha))
    if (dest / "done.json").exists() and not ctx.force and load_json(dest / "done.json").get("digest") == dig:
        return load_json(dest / "result.json")
    disk_guard(ctx.paths)
    shutil.rmtree(dest, ignore_errors=True)
    stage = sio.confine(Path(ctx.paths.staging) / "calibration" / f"r{radius:03d}_run{rng_run}")
    shutil.rmtree(stage, ignore_errors=True)
    a, q = calibration_arrivals()
    kpi, meta = nrrun.run_open_loop(a, q, cls, cal.contract.delay_ms, stage, rng_run=rng_run, disc_radius=radius,
                                    binary=ctx.binary, runner=runner, timeout=ctx.timeout_s, **nrrun.E7_FLAGS)
    s = kpi[kpi["epoch"] >= CAL_SKIP]
    used = float(s["prb_used_mean"].sum())                     # PRB-seconds (1-s epochs)
    tb = float(s["tb_new_mbit"].sum())
    by = {str(c): float(g["tb_new_mbit"].sum() / g["prb_used_mean"].sum()) if g["prb_used_mean"].sum() > 0
          else float("nan") for c, g in s.groupby("cls")}
    rec = dict(radius=radius, rng_run=rng_run, tb_new_mbit=tb, used_prb_s=used,
               rate=tb / used if used > 0 else float("nan"), rate_by_class=by,
               quota_use=float(s["quota_used_frac"].mean()), mcs_mean=float(s["mcs_mean"].mean()),
               saturated=bool(s["quota_used_frac"].mean() >= CAL_SATURATION_MIN), ns3_wall_s=float(meta["wall_s"]),
               epochs=CAL_EPOCHS, skipped=CAL_SKIP, digest=dig)
    rec.update(_collect_run(stage, dest, rng_run))
    dump_json(dest / "result.json", rec)
    dump_json(dest / "done.json", dict(digest=dig))
    return rec


def choose_radius(rows: list[dict], r_bar: float, tol: float = RATE_TOL) -> dict:
    """Pool the runs per radius and choose (module docstring): within +-tol of ``r_bar``, closest in log ratio,
    ties -> smaller radius; ``chosen`` is None if no radius qualifies."""
    table = []
    for R in sorted({int(r["radius"]) for r in rows}):
        rr = [r for r in rows if int(r["radius"]) == R]
        used = sum(r["used_prb_s"] for r in rr)
        rate = sum(r["tb_new_mbit"] for r in rr) / used if used > 0 else float("nan")
        ratio = rate / r_bar
        table.append(dict(radius=R, rate=rate, ratio=ratio, within=bool(abs(ratio - 1.0) <= tol + 1e-12),
                          runs=len(rr), saturated=all(r["saturated"] for r in rr)))
    ok = [t for t in table if t["within"] and np.isfinite(t["ratio"])]
    chosen = min(ok, key=lambda t: (round(abs(math.log(t["ratio"])), 9), t["radius"]))["radius"] if ok else None
    return dict(chosen=chosen, r_bar_train=r_bar, tol=tol, table=table)


def stage_calibrate(ctx: Ctx, workers: int, runner=subprocess.run) -> dict:
    from slicemech.data.calibrate import load_calibration
    cal = load_calibration(CAL_NAME)
    jobs = [(R, r, ctx, runner) for R in RADII for r in CAL_RNG_RUNS]
    rows = _unwrap(_map(run_calibration, jobs, workers), "calibration run")
    res = choose_radius(rows, float(cal.r_bar))
    res.update(runs=rows, radii=RADII, rng_runs=CAL_RNG_RUNS, epochs=CAL_EPOCHS, skipped=CAL_SKIP,
               quotas=CAL_QUOTAS, offer_per_prb=CAL_OFFER_PER_PRB, binary_sha256=ctx.binary_sha,
               flags=nrrun.E7_FLAGS, calibration=CAL_NAME, calibration_sha256=_calibration_sha())
    dump_json(Path(ctx.paths.out) / "calibration" / "nr_calibration.json", res)
    _write_summary_copy(ctx.paths, "nr_calibration.json", res)
    for t in res["table"]:
        print(f"[e7] radius {t['radius']:>3d} m: NR rate {t['rate']:.4f} Mbit/s/PRB = {t['ratio']:.3f} x r_bar "
              f"({'within' if t['within'] else 'outside'} +-{RATE_TOL:.0%}; saturated={t['saturated']})")
    print(f"[e7] chosen --discRadius: {res['chosen']}")
    if res["chosen"] is None:
        raise RuntimeError("no radius gives an NR per-PRB rate within +-20 % of r_bar_TRAIN (see the table)")
    return res


def calibrated_radius(paths: Paths) -> float | None:
    p = Path(paths.summary) / "nr_calibration.json"
    if not p.exists():
        return None
    c = load_json(p)
    return None if c.get("chosen") is None else float(c["chosen"])


def calibrated_rates(paths: Paths, radius, classes=None) -> tuple | None:
    """The calibration stage's NR per-PRB rate (Mbit/s per PRB) of each slice class at ``radius``: the mean over the
    calibration RngRuns of each run's class rate (new-TB Mbit per used PRB-second), in tenant order ``classes``
    (default ``nr.schedule.NR_SLICE_CLASS``). Fixed before any case runs, so ``fluid_cal`` is out of sample (round-3
    review). None when the calibration file or that radius is missing (smoke)."""
    p = Path(paths.summary) / "nr_calibration.json"
    if radius is None or not p.exists():
        return None
    runs = [r for r in load_json(p).get("runs", []) if float(r.get("radius", -1)) == float(radius)]
    if not runs:
        return None
    classes = tuple(classes or nrs.NR_SLICE_CLASS)
    out = []
    for c in classes:
        v = [float(r["rate_by_class"][c]) for r in runs if np.isfinite(float(r["rate_by_class"].get(c, np.nan)))]
        if not v:
            return None
        out.append(float(np.mean(v)))
    return tuple(out)


def check_frozen_calibration(paths: Paths, ctx: Ctx, freeze_file) -> dict:
    """TEST gate of the NR inputs (pre-freeze audit F6): ``nr_calibration.json`` (radius and the fluid_cal rates)
    must be listed with its SHA-256 in the freeze block (``registry.verify_frozen``), must have chosen the frozen
    radius :data:`E7_RADIUS_M`, and the ns-3 scenario binary and nr library used now (``ctx.binary_sha``) must be the
    ones the calibration was measured with. PermissionError otherwise; returns the calibration record."""
    import registry as R
    p = Path(paths.summary) / "nr_calibration.json"
    if not p.is_file():
        raise PermissionError(f"TEST cases need the frozen NR calibration {p}")
    R.verify_frozen([p], freeze_file)
    cal = load_json(p)
    if cal.get("chosen") is None or float(cal["chosen"]) != float(E7_RADIUS_M):
        raise PermissionError(f"the NR calibration chose radius {cal.get('chosen')}, the frozen code "
                              f"E7_RADIUS_M = {E7_RADIUS_M}")
    if ctx.binary_sha != cal.get("binary_sha256"):
        raise PermissionError("the ns-3 binary or nr library differs from the one the frozen NR calibration was "
                              f"measured with ({ctx.binary_sha[:12]}... vs {str(cal.get('binary_sha256'))[:12]}...)")
    return cal


# ------------------------------------------------------------------------------------------------------- H7 summary
def _victim_sum(case_rec: dict, key: str, victims) -> float:
    per = case_rec["summary"]["per_tenant"]
    return float(sum(per[str(v)][key] if str(v) in per else per[v][key] for v in victims))


def h7_summary(recs: list[dict]) -> dict:
    """H7a-H7c from finished case records (windows with all three arms only for H7a/H7b)."""
    done = [r for r in recs if "summary" in r]
    by_w: dict = {}
    for r in done:
        by_w.setdefault(r["case"]["date"] + f"_o{r['case']['offset']:05d}", {})[r["case"]["arm"]] = r
    windows = []
    for wt, arms in sorted(by_w.items()):
        if set(arms) != set(ARMS):
            continue
        br = int(arms["unconstrained_br"]["br_tenant"])
        vic = [i for i in range(3) if i != br]
        ut, ub, pt = arms["unconstrained_truthful"], arms["unconstrained_br"], arms["protected_truthful"]
        eff = {"nr": _victim_sum(ub, "nr_delivered_pdb_mbit", vic) - _victim_sum(ut, "nr_delivered_pdb_mbit", vic),
               "nr_delivered": _victim_sum(ub, "nr_delivered_mbit", vic) - _victim_sum(ut, "nr_delivered_mbit", vic)}
        for tag, _, _ in FLUID_VARIANTS:
            eff[tag] = _victim_sum(ub, f"{tag}_on_time", vic) - _victim_sum(ut, f"{tag}_on_time", vic)
        cmp = {}
        for tag, _, _ in FLUID_VARIANTS:
            f = eff[tag]
            ratio = eff["nr"] / f if np.isfinite(f) and f != 0 else float("nan")
            # round-3 review: the same NONZERO sign (two zero effects carry no sign information)
            cmp[tag] = dict(sign_agree=bool(np.isfinite(f) and eff["nr"] * f > 0), ratio=ratio,
                            within_2x=bool(np.isfinite(ratio) and 1 / H7A_RATIO <= ratio <= H7A_RATIO))
        h7b = {k: _victim_sum(pt, k, vic) - _victim_sum(ub, k, vic)
               for k in ("nr_delivered_mbit", "nr_delivered_pdb_mbit")}
        windows.append(dict(window=wt, br_tenant=br, victims=vic, victim_effect=eff, h7a=cmp,
                            h7b_protected_minus_br=h7b, h7b_holds=bool(h7b["nr_delivered_mbit"] > 0)))
    def corr_of(tag):
        c = {r["tag"]: {str(i): float(d.get(f"corr_delivered_{tag}", float("nan")))
                        for i, d in r["summary"]["per_tenant"].items()} for r in done}
        flat = [v for d in c.values() for v in d.values() if np.isfinite(v)]
        return dict(corr_by_case_tenant=c, min=float(min(flat)) if flat else float("nan"),
                    median=float(np.median(flat)) if flat else float("nan"), threshold=H7C_CORR,
                    share_at_least=float(np.mean([v >= H7C_CORR for v in flat])) if flat else float("nan"),
                    holds=bool(flat) and bool(min(flat) >= H7C_CORR), n=len(flat))

    nW = len(windows)
    h7a = {}
    for tag, _, _ in FLUID_VARIANTS:
        s = sum(w["h7a"][tag]["sign_agree"] for w in windows)
        m = sum(w["h7a"][tag]["within_2x"] for w in windows)
        sc, mc = bool(nW and s >= H7A_SIGN_SHARE * nW), bool(nW and m >= H7A_SIGN_SHARE * nW)
        h7a[tag] = dict(sign_agree=s, within_2x=m, windows=nW, sign_criterion=sc, magnitude_criterion=mc,
                        holds=bool(sc and mc))
    h7c = dict(variant=H7C_VARIANT, **corr_of(H7C_VARIANT))
    h7c["in_sample_fluid_nr"] = dict(corr_of("fluid_nr"), note="decomposition only: the fluid_nr rates are the "
                                     "same run's measured rates, so its correlation with NR delivery is not a test")
    return dict(windows=windows, h7a=h7a,
                h7b=dict(holds=sum(w["h7b_holds"] for w in windows), windows=nW),
                h7c=h7c, binding={r["tag"]: r["summary"]["binding"] for r in done},
                n_cases=len(done))


def summary_rows(recs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in recs:
        if "summary" not in r:
            continue
        for i, d in r["summary"]["per_tenant"].items():
            rows.append(dict(case=r["tag"], arm=r["case"]["arm"], date=r["case"]["date"], offset=r["case"]["offset"],
                             tenant=int(i), br_tenant=r["br_tenant"], rng_run=r["rng_run"],
                             disc_radius=r["disc_radius"], offered_load=r["offered_load"],
                             capacity_load=r["capacity_load"], ns3_wall_s=r["ns3_wall_s"],
                             adjusted_intervals=r["summary"]["adjusted_intervals"],
                             binding_share=r["summary"]["binding"]["share_any"], **d))
    return pd.DataFrame(rows)


def report(group: str, split: str, paths: Paths) -> dict:
    """Collect finished cases of a group/split and write the summary files (NR volume + project copy)."""
    base = Path(paths.out) / f"{group}_{split}"
    recs = []
    for cj in sorted((base / "cases").glob("*/case.json")):
        if (cj.parent / "done.json").exists():
            recs.append(load_json(cj))
    out = dict(group=group, split=split, h7=h7_summary(recs) if recs else None, n_cases=len(recs),
               ns3_wall_s={r["tag"]: r["ns3_wall_s"] for r in recs})
    rows = summary_rows(recs)
    if len(rows):
        (base / "summary").mkdir(parents=True, exist_ok=True)
        rows.to_csv(base / "summary" / "cases.csv", index=False)
        _write_summary_copy(paths, f"{group}_{split}/cases.csv", rows)
    dump_json(base / "summary" / "h7.json", out)
    _write_summary_copy(paths, f"{group}_{split}/h7.json", out)
    return out


# ------------------------------------------------------------------------------------------------------- stages
def run_cases(cases: list[Case], ctx: Ctx, workers: int, with_br: bool, runner=subprocess.run) -> list[dict]:
    """Episodes (phase 1), deviating-class selection (phase 2), NR runs (phase 3)."""
    eps = episodes_for(cases, with_br)
    t0 = time.perf_counter()
    recs = _unwrap(_map(run_episode, [(e, ctx) for e in eps], workers), "episode")
    print(f"[e7] {len(recs)} fluid episodes ready ({time.perf_counter() - t0:.0f} s)")
    br_of, checks = {}, {}
    if with_br:
        for wt in sorted({c.wtag for c in cases}):
            mine = [r for r in recs if f"{r['episode']['date']}_o{r['episode']['offset']:05d}" == wt]
            br_of[wt] = select_br_tenant(mine)
            prot = [r for r in mine if r["episode"]["kind"] == "protected" and r["episode"]["br_tenant"] >= 0]
            checks[wt] = dict(protected_br_equals_truth=all(r["br_reports_equal_truth"] for r in prot),
                              unconstrained_br_gain={r["episode"]["br_tenant"]: r["br_gain_mean"] for r in mine
                                                     if r["episode"]["kind"] == "unconstrained"
                                                     and r["episode"]["br_tenant"] >= 0},
                              br_tenant=br_of[wt])
            print(f"[e7] {wt}: deviating tenant {br_of[wt]} (gains {checks[wt]['unconstrained_br_gain']}); "
                  f"protected best response = truth: {checks[wt]['protected_br_equals_truth']}")
        g = cases[0]
        dump_json(Path(ctx.paths.out) / f"{g.group}_{g.split}" / "summary" / "br_selection.json", checks)
        _write_summary_copy(ctx.paths, f"{g.group}_{g.split}/br_selection.json", checks)
    jobs = [(c, br_of.get(c.wtag, -1), ctx, runner) for c in cases]
    jobs.sort(key=lambda j: j[0].arm != "unconstrained_br")
    return _unwrap(_map(run_case, jobs, workers), "NR case")


def _ctx_from_args(a, paths: Paths) -> Ctx:
    ckpts = tuple((k, str(Path(p).resolve())) for k, p in (("protected", a.ckpt_protected),
                                                            ("unconstrained", a.ckpt_unconstrained)) if p)
    binary = str(a.binary)
    sha = sio.file_sha256(binary) if Path(binary).exists() else "missing"
    # The scenario executable links the nr library dynamically, and the private ns-3 copy patches that library
    # (nr/patches/: gnb-phy statistics, RLC-UM and AMC v5.0 backports). Fingerprint both, so a library change reruns.
    nr_lib = Path(binary).resolve().parent.parent / "lib" / "libns3.46-nr-default.so"
    if nr_lib.exists():
        sha = sha + "+nrlib:" + sio.file_sha256(str(nr_lib))
    return Ctx(paths=paths, ckpts=ckpts, allow_standin=a.split != "test", test_freeze=a.test_freeze,
               force=a.force, binary=binary, binary_sha=sha, timeout_s=a.timeout_h * 3600.0, dry_run=a.dry_run)


def main(argv=None, runner=subprocess.run, paths: Paths | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=SPLITS, default="train")
    ap.add_argument("--stage", choices=STAGES, default="cases")
    ap.add_argument("--smoke", action="store_true", help="tiny TRAIN run: 8 short episodes + one 20-epoch NR run")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ckpt-protected", default=None, help="save_net file of the protected net")
    ap.add_argument("--ckpt-unconstrained", default=None, help="save_net file of the unconstrained net")
    ap.add_argument("--test-freeze", default=None, help="frozen pre-registration (default: the recorded one)")
    ap.add_argument("--disc-radius", type=float, default=None, help="override the calibrated --discRadius (m)")
    ap.add_argument("--binary", default=str(nrrun.NS3_BIN))
    ap.add_argument("--timeout-h", type=float, default=NR_TIMEOUT_S / 3600.0)
    ap.add_argument("--force", action="store_true", help="recompute finished episodes and cases")
    ap.add_argument("--dry-run", action="store_true", help="write the NR input files without running ns-3")
    ap.add_argument("--list-cases", action="store_true", help="print the case definitions of --split and exit")
    ap.add_argument("--case-offset", type=int, default=None,
                    help="window offset (s) of the pilot and case windows; default: the frozen CASE_OFFSET_S "
                         f"({CASE_OFFSET_S} s = 23:01 UTC, the 5th TEST window of a date; None would mean the design's "
                         "first window, 60 s); must be pre-declared on the split (60 and 43260 are on all); on TEST "
                         "only CASE_OFFSET_S itself is accepted")
    a = ap.parse_args(argv)
    paths = paths or Paths()

    if a.list_cases:
        out = dict(split=a.split, cases=[dict(asdict(c), tag=c.tag, rng_run=c.rng_run)
                                         for c in define_cases(a.split, a.case_offset)],
                   pilot=[dict(asdict(c), tag=c.tag, rng_run=c.rng_run) for c in define_pilot(a.case_offset)])
        print(json.dumps(out, indent=1))
        return out

    stage = "smoke" if a.smoke else a.stage
    if a.split == "test":                       # round-3 review: the TEST protocol comes from the frozen code
        bad = ["--disc-radius"] if a.disc_radius is not None else []
        frozen_off = CASE_OFFSET_S
        if a.case_offset is not None and (frozen_off is None or int(a.case_offset) != int(frozen_off)):
            bad.append("--case-offset")
        if a.dry_run:                           # pre-freeze audit PV-8/F10: no partial TEST cases without NR runs
            bad.append("--dry-run")
        if bad:
            raise SystemExit(f"refusing --split test with {', '.join(bad)}: the radius is the calibration's, the "
                             f"case offset the frozen CASE_OFFSET_S = {frozen_off}, and every TEST case runs ns-3")
        if stage == "cases":
            a.ckpt_protected = a.ckpt_protected or str(expected_ckpt("protected"))
            a.ckpt_unconstrained = a.ckpt_unconstrained or str(expected_ckpt("unconstrained"))
    ckpts = {k: v for k, v in (("protected", a.ckpt_protected), ("unconstrained", a.ckpt_unconstrained)) if v}
    rec = require_split_allowed(a.split, stage, a.test_freeze, ckpts)          # TEST: freeze first, nothing read
    ctx = _ctx_from_args(a, paths)
    if rec is not None:
        ctx = replace(ctx, test_freeze=rec["path"], allow_standin=False)
        if stage == "cases":                    # F6: frozen radius, fluid_cal rates and ns-3 build
            check_frozen_calibration(paths, ctx, rec["path"])
    if not a.dry_run and stage != "report" and not Path(ctx.binary).exists():
        raise FileNotFoundError(f"ns-3 scenario binary not found: {ctx.binary}")
    disk_guard(paths)
    t0 = time.perf_counter()

    if stage == "calibrate":
        return stage_calibrate(ctx, a.workers, runner)
    if stage == "report":
        return report("cases", a.split, paths)

    radius = a.disc_radius if a.disc_radius is not None else calibrated_radius(paths)
    ctx = replace(ctx, cal_rates=calibrated_rates(paths, radius))
    if ctx.cal_rates is None:
        print(f"[e7] no calibration-stage NR rates at radius {radius}: fluid_cal (H7c) is not computed")
    if stage == "smoke":
        ctx = replace(ctx, K=SMOKE_K, warmup=SMOKE_WARMUP, disc_radius=radius)
        cases, group, with_br = define_smoke(), "smoke", True
        print(f"[e7] smoke: --discRadius {radius if radius is not None else 'scenario default (60 m)'}")
    else:
        if radius is None:
            raise RuntimeError("no calibrated --discRadius: run --stage calibrate first (or pass --disc-radius)")
        ctx = replace(ctx, disc_radius=radius)
        if stage == "pilot":
            cases, group, with_br = define_pilot(a.case_offset), "pilot", False
        else:
            cases, group, with_br = define_cases(a.split, a.case_offset), "cases", True
    recs = run_cases(cases, ctx, a.workers, with_br, runner)
    out = report(group, a.split, paths)
    out["wall_s"] = time.perf_counter() - t0
    if stage == "pilot":
        b = {r["tag"]: r["summary"]["binding"] for r in recs if "summary" in r}
        out["pilot_gate"] = dict(cases=b, passes=bool(b) and all(v["passes"] for v in b.values()))
        dump_json(Path(paths.out) / f"pilot_{a.split}" / "summary" / "pilot_gate.json", out["pilot_gate"])
        _write_summary_copy(paths, f"pilot_{a.split}/pilot_gate.json", out["pilot_gate"])
        print(f"[e7] pilot binding gate passes: {out['pilot_gate']['passes']}")
    for r in recs:
        b = r.get("summary", {}).get("binding", {})
        print(f"[e7] {r['tag']}: ns-3 {r['ns3_wall_s']:.1f} s for {r['K']} epochs "
              f"({r['ns3_wall_s'] / r['K']:.2f} s/epoch); min-PRB adjusted intervals {r['n_adjusted_intervals']}; "
              f"binding share {b.get('share_any', float('nan')):.2f}; offered load {r['offered_load']:.3f}")
    if stage == "smoke":
        dump_json(Path(paths.out) / "smoke_train" / "summary" / "smoke.json",
                  dict(wall_s=out["wall_s"], cases=recs))
        _write_summary_copy(paths, "smoke_train/smoke.json",
                            dict(wall_s=out["wall_s"], ns3_wall_s={r["tag"]: r["ns3_wall_s"] for r in recs},
                                 disc_radius=radius, summaries={r["tag"]: r.get("summary") for r in recs}))
    print(f"[e7] {stage} done in {out['wall_s']:.0f} s; outputs under {Path(paths.out) / f'{group}_{a.split}'}")
    return out


if __name__ == "__main__":
    main()
