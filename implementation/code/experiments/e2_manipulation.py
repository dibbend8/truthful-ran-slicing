"""E2 - manipulation of unprotected designs (experiment_design.md §6-§7 "E2"; PREREG_DRAFT.md H2a, H2b, H2c; the
protected + full-price bandit design also feeds H3b).

Run from ``code/`` (CPU only, deterministic seeds; ``PYTHONDONTWRITEBYTECODE=1``):

    /home/darpan/anaconda3/bin/python experiments/e2_manipulation.py --split dev --workers 20
    /home/darpan/anaconda3/bin/python experiments/e2_manipulation.py --smoke            # TRAIN, tiny, < 10 min
    /home/darpan/anaconda3/bin/python experiments/e2_manipulation.py --split dev --plan # job list + core-hours

``--split test`` is refused unless the pre-registration freeze is recorded (``puffer_fetch.require_test_read_gate``;
``--test-freeze prereg/PREREG.md`` names it explicitly). Every job writes its own file atomically and is skipped when
the file exists, so an interrupted run resumes where it stopped; the summary step re-reads every job file.

What is measured (setting: n = 3, rho = 0.9, d = 2 s, calibration ``data/calibration/n3_default.json``)
---------------------------------------------------------------------------------------------------------
Arms (``ARMS``; labels and colours from ``spec/controllers.yaml`` via the ``controller_key`` column):

* ``prot_fixed``  trained protected score net, fixed-priority price (manipulable whenever the premium T > 0);
* ``unc_fixed``   trained unconstrained net, fixed-priority price;
* ``unc_usage``   trained unconstrained net, usage tariff ``nu_i Q_i`` (``nu`` revenue-neutral on TRAIN:
  ``nu_i = sum p_i / sum Q_i`` of the protected net, truthful, full price, all TRAIN windows; cached under
  ``results/raw/train/e2/usage_tariff/``);
* ``unc_env``     trained unconstrained net billed with the envelope formula on its non-monotone own-quota path
  (4096-point grid plus bisection; not a truthful price, uncertified).

1. **Per-interval best response (H2a).** One deviating tenant at a time (the role rotates over all n tenants, i.e.
   over the classes) best-responds ex post to truthful others (``pricing.response.best_response``: exact cell
   enumeration for the monotone protected map, 512-point grid + golden section, a lower bound, for the
   unconstrained net) on every 4th scored interval (``BR_EVERY``) of every window. Recorded per tenant-interval:
   deviation gain (credits per interval), report multiple ``z*/theta``, the deviator's PRB shift ``Q_i(z*) - Q_i(theta)``,
   the honest tenants' PRB shift and their declared-value change. Output ``stage/<arm>/s<seed>/<window>.parquet``.
2. **Realized consequences (H2c, D4).** On the replay windows (``--replay-per-date`` per date, spread over the
   time of day; 2 per date gives the planned 32 TEST windows) each tenant in turn plays its best-response
   sequence on every scored interval (truthful in the warm-up, so both runs enter scoring from the same state),
   and the window is replayed by the fluid model. Harm is measured on the honest tenants' own outcomes
   (``eval.metrics.victim_effects``: on-time Mbit, deadline-miss %, their realized value); total tariff-weighted
   welfare is descriptive only (D4). On these windows the stage records of item 1 are read from the replays (the
   per-interval best response depends only on the interval's report-free context, types and rate, so it is the same
   computation). Output ``replay/<arm>/s<seed>/<window>.parquet`` (profiles ``truthful`` and ``br<i>``).
3. **Non-monotonicity of the trained unconstrained net (H2b).** On every scored interval and tenant, the own quota
   ``Q_i(u, theta_-i)`` on a 513-point grid of ``u`` in ``[0, C_i]``; a decrease anywhere is a violation. Rate by
   class, largest PRB drop; the exploitable share (violating states in which the envelope-formula best response
   gains) is computed in the summary from item 1. The protected net is run through the same check as a negative
   control (it must give 0). Output ``nonmono/<net>/s<seed>/<window>.parquet``.
4. **Learning tenants.** All n tenants learn simultaneously with EXP3 or eps-greedy over the multipliers
   ``{0.5, 0.67, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2}`` (``tenants.bandits``); 400 episodes x 4 runs per (design,
   algorithm); designs ``prot_full`` (H3b), ``prot_fixed``, ``unc_fixed``, ``unc_usage``. Episodes are windows drawn
   from the split's windows; evaluations are memoised by (window, multiplier profile) in a per-design SQLite file
   shared by the runs. Output ``bandit/<design>/s<seed>/<algo>_run<r>.npz`` (multiplier trajectories, gains,
   greedy multipliers, final probabilities, per-episode realized welfare).

Controllers come from checkpoints (``resolve_assets``): by default the training registry
(``results/models/default/<kind>/seed<s>/model.pt``, written by ``experiments/train.py``; ``static/static.json``),
then the E0/E1 layout (``results/checkpoints/default/<kind>_seed<s>.pt``, ``static.json``); ``--ckpt-dir`` names
another directory and ``--ckpt kind=path`` one file. A shared ``usage_nu.json`` next to the checkpoints is used
when present. ``--train-missing`` trains a missing net with the PPO defaults (``PPOConfig(horizon=3)``, 400
iterations, TRAIN windows, DEV checkpoint selection; not the DEV-tuned cell) or fits a missing static vector on TRAIN
(rehearsals only). ``--smoke`` always trains tiny nets (2 PPO iterations on the smoke windows) into
``results/raw/train/smoke_assets/``; smoke numbers are plumbing checks, not results.

TEST (round-3 review: E2-E4 used to load whatever file the options or the fallback chain found, and resumed over
outputs of other models without a trace):

* only the training registry is searched (no ``results/checkpoints`` fallback), and every net, the static fit and
  the usage-tariff file must be listed with its SHA-256 in the freeze block (``registry.verify_frozen``);
* the protocol options (``TEST_FIXED_OPTIONS``: dates, window caps, seeds, arms, grids, model locations,
  ``--train-missing``, ``--tag``) are refused; their frozen values are the module constants (e.g.
  :data:`CHECKPOINT_SEEDS`), covered by the code hash of the freeze;
* every job output gets a provenance sidecar ``<output>.prov.json`` (``run_jobs``) with a fingerprint of the job's
  inputs: the SHA-256 of every asset of its checkpoint seed, the code fingerprint (``slicemech`` and the driver
  modules), the calibration, the split and the job arguments. A job is complete only when its output exists AND its
  sidecar carries the current fingerprint; a stale (or partial) output is moved to
  ``results/raw/<split>/archive/<UTC stamp>/<experiment>/...`` before the job reruns (``run_jobs``; never deleted,
  PREREG §8), so outputs of different models never mix. ``config.json`` holds the latest invocation and
  ``config_history.jsonl`` every invocation.

Checkpoint seeds (STATUS.md §8.3 item 2; pre-registration item): E2-E4 run :data:`CHECKPOINT_SEEDS` = (0,) unless
the orchestrator sets all five seeds before the freeze; the analysis records ``n_seeds`` next to H2c, H3a and H4b.

Orchestrator decisions in force (binding; code/STATUS.md §0)
-------------------------------------------------------------
* D1: GBR fractions x 0.8 (g/mu = 0.40 / 0.64 / 0.16) - in the calibration this driver loads.
* D2: per-date offered-load normalization on every split - windows are composed with ``cal.counts_for(split, date)``
  (``Calibration.window``); realized per-window offered and capacity loads are recorded per window (``loads.parquet``).
* D4: harm is measured on the honest tenants' own outcomes (H2c primary = their on-time Mbit and deadline-miss %;
  realized value next to it); total tariff-weighted welfare is descriptive only.
* D6: every PPO run started here (smoke, ``--train-missing``) uses the lookahead return with h = 3
  (``PPOConfig(horizon=3)``, set explicitly).
* D3 (G4 dropped, load subgroups), D5 (E5 protocol) and D7 (E4b frontier, ``e4_cost.py``) do not change E2; the
  per-window offered load needed for D3-style subgroups is written to ``loads.parquet``.

This module also hosts the plumbing shared with ``e3_protection.py`` and ``e4_cost.py`` (split guard, window lists,
checkpoint and tariff resolution, arms, interval rules, the scored-interval best responder, the resumable job pool).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from slicemech import io as sio  # noqa: E402

# ============================================================================================ constants
CAL_NAME = "n3_default"
D = 2                          # measurement delay (intervals of 1 s), spec/params.yaml information.measurement_delay_s
HORIZON = 3                    # D6: PPO lookahead return horizon
BR_EVERY = 4                   # best responses on every 4th scored interval (params.yaml BrEvery)
NONMONO_GRID = 513             # own-report grid of the H2b check
EPS_REL = 1e-8                 # H3a allowance: 1e-8 pi_i mu_i Delta
BANDIT_EPISODES = 400
BANDIT_RUNS = 4
BANDIT_ALGOS = ("exp3", "eps")
BANDIT_DESIGNS = ("prot_full", "prot_fixed", "unc_fixed", "unc_usage")
E2_ARMS = ("prot_fixed", "unc_fixed", "unc_usage", "unc_env")
E2_MODULES = ("e2_manipulation.py",)          # code fingerprint of E2 jobs (with the slicemech package)
SMOKE_WINDOWS = (("2024-01-01", 60), ("2024-01-14", 21660))
SMOKE_K, SMOKE_WARMUP = 30, 10           # smoke windows are shortened (same date counts, Calibration.window)
SMOKE_EPISODES, SMOKE_RUNS = 6, 1
SPLITS = ("train", "dev", "test")
#: Checkpoint seeds of the learned nets in E2-E4 (pre-registration item, STATUS.md §8.3 item 2): seed 0 only unless
#: the orchestrator budgets all five (``(0, 1, 2, 3, 4)``) before the freeze. On TEST ``--seeds`` cannot change it.
CHECKPOINT_SEEDS = (0,)
#: Options refused on TEST (``registry.refuse_test_overrides``): the protocol comes from the frozen code.
TEST_FIXED_OPTIONS = ("--dates", "--max-windows", "--seeds", "--tag", "--ckpt-dir", "--ckpt", "--train-missing",
                      "--smoke-windows", "--arms", "--replay-per-date", "--stage-limit", "--nonmono-nets",
                      "--bandit-designs", "--bandit-episodes", "--bandit-runs", "--cert-arms", "--ident-arms",
                      "--ident-limit", "--h3d-arms", "--pay-arms", "--kappa-scales", "--learn-designs", "--h4b-arms",
                      "--bench-windows", "--e2-root")
RAW = sio.ROOT / "results" / "raw"
MODELS_DIR = sio.ROOT / "results" / "models" / "default"          # experiments/registry.py layout (train.py)
CKPT_DIR = sio.ROOT / "results" / "checkpoints" / "default"      # e0_setup.py / e1_value.py layout
CONFIG = "default"                                                # registry configuration of n = 3, d = 2, rho = 0.9
SMOKE_ASSETS = RAW / "train" / "smoke_assets"
NET_KINDS = ("protected", "unconstrained", "kappa0")

DECISIONS = {
    "D1": "GBR fractions x 0.8: g/mu = 0.40 (video), 0.64 (URLLC-like), 0.16 (mMTC-like); TRAIN-only decision",
    "D2": "per-date offered-load normalization on every split: counts so that each date's mean offered traffic of "
          "class c equals share_c x rho x B x rbar_TRAIN (shares 0.6/0.3/0.1)",
    "D3": "gate G4 dropped; H1c/H1d secondary; load subgroups by window offered load",
    "D4": "harm measured on honest tenants' own outcomes (on-time Mbit, deadline-miss %, realized value); total "
          "tariff-weighted welfare descriptive only",
    "D5": "E5: interleaved 2-s block halves, alpha = 1 stored-report reference, 8 windows per date",
    "D6": "learner objective = lookahead return, h = 3 intervals (PPOConfig.horizon=3)",
    "D7": "E4b: welfare vs honest payment frontier from scaling trained kappa by s in {0, 0.5, 1, 2}",
}

# arm -> (controller kind, billed price, spec/controllers.yaml key)
ARMS = {
    "prot_full": ("protected", "full", "protected"),
    "prot_fixed": ("protected", "fixed", "fixed_price"),
    "unc_fixed": ("unconstrained", "fixed", "unprotected"),
    "unc_usage": ("unconstrained", "usage", "unprotected_usage"),
    "unc_env": ("unconstrained", "full", "unprotected_envelope"),
    "unc_truth": ("unconstrained", "fixed", "unprotected"),      # E4: p (grid envelope), p_fix and p_usage filled
    "kappa0_full": ("kappa0", "full", "kappa0"),
    "static_full": ("static", "full", "static"),
    "equal_full": ("equal", "full", "equal"),
    "direct_full": ("direct", "full", "direct"),
    "directtrue_full": ("direct_true", "full", "direct_truerates"),
}
SEED_FREE_KINDS = ("static", "equal", "direct", "direct_true")


def scaled_arm(s: float) -> str:
    """E4b arm name of the protected net with kappa scaled by ``s`` (D7)."""
    return f"kappa_x{float(s):g}"


def arm_kind(arm: str) -> tuple[str, str, str]:
    """(controller kind, billed price, controllers.yaml key) of an arm, including the E4b scaled-kappa arms."""
    if arm in ARMS:
        return ARMS[arm]
    if arm.startswith("kappa_x"):
        return f"protected_scaled:{float(arm[len('kappa_x'):]):g}", "full", "protected"
    raise KeyError(f"unknown arm {arm!r}")


def arm_needs(arm: str) -> set:
    """Assets an arm needs: net kinds, 'static', 'nu'."""
    kind, price, _ = arm_kind(arm)
    need = set()
    if kind.startswith("protected"):
        need.add("protected")
    elif kind in NET_KINDS:
        need.add(kind)
    elif kind == "static":
        need.add("static")
    if price == "usage" or arm == "unc_truth":
        need |= {"nu", "protected"}
    return need


def seed_free(arm: str) -> bool:
    return arm_kind(arm)[0] in SEED_FREE_KINDS


# ============================================================================================ split / windows
def guard_split(split: str, smoke: bool = False, test_freeze=None) -> str | None:
    """Refuse unknown splits, smoke runs off TRAIN, and TEST before the pre-registration freeze (PermissionError,
    raised before any TEST source is opened). Returns the frozen pre-registration's path on TEST, else None."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if smoke and split != "train":
        raise ValueError("--smoke runs on TRAIN windows only")
    if split == "test":
        from slicemech.data import puffer_fetch
        return puffer_fetch.require_test_read_gate(test_freeze)["path"]
    return None


def split_keys(split: str) -> list[tuple[str, int]]:
    """All pre-declared (date, offset) windows of a split (TRAIN 15 x 12, DEV 21 x 6, TEST 16 x 8)."""
    from slicemech.data import compose, puffer
    return [(d, int(o)) for d in puffer.split_dates(split) for o in compose.window_offsets(split)]


def window_keys(split: str, smoke: bool = False, n_smoke: int = 2, dates=None, max_windows: int | None = None
                ) -> list[tuple[str, int]]:
    """Windows of a run: the smoke windows, or every window of the split (optionally only the given dates, and only
    the first ``max_windows`` of them, e.g. for a timing pilot)."""
    if smoke:
        return [tuple(x) for x in SMOKE_WINDOWS[:max(1, int(n_smoke))]]
    keys = split_keys(split)
    if dates:
        keys = [k for k in keys if k[0] in set(dates)]
        if not keys:
            raise ValueError(f"none of the dates {dates} belongs to split {split!r}")
    return keys[:max_windows] if max_windows else keys


def run_keys(args) -> list[tuple[str, int]]:
    """``window_keys`` from the parsed common arguments."""
    return window_keys(args.split, args.smoke, args.smoke_windows, args.dates, getattr(args, "max_windows", None))


def replay_keys(keys, per_date: int = 2) -> list[tuple[str, int]]:
    """Replay windows: ``per_date`` windows of every date, spread over the day and rotated over dates (window j of
    date number t is chosen for j = (t + q * W // per_date) mod W, q = 0..per_date-1; W windows per date)."""
    by_date = collections.OrderedDict()
    for d, o in keys:
        by_date.setdefault(d, []).append(o)
    out = []
    for t, (d, offs) in enumerate(by_date.items()):
        offs = sorted(offs)
        W = len(offs)
        m = min(int(per_date), W)
        pick = sorted({(t + q * W // max(m, 1)) % W for q in range(m)})
        out += [(d, offs[j]) for j in pick]
    return out


def wkey(date: str, offset: int) -> str:
    return f"{date}_o{int(offset):05d}"


def out_root(exp: str, split: str, smoke: bool, tag: str | None = None) -> Path:
    """``results/raw/<split>/<exp>`` (smoke: ``<exp>_smoke``; ``tag``: ``<exp>_<tag>``, e.g. a timing pilot)."""
    name = exp + ("_smoke" if smoke else "") + (f"_{tag}" if tag else "")
    return RAW / split / name


def scored_every(warmup: int, K: int, every: int = BR_EVERY, limit: int | None = None) -> np.ndarray:
    ks = np.arange(int(warmup), int(K), int(every))
    return ks[:limit] if limit else ks


# ============================================================================================ atomic I/O
def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    p = sio.confine(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, p)
    return p


def write_npz(path: Path, **arrays) -> Path:
    """Atomic ``np.savez_compressed`` to a confined path. The temp file ``<name>.tmp<pid>`` does not end in ``.npz``
    (written through a file handle, so numpy appends no suffix): an interrupted write is never picked up by the
    ``*.npz`` globs of the summaries and the analysis (pre-freeze audit HC-10)."""
    p = sio.confine(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
    return p


def f32(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Downcast the given float columns to float32 (compact storage; certificate columns stay float64)."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)
    return df


def read_parquets(paths) -> pd.DataFrame:
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


# ============================================================================================ assets
@dataclasses.dataclass
class AssetSpec:
    """Picklable description of the controllers a worker loads (one per checkpoint seed)."""

    seed: int
    nets: dict                        # kind -> checkpoint path
    static_w: list | None = None
    nu: list | None = None
    sha256: dict = dataclasses.field(default_factory=dict)
    smoke: bool = False


def _search_dirs(ckpt_dir: Path | None, registry_only: bool = False) -> list[Path]:
    """Asset directories in search order (``registry_only``: TEST, the training registry alone)."""
    if registry_only:
        return [MODELS_DIR]
    return [Path(ckpt_dir)] if ckpt_dir else [MODELS_DIR, CKPT_DIR]


def _net_candidates(kind: str, seed: int, ckpt_dir: Path | None, registry_only: bool = False) -> list[Path]:
    """Where a trained net may live, in search order: an explicit ``--ckpt-dir`` (flat ``<kind>_seed<s>.pt``,
    ``<kind>/seed<s>.pt`` or ``<kind>/seed<s>/model.pt``), else the training registry
    (``results/models/default/<kind>/seed<s>/model.pt``, written by ``experiments/train.py``), else the E0/E1
    layout ``results/checkpoints/default/`` (never on TEST: ``registry_only``)."""
    dirs = _search_dirs(ckpt_dir, registry_only)
    out = []
    for d in dirs:
        out += [d / kind / f"seed{seed}" / "model.pt", d / f"{kind}_seed{seed}.pt", d / kind / f"seed{seed}.pt"]
    return out


def _registry_check(path: Path) -> None:
    """A registry model must carry a finished ``config.json`` whose recorded SHA-256 matches the file."""
    rec_p = path.parent / "config.json"
    if path.name == "model.pt" and rec_p.is_file():
        rec = sio.read_json(rec_p)
        if rec.get("status") != "done":
            raise FileNotFoundError(f"{path}: training job not finished ({rec_p})")
        want = rec.get("result", {}).get("model_sha256")
        if want and sio.file_sha256(path) != want:
            raise ValueError(f"{path} differs from the SHA-256 recorded in {rec_p}")


def _static_candidates(ckpt_dir: Path | None, registry_only: bool = False) -> list[Path]:
    dirs = _search_dirs(ckpt_dir, registry_only)
    return [p for d in dirs for p in (d / "static" / "static.json", d / "static.json")]


def _nu_candidates(ckpt_dir: Path | None, registry_only: bool = False) -> list[Path]:
    return [d / "usage_nu.json" for d in _search_dirs(ckpt_dir, registry_only)]


def _first(paths) -> Path | None:
    return next((p for p in paths if Path(p).is_file()), None)


def _parse_overrides(items) -> dict:
    out = {}
    for it in items or ():
        if "=" not in it:
            raise ValueError(f"--ckpt expects kind=path, got {it!r}")
        k, v = it.split("=", 1)
        if k not in NET_KINDS + ("static",):
            raise ValueError(f"--ckpt kind must be one of {NET_KINDS + ('static',)}")
        out[k] = v
    return out


def smoke_windows(cal, split: str = "train", n_smoke: int = 2):
    return [cal.window(split, d, o, K=SMOKE_K, warmup=SMOKE_WARMUP) for d, o in SMOKE_WINDOWS[:max(1, n_smoke)]]


def compose_window(cal, split: str, date: str, offset: int, smoke: bool, test_freeze=None):
    if smoke:
        return cal.window(split, date, offset, K=SMOKE_K, warmup=SMOKE_WARMUP)
    return cal.window(split, date, offset, test_freeze=test_freeze)


def _train_net(kind: str, seed: int, path: Path, smoke: bool, n_smoke: int = 2) -> dict:
    """Train one priority net (D6: lookahead h = 3) and save it; returns a small training record."""
    import torch
    torch.set_num_threads(1)
    from slicemech.config import Mechanism
    from slicemech.control import nets, ppo
    from slicemech.data.calibrate import load_calibration
    cal = load_calibration(CAL_NAME)
    mech = Mechanism()
    if smoke:
        ws = smoke_windows(cal, "train", n_smoke)
        dev = None
        cfg = ppo.PPOConfig(iterations=2, windows_per_iter=len(ws), horizon=HORIZON, eval_every=0)
    else:
        ws = [cal.window("train", d, o) for d, o in split_keys("train")]
        dev = [cal.window("dev", d, o) for d, o in split_keys("dev")]
        cfg = ppo.PPOConfig(horizon=HORIZON)
    t0 = time.perf_counter()
    res = ppo.train(nets.make_net(kind, seed=seed), ws, dev, d=D, mech=mech, config=cfg,
                    forecaster=cal.operator_forecaster(D), seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    nets.save_net(res.net, tmp)
    os.replace(tmp, path)
    rec = dict(kind=kind, seed=seed, smoke=smoke, iterations=cfg.iterations, horizon=cfg.horizon,
               best_iteration=res.best_iteration, best_dev_welfare=res.best_dev_welfare,
               train_windows=len(ws), dev_windows=0 if dev is None else len(dev), seconds=time.perf_counter() - t0)
    sio.write_json(path.with_suffix(".train.json"), rec)
    return rec


def _fit_static(path: Path, smoke: bool, n_smoke: int = 2) -> dict:
    from slicemech.config import Mechanism
    from slicemech.control.static import fit_static
    from slicemech.data.calibrate import load_calibration
    cal = load_calibration(CAL_NAME)
    ws = smoke_windows(cal, "train", n_smoke) if smoke else [cal.window("train", d, o) for d, o in split_keys("train")]
    fit = fit_static(ws, Mechanism(), method="grid", steps=4 if smoke else 20)
    rec = dict(w=fit.w.tolist(), welfare=fit.welfare, method=fit.method, windows=len(ws), smoke=smoke,
               seconds=fit.seconds)
    sio.write_json(path, rec)
    return rec


def _nu_job(args) -> tuple:
    """Per-tenant sums (price, quota) of the protected net, truthful, full price, on one TRAIN window."""
    seed, date, offset, ckpt, smoke = args
    from slicemech.config import Mechanism
    from slicemech.data.calibrate import load_calibration
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    cal = load_calibration(CAL_NAME)
    net = _cached_net(ckpt)
    w = compose_window(cal, "train", date, offset, smoke)
    df = E.run_window(w, net, E.PriceRule("full"), [Truthful() for _ in range(w.n)], D, Mechanism(), cal.shape,
                      record=False, price_columns="billed")
    sc = df[df["scored"]]
    g = sc.groupby("tenant")
    return g["price"].sum().to_numpy(), g["quota"].sum().to_numpy()


_NET_CACHE: dict = {}


def _cached_net(path):
    path = str(path)
    if path not in _NET_CACHE:
        import torch
        torch.set_num_threads(1)
        from slicemech.control import nets
        net = nets.load_net(path)
        if net.forecaster is None or (net.forecaster.horizon not in (None, D)):
            raise ValueError(f"checkpoint {path} carries no operator forecaster for d={D}")
        net.eval()
        _NET_CACHE[path] = net
    return _NET_CACHE[path]


def usage_tariff_path(seed: int, ckpt: Path, smoke: bool) -> Path:
    """Cache file of :func:`usage_tariff` for the protected checkpoint ``ckpt`` of seed ``seed`` (TRAIN:
    ``results/raw/train/e2/usage_tariff/nu_protected_seed<s>_<sha12>.json``, frozen with the models via
    ``registry.checkpoint_paths``; smoke: below ``SMOKE_ASSETS``). E1's default configuration reads the same file."""
    sha = sio.file_sha256(ckpt)
    base = SMOKE_ASSETS if smoke else RAW / "train" / "e2" / "usage_tariff"
    return base / f"nu_protected_seed{seed}_{sha[:12]}.json"


def usage_tariff(seed: int, ckpt: Path, smoke: bool, workers: int, n_smoke: int = 2) -> tuple[list, Path]:
    """Revenue-neutral usage tariff on TRAIN (credits per PRB per interval): ``nu_i = sum p_i / sum Q_i`` of the
    protected net (seed ``seed``), truthful reports, full-decision price, scored intervals of every TRAIN window
    (smoke: the smoke windows). Cached by the checkpoint's SHA-256 (:func:`usage_tariff_path`)."""
    sha = sio.file_sha256(ckpt)
    path = usage_tariff_path(seed, ckpt, smoke)
    if path.exists():
        return sio.read_json(path)["nu"], path
    keys = list(SMOKE_WINDOWS[:n_smoke]) if smoke else split_keys("train")
    jobs = [(seed, d, o, str(ckpt), smoke) for d, o in keys]
    t0 = time.perf_counter()
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_nu_job, jobs))
    else:
        res = [_nu_job(j) for j in jobs]
    P = np.sum([r[0] for r in res], axis=0)
    Q = np.sum([r[1] for r in res], axis=0)
    nu = (P / Q).tolist()
    sio.write_json(path, dict(nu=nu, sum_price=P.tolist(), sum_quota=Q.tolist(), windows=len(jobs), seed=seed,
                              checkpoint=str(ckpt), checkpoint_sha256=sha, smoke=smoke,
                              rule="nu_i = sum_k p_i / sum_k Q_i, protected net, truthful, full price, TRAIN scored",
                              seconds=time.perf_counter() - t0))
    return nu, path


def resolve_assets(arms, seeds, *, smoke: bool, ckpt_dir=None, overrides=None, train_missing: bool = False,
                   workers: int = 1, n_smoke: int = 2, test_freeze_path=None) -> dict:
    """Locate (or, in smoke / ``train_missing`` mode, create) every input the arms need; returns
    ``{seed: AssetSpec}``. Raises FileNotFoundError listing what is missing otherwise.

    Nets: ``--ckpt kind=path`` (``{seed}`` substituted), else :func:`_net_candidates`. Static weights: ``--ckpt
    static=path``, else ``static/static.json`` or ``static.json`` in the same places (``{"w": [...]}``). Usage
    tariff: a shared ``usage_nu.json`` (``{"nu": [...]}``) in the checkpoint directory if one exists, else
    :func:`usage_tariff` (fitted on TRAIN and cached). Smoke: tiny nets and a coarse static fit in
    ``results/raw/train/smoke_assets/``.

    TEST (``test_freeze_path`` given; round-3 review): ``ckpt_dir``, ``overrides`` and ``train_missing`` are refused,
    only the training registry is searched, and every resolved file (nets, static fit, usage tariff) must be listed
    in the freeze block with its SHA-256 (``registry.verify_frozen``, PermissionError otherwise)."""
    test = test_freeze_path is not None
    if test and (ckpt_dir or overrides or train_missing or smoke):
        raise PermissionError("on TEST the assets come from the frozen training registry only: --ckpt-dir, --ckpt, "
                              "--train-missing and --smoke are refused")
    need = set().union(*[arm_needs(a) for a in arms]) if arms else set()
    if smoke and not ckpt_dir:
        ckpt_dir = SMOKE_ASSETS
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
    overrides = _parse_overrides(overrides)
    specs, missing = {}, []
    for seed in seeds:
        paths = {}
        for kind in (k for k in NET_KINDS if k in need):
            if kind in overrides:
                p = Path(str(overrides[kind]).format(seed=seed))
                p = p if p.is_file() else None
            else:
                p = _first(_net_candidates(kind, seed, ckpt_dir, registry_only=test))
            if p is None:
                if smoke or train_missing:
                    p = (ckpt_dir or CKPT_DIR) / f"{kind}_seed{seed}.pt"     # never into train.py's registry
                    print(f"[assets] training {kind} seed {seed} -> {p} "
                          f"({'smoke: 2 PPO iterations' if smoke else 'PPO defaults, horizon 3'})", flush=True)
                    _train_net(kind, seed, p, smoke, n_smoke)
                else:
                    missing.append(f"{kind} seed {seed} (searched "
                                   f"{[str(c) for c in _net_candidates(kind, seed, ckpt_dir, registry_only=test)]})")
                    continue
            _registry_check(p)
            paths[kind] = p
        specs[seed] = AssetSpec(seed=seed, nets={k: str(p) for k, p in paths.items()}, smoke=smoke)
    if "static" in need:
        sp = Path(overrides["static"]) if "static" in overrides else _first(_static_candidates(ckpt_dir, test))
        if sp is None or not sp.is_file():
            if smoke or train_missing:
                sp = (ckpt_dir or CKPT_DIR) / "static.json"
                print(f"[assets] fitting static weights -> {sp}", flush=True)
                _fit_static(sp, smoke, n_smoke)
            else:
                missing.append(f"static weights (searched {[str(c) for c in _static_candidates(ckpt_dir, test)]})")
        if sp is not None and sp.is_file():
            rec = sio.read_json(sp)
            if rec.get("status", "done") != "done":
                raise FileNotFoundError(f"static fit {sp} is not finished")
            for s in specs.values():
                s.static_w = [float(x) for x in rec["w"]]
                s.sha256["static"] = sio.file_sha256(sp)
                s.nets["static_file"] = str(sp)
    if missing:
        raise FileNotFoundError("missing inputs (train them with experiments/train.py, pass --ckpt kind=path, or "
                                "--train-missing): " + "; ".join(missing))
    for s in specs.values():
        for k, p in list(s.nets.items()):
            if k in NET_KINDS:
                s.sha256[k] = sio.file_sha256(p)
        if "nu" in need:
            shared = None if smoke else _first(_nu_candidates(ckpt_dir, test))
            if shared is not None:
                s.nu, npath = [float(x) for x in sio.read_json(shared)["nu"]], shared
            else:
                if test and not usage_tariff_path(s.seed, Path(s.nets["protected"]), smoke).is_file():
                    raise PermissionError("TEST never refits the usage tariff: the TRAIN cache "
                                          f"{usage_tariff_path(s.seed, Path(s.nets['protected']), smoke)} is missing")
                nu, npath = usage_tariff(s.seed, Path(s.nets["protected"]), smoke, workers, n_smoke)
                s.nu = [float(x) for x in nu]
            s.sha256["nu"] = sio.file_sha256(npath)
            s.nets["nu_file"] = str(npath)
    if test:
        import registry as R
        files = sorted({p for s in specs.values() for k, p in s.nets.items()})
        R.verify_frozen(files, test_freeze_path)
    return specs


# ============================================================================================ controllers
class ScaledKappa:
    """Protected net with the report sensitivity scaled at deployment: ``y_i = a_i + s kappa_i psi(z_i)`` (D7).

    ``s >= 0`` keeps the map protected (``ProtectedMap``), hence monotone and priced by the full-decision price."""

    alloc_input = "reports"
    uses_context_features = True

    def __init__(self, net, s: float):
        if not s >= 0:
            raise ValueError("kappa scale must be >= 0")
        self.net, self.s = net, float(s)
        self.forecaster = net.forecaster
        self.name = f"protected_kappa_x{self.s:g}"

    def priority_map(self, ctx, z=None):
        from slicemech.model.scores import ProtectedMap
        a, kappa = self.net.heads_np(ctx)
        return ProtectedMap(a, self.s * kappa, np.asarray(ctx.mu, float))


def make_arm(arm: str, spec: AssetSpec):
    """(controller, PriceRule) of an arm for one asset spec."""
    from slicemech.sim import episode as E
    kind, price, _ = arm_kind(arm)
    nu = None if spec.nu is None else np.asarray(spec.nu, float)
    if kind.startswith("protected_scaled:"):
        ctrl = ScaledKappa(_cached_net(spec.nets["protected"]), float(kind.split(":", 1)[1]))
    elif kind in NET_KINDS:
        ctrl = _cached_net(spec.nets[kind])
    elif kind == "static":
        ctrl = E.FixedWeights(np.asarray(spec.static_w, float), name="static")
    elif kind == "equal":
        ctrl = E.FixedWeights()
    elif kind == "direct":
        ctrl = E.DirectController("hat")
    elif kind == "direct_true":
        ctrl = E.DirectController("true")
    else:
        raise KeyError(kind)
    if price == "usage":
        pr = E.PriceRule("usage", nu=nu)
    elif arm == "unc_truth":
        pr = E.PriceRule("fixed", nu=nu, nonmonotone_full=True)
    else:
        pr = E.PriceRule(price, nu=nu)
    return ctrl, pr


def interval_rules(w, ctrl, ks, mech, shape) -> dict:
    """The quota rule of each interval in ``ks`` exactly as ``sim.episode.run_window`` builds it: context from the
    window's released data (``features.window_features``; equal to the episode's per-interval context, tested in
    ``tests/test_episode.py``), priority map or direct rule from the context alone, allocator rate ``rate_hat[k]``
    (``rate_true[k]`` for the true-rate diagnostic). Returns ``{k: (QuotaRule, r)}``."""
    from slicemech.control import features as F
    from slicemech.model.allocation import QuotaRule
    fc = getattr(ctrl, "forecaster", None)
    if fc is None:
        if getattr(ctrl, "uses_context_features", True):
            raise ValueError("controller reads context features but carries no operator forecaster")
        fc = F.OperatorAR.default()
    wf = F.window_features(w, D, w.contract, forecaster=fc, mech=mech)
    R = np.asarray(w.rate_hat if getattr(ctrl, "rate_source", "hat") == "hat" else w.rate_true, float)
    out = {}
    for k in ks:
        k = int(k)
        ctx = wf.context(k)
        r = R[k].copy()
        if hasattr(ctrl, "quota_policy"):
            pol = ctrl.quota_policy(ctx, ct=w.contract, mech=mech, shape=shape, r=r)
        else:
            pol = ctrl.priority_map(ctx)
        out[k] = (QuotaRule(w.contract, mech, r, pol), r)
    return out


def _scored_best_responder_cls():
    from slicemech.tenants.strategies import BestResponder, admit

    class ScoredBestResponder(BestResponder):
        """Truthful in the warm-up (``k < warmup``; ``k`` = length of the own history), then the ex-post best
        response of ``BestResponder`` on every scored interval, so a deviation run and its truthful run enter
        scoring from the same queue state."""

        def __init__(self, price: str, warmup: int):
            super().__init__(price)
            self.warmup = int(warmup)
            self.name = f"best_response[{price}]@scored"

        def report(self, i, theta, C, history=None, game=None):
            k = 0 if history is None else len(history)
            if k < self.warmup:
                z, clipped = admit(np.asarray(theta, float), np.asarray(C, float))
                self.n_clipped += int(np.sum(clipped))
                self.last = None
                return float(z)
            return super().report(i, theta, C, history=history, game=game)

    return ScoredBestResponder


def ScoredBestResponder(price: str, warmup: int):     # noqa: N802 (factory keeps the import lazy for spawn workers)
    return _scored_best_responder_cls()(price, warmup)


def _scored_multiplier_cls():
    from slicemech.tenants.strategies import Multiplier, admit

    class ScoredMultiplier(Multiplier):
        """Truthful in the warm-up (``k < warmup``; ``k`` = length of the own history), then the constant multiplier
        report ``min(m theta, C)`` on every scored interval, so a multiplier run and its truthful run enter scoring
        from the same queue state (pre-freeze audit HC-7; as :func:`ScoredBestResponder`)."""

        def __init__(self, m: float, warmup: int):
            super().__init__(m)
            self.warmup = int(warmup)
            self.name = f"x{self.m:g}@scored"

        def report(self, i, theta, C, history=None, game=None):
            k = 0 if history is None else len(history)
            if k < self.warmup:
                z, clipped = admit(np.asarray(theta, float), np.asarray(C, float))
                self.n_clipped += int(np.sum(clipped))
                return float(z)
            return super().report(i, theta, C, history=history, game=game)

        def report_series(self, theta, C):
            raise NotImplementedError("ScoredMultiplier depends on the interval index: use report() per interval")

    return ScoredMultiplier


def ScoredMultiplier(m: float, warmup: int):          # noqa: N802 (factory keeps the import lazy for spawn workers)
    return _scored_multiplier_cls()(m, warmup)


# ============================================================================================ worker state
_W: dict = {}


def init_worker(specs: dict, split: str, smoke: bool, test_freeze) -> None:
    """Process initializer: single-threaded torch/BLAS, calibration and asset specs."""
    import torch
    torch.set_num_threads(1)
    from slicemech.config import Mechanism
    from slicemech.data.calibrate import load_calibration
    _W.clear()
    _W.update(specs=specs, split=split, smoke=smoke, test_freeze=test_freeze, cal=load_calibration(CAL_NAME),
              mech=Mechanism(), windows=collections.OrderedDict())


def get_window(date: str, offset: int, split: str | None = None, cache: int = 24):
    """Composed window (LRU cache per worker)."""
    split = split or _W["split"]
    key = (split, date, int(offset))
    win = _W["windows"]
    if key in win:
        win.move_to_end(key)
        return win[key]
    w = compose_window(_W["cal"], split, date, int(offset), _W["smoke"], _W["test_freeze"])
    win[key] = w
    while len(win) > cache:
        win.popitem(last=False)
    return w


def spec_of(seed) -> AssetSpec:
    s = _W["specs"]
    return s[seed] if seed in s else s[next(iter(s))]


# ============================================================================================ job runner
@dataclasses.dataclass
class Job:
    """One unit of work: ``fn(**kw)`` writes ``out``. ``cost`` = estimated core-seconds. ``fp`` = provenance
    fingerprint of the job's inputs (:func:`attach_provenance`); with a fingerprint the job is complete only when
    ``out`` exists and its sidecar ``<out>.prov.json`` carries the same fingerprint (without one: when ``out``
    exists). ``prov`` is written into the sidecar."""

    name: str
    fn: str            # "module:function"
    kw: dict
    out: str
    cost: float = 1.0
    fp: str = ""
    prov: dict = dataclasses.field(default_factory=dict)

    def outputs(self) -> list[Path]:
        """Every file the job writes: ``out`` and the ``*_out`` / ``out`` paths among its arguments."""
        outs = {str(self.out)}
        for k, v in self.kw.items():
            if (k == "out" or k.endswith("_out")) and isinstance(v, (str, Path)) and v:
                outs.add(str(v))
        return [Path(x) for x in sorted(outs)]


def prov_path(out) -> Path:
    return Path(str(out) + ".prov.json")


def job_status(job: Job) -> str:
    """'done' | 'pending' | 'stale' (output from other inputs: other models, code, calibration or arguments)."""
    if not Path(job.out).exists():
        return "pending"
    if not job.fp:
        return "done"
    pp = prov_path(job.out)
    try:
        return "done" if pp.is_file() and sio.read_json(pp).get("fp") == job.fp else "stale"
    except (ValueError, OSError):
        return "stale"


def code_fingerprint(*modules: str) -> str:
    """SHA-256 over the ``slicemech`` package files and the given experiment modules (file names in this folder)."""
    import registry as R
    h = sio.digest(dict(slicemech=R.package_sha256(),
                        modules={m: sio.file_sha256(_HERE / m) for m in sorted(set(modules))}))
    return h


def attach_provenance(jobs, specs: dict, *, split: str, smoke: bool, modules: tuple, extra: dict | None = None
                      ) -> list[Job]:
    """Set every job's provenance fingerprint (round-3 review): the SHA-256 of every asset of its checkpoint seed
    (nets, static fit, usage tariff), the code fingerprint, the calibration file, the split, the smoke flag and the
    job arguments other than its output paths. Returns the jobs."""
    from slicemech.data.calibrate import CALIB_DIR
    code = code_fingerprint(*modules)
    cal_sha = sio.file_sha256(CALIB_DIR / f"{CAL_NAME}.json")
    for j in jobs:
        seed = j.kw.get("seed")
        spec = specs.get(seed) if seed in specs else (specs[next(iter(specs))] if specs else None)
        if "arm" in j.kw:
            need = arm_needs(j.kw["arm"])
        elif "design" in j.kw:
            need = arm_needs(j.kw["design"])
        elif "kind" in j.kw:
            need = {j.kw["kind"]}
        else:
            need = set()
        assets = {k: v for k, v in (spec.sha256 if spec is not None else {}).items() if k in need}
        args = {k: v for k, v in j.kw.items() if not (k == "out" or k.endswith("_out"))}
        j.prov = dict(job=j.name, fn=j.fn, args=args, assets=assets, code=code, calibration_sha256=cal_sha,
                      split=split, smoke=bool(smoke), **(extra or {}))
        j.fp = sio.digest(j.prov)
    return jobs


def _call(job: Job):
    import importlib
    mod, fn = job.fn.split(":")
    t0 = time.perf_counter()
    getattr(importlib.import_module(mod), fn)(**job.kw)
    sec = time.perf_counter() - t0
    if job.fp:            # one sidecar per output (replays too); ``out``'s is written last: the job is then complete
        rec = dict(fp=job.fp, seconds=sec, finished=time.strftime("%Y-%m-%d %H:%M:%S"), **job.prov)
        for f in [f for f in job.outputs() if str(f) != str(job.out)] + [Path(job.out)]:
            if f.exists():
                sio.write_json(prov_path(f), rec)
    return job.name, sec


def clean_temp_files(jobs) -> int:
    """Delete leftover temp files ``<output name>.tmp*`` of interrupted atomic writes next to the jobs' outputs
    (partial files, never results; an old-style ``<name>.tmp<pid>.npz`` would otherwise match the ``*.npz`` globs).
    Returns the number removed."""
    names: dict = collections.defaultdict(set)
    for j in jobs:
        for f in j.outputs():
            names[f.parent].add(f.name)
    n = 0
    for d, outs in names.items():
        if not d.is_dir():
            continue
        for p in d.iterdir():
            base, sep, _ = p.name.partition(".tmp")
            if sep and base in outs and p.is_file():
                p.unlink(missing_ok=True)
                n += 1
    return n


def retire_outputs(jobs, root: Path, *, stamp: str | None = None) -> Path | None:
    """Move every existing output and sidecar of ``jobs`` (stale outputs of other inputs, partial outputs of
    interrupted jobs) to the archive before the jobs are recomputed (``e0_setup.archive_files``:
    ``results/raw/<split>/archive/<stamp>/<experiment>/...``); nothing is deleted (pre-freeze audit F7/PV-10,
    PREREG §8). Returns the archive folder or None."""
    import e0_setup as S
    files = [g for j in jobs for f in j.outputs() for g in (f, prov_path(f))]
    return S.archive_files(files, root, stamp=stamp, reason="E2-E4 job recomputed (stale provenance or interrupted)")


def run_jobs(jobs, *, workers: int, initargs: tuple, errors_path: Path, plan: bool = False) -> dict:
    """Run the pending jobs on a spawn process pool, longest first (:func:`job_status`: output missing, or present
    with another provenance fingerprint; such stale outputs and their sidecars are moved to the archive before the
    rerun (:func:`retire_outputs`), so outputs of different inputs never mix and earlier numbers are kept). Leftover
    temp files are removed first (:func:`clean_temp_files`). Failures are logged to ``errors_path`` (JSON lines; the
    root of the outputs is ``errors_path.parent``) and do not stop the other jobs. Returns counts."""
    status = {id(j): job_status(j) for j in jobs}
    pending = [j for j in jobs if status[id(j)] != "done"]
    stale = [j for j in pending if status[id(j)] == "stale"]
    total_cost = sum(j.cost for j in pending)
    print(f"[jobs] {len(jobs)} jobs, {len(jobs) - len(pending)} done, {len(pending)} pending "
          f"({len(stale)} with outputs of other inputs), estimated {total_cost / 3600:.2f} core-hours", flush=True)
    if plan:
        return dict(total=len(jobs), pending=len(pending), stale=len(stale), failed=0,
                    est_core_hours=total_cost / 3600)
    n_tmp = clean_temp_files(jobs)
    if not pending:
        return dict(total=len(jobs), pending=0, stale=0, failed=0, est_core_hours=0.0, temp_removed=n_tmp,
                    archived=None)
    import e0_setup as S
    stamp = S.archive_stamp()
    archived = retire_outputs(pending, errors_path.parent, stamp=stamp)
    if archived is not None:
        print(f"[jobs] earlier outputs of the pending jobs archived in {archived}", flush=True)
    pending.sort(key=lambda j: -j.cost)
    if errors_path.exists():                     # keep the previous run's failures next to the new log
        prev = errors_path.with_name(errors_path.stem + ".prev.jsonl")
        S.archive_files([prev], errors_path.parent, stamp=stamp, reason="errors log of an earlier invocation")
        os.replace(errors_path, prev)
    import multiprocessing as mp
    failed = 0
    t0 = time.perf_counter()
    done_cost = 0.0
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(pending))), mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=initargs) as ex:
        futs = {ex.submit(_call, j): j for j in pending}
        for n_done, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            done_cost += j.cost
            try:
                _, sec = fut.result()
                msg = f"ok {sec:7.1f}s"
            except Exception as e:      # noqa: BLE001 - logged, other jobs continue
                failed += 1
                msg = f"FAILED {type(e).__name__}: {e}"
                errors_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sio.confine(errors_path), "a") as f:
                    f.write(json.dumps(dict(job=j.name, error=repr(e), tb=traceback.format_exc(),
                                            time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
            el = time.perf_counter() - t0
            eta = el / max(done_cost, 1e-9) * (total_cost - done_cost)
            print(f"[{n_done}/{len(pending)}] {j.name}: {msg} | elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min",
                  flush=True)
    return dict(total=len(jobs), pending=len(pending), stale=len(stale), failed=failed,
                est_core_hours=total_cost / 3600, temp_removed=n_tmp,
                archived=None if archived is None else str(archived))


def write_config(root: Path, args, specs: dict, extra: dict) -> None:
    """``config.json`` (this invocation) and one appended line of ``config_history.jsonl`` per invocation, with the
    asset SHA-256s and the code fingerprint (round-3 review: the config used to be overwritten silently)."""
    from slicemech.data.calibrate import CALIB_DIR
    cfg = dict(args=vars(args), decisions=DECISIONS, calibration=CAL_NAME,
               calibration_sha256=sio.file_sha256(CALIB_DIR / f"{CAL_NAME}.json"), d=D, horizon=HORIZON,
               assets={str(s): dataclasses.asdict(a) for s, a in specs.items()},
               time=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
    sio.write_json(root / "config.json", cfg)
    hist = sio.confine(root / "config_history.jsonl")
    with open(hist, "a") as f:
        f.write(json.dumps(cfg, sort_keys=True, default=sio._jsonable) + "\n")


def common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--split", default=None, choices=SPLITS, help="required except with --smoke (TRAIN)")
    ap.add_argument("--smoke", action="store_true", help="tiny TRAIN run: 1-2 short windows, 2 PPO iterations")
    ap.add_argument("--smoke-windows", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CHECKPOINT_SEEDS),
                    help="checkpoint seeds of the learned nets (default CHECKPOINT_SEEDS; fixed on TEST)")
    ap.add_argument("--dates", nargs="*", default=None, help="restrict to these dates of the split")
    ap.add_argument("--max-windows", type=int, default=None, help="only the first N windows (pilots)")
    ap.add_argument("--tag", default=None, help="output folder suffix: results/raw/<split>/<exp>_<tag>")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt", nargs="*", default=None, help="kind=path overrides ({seed} is substituted)")
    ap.add_argument("--train-missing", action="store_true")
    ap.add_argument("--test-freeze", default=None, help="frozen prereg/PREREG.md (TEST only)")
    ap.add_argument("--plan", action="store_true", help="list jobs and estimated core-hours, run nothing")
    ap.add_argument("--no-summary", action="store_true")
    return ap


def start(args, exp: str, ap: argparse.ArgumentParser | None = None):
    """Shared start-up: split guard (TEST needs the freeze; ``args.freeze_path`` is set to the frozen file),
    TEST protocol options (``TEST_FIXED_OPTIONS`` refused, given the parser ``ap``), output root, disk guard."""
    if args.split is None:
        if not args.smoke:
            raise SystemExit("--split is required (train, dev or test)")
        args.split = "train"
    try:
        args.freeze_path = guard_split(args.split, args.smoke, args.test_freeze)
    except PermissionError as e:
        raise SystemExit(f"refusing --split {args.split}: {e}") from None
    if ap is not None:
        import registry as R
        R.refuse_test_overrides(args.split, args, ap, TEST_FIXED_OPTIONS)
    root = out_root(exp, args.split, args.smoke, args.tag)
    if not args.plan:
        sio.require_free_gb(3)
        root.mkdir(parents=True, exist_ok=True)
    return root


# ============================================================================================ E2 job bodies
REPLAY_COLS = ("k", "tenant", "theta_adm", "report", "quota", "w", "price", "V_declared", "on_time_Mbit",
               "dropped_Mbit", "realized_value", "utility_declared", "utility_realized", "backlog_Mbit",
               "br_z_star", "br_gain", "br_cert", "br_U_truth", "br_exact")
# float32 only for descriptive columns; outcomes and prices stay float64 so that paired differences (victim effects,
# H4b) are exact and an unchanged run differs from its reference by exactly 0
F32_COLS = ("theta_adm", "w", "backlog_Mbit")


def _stage_from_tables(truth: pd.DataFrame, devs: dict, ks, ct, warmup) -> pd.DataFrame:
    """Stage records at intervals ``ks`` from a truthful run and one best-response run per deviator."""
    n = ct.n
    tq = truth.pivot(index="k", columns="tenant", values="quota")
    tv = truth.pivot(index="k", columns="tenant", values="V_declared")
    th = truth.pivot(index="k", columns="tenant", values="theta_adm")
    rows = []
    for i, df in devs.items():
        bq = df.pivot(index="k", columns="tenant", values="quota")
        bv = df.pivot(index="k", columns="tenant", values="V_declared")
        r = df[df["tenant"] == i].set_index("k")
        for k in ks:
            vic = [j for j in range(n) if j != i]
            dq = bq.loc[k].to_numpy() - tq.loc[k].to_numpy()
            dv = bv.loc[k].to_numpy() - tv.loc[k].to_numpy()
            rows.append(dict(k=int(k), tenant=i, cls=ct.cls[i], theta=float(th.loc[k, i]),
                             z_star=float(r.loc[k, "br_z_star"]), gain=float(r.loc[k, "br_gain"]),
                             U_truth=float(r.loc[k, "br_U_truth"]),
                             U_star=float(r.loc[k, "br_U_truth"] + r.loc[k, "br_gain"]),
                             exact=bool(r.loc[k, "br_exact"]), cert=float(r.loc[k, "br_cert"]),
                             q_truth=float(tq.loc[k, i]), q_br=float(bq.loc[k, i]), dq_dev=float(dq[i]),
                             dq_victims=float(dq[vic].sum()), dV_victims_decl=float(dv[vic].sum()),
                             dV_dev_decl=float(dv[i]), seconds=np.nan, source="replay"))
    return pd.DataFrame(rows)


def stage_records(w, ctrl, pr, ks, mech, shape) -> pd.DataFrame:
    """Per-interval best response of each tenant in turn against truthful others at the intervals ``ks``."""
    from slicemech.model.shapes import stack_shape, value
    from slicemech.pricing.response import best_response
    ct = w.contract
    n = ct.n
    beta, rho = stack_shape(shape)
    th = np.minimum(np.asarray(w.theta, float), ct.C)
    rules = interval_rules(w, ctrl, ks, mech, shape)
    rows = []
    for k in ks:
        rule, r = rules[int(k)]
        q_t = np.asarray(rule(th[k][None, :]))[0].astype(float)
        v_t = value(q_t, th[k], r, ct.pi, beta, rho, mech.Delta)
        for i in range(n):
            t0 = time.perf_counter()
            br = best_response(i, float(th[k, i]), np.delete(th[k], i), rule, price=pr.charge, ct=ct, shape=shape,
                               mech=mech, r=r, nu=pr.nu, tol=None)
            sec = time.perf_counter() - t0
            z = th[k].copy()
            z[i] = br.z_star
            q_b = np.asarray(rule(z[None, :]))[0].astype(float)
            v_b = value(q_b, th[k], r, ct.pi, beta, rho, mech.Delta)
            dq, dv = q_b - q_t, v_b - v_t
            vic = [j for j in range(n) if j != i]
            rows.append(dict(k=int(k), tenant=i, cls=ct.cls[i], theta=float(th[k, i]), z_star=float(br.z_star),
                             gain=float(br.gain), U_truth=float(br.U_truth), U_star=float(br.U_star),
                             exact=bool(br.exact), cert=float(br.cert), q_truth=float(q_t[i]), q_br=float(q_b[i]),
                             dq_dev=float(dq[i]), dq_victims=float(dq[vic].sum()),
                             dV_victims_decl=float(dv[vic].sum()), dV_dev_decl=float(dv[i]), seconds=sec,
                             source="stage"))
    return pd.DataFrame(rows)


def _finish_stage(df: pd.DataFrame, w, arm: str, seed, mech) -> pd.DataFrame:
    ct = w.contract
    t = df["tenant"].to_numpy()
    df["multiple"] = np.where(df["theta"] > 0, df["z_star"] / df["theta"].where(df["theta"] > 0, 1.0), np.nan)
    df["misreport"] = df["z_star"] != df["theta"]
    df["eps"] = EPS_REL * ct.pi[t] * ct.mu[t] * float(mech.Delta)      # H3a allowance 1e-8 pi_i mu_i Delta (HC-5)
    df.insert(0, "seed", -1 if seed_free(arm) else int(seed))
    df.insert(0, "arm", arm)
    df.insert(0, "date", w.date)
    df.insert(0, "window", w.id)
    return df


def job_e2_window(arm: str, seed: int, date: str, offset: int, replay: bool, stage_out: str, replay_out: str | None,
                  stage_limit: int | None = None):
    """E2 items 1-2 on one (window, arm): stage best responses at every 4th scored interval; on replay windows the
    truthful run and one best-response replay per deviator, with the stage records read from them."""
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    spec = spec_of(seed)
    cal, mech = _W["cal"], _W["mech"]
    w = get_window(date, offset)
    ctrl, pr = make_arm(arm, spec)
    ks = scored_every(w.warmup, w.K, BR_EVERY, stage_limit)
    if replay:
        tabs = {}
        t0 = time.perf_counter()
        truth = E.run_window(w, ctrl, pr, [Truthful() for _ in range(w.n)], D, mech, cal.shape,
                             price_columns="billed")
        tabs["truthful"] = truth
        for i in range(w.n):
            pols = [Truthful() for _ in range(w.n)]
            pols[i] = ScoredBestResponder(pr.charge, w.warmup)
            tabs[f"br{i}"] = E.run_window(w, ctrl, pr, pols, D, mech, cal.shape, price_columns="billed")
        sec = time.perf_counter() - t0
        out = []
        for prof, df in tabs.items():
            x = df.loc[:, list(REPLAY_COLS)].copy()
            x.insert(0, "profile", prof)
            out.append(x)
        rep = pd.concat(out, ignore_index=True)
        rep["deviator"] = rep["profile"].map(lambda p: -1 if p == "truthful" else int(p[2:])).astype(np.int8)
        rep.insert(0, "seed", -1 if seed_free(arm) else int(seed))
        rep.insert(0, "arm", arm)
        rep.insert(0, "date", w.date)
        rep.insert(0, "window", w.id)
        rep.attrs = {}
        rep["warmup"] = np.int16(w.warmup)
        rep["seconds"] = sec
        write_parquet(f32(rep, F32_COLS), Path(replay_out))
        devs = {i: tabs[f"br{i}"] for i in range(w.n)}
        st = _stage_from_tables(truth, devs, ks, w.contract, w.warmup)
    else:
        st = stage_records(w, ctrl, pr, ks, mech, cal.shape)
    write_parquet(_finish_stage(st, w, arm, seed, mech), Path(stage_out))


def job_e2_nonmono(kind: str, seed: int, date: str, offset: int, out: str):
    """H2b: own-quota path of every tenant on a 513-point grid of [0, C_i] at every scored interval (others truthful)."""
    spec = spec_of(seed)
    cal, mech = _W["cal"], _W["mech"]
    w = get_window(date, offset)
    net = _cached_net(spec.nets[kind])
    ct = w.contract
    th = np.minimum(np.asarray(w.theta, float), ct.C)
    ks = np.arange(w.warmup, w.K)
    rules = interval_rules(w, net, ks, mech, cal.shape)
    rows = []
    t0 = time.perf_counter()
    for k in ks:
        rule, _ = rules[int(k)]
        for i in range(ct.n):
            u = np.linspace(0.0, float(ct.C[i]), NONMONO_GRID)
            Z = np.repeat(th[k][None, :], u.size, axis=0)
            Z[:, i] = u
            q = np.asarray(rule(Z))[:, i].astype(np.int64)
            drop = np.maximum.accumulate(q) - q
            dec = np.diff(q) < 0
            j_t = int(np.searchsorted(u, th[k, i]))
            rows.append((int(k), i, int(drop.max()), int(dec.sum()), int(q[0]), int(q[-1]),
                         int(q[min(j_t, u.size - 1)]), float(th[k, i])))
    df = pd.DataFrame(rows, columns=["k", "tenant", "max_drop", "n_decreases", "q_at_0", "q_at_C", "q_near_theta",
                                     "theta"])
    df["nonmono"] = df["max_drop"] > 0
    df["cls"] = [ct.cls[i] for i in df["tenant"]]
    df.insert(0, "seed", int(seed))
    df.insert(0, "net", kind)
    df.insert(0, "date", w.date)
    df.insert(0, "window", w.id)
    df["seconds"] = time.perf_counter() - t0
    for c in ("max_drop", "n_decreases", "q_at_0", "q_at_C", "q_near_theta"):
        df[c] = df[c].astype(np.int16)
    write_parquet(f32(df, ("theta",)), Path(out))


# ---------------------------------------------------------------------------------------------- bandits
class PersistentMemo:
    """SQLite cache of window evaluations keyed by (window key, multiplier profile), shared by concurrent runs."""

    def __init__(self, path: Path):
        self.path = sio.confine(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path), timeout=600)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS memo (key TEXT PRIMARY KEY, val TEXT)")
        self.con.commit()
        self.hits = self.misses = 0

    def get(self, key: str):
        row = self.con.execute("SELECT val FROM memo WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, val: dict) -> None:
        self.con.execute("INSERT OR IGNORE INTO memo (key, val) VALUES (?, ?)", (key, json.dumps(val)))
        self.con.commit()

    def count(self) -> int:
        return int(self.con.execute("SELECT COUNT(*) FROM memo").fetchone()[0])


def bandit_evaluator(arm: str, spec: AssetSpec, memo: PersistentMemo):
    """``evaluate(window_key, m) -> dict`` for ``tenants.bandits``: every tenant reports ``min(m_i theta_i, C_i)``
    for the whole window (record=False, billed price only); declared utility summed over the scored intervals is
    the bandit reward input (as ``sim.episode.make_bandit_evaluator``), plus realized-welfare diagnostics."""
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import multiplier_profile
    ctrl, pr = make_arm(arm, spec)
    cal, mech = _W["cal"], _W["mech"]

    def evaluate(window_key, m):
        m = np.asarray(m, float)
        key = f"{window_key}|" + ",".join(f"{x:g}" for x in m)
        hit = memo.get(key)
        if hit is not None:
            memo.hits += 1
            return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in hit.items()}
        memo.misses += 1
        date, off = window_key.rsplit("_o", 1)
        w = get_window(date, int(off))
        df = E.run_window(w, ctrl, pr, multiplier_profile(m), D, mech, cal.shape, record=False,
                          price_columns="billed")
        sc = df[df["scored"]]
        tot = sc.groupby("tenant")[["utility_declared", "utility_realized", "on_time_Mbit", "price",
                                    "realized_value"]].sum()
        per_k = sc.groupby("k")[["realized_value", "V_declared"]].sum().mean()
        res = dict(utility=tot["utility_declared"].to_numpy(), utility_realized=tot["utility_realized"].to_numpy(),
                   on_time_Mbit=tot["on_time_Mbit"].to_numpy(), price=tot["price"].to_numpy(),
                   realized_value=tot["realized_value"].to_numpy(),
                   W_realized=float(per_k["realized_value"]), W_declared=float(per_k["V_declared"]))
        memo.put(key, {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()})
        return res

    return evaluate


def job_e2_bandit(design: str, seed: int, algo: str, run: int, episodes: int, window_keys_: list, memo_path: str,
                  out: str):
    from slicemech.tenants import bandits as Bd
    spec = spec_of(seed)
    memo = PersistentMemo(Path(memo_path))
    ev = bandit_evaluator(design, spec, memo)
    n = _W["cal"].n
    t0 = time.perf_counter()
    br = Bd.run_learning_tenants(ev, list(window_keys_), n, algo=algo, episodes=episodes, seed=run)
    extras = {f"x_{k}": v for k, v in br.extras.items()}
    write_npz(Path(out), arm=br.arm, m=br.m, gain=br.gain, reward=br.reward, greedy_m=br.greedy_m,
              utility=br.utility, final_prob=br.final_prob, arms=br.arms, window=np.asarray(br.window),
              meta=np.asarray(json.dumps(dict(design=design, seed=seed, algo=algo, run=run, episodes=episodes,
                                              memo_hits=memo.hits, memo_misses=memo.misses,
                                              seconds=time.perf_counter() - t0, pool=len(window_keys_)))),
              **extras)


def job_loads(date: str, offset: int, out: str):
    """D2/D3: realized per-window loads: ``offered_load`` = mean over ALL K intervals of the summed measured demand /
    (B rbar_TRAIN) (the E0/E1 D2 quantity and subgroup variable), ``offered_load_scored`` the same over the scored
    intervals, and ``capacity_load_scored`` = ``eval.headroom.window_load`` (scored mean demand / (B x the window's
    mean true per-PRB rate)); all exogenous (no controller outcome). Round-3 review: one name, one definition."""
    from slicemech.eval.headroom import offered_load, window_load
    cal, mech = _W["cal"], _W["mech"]
    w = get_window(date, offset)
    cap = float(window_load(w, mech))
    write_parquet(pd.DataFrame([dict(window=w.id, date=w.date, offset=int(offset),
                                     offered_load=offered_load(w, cal.r_bar, mech),
                                     offered_load_scored=offered_load(w, cal.r_bar, mech, scored_only=True),
                                     capacity_load_scored=cap, counts=json.dumps(list(w.meta.get("counts", []))))]),
                  Path(out))


# ============================================================================================ E2 plan
#: single-core costs (s) measured in the 2-window TRAIN pilot (2026-10-19, K = 200, machine load 9-15, smoke nets):
#: per best response (stage and replay jobs) and per billed-price episode (bandit evaluations, replays)
BR_COST = {"prot_fixed": 0.27, "unc_fixed": 2.0, "unc_usage": 0.12, "unc_env": 0.6, "prot_full": 0.05}
EPISODE_COST = {"prot_full": 4.5, "prot_fixed": 4.5, "unc_fixed": 4.0, "unc_usage": 0.9, "unc_env": 20.0}


def e2_jobs(args, root: Path, specs: dict) -> list[Job]:
    keys = run_keys(args)
    # smoke: replay the first window only, so the direct stage path runs on the other one
    rkeys = set(replay_keys(keys, args.replay_per_date)) if not args.smoke else set(keys[:1])
    K, warm = (SMOKE_K, SMOKE_WARMUP) if args.smoke else (200, 40)
    n_ks = len(scored_every(warm, K, BR_EVERY, args.stage_limit))
    n = 3
    mod = "e2_manipulation"
    jobs = []
    for d, o in keys:
        lo = root / "loads" / f"{wkey(d, o)}.parquet"
        jobs.append(Job(f"loads {wkey(d, o)}", f"{mod}:job_loads", dict(date=d, offset=o, out=str(lo)), str(lo), 1.0))
    for seed in args.seeds:
        for arm in args.arms:
            for d, o in keys:
                rep = (d, o) in rkeys
                stage_out = root / "stage" / arm / f"s{seed}" / f"{wkey(d, o)}.parquet"
                rep_out = root / "replay" / arm / f"s{seed}" / f"{wkey(d, o)}.parquet"
                if rep:
                    cost = n * (K - warm) * BR_COST[arm] + (n + 1) * EPISODE_COST[arm] * K / 200
                else:
                    cost = n * n_ks * BR_COST[arm]
                jobs.append(Job(f"e2 {arm} s{seed} {wkey(d, o)}{' replay' if rep else ''}", f"{mod}:job_e2_window",
                                dict(arm=arm, seed=seed, date=d, offset=o, replay=rep, stage_out=str(stage_out),
                                     replay_out=str(rep_out) if rep else None, stage_limit=args.stage_limit),
                                str(rep_out if rep else stage_out), cost))
                if rep:      # the stage file is written last, so the job is complete only when it exists
                    jobs[-1].out = str(stage_out)
        for kind in args.nonmono_nets:
            for d, o in keys:
                jobs.append(Job(f"nonmono {kind} s{seed} {wkey(d, o)}", f"{mod}:job_e2_nonmono",
                                dict(kind=kind, seed=seed, date=d, offset=o,
                                     out=str(root / "nonmono" / kind / f"s{seed}" / f"{wkey(d, o)}.parquet")),
                                str(root / "nonmono" / kind / f"s{seed}" / f"{wkey(d, o)}.parquet"),
                                (6.0 if kind == "unconstrained" else 1.0) * (K - warm) / 160))
        pool = [wkey(d, o) for d, o in keys]
        for design in args.bandit_designs:
            # the memo is keyed by (window, multiplier profile) only, so its file is tied to the design's assets and
            # the code: evaluations of other models are never reused (round-3 review)
            assets = {k: specs[seed].sha256.get(k) for k in sorted(arm_needs(design))} if seed in specs else {}
            mtag = sio.digest(dict(assets=assets, code=code_fingerprint(*E2_MODULES)))[:12]
            memo = root / "bandit" / design / f"s{seed}" / f"memo_{mtag}.sqlite"
            for algo in BANDIT_ALGOS:
                for run in range(args.bandit_runs):
                    out = root / "bandit" / design / f"s{seed}" / f"{algo}_run{run}.npz"
                    cost = args.bandit_episodes * EPISODE_COST[design] * K / 200     # ~1 memo miss per episode
                    jobs.append(Job(f"bandit {design} s{seed} {algo} run{run}", f"{mod}:job_e2_bandit",
                                    dict(design=design, seed=seed, algo=algo, run=run, episodes=args.bandit_episodes,
                                         window_keys_=pool, memo_path=str(memo), out=str(out)), str(out), cost))
    return jobs


# ============================================================================================ E2 summary
def boot(values, dates, seed: int = 0) -> dict:
    """Date-cluster bootstrap (two-stage, 10,000 replicates, percentile 95% CI) of a per-window mean."""
    from slicemech.eval.stats import cluster_bootstrap
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return dict(estimate=float("nan"), lo=float("nan"), hi=float("nan"), n_windows=0, n_dates=0)
    r = cluster_bootstrap(v[ok], np.asarray(dates)[ok], n_boot=10_000, seed=seed)
    return dict(estimate=r.estimate, lo=r.lo, hi=r.hi, n_windows=r.n_windows, n_dates=r.n_dates)


def e2_summary(root: Path, args) -> dict:
    from slicemech.eval.metrics import victim_effects
    out = dict(split=args.split, smoke=args.smoke, decisions=DECISIONS)
    stage = read_parquets(sorted(root.glob("stage/*/*/*.parquet")))
    loads = read_parquets(sorted(root.glob("loads/*.parquet")))
    if len(loads):
        write_parquet(loads, root / "loads.parquet")
        out["loads"] = dict(offered_mean=float(loads["offered_load_scored"].mean()),
                            capacity_mean=float(loads["capacity_load_scored"].mean()), windows=int(len(loads)))
    if len(stage):
        write_parquet(stage, root / "stage_all.parquet")
        h2a = {}
        for (arm, cls), g in stage.groupby(["arm", "cls"]):
            pw = g.groupby(["window", "date"])["gain"].mean().reset_index()
            pos = g["gain"] > np.maximum(g["eps"], np.where(np.isfinite(g["cert"]), g["cert"], 0.0))
            h2a[f"{arm}|{cls}"] = dict(
                n=int(len(g)), mean_gain=float(g["gain"].mean()), max_gain=float(g["gain"].max()),
                share_positive=float(pos.mean()), share_misreport=float(g["misreport"].mean()),
                mean_multiple_when_misreport=float(g.loc[g["misreport"], "multiple"].mean()),
                mean_dq_dev=float(g["dq_dev"].mean()), mean_dq_victims=float(g["dq_victims"].mean()),
                mean_dV_victims_decl=float(g["dV_victims_decl"].mean()),
                exact_share=float(g["exact"].mean()), ci_mean_gain=boot(pw["gain"], pw["date"]))
        out["H2a_stage_best_response"] = h2a
    # replays -> victim effects (D4)
    rep_files = sorted(root.glob("replay/*/*/*.parquet"))
    vrows = []
    for f in rep_files:
        rep = pd.read_parquet(f)
        warm = int(rep["warmup"].iloc[0])
        tr = rep[rep["profile"] == "truthful"]
        for prof, g in rep[rep["profile"] != "truthful"].groupby("profile"):
            i = int(prof[2:])
            ve = victim_effects(tr, g, i, warm)
            sc_g, sc_t = g[g["k"] >= warm], tr[tr["k"] >= warm]
            dev = sc_g[sc_g["tenant"] == i]
            vrows.append(dict(window=rep["window"].iloc[0], date=rep["date"].iloc[0], arm=rep["arm"].iloc[0],
                              seed=int(rep["seed"].iloc[0]), deviator=i,
                              mean_gain=float(dev["br_gain"].mean()),
                              share_misreport=float((dev["report"] != dev["theta_adm"].astype(float)).mean()),
                              d_on_time_dev=float(dev["on_time_Mbit"].mean()
                                                  - sc_t[sc_t["tenant"] == i]["on_time_Mbit"].mean()),
                              **ve))
    if vrows:
        vic = pd.DataFrame(vrows)
        write_parquet(vic, root / "victims.parquet")
        h2c = {}
        for (arm, i), g in vic.groupby(["arm", "deviator"]):
            # checkpoint seeds are averaged within a window before the bootstrap (experiment_design.md §7)
            g = g.groupby(["window", "date"])[["d_on_time_victims", "d_miss_pct_victims", "d_value_victims",
                                               "dW_realized", "mean_gain"]].mean().reset_index()
            h2c[f"{arm}|deviator{i}"] = dict(
                windows=int(len(g)), d_on_time_victims=boot(g["d_on_time_victims"], g["date"]),
                d_miss_pct_victims=boot(g["d_miss_pct_victims"], g["date"]),
                d_value_victims=boot(g["d_value_victims"], g["date"]),
                dW_realized_descriptive=float(g["dW_realized"].mean()), mean_gain=float(g["mean_gain"].mean()))
        out["H2c_victims"] = h2c
    nm = read_parquets(sorted(root.glob("nonmono/*/*/*.parquet")))
    if len(nm):
        write_parquet(nm, root / "nonmono_all.parquet")
        h2b = {}
        env = stage[stage["arm"] == "unc_env"] if len(stage) else pd.DataFrame()
        for (net, cls), g in nm.groupby(["net", "cls"]):
            d = dict(tenant_intervals=int(len(g)), nonmono_count=int(g["nonmono"].sum()),
                     rate=float(g["nonmono"].mean()), max_prb_drop=int(g["max_drop"].max()))
            if net == "unconstrained" and len(env):
                j = g.merge(env[["window", "k", "tenant", "gain", "eps"]], on=["window", "k", "tenant"])
                jn = j[j["nonmono"]]
                d["exploitable_share_envelope"] = (float((jn["gain"] > jn["eps"]).mean()) if len(jn) else None)
                d["checked_with_envelope_br"] = int(len(jn))
            h2b[f"{net}|{cls}"] = d
        out["H2b_nonmonotonicity"] = h2b
    bfiles = sorted(root.glob("bandit/*/*/*.npz"))
    bnd = {}
    for f in bfiles:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        last = min(50, z["arm"].shape[0])
        bnd[f"{meta['design']}|s{meta['seed']}|{meta['algo']}|run{meta['run']}"] = dict(
            final_greedy_m=z["greedy_m"][-1].tolist(),
            last50_mean_m=z["m"][-last:].mean(0).tolist(),
            last50_share_truth=(z["m"][-last:] == 1.0).mean(0).tolist(),
            last50_mean_gain=z["gain"][-last:].mean(0).tolist(),
            episodes=int(z["arm"].shape[0]), memo_hits=meta["memo_hits"], memo_misses=meta["memo_misses"])
    if bnd:
        out["bandits"] = bnd
    sio.write_json(root / "summary.json", out)
    return out


# ============================================================================================ main
def e2_parser() -> argparse.ArgumentParser:
    """The E2 command line (shared with :func:`e2_completeness`, which rebuilds the pre-declared job list)."""
    ap = common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--arms", nargs="+", default=list(E2_ARMS), choices=list(E2_ARMS) + ["prot_full"])
    ap.add_argument("--replay-per-date", type=int, default=2)
    ap.add_argument("--stage-limit", type=int, default=None, help="max stage intervals per window (smoke: 3)")
    ap.add_argument("--nonmono-nets", nargs="*", default=["unconstrained", "protected"])
    ap.add_argument("--bandit-designs", nargs="*", default=list(BANDIT_DESIGNS))
    ap.add_argument("--bandit-episodes", type=int, default=None)
    ap.add_argument("--bandit-runs", type=int, default=None)
    return ap


def e2_defaults(args):
    """Fill the defaults that depend on ``--smoke`` (after the TEST option check); returns ``args``."""
    if args.smoke:
        args.stage_limit = args.stage_limit or 3
    args.bandit_episodes = args.bandit_episodes or (SMOKE_EPISODES if args.smoke else BANDIT_EPISODES)
    args.bandit_runs = args.bandit_runs or (SMOKE_RUNS if args.smoke else BANDIT_RUNS)
    return args


def e2_completeness(split: str, smoke: bool, specs: dict, *, tag: str | None = None, root=None) -> dict:
    """Status of every pre-declared E2 job of a split (the default options, i.e. the frozen protocol on TEST) under
    the CURRENT provenance fingerprints (assets of ``specs``, code, calibration): ``complete`` is True only when every
    job is 'done' (output present with a sidecar carrying the current fingerprint), which also means one code
    version. E4 refuses to start on TEST otherwise (pre-freeze audit F9, item 35)."""
    argv = ["--smoke"] if smoke else ["--split", split]
    if tag:
        argv += ["--tag", tag]
    a = e2_defaults(e2_parser().parse_args(argv))
    a.split = a.split or "train"
    root = Path(root) if root else out_root("e2", a.split, a.smoke, a.tag)
    jobs = attach_provenance(e2_jobs(a, root, specs), specs, split=a.split, smoke=a.smoke, modules=E2_MODULES)
    st = collections.Counter(job_status(j) for j in jobs)
    codes = set()
    for j in jobs:
        pp = prov_path(j.out)
        if pp.is_file():
            try:
                codes.add(sio.read_json(pp).get("code"))
            except (ValueError, OSError):
                codes.add(None)
    return dict(root=str(root), split=a.split, jobs=len(jobs), done=st["done"], pending=st["pending"],
                stale=st["stale"], code_versions=len(codes), current_code=code_fingerprint(*E2_MODULES),
                complete=bool(jobs) and st["done"] == len(jobs) and len(codes) == 1)


def main(argv=None):
    ap = e2_parser()
    args = ap.parse_args(argv)
    root = start(args, "e2", ap)                 # TEST options are checked before any default is filled in
    e2_defaults(args)
    arms_all = set(args.arms) | set(args.bandit_designs)
    if args.nonmono_nets:
        arms_all |= {"unc_fixed" if k == "unconstrained" else "prot_full" for k in args.nonmono_nets}
    if args.plan:
        specs = {s: AssetSpec(seed=s, nets={}) for s in args.seeds}
        run_jobs(e2_jobs(args, root, specs), workers=args.workers, initargs=(), errors_path=root / "errors.jsonl",
                 plan=True)
        return
    specs = resolve_assets(sorted(arms_all), args.seeds, smoke=args.smoke, ckpt_dir=args.ckpt_dir,
                           overrides=args.ckpt, train_missing=args.train_missing, workers=args.workers,
                           n_smoke=args.smoke_windows, test_freeze_path=args.freeze_path)
    write_config(root, args, specs, dict(experiment="E2", br_every=BR_EVERY, nonmono_grid=NONMONO_GRID,
                                         code_fingerprint=code_fingerprint(*E2_MODULES), freeze=args.freeze_path,
                                         replay_windows=[list(k) for k in replay_keys(
                                             run_keys(args), args.replay_per_date)] if not args.smoke else "first smoke window"))
    jobs = attach_provenance(e2_jobs(args, root, specs), specs, split=args.split, smoke=args.smoke,
                             modules=E2_MODULES)
    t0 = time.perf_counter()
    res = run_jobs(jobs, workers=args.workers, initargs=(specs, args.split, args.smoke, args.test_freeze),
                   errors_path=root / "errors.jsonl")
    res["wall_s"] = time.perf_counter() - t0
    print(json.dumps(res), flush=True)
    if not args.no_summary:
        s = e2_summary(root, args)
        print(json.dumps({k: v for k, v in s.items() if k != "decisions"}, indent=1, default=str)[:6000])
    if res["failed"]:
        raise SystemExit(f"{res['failed']} job(s) failed; see {root / 'errors.jsonl'}")


if __name__ == "__main__":
    # run through the importable module so that the pool's initializer, the job functions and the worker state
    # (``_W``) all live in ``e2_manipulation`` (with spawn, ``__main__`` would be a different module object)
    import e2_manipulation as _self
    _self.main()
