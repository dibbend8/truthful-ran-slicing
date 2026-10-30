"""E0 - setup diagnostics: regime, realized loads, V calibration and one worked interval (experiment_design.md §7 E0).

Run from ``code/`` (PYTHONDONTWRITEBYTECODE=1, anaconda python):

    python experiments/e0_setup.py --split dev [--config default] [--workers 8]
    python experiments/e0_setup.py --smoke                       # 2 short TRAIN windows (see --smoke below)
    python experiments/e0_setup.py --split test --test-freeze ../prereg/PREREG.md   # only after the freeze

This module also holds the helpers shared by the E0 and E1 drivers (the TEST gate, window composition, model
discovery and checks, atomic parquet output, quick PPO nets for a standalone ``--smoke``); ``e1_value.py`` imports
them. Configurations, calibrations, model files and static fits are resolved through ``experiments/registry.py``,
the layout ``experiments/train.py`` writes: configurations ``default, d1, d5, d15, rho0.5, rho0.7, rho1.1, lead0,
lead1.05, n6, n9``, each with its own TRAIN calibration (``n3_default``; load and lead cells ``n3_rho<rho>`` /
``n3_lead<lead>``; ``n6_default``, ``n9_default``), models ``results/models/<config>/<arm>/seed<k>/model.pt``.

What E0 computes, per window of the split (every pre-declared date x offset), in parallel over windows:

1. **Realized loads** (orchestrator decision D2). ``offered_load`` = sum_i mean_k m_ik / (B r_bar_TRAIN) over all K
   intervals (the quantity the D2 rule matches to rho on each date), ``capacity_load`` =
   ``eval.headroom.window_load`` (scored intervals over B x the window's own mean true per-PRB rate), per-tenant
   offered Mbit/s and the D2 counts; the summary adds per-date offered loads and their deviation from rho.
2. **Regime diagnostics** (``data.calibrate.regime_diagnostics``; truthful reports z = min(theta, C), allocator
   rate rate_hat, equal weights, scored intervals) per window; the pooled value is the interval-weighted mean and
   is compared with ``spec/params.yaml: regime_targets`` (descriptive on DEV and TEST; decisions are TRAIN-only).
3. **Rate distribution**: value counts of the true per-PRB rate (scored intervals; CQI gives a finite set).
4. **V calibration** (experiment_design.md §3, condition C3). For every scored interval k and tenant i with
   z_ik > 0 and every grid factor c in ``VCAL_GRID`` = 0.25, 0.5, ..., 2: quota q = c z_ik / rhat_ik (PRB,
   fractional, clipped to [0, B]); x = V_i(q, z_ik) / pi_i (declared expected served volume, Mbit, public burst
   shape, stale rate) and y = on-time Mbit of tenant i in interval k when it holds q, from the fluid replay with
   TRUE per-ms arrivals and TRUE rates, started from the backlog of the equal-weights truthful path (quotas are
   isolated, so tenant i's service depends on its own quota only). Stored as sufficient statistics per (window,
   tenant, c); the summary gives per class the OLS slope, intercept and R^2 of y on x with date-cluster bootstrap
   CIs, the calibration curve (mean x, mean y per c) and the flag R^2 < ``RsqFlag`` = 0.5.
5. **Worked interval** (main process): the own quota, the full-decision price and the fixed-priority price
   versus the own report u in [0, C_i] (others truthful) for the protected map and the unconstrained map of one
   interval, plus each map's exact own-quota staircase and the declared utilities V_i(Q_i(u), theta_i) - price.
   For the unconstrained map the "full" price is the envelope formula on a 4096-point grid path (not truthful,
   uncertified; ``pricing.envelope.price_nonmonotone``). Pre-declared, illustrative selection: the first window
   of the split, the first URLLC-like tenant, and the scored interval maximising, in this order, (the
   unconstrained own-quota path drops somewhere on a 257-point grid, the protected premium T(theta) at truth, the
   size of that drop); ``--worked-window/--worked-k/--worked-tenant`` override it. Nets: the registry models
   ``<models-root>/<config>/{protected,unconstrained}/seed<--seed>/model.pt`` (their record's SHA-256 and parameter
   digest are checked); without them this stage is skipped and the summary says so.

``--smoke``: the two TRAIN windows of ``registry.SMOKE_TRAIN_WINDOWS`` cut to K = 60 s (20 s warm-up), the models of
the training driver's smoke run (``results/raw/smoke_train/<split>/models``) when they exist, else 2-iteration PPO
nets trained here (h = 3) and saved under ``<out>/ckpt``.

Outputs (float32 where exactness is not needed; zstd parquet):
``results/raw/<split>/e0/<config>/`` (``results/raw/smoke/e0/<config>/`` with ``--smoke``):
``parts/<date>_o<offset>__{win,vcal,rates}.parquet`` (per-window work units; a rerun skips windows whose three
parts exist: resumable), merged ``windows.parquet``, ``vcal.parquet``, ``rates.parquet``,
``worked_interval.parquet`` (curves), ``worked_interval.json`` (its metadata), ``worked_staircase.parquet``,
``summary.json``. Nothing is written to ``results/summary/``: ``experiments/analyze.py`` turns these raw files (the
worked interval included) into ``e0.json`` in the split's summary directory (TEST: ``results/summary/``; DEV:
``results/summary/dev/``), which the figure scripts read (integrator, 2026-10-19: a DEV run used to write
``results/summary/e0_dev_<config>.json`` into the TEST-only summary directory).

TEST is refused unless the frozen pre-registration exists and passes the freeze gate
(``data.puffer_fetch.require_freeze_for_test_reads``): ``--test-freeze`` (default ``prereg/PREREG.md``); the check
runs before any window is composed. On TEST the worked-interval checkpoints must be listed in the freeze block with
their SHA-256.

Orchestrator decisions (binding, 2026-10-19; code/STATUS.md §0), as used by the E0/E1 drivers:

* **D1** GBR fractions x 0.8 (g/mu = 0.40 video, 0.64 URLLC-like, 0.16 mMTC-like): carried by the calibration's
  contract (``data/calibration/n*_default.json``); TRAIN-only decision.
* **D2** per-date offered-load normalization on every split: windows are composed with ``Calibration.counts_for``
  (counts chosen from offered traffic only, r_bar_TRAIN, shares 0.6/0.3/0.1 split equally within a class at
  n = 6, 9); load sweeps scale rho (nested counts); realized per-window loads are reported (``offered_load``,
  ``capacity_load``).
* **D3** gate G4 dropped; H1c/H1d secondary with CIs; offered-load subgroups for H1a/H1c/H1d use the exogenous
  per-window ``offered_load`` written here and by E1.
* **D4** harm is measured on the honest tenants' own outcomes (``eval.metrics.victim_effects``): E0/E1 store
  per-tenant on-time Mbit, deadline-miss % and realized value; total welfare is descriptive.
* **D5** E5 protocol (interleaved 2-s halves, alpha = 1 reference, 4 + 4 windows per date): not used by E0/E1.
* **D6** learner objective = lookahead return with h = 3 (``PPOConfig.horizon=3``): the ``--smoke`` nets are
  trained with ``horizon=3`` explicitly and E1's oracle uses ``horizon=3``; the paper calls it short-horizon RL.
* **D7** E4b frontier (kappa scaled by s in {0, 0.5, 1, 2}): not part of E0/E1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True
_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import registry as R  # noqa: E402
from slicemech import io as sio  # noqa: E402

RESULTS = sio.ROOT / "results"
PREREG_FREEZE = sio.ROOT / "prereg" / "PREREG.md"

#: Smoke mode: the two TRAIN windows of ``registry.SMOKE_TRAIN_WINDOWS``, shortened to K = 60 s with 20 s warm-up.
SMOKE_WINDOWS = tuple((str(d), int(o)) for d, o in R.SMOKE_TRAIN_WINDOWS)
SMOKE_K, SMOKE_WARMUP = 60, 20
SMOKE_PPO_ITERS = 2
#: Horizon of the learner's lookahead return (orchestrator decision D6), passed explicitly.
PPO_HORIZON = R.HORIZON

#: Quota grid of the V calibration, as multiples of theta / rhat (experiment_design.md §3).
VCAL_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
RSQ_FLAG = 0.5        # spec/params.yaml experiment_settings.RsqFlag
WORKED_GRID = 257     # own-report grid points of the worked interval (plus the truth)


# ============================================================================================ configurations
def get_config(name: str) -> "R.RunConfig":
    """The named configuration of ``registry.CONFIGS`` (the one the training driver trained)."""
    if name not in R.CONFIGS:
        raise SystemExit(f"unknown config {name!r}; known: {', '.join(R.CONFIGS)}")
    return R.CONFIGS[name]


def config_record(cfg) -> dict:
    """JSON-able configuration (``registry.config_record``: name, sweep, n, d, rho, lead_s, calibration)."""
    return R.config_record(cfg)


def smoke_calib_root() -> Path | None:
    """Calibration directory of the training driver's smoke runs (sweep calibrations), if one exists."""
    for s in ("train", "dev"):
        d = R.SMOKE_DIR / s / "calibration"
        if d.is_dir():
            return d
    return None


def load_cal(cfg, calib_root=None):
    """The configuration's TRAIN calibration. Sweep calibrations (``registry.CALIB_SPECS``: n3_rho*, n3_lead*) are
    read from ``calib_root`` when given (default ``data/calibration``); the n3/n6/n9 defaults always from
    ``data/calibration`` (as ``train.Ctx.calib_dir``)."""
    root = calib_root if (calib_root and cfg.calibration in R.CALIB_SPECS) else None
    return R.load_calibration(cfg.calibration, root)


def calibration_sha(cfg, calib_root=None) -> str:
    root = calib_root if (calib_root and cfg.calibration in R.CALIB_SPECS) else None
    return sio.file_sha256(R.calibration_path(cfg.calibration, root))


# ============================================================================================ TEST gate
def test_gate(split: str, test_freeze=None) -> str | None:
    """Refuse TEST unless the frozen pre-registration exists and passes the freeze gate; returns its path (or None
    for TRAIN/DEV). Runs before any TEST source is opened."""
    if split != "test":
        return None
    path = Path(test_freeze) if test_freeze else PREREG_FREEZE
    if not path.is_file():
        raise PermissionError(f"refusing --split test: the frozen pre-registration {path} does not exist (TEST runs "
                              "happen only after the pre-registration freeze)")
    R.require_split("test", test_freeze=str(path))
    return str(path.resolve())


def verify_frozen_checkpoints(paths, freeze_file) -> None:
    """On TEST every model / static fit used must be listed in the freeze block with its SHA-256 (PermissionError).
    One shared check for every driver: ``registry.verify_frozen``."""
    R.verify_frozen(paths, freeze_file)


# ============================================================================================ windows
def window_keys(split: str, smoke: bool = False, dates=None, limit: int | None = None) -> list[tuple[str, int]]:
    """(date, offset_s) work units: every pre-declared window of the split (``SMOKE_WINDOWS`` with ``smoke``)."""
    from slicemech.data import compose as _compose, puffer
    if smoke:
        keys = list(SMOKE_WINDOWS)
    else:
        ds = puffer.split_dates(split) if not dates else tuple(dates)
        bad = [d for d in ds if puffer.split_of_date(d) != split]
        if bad:
            raise ValueError(f"dates {bad} are not in split {split}")
        keys = [(d, int(o)) for d in ds for o in _compose.window_offsets(split)]
    return keys[:limit] if limit else keys


def compose(cal, split: str, date: str, offset: int, *, smoke: bool = False, test_freeze=None):
    """Compose one window with the configuration's calibration (D2 per-date counts at the calibration's rho and lead;
    TEST only with the freeze; smoke: K = 60 s, 20 s warm-up)."""
    kw = {}
    if smoke:
        kw.update(K=SMOKE_K, warmup=SMOKE_WARMUP)
    if split == "test":
        kw["test_freeze"] = test_freeze
    return cal.window(split, date, int(offset), **kw)


def key_of(date: str, offset: int) -> str:
    return f"{date}_o{int(offset):05d}"


def out_dir(exp: str, split: str, config: str, smoke: bool = False) -> Path:
    """``results/raw/<split>/<exp>/<config>`` (``results/raw/smoke/<exp>/<config>`` in smoke mode)."""
    return RESULTS / "raw" / ("smoke" if smoke else split) / exp / config


def write_parquet(df: pd.DataFrame, path) -> Path:
    """Atomic zstd parquet write to a confined path (attrs dropped)."""
    p = sio.confine(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.attrs = {}
    tmp = p.with_name(p.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, p)
    return p


def f32(df: pd.DataFrame, exclude=()) -> pd.DataFrame:
    """Cast float64 columns to float32 (compact raw tables)."""
    for c in df.columns:
        if c not in exclude and df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


# ============================================================================================ archive
#: Name of the folder that receives every output a driver would overwrite or delete (pre-freeze audit F7/PV-10;
#: PREREG §8: "both the original and the corrected numbers are kept in the evidence").
ARCHIVE_DIR = "archive"


def archive_stamp() -> str:
    """UTC time stamp naming one invocation's archive folder (``YYYYmmddTHHMMSSZ``)."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def archive_anchor(root) -> Path:
    """Folder that holds the archive of the outputs below ``root``: the split folder ``results/raw/<split>`` when
    ``root`` lies below a ``results/raw`` folder (so the archive sits next to the experiment folders and never inside
    one: the analysis globs ``<experiment>/**`` and E0/E1 treat every sub-folder as a configuration), else
    ``root``'s parent (e.g. a ``--out`` folder or a test directory)."""
    root = Path(root).resolve()
    for p in (root, *root.parents):
        if p.parent.name == "raw" and p.parent.parent.name == "results":
            return p
    return root.parent


def archive_files(files, root, *, stamp: str | None = None, reason: str = "", copy: bool = False) -> Path | None:
    """Move (``copy=True``: copy) every existing file of ``files`` to
    ``<archive_anchor(root)>/archive/<stamp>/<path relative to the anchor>`` instead of deleting or overwriting it.
    An already archived file is never overwritten (a ``.dup<i>`` name is used). Each file gets one line in
    ``<archive>/<stamp>/archive_log.jsonl`` (source, destination, reason, time). Returns the archive folder, or
    None when nothing existed."""
    import shutil
    anchor = archive_anchor(root)
    base = anchor / ARCHIVE_DIR / (stamp or archive_stamp())
    moved = []
    for f in files:
        f = Path(f)
        if not f.is_file():
            continue
        try:
            rel = f.resolve().relative_to(anchor)
        except ValueError:
            rel = Path("_outside") / f.name
        dest = sio.confine(base / rel, root=anchor)
        i = 0
        while dest.exists():
            i += 1
            dest = base / rel.parent / f"{rel.stem}.dup{i}{rel.suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(f, dest)
        else:
            shutil.move(str(f), str(dest))
        moved.append(dict(src=str(f), dst=str(dest), reason=reason, copy=bool(copy),
                          time=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid()))
    if not moved:
        return None
    with open(base / "archive_log.jsonl", "a") as fh:            # one short line per file (O_APPEND)
        for m in moved:
            fh.write(json.dumps(m) + "\n")
    return base


# ============================================================================================ models
#: Learned arms of the training registry (``control.nets`` kinds of the same names).
NET_KINDS = tuple(R.ARMS)


def find_models(models_root, config: str, arm: str) -> dict[int, Path]:
    """Finished models of the training registry: ``<root>/<config>/<arm>/seed<k>/model.pt`` whose ``config.json``
    has status 'done' (tuning runs below ``tune/`` are never picked) -> {seed: path}."""
    base = Path(models_root) / config / arm
    out = {}
    if base.is_dir():
        for d in base.iterdir():
            m = re.fullmatch(r"seed(\d+)", d.name)
            if m and (d / R.MODEL_FILE).is_file() and R.read_done(d / R.CONFIG_FILE) is not None:
                out[int(m.group(1))] = d / R.MODEL_FILE
    return dict(sorted(out.items()))


def find_checkpoints(ckpt_dir, kind: str) -> dict[int, Path]:
    """Flat smoke checkpoints written by :func:`quick_nets`: ``<dir>/<kind>_seed<s>.pt`` -> {seed: path}."""
    d = Path(ckpt_dir)
    out = {}
    if d.is_dir():
        for p in d.glob(f"{kind}_seed*.pt"):
            m = re.fullmatch(rf"{re.escape(kind)}_seed(\d+)\.pt", p.name)
            if m:
                out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def load_checked_net(path, d: int):
    """``control.nets.load_net`` with checks: a registry model must match its record (file SHA-256 and parameter
    digest, as ``registry.load_model``) and record the D6 lookahead horizon 3 when its spec names one, and the net
    must carry an operator forecaster fitted for delay d."""
    from slicemech.control import nets
    path = Path(path)
    net = nets.load_net(path)
    rec = R.read_done(path.parent / R.CONFIG_FILE) if path.name == R.MODEL_FILE else None
    if rec is not None:
        res = rec.get("result", {})
        if sio.file_sha256(path) != res.get("model_sha256") or nets.param_digest(net) != res.get("param_digest"):
            raise ValueError(f"{path} differs from the digests recorded in its {R.CONFIG_FILE}")
        spec = rec.get("spec") or (rec.get("copied_from") or {}).get("spec") or {}
        h = (spec.get("ppo") or {}).get("horizon")
        if h is not None and int(h) != PPO_HORIZON:                         # orchestrator decision D6
            raise ValueError(f"{path} was trained with lookahead horizon {h}, D6 requires {PPO_HORIZON}")
    fc = getattr(net, "forecaster", None)
    if fc is None:
        raise ValueError(f"model {path} carries no operator forecaster")
    if fc.horizon is not None and int(fc.horizon) != int(d):
        raise ValueError(f"model {path} was trained with forecaster horizon {fc.horizon}, config d = {d}")
    net.eval()
    return net


def model_paths(cfg, arms, models_root=None, flat_dir=None) -> dict[str, dict[int, Path]]:
    """{registry arm: {seed: model file}}: the registry models below ``models_root``, else (smoke only) the flat
    quick nets in ``flat_dir``."""
    out = {}
    for arm in arms:
        found = find_models(models_root, cfg.name, arm) if models_root else {}
        if not found and flat_dir:
            found = find_checkpoints(flat_dir, arm)
        out[arm] = found
    return out


def smoke_models_root(cfg) -> Path | None:
    """The training driver's smoke models (``results/raw/smoke_train/<split>/models``) if they hold this
    configuration's models, else None."""
    for s in ("train", "dev"):
        root = R.SMOKE_DIR / s / "models"
        if any(find_models(root, cfg.name, a) for a in NET_KINDS):
            return root
    return None


def quick_nets(cal, cfg, windows, kinds, seeds, ckpt_dir, *, iterations: int | None = None) -> dict:
    """Smoke fallback when no training-driver models exist: train each (kind, seed) for ``iterations`` PPO
    iterations (lookahead h = 3, D6) on the given TRAIN windows; save ``<kind>_seed<s>.pt`` in ``ckpt_dir``."""
    import torch
    from slicemech.control import nets, ppo
    torch.set_num_threads(1)
    mech = R.mechanism(cal)
    fc = cal.operator_forecaster(cfg.d)
    iterations = SMOKE_PPO_ITERS if iterations is None else int(iterations)
    pcfg = ppo.PPOConfig(iterations=iterations, windows_per_iter=len(windows), horizon=PPO_HORIZON,
                         eval_every=0)
    out = {}
    for kind in kinds:
        for s in seeds:
            res = ppo.train(nets.make_net(kind, seed=int(s)), windows, None, d=cfg.d, mech=mech, config=pcfg,
                            forecaster=fc, seed=int(s))
            p = sio.confine(Path(ckpt_dir) / f"{kind}_seed{int(s)}.pt")
            p.parent.mkdir(parents=True, exist_ok=True)
            nets.save_net(res.net, p)
            out[(kind, int(s))] = p
    return out


# ============================================================================================ E0 per window
def vcal_stats(w, shape, mech, grid=VCAL_GRID, *, raw: bool = False):
    """V-calibration sufficient statistics of one window (module docstring, step 4).

    Returns a DataFrame with one row per (tenant, c): N, Sx, Sy, Sxx, Syy, Sxy (x = V/pi and y = on-time, Mbit) and
    the number of clipped quotas; with ``raw=True`` also the raw (x, y, valid) arrays of shape (S, G, n).
    """
    from slicemech.model import allocation
    from slicemech.model.shapes import stack_shape, value
    from slicemech.sim import fluid
    ct = w.contract
    n, K, warm = ct.n, w.K, int(w.warmup)
    z = np.minimum(np.asarray(w.theta, float), ct.C)
    r = np.asarray(w.rate_hat, float)
    rt = np.asarray(w.rate_true, float)
    slots = int(round(mech.Delta * fluid.SLOTS_PER_S))
    A = fluid.cumulative_arrivals(w.arrivals_ms)
    Q_eq = allocation.integer(z, np.ones(n), r, ct, mech)                 # reference path: equal weights
    _, c_start, _ = fluid.replay(A, Q_eq, rt, ct.delay_ms, slots_per_interval=slots)
    ks = np.arange(warm, K)
    g = np.asarray(grid, float)
    zk, rk = z[ks], r[ks]
    valid = (zk > 0) & (rk > 0)                                            # (S, n)
    q_raw = np.where(valid[:, None, :], g[None, :, None] * zk[:, None, :] / np.where(rk > 0, rk, 1.0)[:, None, :],
                     0.0)                                                  # (S, G, n) PRB
    q = np.minimum(q_raw, float(mech.B))
    clipped = q_raw > mech.B
    on = fluid.counterfactual_window(A, ks * slots, c_start[ks], q, rt[ks], ct.delay_ms,
                                     slots_per_interval=slots).on_time_Mbit                  # (S, G, n)
    beta, rho = stack_shape(shape)
    x = value(q, zk[:, None, :], np.where(rk > 0, rk, 1.0)[:, None, :], 1.0, beta, rho, mech.Delta)
    rows = []
    for i in range(n):
        for j, c in enumerate(g):
            m = valid[:, i]
            xi, yi = x[m, j, i], on[m, j, i]
            rows.append(dict(tenant=i, cls=ct.cls[i], c=float(c), N=int(m.sum()), Sx=float(xi.sum()),
                             Sy=float(yi.sum()), Sxx=float(xi @ xi), Syy=float(yi @ yi), Sxy=float(xi @ yi),
                             n_clipped=int(clipped[m, j, i].sum())))
    df = pd.DataFrame(rows)
    if raw:
        return df, dict(x=x, y=on, valid=np.broadcast_to(valid[:, None, :], x.shape).copy(), q=q)
    return df


def ols_from_sums(N, Sx, Sy, Sxx, Syy, Sxy) -> dict:
    """OLS of y on x (with intercept) from sufficient statistics: slope, intercept, R^2 (NaN when degenerate)."""
    N = float(N)
    if N < 2:
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan, N=N)
    vx, vy, cxy = Sxx - Sx * Sx / N, Syy - Sy * Sy / N, Sxy - Sx * Sy / N
    slope = cxy / vx if vx > 0 else np.nan
    r2 = cxy * cxy / (vx * vy) if vx > 0 and vy > 0 else np.nan
    return dict(slope=float(slope), intercept=float((Sy - slope * Sx) / N) if np.isfinite(slope) else np.nan,
                r2=float(r2), N=N, mean_x=Sx / N, mean_y=Sy / N)


def _ols_stat(which: str):
    def f(means):
        N, Sx, Sy, Sxx, Syy, Sxy = (means[..., j] for j in range(6))
        vx, vy, cxy = Sxx - Sx * Sx / N, Syy - Sy * Sy / N, Sxy - Sx * Sy / N
        return cxy / vx if which == "slope" else cxy * cxy / (vx * vy)
    return f


def window_diagnostics(w, cal, mech, grid=VCAL_GRID) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """E0 quantities of one window: (window row, V-calibration statistics, rate value counts)."""
    from slicemech.data.calibrate import regime_diagnostics
    from slicemech.eval.headroom import window_load
    ct = w.contract
    n, K, warm = ct.n, w.K, int(w.warmup)
    z = np.minimum(np.asarray(w.theta, float), ct.C)
    r, rt = np.asarray(w.rate_hat, float), np.asarray(w.rate_true, float)
    m = np.asarray(w.measured, float)
    B_rbar = mech.B * float(cal.r_bar)
    row = dict(split=w.split, date=w.date, offset_s=int(w.meta["offset_s"]), window_id=w.id, K=K, warmup=warm,
               n=n, tod_start_s=float(w.meta["tod_start_s"]),
               offered_load=float(m.mean(0).sum() / B_rbar),                 # D2 quantity (all K intervals)
               offered_load_scored=float(m[warm:].sum(1).mean() / B_rbar),
               capacity_load=window_load(w, mech),                          # scored, window's own true rate
               rate_true_mean=float(rt[warm:].mean()), rate_hat_mean=float(r[warm:].mean()))
    for i in range(n):
        row[f"count_{i}"] = int(w.meta["counts"][i])
        row[f"offered_mbps_{i}"] = float(m[:, i].mean())
        row[f"scored_mbps_{i}"] = float(m[warm:, i].mean())
        row[f"theta_mbps_{i}"] = float(z[warm:, i].mean())
        row[f"report_mae_over_mu_{i}"] = float(np.abs(np.asarray(w.theta, float)[warm:, i] - m[warm:, i]).mean()
                                               / ct.mu[i])
        row[f"frac_theta_above_C_{i}"] = float((np.asarray(w.theta, float)[warm:, i] > ct.C[i]).mean())
    diag = regime_diagnostics(z[warm:], r[warm:], ct, mech)
    row.update({f"rg_{k}": float(v) for k, v in diag.items() if k not in ("tilt",)})
    vc = vcal_stats(w, cal.shape, mech, grid)
    rates = []
    for i in range(n):
        vals, cnt = np.unique(np.round(rt[warm:, i], 9), return_counts=True)
        rates += [dict(tenant=i, cls=ct.cls[i], rate=float(v), count=int(c)) for v, c in zip(vals, cnt)]
    keys = dict(date=w.date, offset_s=row["offset_s"], window_id=w.id)
    vc = pd.concat([pd.DataFrame({k: [v] * len(vc) for k, v in keys.items()}), vc], axis=1)
    rt_df = pd.DataFrame(rates)
    rt_df = pd.concat([pd.DataFrame({k: [v] * len(rt_df) for k, v in keys.items()}), rt_df], axis=1)
    return row, vc, rt_df


_E0_CTX: dict = {}


def _e0_ctx(cfg_name: str, calib_root):
    key = (cfg_name, calib_root)
    if key not in _E0_CTX:
        _E0_CTX.clear()
        cal = load_cal(get_config(cfg_name), calib_root)
        _E0_CTX[key] = (cal, R.mechanism(cal))
    return _E0_CTX[key]


def _e0_job(args) -> dict:
    """One window: compose, diagnose, write the three part files. Skipped when the parts exist with the same input
    fingerprint (configuration, calibration file, smoke) unless ``force``; parts that are recomputed are archived
    first (:func:`archive_files`), never overwritten."""
    cfg_name, calib_root, fp, split, date, off, smoke, test_freeze, out, force, *rest = args
    stamp = rest[0] if rest else None
    parts = Path(out) / "parts"
    key = key_of(date, off)
    files = {t: parts / f"{key}__{t}.parquet" for t in ("win", "vcal", "rates")}
    if (not force and all(f.exists() for f in files.values())
            and set(pd.read_parquet(files["win"], columns=["fp"])["fp"]) == {fp}):
        return dict(key=key, ran=False, seconds=0.0)
    t0 = time.perf_counter()
    cal, mech = _e0_ctx(cfg_name, calib_root)
    w = compose(cal, split, date, off, smoke=smoke, test_freeze=test_freeze)
    row, vc, rates = window_diagnostics(w, cal, mech)
    row.update(config=cfg_name, fp=fp, seconds=time.perf_counter() - t0)
    archive_files([files["win"], files["vcal"], files["rates"]], out, stamp=stamp,
                  reason="E0 window recomputed (--force or other inputs)")
    write_parquet(f32(vc, exclude=("c", "Sx", "Sy", "Sxx", "Syy", "Sxy")), files["vcal"])
    write_parquet(rates, files["rates"])
    write_parquet(pd.DataFrame([row]), files["win"])                     # written last: the completion marker
    return dict(key=key, ran=True, seconds=row["seconds"])


# ============================================================================================ E0 summaries
def _dist(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {}
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if x.size > 1 else 0.0, min=float(x.min()),
                p5=float(np.percentile(x, 5)), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)),
                max=float(x.max()), n=int(x.size))


def regime_summary(win: pd.DataFrame) -> dict:
    """Pooled regime diagnostics (interval-weighted means of the per-window values) and the target check."""
    from slicemech.data.calibrate import check_targets
    wts = win["rg_intervals"].to_numpy(float)
    cols = [c for c in win.columns if c.startswith("rg_") and c != "rg_intervals"]
    pooled = {c[3:]: float(np.average(win[c].to_numpy(float), weights=wts)) for c in cols}
    pooled["intervals"] = int(wts.sum())
    return dict(pooled=pooled, targets=check_targets(pooled),
                per_window={c[3:]: _dist(win[c]) for c in cols})


def load_summary(win: pd.DataFrame, rho: float, n: int) -> dict:
    """Per-window and per-date realized loads (D2) and the per-date counts."""
    by_date = win.groupby("date").agg(offered_load=("offered_load", "mean"), windows=("offered_load", "size"),
                                      **{f"count_{i}": (f"count_{i}", "first") for i in range(n)})
    dev = by_date["offered_load"] / rho - 1.0
    return dict(rho=rho, window_offered_load=_dist(win["offered_load"]),
                window_capacity_load=_dist(win["capacity_load"]),
                date_offered_load=_dist(by_date["offered_load"]),
                date_max_abs_rel_dev_from_rho=float(np.abs(dev).max()),
                counts_range={i: [int(by_date[f"count_{i}"].min()), int(by_date[f"count_{i}"].max())]
                              for i in range(n)},
                per_date={d: dict(offered_load=float(r["offered_load"]), windows=int(r["windows"]),
                                  counts=[int(r[f"count_{i}"]) for i in range(n)])
                          for d, r in by_date.iterrows()})


def vcal_summary(vc: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Per class: pooled OLS (all grid factors), date-cluster bootstrap CIs of slope and R^2, and the per-c curve."""
    from slicemech.eval.stats import cluster_bootstrap
    out = {}
    S = ["N", "Sx", "Sy", "Sxx", "Syy", "Sxy"]
    for cls, g in vc.groupby("cls", sort=False):
        tot = g[S].astype(float).sum()
        res = ols_from_sums(*tot.to_numpy())
        per_w = g.groupby(["date", "offset_s"])[S].sum().astype(float).reset_index()
        if per_w["date"].nunique() >= 2 and n_boot:
            vals, dates = per_w[S].to_numpy(), per_w["date"].to_numpy()
            for which in ("slope", "r2"):
                b = cluster_bootstrap(vals, dates, n_boot=n_boot, seed=0,
                                      stat=_ols_stat("slope" if which == "slope" else "r2"))
                res[f"{which}_ci"] = [b.lo, b.hi]
        res["flag_r2_below"] = bool(np.isfinite(res["r2"]) and res["r2"] < RSQ_FLAG)
        res["clipped_share"] = float(g["n_clipped"].sum() / max(1.0, g["N"].sum()))
        curve = []
        for c, gc in g.groupby("c"):
            t = gc[S].astype(float).sum()
            o = ols_from_sums(*t.to_numpy())
            curve.append(dict(c=float(c), N=int(t["N"]), mean_V_over_pi_Mbit=o.get("mean_x", np.nan),
                              mean_on_time_Mbit=o.get("mean_y", np.nan), slope=o["slope"], r2=o["r2"]))
        res["curve"] = curve
        out[cls] = res
    return out


def rate_summary(rates: pd.DataFrame) -> dict:
    out = {}
    for cls, g in rates.groupby("cls", sort=False):
        t = g.groupby("rate")["count"].sum().sort_index()
        v, c = t.index.to_numpy(float), t.to_numpy(float)
        cdf = np.cumsum(c) / c.sum()
        q = {f"p{p}": float(v[np.searchsorted(cdf, p / 100.0)]) for p in (5, 50, 95)}
        out[cls] = dict(mean=float((v * c).sum() / c.sum()), min=float(v.min()), max=float(v.max()),
                        intervals=int(c.sum()), **q)
    return out


# ============================================================================================ worked interval
def _choose_interval(w, cal, cfg, mech, prot, unc, i, n_grid):
    from slicemech.control.features import window_features
    from slicemech.model.allocation import QuotaRule
    from slicemech.pricing import envelope as env
    from slicemech.pricing import thresholds as thr
    ct = w.contract
    z_all = np.minimum(np.asarray(w.theta, float), ct.C)
    r_all = np.asarray(w.rate_hat, float)
    fp = window_features(w, cfg.d, ct, forecaster=prot.forecaster, mech=mech)
    fu = window_features(w, cfg.d, ct, forecaster=unc.forecaster, mech=mech)
    u_chk = np.linspace(0.0, float(ct.C[i]), n_grid)
    best, best_key = None, None
    for k in range(int(w.warmup), w.K):
        z = z_all[k]
        if z[i] <= 0:
            continue
        rp = QuotaRule(ct, mech, r_all[k], prot.priority_map(fp.context(k)))
        ru = QuotaRule(ct, mech, r_all[k], unc.priority_map(fu.context(k)))
        T = env.premium(i, z, rp, ct, cal.shape, mech, r_all[k]).T
        drop = thr.max_drop(thr.own_path(i, z, ru, u_chk))[0]
        key = (drop > 0, round(float(T), 12), float(drop), -k)
        if best_key is None or key > best_key:
            best, best_key = k, key
    if best is None:
        raise ValueError("no scored interval with a positive report for the worked tenant")
    return best, dict(unconstrained_drop_on_grid=best_key[2], protected_T_truth=best_key[1],
                      rule="max (unconstrained path drops, protected premium T(theta), drop size), earliest k")


def worked_interval(w, cal, cfg, mech, prot, unc, *, tenant: int | None = None, k: int | None = None,
                    n_grid: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Own quota and both prices versus the own report for the protected and the unconstrained map (module
    docstring, step 5). Returns (curves, staircases, meta)."""
    from slicemech.control.features import window_features
    from slicemech.model.allocation import QuotaRule
    from slicemech.model.shapes import stack_shape, value
    from slicemech.pricing import envelope as env
    from slicemech.pricing import thresholds as thr
    ct = w.contract
    n_grid = WORKED_GRID if n_grid is None else int(n_grid)
    i = ct.cls.index("urllc") if tenant is None else int(tenant)
    sel = None
    if k is None:
        k, sel = _choose_interval(w, cal, cfg, mech, prot, unc, i, n_grid)
    k = int(k)
    z = np.minimum(np.asarray(w.theta, float), ct.C)[k].copy()
    r = np.asarray(w.rate_hat, float)[k].copy()
    th, C_i = float(z[i]), float(ct.C[i])
    beta, rho = stack_shape(cal.shape)
    vargs = (r[i], ct.pi[i], beta[i], rho[i], mech.Delta)
    ctx_p = window_features(w, cfg.d, ct, forecaster=prot.forecaster, mech=mech).context(k)
    ctx_u = window_features(w, cfg.d, ct, forecaster=unc.forecaster, mech=mech).context(k)
    maps = {"protected": (QuotaRule(ct, mech, r, prot.priority_map(ctx_p)), True),
            "unconstrained": (QuotaRule(ct, mech, r, unc.priority_map(ctx_u)), False)}
    u = np.unique(np.concatenate([np.linspace(0.0, C_i, n_grid), [th]]))
    curves, stairs, meta = [], [], {}
    for name, (rule, monotone) in maps.items():
        Q = thr.own_path(i, z, rule, u).astype(float)
        p_full, p_fix = np.empty(u.size), np.empty(u.size)
        for j, uu in enumerate(u):
            zz = z.copy()
            zz[i] = uu
            full = (env.full_price(i, zz, rule, ct, cal.shape, mech, r) if monotone
                    else env.price_nonmonotone(i, zz, rule, ct, cal.shape, mech, r))
            p_full[j] = full.p
            p_fix[j] = env.fixed_priority_price(i, zz, rule, ct, cal.shape, mech, r).p
        V_true = value(Q, th, *vargs)
        V_rep = value(Q, u, *vargs)
        curves.append(pd.DataFrame(dict(map=name, u=u, quota=Q, p_full=p_full, p_fix=p_fix, V_true=V_true,
                                        V_report=V_rep, U_full=V_true - p_full, U_fix=V_true - p_fix,
                                        is_truth=np.isclose(u, th, rtol=0, atol=0))))
        starts, quotas = env.staircase(rule, z, i, C_i)
        change = np.concatenate([[True], quotas[1:] != quotas[:-1]])      # merge equal-quota grid points into cells
        starts, quotas = starts[change], quotas[change]
        stairs.append(pd.DataFrame(dict(map=name, start=starts, quota=quotas)))
        jt = int(np.flatnonzero(u == th)[0])
        U_fix = V_true - p_fix
        jb = int(np.argmax(U_fix))
        meta[name] = dict(weights_truth=rule.weights(z).tolist(), quota_truth=float(Q[jt]),
                          p_full_truth=float(p_full[jt]), p_fix_truth=float(p_fix[jt]),
                          premium_truth=float(p_full[jt] - p_fix[jt]),
                          max_drop_grid=float(thr.max_drop(Q)[0]), n_cells=int(starts.size),
                          fixed_price_grid_best_report=float(u[jb]),
                          fixed_price_grid_gain=float(U_fix[jb] - U_fix[jt]),
                          full_price_grid_gain=float(np.max(V_true - p_full) - (V_true - p_full)[jt]))
    a, kappa = prot.heads_np(ctx_p)
    meta.update(window_id=w.id, date=w.date, offset_s=int(w.meta["offset_s"]), k=k, tenant=i, cls=ct.cls[i],
                theta_adm=th, theta_raw=float(np.asarray(w.theta, float)[k, i]), C=C_i, r_hat=float(r[i]),
                pi=float(ct.pi[i]), z_profile=z.tolist(), r_profile=r.tolist(), protected_a=a.tolist(),
                protected_kappa=kappa.tolist(), selection=sel, n_grid=int(n_grid),
                units=dict(u="Mbit/s", quota="PRB", prices="credits per interval"),
                note="unconstrained p_full is the envelope formula on a 4096-point grid path (not truthful, "
                     "uncertified); p_fix freezes the priority vector at the report")
    return pd.concat(curves, ignore_index=True), pd.concat(stairs, ignore_index=True), meta


def _controllers_style() -> dict:
    import yaml
    rows = yaml.safe_load(open(sio.ROOT / "spec" / "controllers.yaml"))
    return {r["key"]: dict(label=r["label"], color=r["color"], marker=r["marker"]) for r in rows}


# ============================================================================================ E0 main
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def run_pool(fn, jobs, workers: int):
    """Map ``fn`` over ``jobs`` in-process (workers <= 1) or in a spawn process pool; yields results in order."""
    if workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            yield fn(j)
        return
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        yield from ex.map(fn, jobs, chunksize=1)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    ap.add_argument("--config", default="default", help=f"one of {', '.join(R.CONFIGS)} (experiments/registry.py)")
    ap.add_argument("--smoke", action="store_true", help="2 short TRAIN windows; models of the training driver's "
                                                       "smoke run, else 2-iteration PPO nets")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models-root", default=None, help="training-registry models for the worked interval "
                                                        "(default results/models; smoke: results/raw/smoke_train)")
    ap.add_argument("--calib-root", default=None, help="directory of the sweep calibrations (default "
                                                       "data/calibration; smoke: the training smoke run's)")
    ap.add_argument("--seed", type=int, default=0, help="model seed of the worked-interval nets")
    ap.add_argument("--skip-worked", action="store_true")
    ap.add_argument("--worked-window", default=None, help="DATE:OFFSET (default: first window of the split)")
    ap.add_argument("--worked-k", type=int, default=None)
    ap.add_argument("--worked-tenant", type=int, default=None)
    ap.add_argument("--test-freeze", default=None, help="frozen pre-registration (default prereg/PREREG.md)")
    ap.add_argument("--dates", default=None, help="comma-separated subset of the split's dates")
    ap.add_argument("--limit", type=int, default=None, help="first N windows only (debugging)")
    ap.add_argument("--force", action="store_true", help="recompute windows whose parts exist")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    split = "train" if a.smoke else a.split
    # round-3 review: no window subsets or model relocations on TEST (the analysis checks completeness too)
    R.refuse_test_overrides(split, a, ap, ("--dates", "--limit", "--models-root", "--calib-root", "--worked-window",
                                           "--worked-k", "--worked-tenant", "--seed", "--skip-worked"))
    freeze = test_gate(split, a.test_freeze)                 # before anything reads TEST data
    sio.require_free_gb(3)
    cfg = get_config(a.config)
    calib_root = a.calib_root or (str(smoke_calib_root()) if a.smoke and smoke_calib_root() else None)
    cal = load_cal(cfg, calib_root)
    mech = R.mechanism(cal)
    out = out_dir("e0", split, cfg.name, a.smoke)
    keys = window_keys(split, a.smoke, a.dates.split(",") if a.dates else None, a.limit)
    fp = sio.digest(dict(config=config_record(cfg), calibration_sha256=calibration_sha(cfg, calib_root),
                         smoke=a.smoke, K=SMOKE_K if a.smoke else cal.K, grid=list(VCAL_GRID)))[:16]
    t0 = time.perf_counter()
    stamp = archive_stamp()
    jobs = [(cfg.name, calib_root, fp, split, d, o, a.smoke, freeze, str(out), a.force, stamp) for d, o in keys]
    ran = 0
    for res in run_pool(_e0_job, jobs, a.workers):
        ran += int(res["ran"])
        print(f"[e0] {res['key']}: {'done' if res['ran'] else 'skipped (parts exist)'} "
              f"{res['seconds']:.1f} s", flush=True)
    parts = out / "parts"
    tabs = {}
    for t in ("win", "vcal", "rates"):
        tabs[t] = pd.concat([pd.read_parquet(parts / f"{key_of(d, o)}__{t}.parquet") for d, o in keys],
                            ignore_index=True)
    write_parquet(tabs["win"], out / "windows.parquet")
    write_parquet(tabs["vcal"], out / "vcal.parquet")
    write_parquet(tabs["rates"], out / "rates.parquet")
    summary = dict(experiment="E0", split=split, config=config_record(cfg), smoke=a.smoke, windows=len(keys),
                   windows_computed_this_run=ran, calibration=dict(name=cfg.calibration, root=calib_root,
                                                                   sha256=calibration_sha(cfg, calib_root)),
                   r_bar_train=float(cal.r_bar), B=int(mech.B), classes=list(cal.classes),
                   regime=regime_summary(tabs["win"]), loads=load_summary(tabs["win"], float(cal.rho), cal.n),
                   vcal=vcal_summary(tabs["vcal"], n_boot=0 if a.smoke else a.n_boot),
                   vcal_grid=list(VCAL_GRID), rsq_flag=RSQ_FLAG, rates=rate_summary(tabs["rates"]),
                   decisions=R.DECISIONS)
    # ---- worked interval
    worked = dict(status="skipped (--skip-worked)")
    if not a.skip_worked:
        roots = a.models_root or (str(smoke_models_root(cfg) or "") if a.smoke else str(R.MODELS_DIR))
        flat = out / "ckpt" if a.smoke else None
        wkey = (a.worked_window.split(":")[0], int(a.worked_window.split(":")[1])) if a.worked_window else keys[0]
        paths = model_paths(cfg, ("protected", "unconstrained"), roots or None, flat)
        paths = {k: v.get(a.seed) for k, v in paths.items()}
        if a.smoke and not all(paths.values()):          # no training-driver smoke models: 2-iteration nets
            wins = [compose(cal, "train", d, o, smoke=True) for d, o in SMOKE_WINDOWS]
            quick_nets(cal, cfg, wins, ("protected", "unconstrained"), (a.seed,), flat)
            paths = {k: find_checkpoints(flat, k).get(a.seed) for k in ("protected", "unconstrained")}
        if not all(paths.values()):
            worked = dict(status=f"skipped: no seed-{a.seed} protected/unconstrained models below {roots}")
        else:
            if freeze:
                verify_frozen_checkpoints(paths.values(), freeze)
            shas = {k: sio.file_sha256(p) for k, p in paths.items()}
            wfp = sio.digest(dict(window=list(wkey), k=a.worked_k, tenant=a.worked_tenant, shas=shas, fp=fp,
                                  n_grid=WORKED_GRID, split=split))
            files = [out / f for f in ("worked_interval.json", "worked_interval.parquet", "worked_staircase.parquet")]
            if all(f.exists() for f in files) and sio.read_json(files[0]).get("fp") == wfp:
                meta = sio.read_json(files[0])["meta"]                     # resumable: same inputs, reuse
                curves, stairs = pd.read_parquet(files[1]), pd.read_parquet(files[2])
                seconds = 0.0
            else:
                prot = load_checked_net(paths["protected"], cfg.d)
                unc = load_checked_net(paths["unconstrained"], cfg.d)
                w = compose(cal, split, wkey[0], wkey[1], smoke=a.smoke, test_freeze=freeze)
                tw = time.perf_counter()
                curves, stairs, meta = worked_interval(w, cal, cfg, mech, prot, unc, tenant=a.worked_tenant,
                                                       k=a.worked_k, n_grid=WORKED_GRID)
                seconds = time.perf_counter() - tw
                write_parquet(curves, files[1])
                write_parquet(stairs, files[2])
                sio.write_json(files[0], _jsonable(dict(fp=wfp, meta=meta)))
            style = _controllers_style()
            worked = dict(status="ok", seconds=seconds, meta=meta,
                          models={k: dict(path=str(p), sha256=shas[k]) for k, p in paths.items()},
                          style={"protected": style.get("protected"), "unconstrained": style.get("unprotected")},
                          curves={m: g.drop(columns="map").to_dict(orient="list") for m, g in curves.groupby("map")},
                          staircases={m: g.drop(columns="map").to_dict(orient="list")
                                      for m, g in stairs.groupby("map")})
    summary["worked_interval"] = worked
    summary["seconds"] = time.perf_counter() - t0
    summary["bytes_on_disk"] = dir_bytes(out)
    summary = _jsonable(summary)
    sio.write_json(out / "summary.json", summary)
    pooled = summary["regime"]["pooled"]
    print(json.dumps(dict(regime={k: round(v, 4) for k, v in pooled.items() if isinstance(v, float)},
                          targets_ok={k: v["ok"] for k, v in summary["regime"]["targets"].items()},
                          loads=summary["loads"]["window_offered_load"],
                          vcal={c: dict(slope=v["slope"], r2=v["r2"], flag=v["flag_r2_below"])
                                for c, v in summary["vcal"].items()},
                          worked=worked.get("status"), worked_models=worked.get("models"), out=str(out)), indent=1),
          flush=True)
    return summary


if __name__ == "__main__":
    main()
