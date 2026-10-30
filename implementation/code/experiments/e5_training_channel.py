"""E5 - the training channel: can a tenant buy later priority by distorting the reports that retraining reads?

Run from ``code/`` (``PYTHONDONTWRITEBYTECODE=1 /home/darpan/anaconda3/bin/python experiments/e5_training_channel.py``):

    --smoke                        tiny TRAIN plumbing run (1 date, 1 + 1 short windows, 2 PPO iterations, tiny
                                   checkpoints trained on the spot); writes results/raw/train/e5/smoke/
    --split dev [--workers 24]     the DEV pilot that fixes the pre-registered H5b sign (results/raw/dev/e5/)
    --split test --test-freeze ../prereg/PREREG.md
                                   TEST, only after the pre-registration freeze (refused otherwise)
    --plan                         print the job list and the cost estimate, run nothing
    --aggregate                    only rebuild the aggregate tables from finished jobs

The run is resumable: one job is one (date, seed, arm); a job is skipped when its ``jobs/<job>.done.json`` exists
AND records the job's current provenance fingerprint (``job_fingerprint``: model file SHA-256, hyper-parameters,
windows, profiles, retraining settings and the code fingerprint of ``slicemech`` and this module); a job whose record
names other inputs is rerun (round-3 review). ``--force`` recomputes. A job that is (re)computed first moves its old
``done.json`` and outputs to ``results/raw/<split>/archive/<UTC stamp>/e5/jobs/`` (``retire_job_files``; pre-freeze
audit E5-2, F7/PV-10), so a failed recompute leaves no stale record and earlier numbers are kept; every
``done.json`` records the code fingerprint ``code``. Aggregates are rebuilt from every finished job at the end of
each run; a job whose ``error.txt`` is newer than its ``done.json`` is reported as an error and left out.

TEST (round-3 review): the protocol options (``TEST_FIXED_OPTIONS``: dates, seeds, arms, targets, alphas, d, the
retraining hyper-parameters, the model locations and ``--out``) are refused, so every choice comes from the frozen
code; every model file must be listed in the freeze block with its SHA-256 (``registry.verify_frozen``).

Protocol (experiment_design.md E5 and §6; tenants/poison.py; orchestrator decision D5)
-----------------------------------------------------------------------------------
For every date of the split, every seed s (default 0, 1, 2) and every arm:

* arm ``protected``   = trained protected checkpoint (own-report score net, kappa >= 0), billed the full-decision
  price; arm ``unprotected`` = trained unconstrained checkpoint, billed the fixed-priority price
  (``spec/controllers.yaml`` keys). Checkpoint of seed s: the deployed model of configuration ``default`` in the
  shared layout of ``experiments/registry.py`` (``results/models/default/<protected|unconstrained>/seed<s>/model.pt``
  with its ``config.json``, written by ``experiments/train.py``; file and parameter digests are checked). The
  fine-tuning learning rate (start lr, end lr / 10) and samples per state M are the model's DEV-selected
  hyper-parameters, and the start exploration is the model's own final ``log_sigma``; ``--lr``, ``--M`` and
  ``--sigma0`` override them. ``--models-root`` relocates the layout; ``--ckpt-template`` reads plain
  ``nets.save_net`` files instead (``{cal}``, ``{kind}``, ``{seed}`` filled; optional sidecar ``<file>.json`` with
  ``log_sigma``).
* windows: the 8 pre-declared E5 windows of the date (``poison.e5_layout``: offsets ``60 + 10800 j`` s); windows
  1-4 are retraining windows composed from pool phase A, windows 5-8 evaluation windows from phase B (disjoint
  interleaved 2-s halves of every TRACTOR capture and alternating Raca-run halves; the halves swap on odd-ordinal
  dates). Counts are the date's D2 counts.
* reporting profiles: the reference (everybody truthful, alpha = 1, ``poison.REFERENCE_ALPHA``) and, for each
  target (the video tenant, the URLLC-like tenant) and alpha in {0.5, 0.7, 1.5, 2}, the target reports
  ``min(alpha theta, C)`` in every interval of windows 1-4 (``poison.plan_date``); everybody else is truthful.
  Windows 1-4 are played with the base checkpoint under each profile (``sim.episode.run_window``), which gives
  the stored report logs and the target's cost of lying during windows 1-4.
* retraining (``control.retrain.retrain``), 50 PPO iterations on windows 1-4 (all four per iteration), D6
  lookahead horizon h = 3, the same fine-tuning seed for every profile and pipeline of the (date, seed):
  - ``measurement``: measurement-only retraining. It receives the *played record* of windows 1-4 - the window
    with ``theta`` replaced by the logged reports and the logged reports, quotas, prices, deliveries and
    utilities in ``meta`` - plus the stored report logs as ``stored_reports``; the pipeline must ignore all of it,
    so its parameters must be bit-identical for every profile (H5a, recorded as SHA-256 parameter digests);
  - ``stored``: stored-report retraining on the logged reports (the channel the paper closes).
* evaluation on windows 5-8 with truthful reports for every retrained net and the base checkpoint (nets with the
  same parameter digest share one evaluation: identical parameters and windows give identical episodes).

Recorded (per window, scored intervals): every tenant's on-time / dropped Mbit, deadline-miss %, quota, price,
declared value and utility, realized value and utility (``eval.metrics.window_summary``); per (pipeline, target,
alpha) against the alpha = 1 arm of the same pipeline and date (H5b reference): the target's later gain in
utility, quota and on-time Mbit on windows 5-8, its cost during windows 1-4 and the net, the honest tenants'
effects (``eval.metrics.victim_effects``, D4) in both phases; parameter digests and L2 distances (to the base
checkpoint and to the alpha = 1 net); each window's class loads (mean measured demand per tenant, Mbit/s), its
D2 offered load (``sum m / (B rbar_TRAIN)``) and capacity load (``eval.headroom.window_load``).

Outputs (``results/raw/<split>/e5/``; ``smoke/`` below it for --smoke):
``jobs/<date>__<arm>__s<seed>.{attack,later,pairs,nets,windows}.parquet`` and ``.done.json``; aggregates
``attack.parquet`` (windows 1-4 summaries), ``later.parquet`` (windows 5-8 summaries), ``pairs.parquet`` (per-window
contrasts vs alpha = 1), ``contrasts.parquet`` (per date x seed x arm x pipeline x target x alpha, window means and
the 8-window net), ``nets.parquet``, ``windows.parquet`` and ``checks.json`` (H5a bit-identity per group, stored-log
checks, the protected arm's per-interval DSIC check during windows 1-4). Optional ``--keep-episodes`` stores the
per-interval episode tables (float32). Signs: every ``d_*`` column is (distorting profile) minus (alpha = 1 arm);
a positive later gain means the distortion paid off later; a negative attack-phase ``d_utility_*`` is a cost.

Orchestrator decisions (binding, 2026-10-19; code/STATUS.md §0)
-----------------------------------------------------------------
D1 GBR fractions x 0.8 (video 0.40, URLLC-like 0.64, mMTC-like 0.16): in the n3 calibration's contract.
D2 offered-load normalization per date, every split (counts so that the date's mean offered traffic of class c is
   share_c x rho x B x rbar_TRAIN; shares 0.6/0.3/0.1): ``Calibration.window`` composes E5 windows with the date's
   counts; per-window realized loads are recorded.
D3 gate G4 dropped; H1c/H1d secondary with load subgroups (not used here).
D4 harm = honest tenants' own outcomes (on-time Mbit, deadline-miss %, their realized value); total welfare is
   descriptive only: the ``pairs``/``contrasts`` victim columns.
D5 E5 = interleaved 2-s block halves with date-alternating roles, the alpha = 1 stored-report arm as the H5b
   reference, 8 windows per date (4 retraining + 4 evaluation) on DEV and TEST: implemented as above.
D6 learner objective = lookahead return, h = 3 (``PPOConfig(horizon=3)``, set explicitly here); short-horizon RL.
D7 E4b kappa-scaling frontier (not used here).

TEST is read only after the pre-registration freeze (``data.puffer_fetch.require_test_read_gate``): ``--split
test`` is refused before any TEST source is opened unless the freeze is recorded or ``--test-freeze`` passes.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True
_CODE = Path(__file__).resolve().parents[1]
for _p in (_CODE, _CODE / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from slicemech import io as sio  # noqa: E402
from slicemech.config import PARAMS  # noqa: E402

import registry as R  # noqa: E402  (experiments/registry.py: shared model layout)

CAL_NAME = "n3_default"
D_DEFAULT = int(PARAMS["information"]["measurement_delay_s"])        # 2 intervals
HORIZON = 3                                                          # D6
FT_ITERS = int(PARAMS["experiment_settings"]["FtIters"])             # 50
SEEDS_DEFAULT = (0, 1, 2)                                            # params: FiveSeeds = 3
#: arm -> (net kind of the checkpoint, billed price) (spec/controllers.yaml keys)
ARMS = {"protected": ("protected", "full"), "unprotected": ("unconstrained", "fixed")}
PIPELINES = ("measurement", "stored")
TARGETS = ("video", "urllc")
REF = ("none", 1.0)
MODEL_CONFIG = R.DEFAULT                                             # registry configuration of n = 3, d = 2
SPLITS_FULL = ("dev", "test")
SMOKE = dict(date="2024-01-01", ckpt_window=("2024-01-14", 21660), K=30, warmup=10, iters=2, alphas=(0.5, 2.0),
             n_train=1, n_eval=1, seeds=(0,))
SUM_COLS = ("utility_declared", "utility_realized", "quota", "on_time_Mbit", "dropped_Mbit", "price", "V_declared",
            "realized_value")
#: Options refused on TEST (round-3 review): the E5 protocol is the frozen code's.
TEST_FIXED_OPTIONS = ("--dates", "--seeds", "--arms", "--targets", "--alphas", "--d", "--models-root",
                      "--ckpt-template", "--ft-iters", "--lr", "--M", "--sigma0", "--out")


# ============================================================================================ small helpers
def out_dir(split: str, smoke: bool) -> Path:
    base = sio.ROOT / "results" / "raw" / split / "e5"
    return base / "smoke" if smoke else base


def ft_seed(date: str, seed: int) -> int:
    """Fine-tuning seed of a (date, checkpoint seed): the same for every profile, pipeline and arm of the date."""
    return int.from_bytes(hashlib.sha256(f"e5|{date}|{int(seed)}".encode()).digest()[:4], "little")


def layout(split: str, smoke: bool) -> tuple:
    """(offset_s, pool phase) of the date's E5 windows in time order (``poison.e5_layout``; smoke: 1 + 1)."""
    from slicemech.tenants import poison
    if smoke:
        return ((poison.E5_OFFSETS[0], "A"), (poison.E5_OFFSETS[poison.E5_N_TRAIN], "B"))
    return poison.e5_layout(split)


def profiles(targets, alphas) -> list:
    """Reporting profiles: the alpha = 1 reference first, then (target, alpha) for alpha != 1."""
    out = [REF]
    for t in targets:
        for a in alphas:
            if float(a) != 1.0:
                out.append((t, float(a)))
    return out


def ptag(p) -> str:
    return f"{p[0]}_a{p[1]:g}"


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomic parquet write (zstd)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.attrs = {}
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=1, sort_keys=True, default=sio._jsonable)
    os.replace(tmp, path)


def as_matrix(df: pd.DataFrame, col: str, K: int, n: int) -> np.ndarray:
    """(K, n) array of an episode column (rows ordered by (k, tenant), checked)."""
    k, t = df["k"].to_numpy(), df["tenant"].to_numpy()
    if len(df) != K * n or not (np.array_equal(k, np.repeat(np.arange(K), n))
                                and np.array_equal(t, np.tile(np.arange(n), K))):
        raise ValueError("episode table is not ordered by (k, tenant)")
    return df[col].to_numpy(float).reshape(K, n)


def played_record(window, df: pd.DataFrame):
    """What the operator holds after playing ``window``: the window with ``theta`` replaced by the logged reports
    (the operator never observes types) and the logged reports, quotas, prices, deliveries and declared utilities in
    ``meta``. Measurement-only retraining must ignore all of it (H5a); ``retrain.MeasurementView`` hides it."""
    K, n = window.K, window.n
    logs = {name: as_matrix(df, col, K, n) for name, col in
            (("stored_reports", "report"), ("quota_log", "quota"), ("price_log", "price"),
             ("on_time_log", "on_time_Mbit"), ("utility_log", "utility_declared"))}
    meta = dict(window.meta)
    meta.update(logs)
    return dataclasses.replace(window, theta=logs["stored_reports"].copy(), meta=meta)


def window_info(w, r_bar: float, mech) -> dict:
    """Exogenous loads of a window (reported next to every E5 result): ``offered_load`` over all K intervals (the
    E0/E1 D2 quantity), ``offered_load_scored`` over the scored intervals, ``capacity_load`` (scored) and the
    per-tenant mean measured demand of the scored intervals (round-3 review: one definition per name)."""
    from slicemech.eval.headroom import offered_load, window_load
    m = np.asarray(w.measured, float)[w.warmup:]
    row = dict(window=w.id, date=w.date, offset_s=int(w.meta["offset_s"]), phase=w.meta.get("pool_part"),
               pool_half=w.meta.get("pool_half"), counts=json.dumps([int(c) for c in w.meta["counts"]]),
               offered_load=offered_load(w, r_bar, mech),
               offered_load_scored=offered_load(w, r_bar, mech, scored_only=True), capacity_load=window_load(w, mech))
    for i in range(w.n):
        row[f"load_mbps_{i}"] = float(m[:, i].mean())
    return row


def target_deltas(ref: pd.DataFrame, dev: pd.DataFrame, tgt: int, warmup: int) -> dict:
    """The target's per-interval mean (and window total) changes, distorting profile minus reference."""
    a = ref[(ref["k"] >= warmup) & (ref["tenant"] == tgt)]
    b = dev[(dev["k"] >= warmup) & (dev["tenant"] == tgt)]
    if not np.array_equal(a["k"].to_numpy(), b["k"].to_numpy()):
        raise ValueError("runs cover different intervals")
    out = {}
    for c in SUM_COLS:
        diff = b[c].to_numpy(float) - a[c].to_numpy(float)
        out[f"d_{c}_target"] = float(diff.mean())
        if c.startswith("utility"):
            out[f"d_{c}_target_total"] = float(diff.sum())
            out[f"d_{c}_target_max_interval"] = float(diff.max())
    def miss(df):                      # volume-weighted deadline-miss % over the window
        on, dr = float(df["on_time_Mbit"].sum()), float(df["dropped_Mbit"].sum())
        return 100.0 * dr / (on + dr) if on + dr > 0 else np.nan

    out["d_miss_pct_target"] = float(miss(b) - miss(a))
    out["frac_report_changed_target"] = float(np.mean(a["report"].to_numpy() != b["report"].to_numpy()))
    return out


def resolve_ckpt(template: str, kind: str, seed: int, cal: str = CAL_NAME) -> Path:
    p = Path(template.format(cal=cal, kind=kind, seed=int(seed)))
    return p if p.is_absolute() else sio.ROOT / p


def model_entry(kind: str, seed: int, *, config: str = MODEL_CONFIG, models_root=None, template=None) -> dict:
    """Where the trained model of (``kind``, ``seed``) lives and what its record says.

    Default: the registry layout ``<models_root>/<config>/<kind>/seed<seed>/{model.pt, config.json}`` (finished
    jobs only); ``template``: a plain ``nets.save_net`` file (+ optional sidecar ``<file>.json`` with ``log_sigma``).
    Returns ``dict(path, record, sha256, param_digest, log_sigma, hyper, missing)``.
    """
    if template:
        path = resolve_ckpt(template, kind, seed)
        side, ls = Path(str(path) + ".json"), None
        if side.is_file():
            try:
                ls = json.loads(side.read_text()).get("log_sigma")
            except (ValueError, OSError):
                ls = None
        return dict(path=str(path), record=None, sha256=None, param_digest=None,
                    log_sigma=None if ls is None else float(ls), hyper={}, missing=not path.is_file())
    d = R.model_dir(config, kind, seed, root=models_root)
    rec = R.read_done(d / R.CONFIG_FILE)
    path = d / R.MODEL_FILE
    res = (rec or {}).get("result") or {}
    hyper = (rec or {}).get("hyperparameters") or {}
    return dict(path=str(path), record=str(d / R.CONFIG_FILE), sha256=res.get("model_sha256"),
                param_digest=res.get("param_digest"),
                log_sigma=None if res.get("log_sigma") is None else float(res["log_sigma"]),
                hyper={k: hyper[k] for k in ("id", "lr", "sigma0", "M") if k in hyper},
                missing=rec is None or not path.is_file())


def load_checked(entry: dict):
    """``nets.load_net`` of a model entry, checking the recorded file SHA-256 and parameter digest when present."""
    from slicemech.control import nets as N
    path = Path(entry["path"])
    sha = sio.file_sha256(path)
    if entry.get("sha256") and sha != entry["sha256"]:
        raise ValueError(f"{path} differs from the SHA-256 recorded in its config.json")
    net = N.load_net(path)
    if entry.get("param_digest") and N.param_digest(net) != entry["param_digest"]:
        raise ValueError(f"{path}: parameter digest differs from its record")
    net.eval()
    return net, sha


# ============================================================================================ one job
def retire_job_files(out: Path, jid: str, stamp: str | None = None) -> Path | None:
    """Before a job is (re)computed: move its earlier ``jobs/<job>.done.json`` (first, so no completion record
    outlives its outputs) and every earlier output ``jobs/<job>.*`` to the archive (``e0_setup.archive_files``;
    pre-freeze audit E5-2 and F7/PV-10: a failed recompute can no longer leave a stale done.json, and the earlier
    numbers are kept). The job's ``error.txt`` stays; leftover temp files are deleted. Returns the archive folder."""
    import e0_setup as S
    jdir = Path(out) / "jobs"
    if not jdir.is_dir():
        return None
    for t in jdir.glob(f"{jid}.*.tmp"):
        t.unlink(missing_ok=True)
    files = [p for p in sorted(jdir.glob(f"{jid}.*")) if not p.name.endswith((".error.txt", ".tmp"))]
    files.sort(key=lambda p: not p.name.endswith(".done.json"))
    return S.archive_files(files, out, stamp=stamp, reason="job recomputed (stale fingerprint or --force)")


def run_job(spec: dict) -> dict:
    """One (date, seed, arm): play windows 1-4 under every profile, retrain under both pipelines, evaluate on
    windows 5-8, write ``jobs/<job>.*`` and return a status dict. See the module docstring. The job's earlier record
    and outputs are archived first (:func:`retire_job_files`); the new ``done.json`` records the code fingerprint
    (``code``)."""
    retire_job_files(Path(spec["out"]), spec["job"], spec.get("archive_stamp"))
    import torch
    torch.set_num_threads(1)
    from slicemech.config import Mechanism
    from slicemech.control import nets as N
    from slicemech.control import ppo as P
    from slicemech.control.retrain import retrain
    from slicemech.data.calibrate import load_calibration
    from slicemech.eval.metrics import victim_effects, window_summary
    from slicemech.sim import episode as E
    from slicemech.tenants import poison
    from slicemech.tenants.strategies import Truthful

    t_job = time.perf_counter()
    out = Path(spec["out"])
    jid = spec["job"]
    jdir = out / "jobs"
    split, date, seed, arm = spec["split"], spec["date"], int(spec["seed"]), spec["arm"]
    kind, price = ARMS[arm]
    d = int(spec["d"])
    mech = Mechanism()
    cal = load_calibration(spec.get("cal", CAL_NAME))
    ckpt = Path(spec["model"]["path"])
    base, ckpt_sha = load_checked(spec["model"])
    if base.kind != kind:
        raise ValueError(f"checkpoint {ckpt} holds a {base.kind!r} net, arm {arm!r} needs {kind!r}")
    if base.forecaster is None:
        raise ValueError(f"checkpoint {ckpt} carries no operator forecaster (nets.save_net after ppo.train)")
    ov = dict(spec.get("overrides") or {})
    tf = {"test_freeze": spec["test_freeze"]} if split == "test" and spec.get("test_freeze") else {}
    wins = [cal.window(split, date, int(o), pool_part=ph, **tf, **ov) for o, ph in spec["layout"]]
    n_train = sum(1 for _, ph in spec["layout"] if ph == "A")
    train_ws, eval_ws = wins[:n_train], wins[n_train:]
    n = cal.n
    keys = dict(split=split, date=date, seed=seed, arm=arm)
    profs = [tuple(p) for p in spec["profiles"]]
    timings = {}

    # ---- phase 1: windows 1-4 with the base checkpoint under every profile (stored logs, cost of lying)
    t0 = time.perf_counter()
    attack, plans = {}, {}
    ids = [w.id for w in wins]
    for p in profs:
        plan = None
        if p != REF:
            plan = poison.plan_date(ids, poison.target_index(cal.classes, p[0]), p[1], n_train=n_train,
                                    n_eval=len(eval_ws))
            plans[p] = plan
        attack[p] = [E.run_window(w, base, price, [Truthful() for _ in range(n)] if plan is None
                                  else plan.policies(w.id, n), d, mech, cal.shape, record=False,
                                  price_columns="billed") for w in train_ws]
    timings["attack_s"] = time.perf_counter() - t0
    checks = dict(stored_log_equals_plan=True, attack_ref_truthful=True)
    for p, plan in plans.items():
        want, _ = poison.stored_reports(train_ws, plan)
        for w, df in zip(train_ws, attack[p]):
            if not np.array_equal(as_matrix(df, "report", w.K, n), want[w.id]):
                checks["stored_log_equals_plan"] = False
    for w, df in zip(train_ws, attack[REF]):
        if not np.array_equal(as_matrix(df, "report", w.K, n), np.minimum(w.theta, w.contract.C)):
            checks["attack_ref_truthful"] = False

    # ---- phase 2: retraining under both pipelines
    lr = float(spec["lr"])
    cfg = P.PPOConfig(iterations=int(spec["ft_iters"]), windows_per_iter=len(train_ws), M=int(spec["M"]),
                      lr_start=lr, lr_end=lr / 10.0, sigma0=float(spec["sigma0"]), eval_every=0, horizon=HORIZON)
    init_ls = spec.get("init_log_sigma")
    fseed = ft_seed(date, seed)
    nets, net_rows = {}, []
    t0 = time.perf_counter()
    for p in profs:
        records = [played_record(w, df) for w, df in zip(train_ws, attack[p])]
        logs = [as_matrix(df, "report", w.K, n) for w, df in zip(train_ws, attack[p])]
        for pipe in PIPELINES:
            t1 = time.perf_counter()
            res = retrain(base, records, pipe, stored_reports=logs, d=d, mech=mech, config=cfg, seed=fseed,
                          tenant_forecaster=cal.forecaster, init_log_sigma=init_ls)
            nets[(pipe, p)] = res.net
            hist = res.history
            net_rows.append(dict(**keys, pipeline=pipe, target=p[0], alpha=p[1], digest=N.param_digest(res.net),
                                 seconds=time.perf_counter() - t1, iterations=len(hist), log_sigma=res.log_sigma,
                                 n_projected=int(res.n_projected), report_rebuild=res.report_rebuild,
                                 frac_skipped=float(np.mean([h["skipped"] for h in hist])),
                                 reward_first=float(hist[0]["reward"]), reward_last=float(hist[-1]["reward"]),
                                 welfare_first=float(hist[0]["welfare"]), welfare_last=float(hist[-1]["welfare"])))
    timings["retrain_s"] = time.perf_counter() - t0
    base_vec = N.param_vector(base)
    vecs = {key: N.param_vector(net) for key, net in nets.items()}
    for row in net_rows:
        v, ref = vecs[(row["pipeline"], (row["target"], row["alpha"]))], vecs[(row["pipeline"], REF)]
        row.update(l2_to_base=float(np.linalg.norm(v - base_vec)), l2_to_ref=float(np.linalg.norm(v - ref)),
                   maxabs_to_ref=float(np.max(np.abs(v - ref))), l2_ref_to_base=float(np.linalg.norm(ref - base_vec)),
                   param_norm_base=float(np.linalg.norm(base_vec)))
    meas_digests = {r["digest"] for r in net_rows if r["pipeline"] == "measurement"}
    stored_ref = next(r["digest"] for r in net_rows if r["pipeline"] == "stored" and (r["target"], r["alpha"]) == REF)
    checks.update(h5a_bit_identical=len(meas_digests) == 1, n_measurement_digests=len(meas_digests),
                  measurement_equals_stored_ref=meas_digests == {stored_ref},
                  n_stored_digests=len({r["digest"] for r in net_rows if r["pipeline"] == "stored"}))

    # ---- phase 3: evaluation on windows 5-8 (truthful), shared by nets with the same parameters
    t0 = time.perf_counter()
    cache = {}

    def evaluate(net):
        dg = N.param_digest(net)
        if dg not in cache:
            cache[dg] = [E.run_window(w, net, price, [Truthful() for _ in range(n)], d, mech, cal.shape,
                                      record=False, price_columns="billed") for w in eval_ws]
        return dg, cache[dg]

    later = {("base", REF): evaluate(base)}
    for key, net in nets.items():
        later[key] = evaluate(net)
    timings["eval_s"] = time.perf_counter() - t0
    timings["eval_unique_nets"] = len(cache)

    # ---- tables
    attack_rows, later_rows, pair_rows = [], [], []
    for p in profs:
        for j, (w, df) in enumerate(zip(train_ws, attack[p])):
            attack_rows.append(window_summary(df, w.warmup, keys=dict(**keys, phase="attack", pipeline="deploy",
                                                                      target=p[0], alpha=p[1], window=w.id,
                                                                      win_index=j + 1)))
    for (pipe, p), (dg, dfs) in later.items():
        for j, (w, df) in enumerate(zip(eval_ws, dfs)):
            later_rows.append(window_summary(df, w.warmup, keys=dict(**keys, phase="later", pipeline=pipe,
                                                                     target=p[0], alpha=p[1], window=w.id,
                                                                     win_index=n_train + j + 1, digest=dg)))
    for p in profs:
        if p == REF:
            continue
        tgt = poison.target_index(cal.classes, p[0])
        phases = [("attack", "deploy", train_ws, attack[REF], attack[p], 1)]
        phases += [("later", pipe, eval_ws, later[(pipe, REF)][1], later[(pipe, p)][1], n_train + 1)
                   for pipe in PIPELINES]
        for phase, pipe, ws, refs, devs, first in phases:
            for j, (w, a, b) in enumerate(zip(ws, refs, devs)):
                row = dict(**keys, phase=phase, pipeline=pipe, target=p[0], target_index=tgt, alpha=p[1],
                           window=w.id, win_index=first + j)
                row.update(target_deltas(a, b, tgt, w.warmup))
                row.update(victim_effects(a, b, tgt, w.warmup))
                pair_rows.append(row)
    info = [{**keys, **window_info(w, cal.r_bar, mech), "win_index": j + 1} for j, w in enumerate(wins)]
    if spec["profiles"]:
        dsic = [r["d_utility_declared_target_max_interval"] for r in pair_rows if r["phase"] == "attack"]
        checks["attack_max_interval_gain_target"] = float(max(dsic)) if dsic else float("nan")

    write_parquet(pd.concat(attack_rows, ignore_index=True), jdir / f"{jid}.attack.parquet")
    write_parquet(pd.concat(later_rows, ignore_index=True), jdir / f"{jid}.later.parquet")
    write_parquet(pd.DataFrame(pair_rows), jdir / f"{jid}.pairs.parquet")
    write_parquet(pd.DataFrame(net_rows), jdir / f"{jid}.nets.parquet")
    write_parquet(pd.DataFrame(info), jdir / f"{jid}.windows.parquet")
    if spec.get("keep_episodes"):
        eps = []
        for p in profs:
            for w, df in zip(train_ws, attack[p]):
                eps.append(df.assign(phase="attack", pipeline="deploy", target=p[0], alpha=p[1], window=w.id))
        for (pipe, p), (dg, dfs) in later.items():
            for w, df in zip(eval_ws, dfs):
                eps.append(df.assign(phase="later", pipeline=pipe, target=p[0], alpha=p[1], window=w.id))
        ep = pd.concat(eps, ignore_index=True)
        for c in ep.columns:
            if ep[c].dtype == np.float64:
                ep[c] = ep[c].astype(np.float32)
        write_parquet(ep.assign(**keys), jdir / f"{jid}.episodes.parquet")
    timings["job_s"] = time.perf_counter() - t_job
    done = dict(job=jid, **keys, fp=spec.get("fp"), ckpt=str(ckpt), ckpt_sha256=ckpt_sha, model=spec["model"],
                hyper_source=spec.get("hyper_source"), ft_seed=fseed,
                config=dataclasses.asdict(cfg), init_log_sigma=init_ls, layout=[list(x) for x in spec["layout"]],
                overrides=ov, profiles=[list(p) for p in profs], window_ids=ids, checks=checks, timings=timings,
                d=d, cal=spec.get("cal", CAL_NAME), smoke=bool(spec.get("smoke")), code=code_fingerprint())
    write_json(jdir / f"{jid}.done.json", done)
    (jdir / f"{jid}.error.txt").unlink(missing_ok=True)
    return dict(job=jid, status="done", seconds=timings["job_s"], checks=checks)


def _run_safe(spec: dict) -> dict:
    try:
        return run_job(spec)
    except Exception:                                   # noqa: BLE001  (reported, job stays unfinished)
        tb = traceback.format_exc()
        p = Path(spec["out"]) / "jobs" / f"{spec['job']}.error.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tb)
        return dict(job=spec["job"], status="error", error=tb.strip().splitlines()[-1])


# ============================================================================================ aggregation
def job_errors(jdir: Path) -> tuple[list, list]:
    """(jobs whose ``error.txt`` has no ``done.json`` or is newer than it, the latter subset). An error newer than
    the record means the job's latest attempt failed: its record is not trusted (pre-freeze audit E5-2)."""
    errs, after = [], []
    if not jdir.is_dir():
        return errs, after
    for p in sorted(jdir.glob("*.error.txt")):
        j = p.name[: -len(".error.txt")]
        d = jdir / f"{j}.done.json"
        if not d.is_file():
            errs.append(j)
        elif p.stat().st_mtime_ns > d.stat().st_mtime_ns:
            errs.append(j)
            after.append(j)
    return errs, after


def aggregate(out: Path) -> dict:
    """Concatenate finished jobs and build ``contrasts.parquet`` and ``checks.json``. Returns the checks. A job whose
    ``error.txt`` is newer than its ``done.json`` is reported in ``jobs_error`` (and ``jobs_error_after_done``) and
    left out of the aggregates; ``code_versions`` counts the code fingerprints of the finished jobs."""
    jdir = out / "jobs"
    done = sorted(jdir.glob("*.done.json")) if jdir.is_dir() else []
    errors, after = job_errors(jdir)
    finished = [p.name[: -len(".done.json")] for p in done if p.name[: -len(".done.json")] not in set(after)]
    tables = {}
    for name in ("attack", "later", "pairs", "nets", "windows"):
        parts = [pd.read_parquet(jdir / f"{j}.{name}.parquet") for j in finished
                 if (jdir / f"{j}.{name}.parquet").is_file()]
        if parts:
            tables[name] = pd.concat(parts, ignore_index=True)
            write_parquet(tables[name].copy(), out / f"{name}.parquet")
    checks = dict(jobs_done=len(finished), jobs_error=errors, jobs_error_after_done=after)
    if "pairs" in tables:
        pr = tables["pairs"]
        idx = ["split", "date", "seed", "arm", "pipeline", "target", "alpha"]
        val = [c for c in pr.columns if c.startswith(("d_", "frac_")) or c == "dW_realized"]
        mean = pr.groupby(idx + ["phase"])[val].mean()
        tot = pr.groupby(idx + ["phase"])[["d_utility_declared_target_total",
                                          "d_utility_realized_target_total"]].sum()
        rows = []
        lat = pr[pr["phase"] == "later"]
        for key, g in lat.groupby(idx):
            dkey = dict(zip(idx, key))
            row = dict(dkey)
            m_l = mean.loc[key + ("later",)]
            row.update({f"later_{c}": float(m_l[c]) for c in val})
            akey = key[:4] + ("deploy",) + key[5:] + ("attack",)
            if akey in mean.index:
                m_a = mean.loc[akey]
                row.update({f"attack_{c}": float(m_a[c]) for c in val})
                for u in ("declared", "realized"):
                    c = f"d_utility_{u}_target_total"
                    row[f"net_{c}"] = float(tot.loc[akey][c] + tot.loc[key + ("later",)][c])
            rows.append(row)
        contrasts = pd.DataFrame(rows)
        if "windows" in tables and len(contrasts):
            wi = tables["windows"]
            loads = wi.groupby(["split", "date", "seed", "arm", "phase"])[
                [c for c in wi.columns if c.startswith("load_mbps_")] + ["offered_load", "capacity_load"]].mean()
            for ph, tag in (("A", "attack"), ("B", "later")):
                sub = loads.xs(ph, level="phase").add_prefix(f"{tag}_win_").reset_index()
                contrasts = contrasts.merge(sub, on=["split", "date", "seed", "arm"], how="left")
        write_parquet(contrasts, out / "contrasts.parquet")
        checks["contrast_rows"] = len(contrasts)
        meas = contrasts[contrasts["pipeline"] == "measurement"] if len(contrasts) else contrasts
        checks["measurement_later_gain_absmax"] = (float(meas["later_d_utility_declared_target"].abs().max())
                                                   if len(meas) else None)
    if finished:
        recs = [json.loads((jdir / f"{j}.done.json").read_text()) for j in finished]
        ck = [r["checks"] for r in recs]
        codes = {r.get("code") for r in recs}
        checks.update(code_versions=len(codes - {None}), jobs_without_code=sum(1 for r in recs if not r.get("code")))
        checks.update(
            h5a_groups=len(ck), h5a_all_bit_identical=all(c["h5a_bit_identical"] for c in ck),
            h5a_failures=[r["job"] for r in recs if not r["checks"]["h5a_bit_identical"]],
            measurement_equals_stored_ref_all=all(c["measurement_equals_stored_ref"] for c in ck),
            stored_log_equals_plan_all=all(c["stored_log_equals_plan"] for c in ck),
            attack_ref_truthful_all=all(c["attack_ref_truthful"] for c in ck),
            stored_digests_min=min(c["n_stored_digests"] for c in ck),
            protected_attack_max_interval_gain=max(
                [c.get("attack_max_interval_gain_target", float("nan")) for r, c in zip(recs, ck)
                 if r["arm"] == "protected"] or [float("nan")]),
            job_seconds_mean=float(np.mean([r["timings"]["job_s"] for r in recs])),
            job_seconds_max=float(np.max([r["timings"]["job_s"] for r in recs])))
    write_json(out / "checks.json", checks)
    return checks


# ============================================================================================ smoke checkpoints
def smoke_checkpoints(models_root: Path, seeds, iters: int, *, cal_name: str = CAL_NAME,
                      config: str = MODEL_CONFIG) -> None:
    """Tiny models for the smoke run (``iters`` PPO iterations on one short TRAIN window; not experiment nets),
    written in the registry layout ``<models_root>/<config>/<kind>/seed<s>/{model.pt, config.json}`` so the smoke
    run loads them exactly as a full run loads the trained models."""
    import torch
    torch.set_num_threads(1)
    from slicemech.config import Mechanism
    from slicemech.control import nets as N
    from slicemech.control import ppo as P
    from slicemech.data.calibrate import load_calibration
    cal = load_calibration(cal_name)
    date, off = SMOKE["ckpt_window"]
    w = None
    hyper = dict(id="smoke", lr=float(PARAMS["training"]["lr"]), sigma0=0.5,
                 M=int(PARAMS["training"]["counterfactual_samples"]))
    for kind in sorted({k for k, _ in ARMS.values()}):
        for s in seeds:
            d = R.model_dir(config, kind, s, root=models_root)
            if R.read_done(d / R.CONFIG_FILE) is not None and (d / R.MODEL_FILE).is_file():
                continue
            if w is None:
                w = cal.window("train", date, off, K=SMOKE["K"], warmup=SMOKE["warmup"])
            cfg = P.PPOConfig(iterations=iters, windows_per_iter=1, M=hyper["M"], lr_start=hyper["lr"],
                              lr_end=hyper["lr"] / 10, sigma0=hyper["sigma0"], eval_every=0, horizon=HORIZON)
            res = P.train(N.make_net(kind, seed=s), [w], None, d=D_DEFAULT, mech=Mechanism(), config=cfg, seed=s,
                          forecaster=cal.operator_forecaster(D_DEFAULT))
            d.mkdir(parents=True, exist_ok=True)
            N.save_net(res.net, d / R.MODEL_FILE)
            write_json(d / R.CONFIG_FILE, dict(
                status="done", smoke=True, config=config, arm=kind, seed=int(s), hyperparameters=hyper,
                result=dict(log_sigma=float(res.log_sigma), model_sha256=sio.file_sha256(d / R.MODEL_FILE),
                            param_digest=N.param_digest(res.net), iterations=iters)))


# ============================================================================================ main
def resolve_hyper(args, entry: dict) -> tuple[dict, dict]:
    """Fine-tuning hyper-parameters of a job: command line > the model's record > ``spec/params.yaml``."""
    h = entry.get("hyper") or {}
    src = {}

    def pick(name, cli, default):
        if cli is not None:
            src[name] = "command line"
            return cli
        if name in h:
            src[name] = "model record"
            return h[name]
        src[name] = "default"
        return default

    lr = float(pick("lr", args.lr, float(PARAMS["training"]["lr"])))
    M = int(pick("M", args.M, int(PARAMS["training"]["counterfactual_samples"])))
    sigma0 = float(pick("sigma0", args.sigma0, 0.5))
    if args.sigma0 is not None:
        init_ls, src["init_log_sigma"] = math.log(args.sigma0), "command line"
    elif entry.get("log_sigma") is not None:
        init_ls, src["init_log_sigma"] = float(entry["log_sigma"]), "model record (final log_sigma)"
    else:
        init_ls, src["init_log_sigma"] = math.log(sigma0), "sigma0"
    return dict(lr=lr, M=M, sigma0=min(max(sigma0, 0.05), 1.0), init_log_sigma=init_ls), src


def code_fingerprint() -> str:
    """SHA-256 over the ``slicemech`` package and this module (part of every job fingerprint)."""
    return sio.digest(dict(slicemech=R.package_sha256(), module=sio.file_sha256(Path(__file__))))


def job_fingerprint(job: dict, code: str) -> str:
    """Provenance fingerprint of one job: everything that determines its outputs (the model file's SHA-256, the
    hyper-parameters, layout, profiles, retraining settings, split, date, seed, arm, d, calibration file and the code
    fingerprint), not its output location or the freeze path."""
    from slicemech.data.calibrate import CALIB_DIR
    path = Path(job["model"]["path"])
    keep = {k: v for k, v in job.items() if k not in ("out", "test_freeze", "keep_episodes", "fp", "model",
                                                      "hyper_source", "archive_stamp")}
    return sio.digest(dict(keep, model_sha256=sio.file_sha256(path) if path.is_file() else None, code=code,
                           calibration_sha256=sio.file_sha256(CALIB_DIR / f"{job.get('cal', CAL_NAME)}.json")))


def job_done(out: Path, job: dict) -> bool:
    """A finished job whose record carries the job's current fingerprint."""
    p = out / "jobs" / f"{job['job']}.done.json"
    if not p.is_file():
        return False
    try:
        return json.loads(p.read_text()).get("fp") == job.get("fp")
    except (ValueError, OSError):
        return False


def build_jobs(args, out: Path) -> list[dict]:
    from slicemech.data import puffer
    from slicemech.tenants import poison
    smoke = args.smoke
    dates = ([SMOKE["date"]] if smoke else
             (args.dates.split(",") if args.dates else list(puffer.split_dates(args.split))))
    seeds = (SMOKE["seeds"] if smoke and not args.seeds else
             tuple(int(s) for s in args.seeds.split(",")) if args.seeds else tuple(SEEDS_DEFAULT))
    alphas = SMOKE["alphas"] if smoke and not args.alphas else (
        tuple(float(a) for a in args.alphas.split(",")) if args.alphas else poison.ALPHAS)
    targets = tuple(args.targets.split(",")) if args.targets else TARGETS
    arms = tuple(args.arms.split(",")) if args.arms else tuple(ARMS)
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from {tuple(ARMS)}")
    ov = dict(K=SMOKE["K"], warmup=SMOKE["warmup"]) if smoke else {}
    iters = SMOKE["iters"] if smoke else args.ft_iters
    lay = layout(args.split, smoke)
    models_root = args.models_root
    if smoke and not models_root and not args.ckpt_template:
        models_root = str(out / "models")
        if not args.plan:
            smoke_checkpoints(Path(models_root), seeds, SMOKE["iters"])
    jobs = []
    for date in dates:
        for s in seeds:
            for arm in arms:
                entry = model_entry(ARMS[arm][0], s, models_root=models_root, template=args.ckpt_template)
                hyper, src = resolve_hyper(args, entry)
                jobs.append(dict(job=f"{date}__{arm}__s{s}", split=args.split, date=date, seed=s, arm=arm,
                                 model=entry, d=args.d, out=str(out), layout=[list(x) for x in lay],
                                 profiles=[list(p) for p in profiles(targets, alphas)], overrides=ov,
                                 ft_iters=iters, **hyper, hyper_source=src, keep_episodes=args.keep_episodes,
                                 smoke=smoke, cal=CAL_NAME, test_freeze=args.test_freeze))
    code = code_fingerprint()
    for j in jobs:
        j["fp"] = job_fingerprint(j, code)
    return jobs


def estimate_core_hours(jobs: list[dict], smoke: bool) -> float:
    """Measured 2026-10-19 on one full DEV job per arm (2024-03-10, 9 profiles, 50 iterations; machine load average
    ~16 on hyper-threaded cores): 486 s per job = 110 s for the 36 windows-1-4 episodes (3.0 s each), 270 s for the
    18 retrainings (0.30 s per PPO iteration on 4 windows) and 100 s for the 40 evaluation episodes (10 distinct
    nets x 4 windows, 2.5 s each)."""
    if not jobs:
        return 0.0
    j = jobs[0]
    n_prof = len(j["profiles"])
    n_tr = sum(1 for _, ph in j["layout"] if ph == "A")
    n_ev = len(j["layout"]) - n_tr
    scale = (j["overrides"].get("K", 200) / 200.0) if smoke else 1.0
    per_job = scale * (n_prof * n_tr * 3.0                             # windows 1-4 under every profile
                       + n_prof * 2 * j["ft_iters"] * 0.075 * n_tr     # retraining, both pipelines
                       + (n_prof + 1) * n_ev * 2.5) + 3.0               # evaluation (distinct nets) + composition
    return per_job * len(jobs) / 3600.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("train", "dev", "test"), default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny TRAIN plumbing run (see module docstring)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dates", default=None, help="comma-separated dates (default: every date of the split)")
    ap.add_argument("--seeds", default=None,
                    help=f"checkpoint seeds (default {','.join(map(str, SEEDS_DEFAULT))}, SEEDS_DEFAULT)")
    ap.add_argument("--arms", default=None, help=f"subset of {tuple(ARMS)}")
    ap.add_argument("--targets", default=None, help=f"subset of {TARGETS}")
    ap.add_argument("--alphas", default=None, help="distortion factors besides the alpha = 1 reference")
    ap.add_argument("--d", type=int, default=D_DEFAULT, help="measurement delay (intervals)")
    ap.add_argument("--models-root", default=None,
                    help="root of the registry model layout (default results/models; smoke: its own tiny models)")
    ap.add_argument("--ckpt-template", default=None,
                    help="read plain nets.save_net files instead: path relative to the project root with {cal}, "
                         "{kind}, {seed}")
    ap.add_argument("--ft-iters", type=int, default=FT_ITERS)
    ap.add_argument("--lr", type=float, default=None,
                    help="fine-tuning start learning rate (end = lr / 10); default: the model's DEV-selected lr")
    ap.add_argument("--M", type=int, default=None, help="samples per state (default: the model's selected M)")
    ap.add_argument("--sigma0", type=float, default=None,
                    help="fine-tuning start exploration std (default: the model's final log_sigma)")
    ap.add_argument("--keep-episodes", action="store_true", help="also store per-interval episode tables (float32)")
    ap.add_argument("--test-freeze", default=None, help="frozen pre-registration file (TEST only)")
    ap.add_argument("--out", default=None, help="output directory (inside the project; default results/raw/...)")
    ap.add_argument("--force", action="store_true", help="recompute finished jobs")
    ap.add_argument("--plan", action="store_true", help="print the job list and cost estimate only")
    ap.add_argument("--aggregate", action="store_true", help="only rebuild the aggregates from finished jobs")
    args = ap.parse_args(argv)

    if args.smoke:
        if args.split not in (None, "train"):
            raise SystemExit("--smoke runs on TRAIN only")
        args.split = "train"
    if args.split is None:
        raise SystemExit("--split is required (or --smoke)")
    freeze = None
    if args.split == "test":
        R.refuse_test_overrides("test", args, ap, TEST_FIXED_OPTIONS)
        from slicemech.data import puffer_fetch
        freeze = puffer_fetch.require_test_read_gate(args.test_freeze)["path"]   # PermissionError before the freeze
        args.test_freeze = freeze
    elif not args.smoke and args.split not in SPLITS_FULL:
        raise SystemExit(f"E5 runs on {SPLITS_FULL} dates (the DEV pilot and TEST); TRAIN only with --smoke")
    out = sio.confine(Path(args.out)) if args.out else out_dir(args.split, args.smoke)
    if args.aggregate:
        print(json.dumps(aggregate(out), indent=1, default=str))
        return 0
    if not args.plan:
        sio.require_free_gb(3)
        out.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args, out)
    missing = sorted({j["model"]["path"] for j in jobs if j["model"]["missing"]})
    if missing and not args.plan:
        raise SystemExit("missing trained models (run experiments/train.py first, or pass --models-root / "
                         "--ckpt-template):\n  " + "\n  ".join(missing))
    if freeze and not args.plan:                                       # every model must be the frozen one
        R.verify_frozen(sorted({j["model"]["path"] for j in jobs}), freeze)
    todo = [j for j in jobs if args.force or not job_done(out, j)]
    stamp = None
    if not args.plan:
        import e0_setup as S
        stamp = S.archive_stamp()
    for j in todo:                       # one archive folder per invocation (not part of the fingerprint)
        j["archive_stamp"] = stamp
    est = estimate_core_hours(jobs, args.smoke)
    print(f"E5 {args.split}{' smoke' if args.smoke else ''}: {len(jobs)} jobs ({len(todo)} to run), "
          f"estimated {est:.2f} core-hours in total; output {out}")
    if args.plan:
        for j in jobs:
            print(" ", j["job"], j["model"]["path"], "" if not j["model"]["missing"] else "(missing)")
        return 0
    t0 = time.perf_counter()
    results = []
    if args.workers <= 1 or len(todo) <= 1:
        for j in todo:
            results.append(_run_safe(j))
            print(f"  {results[-1]['job']}: {results[-1]['status']} "
                  f"({results[-1].get('seconds', float('nan')):.1f} s)", flush=True)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(_run_safe, j) for j in todo]
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                print(f"  [{len(results)}/{len(todo)}] {r['job']}: {r['status']} "
                      f"({r.get('seconds', float('nan')):.1f} s){'  ' + r['error'] if 'error' in r else ''}",
                      flush=True)
    checks = aggregate(out)
    print(json.dumps(checks, indent=1, default=str))
    print(f"wall {time.perf_counter() - t0:.1f} s")
    return 1 if any(r["status"] != "done" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
