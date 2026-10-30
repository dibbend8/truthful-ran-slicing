"""Resumable training driver: PPO priority networks, DEV hyper-parameter selection and static fits (TRAIN + DEV only).

Run from ``code/`` with ``PYTHONDONTWRITEBYTECODE=1 /home/darpan/anaconda3/bin/python``:

    experiments/train.py --plan                         # job list, status and estimated core-hours; runs nothing
    experiments/train.py --workers 22                   # everything (resumes: finished jobs are skipped)
    experiments/train.py --only kind=calib              # only the sweep calibrations
    experiments/train.py --only kind=tune --only arm=protected,kappa0
    experiments/train.py --smoke --split train          # tiny end-to-end run on 2 TRAIN windows (no DEV access)
    experiments/train.py --smoke --split dev            # the same with DEV checkpoint selection on 2 DEV windows

Orchestrator decisions (binding, 2026-10-19; ``code/STATUS.md`` §0, ``prereg/PREREG_DRAFT.md`` §0):

* **D1** GBR fractions x 0.8 (g/mu = 0.40 video, 0.64 URLLC-like, 0.16 mMTC-like): in every calibration used here,
  including the sweep calibrations this driver creates (``spec/params.yaml`` -> ``data.calibrate``).
* **D2** Per-date offered-load normalization on every split: every TRAIN and DEV window is composed with its date's
  counts (``Calibration.window``); a load cell rho in {0.5, 0.7, 1.1} has its own TRAIN calibration at that rho
  (nested per-date counts, contract mu = its TRAIN mean demand), as STATUS.md §6 item 14 prescribes. A lead cell
  keeps the default per-date counts (count lead 4.004 s) and has its own calibration at its lead (video report,
  burst shape and operator AR change with the lead).
* **D6** The learner maximises the lookahead return with horizon h = 3 intervals (``PPOConfig.horizon = 3`` set
  explicitly, ``registry.ppo_config``): short-horizon reinforcement learning, not a pure contextual bandit.
* D3 (G4 dropped; load subgroups), D4 (harm on honest tenants), D5 (E5 protocol) and D7 (E4b kappa-scaling frontier)
  apply to the analysis drivers; this driver produces their models.

Jobs (``registry`` describes the configurations, arms and files):

* ``calib``  -- sweep calibrations ``n3_rho{0.5,0.7,1.1}`` and ``n3_lead{0,1.05}`` (``data.calibrate.calibrate`` on
  TRAIN, then the DEV per-date counts), written to ``data/calibration/``. Existing files are kept.
* ``static`` -- the static priority fit per configuration (``control.static.fit_static`` on all TRAIN windows, truthful
  reports; simplex grid at n = 3, CEM at n = 6, 9). The deployed vector is the one with the highest DEV
  selection-half welfare among the TRAIN fit's ``registry.STATIC_DEV_CANDIDATES`` = 72 best candidates, so the
  static comparator of H1c has the same DEV access as a learned arm (round-3 review; the TRAIN-best vector is kept
  as ``w_train_best``). The d cells share the default's fit (the static map reads no context).
* ``tune``   -- DEV tuning grid on the default configuration only: lr in {1e-4, 3e-4} (linear decay to lr/10) x
  sigma0 in {0.3, 0.6} x M in {4, 8}, for every learned arm (protected, unconstrained, meas_only, kappa0) and every
  tuning seed (default: the 5 main seeds).
* selection  -- per arm, the cell with the highest mean (over tuning seeds) DEV realized welfare of the
  checkpoint-selected net; ties keep the earlier grid cell. Written to ``results/models/selection.json`` and REUSED
  for every other configuration (each main job records the cell and this source).
* ``copy``   -- the selected cell's tuning runs become the default configuration's models (identical to retraining:
  same seed and inputs give bit-identical parameters).
* ``main``   -- sweep cells d in {1, 5, 15}, rho in {0.5, 0.7, 1.1}, lead in {0, 1.05} s, n = 6, n = 9: every arm x
  3 seeds, with the selected cell.

Training (``control.ppo.train``): all 180 TRAIN windows (32 sampled per iteration, 400 iterations, PPO clip 0.2, 4
passes, minibatch 2048, entropy 1e-3, grad clip 1), truthful reports; the net and the PPO seed are the run seed. The
checkpoint is selected every 50 iterations on the 66 windows of the DEV SELECTION half (``registry.dev_role``: the 11
DEV dates at even positions; realized welfare, credits per scored interval, truthful reports); the selected net is
re-evaluated there (must equal the selection value) and on TRAIN. The 10 dates of the DEV ESTIMATION half are never
read here: they give the unbiased DEV estimates of the power analysis (round-3 review: selecting and estimating on
the same 126 windows inflated the DEV effect sizes by an arm-dependent winner's curse).

Data rules. Training uses TRAIN; selection uses the DEV selection half. TEST is never used: ``--split test`` raises PermissionError
while the pre-registration freeze is absent (``data.puffer_fetch.require_test_read_gate``), and is refused after the
freeze as well, because no training or selection step may see TEST. ``--split train`` (selection on TRAIN, no DEV
read) is allowed only with ``--smoke``.

Memory and parallelism. A window set (180 TRAIN + 66 DEV selection windows of one calibration; about 0.75 GB at n = 3, 2.3 GB
at n = 9) is composed once in the parent when the first job that needs it starts; every job then runs in its own
forked child (copy-on-write; the child calls ``gc.freeze`` so the window pages stay shared), at most ``--workers`` at
a time and within ``--mem-gb`` (memory model measured with full window sets: see the constants below). Without the
sharing each worker would hold 2.7 GB (n = 3) to 4.7 GB (n = 9). The parent starts no Python threads; the only
native threads at fork time are Arrow's idle Parquet-reader pool and jemalloc's background thread (left by reading
the Puffer caches), which register at-fork handlers; the children never read Parquet. PyTorch cannot run autograd in
a child forked from a process that has already trained; the driver never trains in the parent, and a fork probe at
start-up falls back to running jobs in-process (with a warning) when it is called from such a process.
Dispatch order: calibrations, the long CEM static fits (n = 6, 9), the tuning runs, the other static fits, then the
sweep jobs, the longest first in each class. After every tuning run the selection is updated: as soon as an arm's
tuning is complete, its default models are copied and its sweep jobs start (no barrier between stages).

Resume. Each job writes its record last (``config.json`` / ``static.json``, atomic) with the SHA-256 of its spec
(configuration, arm, seed, hyper-parameters, PPO settings, calibration file hash, window keys, selection split).
Finished jobs are skipped; a record whose spec changed is "stale" and reruns only with ``--redo-stale``;
``--force`` reruns finished jobs. A killed run restarts the unfinished jobs from scratch.

Outputs: ``results/models/<config>/<arm>/seed<k>/{model.pt, config.json, history.csv}``, tuning runs below
``results/models/default/<arm>/tune/<cell>/``, static fits, ``selection.json``, ``index.csv``; logs in
``results/logs/train/`` (one driver log per invocation and one log per job with the per-iteration PPO lines). The
smoke run writes below ``results/raw/smoke_train/<split>/`` and ``results/logs/train_smoke_<split>/``.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import gc
import math
import os
import pathlib
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True
_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import registry as R  # noqa: E402
from slicemech import io as sio  # noqa: E402

KINDS = ("calib", "static", "tune", "copy", "main")
FILTER_KEYS = ("kind", "config", "arm", "seed", "cell", "calib")
REL_TOL_SELECTION = 1e-9      # re-evaluated DEV welfare vs the value ppo.train selected on (deterministic replay)

# Cost model (seconds), measured on this machine on 2026-10-19 (one process, light load; experiments/train.py
# --plan multiplies by CONTENTION for 22 concurrent workers on 24 cores / 32 threads).
PPO_S_PER_WINDOW_ITER_TENANT = {4: 0.0150, 8: 0.0235}   # rollout + M-sample lookahead + PPO update
EVAL_S_PER_WINDOW_TENANT = 0.0025                        # deterministic replay of one window
CALIB_S = 150.0                                          # calibrate() + DEV per-date counts
CONTENTION = 1.4
# Memory model (GB), measured 2026-10-19 with 8 forked workers on full window sets (total PSS): the parent holds
# 1.85 GB after composing the n = 3 set and 3.28 GB after the n = 9 set; each worker adds 0.35 (n = 3) / 0.52 (n = 9).
BASE_GB = 1.2                        # parent: interpreter, numba, torch import, data-module residue
SHARED_GB_PER_TENANT = 0.25          # one window set (180 TRAIN + 126 DEV windows, measured), shared by every worker
SHARED_GB_PER_TENANT_SMOKE = 0.01
PRIVATE_GB = 0.30                    # per running job: features, replay buffers, torch autograd
PRIVATE_GB_PER_TENANT = 0.03
CALIB_GB = 3.0                       # per calibration job (composes its own 180 TRAIN windows)


# ================================================================================================= jobs
@dataclass(frozen=True)
class Job:
    kind: str                 # one of KINDS
    config: str               # configuration (calib: the first configuration that needs the calibration)
    arm: str | None = None
    seed: int | None = None
    cell: str | None = None   # tuning cell id ('tune'); 'main' and 'copy' take the arm's selected cell
    calib: str | None = None  # calibration name ('calib')

    @property
    def id(self) -> str:
        if self.kind == "calib":
            return f"calib/{self.calib}"
        if self.kind == "static":
            return f"static/{self.config}"
        if self.kind == "tune":
            return f"tune/{self.config}/{self.arm}/{self.cell}/seed{self.seed}"
        return f"{self.kind}/{self.config}/{self.arm}/seed{self.seed}"

    @property
    def cfg(self) -> R.RunConfig:
        return R.CONFIGS[self.config]

    @property
    def data_key(self) -> str | None:
        return None if self.kind == "calib" else self.cfg.data_key


@dataclass(frozen=True)
class Ctx:
    """Run settings shared by the parent and the workers (picklable)."""

    split: str = "dev"                     # selection split: 'dev' (full runs) or 'train' (smoke only)
    smoke: bool = False
    models_root: str = str(R.MODELS_DIR)
    logs_dir: str = str(R.LOGS_DIR / "train")
    calib_root: str = str(R.CALIB_DIR)     # where this run writes/reads the sweep calibrations
    tune_seeds: tuple = R.SEEDS_MAIN
    workers: int = 22
    mem_gb: float = 20.0
    code_sha: str = ""

    def calib_dir(self, name: str) -> pathlib.Path:
        """Directory of calibration ``name``: sweep calibrations live in this run's ``calib_root``; the n3/n6/n9
        defaults always come from ``data/calibration``."""
        return pathlib.Path(self.calib_root) if name in R.CALIB_SPECS else R.CALIB_DIR


def make_ctx(split: str = "dev", smoke: bool = False, *, workers: int = 22, mem_gb: float = 20.0,
             tune_seeds=None, root=None) -> Ctx:
    """Run settings; ``root`` (tests) relocates models, logs and sweep calibrations below one directory."""
    if smoke:
        base = pathlib.Path(root) if root else R.SMOKE_DIR / split
        models, logs, calib = base / "models", (base / "logs" if root else R.LOGS_DIR / f"train_smoke_{split}"), \
            base / "calibration"
        seeds = (0,)
    else:
        base = pathlib.Path(root) if root else None
        models = base / "models" if base else R.MODELS_DIR
        logs = base / "logs" if base else R.LOGS_DIR / "train"
        calib = base / "calibration" if base else R.CALIB_DIR
        seeds = R.SEEDS_MAIN
    if tune_seeds is not None:
        seeds = tuple(int(s) for s in tune_seeds)
    return Ctx(split=split, smoke=bool(smoke), models_root=str(models), logs_dir=str(logs), calib_root=str(calib),
               tune_seeds=tuple(seeds), workers=int(workers), mem_gb=float(mem_gb))


def config_seeds(cfg: R.RunConfig, ctx: Ctx) -> tuple:
    return (0,) if ctx.smoke else tuple(cfg.seeds)


def build_jobs(ctx: Ctx) -> list[Job]:
    """Every job of a full (or smoke) run, in dependency order: calib, static, tune, copy/main."""
    jobs, seen = [], set()
    for cfg in R.CONFIGS.values():
        if cfg.calibration in R.CALIB_SPECS and cfg.calibration not in seen:
            seen.add(cfg.calibration)
            jobs.append(Job("calib", cfg.name, calib=cfg.calibration))
    jobs += [Job("static", c.name) for c in R.CONFIGS.values() if c.static_config == c.name]
    for arm in R.ARMS:
        for cell in R.tuning_cells():
            jobs += [Job("tune", R.DEFAULT, arm, int(s), cell.id) for s in ctx.tune_seeds]
    for cfg in R.CONFIGS.values():
        for arm in R.ARMS:
            for s in config_seeds(cfg, ctx):
                kind = "copy" if cfg.name == R.DEFAULT and s in ctx.tune_seeds else "main"
                jobs.append(Job(kind, cfg.name, arm, int(s)))
    return jobs


def parse_filters(items) -> dict:
    """``['kind=tune', 'arm=protected,kappa0']`` -> ``{'kind': {'tune'}, 'arm': {'protected', 'kappa0'}}``."""
    out: dict = {}
    for it in items or ():
        key, sep, vals = str(it).partition("=")
        key = key.strip()
        if not sep or key not in FILTER_KEYS or not vals.strip():
            raise ValueError(f"--only takes key=v1,v2 with key in {FILTER_KEYS}; got {it!r}")
        out.setdefault(key, set()).update(v.strip() for v in vals.split(",") if v.strip())
    return out


def matches(job: Job, filters: dict) -> bool:
    for key, vals in filters.items():
        v = getattr(job, key)
        if v is None or str(v) not in vals:
            return False
    return True


# ================================================================================================= specs & status
def _window_keys(split: str, smoke: bool) -> list[list]:
    return [[d, int(o)] for d, o in R.window_keys(split, smoke=smoke)]


def _selection_keys(ctx: Ctx) -> list[list]:
    """Windows of the checkpoint / cell / static-vector choice: the DEV selection half (``registry.dev_role``), or the
    smoke windows."""
    role = "select" if ctx.split == "dev" and not ctx.smoke else None
    return [[d, int(o)] for d, o in R.window_keys(ctx.split, smoke=ctx.smoke, role=role)]


class Resolver:
    """Specs and statuses of jobs; caches calibration hashes and the selection record."""

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self._sha: dict = {}
        self.selection: dict | None = None
        self.reload_selection()

    def reload_selection(self):
        self.selection = R.read_done(R.selection_path(root=self.ctx.models_root))

    def calib_path(self, name: str) -> pathlib.Path:
        return R.calibration_path(name, self.ctx.calib_dir(name))

    def calib_sha(self, name: str) -> str | None:
        p = self.calib_path(name)
        if not p.is_file():
            return None
        key = (str(p), p.stat().st_mtime_ns, p.stat().st_size)
        if key not in self._sha:
            self._sha[key] = sio.file_sha256(p)
        return self._sha[key]

    def selected(self, arm: str) -> dict | None:
        if self.selection is None:
            return None
        return self.selection["arms"].get(arm)

    # -- specs --------------------------------------------------------------------------------------------
    def spec(self, job: Job) -> dict | None:
        """The job's full spec, or None while it is blocked (missing calibration or selection)."""
        ctx = self.ctx
        if job.kind == "calib":
            spec = dict(kind="calib", calibration=job.calib, calibrate=R.CALIB_SPECS[job.calib], smoke=ctx.smoke,
                        train_windows=_window_keys("train", True) if ctx.smoke else "all",
                        dev_counts=bool(ctx.split == "dev" and not ctx.smoke))
            return spec
        cfg = job.cfg
        sha = self.calib_sha(cfg.calibration)
        if sha is None:
            return None
        base = dict(calibration=dict(name=cfg.calibration, sha256=sha), smoke=ctx.smoke,
                    train_windows=_window_keys("train", ctx.smoke),
                    selection=dict(split=ctx.split, windows=_selection_keys(ctx)))
        if job.kind == "static":
            return dict(kind="static", config=R.config_record(cfg), method=_static_kwargs(cfg.n, ctx.smoke),
                        report_source="theta", dev_choice=dict(top_k=R.STATIC_DEV_CANDIDATES,
                                                               rule=STATIC_CHOICE_RULE), **base)
        if job.kind == "tune":
            cell = R.cell_by_id(job.cell)
            source = "tuning grid"
        elif job.kind in ("main", "copy"):
            sel = self.selected(job.arm)
            if sel is None:
                return None
            cell = R.cell_by_id(sel["cell"])
            source = "DEV selection on the default configuration (results/models/selection.json)"
            if job.kind == "copy":
                src = Job("tune", R.DEFAULT, job.arm, job.seed, cell.id)
                src_spec = self.spec(src)
                rec = R.read_done(self.marker(src))
                if src_spec is None or rec is None or rec.get("spec_digest") != R.spec_digest(src_spec):
                    return None
                return dict(kind="copy", source=src.id, source_spec_digest=rec["spec_digest"], cell=cell.id)
        else:
            raise ValueError(job.kind)
        pcfg = R.ppo_config(cell, smoke=ctx.smoke)
        return dict(kind="train", config=R.config_record(cfg), arm=job.arm, seed=int(job.seed),
                    net=dict(kind=R.ARM_NET_KIND[job.arm], seed=int(job.seed)),
                    cell=dict(id=cell.id, **cell.as_dict()), hyperparameter_source=source,
                    ppo=dataclasses.asdict(pcfg), report_source="theta", dev_report_source="theta", **base)

    def marker(self, job: Job) -> pathlib.Path:
        root = self.ctx.models_root
        if job.kind == "calib":
            return self.calib_path(job.calib)
        if job.kind == "static":
            return R.static_dir(job.config, root=root) / R.STATIC_FILE
        if job.kind == "tune":
            return R.tune_dir(job.arm, job.cell, job.seed, root=root) / R.CONFIG_FILE
        return R.model_dir(job.config, job.arm, job.seed, root=root) / R.CONFIG_FILE

    def status(self, job: Job) -> tuple[str, dict | None]:
        """('done' | 'stale' | 'pending' | 'blocked:<why>', spec)."""
        if job.kind == "calib":
            return ("done" if self.marker(job).is_file() else "pending"), self.spec(job)
        if self.calib_sha(job.cfg.calibration) is None:
            return f"blocked:calibration {job.cfg.calibration}", None
        spec = self.spec(job)
        if spec is None:
            why = "selection" if self.selected(job.arm) is None else "tuning run"
            return f"blocked:{why}", None
        rec = R.read_done(self.marker(job))
        if rec is None:
            return "pending", spec
        return ("done" if rec.get("spec_digest") == R.spec_digest(spec) else "stale"), spec


def _static_kwargs(n: int, smoke: bool) -> dict:
    kw = dict(method="grid" if n == 3 else "cem", seed=0)
    if n == 3:
        kw.update(steps=20, floor=0.05)
    else:
        kw.update(pop=16, iters=2, elite=4) if smoke else kw.update(pop=64, iters=20, elite=8)
    return kw


# ================================================================================================= cost / memory model
def n_windows(split: str, smoke: bool) -> int:
    role = "select" if split == "dev" and not smoke else None
    return len(R.window_keys(split, smoke=smoke, role=role))


def est_seconds(job: Job, ctx: Ctx, M: int | None = None) -> float:
    """Single-process estimate (s) on an otherwise idle machine. ``M`` overrides the cell's sample count (main jobs
    before the selection: 8, the upper bound)."""
    if job.kind == "copy":
        return 0.0
    if job.kind == "calib":
        return 15.0 if ctx.smoke else CALIB_S
    n = job.cfg.n
    wt, ws = n_windows("train", ctx.smoke), n_windows(ctx.split, ctx.smoke)
    ev = EVAL_S_PER_WINDOW_TENANT * n
    if job.kind == "static":
        kw = _static_kwargs(n, ctx.smoke)
        cands = 232 * 0.6 if kw["method"] == "grid" else kw["pop"] * kw["iters"]
        return cands * wt * ev + (R.STATIC_DEV_CANDIDATES + 1) * ws * ev
    if M is None:
        M = R.cell_by_id(job.cell).M if job.kind == "tune" else _SELECTED_M.get(job.arm, 8)
    pc = R.ppo_config(R.Cell(3e-4, 0.6, M), smoke=ctx.smoke)
    wpi = min(pc.windows_per_iter, wt)
    evals = pc.iterations // pc.eval_every + 1 if pc.eval_every else 0
    return pc.iterations * wpi * n * PPO_S_PER_WINDOW_ITER_TENANT[M] + (evals + 1) * ws * ev + wt * ev


def shared_gb(key: str, ctx: Ctx) -> float:
    """Memory of one composed window set (TRAIN + selection windows) in the parent, shared by forked workers."""
    per = SHARED_GB_PER_TENANT_SMOKE if ctx.smoke else SHARED_GB_PER_TENANT
    return R.CONFIGS[_cfg_of_key(key)].n * per


def job_gb(job: Job) -> float:
    """Private memory of a running job (calibration jobs compose their own windows)."""
    return CALIB_GB if job.kind == "calib" else PRIVATE_GB + PRIVATE_GB_PER_TENANT * job.cfg.n


def _cfg_of_key(key: str) -> str:
    for c in R.CONFIGS.values():
        if c.data_key == key:
            return c.name
    raise KeyError(key)


#: Dispatch order: calibrations (they unblock jobs), the long CEM static fits, tuning, the other static fits, then the
#: sweep jobs; within a class the longest job first.
_RANK = {"calib": 0, "static_cem": 1, "tune": 2, "static": 3, "main": 4, "copy": 5}
_SELECTED_M: dict = {}


def priority(job: Job, ctx: Ctx) -> tuple:
    kind = "static_cem" if job.kind == "static" and job.cfg.n > 3 else job.kind
    return (_RANK[kind], -est_seconds(job, ctx), job.id)


def simulate(jobs: list[Job], ctx: Ctx, workers: int) -> float:
    """Wall-clock (s) of a list-scheduling run of ``jobs`` on ``workers`` with the cost model (x CONTENTION); main
    jobs of an arm wait for all its tuning runs and sweep-cell jobs for their calibration. Memory is ignored."""
    pending = sorted((j for j in jobs if j.kind != "copy"), key=lambda j: priority(j, ctx))
    tune_left = {a: sum(1 for j in pending if j.kind == "tune" and j.arm == a) for a in R.ARMS}
    calib_left = {j.calib for j in pending if j.kind == "calib"}
    t, running = 0.0, []                                   # (end, job)

    def ready(j):
        if j.kind != "calib" and j.cfg.calibration in calib_left:
            return False
        return not (j.kind == "main" and tune_left.get(j.arm, 0) > 0)

    while pending or running:
        while len(running) < workers:
            j = next((j for j in pending if ready(j)), None)
            if j is None:
                break
            pending.remove(j)
            running.append((t + est_seconds(j, ctx) * CONTENTION, j))
        if not running:
            break                                          # blocked forever (e.g. tuning filtered out)
        running.sort(key=lambda x: x[0])
        t, j = running.pop(0)
        if j.kind == "tune":
            tune_left[j.arm] -= 1
        elif j.kind == "calib":
            calib_left.discard(j.calib)
    return t


# ================================================================================================= logging
class Log:
    def __init__(self, ctx: Ctx, name: str = "train"):
        d = pathlib.Path(ctx.logs_dir)
        d.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = d / f"{name}_{stamp}_{os.getpid()}.log"

    def __call__(self, *msg):
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] " + " ".join(str(m) for m in msg)
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def job_log_path(job: Job, ctx: Ctx) -> pathlib.Path:
    return pathlib.Path(ctx.logs_dir) / "jobs" / (job.id.replace("/", "__") + ".log")


# ================================================================================================= workers
#: data key -> dict(cal, train, sel): composed in the parent before the pool is forked (shared copy-on-write).
_SHARED: dict = {}


def compose_data(key: str, ctx: Ctx) -> dict:
    cal = R.load_calibration(key, ctx.calib_dir(key))
    tk = R.window_keys("train", smoke=ctx.smoke)
    train = R.compose_windows(cal, "train", tk)
    if ctx.split == "dev":
        sel = R.compose_windows(cal, "dev", R.window_keys("dev", smoke=ctx.smoke,
                                                         role=None if ctx.smoke else "select"))
    elif ctx.split == "train" and ctx.smoke:
        sel = train
    else:
        raise PermissionError(f"selection split {ctx.split!r} is not allowed for training")
    return dict(cal=cal, train=train, sel=sel)


def _execute(payload) -> dict:
    """Run one job (worker process, or the parent with --workers 1). Never raises: failures are returned."""
    job, spec, ctx = payload
    t0 = time.perf_counter()
    log = job_log_path(job, ctx)
    log.parent.mkdir(parents=True, exist_ok=True)
    import torch
    threads = torch.get_num_threads()
    torch.set_num_threads(1)                 # restored below: with --workers 1 this runs in the parent
    try:
        with open(log, "a") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            print(f"=== {job.id} start {_dt.datetime.now().isoformat(timespec='seconds')} pid {os.getpid()}")
            out = _RUNNERS[job.kind](job, spec, ctx)
            print(f"=== {job.id} done in {time.perf_counter() - t0:.1f} s: {out}")
        return dict(id=job.id, status="done", seconds=time.perf_counter() - t0, summary=out)
    except Exception as exc:   # noqa: BLE001 -- the parent logs it and continues with the other jobs
        tb = traceback.format_exc()
        with open(log, "a") as fh:
            fh.write(tb)
        return dict(id=job.id, status="failed", seconds=time.perf_counter() - t0, error=repr(exc), log=str(log))
    finally:
        torch.set_num_threads(threads)


def _versions() -> dict:
    import torch
    return dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)


def _record(job: Job, spec: dict, ctx: Ctx, **fields) -> dict:
    return dict(status="done", job=job.id, spec_digest=R.spec_digest(spec), spec=spec, decisions=R.DECISIONS,
                code=dict(slicemech_sha256=ctx.code_sha), versions=_versions(), host=platform.node(),
                created=_dt.datetime.now().isoformat(timespec="seconds"), **fields)


def _run_train(job: Job, spec: dict, ctx: Ctx) -> dict:
    from slicemech.control import nets, ppo
    cfg = job.cfg
    data = _SHARED[cfg.data_key]
    cal = data["cal"]
    mech = R.mechanism(cal)
    fc = cal.operator_forecaster(cfg.d)
    cell = R.cell_by_id(spec["cell"]["id"])
    pcfg = R.ppo_config(cell, smoke=ctx.smoke)
    if dataclasses.asdict(pcfg) != spec["ppo"]:
        raise RuntimeError("PPO configuration differs from the job spec")
    net0 = nets.make_net(R.ARM_NET_KIND[job.arm], seed=job.seed)
    dev = data["sel"] if ctx.split == "dev" else None
    res = ppo.train(net0, data["train"], dev, d=cfg.d, mech=mech, config=pcfg, seed=job.seed, forecaster=fc,
                    verbose=True)
    ev = ppo.evaluate(res.net, data["sel"], d=cfg.d, mech=mech)
    if dev is not None:
        ref = res.best_dev_welfare
        if ref is None or abs(ev["welfare"] - ref) > REL_TOL_SELECTION * max(1.0, abs(ref)):
            raise ArithmeticError(f"DEV re-evaluation {ev['welfare']!r} differs from the selection value {ref!r}")
    tr = ppo.evaluate(res.net, data["train"], d=cfg.d, mech=mech)
    out = (R.tune_dir(job.arm, job.cell, job.seed, root=ctx.models_root) if job.kind == "tune"
           else R.model_dir(job.config, job.arm, job.seed, root=ctx.models_root))
    sio.require_free_gb(3)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (R.MODEL_FILE + ".tmp")
    nets.save_net(res.net, tmp)
    os.replace(tmp, out / R.MODEL_FILE)
    import pandas as pd
    hist = pd.DataFrame(res.history)
    hist.to_csv(out / R.HISTORY_FILE, index=False, float_format="%.10g")
    result = dict(
        best_iteration=int(res.best_iteration), best_dev_welfare=res.best_dev_welfare,
        selection_split=ctx.split, selection_welfare=float(ev["welfare"]),
        selection_per_window=[float(x) for x in ev["per_window"]],
        train_welfare=float(tr["welfare"]), log_sigma=float(res.log_sigma), sigma=math.exp(res.log_sigma),
        n_projected=int(res.n_projected), iterations=len(res.history),
        final_train_welfare=float(res.history[-1]["welfare"]),
        dev_curve={int(h["iteration"]): h["dev_welfare"] for h in res.history if "dev_welfare" in h},
        param_digest=nets.param_digest(res.net), n_parameters=int(res.net.n_parameters()),
        model_sha256=sio.file_sha256(out / R.MODEL_FILE), forecaster_horizon=fc.horizon,
        train_seconds=float(sum(h["seconds"] for h in res.history)))
    sio.write_json(out / R.CONFIG_FILE, _record(job, spec, ctx, result=result, net_spec=res.net.spec(),
                                                hyperparameters=spec["cell"]))
    return dict(selection_welfare=round(result["selection_welfare"], 6), best_iteration=result["best_iteration"],
                train_welfare=round(result["train_welfare"], 6))


def _run_static(job: Job, spec: dict, ctx: Ctx) -> dict:
    from slicemech.control import ppo, static
    cfg = job.cfg
    data = _SHARED[cfg.data_key]
    mech = R.mechanism(data["cal"])
    kw = dict(spec["method"])
    fit = static.fit_static(data["train"], mech, **kw)
    sel = ppo.prepare(data["sel"], mech=mech, report_source="theta")
    tr = ppo.prepare(data["train"], mech=mech, report_source="theta")
    eq = np.full((1, cfg.n), 1.0 / cfg.n)
    roll = dict(mech=mech, env_factory=ppo.fluid_env_factory, quota_fn=ppo.default_quota_fn)
    # round-3 review (H1c information symmetry): the deployed vector is the DEV-selection-half best of the TRAIN
    # fit's top-K candidates, exactly as a learned arm's checkpoint and cell are DEV-selected TRAIN candidates
    choice = static_dev_choice(fit, sel, roll, int(spec["dev_choice"]["top_k"]))
    w_dep = np.asarray(choice["w"], float)
    w_sel = float(choice["dev_welfare"])
    eq_sel = float(static.rollout_fixed(sel, eq, **roll)[0])
    eq_tr = float(static.rollout_fixed(tr, eq, **roll)[0])
    out = R.static_dir(job.config, root=ctx.models_root)
    sio.require_free_gb(3)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / ("tmp_" + R.STATIC_CANDIDATES)
    np.savez_compressed(tmp, candidates=np.asarray(fit.candidates, float), scores=np.asarray(fit.scores, float))
    os.replace(tmp, out / R.STATIC_CANDIDATES)
    shared_with = [c.name for c in R.CONFIGS.values() if c.static_config == job.config]
    sio.write_json(out / R.STATIC_FILE, _record(
        job, spec, ctx, w=w_dep.tolist(), method=fit.method, welfare_train=float(choice["train_welfare"]),
        welfare_selection=w_sel, selection_split=ctx.split, equal_welfare_train=eq_tr, equal_welfare_selection=eq_sel,
        w_train_best=fit.w.tolist(), welfare_train_best=float(fit.welfare),
        welfare_selection_train_best=float(choice["dev_welfare_train_best"]), dev_choice=choice["table"],
        dev_choice_rank=int(choice["rank"]), n_candidates=int(len(fit.scores)), seconds=float(fit.seconds),
        info=fit.info, used_by_configs=shared_with))
    return dict(w=np.round(w_dep, 4).tolist(), welfare_train=round(float(choice["train_welfare"]), 6),
                welfare_selection=round(w_sel, 6), train_rank=int(choice["rank"]))


STATIC_CHOICE_RULE = ("deployed static vector = highest DEV selection-half realized welfare among the TRAIN fit's top-K "
                      "candidates by TRAIN welfare (ties: better TRAIN rank)")


def static_dev_choice(fit, sel, roll: dict, top_k: int) -> dict:
    """The DEV choice of the static vector (``STATIC_CHOICE_RULE``): the TRAIN fit's ``top_k`` best distinct
    candidates are replayed on the selection windows ``sel`` (prepared) and the best is deployed. Returns
    ``dict(w, rank, train_welfare, dev_welfare, dev_welfare_train_best, table)``; rank 0 = the TRAIN-best vector."""
    from slicemech.control import static
    C = np.asarray(fit.candidates, float)
    sc = np.asarray(fit.scores, float)
    order = np.argsort(-sc, kind="stable")
    keep, seen = [], set()
    for j in order:
        key = tuple(np.round(C[j], 12))
        if key in seen:
            continue
        seen.add(key)
        keep.append(int(j))
        if len(keep) >= int(top_k):
            break
    if not keep or not np.allclose(C[keep[0]], np.asarray(fit.w, float)):
        keep = [None] + keep                                   # the fit's own vector first (CEM's mean may differ)
    W = np.stack([np.asarray(fit.w, float) if j is None else C[j] for j in keep])
    trw = np.array([float(fit.welfare) if j is None else float(sc[j]) for j in keep])
    dev = np.asarray(static.rollout_fixed(sel, W, **roll), float)
    best = int(np.argmax(dev))                                 # first maximum = better TRAIN rank on ties
    table = [dict(rank=r, w=W[r].tolist(), train_welfare=float(trw[r]), dev_welfare=float(dev[r]))
             for r in range(len(W))]
    return dict(w=W[best].tolist(), rank=best, train_welfare=float(trw[best]), dev_welfare=float(dev[best]),
                dev_welfare_train_best=float(dev[0]), table=table)


def _run_calib(job: Job, spec: dict, ctx: Ctx) -> dict:
    from slicemech.data.calibrate import attach_date_counts, calibrate
    kw = dict(spec["calibrate"])
    keys = [tuple(k) for k in spec["train_windows"]] if ctx.smoke else None
    cal = calibrate(n=int(kw["n"]), rho=float(kw["rho"]), lead_s=float(kw["lead_s"]), train_windows=keys,
                    diag_windows=keys)
    if spec["dev_counts"]:
        cal = attach_date_counts(cal, ("dev",))
    sio.require_free_gb(3)
    path = R.calibration_path(job.calib, ctx.calib_dir(job.calib))
    sio.write_json(path, cal.to_dict())
    targets = cal.diagnostics.get("targets", {})
    marker = pathlib.Path(ctx.logs_dir) / "calib" / f"{job.calib}.json"
    sio.write_json(marker, _record(job, spec, ctx, path=str(path), sha256=sio.file_sha256(path),
                                   offered_load=cal.meta.get("offered_load"), regime_targets=targets,
                                   mu=np.asarray(cal.contract.mu).tolist(), counts_ref=list(cal.counts)))
    print("regime diagnostics (TRAIN; targets are pre-registered for the default configuration only):",
          {k: v for k, v in cal.diagnostics.items() if k not in ("windows",)})
    return dict(path=str(path), offered_load=round(float(cal.meta.get("offered_load", float("nan"))), 4),
                targets_pass=bool(all(v["ok"] for v in targets.values())))


_RUNNERS = {"tune": _run_train, "main": _run_train, "static": _run_static, "calib": _run_calib}


# ================================================================================================= parent
def _warm_kernels(key: str):
    """Compile the numba fluid kernels in the parent so forked workers inherit them (numba cache=False)."""
    from slicemech.control import ppo
    w = _SHARED[key]["train"][0]
    p = ppo.prepare([w], mech=R.mechanism(_SHARED[key]["cal"]), report_source="theta")[0]
    env = ppo.fluid_env_factory(w, R.mechanism(_SHARED[key]["cal"]))
    eq = np.full(p.n, 1.0 / p.n)
    Q = ppo.quotas(ppo.default_quota_fn, p.z_alloc, eq, p.r_hat, p.contract, R.mechanism(_SHARED[key]["cal"]))
    env.replay(Q)
    ks = p.scored[:2]
    env.lookahead_at(ks, np.repeat(Q[ks][:, None, :], 2, axis=1), R.HORIZON)
    env.counterfactual_at(ks, np.repeat(Q[ks][:, None, :], 2, axis=1))


def _clear_data_caches():
    from slicemech.data import calibrate, puffer, raca, tractor
    for mod in (puffer, raca, tractor, calibrate):
        for v in vars(mod).values():
            if callable(getattr(v, "cache_clear", None)):
                v.cache_clear()


@dataclass
class _Running:
    job: Job
    proc: object
    conn: object
    keys: frozenset             # window sets the child shares (composed when it was forked)
    gb: float
    t0: float


def _child(payload, conn):
    """Forked worker: run one job and send its result. ``gc.freeze`` first, so the child's collector never writes to
    the parent's objects (the composed windows stay shared copy-on-write)."""
    gc.freeze()
    try:
        r = _execute(payload)
    except BaseException as exc:   # noqa: BLE001 -- e.g. KeyboardInterrupt; report and exit
        r = dict(id=payload[0].id, status="failed", seconds=0.0, error=repr(exc), log="")
    if len(str(r.get("error", ""))) > 2000:
        r["error"] = str(r["error"])[:2000] + " ..."
    conn.send(r)
    conn.close()


def _probe_child(conn):
    try:
        import torch
        torch.set_num_threads(1)
        x = torch.ones(2, dtype=torch.float64, requires_grad=True)
        (x * x).sum().backward()
        conn.send(None)
    except Exception as exc:   # noqa: BLE001
        conn.send(repr(exc))
    conn.close()


def fork_probe(timeout: float = 120.0) -> str | None:
    """None when a forked child can train; else the reason. PyTorch refuses autograd in a child forked from a process
    that has already run autograd (its engine threads cannot be forked); the CLI parent never trains, but a caller
    that trained in-process first (an interactive session, a test suite) would make every job fail."""
    import multiprocessing as mp
    mpc = mp.get_context("fork")
    rx, tx = mpc.Pipe(duplex=False)
    proc = mpc.Process(target=_probe_child, args=(tx,), name="fork-probe")
    proc.start()
    tx.close()
    msg = rx.recv() if rx.poll(timeout) else "fork probe timed out"
    proc.join()
    rx.close()
    return msg


class Scheduler:
    """Dynamic list scheduler: one forked child per job, at most ``ctx.workers`` at once, within ``ctx.mem_gb``.

    The parent composes a window set when the first job that needs it is dispatched and drops it when no job of the
    selection needs it any more; a child forked while a set was composed keeps it mapped until it exits, so such
    sets keep counting against the budget ("ghosts"). The parent starts no Python threads (see the module
    docstring). After every tuning run the DEV selection is updated; when an arm's tuning is complete its selection
    is written, the default configuration's models are copied from the selected cell and the arm's sweep jobs become
    dispatchable. A child's result is a small dict (errors truncated), so the pipe never fills before it exits.
    """

    def __init__(self, res: Resolver, log: Log, jobs: list[Job], *, force: bool, redo_stale: bool):
        self.res, self.log, self.ctx = res, log, res.ctx
        self.jobs = jobs
        self.todo: list[Job] = []
        self.skipped: dict = {}
        for j in jobs:
            st, _ = res.status(j)
            if st == "done" and not (force and j.kind != "calib"):
                continue
            if st == "stale" and not (force or redo_stale):
                self.skipped.setdefault("stale (use --redo-stale)", []).append(j.id)
                continue
            self.todo.append(j)
        self.running: dict = {}
        self.composed: dict = {}
        self.results: list = []
        self.n_total = len([j for j in self.todo if j.kind != "copy"])
        self._warm = False
        self.workers = max(1, int(self.ctx.workers))

    # -- memory ----------------------------------------------------------------------------------------
    def mem_used(self, extra_keys=()) -> float:
        keys = set(self.composed) | set(extra_keys)
        for r in self.running.values():
            keys |= r.keys
        return BASE_GB + sum(shared_gb(k, self.ctx) for k in keys) + sum(r.gb for r in self.running.values())

    def _needed_keys(self) -> set:
        return {j.data_key for j in self.todo if j.data_key} | {r.job.data_key for r in self.running.values()
                                                               if r.job.data_key}

    def _evict_unneeded(self):
        for k in [k for k in self.composed if k not in self._needed_keys()]:
            del self.composed[k]
            _SHARED.pop(k, None)
            self.log(f"dropped window set {k} (no remaining job needs it)")
        gc.collect()

    def _compose(self, key: str):
        t1 = time.perf_counter()
        _SHARED[key] = compose_data(key, self.ctx)
        self.composed[key] = shared_gb(key, self.ctx)
        d = _SHARED[key]
        nsel = len(d["sel"]) if d["sel"] is not d["train"] else 0
        self.log(f"composed {key}: {len(d['train'])} TRAIN + {nsel} DEV windows in {time.perf_counter() - t1:.1f} s"
                 + ("" if nsel else " (selection on the TRAIN windows: smoke)"))
        if not self._warm:
            _warm_kernels(key)
            self._warm = True
        _clear_data_caches()
        gc.collect()

    # -- dispatch --------------------------------------------------------------------------------------
    def _dispatchable(self) -> list[Job]:
        out = [j for j in self.todo if j.kind != "copy" and self.res.spec(j) is not None]
        return sorted(out, key=lambda j: priority(j, self.ctx))

    def _fits(self, job: Job) -> bool:
        extra = (job.data_key,) if job.data_key and job.data_key not in self.composed else ()
        return not self.running or self.mem_used(extra) + job_gb(job) <= self.ctx.mem_gb

    def _start(self, job: Job):
        spec = self.res.spec(job)
        self.todo.remove(job)
        if job.data_key and job.data_key not in self.composed:
            self._evict_unneeded()
            try:
                self._compose(job.data_key)
            except Exception as exc:   # noqa: BLE001 -- the job fails; the others continue
                self._finish(job, dict(id=job.id, status="failed", seconds=0.0, log=str(self.log.path),
                                       error=f"composing window set {job.data_key} failed: {exc!r}"))
                return
        payload = (job, spec, self.ctx)
        if self.workers <= 1:
            self._finish(job, _execute(payload))
            return
        import multiprocessing as mp
        mpc = mp.get_context("fork")
        rx, tx = mpc.Pipe(duplex=False)
        proc = mpc.Process(target=_child, args=(payload, tx), name=job.id, daemon=False)
        proc.start()
        tx.close()
        self.running[proc.sentinel] = _Running(job, proc, rx, frozenset(self.composed), job_gb(job),
                                               time.perf_counter())

    def _finish(self, job: Job, r: dict):
        self.results.append(r)
        k = len(self.results)
        if r["status"] == "done":
            self.log(f"[{k}/{self.n_total}] done {r['id']} {r['seconds']:.0f} s {r.get('summary', '')}")
        else:
            self.log(f"[{k}/{self.n_total}] FAILED {r['id']}: {r.get('error')} (log: {r.get('log')})")
        if job.kind == "tune" and r["status"] == "done":
            self._after_tuning()

    def _after_tuning(self):
        update_selection(self.res, self.log)
        for j in [j for j in self.todo if j.kind == "copy" and self.res.spec(j) is not None]:
            run_copy(j, self.res, self.log)
            self.todo.remove(j)

    def _reap(self, timeout=None):
        from multiprocessing.connection import wait
        ready = wait(list(self.running), timeout=timeout)
        for s in ready:
            run = self.running.pop(s)
            r = None
            try:
                if run.conn.poll():
                    r = run.conn.recv()
            except (EOFError, OSError):
                r = None
            run.proc.join()
            run.conn.close()
            if r is None:
                r = dict(id=run.job.id, status="failed", seconds=time.perf_counter() - run.t0,
                         error=f"worker exited with code {run.proc.exitcode} without a result",
                         log=str(job_log_path(run.job, self.ctx)))
            self._finish(run.job, r)

    def run(self) -> list[dict]:
        self._after_tuning()                     # selection and copies from earlier runs
        if self.workers > 1 and any(j.kind != "copy" for j in self.todo):
            why = fork_probe()
            if why is not None:
                self.log(f"WARNING: forked workers cannot train in this process ({why}); running the jobs one by one "
                         "in this process instead. Start train.py in a fresh process to use --workers.")
                self.workers = 1
        self.log(f"{self.n_total} job(s) to run on up to {self.workers} worker(s), memory budget "
                 f"{self.ctx.mem_gb:.0f} GB")
        try:
            while True:
                started = False
                while len(self.running) < self.workers:
                    job = next((j for j in self._dispatchable() if self._fits(j)), None)
                    if job is None:
                        break
                    self._start(job)
                    started = True
                if not self.running:
                    if not started:
                        break
                    continue
                self._reap()
        except BaseException:
            for run in self.running.values():
                run.proc.terminate()
            raise
        finally:
            _SHARED.clear()
            self.composed.clear()
        for j in self.todo:
            st, _ = self.res.status(j)
            self.skipped.setdefault(st if st.startswith("blocked") else f"not run ({st})", []).append(j.id)
        return self.results


def update_selection(res: Resolver, log: Log) -> dict | None:
    """Recompute the per-arm DEV selection from the finished tuning runs; write it when it changed."""
    ctx = res.ctx
    cells = R.tuning_cells()
    arms, digests = {}, {}
    for arm in R.ARMS:
        per_cell, ok, dig = {}, True, []
        for cell in cells:
            vals = {}
            for s in ctx.tune_seeds:
                job = Job("tune", R.DEFAULT, arm, int(s), cell.id)
                st, spec = res.status(job)
                if st != "done":
                    ok = False
                    break
                rec = R.read_done(res.marker(job))
                vals[str(s)] = float(rec["result"]["selection_welfare"])
                dig.append(rec["spec_digest"])
            if not ok:
                break
            per_cell[cell.id] = dict(params=cell.as_dict(), mean=float(np.mean(list(vals.values()))),
                                     sd=float(np.std(list(vals.values()), ddof=1)) if len(vals) > 1 else 0.0,
                                     per_seed=vals)
        if not ok:
            continue
        best = cells[0]
        for c in cells[1:]:
            if per_cell[c.id]["mean"] > per_cell[best.id]["mean"]:
                best = c
        ranked = sorted(per_cell.values(), key=lambda v: -v["mean"])
        arms[arm] = dict(cell=best.id, params=best.as_dict(), mean_selection_welfare=per_cell[best.id]["mean"],
                         margin_over_second=(ranked[0]["mean"] - ranked[1]["mean"]) if len(ranked) > 1 else None,
                         per_cell=per_cell)
        digests[arm] = sorted(dig)
    path = R.selection_path(root=ctx.models_root)
    old = R.read_done(path)
    if not arms and old is None:
        return None
    # arms whose tuning is incomplete or stale are dropped, so their main jobs block instead of using an old cell
    body = dict(criterion=("per arm: highest mean over the tuning seeds of the checkpoint-selected realized welfare "
                           "(credits per scored interval, truthful reports) on the selection windows; ties keep the "
                           "earlier grid cell (order lr, sigma0, M)"),
                split=ctx.split, tune_seeds=list(ctx.tune_seeds), grid=R.TUNING_GRID, lr_end_ratio=R.LR_END_RATIO,
                config=R.DEFAULT, reused_for=[c for c in R.CONFIGS if c != R.DEFAULT], arms=arms,
                tune_spec_digests=digests, decisions=R.DECISIONS)
    digest = R.spec_digest(body)
    if old is None or old.get("spec_digest") != digest:
        sio.write_json(path, dict(status="done", spec_digest=digest,
                                  created=_dt.datetime.now().isoformat(timespec="seconds"), **body))
        log("selection written: " + (", ".join(f"{a}: {v['cell']} ({v['mean_selection_welfare']:.4f})"
                                               for a, v in arms.items()) or "no arm has complete tuning"))
    res.reload_selection()
    for a, v in arms.items():
        _SELECTED_M[a] = int(v["params"]["M"])
    return res.selection


def run_copy(job: Job, res: Resolver, log: Log) -> dict:
    ctx = res.ctx
    st, spec = res.status(job)
    if spec is None:
        return dict(id=job.id, status=st)
    cell = spec["cell"]
    src = R.tune_dir(job.arm, cell, job.seed, root=ctx.models_root)
    dst = R.model_dir(job.config, job.arm, job.seed, root=ctx.models_root)
    rec = R.read_done(src / R.CONFIG_FILE)
    dst.mkdir(parents=True, exist_ok=True)
    for name in (R.MODEL_FILE, R.HISTORY_FILE):
        tmp = dst / (name + ".tmp")
        shutil.copyfile(src / name, tmp)
        os.replace(tmp, dst / name)
    if sio.file_sha256(dst / R.MODEL_FILE) != rec["result"]["model_sha256"]:
        raise RuntimeError(f"copy of {src / R.MODEL_FILE} does not match its recorded SHA-256")
    new = dict(rec)
    new.update(job=job.id, spec=spec, spec_digest=R.spec_digest(spec), created=_dt.datetime.now().isoformat(
        timespec="seconds"), copied_from=dict(job=spec["source"], dir=str(src), spec=rec["spec"],
                                              spec_digest=rec["spec_digest"]),
        hyperparameter_source="DEV selection on the default configuration (results/models/selection.json)")
    sio.write_json(dst / R.CONFIG_FILE, new)
    log(f"copied {spec['source']} -> {dst}")
    return dict(id=job.id, status="done")


def write_index(res: Resolver, jobs: list[Job]) -> pathlib.Path:
    import pandas as pd
    rows = []
    for j in jobs:
        if j.kind == "calib":
            continue
        rec = R.read_done(res.marker(j))
        if rec is None:
            continue
        row = dict(job=j.id, kind=j.kind, config=j.config, arm=j.arm, seed=j.seed,
                   cell=(rec.get("hyperparameters") or {}).get("id") if j.kind != "static" else None,
                   path=str(res.marker(j).parent.relative_to(R.ROOT)) if res.marker(j).is_relative_to(R.ROOT)
                   else str(res.marker(j).parent), created=rec.get("created"))
        if j.kind == "static":
            row.update(selection_welfare=rec["welfare_selection"], train_welfare=rec["welfare_train"],
                       w=" ".join(f"{x:.4f}" for x in rec["w"]))
        else:
            r = rec["result"]
            row.update(selection_welfare=r["selection_welfare"], train_welfare=r["train_welfare"],
                       best_iteration=r["best_iteration"], sigma=r["sigma"], param_digest=r["param_digest"],
                       train_seconds=r["train_seconds"])
        rows.append(row)
    path = pathlib.Path(res.ctx.models_root) / R.INDEX_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.10g")
    return path


def check_split(split: str, smoke: bool) -> None:
    """Training split rule: 'dev' (selection on DEV); 'train' only for --smoke; 'test' never.

    'test' first goes through the freeze gate, so before the freeze the error is the gate's PermissionError (TEST is
    locked); after the freeze it is refused anyway, because training and selection never see TEST."""
    if split == "test":
        R.require_split("test")          # PermissionError while the pre-registration freeze is absent
        raise PermissionError("training and checkpoint/hyper-parameter selection never use TEST, even after the "
                              "freeze; TEST evaluation belongs to the experiment drivers")
    R.require_split(split)
    if split == "train" and not smoke:
        raise ValueError("selection on TRAIN (--split train) is only for --smoke; full runs select on DEV")


def run(ctx: Ctx, filters: dict | None = None, *, force: bool = False, redo_stale: bool = False) -> int:
    """Run every unfinished job that matches ``filters``; returns the number of selected jobs not done."""
    check_split(ctx.split, ctx.smoke)
    sio.require_free_gb(3)
    ctx = dataclasses.replace(ctx, code_sha=ctx.code_sha or R.package_sha256())
    log = Log(ctx)
    res = Resolver(ctx)
    jobs = [j for j in build_jobs(ctx) if matches(j, filters or {})]
    log(f"train.py: {len(jobs)} jobs selected; split={ctx.split} smoke={ctx.smoke} workers={ctx.workers} "
        f"mem_gb={ctx.mem_gb} tune_seeds={list(ctx.tune_seeds)} models={ctx.models_root}")
    sched = Scheduler(res, log, jobs, force=force, redo_stale=redo_stale)
    results = sched.run()
    idx = write_index(res, build_jobs(ctx))
    final = {j.id: res.status(j)[0] for j in jobs}
    unfinished = [k for k, v in final.items() if v != "done"]
    failed = [r for r in results if r["status"] != "done"]
    log(f"index: {idx}")
    for k, v in sched.skipped.items():
        log(f"{len(v)} job(s) {k}: {', '.join(v[:6])}{' ...' if len(v) > 6 else ''}")
    log(f"finished: {len(jobs) - len(unfinished)}/{len(jobs)} selected jobs done, {len(failed)} failed this run")
    return len(unfinished)


def print_plan(ctx: Ctx, filters: dict | None = None) -> dict:
    """Print the job list with status and estimated cost; returns the totals (core-seconds and wall seconds)."""
    res = Resolver(ctx)
    for a, v in (res.selection or {"arms": {}})["arms"].items():
        _SELECTED_M[a] = int(v["params"]["M"])
    jobs = [j for j in build_jobs(ctx) if matches(j, filters or {})]
    tot = dict(lo=0.0, hi=0.0, todo_lo=0.0, todo_hi=0.0)
    counts: dict = {}
    todo = []
    print(f"{'job':58s} {'status':30s} {'n':>2s} {'est. core-min':>14s}")
    for j in jobs:
        st, _ = res.status(j)
        if j.kind == "main" and j.arm not in _SELECTED_M:
            lo, hi = est_seconds(j, ctx, M=4), est_seconds(j, ctx, M=8)
        else:
            lo = hi = est_seconds(j, ctx)
        lo, hi = lo * CONTENTION, hi * CONTENTION
        cost = f"{lo / 60:.1f}" if lo == hi else f"{lo / 60:.1f}-{hi / 60:.1f}"
        print(f"{j.id:58s} {st:30s} {j.cfg.n:2d} {cost:>14s}")
        tot["lo"] += lo
        tot["hi"] += hi
        if st != "done":
            tot["todo_lo"] += lo
            tot["todo_hi"] += hi
            todo.append(j)
        counts[(j.kind, st.split(":")[0])] = counts.get((j.kind, st.split(":")[0]), 0) + 1
    print()
    for (k, st), c in sorted(counts.items()):
        print(f"  {k:7s} {st:8s} {c:4d}")
    by_kind: dict = {}
    for j in todo:
        by_kind[j.kind] = by_kind.get(j.kind, 0.0) + est_seconds(j, ctx) * CONTENTION
    span = f"{tot['lo'] / 3600:.1f}" + ("" if tot["lo"] == tot["hi"] else f"-{tot['hi'] / 3600:.1f}")
    print(f"\nestimated core-hours (single-process cost model x{CONTENTION} for a full, otherwise idle machine): "
          f"all selected jobs {span}; not yet done {tot['todo_lo'] / 3600:.1f}-{tot['todo_hi'] / 3600:.1f}")
    print("  not yet done, by kind: " + ", ".join(f"{k} {v / 3600:.1f}" for k, v in sorted(by_kind.items())))
    wall = simulate(todo, ctx, ctx.workers)
    tot["wall"] = wall
    keys = sorted({j.data_key for j in todo if j.data_key})
    peak = BASE_GB + sum(shared_gb(k, ctx) for k in keys)
    g3, g9 = job_gb(Job("main", "default", "protected", 0)), job_gb(Job("main", "n9", "protected", 0))
    print(f"wall-clock (list-scheduling simulation, {ctx.workers} workers): {wall / 3600:.1f} h, plus window "
          f"composition (~30-60 s per window set); memory: {len(keys)} window set(s) {peak:.1f} GB if all are "
          f"composed at once, + {g3:.2f}-{g9:.2f} GB per running job; the scheduler keeps the total under --mem-gb "
          f"{ctx.mem_gb:.0f}")
    return tot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=R.SPLITS,
                    help="selection split: dev (default); train only with --smoke; test is refused")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 2 TRAIN windows, 2 PPO iterations, 1 seed")
    ap.add_argument("--workers", type=int, default=22, help="worker processes (1: run in this process)")
    ap.add_argument("--mem-gb", type=float, default=20.0, help="memory budget for shared windows + workers (GB)")
    ap.add_argument("--plan", action="store_true", help="print the job list and estimated core-hours; run nothing")
    ap.add_argument("--only", action="append", default=[], metavar="KEY=V1,V2",
                    help=f"filter jobs (repeatable, AND-ed); keys {', '.join(FILTER_KEYS)}")
    ap.add_argument("--tune-seeds", type=int, default=None,
                    help="number of tuning seeds (first k main seeds; default all 5 main seeds)")
    ap.add_argument("--redo-stale", action="store_true", help="retrain jobs whose finished spec differs")
    ap.add_argument("--force", action="store_true", help="retrain finished jobs that match the filters")
    ap.add_argument("--root", default=None,
                    help="relocate models, logs and sweep calibrations below this directory (inside the project; "
                         "tests)")
    a = ap.parse_args(argv)
    try:
        check_split(a.split, a.smoke)
    except (PermissionError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    tune = None if a.tune_seeds is None else R.SEEDS_MAIN[:max(1, a.tune_seeds)]
    ctx = make_ctx(a.split, a.smoke, workers=a.workers, mem_gb=a.mem_gb, tune_seeds=tune, root=a.root)
    filters = parse_filters(a.only)
    if a.plan:
        print_plan(ctx, filters)
        return 0
    return 1 if run(ctx, filters, force=a.force, redo_stale=a.redo_stale) else 0


if __name__ == "__main__":
    sys.exit(main())
