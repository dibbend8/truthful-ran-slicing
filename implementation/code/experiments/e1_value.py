"""E1 - value of reports and of learning under truthful reports (experiment_design.md §7 E1; prereg H1a-H1d).

Run from ``code/`` (PYTHONDONTWRITEBYTECODE=1, anaconda python):

    python experiments/e1_value.py --split dev [--config default] [--workers 20]
    python experiments/e1_value.py --split dev --allow-missing      # arms without models are skipped
    python experiments/e1_value.py --smoke                           # 2 short TRAIN windows (see --smoke below)
    python experiments/e1_value.py --split test                      # refused until prereg/PREREG.md is frozen

For one split and one configuration of ``registry.CONFIGS`` every window of every date of the split is played by
every arm with truthful tenants. All arms of a window use the same composed window object (same per-ms arrivals,
CQI run, reports; paired by construction); the episode itself has no randomness and the oracle's CEM stream is
seeded by the window id.

Arms (keys of ``spec/controllers.yaml`` where one exists, so figures take labels and colours from there; the CLI
also accepts ``unconstrained`` for ``unprotected`` and ``direct_true_rates`` for ``direct_truerates``):

=================  ============================================================  ===========================
arm                controller                                                    billed price
=================  ============================================================  ===========================
oracle             lookahead hindsight oracle (``control.oracle``, pop 64, 8      none
                   iterations, h = 3, base policy = the static weights); a
                   rollout reference, not an upper bound (D3)
protected          trained ``ProtectedNet`` (kappa >= 0), every seed              full-decision price
kappa0             trained ``Kappa0Net`` (kappa = 0 ablation), every seed         full-decision price
meas_only          trained ``MeasOnlyNet`` (allocates on the operator forecast;   usage tariff nu Q when nu
                   reports never enter), every seed                               is given (``--nu-file`` /
                                                                                  ``--fit-nu``), else none
static             TRAIN-fitted static weights (``control.static.fit_static`` on  full-decision price
                   all 180 TRAIN windows of the configuration)
equal              equal weights                                                  full-decision price
direct             direct declared-welfare optimiser, stale rate                  full-decision formula
unprotected        trained ``UnconstrainedNet`` (CallableMap, not monotone),      fixed-priority price
                   every seed                                                     (+ the grid envelope p with
                                                                                  ``--unconstrained-envelope``)
direct_truerates   direct optimiser with the true current rates (diagnostic)      full-decision formula
=================  ============================================================  ===========================

Every run also records ``p`` (full-decision envelope; certified for monotone maps), ``p_fix`` (fixed-priority
price), ``T = p - p_fix`` and, when nu is known, ``p_usage`` - whatever the arm bills.

Inputs (``experiments/registry.py``, the layout ``experiments/train.py`` writes). Configurations: ``default, d1, d5,
d15, rho0.5, rho0.7, rho1.1, lead0, lead1.05, n6, n9``; each composes its windows with its own TRAIN calibration
(``n3_default``; ``n3_rho<rho>`` / ``n3_lead<lead>`` for the load and lead cells, read from ``--calib-root`` or
``data/calibration``; ``n6_default``, ``n9_default``). Learned arms: every finished model
``<models-root>/<config>/<arm>/seed<k>/model.pt`` (arm ``unconstrained`` for ``unprotected``), loaded with
``control.nets.load_net`` and checked against its ``config.json`` record (file SHA-256, parameter digest) and against
the configuration's d (operator forecaster horizon). Missing learned arms are an error unless ``--allow-missing``.
Static weights: ``registry.load_static`` (the d cells share the default's fit); without it (not on TEST) a fit with
the training driver's settings is made and cached in ``<out>/static_fit.json``. Usage tariff nu (credits per PRB per
interval, revenue-neutral on TRAIN): ``--nu-file``, or the TRAIN fit (protected seed 0 on all TRAIN windows of the
configuration), read from / with ``--fit-nu`` written to ``tariff_file``: for ``default`` the E2-E4 cache
(``e2_manipulation.usage_tariff``, ``results/raw/train/e2/usage_tariff/``), for every sweep configuration
``registry.sweep_tariff_path`` (``results/raw/train/e1/<config>/usage_nu.json``; pre-freeze audit E5-7/F8), smoke
runs ``<out>/usage_nu.json``; without it meas_only is billed 'none' and nu x quota can be applied later. On TEST
every model, the static fit and the usage-tariff file must be listed with their SHA-256 in the freeze block of the
frozen pre-registration, and the tariff is loaded, never refitted.

``--smoke``: the two TRAIN windows of ``registry.SMOKE_TRAIN_WINDOWS`` cut to K = 60 s (20 s warm-up), oracle with
population 16 and 2 iterations, the training driver's smoke models (``results/raw/smoke_train/<split>/models``)
when they exist, else 2-iteration PPO nets trained here (seeds 0 and 1, h = 3) under ``<out>/ckpt``.

Outputs (``results/raw/<split>/e1/<config>/``; ``results/raw/smoke/e1/<config>/`` with ``--smoke``):

* ``windows/<date>_o<offset>.parquet``: one row per (arm, seed, tenant) with tenant = -1 the per-interval sum over
  tenants (``eval.metrics.window_summary`` over the scored intervals: on-time / dropped Mbit per interval,
  deadline-miss %, quota PRB, billed price, declared and realized value and utilities, credits per interval), plus
  the means of p, p_fix, T, p_usage, the share of tenant-intervals with a certified positive premium, payment as %
  of declared value, idle PRB, stage shares, the invariant checks of ``sim.episode.episode_checks``, the window's
  ``offered_load`` (D2 quantity: sum_i mean_k m_ik / (B r_bar_TRAIN), all K intervals) and ``capacity_load``
  (``eval.headroom.window_load``), the run fingerprint ``fp``, the code fingerprint ``code_fp``
  (:func:`code_fingerprint`), the output ``schema`` (:data:`SCHEMA`) and wall seconds. Realized welfare of a run is
  the ``realized_value`` of its tenant = -1 row. PRB efficiency (G6 inputs; pre-freeze audit HC-4/NUM-06), means
  per scored interval (tenant = -1: of the per-interval sums over tenants): ``cap_Mbit`` = quota x r_true x Delta
  (Mbit the held PRBs could carry at the true per-PRB rate), ``unused_Mbit`` (capacity that found an empty queue),
  ``prb_busy`` = on_time / (r_true x Delta) (PRB-equivalents spent on on-time traffic) and, per tenant, ``r_true``
  (Mbit/s per PRB). With deadline expiry on_time + unused = cap (``sim.fluid``).
* ``intervals/<date>_o<offset>.parquet`` (``--intervals auto``: only for the default configuration and smoke runs,
  about 32 MB per split at n = 3; ``on``/``off`` force it): scored (arm, seed, k, tenant) rows with quota, w, p,
  p_fix, price, V_declared, on-time, dropped and unused Mbit, backlog and the true per-PRB rate r_true (float32).
* merged ``windows.parquet``; ``manifest.json`` (configuration, models and hashes, static weights, nu, oracle
  settings, code hash, timings, disk use) with a **quicklook**: per-arm mean welfare and paired differences with a
  date-cluster bootstrap and offered-load terciles (descriptive; not the pre-registered analysis).

Resumable: the unit of work is a window; its file lists the run fingerprints already done (arm, seed, model
SHA-256, static weights, nu, settings, output schema and code fingerprint), so a rerun computes only missing or stale
(arm, seed) runs and keeps rows of arms that are not part of the current invocation. A code change therefore makes
every run stale once (pre-freeze audit F7/PV-10). ``--force`` recomputes every planned run. A window file whose rows
are replaced is first copied to ``results/raw/<split>/archive/<UTC stamp>/e1/<config>/...``
(``e0_setup.archive_files``; one stamp per invocation, listed in ``manifest.json``), so earlier numbers are never
lost (PREREG §8).

TEST: besides the window, arm, model and tariff overrides, ``--intervals`` and ``--unconstrained-envelope`` are
refused, and the usage tariff is always required (``fit_nu_required``): it is loaded from its frozen TRAIN file
(``tariff_file``) and checked against the freeze block, never refitted.

Orchestrator decisions (binding, 2026-10-19; code/STATUS.md §0), as used here:

* **D1** GBR fractions x 0.8 (0.40 / 0.64 / 0.16 of mu): in the calibration's contract; TRAIN-only decision.
* **D2** per-date offered-load normalization on every split: windows are composed with the date's counts
  (``Calibration.counts_for``; rho sweeps use the nested counts at the cell's rho); realized per-window loads
  ``offered_load`` and ``capacity_load`` are written on every row.
* **D3** gate G4 dropped (the oracle is a reference, not an upper bound); H1c/H1d secondary, reported with CIs and
  by offered-load subgroup: the exogenous subgroup variable is ``offered_load`` (quicklook: terciles).
* **D4** harm is measured on honest tenants' own outcomes: E1 has no deviator; per-tenant on-time Mbit, deadline
  miss and realized value are stored so E2/E4 can pair them with deviation runs; total welfare is descriptive there.
* **D5** E5 protocol: not used by E1.
* **D6** learner objective = lookahead return, h = 3 (``PPOConfig.horizon=3``): a registry model whose record names
  another PPO horizon is refused (``e0_setup.load_checked_net``), standalone smoke nets are trained with h = 3, and
  the oracle uses ``horizon=3``.
* **D7** E4b kappa-scaling frontier: not part of E1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

import e0_setup as S  # noqa: E402
import registry as R  # noqa: E402
from slicemech import io as sio  # noqa: E402

ARMS = ("oracle", "protected", "kappa0", "meas_only", "static", "equal", "direct", "unprotected",
        "direct_truerates")
ALIASES = {"unconstrained": "unprotected", "direct_true_rates": "direct_truerates", "kappa_0": "kappa0",
           "measurement_only": "meas_only"}
#: Learned arms and the net kind of their checkpoints.
LEARNED = {"protected": "protected", "kappa0": "kappa0", "meas_only": "meas_only", "unprotected": "unconstrained"}
ORACLE = dict(pop=64, iters=8, seed=0, horizon=S.PPO_HORIZON)
SMOKE_ORACLE = dict(pop=16, iters=2, seed=0, horizon=S.PPO_HORIZON)
SMOKE_SEEDS = (0, 1)
T_POS = 0.0           # a premium counts as positive when its certified lower end T_lo exceeds this (credits)
INTERVAL_COLS = ("k", "tenant", "quota", "w", "p", "p_fix", "price", "V_declared", "on_time_Mbit", "dropped_Mbit",
                 "backlog_Mbit", "unused_Mbit", "r_true")
#: Version of E1's output schema. It is part of every run fingerprint, so a resumed run recomputes rows written
#: under an older schema. 2 (2026-10-20, pre-freeze audit HC-4/NUM-06): interval columns ``unused_Mbit`` and
#: ``r_true``; window means ``cap_Mbit``, ``unused_Mbit``, ``prb_busy`` and ``r_true``; the ``code_fp`` column.
SCHEMA = 2
#: Driver modules whose source enters E1's code fingerprint, next to the ``slicemech`` package (audit F7/PV-10).
CODE_MODULES = ("e1_value.py", "e0_setup.py", "registry.py")


def resolve_arms(spec: str | None) -> tuple[str, ...]:
    if not spec:
        return ARMS
    out = []
    for a in spec.split(","):
        a = ALIASES.get(a.strip(), a.strip())
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {', '.join(ARMS)} (aliases {ALIASES})")
        if a not in out:
            out.append(a)
    return tuple(a for a in ARMS if a in out)


# ============================================================================================ plan
def _digest(obj) -> str:
    return sio.digest(obj)[:16]


def code_fingerprint() -> str:
    """SHA-256 over the ``slicemech`` package and the driver modules :data:`CODE_MODULES` (audit F7/PV-10: after a
    logged code fix every E1 run is stale once, so rows of two code versions never share one window file)."""
    return sio.digest(dict(slicemech=R.package_sha256(),
                           modules={m: sio.file_sha256(_HERE / m) for m in CODE_MODULES}))


def planned_runs(plan: dict) -> list[dict]:
    """(arm, seed, fp) of every run of a window; ``fp`` digests every input that changes the run's numbers: the
    configuration, calibration, smoke flag, nu, the output schema (:data:`SCHEMA`), the code fingerprint
    (``plan["code_fp"]``, :func:`code_fingerprint`) and the arm's own inputs."""
    runs = []
    # nu enters every run (p_usage column; the meas_only bill), the configuration enters every composition
    common = dict(config=plan["config"], cfg=plan["cfg"], cal_sha=plan["cal_sha"], smoke=plan["smoke"],
                  nu=plan["nu"], schema=SCHEMA, code=plan.get("code_fp"))
    for arm in plan["arms"]:
        extra = {}
        if arm == "unprotected":
            extra["envelope"] = plan["envelope"]
        if arm in ("static", "oracle"):
            extra["static_w"] = plan["static_w"]
        if arm == "oracle":
            extra["oracle"] = plan["oracle"]
        if arm in LEARNED:
            for seed, (path, sha) in sorted(plan["ckpts"].get(arm, {}).items()):
                runs.append(dict(arm=arm, seed=int(seed), ckpt_sha=sha,
                                 fp=_digest(dict(arm=arm, seed=int(seed), sha=sha, **extra, **common))))
        else:
            runs.append(dict(arm=arm, seed=-1, ckpt_sha="", fp=_digest(dict(arm=arm, **extra, **common))))
    return runs


_CTX: dict = {}


def _context(plan: dict) -> dict:
    """Per-process cache: calibration, mechanism, configuration, loaded nets (by arm, seed)."""
    key = plan["key"]
    if key in _CTX:
        return _CTX[key]
    import torch
    torch.set_num_threads(1)
    cfg = S.get_config(plan["config"])
    cal = S.load_cal(cfg, plan["calib_root"])
    nets = {}
    for arm, seeds in plan["ckpts"].items():
        for seed, (path, sha) in seeds.items():
            if sio.file_sha256(path) != sha:
                raise RuntimeError(f"checkpoint {path} changed during the run")
            nets[(arm, int(seed))] = S.load_checked_net(path, cfg.d)
    _CTX.clear()
    _CTX[key] = dict(cal=cal, mech=R.mechanism(cal), cfg=cfg, nets=nets,
                     static_w=np.asarray(plan["static_w"], float),
                     nu=None if plan["nu"] is None else np.asarray(plan["nu"], float))
    return _CTX[key]


# ============================================================================================ one run
def arm_controller(arm: str, seed: int, ctx: dict, plan: dict, window):
    """(controller, PriceRule, price_columns, extra dict) of one arm on one window."""
    from slicemech.control.oracle import oracle_window
    from slicemech.sim import episode as E
    nu = ctx["nu"]
    if arm == "oracle":
        o = plan["oracle"]
        res = oracle_window(window, ctx["mech"], pop=o["pop"], iters=o["iters"], seed=o["seed"],
                            elite=o.get("elite", min(8, o["pop"] // 2)), horizon=o["horizon"],
                            base_weights=ctx["static_w"])
        return res.controller(), E.PriceRule("none", nu=nu), "billed", dict(oracle_W=res.mean_welfare,
                                                                              oracle_evals=float(res.evaluations.mean()))
    if arm == "equal":
        return E.FixedWeights(), E.PriceRule("full", nu=nu), "all", {}
    if arm == "static":
        return E.FixedWeights(ctx["static_w"], name="static"), E.PriceRule("full", nu=nu), "all", {}
    if arm == "direct":
        return E.DirectController("hat"), E.PriceRule("full", nu=nu), "all", {}
    if arm == "direct_truerates":
        return E.DirectController("true"), E.PriceRule("full", nu=nu), "all", {}
    net = ctx["nets"][(arm, int(seed))]
    if arm in ("protected", "kappa0"):
        return net, E.PriceRule("full", nu=nu), "all", {}
    if arm == "meas_only":
        pr = E.PriceRule("usage", nu=nu) if nu is not None else E.PriceRule("none")
        return net, pr, "billed", {}
    if arm == "unprotected":
        return net, E.PriceRule("fixed", nu=nu, nonmonotone_full=bool(plan["envelope"])), "all", {}
    raise ValueError(arm)


def prb_columns(sc: pd.DataFrame, Delta: float) -> pd.DataFrame:
    """PRB-efficiency inputs of the G6 table (audit HC-4/NUM-06), per (interval, tenant) row of an episode table:
    ``cap_Mbit = quota * r_true * Delta`` (capacity of the held PRBs at the true per-PRB rate, Mbit per interval)
    and ``prb_busy = on_time_Mbit / (r_true * Delta)`` (PRB-equivalents spent on on-time traffic; 0 where
    ``r_true = 0``). With the fluid model's deadline expiry (no late service) ``on_time + unused = cap`` up to
    rounding (``sim.fluid``)."""
    r = sc["r_true"].to_numpy(float)
    cap = sc["quota"].to_numpy(float) * r * float(Delta)
    rd = r * float(Delta)
    busy = np.divide(sc["on_time_Mbit"].to_numpy(float), rd, out=np.zeros(len(sc)), where=rd > 0)
    return sc.assign(cap_Mbit=cap, prb_busy=busy)


def summarize_run(df: pd.DataFrame, window, mech, keys: dict) -> pd.DataFrame:
    """Per-tenant and all-tenant (tenant = -1) window summary of one episode table (scored intervals).

    Besides ``eval.metrics.window_summary`` and the price columns, every row carries the means per scored interval
    of ``cap_Mbit``, ``unused_Mbit`` and ``prb_busy`` (:func:`prb_columns`; tenant = -1: of the per-interval sums
    over tenants) and, per tenant, the mean true per-PRB rate ``r_true`` (Mbit/s per PRB; NaN for tenant = -1)."""
    from slicemech.eval.metrics import window_summary
    from slicemech.sim.episode import episode_checks
    warm = int(window.warmup)
    ws = window_summary(df, warm)
    sc = prb_columns(df[df["scored"]], float(mech.Delta))
    nk = sc["k"].nunique()
    t_lo = sc["T_lo"].to_numpy(float)
    sc = sc.assign(T_pos=np.where(np.isfinite(t_lo), (t_lo > T_POS).astype(float), np.nan))
    per_t = sc.groupby("tenant").agg(p=("p", "mean"), p_fix=("p_fix", "mean"), T=("T", "mean"),
                                     p_usage=("p_usage", "mean"), frac_T_pos=("T_pos", "mean"), w=("w", "mean"),
                                     offered_mbps=("arrivals_Mbit", "mean"), theta=("theta_adm", "mean"),
                                     backlog_Mbit=("backlog_Mbit", "mean"), r_alloc=("r_alloc", "mean"),
                                     cap_Mbit=("cap_Mbit", "mean"), unused_Mbit=("unused_Mbit", "mean"),
                                     prb_busy=("prb_busy", "mean"), r_true=("r_true", "mean"))
    per_t["offered_mbps"] /= float(mech.Delta)
    sums = sc.groupby("k")[["p", "p_fix", "T", "p_usage", "arrivals_Mbit", "theta_adm", "backlog_Mbit", "cap_Mbit",
                            "unused_Mbit", "prb_busy"]].sum(min_count=1).mean()
    all_row = dict(p=sums["p"], p_fix=sums["p_fix"], T=sums["T"], p_usage=sums["p_usage"],
                   frac_T_pos=float(sc["T_pos"].mean()), w=np.nan,
                   offered_mbps=sums["arrivals_Mbit"] / float(mech.Delta), theta=sums["theta_adm"],
                   backlog_Mbit=sums["backlog_Mbit"], r_alloc=np.nan, cap_Mbit=sums["cap_Mbit"],
                   unused_Mbit=sums["unused_Mbit"], prb_busy=sums["prb_busy"], r_true=np.nan)
    per_t.loc[-1] = all_row
    per_t.index.name = None
    per_t = per_t.rename_axis("tenant").reset_index()
    per_t["tenant"] = per_t["tenant"].astype(int)
    ws = ws.merge(per_t, on="tenant", how="left", validate="one_to_one")
    ws["payment_pct_of_V"] = np.where(ws["V_declared"] > 0, 100.0 * ws["price"] / ws["V_declared"], np.nan)
    k0 = sc[sc["tenant"] == 0]
    chk = episode_checks(df, window.contract, mech)
    extra_all = dict(idle_prb=float(mech.B - sc.groupby("k")["quota"].sum().mean()),
                     share_overload=float(k0["overload"].mean()), share_leftover=float(k0["leftover"].mean()),
                     share_need=float(k0["need"].mean()), scored_intervals=int(nk),
                     **{f"chk_{k}": float(v) for k, v in chk.items() if not k.startswith("br_")})
    for c, v in extra_all.items():
        ws[c] = np.where(ws["tenant"] == -1, v, np.nan)
    for j, (c, v) in enumerate(keys.items()):
        ws.insert(j, c, v)
    return ws


def interval_rows(df: pd.DataFrame, keys: dict) -> pd.DataFrame:
    sc = df.loc[df["scored"], list(INTERVAL_COLS)].copy()
    sc["k"] = sc["k"].astype(np.int16)
    sc["tenant"] = sc["tenant"].astype(np.int8)
    sc["quota"] = sc["quota"].astype(np.int16)
    sc = S.f32(sc)
    for j, (c, v) in enumerate(keys.items()):
        sc.insert(j, c, v)
    sc["seed"] = sc["seed"].astype(np.int8)
    return sc


def run_one(window, run: dict, ctx: dict, plan: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Play one (arm, seed) on a window with truthful tenants; return (summary rows, interval rows)."""
    from slicemech.eval.headroom import window_load
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    t0 = time.perf_counter()
    arm, seed = run["arm"], int(run["seed"])
    ctrl, pr, pc, extra = arm_controller(arm, seed, ctx, plan, window)
    mech, cal, cfg = ctx["mech"], ctx["cal"], ctx["cfg"]
    df = E.run_window(window, ctrl, pr, [Truthful() for _ in range(window.n)], cfg.d, mech, cal.shape,
                      price_columns=pc)
    m = np.asarray(window.measured, float)
    keys = dict(split=window.split, config=plan["config"], date=window.date, offset_s=int(window.meta["offset_s"]),
                window_id=window.id, arm=arm, seed=seed, fp=run["fp"], ckpt_sha=run["ckpt_sha"],
                code_fp=plan.get("code_fp"), schema=SCHEMA,
                price_rule=pr.charge, offered_load=float(m.mean(0).sum() / (mech.B * float(cal.r_bar))),
                capacity_load=window_load(window, mech), tod_start_s=float(window.meta["tod_start_s"]))
    summ = summarize_run(df, window, mech, keys)
    W = float(summ.loc[summ["tenant"] == -1, "realized_value"].iloc[0])
    if "oracle_W" in extra:                          # the replayed oracle path must reproduce the oracle's welfare
        extra["oracle_selfcheck"] = abs(W - extra["oracle_W"]) / max(1.0, abs(W))
    for c, v in extra.items():
        summ[c] = np.where(summ["tenant"] == -1, v, np.nan)
    summ["seconds"] = time.perf_counter() - t0
    summ["episode_seconds"] = float(df.attrs.get("seconds", np.nan))
    ivl = interval_rows(df, dict(arm=arm, seed=seed)) if plan["intervals"] else None
    return summ, ivl


# ============================================================================================ one window
def _merge(old: pd.DataFrame | None, new: list, replaced: set) -> pd.DataFrame:
    keep = []
    if old is not None and len(old):
        mask = [(a, int(s)) not in replaced for a, s in zip(old["arm"], old["seed"])]
        keep.append(old[np.asarray(mask, bool)])
    frames = keep + [x for x in new if x is not None and len(x)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def window_job(args) -> dict:
    """All planned (arm, seed) runs of one window that are not done yet (module docstring, "Resumable"; with
    ``plan["force"]`` every planned run). Before a window file is rewritten with rows that replace earlier rows
    (stale or forced runs), the old file is copied to the archive (``e0_setup.archive_files``), so the earlier
    numbers are kept (PREREG §8)."""
    plan, date, off = args
    t0 = time.perf_counter()
    out = Path(plan["out"])
    key = S.key_of(date, off)
    sfile, ifile = out / "windows" / f"{key}.parquet", out / "intervals" / f"{key}.parquet"
    runs = planned_runs(plan)
    old_s = pd.read_parquet(sfile) if sfile.exists() else None
    old_i = pd.read_parquet(ifile) if (ifile.exists() and plan["intervals"]) else None
    done = set(old_s["fp"]) if old_s is not None else set()
    if plan["intervals"]:                  # a run without its interval rows (stored earlier with intervals off)
        have = set() if old_i is None else {(a, int(s)) for a, s in old_i[["arm", "seed"]].drop_duplicates()
                                            .itertuples(index=False)}
        done = {r["fp"] for r in runs if r["fp"] in done and (r["arm"], int(r["seed"])) in have}
    if plan.get("force"):
        done = set()
    todo = [r for r in runs if r["fp"] not in done]
    if not todo:
        return dict(key=key, ran=0, planned=len(runs), seconds=time.perf_counter() - t0, archived=None)
    ctx = _context(plan)
    w = S.compose(ctx["cal"], plan["split"], date, off, smoke=plan["smoke"], test_freeze=plan["test_freeze"])
    stamp, archived = plan.get("archive_stamp"), None
    if old_s is not None and len(old_s) and set(old_s["window_id"]) != {w.id}:
        archived = S.archive_files([sfile, ifile], out, stamp=stamp, copy=True,
                                   reason="E1 window composition changed: every run redone")
        old_s = old_i = None                                   # the window composition changed: redo everything
        todo = runs
        if not plan["intervals"] and ifile.exists():
            ifile.unlink()                                     # archived above: interval rows of the old composition
    new_s, new_i = [], []
    for r in todo:
        s, iv = run_one(w, r, ctx, plan)
        new_s.append(s)
        new_i.append(iv)
    replaced = {(r["arm"], int(r["seed"])) for r in todo}
    if archived is None:                      # rows of earlier runs that are replaced now: keep the old files
        old_runs = set() if old_s is None else {(a, int(s)) for a, s in zip(old_s["arm"], old_s["seed"])}
        if old_i is not None:
            old_runs |= {(a, int(s)) for a, s in zip(old_i["arm"], old_i["seed"])}
        if replaced & old_runs:
            archived = S.archive_files([sfile, ifile] if plan["intervals"] else [sfile], out, stamp=stamp, copy=True,
                                       reason="E1 rows replaced (stale fingerprint or --force)")
    if plan["intervals"]:
        S.write_parquet(_merge(old_i, new_i, replaced), ifile)
    S.write_parquet(_merge(old_s, new_s, replaced), sfile)   # written last: the completion marker
    return dict(key=key, ran=len(todo), planned=len(runs), seconds=time.perf_counter() - t0,
                archived=None if archived is None else str(archived))


# ============================================================================================ inputs
def static_kwargs(n: int, smoke: bool) -> dict:
    """The training driver's static-fit settings (``train._static_kwargs``): simplex grid (20 steps, floor 0.05) at
    n = 3, CEM over log-weights at n > 3."""
    kw = dict(method="grid" if n == 3 else "cem", seed=0)
    if n == 3:
        kw.update(steps=20, floor=0.05)
    else:
        kw.update(pop=16, iters=2, elite=4) if smoke else kw.update(pop=64, iters=20, elite=8)
    return kw


def static_weights(cal, cfg, models_root, out: Path, smoke: bool, *, test: bool = False) -> tuple[list, str]:
    """Static weights: the training registry's fit (``registry.load_static``; the d cells share the default's),
    else (not on TEST) a fit with the same settings on the configuration's TRAIN windows (smoke: the two smoke
    windows), cached in ``<out>/static_fit.json``."""
    if models_root:
        try:
            fit = R.load_static(cfg.name, root=models_root)
            return [float(x) for x in fit.w], str(R.static_dir(cfg.name, root=models_root) / R.STATIC_FILE)
        except FileNotFoundError:
            pass
    if test:
        raise SystemExit(f"TEST needs the frozen static fit of {cfg.name} below {models_root}")
    cache = out / "static_fit.json"
    rec_cfg = dict(config=S.config_record(cfg), calibration_sha256=S.calibration_sha(cfg, None if not smoke
                                                                                     else S.smoke_calib_root()))
    if cache.exists():
        c = sio.read_json(cache)
        if c.get("inputs") == S._jsonable(rec_cfg) and c.get("smoke") == smoke:
            return [float(x) for x in c["w"]], str(cache)
    from slicemech.control.static import fit_static
    from slicemech.data.calibrate import train_window_keys
    keys = S.SMOKE_WINDOWS if smoke else train_window_keys()
    t0 = time.perf_counter()
    wins = [S.compose(cal, "train", d, o, smoke=smoke) for d, o in keys]
    fit = fit_static(wins, R.mechanism(cal), **static_kwargs(cal.n, smoke))
    rec = dict(w=fit.w.tolist(), train_welfare=fit.welfare, method=fit.method, info=fit.info, windows=len(keys),
               inputs=rec_cfg, smoke=smoke, seconds=time.perf_counter() - t0,
               note="fallback fit: no registry static fit found")
    sio.write_json(cache, S._jsonable(rec))
    return rec["w"], str(cache)


def fit_usage_tariff(net, windows, cfg, cal, mech) -> list:
    """Revenue-neutral usage tariff (experiment_design.md §5): nu_i = sum of the protected net's full prices /
    sum of its quotas over the scored intervals of TRAIN windows (credits per PRB per interval)."""
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    P, Q = np.zeros(cal.n), np.zeros(cal.n)
    for w in windows:
        df = E.run_window(w, net, "full", [Truthful() for _ in range(w.n)], cfg.d, mech, cal.shape,
                          price_columns="billed", record=False)
        sc = df[df["scored"]]
        P += sc.groupby("tenant")["price"].sum().to_numpy()
        Q += sc.groupby("tenant")["quota"].sum().to_numpy()
    return (P / np.maximum(Q, 1e-12)).tolist()


_NU_NETS: dict = {}


def _nu_window_job(args) -> tuple[list, list]:
    """Per-tenant sums (full price, quota) of the protected net on one TRAIN window (spawned worker)."""
    config, calib_root, path, date, off, smoke = args
    import torch
    torch.set_num_threads(1)
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    cfg = S.get_config(config)
    cal = S.load_cal(cfg, calib_root)
    if path not in _NU_NETS:
        _NU_NETS.clear()
        _NU_NETS[path] = S.load_checked_net(path, cfg.d)
    w = S.compose(cal, "train", date, off, smoke=smoke)
    df = E.run_window(w, _NU_NETS[path], "full", [Truthful() for _ in range(w.n)], cfg.d, R.mechanism(cal),
                      cal.shape, price_columns="billed", record=False)
    sc = df[df["scored"]]
    return (sc.groupby("tenant")["price"].sum().to_numpy().tolist(),
            sc.groupby("tenant")["quota"].sum().to_numpy().tolist())


def tariff_file(cfg, out: Path, smoke: bool, protected_seed0=None) -> Path | None:
    """Where the configuration's usage tariff is cached (pre-freeze audit E5-7/F8):

    * smoke runs: ``<out>/usage_nu.json`` (smoke windows and nets; never a frozen location);
    * ``default``: the E2-E4 TRAIN cache of the protected seed-0 model
      (``e2_manipulation.usage_tariff_path``, ``results/raw/train/e2/usage_tariff/``; None without that model);
    * every sweep configuration: ``registry.sweep_tariff_path(<config>)`` =
      ``results/raw/train/e1/<config>/usage_nu.json`` (a TRAIN fit, frozen with the models)."""
    if smoke:
        return Path(out) / "usage_nu.json"
    if cfg.name == R.DEFAULT:
        import e2_manipulation as X
        return None if protected_seed0 is None else X.usage_tariff_path(0, Path(protected_seed0), False)
    return R.sweep_tariff_path(cfg.name)


def usage_tariff(cal, cfg, out: Path, smoke: bool, fit: bool, models: dict, nu_file=None, *, workers: int = 1,
                 calib_root=None, test: bool = False):
    """Usage tariff nu (credits per PRB per interval) -> ``(nu, source, file)``: ``--nu-file``, else the TRAIN fit
    cached in :func:`tariff_file`, else (with ``fit``) the revenue-neutral TRAIN fit of experiment_design.md §5
    (protected seed-0 model, truthful reports, full-decision price, ``nu_i = sum p_i / sum Q_i`` over the scored
    intervals of ALL TRAIN windows of the configuration), written to :func:`tariff_file`; else None.

    Integrator (2026-10-19): the default configuration delegates to ``e2_manipulation.usage_tariff`` (same rule,
    same windows, same seed-0 net, shared cache), so the meas_only bill of E1 and the usage tariff of E2-E4 are one
    number; other configurations fit their own protected net on their own TRAIN windows in parallel. Smoke runs use
    the smoke windows.

    Pre-freeze audit E5-7/F8: the sweep tariffs are fitted on TRAIN before the freeze and stored at
    ``registry.sweep_tariff_path`` (no longer below the split's output folder, where TEST used to refit them); on
    TEST (``test``) the tariff file must exist and was fitted for the frozen protected seed-0 model: it is loaded,
    never refitted (``main`` checks it against the freeze block)."""
    if nu_file:
        return [float(x) for x in sio.read_json(nu_file)["nu"]], str(nu_file), Path(nu_file)
    src = models.get("protected", {}).get(0, (None, None))
    inputs = S._jsonable(dict(config=S.config_record(cfg), protected_seed0_sha256=src[1], smoke=smoke,
                              windows="smoke windows" if smoke else "all TRAIN windows"))
    path = tariff_file(cfg, out, smoke, src[0])
    if path is not None and path.is_file():
        c = sio.read_json(path)
        if cfg.name == R.DEFAULT and not smoke:                     # the E2-E4 cache: keyed by the model SHA-256
            return ([float(x) for x in c["nu"]],
                    f"protected seed 0: {src[0]}; e2_manipulation.usage_tariff (shared with E2-E4): {path}", path)
        if c.get("inputs") == inputs or (src[0] is None and not smoke and not test):
            return [float(x) for x in c["nu"]], str(path), path
        if test:
            raise SystemExit(f"the frozen usage tariff {path} was fitted for other inputs ({c.get('inputs')}); "
                             "TEST never refits it")
    if test:
        raise SystemExit(f"TEST needs the usage tariff of {cfg.name} fitted on TRAIN before the freeze at {path}; "
                         "TEST never refits it (run e1_value.py --fit-nu on DEV first)")
    if not fit:
        return None, "none (meas_only billed 'none'; nu x quota can be applied later)", None
    if src[0] is None:
        if smoke:
            return None, "none (smoke run without the protected arm)", None
        raise SystemExit("--fit-nu needs the protected seed-0 model")
    if smoke:
        keys = list(S.SMOKE_WINDOWS)
        wins = [S.compose(cal, "train", d, o, smoke=smoke) for d, o in keys]
        nu = fit_usage_tariff(S.load_checked_net(src[0], cfg.d), wins, cfg, cal, R.mechanism(cal))
        source = f"protected seed 0: {src[0]}"
    elif cfg.name == R.DEFAULT:
        import e2_manipulation as X
        nu, npath = X.usage_tariff(0, Path(src[0]), False, workers)
        return ([float(x) for x in nu],
                f"protected seed 0: {src[0]}; e2_manipulation.usage_tariff (shared with E2-E4): {npath}", Path(npath))
    else:
        from slicemech.data.calibrate import train_window_keys
        keys = train_window_keys()
        jobs = [(cfg.name, calib_root, str(src[0]), d, int(o), False) for d, o in keys]
        P, Q = np.zeros(cal.n), np.zeros(cal.n)
        for p_i, q_i in S.run_pool(_nu_window_job, jobs, workers):
            P += np.asarray(p_i, float)
            Q += np.asarray(q_i, float)
        nu = (P / np.maximum(Q, 1e-12)).tolist()
        source = f"protected seed 0: {src[0]}"
    S.archive_files([path], path.parent, reason="usage tariff refitted for other inputs")   # never overwritten
    sio.write_json(path, dict(nu=nu, inputs=inputs, config=cfg.name, windows=[list(k) for k in keys], source=source,
                              rule="sum p / sum Q, scored intervals, truthful, full-decision price"))
    return nu, str(path), path


# ============================================================================================ quicklook
COMPARISONS = (("protected", "meas_only"), ("protected", "static"), ("protected", "direct"),
               ("protected", "unprotected"), ("protected", "kappa0"), ("protected", "equal"),
               ("oracle", "protected"), ("direct_truerates", "direct"), ("static", "equal"), ("direct", "equal"))


def quicklook(win: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Descriptive per-arm welfare and paired differences (learned arms averaged over seeds within a window)."""
    from slicemech.eval.stats import cluster_bootstrap
    A = win[win["tenant"] == -1]
    W = A.groupby(["date", "offset_s", "arm"])["realized_value"].mean().unstack("arm")
    load = A.groupby(["date", "offset_s"])["offered_load"].first().reindex(W.index)
    out = dict(windows=int(len(W)), dates=int(W.index.get_level_values(0).nunique()),
               welfare={a: dict(mean=float(W[a].mean()), windows=int(W[a].notna().sum())) for a in W.columns},
               seeds={a: int(A[A["arm"] == a]["seed"].nunique()) for a in A["arm"].unique()},
               miss_pct={a: A[A["arm"] == a]["deadline_miss_pct"].mean() for a in A["arm"].unique()},
               payment_pct_of_V={a: A[A["arm"] == a]["payment_pct_of_V"].mean() for a in A["arm"].unique()})
    terc = pd.qcut(load, 3, labels=("low", "mid", "high")) if load.nunique() >= 3 else None
    comps = {}
    for a, b in COMPARISONS:
        if a not in W.columns or b not in W.columns:
            continue
        d = (W[a] - W[b]).dropna()
        if d.empty:
            continue
        dates = d.index.get_level_values(0).to_numpy()
        rec = dict(mean_diff=float(d.mean()), pct_of_b=float(100 * d.mean() / W[b].loc[d.index].mean()),
                   windows=int(d.size), windows_positive=int((d > 0).sum()))
        if n_boot and np.unique(dates).size >= 2:
            bt = cluster_bootstrap(d.to_numpy(), dates, n_boot=n_boot, seed=0)
            rec["ci95"] = [bt.lo, bt.hi]
        if terc is not None:
            rec["by_offered_load_tercile"] = {str(t): float(d[terc.loc[d.index] == t].mean())
                                              for t in ("low", "mid", "high")}
        comps[f"{a}-{b}"] = rec
    out["paired_differences"] = comps
    out["note"] = ("descriptive quicklook: credits per scored interval; CIs are date-cluster bootstrap "
                   f"({n_boot} replicates); not the pre-registered analysis")
    return out


# ============================================================================================ main
#: Options refused on TEST (round-3 review): window subsets, arm subsets, model / tariff / oracle overrides; since the
#: pre-freeze audit (PV-8/F10) also the protocol-changing ``--intervals`` and ``--unconstrained-envelope`` (the usage
#: tariff is always fitted on TEST, :func:`fit_nu_required`). ``--force`` stays allowed: a logged post-freeze rerun.
TEST_FIXED_OPTIONS = ("--dates", "--limit", "--arms", "--allow-missing", "--models-root", "--calib-root", "--nu-file",
                      "--oracle-pop", "--oracle-iters", "--intervals", "--unconstrained-envelope")


def fit_nu_required(a, split: str) -> bool:
    """The usage tariff is required with ``--fit-nu``, in smoke runs and always on TEST (audit PV-8: on TEST the
    meas_only arm is billed nu Q, never 'none'). On TEST it is loaded from the frozen TRAIN file and never refitted
    (:func:`usage_tariff` with ``test``)."""
    return bool(a.fit_nu or a.smoke or split == "test")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    ap.add_argument("--config", default="default", help=f"one of {', '.join(R.CONFIGS)} (experiments/registry.py)")
    ap.add_argument("--smoke", action="store_true", help="2 short TRAIN windows; the training driver's smoke models, "
                                                       "else 2-iteration PPO nets (seeds 0, 1)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--arms", default=None, help=f"comma-separated subset of {', '.join(ARMS)}")
    ap.add_argument("--models-root", default=None, help="training-registry models (default results/models; smoke: "
                                                        "results/raw/smoke_train/<split>/models)")
    ap.add_argument("--calib-root", default=None, help="directory of the sweep calibrations (default "
                                                       "data/calibration; smoke: the training smoke run's)")
    ap.add_argument("--allow-missing", action="store_true", help="skip learned arms without models")
    ap.add_argument("--unconstrained-envelope", action="store_true",
                    help="also compute the grid envelope formula p of the unconstrained net (~4x its cost)")
    ap.add_argument("--fit-nu", action="store_true", help="fit the usage tariff on all TRAIN windows (protected "
                                                         "seed 0; default config: shared with E2-E4)")
    ap.add_argument("--nu-file", default=None, help="JSON with the usage tariff {'nu': [...]}")
    ap.add_argument("--intervals", choices=("auto", "on", "off"), default="auto",
                    help="store per-interval rows (auto: only for the default configuration, to save disk)")
    ap.add_argument("--oracle-pop", type=int, default=None)
    ap.add_argument("--oracle-iters", type=int, default=None)
    ap.add_argument("--test-freeze", default=None, help="frozen pre-registration (default prereg/PREREG.md)")
    ap.add_argument("--dates", default=None, help="comma-separated subset of the split's dates")
    ap.add_argument("--limit", type=int, default=None, help="first N windows only (debugging)")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap replicates of the quicklook")
    ap.add_argument("--force", action="store_true",
                    help="recompute every planned run (the window files whose rows are replaced are archived first)")
    a = ap.parse_args(argv)
    split = "train" if a.smoke else a.split
    # round-3 review: on TEST every window, arm and model location comes from the frozen protocol (no subsets)
    R.refuse_test_overrides(split, a, ap, TEST_FIXED_OPTIONS)
    freeze = S.test_gate(split, a.test_freeze)                # before anything reads TEST data
    sio.require_free_gb(3)
    cfg = S.get_config(a.config)
    arms = resolve_arms(a.arms)
    calib_root = a.calib_root or (str(S.smoke_calib_root()) if a.smoke and S.smoke_calib_root() else None)
    cal = S.load_cal(cfg, calib_root)
    out = S.out_dir("e1", split, cfg.name, a.smoke)
    t0 = time.perf_counter()

    # ---- learned-arm models: the training registry (smoke: its smoke run, else 2-iteration PPO nets, h = 3)
    if a.models_root:
        models_root = Path(a.models_root)
    elif a.smoke:
        models_root = S.smoke_models_root(cfg)
    else:
        models_root = R.MODELS_DIR
    flat = out / "ckpt" if a.smoke else None
    learned = [LEARNED[x] for x in arms if x in LEARNED]
    found = S.model_paths(cfg, learned, models_root, flat)
    if a.smoke and models_root is None:
        need = [k for k in learned if not all(s in found[k] for s in SMOKE_SEEDS)]
        if need:
            wins = [S.compose(cal, "train", d, o, smoke=True) for d, o in S.SMOKE_WINDOWS]
            S.quick_nets(cal, cfg, wins, tuple(need), SMOKE_SEEDS, flat)
            found = S.model_paths(cfg, learned, None, flat)
    models, missing = {}, []
    for arm in arms:
        if arm in LEARNED:
            fd = found.get(LEARNED[arm], {})
            if not fd:
                missing.append(arm)
                continue
            models[arm] = {int(s): (str(p), sio.file_sha256(p)) for s, p in fd.items()}
            for p in fd.values():                   # fail early: record digests and the forecaster horizon
                S.load_checked_net(p, cfg.d)
    if missing and not a.allow_missing:
        raise SystemExit(f"no finished models below {models_root} for arm(s) {missing} of {cfg.name}; run "
                         "experiments/train.py first or pass --allow-missing to skip these arms")
    arms = tuple(x for x in arms if x not in missing)
    static_w, static_src = static_weights(cal, cfg, models_root, out, a.smoke, test=bool(freeze))
    nu, nu_src, nu_path = usage_tariff(cal, cfg, out, a.smoke, fit_nu_required(a, split), models, a.nu_file,
                                       workers=a.workers, calib_root=calib_root, test=bool(freeze))
    if freeze:                         # every model, the static fit and the TRAIN usage tariff must be the frozen ones
        S.verify_frozen_checkpoints([p for v in models.values() for p, _ in v.values()]
                                    + [x for x in (static_src, nu_path) if x is not None], freeze)
    oracle = dict(SMOKE_ORACLE if a.smoke else ORACLE)
    if a.oracle_pop:
        oracle["pop"] = a.oracle_pop
    if a.oracle_iters:
        oracle["iters"] = a.oracle_iters
    plan = dict(split=split, config=cfg.name, cfg=S.config_record(cfg), cal_sha=S.calibration_sha(cfg, calib_root),
                calib_root=calib_root, smoke=a.smoke, test_freeze=freeze, arms=arms,
                ckpts={arm: {int(s): v for s, v in d.items()} for arm, d in models.items()},
                static_w=static_w, nu=nu, oracle=oracle, envelope=bool(a.unconstrained_envelope),
                intervals=(a.intervals == "on" or (a.intervals == "auto" and (cfg.name == "default" or a.smoke))),
                code_fp=code_fingerprint(), schema=SCHEMA, out=str(out))
    plan["key"] = _digest({k: v for k, v in plan.items() if k != "out"})
    plan.update(force=bool(a.force), archive_stamp=S.archive_stamp())        # not part of the context key
    keys = S.window_keys(split, a.smoke, a.dates.split(",") if a.dates else None, a.limit)
    print(f"[e1] {split}/{cfg.name}: {len(keys)} windows, arms {arms}, seeds "
          f"{ {x: sorted(v) for x, v in plan['ckpts'].items()} } from {models_root or flat}, static w "
          f"{np.round(static_w, 4).tolist()} ({static_src}), nu {nu} ({nu_src})", flush=True)

    jobs = [(plan, d, o) for d, o in keys]
    ran = 0
    archived = set()
    for res in S.run_pool(window_job, jobs, a.workers):
        ran += res["ran"]
        if res.get("archived"):
            archived.add(res["archived"])
        print(f"[e1] {res['key']}: {res['ran']}/{res['planned']} runs computed, {res['seconds']:.1f} s", flush=True)
    win = pd.concat([pd.read_parquet(out / "windows" / f"{S.key_of(d, o)}.parquet") for d, o in keys],
                    ignore_index=True)
    S.write_parquet(win, out / "windows.parquet")
    from slicemech.data.puffer_fetch import analysis_code_sha256
    manifest = dict(
        experiment="E1", split=split, config=S.config_record(cfg), smoke=a.smoke, windows=len(keys),
        runs_computed_this_invocation=ran, arms=list(arms), arms_missing_models=missing,
        expected_seeds=list(cfg.seeds), models_root=str(models_root or flat),
        models={arm: {str(s): dict(path=p, sha256=h) for s, (p, h) in d.items()} for arm, d in plan["ckpts"].items()},
        static_w=static_w, static_source=static_src, usage_nu=nu, usage_nu_source=nu_src,
        usage_nu_file=None if nu_path is None else str(nu_path), oracle=oracle,
        unconstrained_envelope=plan["envelope"], intervals_stored=plan["intervals"],
        calibration=dict(name=cfg.calibration, root=calib_root, sha256=plan["cal_sha"]),
        analysis_code_sha256=analysis_code_sha256(), code_fingerprint=plan["code_fp"], code_modules=list(CODE_MODULES),
        schema=SCHEMA, force=bool(a.force), archived_to=sorted(archived),
        code_fp_values=sorted(str(x) for x in win["code_fp"].dropna().unique()) if "code_fp" in win else [],
        test_freeze=freeze, decisions=R.DECISIONS,
        worst_checks={c: float(np.nanmax(win[c].to_numpy(float))) for c in win.columns if c.startswith("chk_")},
        oracle_selfcheck_max=(float(np.nanmax(win["oracle_selfcheck"].to_numpy(float)))
                              if "oracle_selfcheck" in win else None),
        seconds_total=time.perf_counter() - t0,
        run_seconds={x: float(win[(win["arm"] == x) & (win["tenant"] == -1)]["seconds"].mean())
                     for x in win["arm"].unique()},
        bytes_on_disk=S.dir_bytes(out),
        quicklook=quicklook(win, n_boot=min(a.n_boot, 200) if a.smoke else a.n_boot))
    manifest = S._jsonable(manifest)
    sio.write_json(out / "manifest.json", manifest)
    print(json.dumps(dict(quicklook=manifest["quicklook"], worst_checks=manifest["worst_checks"],
                          oracle_selfcheck_max=manifest["oracle_selfcheck_max"], run_seconds=manifest["run_seconds"],
                          bytes_on_disk=manifest["bytes_on_disk"], out=str(out)), indent=1), flush=True)
    return manifest


if __name__ == "__main__":
    main()
