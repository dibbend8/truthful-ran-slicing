"""E6 - scaling to n = 6 and n = 9 tenants: certified gain, pricing time, cost of protection and payments.

Run from ``code/`` (``PYTHONDONTWRITEBYTECODE=1 /home/darpan/anaconda3/bin/python experiments/e6_scaling.py``):

    --smoke                        tiny TRAIN plumbing run (1 short window per n, 2 PPO iterations for tiny
                                   checkpoints trained on the spot, best responses on a few intervals);
                                   writes results/raw/train/e6/smoke/
    --split dev [--workers 24]     DEV (descriptive; results/raw/dev/e6/)
    --split test --test-freeze ../prereg/PREREG.md
                                   TEST, only after the pre-registration freeze (refused otherwise)
    --plan / --aggregate / --force as in e5_training_channel.py

The run is resumable: one job is one (n, window); it is skipped when ``jobs/<job>.done.json`` exists AND records the
job's current provenance fingerprint (model file SHA-256s, arms, seeds, d, best-response period, calibration and the
code fingerprint of ``slicemech`` and this module; round-3 review). A (re)computed job first moves its old record and
outputs to ``results/raw/<split>/archive/<UTC stamp>/e6/jobs/``; ``done.json`` records the code fingerprint ``code``;
an ``error.txt`` newer than its ``done.json`` is an error (pre-freeze audit E5-2, F7/PV-10). On TEST the protocol options
(``TEST_FIXED_OPTIONS``) are refused and every model must be listed in the freeze block (``registry.verify_frozen``).

Design (experiment_design.md E6; H6a, H6b; spec/params.yaml ``n_tenants.scaling``)
----------------------------------------------------------------------------------
Calibrations ``data/calibration/n6_default.json`` (2 video tenants on channel groups {abc, cbs, cw} and
{fox, nbc, pbs, univision}, 2 URLLC-like, 2 mMTC-like) and ``n9_default.json`` (3 of each; video groups
{abc, cbs}, {cw, fox}, {nbc, pbs, univision}); the same B = 106 PRB and rho = 0.9. Every pre-declared window of
the split (TRAIN 12, DEV 6, TEST 8 per date) is played, with measurement delay d = 2 intervals, by

* ``equal``       equal weights, full-decision price (seed-free);
* ``direct``      direct declared-welfare optimiser, full-decision price (seed-free);
* ``protected``   trained protected model of each seed, full-decision price;
* ``unprotected`` trained unconstrained model of each seed, fixed-priority price, truthful reports.

Trained models come from the shared layout of ``experiments/registry.py``: configuration ``n6`` / ``n9``,
``results/models/n<n>/<protected|unconstrained>/seed<s>/model.pt`` with its finished ``config.json`` (written by
``experiments/train.py``; file and parameter digests are checked). ``--models-root`` relocates the layout and
``--ckpt-template`` reads plain ``nets.save_net`` files instead (``{cal}``, ``{n}``, ``{kind}``, ``{seed}``).

Certified gain (H6a, as H3a): in the three monotone arms every tenant is an ex-post full-price best responder
against truthful opponents in every scored interval (``--br-every m``: every m-th scored interval);
``best_response`` enumerates the own-quota cells exactly and certifies the gain. A tenant-interval violates the
claim when ``gain > max(cert, 1e-8 pi_i mu_i Delta)``. The full price makes truth a best response, so all
reports equal the (admitted) types and the best-response run is also the truthful run (checked per run; a
separate truthful run is played otherwise).

Pricing time (H6a: p95 <= 100 ms per interval at n = 9 on one core): after each monotone run, every interval is
re-priced from the same released context, priority map and reports (``envelope.full_prices`` for all n tenants,
and ``fixed_priority_prices`` for the priority maps), timed with ``perf_counter`` in a single-threaded worker,
and checked against the episode's logged prices (``reprice_max_abs_diff``, expected 0). The timings include the
machine load of concurrent workers (recorded as the load average).

Cost of protection (H6b, descriptive; D4): realized welfare ``W = sum_i pi_i on_time_i`` per scored interval of the
unprotected and the protected arm with truthful reports on the same window and seed; ``W_unprot - W_prot`` in
credits per interval and in % of ``W_prot``; equal and direct as references. Payments: billed price, full price,
fixed-priority price and premium ``T = p - p_fix`` as % of declared value; rounding loss = idle PRB
``B - sum_i Q_i`` (integer flooring).

Outputs (``results/raw/<split>/e6/``; ``smoke/`` for --smoke): ``jobs/n<n>__<date>__o<offset>.{runs,tenants,br,
timing}.parquet`` + ``.done.json``; aggregates ``runs.parquet`` (one row per n x window x arm x seed),
``tenants.parquet`` (``eval.metrics.window_summary`` rows), ``br.parquet`` (every best-responding scored
tenant-interval: gain, certificate, allowance, excess), ``timing.parquet`` (per interval, ms),
``protection.parquet`` (per n x window x seed) and ``checks.json`` (per n: certified-gain maximum and violations
per arm, pricing time median/p95, cost of protection and payments, means over windows; CIs are the analysis
step's job). Signs: ``d_W_unprot_minus_prot`` > 0 means protection costs welfare.

Orchestrator decisions (binding, 2026-10-19; code/STATUS.md §0)
-----------------------------------------------------------------
D1 GBR fractions x 0.8 (video 0.40, URLLC-like 0.64, mMTC-like 0.16): in the n6/n9 calibrations' contracts.
D2 offered-load normalization per date, every split; at n = 6 and 9 each class share is split equally among the
   tenants of the class: ``Calibration.window`` composes with the date's counts; per-window loads are recorded.
D3 gate G4 dropped (not used here). D4 harm on honest tenants' own outcomes (no deviation is played here: under the
   full price the best response is the truth; total welfare is reported for the protection cost only).
D5 E5 protocol (not used here). D6 the checkpoints are trained with the lookahead return, h = 3 (training driver).
D7 E4b kappa-scaling frontier (not used here).

TEST is read only after the pre-registration freeze (``data.puffer_fetch.require_test_read_gate``).
"""
from __future__ import annotations

import argparse
import json
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

N_DEFAULT = tuple(int(x) for x in PARAMS["n_tenants"]["scaling"])    # (6, 9)
D_DEFAULT = int(PARAMS["information"]["measurement_delay_s"])        # 2 intervals
HORIZON = 3                                                          # D6 (smoke checkpoints)
SEEDS_DEFAULT = (0, 1, 2)                                            # params: SeedsSweep = 3
EPS_REL = 1e-8                                                       # H3a / H6a allowance: 1e-8 pi mu Delta
#: arm -> (net kind or None, billed price, best responders?)
ARMS = {"equal": (None, "full", True), "direct": (None, "full", True),
        "protected": ("protected", "full", True), "unprotected": ("unconstrained", "fixed", False)}
SMOKE = dict(window=("2024-01-14", 60), ckpt_window=("2024-01-01", 60), K=20, warmup=8, iters=2, seeds=(0,))


# ============================================================================================ helpers
def out_dir(split: str, smoke: bool) -> Path:
    base = sio.ROOT / "results" / "raw" / split / "e6"
    return base / "smoke" if smoke else base


def write_parquet(df: pd.DataFrame, path: Path) -> None:
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


def resolve_ckpt(template: str, n: int, kind: str, seed: int) -> Path:
    p = Path(template.format(cal=f"n{n}_default", n=n, kind=kind, seed=int(seed)))
    return p if p.is_absolute() else sio.ROOT / p


#: Options refused on TEST (round-3 review): the E6 protocol is the frozen code's.
TEST_FIXED_OPTIONS = ("--n", "--dates", "--offsets", "--seeds", "--arms", "--d", "--br-every", "--models-root",
                      "--ckpt-template", "--out")


def model_entry(n: int, kind: str, seed: int, *, models_root=None, template=None) -> dict:
    """Trained model of (n, kind, seed): the registry layout ``<models_root>/n<n>/<kind>/seed<seed>/model.pt`` (with
    its finished ``config.json``), or a plain ``nets.save_net`` file from ``template``."""
    if template:
        path = resolve_ckpt(template, n, kind, seed)
        return dict(path=str(path), sha256=None, param_digest=None, missing=not path.is_file())
    d = R.model_dir(f"n{n}", kind, seed, root=models_root)
    rec = R.read_done(d / R.CONFIG_FILE)
    res = (rec or {}).get("result") or {}
    return dict(path=str(d / R.MODEL_FILE), sha256=res.get("model_sha256"), param_digest=res.get("param_digest"),
                missing=rec is None or not (d / R.MODEL_FILE).is_file())


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


class SparseBestResponder:
    """Ex-post full-price (or other) best responder on every ``every``-th scored interval, truthful otherwise.

    Wraps ``tenants.strategies.BestResponder``; intervals are counted from ``reset()`` (one ``report`` call per
    interval, as ``sim.episode.run_window`` makes them). ``last`` is None on intervals without a best response,
    so the episode records best-response columns only where one was computed.
    """

    needs_game = True

    def __init__(self, price: str, warmup: int, every: int = 1):
        from slicemech.tenants.strategies import BestResponder
        if int(every) < 1 or int(warmup) < 0:
            raise ValueError("every >= 1 and warmup >= 0 required")
        self._br = BestResponder(price)
        self.price, self.warmup, self.every = price, int(warmup), int(every)
        self.name = f"best_response[{price}]/every{self.every}"
        self.n_clipped, self.last, self._k = 0, None, 0

    def reset(self) -> None:
        self._br.reset()
        self.last, self._k = None, 0

    def active(self, k: int) -> bool:
        return k >= self.warmup and (k - self.warmup) % self.every == 0

    def report(self, i, theta, C, history=None, game=None) -> float:
        k, self._k = self._k, self._k + 1
        if self.active(k):
            before = self._br.n_clipped
            z = self._br.report(i, theta, C, history=history, game=game)
            self.n_clipped += self._br.n_clipped - before
            self.last = self._br.last
            return z
        self.last = None
        if theta > C:
            self.n_clipped += 1
        return float(min(theta, C))


def pricing_times(window, controller, reports: np.ndarray, d: int, mech, shape, *, fixed: bool,
                  logged_p: np.ndarray, logged_p_fix: np.ndarray | None) -> dict:
    """Re-price every interval of a played window and time it (single thread; ms).

    Rebuilds the context from released data, the controller's map (or direct rule) and the quota rule exactly as
    ``sim.episode.run_window`` does, then times ``envelope.full_prices`` (all n tenants) and, for priority maps,
    ``envelope.fixed_priority_prices``. Returns per-interval times and the largest difference to the logged prices.
    """
    from slicemech.control import features as F
    from slicemech.model import allocation as A
    from slicemech.pricing import envelope as Env
    from slicemech.sim import episode as E
    ct = window.contract
    K = window.K
    fc = getattr(controller, "forecaster", None) or F.OperatorAR.default()
    direct = hasattr(controller, "quota_policy")
    view = E.ReleasedData(window, d)
    view.advance(0)
    t_full, t_fix = np.full(K, np.nan), np.full(K, np.nan)
    d_full = d_fix = 0.0
    for k in range(K):
        ctx = F.context(view, k, d, ct, forecaster=fc, mech=mech)
        r = np.asarray(window.rate_hat[k], float).copy()
        pol = (controller.quota_policy(ctx, ct=ct, mech=mech, shape=shape, r=r) if direct
               else controller.priority_map(ctx))
        rule = A.QuotaRule(ct, mech, r, pol)
        z = np.asarray(reports[k], float)
        t0 = time.perf_counter()
        full = Env.full_prices(z, rule, ct, shape, mech, r)
        t_full[k] = 1e3 * (time.perf_counter() - t0)
        d_full = max(d_full, float(np.max(np.abs(np.array([x.p for x in full]) - logged_p[k]))))
        if fixed and not direct:
            t0 = time.perf_counter()
            fx = Env.fixed_priority_prices(z, rule, ct, shape, mech, r)
            t_fix[k] = 1e3 * (time.perf_counter() - t0)
            if logged_p_fix is not None:
                d_fix = max(d_fix, float(np.max(np.abs(np.array([x.p for x in fx]) - logged_p_fix[k]))))
        if k + 1 < K:
            view.advance(k + 1)
    return dict(t_full_ms=t_full, t_fixed_ms=t_fix, reprice_max_abs_diff=d_full, reprice_fix_max_abs_diff=d_fix)


def _matrix(df: pd.DataFrame, col: str, K: int, n: int) -> np.ndarray:
    return df[col].to_numpy(float).reshape(K, n)


def _q(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


# ============================================================================================ one job
def run_job(spec: dict) -> dict:
    """One (n, window): every arm and seed (module docstring). Writes ``jobs/<job>.*`` and returns a status. The
    job's earlier record and outputs are archived first (``e5_training_channel.retire_job_files``); the new
    ``done.json`` records the code fingerprint (``code``)."""
    from e5_training_channel import retire_job_files
    retire_job_files(Path(spec["out"]), spec["job"], spec.get("archive_stamp"))
    import torch
    torch.set_num_threads(1)
    from slicemech.config import Mechanism
    from slicemech.data.calibrate import load_calibration
    from slicemech.eval.headroom import window_load
    from slicemech.eval.metrics import window_summary
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful

    t_job = time.perf_counter()
    out, jid = Path(spec["out"]), spec["job"]
    jdir = out / "jobs"
    split, n, date, off, d = spec["split"], int(spec["n"]), spec["date"], int(spec["offset"]), int(spec["d"])
    mech = Mechanism()
    cal = load_calibration(f"n{n}_default")
    if cal.n != n:
        raise ValueError(f"calibration n{n}_default has n = {cal.n}")
    tf = {"test_freeze": spec["test_freeze"]} if split == "test" and spec.get("test_freeze") else {}
    w = cal.window(split, date, off, **tf, **dict(spec.get("overrides") or {}))
    ct = w.contract
    K, warm = w.K, int(w.warmup)
    from slicemech.eval.headroom import offered_load
    keys = dict(split=split, n=n, date=date, offset_s=off, window=w.id)
    winfo = dict(offered_load=offered_load(w, cal.r_bar, mech),                  # all K (E0/E1 D2 quantity)
                 offered_load_scored=offered_load(w, cal.r_bar, mech, scored_only=True),
                 capacity_load=window_load(w, mech), counts=json.dumps([int(c) for c in w.meta["counts"]]))
    allow_t = EPS_REL * ct.pi * ct.mu * mech.Delta                       # (n,) credits
    runs, tenants, brs, timing = [], [], [], []
    ckpt_sha = {}
    plan = []
    for arm in spec["arms"]:
        kind, price, br = ARMS[arm]
        if kind is None:
            plan.append((arm, -1, None, price, br))
        else:
            for s in spec["seeds"]:
                plan.append((arm, int(s), spec["models"][f"{kind}|{s}"], price, br))
    for arm, seed, ck, price, br in plan:
        t0 = time.perf_counter()
        if arm == "equal":
            ctrl = E.FixedWeights()
        elif arm == "direct":
            ctrl = E.DirectController()
        else:
            ctrl, ckpt_sha[ck["path"]] = load_checked(ck)
            if ctrl.kind != ARMS[arm][0] or ctrl.forecaster is None:
                raise ValueError(f"checkpoint {ck}: kind {ctrl.kind!r} / forecaster {ctrl.forecaster} unusable")
        if br:
            pols = [SparseBestResponder(price, warm, spec["br_every"]) for _ in range(n)]
            df = E.run_window(w, ctrl, price, pols, d, mech, cal.shape, price_columns="all")
            reused = bool(np.array_equal(df["report"].to_numpy(), df["theta_adm"].to_numpy()))
            truth = df if reused else E.run_window(w, ctrl, price, [Truthful() for _ in range(n)], d, mech,
                                                   cal.shape, price_columns="all")
        else:
            df = truth = E.run_window(w, ctrl, price, [Truthful() for _ in range(n)], d, mech, cal.shape,
                                      price_columns="billed")
            reused = None
        sec_play = time.perf_counter() - t0
        sc = truth[truth["scored"]]
        per_k = sc.groupby("k")
        V = float(sc["V_declared"].sum())

        def pct_V(col, _sc=sc, _V=V):                   # NaN when the column was not computed for this arm
            return 100.0 * float(np.sum(_sc[col].to_numpy(float))) / _V if _V > 0 else np.nan

        T_lo = sc["T_lo"].to_numpy(float)
        chk = E.episode_checks(truth, ct, mech)
        row = dict(**keys, **winfo, arm=arm, seed=seed, price_rule=price, seconds_play=sec_play,
                   W_realized=float(per_k["realized_value"].sum().mean()),
                   W_declared=float(per_k["V_declared"].sum().mean()),
                   revenue=float(per_k["price"].sum().mean()),
                   idle_prb=float(mech.B - per_k["quota"].sum().mean()),
                   payment_pct_V=pct_V("price"), p_full_pct_V=pct_V("p"), p_fix_pct_V=pct_V("p_fix"),
                   T_pct_V=pct_V("T"), T_mean=float(np.mean(sc["T"].to_numpy(float))),
                   T_pos_share=float(np.mean(T_lo > 0)) if np.all(np.isfinite(T_lo)) else np.nan,
                   share_overload=float(sc.groupby("k")["overload"].first().mean()),
                   share_leftover=float(sc.groupby("k")["leftover"].first().mean()),
                   on_time_Mbit=float(per_k["on_time_Mbit"].sum().mean()),
                   dropped_Mbit=float(per_k["dropped_Mbit"].sum().mean()),
                   br_truth_reused=reused, **{f"chk_{k}": v for k, v in chk.items() if not k.startswith("br_")})
        if br:
            b = df[df["scored"] & np.isfinite(df["br_gain"].to_numpy())]
            t = b["tenant"].to_numpy()
            allow = np.maximum(b["br_cert"].to_numpy(float), allow_t[t])
            exc = b["br_gain"].to_numpy(float) - allow
            brs.append(pd.DataFrame(dict(**{k: [v] * len(b) for k, v in keys.items()}, arm=arm, seed=seed,
                                         k=b["k"].to_numpy(), tenant=t, cls=b["cls"].to_numpy(),
                                         gain=b["br_gain"].to_numpy(float), cert=b["br_cert"].to_numpy(float),
                                         allowance=allow, excess=exc, exact=b["br_exact"].to_numpy(bool),
                                         misreport=b["report"].to_numpy() != b["theta_adm"].to_numpy())))
            row.update(br_rows=int(len(b)), br_exact_share=float(b["br_exact"].mean()) if len(b) else np.nan,
                       br_gain_max=float(b["br_gain"].max()) if len(b) else np.nan,
                       br_excess_max=float(exc.max()) if len(b) else np.nan,
                       br_violations=int((exc > 0).sum()), br_misreport_share=float(
                           (b["report"].to_numpy() != b["theta_adm"].to_numpy()).mean()) if len(b) else np.nan)
            t0 = time.perf_counter()
            tm = pricing_times(w, ctrl, _matrix(truth, "report", K, n), d, mech, cal.shape, fixed=arm != "direct",
                               logged_p=_matrix(truth, "p", K, n),
                               logged_p_fix=None if arm == "direct" else _matrix(truth, "p_fix", K, n))
            ks = np.arange(K)
            timing.append(pd.DataFrame(dict(**{k: [v] * K for k, v in keys.items()}, arm=arm, seed=seed, k=ks,
                                            scored=ks >= warm, t_full_ms=tm["t_full_ms"].astype(np.float32),
                                            t_fixed_ms=tm["t_fixed_ms"].astype(np.float32))))
            s_full, s_fix = tm["t_full_ms"][warm:], tm["t_fixed_ms"][warm:]
            row.update(t_full_ms_median=_q(s_full, 0.5), t_full_ms_p95=_q(s_full, 0.95),
                       t_full_ms_max=float(np.nanmax(s_full)), t_fixed_ms_median=_q(s_fix, 0.5),
                       t_fixed_ms_p95=_q(s_fix, 0.95), reprice_max_abs_diff=tm["reprice_max_abs_diff"],
                       reprice_fix_max_abs_diff=tm["reprice_fix_max_abs_diff"],
                       seconds_reprice=time.perf_counter() - t0, loadavg_1min=os.getloadavg()[0])
        runs.append(row)
        tenants.append(window_summary(truth, warm, keys=dict(**keys, arm=arm, seed=seed)))

    jdir.mkdir(parents=True, exist_ok=True)
    write_parquet(pd.DataFrame(runs), jdir / f"{jid}.runs.parquet")
    write_parquet(pd.concat(tenants, ignore_index=True), jdir / f"{jid}.tenants.parquet")
    if brs:
        write_parquet(pd.concat(brs, ignore_index=True), jdir / f"{jid}.br.parquet")
    if timing:
        write_parquet(pd.concat(timing, ignore_index=True), jdir / f"{jid}.timing.parquet")
    sec = time.perf_counter() - t_job
    write_json(jdir / f"{jid}.done.json", dict(job=jid, **keys, fp=spec.get("fp"), seeds=list(spec["seeds"]),
                                                arms=list(spec["arms"]),
                                                br_every=int(spec["br_every"]), d=d, ckpt_sha256=ckpt_sha,
                                                overrides=dict(spec.get("overrides") or {}), seconds=sec,
                                                smoke=bool(spec.get("smoke")), code=code_fingerprint()))
    (jdir / f"{jid}.error.txt").unlink(missing_ok=True)
    return dict(job=jid, status="done", seconds=sec)


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
def aggregate(out: Path) -> dict:
    """Concatenate finished jobs; build ``protection.parquet`` and ``checks.json`` (descriptive means). A job whose
    ``error.txt`` is newer than its ``done.json`` is reported in ``jobs_error`` (``jobs_error_after_done``) and left
    out (``e5_training_channel.job_errors``; pre-freeze audit E5-2)."""
    from e5_training_channel import job_errors
    jdir = out / "jobs"
    errors, after = job_errors(jdir)
    finished = ([p.name[: -len(".done.json")] for p in sorted(jdir.glob("*.done.json"))
                 if p.name[: -len(".done.json")] not in set(after)] if jdir.is_dir() else [])
    tables = {}
    for name in ("runs", "tenants", "br", "timing"):
        parts = [pd.read_parquet(jdir / f"{j}.{name}.parquet") for j in finished
                 if (jdir / f"{j}.{name}.parquet").is_file()]
        if parts:
            tables[name] = pd.concat(parts, ignore_index=True)
            write_parquet(tables[name].copy(), out / f"{name}.parquet")
    codes = []
    for j in finished:
        try:
            codes.append(json.loads((jdir / f"{j}.done.json").read_text()).get("code"))
        except (ValueError, OSError):
            codes.append(None)
    checks: dict = dict(jobs_done=len(finished), jobs_error=errors, jobs_error_after_done=after,
                        code_versions=len(set(codes) - {None}), jobs_without_code=sum(1 for c in codes if not c),
                        by_n={})
    if "runs" not in tables:
        write_json(out / "checks.json", checks)
        return checks
    runs = tables["runs"]
    idx = ["split", "n", "date", "offset_s", "window"]
    prot = runs[runs["arm"] == "protected"].set_index(idx + ["seed"])
    unp = runs[runs["arm"] == "unprotected"].set_index(idx + ["seed"])
    pc = pd.DataFrame()
    common = prot.index.intersection(unp.index)
    if len(common):
        pc = pd.DataFrame(dict(W_prot=prot.loc[common, "W_realized"], W_unprot=unp.loc[common, "W_realized"],
                               pay_prot_pct_V=prot.loc[common, "payment_pct_V"],
                               pay_unprot_pct_V=unp.loc[common, "payment_pct_V"],
                               offered_load=prot.loc[common, "offered_load"])).reset_index()
        pc["d_W_unprot_minus_prot"] = pc["W_unprot"] - pc["W_prot"]
        pc["d_W_pct_of_prot"] = 100.0 * pc["d_W_unprot_minus_prot"] / pc["W_prot"]
        for ref in ("equal", "direct"):
            r = runs[runs["arm"] == ref].set_index(idx)["W_realized"].rename(f"W_{ref}")
            pc = pc.merge(r.reset_index(), on=idx, how="left")
        write_parquet(pc, out / "protection.parquet")
    br, tm = tables.get("br"), tables.get("timing")
    for n, g in runs.groupby("n"):
        rec: dict = dict(windows=int(g["window"].nunique()), arms={})
        for arm, ga in g.groupby("arm"):
            a = dict(runs=int(len(ga)), W_realized_mean=float(ga["W_realized"].mean()),
                     payment_pct_V_mean=float(ga["payment_pct_V"].mean()), idle_prb_mean=float(ga["idle_prb"].mean()),
                     T_pct_V_mean=float(ga["T_pct_V"].mean()))
            if br is not None and arm in set(br["arm"]):
                b = br[(br["n"] == n) & (br["arm"] == arm)]
                a.update(br_tenant_intervals=int(len(b)), br_exact_share=float(b["exact"].mean()),
                         br_gain_max=float(b["gain"].max()), br_excess_max=float(b["excess"].max()),
                         br_violations=int((b["excess"] > 0).sum()), br_misreports=int(b["misreport"].sum()))
            if tm is not None and arm in set(tm["arm"]):
                t = tm[(tm["n"] == n) & (tm["arm"] == arm) & tm["scored"]]
                a.update(t_full_ms_median=_q(t["t_full_ms"], 0.5), t_full_ms_p95=_q(t["t_full_ms"], 0.95),
                         t_full_ms_max=float(np.nanmax(t["t_full_ms"])) if len(t) else np.nan,
                         t_fixed_ms_median=_q(t["t_fixed_ms"], 0.5), t_fixed_ms_p95=_q(t["t_fixed_ms"], 0.95),
                         reprice_max_abs_diff=float(ga["reprice_max_abs_diff"].max()))
            rec["arms"][arm] = a
        if len(pc):
            p = pc[pc["n"] == n]
            rec["protection"] = dict(pairs=int(len(p)), d_W_mean=float(p["d_W_unprot_minus_prot"].mean()),
                                     d_W_pct_mean=float(p["d_W_pct_of_prot"].mean()),
                                     d_W_pct_of_mean=float(100 * p["d_W_unprot_minus_prot"].mean()
                                                           / p["W_prot"].mean()))
        mono = [rec["arms"][a] for a in ("equal", "direct", "protected") if a in rec["arms"]]
        rec["h6a_zero_certified_gain"] = bool(mono and all(a.get("br_violations", 1) == 0 for a in mono))
        if "protected" in rec["arms"] and "t_full_ms_p95" in rec["arms"]["protected"]:
            rec["h6a_p95_ms_protected"] = rec["arms"]["protected"]["t_full_ms_p95"]
        checks["by_n"][str(int(n))] = rec
    write_json(out / "checks.json", checks)
    return checks


# ============================================================================================ smoke checkpoints
def smoke_checkpoints(models_root: Path, ns, seeds, iters: int) -> None:
    """Tiny models per n for the smoke run (``iters`` PPO iterations on one short TRAIN window; not experiment nets)
    in the registry layout ``<models_root>/n<n>/<kind>/seed<s>/{model.pt, config.json}``."""
    import torch
    torch.set_num_threads(1)
    from slicemech.config import Mechanism
    from slicemech.control import nets as N
    from slicemech.control import ppo as P
    from slicemech.data.calibrate import load_calibration
    for n in ns:
        cal = load_calibration(f"n{n}_default")
        date, off = SMOKE["ckpt_window"]
        w = None
        for kind in ("protected", "unconstrained"):
            for s in seeds:
                d = R.model_dir(f"n{n}", kind, s, root=models_root)
                if R.read_done(d / R.CONFIG_FILE) is not None and (d / R.MODEL_FILE).is_file():
                    continue
                if w is None:
                    w = cal.window("train", date, off, K=SMOKE["K"], warmup=SMOKE["warmup"])
                cfg = P.PPOConfig(iterations=iters, windows_per_iter=1, eval_every=0, horizon=HORIZON)
                res = P.train(N.make_net(kind, seed=s), [w], None, d=D_DEFAULT, mech=Mechanism(), config=cfg,
                              seed=s, forecaster=cal.operator_forecaster(D_DEFAULT))
                d.mkdir(parents=True, exist_ok=True)
                N.save_net(res.net, d / R.MODEL_FILE)
                write_json(d / R.CONFIG_FILE, dict(
                    status="done", smoke=True, config=f"n{n}", arm=kind, seed=int(s),
                    result=dict(log_sigma=float(res.log_sigma), model_sha256=sio.file_sha256(d / R.MODEL_FILE),
                                param_digest=N.param_digest(res.net), iterations=iters)))


# ============================================================================================ main
def code_fingerprint() -> str:
    """SHA-256 over the ``slicemech`` package and this module (part of every job fingerprint)."""
    return sio.digest(dict(slicemech=R.package_sha256(), module=sio.file_sha256(Path(__file__))))


def job_fingerprint(job: dict, code: str) -> str:
    """Provenance fingerprint of one job (module docstring): not its output location or the freeze path."""
    from slicemech.data.calibrate import CALIB_DIR
    keep = {k: v for k, v in job.items() if k not in ("out", "test_freeze", "fp", "models", "archive_stamp")}
    shas = {k: (sio.file_sha256(m["path"]) if Path(m["path"]).is_file() else None)
            for k, m in sorted(job["models"].items())}
    return sio.digest(dict(keep, model_sha256=shas, code=code,
                           calibration_sha256=sio.file_sha256(CALIB_DIR / f"n{int(job['n'])}_default.json")))


def job_done(out: Path, job: dict) -> bool:
    p = out / "jobs" / f"{job['job']}.done.json"
    if not p.is_file():
        return False
    try:
        return json.loads(p.read_text()).get("fp") == job.get("fp")
    except (ValueError, OSError):
        return False


def build_jobs(args, out: Path) -> list[dict]:
    from slicemech.data import compose, puffer
    smoke = args.smoke
    ns = tuple(int(x) for x in args.n.split(",")) if args.n else N_DEFAULT
    for n in ns:
        if n not in (6, 9):
            raise SystemExit(f"E6 runs n in (6, 9), not {n}")
    seeds = (SMOKE["seeds"] if smoke and not args.seeds else
             tuple(int(s) for s in args.seeds.split(",")) if args.seeds else tuple(SEEDS_DEFAULT))
    arms = tuple(args.arms.split(",")) if args.arms else tuple(ARMS)
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from {tuple(ARMS)}")
    if smoke:
        wins = [SMOKE["window"]]
        ov = dict(K=SMOKE["K"], warmup=SMOKE["warmup"])
    else:
        dates = args.dates.split(",") if args.dates else list(puffer.split_dates(args.split))
        offs = ([int(o) for o in args.offsets.split(",")] if args.offsets else list(compose.window_offsets(args.split)))
        wins = [(dd, o) for dd in dates for o in offs]
        ov = {}
    models_root = args.models_root
    if smoke and not models_root and not args.ckpt_template:
        models_root = str(out / "models")
        if not args.plan:
            smoke_checkpoints(Path(models_root), ns, seeds, SMOKE["iters"])
    jobs = []
    for n in ns:
        models = {}
        for a in arms:
            kind = ARMS[a][0]
            if kind is None:
                continue
            for s in seeds:
                models[f"{kind}|{s}"] = model_entry(n, kind, s, models_root=models_root, template=args.ckpt_template)
        for date, off in wins:
            jobs.append(dict(job=f"n{n}__{date}__o{int(off):05d}", split=args.split, n=n, date=date, offset=int(off),
                             seeds=list(seeds), arms=list(arms), models=models, d=args.d, br_every=args.br_every,
                             overrides=ov, out=str(out), smoke=smoke, test_freeze=args.test_freeze))
    code = code_fingerprint()
    shas: dict = {}
    for j in jobs:                                    # model hashes once per file
        for m in j["models"].values():
            if m["path"] not in shas:
                shas[m["path"]] = sio.file_sha256(m["path"]) if Path(m["path"]).is_file() else None
        j["fp"] = sio.digest(dict(base=job_fingerprint(dict(j, models={}), code),
                                  models={k: shas[m["path"]] for k, m in sorted(j["models"].items())}))
    return jobs


#: Seconds per 200-s window and arm (one seed), measured 2026-10-19 on a DEV window (2024-03-10 +14460 s) with
#: --br-every 1 at a machine load average of ~16 (hyper-threaded cores): (play, re-pricing).
_COST_S = {6: dict(equal=(34.6, 5.2), direct=(38.9, 9.2), protected=(44.0, 6.0), unprotected=(6.5, 0.0)),
           9: dict(equal=(37.2, 6.3), direct=(43.8, 7.6), protected=(63.6, 10.7), unprotected=(4.1, 0.0))}


def estimate_core_hours(jobs: list[dict], smoke: bool) -> float:
    """Core-hours from :data:`_COST_S` (best responses take ~85 % of a best-responding arm's play time and scale
    with 1 / --br-every); smoke windows scale with K / 200."""
    tot = 0.0
    for j in jobs:
        cost = _COST_S.get(int(j["n"]), _COST_S[9])
        scale = (j["overrides"].get("K", 200) / 200.0) if smoke else 1.0
        br = 0.15 + 0.85 / max(1, int(j["br_every"]))
        for arm in j["arms"]:
            play, rep = cost[arm]
            reps = 1 if ARMS[arm][0] is None else len(j["seeds"])
            tot += reps * scale * (play * (br if ARMS[arm][2] else 1.0) + rep)
        tot += 2.0                                                    # composition
    return tot / 3600.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("train", "dev", "test"), default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny TRAIN plumbing run (see module docstring)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", default=None, help="comma-separated tenant counts (default 6,9)")
    ap.add_argument("--dates", default=None, help="comma-separated dates (default: every date of the split)")
    ap.add_argument("--offsets", default=None, help="window offsets in s (default: the split's pre-declared ones)")
    ap.add_argument("--seeds", default=None,
                    help=f"checkpoint seeds (default {','.join(map(str, SEEDS_DEFAULT))}, SEEDS_DEFAULT)")
    ap.add_argument("--arms", default=None, help=f"subset of {tuple(ARMS)}")
    ap.add_argument("--d", type=int, default=D_DEFAULT, help="measurement delay (intervals)")
    ap.add_argument("--br-every", type=int, default=1, help="best responses on every m-th scored interval")
    ap.add_argument("--models-root", default=None,
                    help="root of the registry model layout (default results/models; smoke: its own tiny models)")
    ap.add_argument("--ckpt-template", default=None,
                    help="read plain nets.save_net files instead: path relative to the project root with {cal}, "
                         "{n}, {kind}, {seed}")
    ap.add_argument("--test-freeze", default=None, help="frozen pre-registration file (TEST only)")
    ap.add_argument("--out", default=None, help="output directory (inside the project; default results/raw/...)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
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
    if args.br_every < 1:
        raise SystemExit("--br-every must be >= 1")
    out = sio.confine(Path(args.out)) if args.out else out_dir(args.split, args.smoke)
    if args.aggregate:
        print(json.dumps(aggregate(out), indent=1, default=str))
        return 0
    if not args.plan:
        sio.require_free_gb(3)
        out.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args, out)
    missing = sorted({m["path"] for j in jobs for m in j["models"].values() if m["missing"]})
    if missing and not args.plan:
        raise SystemExit("missing trained models (run experiments/train.py first, or pass --models-root / "
                         "--ckpt-template):\n  " + "\n  ".join(missing))
    if freeze and not args.plan:                                       # every model must be the frozen one
        R.verify_frozen(sorted({m["path"] for j in jobs for m in j["models"].values()}), freeze)
    todo = [j for j in jobs if args.force or not job_done(out, j)]
    stamp = None
    if not args.plan:
        import e0_setup as S
        stamp = S.archive_stamp()
    for j in todo:                       # one archive folder per invocation (not part of the fingerprint)
        j["archive_stamp"] = stamp
    print(f"E6 {args.split}{' smoke' if args.smoke else ''}: {len(jobs)} jobs ({len(todo)} to run), estimated "
          f"{estimate_core_hours(jobs, args.smoke):.2f} core-hours in total; output {out}")
    if args.plan:
        for j in jobs:
            print(" ", j["job"])
        return 0
    t0 = time.perf_counter()
    results = []
    if args.workers <= 1 or len(todo) <= 1:
        for j in todo:
            results.append(_run_safe(j))
            print(f"  {results[-1]['job']}: {results[-1]['status']} ({results[-1].get('seconds', float('nan')):.1f} s)"
                  f"{'  ' + results[-1]['error'] if 'error' in results[-1] else ''}", flush=True)
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
