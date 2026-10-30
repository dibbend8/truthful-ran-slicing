"""E4 - cost of protection (experiment_design.md §7 "E4"; PREREG_DRAFT.md H4b, H4c; E4b frontier, decision D7;
the truthful welfare difference behind H4a is written as a descriptive by-product, E1 owns H4a).

Run from ``code/`` (CPU only, deterministic; ``PYTHONDONTWRITEBYTECODE=1``), after ``e2_manipulation.py`` on the
same split if H4b is wanted (E4 reads E2's replays and bandit end states):

    /home/darpan/anaconda3/bin/python experiments/e4_cost.py --split dev --workers 20
    /home/darpan/anaconda3/bin/python experiments/e4_cost.py --smoke
    /home/darpan/anaconda3/bin/python experiments/e4_cost.py --split dev --plan

``--split test`` is refused until the pre-registration freeze is recorded, and after it until every pre-declared E2
job of the split is done under the current fingerprints (one code version; ``require_e2_complete``). Jobs are
resumable; outputs of recomputed jobs are archived, not deleted (``e2_manipulation.run_jobs``). Outputs:
``results/raw/<split>/e4/`` (smoke: ``e4_smoke/``).

What is measured (n = 3, rho = 0.9, d = 2 s; calibration ``n3_default``)
------------------------------------------------------------------------
1. **Honest payments and surplus (H4c).** Truthful episodes of every arm with both envelope prices computed in each
   interval (``price_columns='all'``): protected net (full price ``p`` and fixed-priority price ``p_fix`` from the
   same run, premium ``T``), unconstrained net (``p_fix``, the uncertified grid envelope ``p`` and the usage tariff
   ``nu Q``), kappa = 0 net, static and equal weights (``T = 0``), direct optimizer (``p``). Per window and tenant:
   payment as % of declared value ``100 sum p / sum V``, surplus ``V - p`` per interval, revenue, realized welfare.
   Output ``pay/<arm>/s<seed>/<window>.parquet`` (one row per scored interval and tenant).
2. **E4b frontier (D7, descriptive).** The trained kappa is scaled by s in {0, 0.5, 1, 2} at deployment
   (``e2_manipulation.ScaledKappa``: each scaled map is a protected map, monotone, billed with the full-decision
   price); truthful welfare against honest payment, with equal, static and direct as reference points. s = 1 is
   the protected arm of item 1. Output ``pay/kappa_x<s>/...``.
3. **Pricing runtime (H4c).** After the pool, single core in the main process: the wall time of
   ``envelope.full_prices`` and ``envelope.fixed_priority_prices`` for all tenants of the protected rule on every
   4th scored interval of ``--bench-windows`` windows (``bench.parquet``; median and p95 per interval).
4. **H4b, honest tenants' realized value (D4).** (a) Best-responding deviator: for each E2 replay window, arm
   (``prot_fixed``, ``unc_fixed``, ``unc_usage``, ``unc_env``) and deviator i, the honest tenants' realized value
   ``sum_{j != i} pi_j on_time_j`` per scored interval under the protected design with truthful reports (item 1)
   minus the same under the arm with tenant i best-responding (E2 ``replay/``); positive = protection serves the
   honest tenants better. (b) Learning deviators: the end state of every E2 bandit run (greedy multiplier profile
   after the last episode) is replayed on the E2 replay windows under its design; tenants whose end multiplier is
   1 are the honest ones. Outputs ``h4b_br.parquet`` and ``learn/<design>/s<seed>/<profile>.parquet``.

Checkpoints, usage tariff and smoke mode as in ``e2_manipulation.py``.

Orchestrator decisions in force (binding; code/STATUS.md §0): D1 and D2 through the calibration and
``Calibration.window``; D4 (H4b on the honest tenants' realized value; total welfare descriptive); D6 (PPO h = 3 for
any net trained here); D7 (the E4b kappa-scale frontier, item 2). D3 and D5 do not change E4.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e2_manipulation as X  # noqa: E402  (shared plumbing)
from slicemech import io as sio  # noqa: E402

PAY_ARMS = ("prot_full", "unc_truth", "kappa0_full", "equal_full", "static_full", "direct_full")
KAPPA_SCALES = (0.0, 0.5, 1.0, 2.0)
H4B_ARMS = ("prot_fixed", "unc_fixed", "unc_usage", "unc_env")
H4B_LEARN_DESIGNS = ("prot_fixed", "unc_fixed", "unc_usage")

PAY_COLS = ("k", "tenant", "cls", "theta_adm", "quota", "w", "V_declared", "p", "p_lo", "p_hi", "p_fix", "p_usage",
            "T", "price", "realized_value", "on_time_Mbit", "dropped_Mbit", "utility_declared")


def e4b_arms(scales=KAPPA_SCALES) -> list[str]:
    """E4b arms other than s = 1 (which is the protected arm ``prot_full``)."""
    return [X.scaled_arm(s) for s in scales if float(s) != 1.0]


# ============================================================================================ job bodies
E4_MODULES = ("e2_manipulation.py", "e4_cost.py")          # code fingerprint of E4 jobs (round-3 review)

def job_e4_pay(arm: str, seed: int, date: str, offset: int, out: str):
    """Truthful episode of one arm with every price column (item 1 / item 2); scored rows only."""
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    spec = X.spec_of(seed)
    cal, mech = X._W["cal"], X._W["mech"]
    w = X.get_window(date, offset)
    ctrl, pr = X.make_arm(arm, spec)
    t0 = time.perf_counter()
    df = E.run_window(w, ctrl, pr, [Truthful() for _ in range(w.n)], X.D, mech, cal.shape, price_columns="all")
    sec = time.perf_counter() - t0
    x = df.loc[df["scored"], list(PAY_COLS)].copy()
    x.insert(0, "seed", -1 if X.seed_free(arm) else int(seed))
    x.insert(0, "arm", arm)
    x.insert(0, "date", w.date)
    x.insert(0, "window", w.id)
    x["controller_key"] = X.arm_kind(arm)[2]
    x["seconds"] = sec
    x["prices_s_per_interval"] = df.attrs["timing_s"]["prices"] / w.K
    x.attrs = {}
    X.write_parquet(X.f32(x, ("theta_adm", "w")), Path(out))


def job_e4_learn(design: str, seed: int, profile: list, windows: list, out: str, n_runs: int = 1):
    """Replay one bandit end-state multiplier profile on the given windows under its design (H4b, learning)."""
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import multiplier_profile
    spec = X.spec_of(seed)
    cal, mech = X._W["cal"], X._W["mech"]
    ctrl, pr = X.make_arm(design, spec)
    rows = []
    for d, o in windows:
        w = X.get_window(d, int(o))
        df = E.run_window(w, ctrl, pr, multiplier_profile(profile), X.D, mech, cal.shape, record=False,
                          price_columns="billed")
        sc = df[df["scored"]]
        nk = int(sc["k"].nunique())
        g = sc.groupby("tenant")
        val, on = g["realized_value"].sum() / nk, g["on_time_Mbit"].sum() / nk
        for i in range(w.n):
            rows.append(dict(window=w.id, date=w.date, design=design, seed=int(seed), tenant=i, cls=w.cls[i],
                             profile_tag=_tag(profile), n_runs=int(n_runs), m_end=float(profile[i]),
                             honest=bool(profile[i] == 1.0), value=float(val[i]), on_time=float(on[i])))
    X.write_parquet(pd.DataFrame(rows), Path(out))


def pricing_bench(args, root: Path, specs: dict) -> pd.DataFrame:
    """Single-core runtime of the protected rule's prices (item 3), in the main process after the pool."""
    from slicemech.pricing.envelope import fixed_priority_prices, full_prices
    keys = X.run_keys(args)[: args.bench_windows]
    X.init_worker(specs, args.split, args.smoke, args.test_freeze)
    cal, mech = X._W["cal"], X._W["mech"]
    seed = args.seeds[0]
    ctrl, _ = X.make_arm("prot_full", X.spec_of(seed))
    rows = []
    for d, o in keys:
        w = X.get_window(d, o)
        th = np.minimum(np.asarray(w.theta, float), w.contract.C)
        ks = X.scored_every(w.warmup, w.K, X.BR_EVERY, 3 if args.smoke else None)
        rules = X.interval_rules(w, ctrl, ks, mech, cal.shape)
        rule, r = rules[int(ks[0])]                       # untimed warm-up (imports, numba, caches)
        full_prices(th[ks[0]], rule, w.contract, cal.shape, mech, r)
        fixed_priority_prices(th[ks[0]], rule, w.contract, cal.shape, mech, r)
        for k in ks:
            rule, r = rules[int(k)]
            t0 = time.perf_counter()
            full = full_prices(th[k], rule, w.contract, cal.shape, mech, r)
            t1 = time.perf_counter()
            fixed_priority_prices(th[k], rule, w.contract, cal.shape, mech, r)
            t2 = time.perf_counter()
            rows.append(dict(window=w.id, k=int(k), full_ms=1e3 * (t1 - t0), fixed_ms=1e3 * (t2 - t1),
                             levels=int(sum(p.n_levels for p in full))))
    df = pd.DataFrame(rows)
    if len(df):
        import os
        df["loadavg_1min"] = os.getloadavg()[0]
        X.write_parquet(df, root / "bench.parquet")
    return df


# ============================================================================================ plan
def e2_root(args) -> Path:
    return Path(args.e2_root) if args.e2_root else X.out_root("e2", args.split, args.smoke, args.tag)


def _tag(profile) -> str:
    return "m" + "_".join(f"{float(v):g}" for v in profile)


def learn_profiles(e2: Path, seed: int, designs) -> dict:
    """{design: {end-state profile: number of runs}} from E2's bandit runs (both algorithms) of checkpoint seed
    ``seed``; the end state is the greedy multiplier vector after the last episode."""
    import collections
    out = {}
    for design in designs:
        prof = collections.Counter()
        for f in sorted((e2 / "bandit" / design / f"s{seed}").glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            prof[tuple(float(v) for v in z["greedy_m"][-1])] += 1
        if prof:
            out[design] = dict(sorted(prof.items()))
    return out


def e4_jobs(args, root: Path) -> list:
    keys = X.run_keys(args)
    rkeys = keys[:1] if args.smoke else X.replay_keys(keys, args.replay_per_date)
    K = X.SMOKE_K if args.smoke else 200
    mod = "e4_cost"
    # pilot (K = 200, load 9-15): seconds per truthful window with every price column
    cost = {"unc_truth": 36.0, "prot_full": 16.0, "kappa0_full": 15.0, "equal_full": 11.0, "static_full": 12.0,
            "direct_full": 11.0}
    jobs = []
    for seed in args.seeds:
        for arm in list(args.pay_arms) + e4b_arms(args.kappa_scales):
            if X.seed_free(arm) and seed != args.seeds[0]:
                continue
            for d, o in keys:
                out = root / "pay" / arm / f"s{seed}" / f"{X.wkey(d, o)}.parquet"
                jobs.append(X.Job(f"pay {arm} s{seed} {X.wkey(d, o)}", f"{mod}:job_e4_pay",
                                  dict(arm=arm, seed=seed, date=d, offset=o, out=str(out)), str(out),
                                  cost.get(arm, 7.0) * K / 200))
        for design, profs in learn_profiles(e2_root(args), seed, args.learn_designs).items():
            for prof, n_runs in profs.items():
                tag = _tag(prof)
                out = root / "learn" / design / f"s{seed}" / f"{tag}.parquet"
                jobs.append(X.Job(f"learn {design} s{seed} {tag}", f"{mod}:job_e4_learn",
                                  dict(design=design, seed=seed, profile=list(prof), windows=[list(k) for k in rkeys],
                                       out=str(out), n_runs=n_runs), str(out),
                                  len(rkeys) * 3.0 * K / 200))
    return jobs


# ============================================================================================ summary
def _honest_value(df: pd.DataFrame, deviator: int) -> float:
    """Honest tenants' realized value per scored interval (credits): sum_{j != i} pi_j on_time_j."""
    nk = df["k"].nunique()
    return float(df.loc[df["tenant"] != deviator, "realized_value"].sum() / nk)


def e4_summary(root: Path, args) -> dict:
    out = dict(split=args.split, smoke=args.smoke, decisions=X.DECISIONS)
    pay = X.read_parquets(sorted(root.glob("pay/*/*/*.parquet")))
    if len(pay):
        pw = []
        for (arm, seed, win, date), g in pay.groupby(["arm", "seed", "window", "date"]):
            nk = g["k"].nunique()
            V = g["V_declared"].sum()
            row = dict(arm=arm, seed=seed, window=win, date=date, W_realized=g["realized_value"].sum() / nk,
                       V_declared=V / nk, surplus_full=(V - g["p"].sum()) / nk if g["p"].notna().all() else np.nan)
            for pc in ("p", "p_fix", "p_usage", "price"):
                s = g[pc]
                row[f"rev_{pc}"] = s.sum() / nk if s.notna().all() else np.nan
                row[f"pay_pct_V_{pc}"] = 100 * s.sum() / V if (s.notna().all() and V > 0) else np.nan
            row["T_mean"] = g["T"].mean()
            for c, h in g.groupby("cls"):
                Vc = h["V_declared"].sum()
                row[f"pay_pct_V_p_{c}"] = 100 * h["p"].sum() / Vc if (h["p"].notna().all() and Vc > 0) else np.nan
                row[f"pay_pct_V_p_fix_{c}"] = (100 * h["p_fix"].sum() / Vc
                                               if (h["p_fix"].notna().all() and Vc > 0) else np.nan)
            pw.append(row)
        pw = pd.DataFrame(pw)
        X.write_parquet(pw, root / "pay_windows.parquet")
        h4c = {}
        for arm, g in pw.groupby("arm"):
            h4c[arm] = {c: (float(g[c].mean()) if g[c].notna().any() else None)
                        for c in g.columns if c not in ("arm", "seed", "window", "date")}
            h4c[arm]["windows"] = int(len(g))
        out["H4c_payments_per_window_means"] = h4c
        # truthful welfare difference unconstrained - protected (E1 owns H4a; descriptive here)
        a = pw[pw["arm"] == "unc_truth"].set_index(["seed", "window"])
        b = pw[pw["arm"] == "prot_full"].set_index(["seed", "window"])
        common = a.index.intersection(b.index)
        if len(common):
            d = pd.DataFrame({"date": b.loc[common, "date"].to_numpy(),
                              "diff": (a.loc[common, "W_realized"] - b.loc[common, "W_realized"]).to_numpy(),
                              "Wp": b.loc[common, "W_realized"].to_numpy()},
                             index=common).groupby(level="window").agg(date=("date", "first"), diff=("diff", "mean"),
                                                                        Wp=("Wp", "mean"))
            out["H4a_descriptive_unc_minus_prot"] = dict(
                credits=X.boot(d["diff"].to_numpy(), d["date"].to_numpy()),
                pct_of_protected=X.boot((100 * d["diff"] / d["Wp"]).to_numpy(), d["date"].to_numpy()),
                note="seed-averaged per window; E1 owns H4a")
        fr = {}
        for s in args.kappa_scales:
            arm = "prot_full" if float(s) == 1.0 else X.scaled_arm(s)
            g = pw[pw["arm"] == arm]
            if len(g):
                fr[f"s={float(s):g}"] = dict(W_realized=float(g["W_realized"].mean()),
                                             pay_pct_V=float(g["pay_pct_V_p"].mean()),
                                             revenue=float(g["rev_p"].mean()), T_mean=float(g["T_mean"].mean()),
                                             windows=int(len(g)))
        for arm in ("equal_full", "static_full", "direct_full"):
            g = pw[pw["arm"] == arm]
            if len(g):
                fr[arm] = dict(W_realized=float(g["W_realized"].mean()), pay_pct_V=float(g["pay_pct_V_p"].mean()),
                               revenue=float(g["rev_p"].mean()), windows=int(len(g)))
        out["E4b_frontier"] = fr
    bench = root / "bench.parquet"
    if bench.exists():
        b = pd.read_parquet(bench)
        out["H4c_pricing_runtime_ms"] = dict(
            intervals=int(len(b)), full_median=float(b["full_ms"].median()), full_p95=float(b["full_ms"].quantile(0.95)),
            fixed_median=float(b["fixed_ms"].median()), fixed_p95=float(b["fixed_ms"].quantile(0.95)),
            loadavg_1min=float(b["loadavg_1min"].iloc[0]))
    # H4b (a): best-responding deviator, from E2 replays
    e2 = e2_root(args)
    rows = []
    if len(pay):
        prot = pay[pay["arm"] == "prot_full"]
        for f in sorted(e2.glob("replay/*/*/*.parquet")):
            rep = pd.read_parquet(f)
            arm = rep["arm"].iloc[0]
            if arm not in args.h4b_arms:
                continue
            seed = int(rep["seed"].iloc[0])
            win = rep["window"].iloc[0]
            warm = int(rep["warmup"].iloc[0])
            pr = prot[(prot["window"] == win) & (prot["seed"] == seed)]
            if not len(pr):
                continue
            sc = rep[rep["k"] >= warm]
            tr = sc[sc["profile"] == "truthful"]
            for prof, g in sc[sc["profile"] != "truthful"].groupby("profile"):
                i = int(prof[2:])
                v_prot = _honest_value(pr, i)
                v_arm = _honest_value(g, i)
                rows.append(dict(window=win, date=rep["date"].iloc[0], arm=arm, seed=seed, deviator=i,
                                 cls=pr[pr["tenant"] == i]["cls"].iloc[0], honest_value_protected=v_prot,
                                 honest_value_arm_br=v_arm, honest_value_arm_truthful=_honest_value(tr, i),
                                 delta=v_prot - v_arm,
                                 delta_pct=100 * (v_prot - v_arm) / v_prot if v_prot > 0 else np.nan))
    if rows:
        h = pd.DataFrame(rows)
        X.write_parquet(h, root / "h4b_br.parquet")
        res = {}
        for (arm, i), g in h.groupby(["arm", "deviator"]):
            cls = g["cls"].iloc[0]
            g = g.groupby(["window", "date"])[["delta", "delta_pct"]].mean().reset_index()   # seeds within window
            res[f"{arm}|deviator{i}|{cls}"] = dict(windows=int(len(g)),
                                                                 delta_credits=X.boot(g["delta"], g["date"]),
                                                                 delta_pct=X.boot(g["delta_pct"], g["date"]))
        pf = h[h["arm"] == "prot_fixed"]
        res["check_prot_fixed_truthful_equals_protected"] = (
            float((pf["honest_value_arm_truthful"] - pf["honest_value_protected"]).abs().max()) if len(pf) else None)
        out["H4b_best_response"] = res
    # H4b (b): learning deviators (bandit end states)
    learn = X.read_parquets(sorted(root.glob("learn/*/*/*.parquet")))
    if len(learn) and len(pay):
        prot = pay[pay["arm"] == "prot_full"]
        nk = prot.groupby(["window", "seed"])["k"].nunique().rename("nk")
        pv = (prot.groupby(["window", "seed", "tenant"])["realized_value"].sum().rename("value_protected")
              .reset_index().merge(nk.reset_index(), on=["window", "seed"]))
        pv["value_protected"] = pv["value_protected"] / pv["nk"]
        j = learn.merge(pv[["window", "seed", "tenant", "value_protected"]], on=["window", "seed", "tenant"],
                        how="left")
        res = {}
        for (design, seed), g in j.groupby(["design", "seed"]):
            per_profile = {}
            for tag, h in g.groupby("profile_tag"):
                h = h.assign(delta=h["value_protected"] - h["value"])
                hon = h[h["honest"]]
                d = dict(n_runs=int(h["n_runs"].iloc[0]), m_end=[float(v) for v in
                                                                  h.sort_values("tenant").groupby("tenant")["m_end"].first()])
                if len(hon):
                    per_w = hon.groupby(["window", "date"])["delta"].sum().reset_index()
                    d.update(honest_tenants=sorted(int(t) for t in hon["tenant"].unique()), windows=int(len(per_w)),
                             delta_honest_credits=X.boot(per_w["delta"], per_w["date"]))
                else:
                    per_w = h.groupby(["window", "date"])["delta"].sum().reset_index()
                    d.update(honest_tenants=[], note="no tenant ends at multiplier 1; delta of all tenants instead",
                             delta_all_credits=X.boot(per_w["delta"], per_w["date"]))
                per_profile[tag] = d
            res[f"{design}|s{seed}"] = per_profile
        out["H4b_learning"] = res
    sio.write_json(root / "summary.json", out)
    return out


# ============================================================================================ main
def require_e2_complete(args, specs: dict) -> dict:
    """E4 reads E2's replays and bandit end states, so on TEST it refuses to start unless every pre-declared E2 job
    is done under the current fingerprints (one code version; ``e2_manipulation.e2_completeness``; pre-freeze audit
    F9, item 35). On TRAIN/DEV an incomplete E2 only gives a warning. Returns the E2 status (recorded in
    ``config.json``)."""
    st = X.e2_completeness(args.split, args.smoke, specs, tag=args.tag, root=args.e2_root)
    if not st["complete"]:
        msg = (f"E2 is incomplete under the current inputs: {st['done']}/{st['jobs']} pre-declared jobs done, "
               f"{st['pending']} pending, {st['stale']} stale, {st['code_versions']} code version(s) in {st['root']}")
        if args.split == "test":
            raise SystemExit(f"refusing to start E4 on TEST: {msg}; run e2_manipulation.py to completion first")
        print(f"[e4] warning: {msg}", flush=True)
    return st


def main(argv=None):
    ap = X.common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--pay-arms", nargs="*", default=list(PAY_ARMS))
    ap.add_argument("--kappa-scales", type=float, nargs="*", default=list(KAPPA_SCALES))
    ap.add_argument("--learn-designs", nargs="*", default=list(H4B_LEARN_DESIGNS))
    ap.add_argument("--h4b-arms", nargs="*", default=list(H4B_ARMS))
    ap.add_argument("--replay-per-date", type=int, default=2)
    ap.add_argument("--bench-windows", type=int, default=8)
    ap.add_argument("--e2-root", default=None, help="E2 output root (default results/raw/<split>/e2[_smoke])")
    args = ap.parse_args(argv)
    root = X.start(args, "e4", ap)               # TEST options are checked before any default is filled in
    if args.smoke:
        args.bench_windows = min(args.bench_windows, 1)
    jobs = e4_jobs(args, root)
    if args.plan:
        X.run_jobs(jobs, workers=args.workers, initargs=(), errors_path=root / "errors.jsonl", plan=True)
        return
    arms = sorted(set(args.pay_arms) | set(e4b_arms(args.kappa_scales)) | set(args.learn_designs) | {"prot_full"})
    specs = X.resolve_assets(arms, args.seeds, smoke=args.smoke, ckpt_dir=args.ckpt_dir, overrides=args.ckpt,
                             train_missing=args.train_missing, workers=args.workers, n_smoke=args.smoke_windows,
                             test_freeze_path=args.freeze_path)
    e2_status = require_e2_complete(args, specs)            # before anything of E4 is written (TEST: refusal)
    X.write_config(root, args, specs, dict(experiment="E4", kappa_scales=list(args.kappa_scales),
                                           e2_root=str(e2_root(args)), code_fingerprint=X.code_fingerprint(*E4_MODULES),
                                           freeze=args.freeze_path, e2_status=e2_status))
    jobs = X.attach_provenance(jobs, specs, split=args.split, smoke=args.smoke, modules=E4_MODULES)
    t0 = time.perf_counter()
    res = X.run_jobs(jobs, workers=args.workers, initargs=(specs, args.split, args.smoke, args.test_freeze),
                     errors_path=root / "errors.jsonl")
    if not (root / "bench.parquet").exists() and args.bench_windows > 0:
        pricing_bench(args, root, specs)
    res["wall_s"] = time.perf_counter() - t0
    print(json.dumps(res), flush=True)
    if not args.no_summary:
        s = e4_summary(root, args)
        print(json.dumps({k: v for k, v in s.items() if k != "decisions"}, indent=1, default=str)[:6000])
    if res["failed"]:
        raise SystemExit(f"{res['failed']} job(s) failed; see {root / 'errors.jsonl'}")


if __name__ == "__main__":
    main()
