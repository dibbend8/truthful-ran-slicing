"""E3 - protection under the full-decision price (experiment_design.md §7 "E3"; PREREG_DRAFT.md H3a, H3c, H3d; H3b is
read from the ``prot_full`` learning-tenant design of ``e2_manipulation.py``).

Run from ``code/`` (CPU only, deterministic; ``PYTHONDONTWRITEBYTECODE=1``):

    /home/darpan/anaconda3/bin/python experiments/e3_protection.py --split dev --workers 20
    /home/darpan/anaconda3/bin/python experiments/e3_protection.py --smoke
    /home/darpan/anaconda3/bin/python experiments/e3_protection.py --split dev --plan

``--split test`` is refused until the pre-registration freeze is recorded (``--test-freeze`` names the file). Jobs are
resumable (one file per job, skipped when present). Outputs: ``results/raw/<split>/e3/`` (smoke: ``e3_smoke/``).

What is measured (n = 3, rho = 0.9, d = 2 s; calibration ``n3_default``)
------------------------------------------------------------------------
1. **H3a, certified maximal deviation gain.** For every monotone arm under the full-decision price
   (``--cert-arms``, default: protected net, kappa = 0 net, static weights, equal weights, direct optimizer, direct
   optimizer with true rates, and the D7 kappa-scaled protected maps ``kappa_x<s>`` of E4b, s in {0, 0.5, 2})
   every tenant computes, on every scored interval, its ex-post best response to
   truthful others by the exact cell enumeration of ``pricing.response.best_response`` (all thresholds of the own
   quota, both bracket ends, the truth, 0 and C_i). Per scored tenant-interval: the gain, its numerical certificate
   (twice the largest price error), the allowance ``eps_i = 1e-8 pi_i mu_i Delta`` and two violation flags:
   ``viol`` = gain > max(cert, eps_i) (the convention of ``sim.episode.episode_checks``) and the stricter
   ``viol_strict`` = gain + cert > eps_i. The same run bills the full price and also computes the fixed-priority
   price, so it yields the truthful premium ``T(theta) = p - p_fix`` (H3c) with certified brackets. If any tenant
   deviates (a violation), the truthful run is added as profile ``truthful``. Output ``cert/<arm>/s<seed>/<window>.parquet``.
2. **H3c, gain identity.** For the protected arm on every 4th scored interval and every tenant, reports
   ``z = min(m theta, C)`` for every bandit multiplier m != 1 and the fixed-price best response ``z*_fix``:
   ``G_full = U_full(z) - U_full(theta)`` and ``G_fix`` from ``pricing.response.utilities`` (staircase over
   ``[0, C_i]``), ``T(z)``, ``T(theta)`` from ``pricing.envelope.premium`` (thresholds over ``[0, z]``; an independent
   computation of the full price). Residuals: ``resid_same = G_fix - (G_full + T(z) - T(theta))`` with every term
   from the premium objects (the identity in floating point; H3c asks < 1e-9) and ``resid_indep`` with the
   utilities from the other engine (bounded by the certificates ``err_bound``). Output ``ident/<arm>/s<seed>/...``.
3. **H3d, realized-utility gain.** For ``--h3d-arms`` (default protected, full price), each tenant in turn reports
   ``min(m theta, C)`` with a constant multiplier m in {0.5, 0.67, 0.8, 0.9, 1.1, 1.25, 1.5, 2} on every scored
   interval and truthfully in the warm-up (``e2_manipulation.ScoredMultiplier``; pre-freeze audit HC-7: the
   multiplier and the truthful run enter scoring from the same queue state, as the E2 replays), others truthful.
   Recorded per (window, tenant, m): realized utility ``sum_k (pi_i on_time_i - p_i)`` and
   declared utility over the scored intervals, the honest tenants' realized value. The summary gives the largest
   pooled relative gain over truth per tenant (H3d: <= 1 % of truthful realized utility, otherwise a limitation).
   Output ``h3d/<arm>/s<seed>/<window>.parquet``.

Checkpoints and smoke mode as in ``e2_manipulation.py`` (``--ckpt-dir``, ``--ckpt kind=path``, ``--train-missing``).

Orchestrator decisions in force (binding; code/STATUS.md §0): D1 (GBR x 0.8) and D2 (per-date load normalization)
through the calibration and ``Calibration.window``; D6 (PPO lookahead h = 3) for any net trained here; D4 is applied
in H3d by recording the honest tenants' realized value next to the deviator's utility. D3, D5, D7 do not change E3.
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

import e2_manipulation as X  # noqa: E402  (shared plumbing; sets threads and sys.path)
import e4_cost as E4  # noqa: E402  (the D7 kappa-scale arms)
from slicemech import io as sio  # noqa: E402

#: Monotone arms certified for H3a. Since the pre-freeze audit (HC-11) they include the D7 kappa-scaled protected
#: maps of E4b (``kappa_x0``, ``kappa_x0.5``, ``kappa_x2``; s = 1 is ``prot_full``), which are monotone and billed
#: with the full-decision price like the other monotone arms.
CERT_ARMS = ("prot_full", "kappa0_full", "static_full", "equal_full", "direct_full", "directtrue_full",
             *E4.e4b_arms())
IDENT_ARMS = ("prot_full",)
H3D_ARMS = ("prot_full",)
H3D_MULTS = (0.5, 0.67, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0)
H3D_LIMIT = 0.01          # 1 % of truthful realized utility

CERT_COLS = ("k", "tenant", "cls", "theta_adm", "report", "quota", "w", "p", "p_lo", "p_hi", "p_fix", "p_fix_lo",
             "p_fix_hi", "T", "T_lo", "T_hi", "price", "V_declared", "V_report", "realized_value", "on_time_Mbit",
             "dropped_Mbit", "br_z_star", "br_gain", "br_cert", "br_U_truth", "br_exact")


# ============================================================================================ job bodies
E3_MODULES = ("e2_manipulation.py", "e3_protection.py")    # code fingerprint of E3 jobs (round-3 review)

def job_e3_cert(arm: str, seed: int, date: str, offset: int, out: str):
    """H3a + H3c premiums: every tenant best-responds (full price) on every scored interval against truthful others."""
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    spec = X.spec_of(seed)
    cal, mech = X._W["cal"], X._W["mech"]
    w = X.get_window(date, offset)
    ctrl, pr = X.make_arm(arm, spec)
    if pr.charge != "full":
        raise ValueError(f"H3a arm {arm} must bill the full-decision price")
    t0 = time.perf_counter()
    pols = [X.ScoredBestResponder("full", w.warmup) for _ in range(w.n)]
    df = E.run_window(w, ctrl, pr, pols, X.D, mech, cal.shape, price_columns="all")
    sec = time.perf_counter() - t0
    chk = E.episode_checks(df, w.contract, mech)
    sc = df["scored"].to_numpy()
    deviated = bool(np.any(df["report"].to_numpy()[sc] != df["theta_adm"].to_numpy()[sc]))
    tabs = {"all_br": df}
    if deviated:
        tabs["truthful"] = E.run_window(w, ctrl, pr, [Truthful() for _ in range(w.n)], X.D, mech, cal.shape,
                                        price_columns="all")
    parts = []
    for prof, t in tabs.items():
        x = t.loc[:, list(CERT_COLS) + ["scored"]].copy()
        x.insert(0, "profile", prof)
        parts.append(x)
    x = pd.concat(parts, ignore_index=True)
    ct = w.contract
    ti = x["tenant"].to_numpy()
    x["eps"] = X.EPS_REL * ct.pi[ti] * ct.mu[ti] * mech.Delta
    g, c = x["br_gain"].to_numpy(), x["br_cert"].to_numpy()
    has = np.isfinite(g)
    x["viol"] = has & (g > np.maximum(np.where(np.isfinite(c), c, 0.0), x["eps"].to_numpy()))
    x["viol_strict"] = has & (g + np.where(np.isfinite(c), c, 0.0) > x["eps"].to_numpy())
    x.insert(0, "seed", -1 if X.seed_free(arm) else int(seed))
    x.insert(0, "arm", arm)
    x.insert(0, "date", w.date)
    x.insert(0, "window", w.id)
    x["warmup"] = np.int16(w.warmup)
    x["deviated"] = deviated
    x["seconds"] = sec
    x["chk_conservation_max"] = chk["conservation_max"]
    x["chk_cert_violation_p"] = chk["cert_violation_p"]
    x["chk_cert_violation_p_fix"] = chk["cert_violation_p_fix"]
    x.attrs = {}
    X.write_parquet(X.f32(x, ("theta_adm", "w")), Path(out))


def job_e3_ident(arm: str, seed: int, date: str, offset: int, out: str, limit: int | None = None):
    """H3c gain identity at multiplier reports and at the fixed-price best response (module docstring, item 2)."""
    from slicemech.model.shapes import stack_shape, value
    from slicemech.pricing.envelope import premium
    from slicemech.pricing.response import best_response, utilities
    from slicemech.tenants.bandits import MULTIPLIERS
    spec = X.spec_of(seed)
    cal, mech = X._W["cal"], X._W["mech"]
    shape = cal.shape
    w = X.get_window(date, offset)
    ctrl, _ = X.make_arm(arm, spec)
    ct = w.contract
    beta, rho = stack_shape(shape)
    th = np.minimum(np.asarray(w.theta, float), ct.C)
    ks = X.scored_every(w.warmup, w.K, X.BR_EVERY, limit)
    rules = X.interval_rules(w, ctrl, ks, mech, shape)
    mults = [float(m) for m in MULTIPLIERS if m != 1.0]
    rows = []
    t0 = time.perf_counter()
    for k in ks:
        rule, r = rules[int(k)]
        for i in range(ct.n):
            t_i = float(th[k, i])
            opp = np.delete(th[k], i)
            kw = dict(ct=ct, shape=shape, mech=mech, r=r)
            brf = best_response(i, t_i, opp, rule, price="fixed", **kw)
            tags = ["truth"] + [f"x{m:g}" for m in mults] + ["br_fixed"]
            zs = np.array([t_i] + [min(m * t_i, float(ct.C[i])) for m in mults] + [float(brf.z_star)])
            U_full = utilities(i, t_i, opp, rule, zs, "full", **kw)
            U_fix = utilities(i, t_i, opp, rule, zs, "fixed", **kw)

            def prof(u):
                z = th[k].copy()
                z[i] = u
                return z

            prem = [premium(i, prof(u), rule, ct, shape, mech, r) for u in zs]
            args = (r[i], ct.pi[i], beta[i], rho[i], mech.Delta)
            Vt = float(value(prem[0].full.quota, t_i, *args))
            for j in range(1, len(zs)):
                Pz, Pt = prem[j], prem[0]
                Vz = float(value(Pz.full.quota, t_i, *args))
                g_full_s = (Vz - Pz.full.p) - (Vt - Pt.full.p)
                g_fix_s = (Vz - Pz.fixed.p) - (Vt - Pt.fixed.p)
                dT = Pz.T - Pt.T
                G_full, G_fix = U_full[j] - U_full[0], U_fix[j] - U_fix[0]
                rows.append(dict(
                    k=int(k), tenant=i, cls=ct.cls[i], tag=tags[j], theta=t_i, z=float(zs[j]),
                    multiple=float(zs[j] / t_i) if t_i > 0 else np.nan, T_theta=Pt.T, T_z=Pz.T, dT=dT,
                    p_theta=Pt.full.p, p_fix_theta=Pt.fixed.p, G_full=float(G_full), G_fix=float(G_fix),
                    resid_same=float(g_fix_s - (g_full_s + dT)), resid_indep=float(G_fix - (G_full + dT)),
                    err_bound=float(2 * (Pz.full.err + Pt.full.err + Pz.fixed.err + Pt.fixed.err)),
                    q_theta=Pt.full.quota, q_z=Pz.full.quota, quota_engines_agree=bool(Pz.full.quota == Pz.fixed.quota),
                    brf_gain=float(brf.gain), brf_exact=bool(brf.exact)))
    df = pd.DataFrame(rows)
    df.insert(0, "seed", -1 if X.seed_free(arm) else int(seed))
    df.insert(0, "arm", arm)
    df.insert(0, "date", w.date)
    df.insert(0, "window", w.id)
    df["seconds"] = time.perf_counter() - t0
    X.write_parquet(df, Path(out))


def job_e3_h3d(arm: str, seed: int, date: str, offset: int, out: str):
    """H3d: realized-utility gain of constant multipliers (one deviating tenant, truthful in the warm-up; others
    truthful)."""
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    spec = X.spec_of(seed)
    cal, mech = X._W["cal"], X._W["mech"]
    w = X.get_window(date, offset)
    ctrl, pr = X.make_arm(arm, spec)
    ct = w.contract
    rows = []
    t0 = time.perf_counter()

    def sums(pols):
        df = E.run_window(w, ctrl, pr, pols, X.D, mech, cal.shape, record=False, price_columns="billed")
        sc = df[df["scored"]]
        return sc.groupby("tenant")[["utility_realized", "utility_declared", "realized_value", "on_time_Mbit",
                                      "price"]].sum(), int(sc["k"].nunique())

    base, nk = sums([Truthful() for _ in range(ct.n)])
    for i in range(ct.n):
        rows.append(dict(tenant=i, m=1.0, **_row(base, i, nk)))
        for m in H3D_MULTS:
            pols = [Truthful() for _ in range(ct.n)]
            pols[i] = X.ScoredMultiplier(m, w.warmup)
            s, _ = sums(pols)
            rows.append(dict(tenant=i, m=float(m), **_row(s, i, nk)))
    df = pd.DataFrame(rows)
    df["cls"] = [ct.cls[i] for i in df["tenant"]]
    df.insert(0, "seed", -1 if X.seed_free(arm) else int(seed))
    df.insert(0, "arm", arm)
    df.insert(0, "date", w.date)
    df.insert(0, "window", w.id)
    df["seconds"] = time.perf_counter() - t0
    X.write_parquet(df, Path(out))


def _row(s: pd.DataFrame, i: int, nk: int) -> dict:
    vic = [j for j in s.index if j != i]
    return dict(U_realized=float(s.loc[i, "utility_realized"]), U_declared=float(s.loc[i, "utility_declared"]),
                value_dev=float(s.loc[i, "realized_value"]), on_time_dev=float(s.loc[i, "on_time_Mbit"]),
                price_dev=float(s.loc[i, "price"]), value_victims=float(s.loc[vic, "realized_value"].sum()),
                on_time_victims=float(s.loc[vic, "on_time_Mbit"].sum()), n_intervals=nk)


# ============================================================================================ plan
def e3_jobs(args, root: Path) -> list:
    keys = X.run_keys(args)
    K, warm = (X.SMOKE_K, X.SMOKE_WARMUP) if args.smoke else (200, 40)
    n, mod = 3, "e3_protection"
    jobs = []
    for seed in args.seeds:
        for arm in args.cert_arms:
            if X.seed_free(arm) and seed != args.seeds[0]:
                continue
            for d, o in keys:
                out = root / "cert" / arm / f"s{seed}" / f"{X.wkey(d, o)}.parquet"
                cost = 40.0 * K / 200          # pilot: 18-59 s per (window, arm) at K = 200, load 9-15
                jobs.append(X.Job(f"cert {arm} s{seed} {X.wkey(d, o)}", f"{mod}:job_e3_cert",
                                  dict(arm=arm, seed=seed, date=d, offset=o, out=str(out)), str(out), cost))
        for arm in args.ident_arms:
            for d, o in keys:
                out = root / "ident" / arm / f"s{seed}" / f"{X.wkey(d, o)}.parquet"
                nks = len(X.scored_every(warm, K, X.BR_EVERY, args.ident_limit))
                jobs.append(X.Job(f"ident {arm} s{seed} {X.wkey(d, o)}", f"{mod}:job_e3_ident",
                                  dict(arm=arm, seed=seed, date=d, offset=o, out=str(out), limit=args.ident_limit),
                                  str(out), n * nks * 0.75))
        for arm in args.h3d_arms:
            if X.seed_free(arm) and seed != args.seeds[0]:
                continue
            for d, o in keys:
                out = root / "h3d" / arm / f"s{seed}" / f"{X.wkey(d, o)}.parquet"
                jobs.append(X.Job(f"h3d {arm} s{seed} {X.wkey(d, o)}", f"{mod}:job_e3_h3d",
                                  dict(arm=arm, seed=seed, date=d, offset=o, out=str(out)), str(out),
                                  (1 + n * len(H3D_MULTS)) * 5.0 * K / 200))
    return jobs


# ============================================================================================ summary
def e3_summary(root: Path, args) -> dict:
    out = dict(split=args.split, smoke=args.smoke, decisions=X.DECISIONS)
    cert = X.read_parquets(sorted(root.glob("cert/*/*/*.parquet")))
    if len(cert):
        br = cert[(cert["profile"] == "all_br") & cert["scored"]]
        h3a, h3c = {}, {}
        for arm, g in br.groupby("arm"):
            d = dict(tenant_intervals=int(len(g)), exact_share=float(g["br_exact"].mean()),
                     max_gain=float(g["br_gain"].max()), max_cert=float(g["br_cert"].max()),
                     max_gain_minus_eps=float((g["br_gain"] - g["eps"]).max()),
                     max_gain_plus_cert_minus_eps=float((g["br_gain"] + g["br_cert"] - g["eps"]).max()),
                     violations=int(g["viol"].sum()), violations_strict=int(g["viol_strict"].sum()),
                     windows_deviated=int(g.groupby("window")["deviated"].first().sum()),
                     max_cert_violation_p=float(g["chk_cert_violation_p"].max()),
                     by_class={c: dict(n=int(len(h)), max_gain=float(h["br_gain"].max()),
                                       min_eps=float(h["eps"].min()), max_cert=float(h["br_cert"].max()),
                                       violations=int(h["viol"].sum()))
                               for c, h in g.groupby("cls")})
            h3a[arm] = d
        out["H3a_certified_gain"] = h3a
        out["H3a_all_arms"] = dict(tenant_intervals=int(len(br)), violations=int(br["viol"].sum()),
                                   violations_strict=int(br["viol_strict"].sum()))
        tr = cert[cert["scored"] & ((cert["profile"] == "truthful") | ~cert["deviated"])]
        for arm, g in tr.groupby("arm"):
            if not np.isfinite(g["T"]).any():
                continue
            p = g["p"].to_numpy()
            T = g["T"].to_numpy()
            pw = g.groupby(["window", "date"])["T"].mean().reset_index()
            h3c[arm] = dict(mean_T=float(np.nanmean(T)), p95_T=float(np.nanquantile(T, 0.95)),
                            max_T=float(np.nanmax(T)), share_T_certified_positive=float((g["T_lo"] > 0).mean()),
                            min_T_hi=float(g["T_hi"].min()), premium_share_of_payment_pct=float(
                                100 * np.nansum(T) / np.nansum(p)) if np.nansum(p) > 0 else None,
                            mean_payment=float(np.nanmean(p)), mean_fixed_payment=float(np.nanmean(g["p_fix"])),
                            ci_mean_T=X.boot(pw["T"], pw["date"]),
                            by_class={c: dict(mean_T=float(h["T"].mean()), share_positive=float((h["T_lo"] > 0).mean()))
                                      for c, h in g.groupby("cls")})
        out["H3c_premium"] = h3c
    idn = X.read_parquets(sorted(root.glob("ident/*/*/*.parquet")))
    if len(idn):
        bf = idn[idn["tag"] == "br_fixed"]
        out["H3c_identity"] = dict(
            rows=int(len(idn)), max_abs_resid_same=float(idn["resid_same"].abs().max()),
            n_resid_same_above_1e9=int((idn["resid_same"].abs() > 1e-9).sum()),
            max_abs_resid_indep=float(idn["resid_indep"].abs().max()),
            max_err_bound=float(idn["err_bound"].max()),
            n_resid_indep_above_bound=int((idn["resid_indep"].abs() > idn["err_bound"] + 1e-12).sum()),
            quota_engines_disagree=int((~idn["quota_engines_agree"]).sum()),
            fixed_price_br=dict(n=int(len(bf)), share_positive_fixed_gain=float((bf["G_fix"] > 1e-9).mean()),
                                mean_fixed_gain=float(bf["G_fix"].mean()),
                                max_full_gain_at_fixed_br=float(bf["G_full"].max()),
                                mean_premium_increase=float(bf["dT"].mean())))
    h3d = X.read_parquets(sorted(root.glob("h3d/*/*/*.parquet")))
    if len(h3d):
        res = {}
        for (arm, i), g in h3d.groupby(["arm", "tenant"]):
            base = g[g["m"] == 1.0].set_index("window")
            per_m = {}
            for m, h in g[g["m"] != 1.0].groupby("m"):
                h = h.set_index("window")
                dU = (h["U_realized"] - base.loc[h.index, "U_realized"])
                dUd = (h["U_declared"] - base.loc[h.index, "U_declared"])
                den = base.loc[h.index, "U_realized"].abs().sum()
                per_m[f"{m:g}"] = dict(pooled_rel_gain_pct=float(100 * dU.sum() / den) if den > 0 else None,
                                       share_windows_gain_over_1pct=float(
                                           (dU > H3D_LIMIT * base.loc[h.index, "U_realized"].abs()).mean()),
                                       mean_declared_gain=float(dUd.mean()), max_declared_gain=float(dUd.max()),
                                       d_value_victims=float((h["value_victims"]
                                                              - base.loc[h.index, "value_victims"]).mean()
                                                             / max(1, int(base["n_intervals"].iloc[0]))))
            best = max(per_m.items(), key=lambda kv: -1e300 if kv[1]["pooled_rel_gain_pct"] is None
                       else kv[1]["pooled_rel_gain_pct"])
            res[f"{arm}|tenant{i}|{g['cls'].iloc[0]}"] = dict(
                windows=int(base.shape[0]), max_pooled_rel_gain_pct=best[1]["pooled_rel_gain_pct"],
                argmax_m=best[0], within_1pct=(best[1]["pooled_rel_gain_pct"] or 0.0) <= 100 * H3D_LIMIT,
                per_multiplier=per_m)
        out["H3d_realized_gain"] = res
    sio.write_json(root / "summary.json", out)
    return out


# ============================================================================================ main
def main(argv=None):
    ap = X.common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--cert-arms", nargs="*", default=list(CERT_ARMS))
    ap.add_argument("--ident-arms", nargs="*", default=list(IDENT_ARMS))
    ap.add_argument("--ident-limit", type=int, default=None, help="max identity intervals per window (smoke: 2)")
    ap.add_argument("--h3d-arms", nargs="*", default=list(H3D_ARMS))
    args = ap.parse_args(argv)
    for a in set(args.cert_arms) | set(args.ident_arms) | set(args.h3d_arms):
        kind, price, _ = X.arm_kind(a)
        if price != "full":
            raise SystemExit(f"E3 arms bill the full-decision price; {a} bills {price}")
    root = X.start(args, "e3", ap)               # TEST options are checked before any default is filled in
    if args.smoke:
        args.ident_limit = args.ident_limit or 2
    jobs_plan = e3_jobs(args, root)
    if args.plan:
        X.run_jobs(jobs_plan, workers=args.workers, initargs=(), errors_path=root / "errors.jsonl", plan=True)
        return
    arms = sorted(set(args.cert_arms) | set(args.ident_arms) | set(args.h3d_arms))
    specs = X.resolve_assets(arms, args.seeds, smoke=args.smoke, ckpt_dir=args.ckpt_dir, overrides=args.ckpt,
                             train_missing=args.train_missing, workers=args.workers, n_smoke=args.smoke_windows,
                             test_freeze_path=args.freeze_path)
    X.write_config(root, args, specs, dict(experiment="E3", eps_rel=X.EPS_REL, h3d_multipliers=list(H3D_MULTS),
                                           code_fingerprint=X.code_fingerprint(*E3_MODULES), freeze=args.freeze_path))
    jobs_plan = X.attach_provenance(jobs_plan, specs, split=args.split, smoke=args.smoke, modules=E3_MODULES)
    t0 = time.perf_counter()
    res = X.run_jobs(jobs_plan, workers=args.workers, initargs=(specs, args.split, args.smoke, args.test_freeze),
                     errors_path=root / "errors.jsonl")
    res["wall_s"] = time.perf_counter() - t0
    print(json.dumps(res), flush=True)
    if not args.no_summary:
        s = e3_summary(root, args)
        print(json.dumps({k: v for k, v in s.items() if k != "decisions"}, indent=1, default=str)[:6000])
    if res["failed"]:
        raise SystemExit(f"{res['failed']} job(s) failed; see {root / 'errors.jsonl'}")


if __name__ == "__main__":
    main()
