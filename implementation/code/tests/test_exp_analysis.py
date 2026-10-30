"""Tests of ``experiments/analyze.py`` and ``experiments/figures/*.py`` on synthetic driver outputs.

A synthetic DEV result tree in the drivers' own layouts (E0 ... E7) with known effects is analysed end to end; the
summaries must recover the effects, follow the pre-registered rules (paired windows, seeds averaged within a window,
two-stage cluster bootstrap, Holm, D3 subgroups, D4 victim measures, D5 reference, D7 frontier) and render every paper
figure. Nothing here reads real data; the TEST gate is checked without a freeze.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "experiments"))
sys.path.insert(0, str(CODE / "experiments" / "figures"))

import analyze as A  # noqa: E402
from slicemech.data import puffer  # noqa: E402
from slicemech.eval import metrics as M  # noqa: E402

DATES = puffer.DEV_DATES[:6]
OFFSETS = (60, 14460, 28860, 43260)
LOADS = (0.7, 0.85, 0.95, 1.05)
CLS = ("video", "urllc", "mmtc")
PI = np.array([1.0, 3.0, 0.5])
MU = np.array([25.0, 12.5, 4.2])
NB = 300
H1A_EFFECT = 1.0                      # credits per interval, protected - meas_only at d = 2
DELAY_EFFECT = {1: 0.5, 2: 1.0, 5: 1.5, 15: 3.0}
STATIC_EFFECT, DIRECT_EFFECT, UNPROT_EFFECT = 0.5, 0.2, 0.3
VICTIM_LOSS = 0.3                     # Mbit per interval lost by the honest tenants under unc_fixed best response


def wid(date, off, tag=""):
    return f"dev-{date}-{tag}o{off:05d}-abcd1234"


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ------------------------------------------------------------------------------------------------ synthetic drivers
def _tenant_rows(rng, extra_value: np.ndarray) -> pd.DataFrame:
    on = np.array([20.0, 10.0, 3.0]) + rng.normal(0, 0.05, 3)
    on = on + extra_value / PI
    dr = np.array([0.1, 0.05, 0.02])
    price = np.array([0.4, 0.6, 0.1])
    V = PI * on * 1.02
    return pd.DataFrame(dict(tenant=[0, 1, 2], cls=list(CLS), n_intervals=160, on_time_Mbit=on, dropped_Mbit=dr,
                             quota=[50.0, 40.0, 10.0], price=price, V_declared=V, realized_value=PI * on,
                             utility_declared=V - price, utility_realized=PI * on - price, p=price, p_fix=price * 0.7,
                             T=price * 0.3))


def _with_total(df: pd.DataFrame) -> pd.DataFrame:
    s = df.drop(columns=["tenant", "cls"]).sum(numeric_only=True)
    tot = dict(s)
    tot.update(tenant=-1, cls="all", n_intervals=160)
    return pd.concat([df, pd.DataFrame([tot])], ignore_index=True)


def make_e1(root: Path, rng) -> None:
    for cfg, d in (("default", 2), ("d1", 1), ("d5", 5), ("d15", 15)):
        for date in DATES:
            for off, load in zip(OFFSETS, LOADS):
                window_noise = rng.normal(0, 0.3)
                rows = []
                arms = [("meas_only", s, 0.0) for s in (0, 1)]
                arms += [("protected", s, DELAY_EFFECT[d] + (0.4 if s == 0 else -0.4)) for s in (0, 1)]
                if cfg == "default":
                    arms += [("static", -1, STATIC_EFFECT), ("direct", -1, DIRECT_EFFECT),
                             ("unprotected", 0, H1A_EFFECT + UNPROT_EFFECT),
                             ("unprotected", 1, H1A_EFFECT + UNPROT_EFFECT),
                             ("direct_truerates", -1, DIRECT_EFFECT + 0.1), ("equal", -1, 0.1)]
                for arm, seed, eff in arms:
                    t = _tenant_rows(np.random.default_rng(1000 + off), np.array([0.0, eff + window_noise, 0.0]))
                    t = _with_total(t)
                    t.insert(0, "seed", seed)
                    t.insert(0, "arm", arm)
                    rows.append(t.assign(split="dev", config=cfg, date=date, offset_s=off,
                                         window_id=wid(date, off), offered_load=load, capacity_load=load * 0.95))
                _write(pd.concat(rows, ignore_index=True), root / "e1" / cfg / "windows" / f"{date}_o{off:05d}.parquet")


def make_e0(root: Path) -> None:
    rows, vc = [], []
    for date in DATES:
        for off, load in zip(OFFSETS, LOADS):
            rows.append(dict(split="dev", date=date, offset_s=off, window_id=wid(date, off), offered_load=load,
                             capacity_load=load * 0.95, rg_frac_sumH_gt_B=0.97, rg_frac_all_caps_fit=0.0,
                             rg_frac_weights_change_quota=0.9, rg_mean_weighted_share=0.4, rg_gbr_outage=0.01,
                             rg_intervals=160))
            for i, c in enumerate(CLS):
                x = np.linspace(1, 10, 20)
                y = 0.9 * x
                vc.append(dict(date=date, offset_s=off, window_id=wid(date, off), tenant=i, cls=c, c=1.0, N=20,
                               Sx=x.sum(), Sy=y.sum(), Sxx=(x * x).sum(), Syy=(y * y).sum(), Sxy=(x * y).sum(),
                               n_clipped=0))
    _write(pd.DataFrame(rows), root / "e0" / "default" / "windows.parquet")
    _write(pd.DataFrame(vc), root / "e0" / "default" / "vcal.parquet")
    make_e0_worked(root / "e0" / "default")


def make_e0_worked(d: Path) -> None:
    """E0 worked-interval files (``e0_setup.worked_interval`` layout): a monotone protected staircase, an
    unconstrained one that drops at u = 30, and utilities whose fixed-price gain peaks at 0.5 credits at u = 30."""
    theta, cap = 10.0, 50.0
    u = np.r_[np.linspace(0, cap, 51), theta]
    rows = []
    for mp in ("protected", "unconstrained"):
        q = np.floor(u / 2.0) if mp == "protected" else np.where(u < 30, np.floor(u / 2.0), 3.0)
        uf = -0.01 * (u - theta) ** 2
        ux = uf + (0.5 * np.exp(-((u - 30.0) ** 2)) if mp == "unconstrained" else 0.0)
        rows.append(pd.DataFrame(dict(map=mp, u=u, quota=q, p_full=0.1 * q, p_fix=0.08 * q, V_true=q, V_report=q,
                                      U_full=uf, U_fix=ux, is_truth=np.r_[np.zeros(51, bool), True])))
    _write(pd.concat(rows, ignore_index=True), d / "worked_interval.parquet")
    st = [dict(map="protected", start=2.0 * j, quota=float(j)) for j in range(26)]
    st += [dict(map="unconstrained", start=2.0 * j, quota=float(j)) for j in range(15)]
    st += [dict(map="unconstrained", start=30.0, quota=3.0)]
    _write(pd.DataFrame(st), d / "worked_staircase.parquet")
    q = dict(quota_truth=5.0, p_full_truth=0.5, p_fix_truth=0.4, premium_truth=0.1, fixed_price_grid_gain=0.0,
             full_price_grid_gain=0.0, n_cells=26, weights_truth=[0.3, 0.3, 0.4])
    meta = dict(date=DATES[0], offset_s=OFFSETS[0], window_id=wid(DATES[0], OFFSETS[0]), k=41, tenant=1,
                cls="urllc", theta_adm=theta, theta_raw=theta, C=cap, pi=3.0, r_hat=0.4, n_grid=51,
                protected=dict(q, max_drop_grid=0.0),
                unconstrained=dict(q, fixed_price_grid_gain=0.5, max_drop_grid=11.0))
    (d / "worked_interval.json").write_text(json.dumps(dict(fp="x", meta=meta)))


def _episode(rng, k0=0, K=12, loss=None) -> pd.DataFrame:
    """Per-(k, tenant) episode rows; ``loss`` = (deviator, Mbit lost per honest tenant per interval)."""
    rows = []
    for k in range(K):
        for i in range(3):
            on = [20.0, 10.0, 3.0][i] + 0.1 * math.sin(k + i)
            if loss is not None:
                dev, lost = loss
                on = on + (0.2 if i == dev else -lost)
            dr = 0.05 * (1 + (k % 3 == 0))
            price = 0.3
            V = PI[i] * on
            rows.append(dict(k=k, tenant=i, theta_adm=on, report=on, quota=30.0, w=1 / 3, price=price, V_declared=V,
                             on_time_Mbit=on, dropped_Mbit=dr, realized_value=PI[i] * on, utility_declared=V - price,
                             utility_realized=PI[i] * on - price, backlog_Mbit=0.0))
    return pd.DataFrame(rows)


def make_e2(root: Path, rng) -> dict:
    base = root / "e2"
    eps_rows = []
    for date in DATES:
        for off in OFFSETS:
            for arm, g in (("unc_fixed", 0.5), ("prot_fixed", 0.2), ("unc_usage", 0.05), ("unc_env", 0.0)):
                rows = []
                for k in range(40, 52, 4):
                    for i, c in enumerate(CLS):
                        gain = g * (1.0 if c == "urllc" else 0.1) + rng.normal(0, 0.01)
                        rows.append(dict(window=wid(date, off), date=date, arm=arm, seed=0, k=k, tenant=i, cls=c,
                                         theta=10.0, z_star=15.0 if gain > 0 else 10.0, gain=max(gain, 0.0),
                                         U_truth=5.0, exact=arm.startswith("prot"), cert=1e-9,
                                         eps=1e-8 * PI[i] * MU[i]))
                _write(pd.DataFrame(rows), base / "stage" / arm / "s0" / f"{date}_o{off:05d}.parquet")
            eps_rows.append((date, off))
    # replays on the first two offsets of every date: unc_fixed with honest loss, prot_fixed without
    episodes = {}
    for date in DATES:
        for off in OFFSETS[:2]:
            for arm, lost in (("unc_fixed", VICTIM_LOSS), ("prot_fixed", 0.0)):
                parts = []
                tr = _episode(rng, K=12)
                parts.append(tr.assign(profile="truthful"))
                for i in range(3):
                    parts.append(_episode(rng, K=12, loss=(i, lost)).assign(profile=f"br{i}"))
                rep = pd.concat(parts, ignore_index=True)
                rep = rep.assign(window=wid(date, off), date=date, arm=arm, seed=0, warmup=2)
                rep["deviator"] = rep["profile"].map(lambda p: -1 if p == "truthful" else int(p[2:]))
                for c in ("br_z_star", "br_gain", "br_cert", "br_U_truth"):
                    rep[c] = np.nan
                rep["br_exact"] = False
                _write(rep, base / "replay" / arm / "s0" / f"{date}_o{off:05d}.parquet")
                episodes[(date, off, arm)] = rep
    # non-monotonicity: unconstrained decreases in 1 of 4 states, protected never
    nm = []
    for date in DATES:
        for off in OFFSETS:
            for k in range(40, 44):
                for i, c in enumerate(CLS):
                    for net, dec in (("unconstrained", 2 if k == 40 else 0), ("protected", 0)):
                        nm.append(dict(window=wid(date, off), date=date, net=net, seed=0, k=k, tenant=i, cls=c,
                                       max_drop=3 if dec else 0, n_decreases=dec, q_at_0=0, q_at_C=60,
                                       q_near_theta=30, theta=10.0, nonmono=bool(dec)))
    df = pd.DataFrame(nm)
    for net, g in df.groupby("net"):
        _write(g, base / "nonmono" / net / "s0" / "all.parquet")
    # bandits: protected converges to 1, unc_fixed to 1.5
    for design, end in (("prot_full", 1.0), ("unc_fixed", 1.5), ("prot_fixed", 1.25)):
        E = 80
        m = np.where(np.arange(E)[:, None] < 40, 0.8, end) * np.ones((E, 3))
        meta = json.dumps(dict(design=design, seed=0, algo="exp3", run=0, episodes=E, memo_hits=0, memo_misses=0,
                               seconds=0.0, pool=4))
        p = base / "bandit" / design / "s0" / "exp3_run0.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, arm=np.zeros((E, 3), int), m=m, gain=np.zeros((E, 3)), reward=np.zeros((E, 3)), greedy_m=m,
                 utility=np.zeros((E, 3)), final_prob=np.zeros((3, 9)), arms=np.ones(9), window=np.array(["w"] * E),
                 meta=np.asarray(meta))
    return episodes


def make_e3(root: Path, rng, violate: bool = False) -> None:
    base = root / "e3"
    for arm in ("prot_full", "equal_full"):
        for date in DATES:
            for off in OFFSETS[:2]:
                rows = []
                for k in range(40, 46):
                    for i, c in enumerate(CLS):
                        eps = 1e-8 * PI[i] * MU[i]
                        gain = 0.0
                        if violate and arm == "prot_full" and date == DATES[0] and k == 40 and i == 1:
                            gain = 10 * eps
                        T = 0.1 + 0.01 * i
                        rows.append(dict(window=wid(date, off), date=date, arm=arm,
                                         seed=0 if arm == "prot_full" else -1,
                                         profile="all_br", k=k, tenant=i, cls=c, theta_adm=10.0, report=10.0,
                                         quota=30.0, p=0.5, p_fix=0.5 - T, T=T, T_lo=T - 1e-10, br_z_star=10.0,
                                         br_gain=gain, br_cert=1e-10, br_exact=True, eps=eps, scored=True,
                                         deviated=False))
                _write(pd.DataFrame(rows), base / "cert" / arm / "s0" / f"{date}_o{off:05d}.parquet")
    rows = []
    for date in DATES:
        for k in (40, 44):
            for i, c in enumerate(CLS):
                for m in (0.5, 2.0):
                    dT = 0.3 * (m - 1)
                    rows.append(dict(window=wid(date, OFFSETS[0]), date=date, arm="prot_full", seed=0, k=k, tenant=i,
                                     cls=c, tag=f"x{m:g}", theta=10.0, z=10 * m, multiple=m, T_theta=0.1,
                                     T_z=0.1 + dT, dT=dT, G_full=-0.2 * abs(m - 1), G_fix=-0.2 * abs(m - 1) + dT,
                                     resid_same=1e-13, resid_indep=2e-12, err_bound=1e-9))
    _write(pd.DataFrame(rows), base / "ident" / "prot_full" / "s0" / "all.parquet")
    rows = []
    for date in DATES:
        for i, c in enumerate(CLS):
            for m in (1.0, 0.5, 2.0):
                rows.append(dict(window=wid(date, OFFSETS[0]), date=date, arm="prot_full", seed=0, tenant=i, cls=c,
                                 m=m, U_realized=100.0 + (0.5 if m == 2.0 else -1.0) * (m != 1.0),
                                 U_declared=110.0 - 2.0 * (m != 1.0), value_dev=0, on_time_dev=0, price_dev=0,
                                 value_victims=500.0, on_time_victims=0, n_intervals=160))
    _write(pd.DataFrame(rows), base / "h3d" / "prot_full" / "s0" / "all.parquet")


def make_e4(root: Path, rng) -> None:
    base = root / "e4"
    for arm, eff, pay in (("prot_full", 0.0, 1.0), ("kappa_x0", -0.5, 0.0), ("kappa_x0.5", -0.2, 0.5),
                          ("kappa_x2", 0.1, 2.0), ("equal_full", -0.6, 0.0), ("static_full", -0.3, 0.0),
                          ("direct_full", 0.2, 3.0)):
        for date in DATES:
            for off in OFFSETS:
                ep = _episode(rng, K=4)
                ep["realized_value"] += eff / 3
                ep["price"] = pay / 3
                ep["p"], ep["p_fix"], ep["T"], ep["p_usage"] = pay / 3, pay / 3 * 0.7, pay / 3 * 0.3, np.nan
                ep["cls"] = [CLS[t] for t in ep["tenant"]]
                ep = ep.assign(window=wid(date, off), date=date, arm=arm,
                               seed=-1 if arm in ("equal_full", "static_full", "direct_full") else 0)
                _write(ep, base / "pay" / arm / "s0" / f"{date}_o{off:05d}.parquet")
    h = []
    for date in DATES:
        for off in OFFSETS[:2]:
            for i, c in enumerate(CLS):
                h.append(dict(window=wid(date, off), date=date, arm="unc_fixed", seed=0, deviator=i, cls=c,
                              honest_value_protected=50.0, honest_value_arm_br=50.0 - 0.8,
                              honest_value_arm_truthful=50.0,
                              delta=0.8, delta_pct=1.6))
    _write(pd.DataFrame(h), base / "h4b_br.parquet")
    _write(pd.DataFrame(dict(window=wid(DATES[0], 60), k=range(20), full_ms=np.linspace(5, 9, 20),
                             fixed_ms=np.linspace(4, 8, 20), levels=30)), base / "bench.parquet")


def make_e5(root: Path, rng) -> None:
    jd = root / "e5" / "jobs"
    for date in DATES:
        for seed in (0, 1):
            job = f"{date}__protected__s{seed}"
            pairs, nets = [], []
            for tgt in ("video", "urllc"):
                for a in (0.5, 2.0):
                    for w in range(4):
                        pairs.append(dict(split="dev", date=date, seed=seed, arm="protected", phase="attack",
                                          pipeline="deploy", target=tgt, alpha=a,
                                          window=wid(date, 60 + 10800 * w, "pA-"),
                                          d_utility_declared_target=-1.0, d_utility_realized_target=-0.9,
                                          n_intervals=160))
                        for pipe, g in (("measurement", 0.0), ("stored", 0.4 if a == A.H5B_PRIMARY_ALPHA else -0.2)):
                            pairs.append(dict(split="dev", date=date, seed=seed, arm="protected", phase="later",
                                              pipeline=pipe, target=tgt, alpha=a,
                                              window=wid(date, 60 + 10800 * (w + 4), "pB-"),
                                              d_utility_declared_target=g + rng.normal(0, 0.02) * (pipe == "stored"),
                                              d_utility_realized_target=g, n_intervals=160))
                    for pipe in ("measurement", "stored"):
                        nets.append(dict(split="dev", date=date, seed=seed, arm="protected", pipeline=pipe,
                                         target=tgt, alpha=a, digest="same" if pipe == "measurement" else f"d{a}",
                                         l2_to_ref=0.0 if pipe == "measurement" else 0.3))
            for pipe in ("measurement", "stored"):
                nets.append(dict(split="dev", date=date, seed=seed, arm="protected", pipeline=pipe, target="none",
                                 alpha=1.0, digest="same", l2_to_ref=0.0))
            _write(pd.DataFrame(pairs), jd / f"{job}.pairs.parquet")
            _write(pd.DataFrame(nets), jd / f"{job}.nets.parquet")
            (jd / f"{job}.done.json").write_text("{}")
    (root / "e5" / "checks.json").write_text(json.dumps(dict(h5a_all_bit_identical=True, jobs_done=12)))


def make_e6(root: Path, rng) -> None:
    jd = root / "e6" / "jobs"
    for n in (6, 9):
        for date in DATES:
            job = f"n{n}__{date}__o00060"
            ten, br, tm = [], [], []
            for arm, eff in (("protected", 0.0), ("unprotected", 0.4), ("equal", -0.3)):
                for t in range(n):
                    on = 5.0 + t + (eff / n if t == 0 else 0.0) * n
                    ten.append(dict(split="dev", n=n, date=date, offset_s=60, window=wid(date, 60), arm=arm,
                                    seed=-1 if arm == "equal" else 0, tenant=t, cls=CLS[t % 3], n_intervals=160,
                                    on_time_Mbit=on, dropped_Mbit=0.1, deadline_miss_pct=1.0, quota=106 / n,
                                    price=0.1, V_declared=on, realized_value=on, utility_declared=on - 0.1,
                                    utility_realized=on - 0.1))
            for k in range(40, 60):
                tm.append(dict(split="dev", n=n, date=date, offset_s=60, window=wid(date, 60), arm="protected", seed=0,
                               k=k, scored=True, t_full_ms=10.0 + n + (k - 40) * 0.1, t_fixed_ms=9.0))
                br.append(dict(split="dev", n=n, date=date, offset_s=60, window=wid(date, 60), arm="protected", seed=0,
                               k=k, tenant=0, cls="video", gain=0.0, cert=1e-10, allowance=2e-7, excess=-2e-7,
                               exact=True, misreport=False))
            _write(pd.DataFrame(ten), jd / f"{job}.tenants.parquet")
            _write(pd.DataFrame(br), jd / f"{job}.br.parquet")
            _write(pd.DataFrame(tm), jd / f"{job}.timing.parquet")
            (jd / f"{job}.done.json").write_text("{}")


def make_e7(path: Path) -> None:
    rows = []
    for date in DATES[:4]:
        for arm in ("protected_truthful", "unconstrained_truthful", "unconstrained_br"):
            for t, c in enumerate(CLS):
                hon = arm == "unconstrained_br" and t != 1
                rows.append(dict(case=f"{date}_{arm}", arm=arm, date=date, offset=60, tenant=t, br_tenant=1, cls=c,
                                 nr_delivered_pdb_mbit=10.0 - (1.0 if hon else 0.0),
                                 nr_delivered_mbit=10.5 - (1.0 if hon else 0.0), nr_pdcp_sdu_mbit=11.0,
                                 nr_rlc_drop_mbit=0.1 * (t + 1), nr_delay_p95_ms_median=20.0 * (t + 1),
                                 fluid_raca_on_time=10.0 - (0.8 if hon else 0.0), fluid_nr_on_time=10.0,
                                 fluid_cal_on_time=10.0, corr_delivered_fluid_cal=0.9 + 0.01 * t,
                                 corr_delivered_fluid_nr=0.999, offered_load=0.9, binding_share=0.7,
                                 adjusted_intervals=3))
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "cases.csv", index=False)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(7)
    make_e0(root)
    make_e1(root, rng)
    episodes = make_e2(root, rng)
    make_e3(root, rng)
    make_e4(root, rng)
    make_e5(root, rng)
    make_e6(root, rng)
    make_e7(root / "e7")
    src = {e: root / e for e in A.EXPERIMENTS}
    return dict(root=root, src=src, episodes=episodes)


@pytest.fixture(scope="module")
def results(tree, tmp_path_factory):
    out = tmp_path_factory.mktemp("summary")
    res = A.run_analysis("dev", out, n_boot=NB, src=tree["src"], crossed=False, confine=False, verbose=False)
    return dict(out=out, res=res)


# ------------------------------------------------------------------------------------------------ tests
def test_every_experiment_produces_strict_json_with_letter_macros(results):
    for e in A.EXPERIMENTS + ("holm", "example", "settings"):
        p = results["out"] / f"{e}.json"
        d = json.loads(p.read_text())                                  # strict JSON (no NaN)
        assert "NaN" not in p.read_text()
        for name, spec in d["macros"].items():
            assert name.isalpha(), name
            txt = spec["fmt"].format(spec["value"]) if not isinstance(spec["value"], str) else spec["value"]
            assert txt and spec["unit"] is not None and spec["source"].startswith(e)


def test_settings_macros_from_params_yaml(results):
    m = results["res"]["settings"]["macros"]
    assert m["PpoIters"]["fmt"].format(m["PpoIters"]["value"]) == "400"
    assert m["EfiveFactor"]["value"] == "0.5, 0.7, 1.5 and 2"
    assert m["LoadRho"]["fmt"].format(m["LoadRho"]["value"]) == "0.9"


def test_h1a_recovers_the_paired_effect_with_seeds_averaged(results):
    h = results["res"]["e1"]["hypotheses"]["H1a"]
    assert h["n_dates"] == len(DATES) and h["n_windows"] == len(DATES) * len(OFFSETS)
    assert h["estimate"] == pytest.approx(H1A_EFFECT, abs=1e-9)      # seed noise +-0.4 cancels within the window
    assert h["lo"] <= H1A_EFFECT <= h["hi"] and h["lo"] > 0
    assert h["dates_positive"] == len(DATES) and h["decision"]["rule_met"] and "supported" not in h["decision"]
    assert "EoneSupported" not in results["res"]["e1"]["macros"]      # the final verdict lives in holm.json
    v = {r["name"]: r for r in results["res"]["holm"]["tables"]["verdicts"]}
    assert v["H1a"]["supported"] == bool(v["H1a"]["rule_met"] and v["H1a"]["holm_reject"])
    assert h["pct"] == pytest.approx(100 * h["estimate"] / h["mean_b"], rel=1e-9)
    assert h["seeds_a"] == 2 and h["unpaired"] == 0
    m = results["res"]["e1"]["macros"]
    assert m["EoneGain"]["value"] == pytest.approx(H1A_EFFECT) and m["nTestDates"]["value"] == len(DATES)
    assert m["EoneGainCI"]["value"].startswith("[")


def test_h1b_slope_and_delay_table(results):
    e1 = results["res"]["e1"]
    tb = {r["d"]: r["estimate"] for r in e1["tables"]["value_by_delay"]}
    for d, eff in DELAY_EFFECT.items():
        assert tb[float(d)] == pytest.approx(eff, abs=1e-9)
    sl = e1["hypotheses"]["H1b"]["d"]
    x = np.array(list(DELAY_EFFECT)), np.array(list(DELAY_EFFECT.values()))
    assert sl["estimate"] == pytest.approx(np.polyfit(x[0], x[1], 1)[0], rel=1e-9)
    assert sl["nondecreasing_supported"]


def test_h1c_h1d_secondary_two_sided_with_rule_out_and_load_subgroups(results):
    hyp = results["res"]["e1"]["hypotheses"]
    for hid, eff in (("H1c", H1A_EFFECT - STATIC_EFFECT), ("H1d", H1A_EFFECT - DIRECT_EFFECT)):
        r = hyp[hid]
        assert r["alternative"] == "two-sided" and r["role"].startswith("secondary")
        assert r["estimate"] == pytest.approx(eff, abs=1e-9)
        assert r["rules_out"]["advantage_above"] == r["hi"]
        sub = r["load_subgroups"]
        assert sub["cut_points"] == list(A.LOAD_BINS)
        assert [b["n_windows"] for b in sub["bins"]] == [len(DATES), 2 * len(DATES), len(DATES)]
    assert hyp["H1a"]["load_subgroups"]["slope"] is not None


def test_h4a_noninferiority_and_holm_family(results):
    h = results["res"]["e4"]["hypotheses"]["H4a"]
    assert h["estimate"] == pytest.approx(UNPROT_EFFECT, abs=1e-9)
    assert h["rule_met"] == (h["pct_upper_one_sided"] < A.H4A_MARGIN_PCT) and "supported" not in h
    holm = results["res"]["holm"]
    assert holm["hypotheses"]["family"] == list(A.HOLM_FAMILY) == ["H1a", "H2c", "H4a", "H5b"]
    tb = pd.DataFrame(holm["tables"]["holm"])
    raw = dict(zip(tb["name"], tb["p"]))
    assert raw["H4a"] == pytest.approx(h["noninferiority_p"])
    assert (tb["p_holm"] >= tb["p"] - 1e-15).all() and holm["hypotheses"]["complete"]


def test_h2a_h2b_and_d4_victims(tree, results):
    e2 = results["res"]["e2"]
    tb = pd.DataFrame(e2["tables"]["br_by_class"])
    u = tb[(tb["arm"] == "unprotected") & (tb["cls"] == "urllc")].iloc[0]
    assert u["mean_gain"] == pytest.approx(0.5, abs=0.01) and u["lo"] > 0
    assert e2["hypotheses"]["H2a"]["supported"]
    h2b = e2["hypotheses"]["H2b"]
    assert h2b["share_states_nonmonotone"] == pytest.approx(0.25) and h2b["protected_negative_control"] == 0
    h2c = e2["hypotheses"]["H2c"]
    assert h2c["estimate"] == pytest.approx(-2 * VICTIM_LOSS, abs=1e-9) and h2c["rule_met"]
    assert h2c["n_seeds"] == 1 and "seed_note" in h2c
    assert e2["hypotheses"]["H2c_all"]["fixed_price"]["d_on_time_victims"]["estimate"] == pytest.approx(0, abs=1e-12)


def test_victim_table_equals_eval_metrics_victim_effects(tree):
    notes = []
    T = A.load_all("dev", False, notes, tree["src"])
    VT = A.victim_table(T["windows"])
    date, off = DATES[0], OFFSETS[0]
    rep = tree["episodes"][(date, off, "unc_fixed")]
    tr = rep[rep["profile"] == "truthful"]
    for i in range(3):
        ve = M.victim_effects(tr, rep[rep["profile"] == f"br{i}"], i, warmup=2)
        row = VT[(VT["unit"] == f"{date}|o{off}") & (VT["arm"] == "unprotected") & (VT["deviator"] == i)].iloc[0]
        for k in ("d_on_time_victims", "d_miss_pct_victims", "d_value_victims", "dW_realized"):
            assert row[k] == pytest.approx(ve[k], abs=1e-9), k


def test_h3a_certificate_and_violation_detection(tree, tmp_path):
    res, by = A.h3a(A.load_all("dev", False, [], tree["src"])["br"].query("exp == 'e3'"))
    assert res["holds"] and res["violations"] == 0 and res["tenant_intervals"] == 2 * len(DATES) * 2 * 6 * 3
    bad = tmp_path / "raw"
    make_e3(bad, np.random.default_rng(0), violate=True)
    r2, _ = A.h3a(A.load_all("dev", False, [], {"e3": bad / "e3"})["br"])
    assert not r2["holds"] and r2["violations"] == 2                            # one per window of DATES[0]
    # PREREG §6 (evidence check 2026-10-20 item 7): the verdict counts gain + certificate > eps_i; the earlier
    # allowance gain > max(cert, eps_i) is a reported check (the same here: gain = 10 eps, cert = 1e-10)
    assert r2["violations"] == r2["violations_strict"] == r2["violations_max_rule"] == 2
    assert res["violations_strict"] == res["violations_max_rule"] == 0 and "certified gain" in res["rule"]


def test_e3_premium_identity_gain_by_mult_and_h3d(results):
    e3 = results["res"]["e3"]
    assert e3["hypotheses"]["H3c"]["premium"] == pytest.approx(0.11, abs=1e-9)
    assert e3["hypotheses"]["H3c_identity"]["holds"]
    gm = pd.DataFrame(e3["tables"]["gain_by_mult"])
    full = gm[(gm["arm"] == "protected") & (gm["mult"] == 2.0)]
    fix = gm[(gm["arm"] == "fixed_price") & (gm["mult"] == 2.0)]
    assert (full["mean_gain"] < 0).all() and (fix["mean_gain"] > 0).all()
    assert e3["hypotheses"]["H3d"]["max_pct"] == pytest.approx(0.5) and e3["hypotheses"]["H3d"]["holds"]
    assert e3["hypotheses"]["H3b"]["holds"]
    assert e3["macros"]["EthreeMult"]["value"] == "Video 1, URLLC-like 1, mMTC-like 1"


def test_e4_h4b_payments_and_d7_frontier(results):
    e4 = results["res"]["e4"]
    assert e4["hypotheses"]["H4b"]["unprotected"]["estimate"] == pytest.approx(0.8, abs=1e-9)
    fr = pd.DataFrame(e4["tables"]["frontier"])
    prot = fr[fr["arm"] == "protected"].sort_values("kappa_scale")
    assert prot["kappa_scale"].tolist() == [0.0, 0.5, 1.0, 2.0] and set(prot["source"]) == {"e4"}
    assert prot["payment"].tolist() == pytest.approx([0.0, 0.5, 1.0, 2.0])
    assert {"equal", "static", "direct"} <= set(fr["arm"])
    pay = pd.DataFrame(e4["tables"]["payments"])
    assert pay[pay["cls"] == "all"]["premium_pct_of_payment"].iloc[0] == pytest.approx(30.0)
    assert e4["tables"]["pricing_time_n3"][0]["p95_ms"] == pytest.approx(np.quantile(np.linspace(5, 9, 20), 0.95))


def test_e5_h5a_h5b_reference_and_cost(results):
    e5 = results["res"]["e5"]
    assert e5["hypotheses"]["H5a"]["holds"] and e5["hypotheses"]["H5a_later_gain_max_abs"] == 0.0
    h = e5["hypotheses"]["H5b"]
    # the synthetic later gain is +0.4 at the pre-registered primary alpha (A.H5B_PRIMARY_ALPHA) and -0.2 elsewhere
    assert h["estimate"] == pytest.approx(0.4, abs=0.02) and h["alpha"] == A.H5B_PRIMARY_ALPHA and h["lo"] > 0
    if A.H5B_SIGN is None:      # before the freeze: no rule; the DEV pilot suggests the sign of the estimate
        assert h["rule_met"] is None and h["dev_sign_suggestion"] == 1
    else:                       # after the freeze (orchestrator 2026-10-20: H5B_SIGN = +1 from the DEV estimation half)
        assert h["rule_met"] is (A.H5B_SIGN == 1)
    assert A.h5b_decision(h, sign=1)["rule_met"] and A.h5b_decision(h, sign=-1)["contrary"]
    tb = pd.DataFrame(e5["tables"]["later_gain"])
    # pre-freeze audit NUM-08: attack_change is signed (negative = cost), cost = -attack_change (positive = cost)
    assert np.allclose(tb["attack_change"].astype(float), -1.0) and np.allclose(tb["cost"].astype(float), 1.0)
    m = e5["macros"]
    assert m["EfiveCost"]["value"] == pytest.approx(1.0) and m["EfiveAttackChange"]["value"] == pytest.approx(-1.0)
    assert m["EfiveNet"]["value"] == pytest.approx(h["estimate"] - 1.0, abs=1e-9)
    # repay: cost 1.0 per retraining interval, later gain ~0.4 per evaluation interval, 4 windows in each phase
    assert m["EfiveRepay"]["value"] == pytest.approx(1.0 / h["estimate"], rel=1e-9)
    assert m["EfiveRepayWindows"]["value"] == pytest.approx(4.0 / h["estimate"], rel=1e-9)
    assert {r["seed"] for r in e5["tables"]["h5b_by_seed"]} == {0, 1} and "seed_spread" in h
    assert e5["macros"]["EfiveDiffParams"]["value"] == 100.0


def test_e6_timing_cost_and_e7(results):
    e6 = results["res"]["e6"]
    t9 = [r for r in e6["tables"]["pricing_time"] if r["n"] == 9][0]
    assert t9["p95_ms"] == pytest.approx(np.quantile(10 + 9 + np.arange(20) * 0.1, 0.95))
    assert e6["hypotheses"]["H6a_time"]["holds"] and e6["hypotheses"]["H6a_gain"]["holds"]
    cost = {r["n"]: r for r in e6["tables"]["cost_by_n"]}
    assert cost[9]["estimate"] == pytest.approx(0.4) and cost[3]["source"] == "e1"
    e7 = results["res"]["e7"]
    h7a = e7["hypotheses"]["H7a"]
    assert h7a["sign_agrees"] == 4 and h7a["within_2x"] == 4 and h7a["sign_criterion"] and h7a["magnitude_criterion"]
    assert h7a["holds"]
    assert e7["macros"]["EsevenVictim"]["value"] == pytest.approx(2.0)
    assert e7["hypotheses"]["H7c"]["holds"] and e7["hypotheses"]["H7c"]["min"] == pytest.approx(0.9)
    assert e7["hypotheses"]["H7c_in_sample_decomposition"]["min"] == pytest.approx(0.999)


def test_e0_regime_and_vcal(results):
    e0 = results["res"]["e0"]
    assert e0["macros"]["EzeroNeedShare"]["value"] == pytest.approx(97.0)
    assert e0["hypotheses"]["regime_targets_met"]
    assert all(abs(r["r2"] - 1.0) < 1e-12 for r in e0["tables"]["vcal"])


def test_e0_worked_interval_reaches_the_summary(results):
    """Integrator: E0's worked interval (raw files) is carried into e0.json for fig:worked; gains are relative to
    truth under each design's billed price (full for Protected, fixed-priority for the Unconstrained net)."""
    e0 = results["res"]["e0"]
    mac, tab = e0["macros"], e0["tables"]
    assert mac["EzeroWorkedFixGainUnc"]["value"] == pytest.approx(0.5)
    assert mac["EzeroWorkedDropUnc"]["value"] == 11 and mac["EzeroWorkedDropProt"]["value"] == 0
    assert mac["EzeroWorkedClass"]["value"] == "URLLC-like" and e0["hypotheses"]["worked_protected_monotone"]
    assert {r["map"] for r in tab["worked_meta"]} == {"protected", "unconstrained"}
    g = {(r["map"], r["u"]): r["gain_billed"] for r in tab["worked_curves"]}
    assert g[("protected", 10.0)] == pytest.approx(0.0) and max(v for (m, _), v in g.items() if m == "protected") <= 0
    assert g[("unconstrained", 30.0)] == pytest.approx(0.5 - 0.01 * 400)
    assert len(tab["worked_staircase"]) == 42


def test_example_matches_the_theory_closed_form():
    ex = A.example_summary()
    m = ex["macros"]
    uc = (5 * math.sqrt(13) - 15) / 2
    assert m["ExFixGain"]["value"] == pytest.approx(45 / 56, abs=1e-15)
    assert m["ExPremium"]["value"] == pytest.approx((275 - 70 * math.sqrt(13)) / 28, abs=1e-14)
    assert m["ExQuotaTrue"]["value"] == pytest.approx(65 / 14) and m["ExQuotaLie"]["value"] == pytest.approx(25 / 4)
    cur = pd.DataFrame(ex["tables"]["curves"])
    hi = cur[cur["report"] >= uc]
    assert np.allclose(hi["utility_full"], uc, atol=1e-12)                       # truth is a best response
    assert (cur[cur["report"] < uc - 1e-9]["utility_full"] < uc).all()
    assert np.all(np.diff(cur["utility_fixed"].to_numpy()) >= -1e-15)          # inflation pays under p_fix
    assert np.array_equal(cur["utility_none"], cur["utility_fixed"])
    try:
        sys.path.insert(0, str(CODE.parent / "theory"))
        w = importlib.import_module("witnesses")
        cf = w.e3_closed_form()
        assert float(cf["T"](8)) == pytest.approx(m["ExPremium"]["value"], abs=1e-12)
    except ImportError:
        pytest.skip("theory/witnesses.py (sympy) not importable")


def test_crossed_bootstrap_with_membership_records():
    p = pd.DataFrame(dict(date=[f"d{i // 2}" for i in range(12)], unit=[f"u{i}" for i in range(12)],
                          a=[1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.05, 0.95, 1.15, 0.85, 1.0, 1.1], b=0.0))
    mem = {f"u{i}": {"date": {p["date"][i]: 1.0}, "raca_run": {str(i % 2): 1.0}} for i in range(12)}
    r = A.contrast(p, n_boot=NB, mem=mem, pct=False)
    assert set(r["crossed"]["factors"]) == {"date", "raca_run"} and r["crossed"]["lo"] <= r["estimate"]
    mem.pop("u0")
    assert "crossed" not in A.contrast(p, n_boot=NB, mem=mem, pct=False)        # all or nothing


def test_resumable_skip_and_test_gate(tree, results, monkeypatch, capsys):
    out = results["out"]
    before = (out / "e1.json").stat().st_mtime_ns
    A.run_analysis("dev", out, n_boot=NB, src=tree["src"], crossed=False, confine=False, exps=("e1",))
    assert (out / "e1.json").stat().st_mtime_ns == before                     # digest unchanged: skipped
    from slicemech.data import puffer_fetch

    def refuse(*a, **k):
        raise PermissionError("no freeze")
    monkeypatch.setattr(puffer_fetch, "require_test_read_gate", refuse)
    assert A.main(["--split", "test", "--out", str(out / "never")]) == 3
    assert not (out / "never").exists()


def test_test_dates_outside_a_test_analysis_are_refused(tmp_path):
    tdate = puffer.test_dates()[0]
    df = pd.DataFrame(dict(window=[wid(tdate, 60)], date=[tdate], offset_s=[60], tenant=[0], cls=["video"],
                           arm=["protected"], seed=[0], realized_value=[1.0], on_time_Mbit=[1.0], dropped_Mbit=[0.0],
                           quota=[1.0], price=[0.0], V_declared=[1.0], utility_declared=[1.0]))
    _write(df, tmp_path / "e6" / "jobs" / "x.tenants.parquet")
    (tmp_path / "e6" / "jobs" / "x.done.json").write_text("{}")
    with pytest.raises(PermissionError):
        A.load_all("dev", False, [], {"e6": tmp_path / "e6"})


@pytest.mark.parametrize("name", ["fig_worked_example", "fig_e0_worked", "fig_e1_value", "fig_e2e3_gain",
                                  "fig_e2_manip", "fig_e3_bandits", "fig_e3_protect", "fig_e4_cost",
                                  "fig_e4_frontier", "fig_e5", "fig_e6", "fig_e7_nr"])
def test_figures_render_from_summaries_only(results, tmp_path, name):
    mod = importlib.import_module(name)
    figdir = tmp_path / "fig"
    assert mod.main(["--split", "dev", "--summary-dir", str(results["out"]), "--out", str(figdir)]) == 0
    for ext in ("pdf", "png"):
        p = figdir / f"{name}.{ext}"
        assert p.is_file() and p.stat().st_size > 2000
    assert (figdir / f"{name}.pdf").read_bytes()[:4] == b"%PDF"


def test_figures_skip_cleanly_without_summaries(tmp_path):
    mod = importlib.import_module("fig_e1_value")
    assert mod.main(["--summary-dir", str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    assert not (tmp_path / "o").exists()


def test_figure_colors_and_labels_come_from_controllers_yaml():
    import _common as C
    assert C.style("protected")["color"] == "#2a78d6" and C.style("unprotected")["color"] == "#eb6834"
    assert C.style("protected")["label"] == "Protected (ours)"
    assert C.style("fixed_price")["label"] == "Protected net, fixed price"
    # pre-freeze audit NUM-09: the report-free controller is 'No reports' everywhere; no figure hard-codes a name
    assert C.style("meas_only")["label"] == "No reports" == A.label("meas_only")
    for f in (CODE / "experiments" / "figures").glob("*.py"):
        assert "Measurement-only" not in f.read_text(), f.name
