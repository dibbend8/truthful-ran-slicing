"""Descriptive result macros added before the pre-registration freeze (orchestrator, 2026-10-20 evening).

experiments/analyze.py: per-class value of reports and the kappa = 0 contrasts (E1), the H2c-cell PRB transfer and
the ceiling share of best responses (E2), the per-seed H4a gap and the learning-tenant H4b macro (E4), the repay
length in minutes and the alpha = 2 later gain (E5), the H7b macros (E7), and H3a's certified-gain rule
(gain + certificate <= eps_i, PREREG §6). Synthetic inputs in the drivers' layouts (the builders of
tests/test_exp_analysis.py where they fit) with known effects; nothing real is read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1]
for _p in (CODE / "experiments", CODE / "tests"):
    sys.path.insert(0, str(_p))

import analyze as A  # noqa: E402
import test_exp_analysis as TA  # noqa: E402

DATES = TA.DATES                                   # six DEV dates
OFFSETS = TA.OFFSETS
LOADS = (0.70, 0.80, 0.95, 1.05)                   # mean 0.875: the video effect's load slope is exactly 1
PI = TA.PI
NB = 300
GAIN = np.array([0.6, 1.0, 0.4])                   # per-class value of reports at the mean load (credits/interval)
KAPPA0 = np.array([0.3, 0.5, 0.2])                 # kappa0 - No reports, per class
SEED_GAP = {0: 0.3, 1: -0.1}                       # Unconstrained - Protected, per training seed (video tenant)
RHO_HIGH_GAIN = 2.5


# ------------------------------------------------------------------------------------------------ synthetic E1
def _e1_rows(extra: np.ndarray, dropped: np.ndarray) -> pd.DataFrame:
    on = np.array([20.0, 10.0, 3.0]) + extra / PI
    t = pd.DataFrame(dict(tenant=[0, 1, 2], cls=list(TA.CLS), n_intervals=160, on_time_Mbit=on, dropped_Mbit=dropped,
                          quota=[50.0, 40.0, 10.0], price=[0.4, 0.6, 0.1], V_declared=PI * on, realized_value=PI * on,
                          utility_declared=PI * on - 0.3, utility_realized=PI * on - 0.3))
    return TA._with_total(t)


def make_e1(root: Path, rng) -> None:
    for cfg in ("default", "rho1.1"):
        for date in DATES:
            for off, load in zip(OFFSETS, LOADS):
                noise = rng.normal(0, 0.3)                       # the same for every arm of the window
                prot = GAIN + np.array([load - 0.875, 0.0, 0.0])
                if cfg == "rho1.1":
                    prot = np.array([RHO_HIGH_GAIN, 0.0, 0.0])
                arms = [("meas_only", s, np.zeros(3), np.array([1.0, 0.5, 0.2])) for s in (0, 1)]
                arms += [("protected", s, prot + np.array([0, 0.4 if s == 0 else -0.4, 0]),
                          np.array([0.5, 0.4, 0.2])) for s in (0, 1)]
                if cfg == "default":
                    arms += [("kappa0", s, KAPPA0.copy(), np.array([0.8, 0.5, 0.2])) for s in (0, 1)]
                    arms += [("unprotected", s, prot + np.array([SEED_GAP[s], 0.4 if s == 0 else -0.4, 0]),
                              np.array([0.5, 0.4, 0.2])) for s in (0, 1)]
                rows = []
                for arm, seed, eff, dr in arms:
                    t = _e1_rows(eff + np.array([0.0, noise, 0.0]), dr)
                    t.insert(0, "seed", seed)
                    t.insert(0, "arm", arm)
                    rows.append(t.assign(split="dev", config=cfg, date=date, offset_s=off,
                                         window_id=TA.wid(date, off), offered_load=load, capacity_load=load * 0.95))
                TA._write(pd.concat(rows, ignore_index=True),
                          root / "e1" / cfg / "windows" / f"{date}_o{off:05d}.parquet")


@pytest.fixture(scope="module")
def e1(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw_e1")
    make_e1(root, np.random.default_rng(3))
    T = A.load_all("dev", False, [], {"e1": root / "e1"})
    cfg = dict(n_boot=NB, split="none")
    return dict(T=T, e1=A.analyze_e1(T, cfg), e4=A.analyze_e4(T, cfg))


def test_e1_per_class_value_of_reports_share_and_kappa0(e1):
    m, tb = e1["e1"]["macros"], e1["e1"]["tables"]
    for c, g in zip(("Video", "Urllc", "Mmtc"), GAIN):
        assert m[f"EoneGain{c}"]["value"] == pytest.approx(g, abs=1e-9)
        assert m[f"EoneGain{c}Lo"]["value"] <= m[f"EoneGain{c}"]["value"] <= m[f"EoneGain{c}Hi"]["value"]
        assert m[f"EoneGain{c}"]["unit"] == "credits per interval"
    assert m["EoneGainVideoLo"]["value"] < m["EoneGainVideoHi"]["value"]          # the video gain varies with load
    assert m["EoneGainVideoShare"]["value"] == pytest.approx(100 * GAIN[0] / GAIN.sum(), abs=1e-9)
    assert m["EoneGainVideoShareLo"]["value"] <= m["EoneGainVideoShare"]["value"] <= m["EoneGainVideoShareHi"]["value"]
    assert m["EoneGainVideoShare"]["unit"].startswith("percent of the total value of reports")
    rows = {r["cls"]: r for r in tb["value_of_reports_by_class"]}
    assert rows["all"]["estimate"] == pytest.approx(m["EoneGain"]["value"], abs=1e-9)
    assert rows["all"]["class_sum"] == pytest.approx(rows["all"]["estimate"], abs=1e-9)   # classes add up to H1a
    assert sum(rows[c]["share_of_total_pct"] for c in TA.CLS) == pytest.approx(100.0, abs=1e-9)
    assert m["EoneKappaZero"]["value"] == pytest.approx((GAIN - KAPPA0).sum(), abs=1e-9)
    assert m["EoneKappaZeroVsFree"]["value"] == pytest.approx(KAPPA0.sum(), abs=1e-9)
    assert m["EoneKappaZero"]["value"] + m["EoneKappaZeroVsFree"]["value"] == pytest.approx(m["EoneGain"]["value"])
    assert "EoneKappaZeroCI" in m and {r["arm_b"] for r in tb["kappa0_contrasts"]} == {"kappa0", "meas_only"}


def test_e1_load_slope_rho_high_and_miss_differences(e1):
    m, tb, h = e1["e1"]["macros"], e1["e1"]["tables"], e1["e1"]["hypotheses"]["H1a"]
    sl = h["load_subgroups"]["slope"]
    assert m["EoneLoadSlope"]["value"] == pytest.approx(sl["estimate"]) == pytest.approx(1.0, abs=1e-9)
    assert m["EoneLoadSlopeLo"]["value"] == pytest.approx(sl["lo"])
    assert m["EoneGainRhoHigh"]["value"] == pytest.approx(RHO_HIGH_GAIN, abs=1e-9)
    assert "EoneGainRhoHighCI" in m
    # independent computation of the per-class miss share (seed-averaged Mbit of the window)
    W = e1["T"]["windows"]
    W = W[(W["exp"] == "e1") & (W["config"] == "default") & (W["tenant"] >= 0)]

    def miss(arm, cls):
        g = W[(W["arm"] == arm) & (W["cls"] == cls)].groupby(["unit", "seed"])[["on_time_Mbit", "dropped_Mbit"]] \
            .sum().groupby("unit").mean()
        return 100 * g["dropped_Mbit"] / (g["on_time_Mbit"] + g["dropped_Mbit"])
    for c, nm in zip(TA.CLS, ("Video", "Urllc", "Mmtc")):
        want = float((miss("protected", c) - miss("meas_only", c)).mean())
        assert m[f"EoneMissDiff{nm}"]["value"] == pytest.approx(want, rel=1e-9, abs=1e-12)
        assert m[f"EoneMissDiff{nm}"]["unit"].startswith("percentage points")
    assert m["EoneMissDiffVideo"]["value"] < 0 and len(tb["miss_diff_by_class"]) == 3


def test_e4_seed_spread(e1):
    m, tb = e1["e4"]["macros"], e1["e4"]["tables"]
    assert m["EfourSeedLo"]["value"] == pytest.approx(min(SEED_GAP.values()), abs=1e-9)
    assert m["EfourSeedHi"]["value"] == pytest.approx(max(SEED_GAP.values()), abs=1e-9)
    by = {r["seed"]: r for r in tb["h4a_by_seed"]}
    assert set(by) == {0, 1} and by[0]["lo"] <= by[0]["estimate"] <= by[0]["hi"]
    assert np.mean([r["estimate"] for r in by.values()]) == pytest.approx(m["EfourGap"]["value"], abs=1e-9)


def test_e4_learning_end_state_macro():
    lt = pd.DataFrame([dict(arm="protected", status="ok", estimate=0.0, lo=-0.1, hi=0.1),
                       dict(arm="unprotected", status="ok", estimate=0.7, lo=0.5, hi=0.9)])
    mac, notes = A.Macros("e4"), []
    A._h4b_learning_macro(lt, mac, notes)
    assert mac.d["EfourHonestLearn"]["value"] == 0.7 and mac.d["EfourHonestLearnLo"]["value"] == 0.5
    assert mac.d["EfourHonestLearnCI"]["value"] == "[0.50, 0.90]" and not notes
    mac2, notes2 = A.Macros("e4"), []
    A._h4b_learning_macro(lt.assign(status="no tenant ends at multiplier 1"), mac2, notes2)
    assert "EfourHonestLearn" not in mac2.d and notes2


# ------------------------------------------------------------------------------------------------ synthetic E2
TRUTH_Q, TRUTH_ON = np.array([50.0, 40.0, 16.0]), np.array([20.0, 10.0, 3.0])


def _replay(dq: float, K: int = 10) -> pd.DataFrame:
    """Profiles truthful and br<i>: the deviator i gains dq PRB (the honest tenants lose dq/2 each), its on-time
    traffic rises by 0.1 dq (1 + 0.5 i) and each honest tenant's falls by 0.05 dq."""
    parts = []
    for prof in ("truthful", "br0", "br1", "br2"):
        dev = None if prof == "truthful" else int(prof[2:])
        for k in range(K):
            for i in range(3):
                q, on = TRUTH_Q[i], TRUTH_ON[i]
                if dev is not None:
                    q += dq if i == dev else -dq / 2
                    on += 0.1 * dq * (1 + 0.5 * dev) if i == dev else -0.05 * dq
                parts.append(dict(profile=prof, k=k, tenant=i, quota=q, on_time_Mbit=on, dropped_Mbit=0.1,
                                  price=0.2, V_declared=PI[i] * on, realized_value=PI[i] * on,
                                  utility_declared=PI[i] * on - 0.2, utility_realized=PI[i] * on - 0.2,
                                  deviator=-1 if dev is None else dev))
    return pd.DataFrame(parts)


@pytest.fixture(scope="module")
def e2(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw_e2")
    dqs = {}
    for j, date in enumerate(DATES):
        for o, off in enumerate(OFFSETS[:2]):
            dq = 3.0 + (j + o) % 3
            dqs[(date, off)] = dq
            rep = _replay(dq).assign(window=TA.wid(date, off), date=date, arm="unc_fixed", seed=0, warmup=2)
            TA._write(rep, root / "e2" / "replay" / "unc_fixed" / "s0" / f"{date}_o{off:05d}.parquet")
    T = A.load_all("dev", False, [], {"e2": root / "e2"})
    return dict(dqs=dqs, res=A.analyze_e2(T, dict(n_boot=NB, split="none")))


def test_e2_prb_transfer_of_the_h2c_cell(e2):
    m, tb = e2["res"]["macros"], e2["res"]["tables"]
    dq = np.array(list(e2["dqs"].values()))
    assert m["EtwoPrbMoved"]["value"] == pytest.approx(dq.mean(), abs=1e-9)
    assert m["EtwoPrbMovedLo"]["value"] <= m["EtwoPrbMoved"]["value"] <= m["EtwoPrbMovedHi"]["value"]
    assert m["EtwoDevMbitPerPrb"]["value"] == pytest.approx(0.15, abs=1e-9)         # 0.1 x mean(1 + 0.5 i)
    assert m["EtwoVictimMbitPerPrb"]["value"] == pytest.approx(0.1, abs=1e-9)       # two honest tenants x 0.05
    honest_truth = np.mean([TRUTH_ON.sum() - TRUTH_ON[i] for i in range(3)])
    assert m["EtwoVictimLossPct"]["value"] == pytest.approx(100 * 0.1 * dq.mean() / honest_truth, abs=1e-9)
    # consistent with the H2c primary (EtwoVictimLoss = honest on-time Mbit lost, roles averaged per window)
    assert m["EtwoVictimLossPct"]["value"] == pytest.approx(100 * m["EtwoVictimLoss"]["value"] / honest_truth,
                                                            abs=1e-9)
    assert m["EtwoVictimLossPct"]["unit"].startswith("percent of the honest tenants' truthful on-time Mbit")
    rows = {r["deviator_role"]: r for r in tb["h2c_transfer"]}
    assert set(rows) == {"pooled", "video", "urllc", "mmtc"}
    assert rows["mmtc"]["dev_mbit_per_prb"] == pytest.approx(0.1 * 2.0, abs=1e-9)
    assert any("EtwoCeilingShare" in n for n in e2["res"]["notes"])                  # no stage rows here
    assert "EtwoAllLoss" not in m


def test_e2_ceiling_share_per_class(monkeypatch):
    C = np.array([100.0, 50.0, 30.0])
    rows = []
    for date in DATES:
        for off in OFFSETS[:2]:
            for k in range(8):
                for i, c in enumerate(TA.CLS):
                    z = 0.5 * C[i]
                    if c == "video" and k % 2 == 0:
                        z = C[i]                                         # at the ceiling: 1/2 of the rows
                    if c == "mmtc" and k == 0:
                        z = C[i] * (1 - 1e-12)                           # within the tolerance: 1/8
                    if c == "mmtc" and k == 1:
                        z = C[i] * (1 - 1e-6)                            # outside it
                    rows.append(dict(arm="unprotected", n=3, cls=c, tenant=i, theta=0.4 * C[i], z_star=z, date=date,
                                     unit=f"{date}|o{off}", seed=0))
    BR = pd.DataFrame(rows)
    monkeypatch.setattr(A, "_ceilings", lambda n: C)
    cs = A.ceiling_share(BR, "unprotected", n_boot=NB).set_index("cls")
    assert cs.loc["video", "share_pct"] == pytest.approx(50.0) and cs.loc["urllc", "share_pct"] == 0.0
    assert cs.loc["mmtc", "share_pct"] == pytest.approx(12.5)
    assert cs.loc["video", "share_inflated_to_ceiling_pct"] == pytest.approx(50.0)
    assert cs.loc["mmtc", "share_within_1pct_of_ceiling"] == pytest.approx(25.0)
    cs2 = A.ceiling_share(BR.assign(C=C[BR["tenant"]]), "unprotected", n_boot=NB).set_index("cls")  # own C column
    assert cs2["share_pct"].to_dict() == cs["share_pct"].to_dict()
    mac, notes, tables = A.Macros("e2"), [], {}
    A._e2_extra(None, A._units(BR), NB, mac, tables, notes)
    assert mac.d["EtwoCeilingShareVideo"]["value"] == pytest.approx(50.0)
    assert mac.d["EtwoCeilingShareMmtc"]["value"] == pytest.approx(12.5)
    assert mac.d["EtwoCeilingShareUrllc"]["value"] == 0.0 and "EtwoCeilingShareVideoCI" in mac.d
    assert mac.d["EtwoCeilingShareVideo"]["unit"].startswith("percent of the class's best responses")


# ------------------------------------------------------------------------------------------------ E3: H3a rule
def test_h3a_uses_the_certified_gain_rule():
    base = dict(price="full", arm="protected", exact=True, n=3, cls="video", seed=0, exp="e3")
    BR = pd.DataFrame([dict(base, gain=0.0, cert=3e-8, eps=2e-8),      # certificate above eps: strict violation only
                       dict(base, gain=1e-8, cert=1e-9, eps=2e-8),     # within eps under both rules
                       dict(base, gain=5e-8, cert=1e-9, eps=2e-8)])    # violation under both rules
    res, by = A.h3a(BR)
    assert res["violations"] == res["violations_strict"] == 2 and res["violations_max_rule"] == 1
    assert res["holds"] is False and "certified gain" in res["rule"]
    assert by.iloc[0]["violations"] == 2 and by.iloc[0]["violations_max_rule"] == 1
    assert res["max_certified_gain_over_eps"] == pytest.approx((5e-8 + 1e-9) / 2e-8)
    ok, _ = A.h3a(BR.iloc[[1]])
    assert ok["holds"] is True and ok["violations"] == 0
    out = A.analyze_e3(dict(br=BR), dict(n_boot=NB))
    m = out["macros"]
    assert m["EthreeViolationsStrict"]["value"] == 2 == m["EthreeViol"]["value"]
    assert m["EthreeEps"]["value"] == pytest.approx(2e-8) and out["hypotheses"]["H3a"]["holds"] is False


def test_h3c_identity_count_above_the_bound():
    ID = pd.DataFrame(dict(resid_same=[1e-13] * 4, resid_indep=[1e-12, 5e-9, -3e-9, 0.0], err_bound=[1e-9] * 4,
                           tag="x2", cls="video", G_full=0.0, G_fix=0.0, date=DATES[0], unit=f"{DATES[0]}|o60",
                           seed=0))
    m = A.analyze_e3(dict(ident=ID), dict(n_boot=NB))["macros"]
    assert m["EthreeIdentityAbove"]["value"] == 2


# ------------------------------------------------------------------------------------------------ E5
def _e5_pairs():
    rng = np.random.default_rng(4)
    rows = []
    for d in DATES:
        for seed in (0, 1):
            for tgt in ("video", "urllc"):
                for a in (0.5, 2.0):
                    for w in range(4):
                        rows.append(dict(date=d, seed=seed, arm="protected", phase="attack", pipeline="deploy",
                                         target=tgt, alpha=a, window=f"dev-{d}-pA-o{60 + 10800 * w:05d}-x",
                                         d_utility_declared_target=-0.8, d_utility_realized_target=-0.8))
                        for pipe in ("measurement", "stored"):
                            g = 0.0 if pipe == "measurement" else (0.4 if a == 0.5 else -0.3)
                            rows.append(dict(date=d, seed=seed, arm="protected", phase="later", pipeline=pipe,
                                             target=tgt, alpha=a, window=f"dev-{d}-pB-o{43260 + 10800 * w:05d}-x",
                                             d_utility_declared_target=g + rng.normal(0, 0.01) * (pipe == "stored"),
                                             d_utility_realized_target=g))
    return A._units(pd.DataFrame(rows))


def test_e5_repay_minutes_and_inflation_later_gain(monkeypatch):
    monkeypatch.setattr(A, "POWER_SIMS", 5)
    res = A.analyze_e5(dict(e5_pairs=_e5_pairs()), dict(n_boot=NB, split="train"))
    m = res["macros"]
    assert A._window_len_s() == 200.0
    assert m["EfiveRepayMinutes"]["value"] == pytest.approx(m["EfiveRepayWindows"]["value"] * 200.0 / 60.0)
    assert m["EfiveRepayMinutes"]["unit"].startswith("minutes")
    assert m["EfiveLaterInflate"]["value"] == pytest.approx(-0.3, abs=0.01)
    assert m["EfiveLaterInflateLo"]["value"] <= m["EfiveLaterInflate"]["value"] <= m["EfiveLaterInflateHi"]["value"]
    q = {r["quantity"]: r for r in res["tables"]["e5_extra"]}
    assert q["repay_minutes"]["window_len_s"] == 200.0 and "later_gain_alpha2" in q


# ------------------------------------------------------------------------------------------------ E7
def test_e7_h7b_macros(tmp_path):
    TA.make_e7(tmp_path / "e7")
    out = A.analyze_e7(dict(nr=pd.read_csv(tmp_path / "e7" / "cases.csv")), {})
    m, tb = out["macros"], out["tables"]
    assert m["EsevenProtMinusManip"]["value"] == pytest.approx(out["hypotheses"]["H7b"]["mean"]) == pytest.approx(2.0)
    assert m["EsevenProtBetter"]["value"] == out["hypotheses"]["H7b"]["windows_holding"] == 4
    assert m["EsevenProtMinusManipPdb"]["value"] == pytest.approx(2.0) and m["EsevenProtBetterPdb"]["value"] == 4
    assert len(tb["h7b_prot_minus_manip"]) == m["EsevenWindows"]["value"]


# ------------------------------------------------------------------------------------------------ JSON / names
def test_new_macros_are_letters_only_and_strict_json(e1, e2):
    names = set()
    for res in (e1["e1"], e1["e4"], e2["res"]):
        json.dumps(A._clean(res), allow_nan=False)
        names |= set(res["macros"])
    for n in ("EoneGainVideo", "EoneGainUrllc", "EoneGainMmtc", "EoneKappaZero", "EoneKappaZeroVsFree",
              "EoneGainVideoShare", "EoneLoadSlope", "EoneGainRhoHigh", "EoneMissDiffVideo", "EtwoPrbMoved",
              "EtwoVictimLossPct", "EtwoDevMbitPerPrb", "EtwoVictimMbitPerPrb", "EfourSeedLo", "EfourSeedHi"):
        assert n in names and n.isalpha(), n
