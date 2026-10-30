"""Pre-freeze audit fixes of the analysis group (review/prefreeze_audit.md section 7; 2026-10-20).

experiments/analyze.py, experiments/figures/**, slicemech/eval/stats.py, tools/make_numbers.py and the params key
EfivePrimary. Synthetic inputs only (the DEV builders of tests/test_exp_analysis.py and hand-made tables); no TEST
data is read: the TEST paths are exercised with a stubbed read gate and DEV-dated tables.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

CODE = Path(__file__).resolve().parents[1]
for _p in (CODE / "experiments", CODE / "experiments" / "figures", CODE / "tests"):
    sys.path.insert(0, str(_p))

import analyze as A  # noqa: E402
import e2_manipulation as X  # noqa: E402
import registry as R  # noqa: E402
from slicemech.data import puffer  # noqa: E402
from slicemech.eval import stats as S  # noqa: E402

TOOLS = CODE.parent / "tools"
DEV = list(puffer.split_dates("dev"))


# ------------------------------------------------------------------------------------------------ shared fixtures
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """The synthetic DEV analysis of tests/test_exp_analysis.py (known effects), power simulation shortened."""
    import test_exp_analysis as TA
    root = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(7)
    TA.make_e0(root)
    TA.make_e1(root, rng)
    TA.make_e2(root, rng)
    TA.make_e3(root, rng)
    TA.make_e4(root, rng)
    TA.make_e5(root, rng)
    TA.make_e6(root, rng)
    TA.make_e7(root / "e7")
    src = {e: root / e for e in A.EXPERIMENTS}
    out = tmp_path_factory.mktemp("summary")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(A, "POWER_SIMS", 30)
        res = A.run_analysis("dev", out, n_boot=200, src=src, crossed=False, confine=False, verbose=False)
    return dict(src=src, out=out, res=res)


def _w1_rows(units, arm, seeds, value=50.0, config="default", sha="a" * 8):
    """E1 tenant = -1 truthful rows of the default cell (canonical table columns)."""
    rows = []
    for u in units:
        d = u.split("|")[0]
        for s in seeds:
            rows.append(dict(exp="e1", arm=arm, arm_raw=arm, seed=s, tenant=-1, policy="truthful", unit=u, date=d,
                             n=3, d=2.0, rho=0.9, lead_s=4.004, kappa_scale=1.0 if arm == "protected" else np.nan,
                             config=config, realized_value=value, ckpt_sha=f"{sha}{arm}{s}" if s >= 0 else None))
    return rows


def _pv2_inputs(tmp_path: Path):
    """Every primary's own inputs complete and of one provenance; an oracle E1 window, an unc_env stage window and an
    unprotected E5 job missing; a checkpoint conflict in an E1 sweep configuration; an E2 stage output and an
    unprotected E5 job record without provenance."""
    units = A.expected_units("dev")
    seeds = tuple(R.CONFIGS["default"].seeds)
    rows = []
    for arm in ("protected", "meas_only", "unprotected", "kappa0"):
        rows += _w1_rows(units, arm, seeds)
    for arm in ("static", "equal", "direct", "direct_truerates"):
        rows += _w1_rows(units, arm, (-1,))
    rows += _w1_rows(units[:-1], "oracle", (-1,))                                  # one oracle window missing
    rows += _w1_rows(units[:3], "meas_only", (0,), config="d1", sha="b" * 8)       # sweep: another checkpoint
    rows += _w1_rows(units[:3], "meas_only", (0,), config="d1", sha="c" * 8)
    rk = [f"{d}|o{o}" for d, o in X.replay_keys(X.split_keys("dev"), A.REPLAY_PER_DATE)]
    for arm_raw, arm in (("unc_fixed", "unprotected"), ("prot_fixed", "fixed_price")):
        for u in rk:
            rows.append(dict(exp="e2", arm=arm, arm_raw=arm_raw, seed=0, tenant=-1, policy="truthful", unit=u,
                             date=u.split("|")[0], n=3, d=2.0, rho=0.9, lead_s=4.004, kappa_scale=np.nan,
                             config=None, realized_value=1.0, ckpt_sha=None))
    W = pd.DataFrame(rows)
    br = pd.DataFrame([dict(exp="e2", arm="unprotected_envelope", arm_raw="unc_env", unit=u, date=u.split("|")[0])
                       for u in units[1:]])                                         # one stage window missing
    import e5_training_channel as E5M
    p5 = []
    for d in DEV:
        for s in E5M.SEEDS_DEFAULT:
            for arm in E5M.ARMS:
                if arm == "unprotected" and d == DEV[0] and s == 0:
                    continue                                                        # one unprotected job missing
                p5.append(dict(date=d, seed=s, arm=arm, phase="later"))
    T = dict(windows=W, br=br, e5_pairs=pd.DataFrame(p5))
    e2 = tmp_path / "e2"
    for sub, side in (("replay/unc_fixed/s0/w0.parquet", True), ("stage/unc_env/s0/w1.parquet", False)):
        f = e2 / sub
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": [1]}).to_parquet(f)
        if side:
            Path(str(f) + ".prov.json").write_text(json.dumps(dict(code="c1", args=dict(seed=0),
                                                                   assets=dict(unconstrained="d" * 64))))
    jd = tmp_path / "e5" / "jobs"
    jd.mkdir(parents=True)
    for d in DEV:
        for s in E5M.SEEDS_DEFAULT:
            (jd / f"{d}__protected__s{s}.done.json").write_text(json.dumps(dict(arm="protected", seed=s, fp="f",
                                                                                ckpt_sha256="e" * 64)))
    (jd / "x__unprotected__s0.done.json").write_text(json.dumps(dict(arm="unprotected", seed=0, ckpt_sha256="f")))
    return T, dict(e2=e2, e5=tmp_path / "e5")


def _holm_inputs(T, src, frozen=None):
    comp, prov = A.completeness(T, "dev"), A.provenance(src, T)
    dates = A.primary_dates(T)
    res = {"e1": dict(hypotheses=dict(H1a=dict(rule_met=True, p=1e-4))),
           "e2": dict(hypotheses=dict(H2c=dict(rule_met=True, p=1e-4))),
           "e4": dict(hypotheses=dict(H4a=dict(rule_met=True, noninferiority_p=1e-4))),
           "e5": dict(hypotheses=dict(H5b=dict(rule_met=True, contrary=False, p=1e-4)))}
    for e in res:
        res[e]["inputs_check"] = A.inputs_check(e, comp, prov, dates, frozen)
    return comp, prov, res


# ====================================================================================== item 4: PV-2
def test_pv2_each_primary_is_gated_only_on_its_own_inputs(tmp_path):
    T, src = _pv2_inputs(tmp_path)
    comp, prov, res = _holm_inputs(T, src)
    # the information records still show every defect ...
    assert comp["e1"]["oracle"]["complete"] is False and comp["e1"]["protected"]["complete"]
    assert comp["e2"]["stage"]["unc_env"]["complete"] is False and comp["e5"]["jobs"]["complete"] is False
    assert comp["e5"]["jobs_by_arm"]["protected"]["complete"] and not comp["e5"]["jobs_by_arm"]["unprotected"]["complete"]
    assert prov["e1"]["ok"] is False and prov["e2"]["ok"] is False and prov["e5"]["ok"] is False
    # ... but none of them is an input of a primary
    for e in ("e1", "e2", "e4", "e5"):
        assert res[e]["inputs_check"]["ok"], (e, res[e]["inputs_check"])
    v = {r["name"]: r for r in A.holm_summary(res, split="test")["tables"]["verdicts"]}
    assert all(v[h]["supported"] is True for h in A.HOLM_FAMILY), v
    assert all("complete, single-provenance" in v[h]["reason"] for h in A.HOLM_FAMILY)
    assert A.PRIMARY_INPUTS == {"e1": {"e1": ("protected", "meas_only")}, "e4": {"e1": ("protected", "unprotected")},
                                "e2": {"e2": {"replay": ("unc_fixed",)}}, "e5": {"e5": {"jobs_by_arm": ("protected",)}}}
    # a missing window of an arm a primary reads fails exactly the primaries that read it, with the reason
    W = T["windows"]
    T2 = dict(T, windows=W.drop(W[(W["arm"] == "unprotected") & (W["exp"] == "e1") & (W["unit"] == W["unit"].iloc[0])]
                                .index))
    _, _, res2 = _holm_inputs(T2, src)
    v2 = {r["name"]: r for r in A.holm_summary(res2, split="test")["tables"]["verdicts"]}
    assert v2["H4a"]["supported"] is False and v2["H4a"]["reason"] == "inputs incomplete"
    assert v2["H1a"]["supported"] and v2["H2c"]["supported"] and v2["H5b"]["supported"]
    assert v2["H4a"]["inputs_ok"] is False


def test_f5_primary_inputs_must_cover_the_frozen_number_of_test_dates(tmp_path):
    T, src = _pv2_inputs(tmp_path)
    dates = A.primary_dates(T)
    assert dates["e1"]["e1"] == {"protected": len(DEV), "meas_only": len(DEV)}
    assert dates["e2"]["e2"] == {"replay/unc_fixed": len(DEV)} and dates["e5"]["e5"] == {"jobs_by_arm/protected": len(DEV)}
    _, _, ok = _holm_inputs(T, src, frozen=len(DEV))
    assert all(ok[e]["inputs_check"]["ok"] for e in ok) and ok["e1"]["inputs_check"]["test_dates"]["ok"]
    _, _, bad = _holm_inputs(T, src, frozen=16)                     # a freeze of 16 dates against 21 dated inputs
    v = {r["name"]: r for r in A.holm_summary(bad, split="test")["tables"]["verdicts"]}
    for h in A.HOLM_FAMILY:
        assert v[h]["supported"] is False and "frozen test_dates = 16" in v[h]["reason"], v[h]
    assert "e1:protected has 21" in bad["e1"]["inputs_check"]["reason"]


def test_f5_main_passes_the_frozen_test_dates_of_the_freeze_record(monkeypatch, tmp_path):
    from slicemech.data import puffer_fetch as PF
    text = "# PREREG\n```freeze\n" + yaml.safe_dump(dict(
        analysis_code_sha256="0" * 64, config_sha256="1" * 64, checkpoints={"results/models/x": "2" * 64},
        date_rule=PF.DATE_RULES[24], test_dates=24), sort_keys=True) + "```\n"
    rec = PF.parse_freeze(text)                                       # a synthetic, validated freeze record
    assert rec["test_dates"] == 24
    seen = {}
    monkeypatch.setattr(PF, "require_test_read_gate", lambda tf=None: dict(path="prereg/PREREG.md",
                                                                           test_dates=rec["test_dates"]))
    monkeypatch.setattr(A, "run_analysis", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(A.sio, "require_free_gb", lambda gb: None)
    assert A.main(["--split", "test"]) == 0 and seen["test_dates"] == 24 and seen["n_boot"] == A.N_BOOT
    monkeypatch.setattr(PF, "require_test_read_gate", lambda tf=None: dict(path="prereg/PREREG.md"))
    assert A.main(["--split", "test"]) == 3                           # a record without the count is refused


# ====================================================================================== item 5: PV-4, PV-1, PV-5
def _dev_pairs(effect: float, b: float = 0.0, date_sd: float = 0.2, win_sd: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in DEV:
        mu = effect + rng.normal(0, date_sd)
        for o in (60, 14460, 28860, 43260, 57660, 72060):
            a = b + mu + rng.normal(0, win_sd)
            rows.append(dict(date=d, unit=f"{d}|o{o}", a=a, b=b))
    return pd.DataFrame(rows)


def test_pv4_power_of_the_registered_test_analytic_and_simulated():
    from scipy.stats import norm
    pw = A.power_inputs(_dev_pairs(0.4), "H1a", n_boot=1000, n_sim=60, sim_boot=400)
    assert pw["dates"] == 10 and pw["windows_per_date"] == 6 and "two-stage" in pw["note"]
    r, sd = pw["boot_inflation_r"], pw["sd_date_means"]
    assert math.isfinite(r) and r > 0
    za, zb = norm.ppf(1 - A.POWER_ALPHA), norm.ppf(A.POWER)
    assert pw["dates_needed_point"] == max(A.MIN_DATES, math.ceil(((r * za + zb) * sd / pw["effect_point"]) ** 2))
    sim = pw["simulation"]
    assert sim["status"] == "ok" and set(sim["by_dates"]) == {"16", "24"} and sim["boot_reps"] == 400
    for G in ("16", "24"):
        for k in ("point", "conservative"):
            c = sim["by_dates"][G][k]
            assert c is None or 0.0 <= c["ci_holm"] <= 1.0
    assert pw["dates_decision"] in (16, 24) and pw["decision_basis"].startswith("simulated power")
    big = A.power_inputs(_dev_pairs(3.0), "H1a", n_boot=500, n_sim=30, sim_boot=300)
    assert big["simulation"]["by_dates"]["16"]["conservative"]["ci_holm"] == 1.0 and big["dates_decision"] == 16
    small = A.power_inputs(_dev_pairs(0.12, seed=3), "H1a", n_boot=500, n_sim=30, sim_boot=300)
    assert small["dates_decision"] == 24
    s16 = small["simulation"]["by_dates"]["16"]["point"]["ci_holm"]
    s24 = small["simulation"]["by_dates"]["24"]["point"]["ci_holm"]
    assert s24 >= s16 - 0.15                                       # more dates, (noisily) more power


def test_pv1_h1a_percentage_part_and_its_status():
    low = A.power_inputs(_dev_pairs(0.2, b=100.0, date_sd=0.02, win_sd=0.05), "H1a", n_boot=500, n_sim=10,
                         sim_boot=200)
    assert low["pct"] == pytest.approx(0.2, abs=0.05) and low["min_effect_pct"] == A.H1A_MIN_PCT
    assert low["pct_rule_status"].startswith("unattainable") and low["pct_rule_dates_needed"] is None
    assert low["dates_needed_rule"] is None and "CI/Holm" in low["power_scope"]
    assert low["sd_date_pct"] is not None and low["pct_lo"] is not None
    assert low["simulation"]["by_dates"]["16"]["point"]["rule"] == 0.0     # the full rule never holds
    high = A.power_inputs(_dev_pairs(2.0, b=100.0, date_sd=0.2, win_sd=0.3), "H1a", n_boot=500, n_sim=10,
                          sim_boot=200)
    assert high["pct_rule_status"] == "attainable" and high["pct_rule_dates_needed"] >= A.MIN_DATES
    assert high["dates_needed_rule"] == max(high["dates_needed_conservative"], high["pct_rule_dates_needed"])


def test_pv5_h5b_power_uses_the_preregistered_sign(monkeypatch):
    monkeypatch.setattr(A, "H5B_SIGN", 1)
    neg = A.power_inputs(_dev_pairs(-0.8), "H5b", n_boot=500, n_sim=10, sim_boot=200)
    assert neg["preregistered_sign"] == 1 and neg["effect_point"] < 0
    assert neg["dates_needed_point"] is None and neg["effect_conservative"] is None
    assert neg["simulation"]["by_dates"]["16"]["point"] is None and neg["dates_decision"] == 24
    pos = A.power_inputs(_dev_pairs(0.8), "H5b", n_boot=500, n_sim=10, sim_boot=200)
    assert pos["effect_point"] > 0 and pos["effect_conservative"] == pytest.approx(pos["lo"])
    assert pos["alpha_one_sided"] == pytest.approx(A.POWER_ALPHA / 2)


def test_power_entries_reach_holm_dev_power(synth):
    dp = {r["name"]: r for r in synth["res"]["holm"]["tables"]["dev_power"]}
    assert {"H1a", "H2c", "H4a", "H5b"} <= set(dp)
    assert "simulation" in dp["H1a"] and "pct_rule_status" in dp["H1a"]
    # 3 estimation-half dates in the synthetic tree (< MIN_DATES): no CI bound, so no 16/24 decision
    assert dp["H1a"]["dates_decision"] is None


# ====================================================================================== item 8: HC-4 (G6)
def _g6_windows(with_cap: bool = True):
    rng = np.random.default_rng(5)
    rows = []
    for d in DEV[:6]:
        for o in (60, 14460):
            for arm, seeds, use in (("protected", (0, 1), 0.8), ("meas_only", (0, 1), 0.6), ("static", (-1,), 0.7)):
                for s in seeds:
                    for t, c in enumerate(("video", "urllc", "mmtc")):
                        q = (50.0, 40.0, 10.0)[t]
                        cap = q * 0.43
                        on = cap * (use + 0.05 * s + rng.normal(0, 0.02))
                        r = dict(exp="e1", arm=arm, seed=s, tenant=t, cls=c, policy="truthful", date=d,
                                 unit=f"{d}|o{o}", n=3, d=2.0, rho=0.9, lead_s=4.004,
                                 kappa_scale=1.0 if arm == "protected" else np.nan, on_time_Mbit=on, quota=q)
                        if with_cap:
                            r.update(cap_Mbit=cap, unused_Mbit=cap - on, prb_busy=on / 0.43, r_true=0.43)
                        rows.append(r)
    return pd.DataFrame(rows)


def test_hc4_g6_table_ratio_of_sums_inside_the_bootstrap_and_macros():
    W = _g6_windows()
    g6, notes = A.g6_table(W, A.MAIN, n_boot=500)
    assert not notes
    x = W[W["arm"] == "protected"]
    pw = x.groupby(["date", "unit", "seed"])[["on_time_Mbit", "cap_Mbit", "prb_busy", "quota"]].sum() \
        .groupby(["date", "unit"]).mean()                           # seeds averaged within a window
    r = g6[(g6["arm"] == "protected") & (g6["cls"] == "all")].iloc[0]
    assert r["pct_reserved_used"] == pytest.approx(100 * pw["on_time_Mbit"].sum() / pw["cap_Mbit"].sum(), rel=1e-12)
    assert r["pct_cell_used"] == pytest.approx(100 * pw["prb_busy"].mean() / 106.0, rel=1e-12)
    assert r["mbps_per_reserved_prb"] == pytest.approx(pw["on_time_Mbit"].sum() / pw["quota"].sum(), rel=1e-12)
    for k in ("pct_reserved_used", "pct_cell_used", "mbps_per_reserved_prb"):
        assert r[f"{k}_lo"] <= r[k] <= r[f"{k}_hi"]
    assert r["identity_max_abs_resid"] == pytest.approx(0.0, abs=1e-9) and r["seeds"] == 2
    assert set(g6["arm"]) == {"protected", "meas_only", "static"} and set(g6["cls"]) == {"all", "video", "urllc", "mmtc"}
    mac = A.Macros("e4")
    A._g6_macros(g6, mac)
    m = mac.d
    assert m["EfourPrbUsed"]["value"] == pytest.approx(r["pct_reserved_used"])
    free = g6[(g6["arm"] == "meas_only") & (g6["cls"] == "all")].iloc[0]
    assert m["EfourPrbUsedFree"]["value"] == pytest.approx(free["pct_reserved_used"])
    assert m["EfourPrbUsedCI"]["value"].startswith("[") and "EfourPrbUsedFreeLo" in m and "EfourCellUsed" in m
    assert m["EfourPrbUsed"]["unit"].startswith("percent of the reserved PRB capacity")


def test_hc4_g6_on_outputs_without_the_new_columns_is_graceful(synth):
    g6, notes = A.g6_table(_g6_windows(with_cap=False), A.MAIN, n_boot=200)
    assert notes and "lack" in notes[0] and "pct_reserved_used" not in g6.columns
    assert g6["mbps_per_reserved_prb"].notna().all()
    e4 = synth["res"]["e4"]                                      # the old-layout synthetic E1: no EfourPrbUsed
    assert "EfourPrbUsed" not in e4["macros"] and any("G6" in n for n in e4["notes"])
    assert e4["tables"]["g6_radio_efficiency"] and "EfourPrbRate" in e4["macros"]


# ====================================================================================== item 9: E5-1
def test_e5_1_efive_primary_is_the_code_constant(tmp_path):
    st = yaml.safe_load(open(A.SPEC / "params.yaml"))["experiment_settings"]
    assert float(st["EfivePrimary"]) == A.H5B_PRIMARY_ALPHA == 0.5
    m = A.settings_summary(models_root=tmp_path)["macros"]
    assert m["EfivePrimary"]["value"] == 0.5 and m["EfivePrimary"]["text"] == "0.5"
    assert "H5B_PRIMARY_ALPHA" in m["EfivePrimary"]["source"] and m["EfiveSign"]["value"] == "+1"


# ====================================================================================== item 10: HC-2
def test_hc2_victim_loss_is_minus_the_signed_change(synth):
    m = synth["res"]["e2"]["macros"]
    assert m["EtwoVictim"]["value"] < 0
    assert m["EtwoVictimLoss"]["value"] == pytest.approx(-m["EtwoVictim"]["value"])
    assert m["EtwoVictimLossLo"]["value"] == pytest.approx(-m["EtwoVictimHi"]["value"])
    assert m["EtwoVictimLossHi"]["value"] == pytest.approx(-m["EtwoVictimLo"]["value"])
    assert "lost" in m["EtwoVictimLoss"]["unit"] and "\\ensuremath{-}" not in m["EtwoVictimLossCI"]["value"]


# ====================================================================================== item 11: NUM-04 (+ NUM-15)
def test_num04_error_bound_and_certificate_macros(synth):
    h = synth["res"]["e3"]["hypotheses"]["H3a"]
    m = synth["res"]["e3"]["macros"]
    assert h["max_err"] == pytest.approx(h["max_cert"] / 2)
    assert m["EthreeErr"]["value"] == pytest.approx(h["max_cert"] / 2) and m["EthreeCert"]["value"] == h["max_cert"]
    assert "EthreeTol" not in m
    for k in ("EthreeErr", "EthreeCert", "EthreeEps", "EthreeIdentity"):
        assert re.fullmatch(r"\\ensuremath\{-?\d(\.\d+)?\\times10\^\{-?\d+\}\}", m[k]["text"]), (k, m[k]["text"])
    for r in synth["res"]["e3"]["tables"]["h3a_by_design"]:
        assert r["max_err"] == pytest.approx(r["max_cert"] / 2)


def test_num15_tex_number_formatter():
    assert A.tex_num(2.0147e-08, "{:.1e}") == "\\ensuremath{2.0\\times10^{-8}}"
    assert A.tex_num(-1.5e-10, "{:.2g}") == "\\ensuremath{-1.5\\times10^{-10}}"
    assert A.tex_num(-0.5) == "\\ensuremath{-}0.50" and A.tex_num(12345, "{:,d}") == "12,345"
    assert A.ci_text(-1, 2) == "[\\ensuremath{-}1.00, 2.00]"


# ====================================================================================== item 12: PV-3
def test_pv3_h2c_secondary_tags_and_the_primary_cell_constants(synth):
    allh = synth["res"]["e2"]["hypotheses"]["H2c_all"]
    for arm, mm in allh.items():
        for metric, r in mm.items():
            prim = arm == A.H2C_PRIMARY_ARM and metric == "d_on_time_victims"
            assert r["role"].startswith("primary" if prim else "secondary"), (arm, metric)
    assert A.H2C_PRIMARY_ARM == "unprotected" and A.ARM_MAP["unc_fixed"][0] == "unprotected"
    for mod in ("e2_manipulation.py", "e4_cost.py"):
        txt = (CODE / "experiments" / mod).read_text()
        assert re.search(r'"--replay-per-date", type=int, default=(\d+)', txt).group(1) == str(A.REPLAY_PER_DATE)
    assert {r["metric"] for r in synth["res"]["e2"]["tables"]["victim_contrasts"]} >= {"d_on_time_victims"}


# ====================================================================================== item 13: PV-8
@pytest.mark.parametrize("opt", [["--n-boot", "500"], ["--no-crossed"], ["--smoke"], ["--out", "x"]])
def test_pv8_test_analysis_refuses_protocol_options(monkeypatch, capsys, opt):
    from slicemech.data import puffer_fetch as PF
    calls = []
    monkeypatch.setattr(PF, "require_test_read_gate", lambda tf=None: calls.append(tf) or dict(path="x", test_dates=16))
    assert A.main(["--split", "test", *opt]) == 3
    err = capsys.readouterr().err
    assert opt[0] in err and not calls                               # refused before the TEST gate
    assert A.refusals_on_test(A.argparse.Namespace(source=[], n_boot=A.N_BOOT, no_crossed=False, smoke=False,
                                                out=None)) == []


# ====================================================================================== item 14: PV-9
def test_pv9_test_holm_never_reuses_a_stale_summary(synth, tmp_path):
    out = tmp_path / "test_summary"
    out.mkdir()
    stale = dict(experiment="e2", split="test", smoke=False, code_sha256="0" * 64, inputs=dict(digest="old"),
                 hypotheses=dict(H2c=dict(rule_met=True, p=1e-6)), macros={}, tables={}, notes=[])
    (out / "e2.json").write_text(json.dumps(stale))
    # TEST label on DEV-dated synthetic inputs: every row is dropped by the split guard (no TEST data is read)
    res = A.run_analysis("test", out, n_boot=100, src=synth["src"], exps=("e1",), crossed=False, confine=False,
                         verbose=False)
    hs = res["holm"]
    assert hs["family_sources"]["e2"] == "this run"                  # re-analysed, not the stale p = 1e-6
    assert json.loads((out / "e2.json").read_text())["inputs"]["digest"] != "old"
    assert "H2c" in hs["hypotheses"]["missing"]


# ====================================================================================== item 19: HC-6
def test_hc6_identity_is_judged_on_the_independent_engine():
    ID = pd.DataFrame(dict(resid_same=[1e-13, 2e-13], resid_indep=[1e-10, 5e-6], err_bound=[1e-9, 1e-9], tag="x2",
                           cls="video", date=DEV[0], unit=f"{DEV[0]}|o60", G_full=0.0, G_fix=0.1))
    res = A.analyze_e3(dict(ident=ID), dict(n_boot=50))
    h = res["hypotheses"]["H3c_identity"]
    assert h["holds_same_engine"] and h["holds"] is False and h["n_indep_above_bound"] == 1
    assert res["macros"]["EthreeIdentityIndep"]["value"] == pytest.approx(5e-6)


# ====================================================================================== item 21: HC-9
def test_hc9_all_nan_sums_stay_nan_and_no_mixed_seed_fallback_on_test():
    L = pd.DataFrame(dict(arm="unprotected", date=DEV[0], unit=f"{DEV[0]}|o60", seed=0, tenant=[0, 1], honest=True,
                          value=1.0, profile_tag="m1_1_1.5", n_runs=2))
    W = pd.DataFrame(dict(exp="e4", arm="protected", policy="truthful", tenant=[0], kappa_scale=1.0,
                          unit=f"{DEV[1]}|o60", seed=0, realized_value=5.0))        # no Protected run on L's window
    t = A._h4b_learning(L, W, n_boot=50)
    assert t.iloc[0]["status"].startswith("no Protected truthful run")    # an all-NaN sum is not a 0 difference
    with pytest.raises(A.MissingInput, match="TEST"):
        A._h4b_from_e1e2(pd.DataFrame(), "unprotected", n_boot=50, split="test")


# ====================================================================================== item 24: HC-13
def test_hc13_no_nan_ci_strings():
    mac = A.Macros("e1")
    mac.ci("EoneX", dict(estimate=1.0, lo=float("nan"), hi=float("nan"), pct=2.0, pct_lo=None, pct_hi=float("nan"),
                        denominator="d"), "u", "s")
    assert "EoneX" in mac.d and "EoneXPct" in mac.d
    assert not {"EoneXCI", "EoneXLo", "EoneXHi", "EoneXPctCI"} & set(mac.d)
    assert all("nan" not in str(v["value"]) for v in mac.d.values())


# ====================================================================================== items 25, 29, 30: E5
def _e5_pairs():
    rng = np.random.default_rng(2)
    rows = []
    for d in DEV[:6]:
        for seed in (0, 1, 2):
            for tgt in ("video", "urllc"):
                for a in (0.5, 2.0):
                    for w in range(4):
                        rows.append(dict(date=d, seed=seed, arm="protected", phase="attack", pipeline="deploy",
                                         target=tgt, alpha=a, window=f"dev-{d}-pA-o{60 + 10800 * w:05d}-x",
                                         d_utility_declared_target=-0.8, d_utility_realized_target=-0.8,
                                         d_on_time_victims=0.0, d_miss_pct_victims=0.0, d_value_victims=0.0))
                        for pipe in ("measurement", "stored"):
                            g = 0.0 if pipe == "measurement" else (0.4 + 0.1 * seed if a == 0.5 else -0.3)
                            v = 0.0 if pipe == "measurement" else -2.0
                            rows.append(dict(date=d, seed=seed, arm="protected", phase="later", pipeline=pipe,
                                             target=tgt, alpha=a, window=f"dev-{d}-pB-o{43260 + 10800 * w:05d}-x",
                                             d_utility_declared_target=g + rng.normal(0, 0.01) * (pipe == "stored"),
                                             d_utility_realized_target=g, d_on_time_victims=v,
                                             d_miss_pct_victims=-v, d_value_victims=1.5 * v))
    return A._units(pd.DataFrame(rows))


def test_e5_by_seed_cost_net_repay_and_victims(monkeypatch):
    monkeypatch.setattr(A, "POWER_SIMS", 5)
    res = A.analyze_e5(dict(e5_pairs=_e5_pairs()), dict(n_boot=300, split="train"))
    m, tb, h = res["macros"], res["tables"], res["hypotheses"]["H5b"]
    bs = {r["seed"]: r["estimate"] for r in tb["h5b_by_seed"]}                    # E5-8
    assert set(bs) == {0, 1, 2} and bs[2] - bs[0] == pytest.approx(0.2, abs=0.02)
    assert h["seed_spread"]["max"] == pytest.approx(max(bs.values()))
    assert m["EfiveAttackChange"]["value"] == pytest.approx(-0.8) and m["EfiveCost"]["value"] == pytest.approx(0.8)
    assert m["EfiveCostLo"]["value"] == pytest.approx(-m["EfiveAttackChangeHi"]["value"])       # NUM-08
    assert m["EfiveNet"]["value"] == pytest.approx(h["estimate"] - 0.8, abs=1e-9)
    assert m["EfiveRepay"]["value"] == pytest.approx(0.8 / h["estimate"], rel=1e-9)
    vt = pd.DataFrame(tb["later_victims"])                                        # NUM-07
    p = vt[(vt["pipeline"] == "stored") & (vt["alpha"] == 0.5) & (vt["target"] == "pooled")].iloc[0]
    assert p["d_on_time_victims"] == pytest.approx(-2.0) and p["d_miss_pct_victims"] == pytest.approx(2.0)
    assert m["EfiveVictimOnTime"]["value"] == pytest.approx(-2.0) and m["EfiveVictimLoss"]["value"] == pytest.approx(2.0)
    assert m["EfiveVictimValue"]["value"] == pytest.approx(-3.0) and set(vt["target"]) == {"pooled", "video", "urllc"}
    assert any("EfiveDiffMeas" in n for n in res["notes"])


def test_num07_efive_diff_from_kept_episodes(tmp_path):
    jd = tmp_path / "jobs"
    jd.mkdir()
    rows = []
    for pipe in ("measurement", "stored"):
        for tgt, a in (("none", 1.0), ("video", 0.5)):
            for k in range(4):
                for t in range(3):
                    w = 1 / 3 + (0.1 if (pipe == "stored" and a == 0.5 and k >= 2 and t == 0) else 0.0)
                    rows.append(dict(phase="later", pipeline=pipe, target=tgt, alpha=a, window="w5", k=k, tenant=t,
                                     w=w, scored=k >= 1, date=DEV[0], seed=0, arm="protected"))
    pd.DataFrame(rows).to_parquet(jd / "j.episodes.parquet")
    (jd / "j.done.json").write_text("{}")
    wd = A._e5_wdiff(jd)
    got = {r["pipeline"]: (r["differ"], r["intervals"]) for _, r in wd.iterrows()}
    assert got == {"measurement": (0, 3), "stored": (2, 3)}
    mac = A.Macros("e5")
    A._e5_wdiff_summary(wd, mac, {})
    assert mac.d["EfiveDiffMeas"]["value"] == 0.0 and mac.d["EfiveDiffStored"]["value"] == pytest.approx(200 / 3)


# ====================================================================================== item 28: NUM-05
def test_num05_per_class_gain_macros_with_cis(synth):
    m = synth["res"]["e2"]["macros"]
    for stem in ("EtwoGain", "EtwoGainFixed", "EtwoGainUsage"):
        for c in ("Video", "Urllc", "Mmtc"):
            assert {stem + c, stem + c + "Lo", stem + c + "Hi", stem + c + "CI"} <= set(m), stem + c
    assert "EtwoGainEnvelopeVideo" not in m
    tb = pd.DataFrame(synth["res"]["e2"]["tables"]["br_by_class"])
    u = tb[(tb["arm"] == "unprotected") & (tb["cls"] == "urllc")].iloc[0]
    assert m["EtwoGainUrllc"]["value"] == pytest.approx(u["mean_gain"]) and "classes pooled" in m["EtwoGain"]["source"]


# ====================================================================================== item 31: NUM-09
def test_num09_labels_come_from_controllers_yaml(synth):
    import _common as C
    import fig_e1_value
    assert A.label("meas_only") == "No reports" == C.style("meas_only")["label"]
    assert "No reports" in fig_e1_value.YLABEL
    for f in (CODE / "experiments" / "figures").glob("*.py"):
        assert "Measurement-only" not in f.read_text(), f.name
    m = synth["res"]["e1"]["macros"]
    assert "No reports" in m["EoneGainPct"]["unit"] and "No reports" in m["EoneGain"]["source"]


# ====================================================================================== item 32: NUM-10
def test_num10_every_outline_figure_with_data_has_a_script():
    import make_all
    for name in ("fig_e2_manip", "fig_e3_protect", "fig_e4_cost"):
        assert name in make_all.FIGURES and (CODE / "experiments" / "figures" / f"{name}.py").is_file()
    outline = yaml.safe_load(open(A.SPEC / "outline.yaml"))
    ids = {f["id"] for f in outline["figures"] if f["id"].startswith("fig:")}
    covered = set(re.findall(r"fig:[a-z]+", (CODE / "experiments" / "figures" / "make_all.py").read_text()))
    data_figs = ids - {"fig:timeline", "fig:arch", "fig:allocator"}           # TikZ / theory-check figures
    assert data_figs <= covered, data_figs - covered


# ====================================================================================== item 33: NUM-11/14/16/17
def test_num11_num16_settings_macros(tmp_path):
    pd.DataFrame(dict(kind=["tune", "tune", "main", "copy", "static"], train_seconds=[1200, 1800, 2400, 60, None])) \
        .to_csv(tmp_path / R.INDEX_FILE, index=False)
    m = A.settings_summary(models_root=tmp_path)["macros"]
    assert m["TrainMinutesNet"]["value"] == pytest.approx(30.0) and m["TrainMinutesNet"]["text"] == "30"
    if "RateMin" in m:
        assert m["RateMin"]["fmt"] == "{:.4f}" and m["RateMax"]["fmt"] == "{:.3f}"


def test_num14_percent_macros_are_bare_numbers(synth):
    for e in A.EXPERIMENTS + ("holm",):
        for name, spec in synth["res"][e]["macros"].items():
            if str(spec["unit"]).startswith("percent"):
                assert "%" not in spec["text"] and "percent" not in str(spec["text"]), (e, name)
                assert isinstance(spec["value"], (int, float)) or spec["value"].startswith("["), (e, name)


def test_num17_e6_e7_macros(synth):
    m6 = synth["res"]["e6"]["macros"]
    assert {"EsixTimeSix", "EsixTimeNine", "EsixTimeMedianSix", "EsixTimeMedianNine"} <= set(m6)
    assert m6["EsixTimeNine"]["value"] == m6["EsixTime"]["value"]
    m7 = synth["res"]["e7"]["macros"]
    assert m7["EsevenCorrMin"]["value"] == pytest.approx(0.9) and m7["EsevenWindows"]["value"] == 4
    assert m7["EsevenVictimChange"]["value"] == pytest.approx(-m7["EsevenVictim"]["value"])
    assert "EsevenWindows" in m7["EsevenSign"]["unit"]


# ====================================================================================== item 34: make_numbers --final
def _make_numbers():
    spec = importlib.util.spec_from_file_location("make_numbers_mod", TOOLS / "make_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_numbers_final_catches_every_missing_macro_and_non_test_summaries(tmp_path):
    MN = _make_numbers()
    paper, sdir = tmp_path / "paper", tmp_path / "summary"
    paper.mkdir()
    sdir.mkdir()
    (paper / "a.tex").write_text("\\providecommand{\\EfourPrbUsed}{\\missingnum{EfourPrbUsed}}\n"
                                 "\\newcommand{\\HarqName}{HARQ}\n"
                                 "We use \\nTestDatesUnset{} dates, \\EoneGain{}, \\EfourPrbUsed{}, \\TestDatesLarge{}, "
                                 "\\HarqName{}, \\theexample{} and \\today{}. % \\EsevenNever{} in a comment\n")
    mac = dict(EoneGain=dict(value=-0.5, fmt="{:.2f}", text="\\ensuremath{-}0.50", unit="u", source="e1: x"),
               EonePct=dict(value=5.0, fmt="{:.0f}%", unit="percent", source="e1: y"))
    (sdir / "e1.json").write_text(json.dumps(dict(split="dev", smoke=False, macros=mac,
                                                  inputs_check=dict(ok=False))))
    (sdir / "_memberships.json").write_text(json.dumps({"u": {"date": {"d": 1.0}}}))
    kw = ["--summary-dir", str(sdir), "--paper", str(paper), "--out", str(tmp_path / "numbers.tex"),
          "--manifest", str(tmp_path / "manifest.json")]
    assert MN.main(kw) == 0                                           # draft: placeholders only
    tex = (tmp_path / "numbers.tex").read_text()
    assert "\\newcommand{\\EoneGain}{\\ensuremath{-}0.50}" in tex and "{5\\%}" in tex       # text; '%' escaped
    assert "\\providecommand{\\nTestDatesUnset}{\\missingnum{nTestDatesUnset}}" in tex
    assert "\\providecommand{\\EfourPrbUsed}{\\missingnum{EfourPrbUsed}}" in tex          # a placeholder is missing
    for name in ("HarqName", "theexample", "today", "TestDatesLarge", "EsevenNever"):
        assert f"\\providecommand{{\\{name}}}" not in tex, name
    assert MN.main(["--final", *kw]) == 1                             # dev summary + missing macros
    mac.update(nTestDatesUnset=dict(value=16, fmt="{:d}", unit="dates", source="e1: z"),
               EfourPrbUsed=dict(value=40.0, fmt="{:.1f}", unit="percent", source="e4: g6"))
    (sdir / "e1.json").write_text(json.dumps(dict(split="test", smoke=False, macros=mac)))
    assert MN.main(["--final", *kw]) == 0
    (sdir / "e1.json").write_text(json.dumps(dict(split="test", smoke=True, macros=mac)))
    assert MN.main(["--final", *kw]) == 1                             # a smoke summary is never final


# ====================================================================================== item 35 (analysis part, F9)
def test_f9_completeness_covers_sweeps_bandits_learn_and_h4b(synth):
    comp = json.loads((synth["out"] / "holm.json").read_text())["input_checks"]["completeness"]
    assert set(comp["e1_sweeps"]) == {k for k in R.CONFIGS if k != "default"}
    assert comp["e1_sweeps"]["d1"]["protected"]["expected"] == len(A.expected_units("dev"))
    assert comp["e2"]["bandit"]["expected"] == (len(X.BANDIT_DESIGNS) * len(X.CHECKPOINT_SEEDS)
                                                * len(X.BANDIT_ALGOS) * X.BANDIT_RUNS)
    assert comp["e4"]["learn"]["expected"] == 2 and comp["e4"]["learn"]["present"] == 0
    assert set(comp["e4"]["h4b"]) >= {"unc_fixed", "prot_fixed", "unc_usage", "unc_env"}
    assert comp["e4"]["h4b"]["unc_fixed"]["expected"] == A.REPLAY_PER_DATE * len(DEV)


# ====================================================================================== stats: shared resamples
def test_cluster_bootstrap_multi_shares_resamples_and_matches_single():
    rng = np.random.default_rng(1)
    V = np.column_stack([rng.normal(1, 1, 60), rng.normal(10, 1, 60)])
    d = np.repeat(np.arange(10), 6)
    st = {"diff": lambda m: m[..., 0], "pct": lambda m: 100 * m[..., 0] / m[..., 1]}
    mm = S.cluster_bootstrap_multi(V, d, st, n_boot=700, seed=4)
    for k, f in st.items():
        one = S.cluster_bootstrap(V, d, n_boot=700, seed=4, stat=f)
        assert np.array_equal(one.reps, mm[k].reps) and one.lo == mm[k].lo and one.estimate == mm[k].estimate
    with pytest.raises(ValueError):
        S.cluster_bootstrap_multi(V, d, {}, n_boot=10)
