"""Regression tests for the round-3 review of the experiment drivers (CHANGELOG.md, "fixer, round 3").

One section per confirmed finding (numbering of CHANGELOG.md / STATUS.md §2c):

* X1  H5b decision ignored the pre-registered sign (-> ``H5B_SIGN``, signed rule, contrary flag, TEST refusal);
* X2  primary verdicts ignored Holm (-> final verdicts in holm.json: rule AND Holm AND, on TEST, clean inputs);
* X3  holm.json was rebuilt from the experiments of one invocation only;
* X4  E2-E6 on TEST were not tied to the frozen assets, code and protocol (-> verify_frozen, refused options,
      provenance fingerprints with stale-output handling, the code hash re-checked on every TEST read);
* X5  no completeness check of the TEST windows (-> ``completeness`` / ``provenance`` / ``inputs_check``);
* X6  DEV selection and DEV power estimates on the same windows (-> DEV selection / estimation halves);
* X7  the static comparator of H1c had no DEV access (-> DEV choice among the TRAIN fit's top candidates);
* X8  bootstrap inference from fewer than two dates (-> ``MIN_DATES``);
* X9  crossed bootstrap missing for H5b, testing another hypothesis for H4a;
* X10 H7a 'holds' checked the sign criterion only; X11 H7c was circular (-> out-of-sample ``fluid_cal``);
* X12 figure series styles not from spec/controllers.yaml; X13 typed setting macros; X14 'offered_load' had two
      meanings; X15 E2-E4 seed choice recorded (n_seeds).

Nothing here reads TEST data: every TEST path is exercised with the freeze gate replaced by a stub, and the TEST
readers that must not be reached are replaced by functions that fail if called.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1]
for _p in (CODE / "experiments", CODE / "experiments" / "figures", CODE / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyze as A  # noqa: E402
import e2_manipulation as X  # noqa: E402
import registry as R  # noqa: E402
from slicemech import io as sio  # noqa: E402
from slicemech.config import Mechanism  # noqa: E402

MECH = Mechanism()


def _calibration_or_skip():
    from slicemech.data.calibrate import load_calibration
    try:
        return load_calibration("n3_default")
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"stored calibration missing: {exc}")


def _synthetic_tree(root: Path, rng_seed: int = 7) -> dict:
    """The synthetic DEV driver outputs of tests/test_exp_analysis.py (known effects, 6 DEV dates)."""
    import test_exp_analysis as TA
    rng = np.random.default_rng(rng_seed)
    TA.make_e0(root)
    TA.make_e1(root, rng)
    TA.make_e2(root, rng)
    TA.make_e3(root, rng)
    TA.make_e4(root, rng)
    TA.make_e5(root, rng)
    TA.make_e6(root, rng)
    TA.make_e7(root / "e7")
    return {e: root / e for e in A.EXPERIMENTS}


# ====================================================================================== X1 H5b sign
def test_x1_h5b_counts_only_the_preregistered_side():
    neg = dict(estimate=-0.99, lo=-1.2, hi=-0.8)               # the smoke run's significant later LOSS
    up = A.h5b_decision(neg, sign=1)
    assert up["rule_met"] is False and up["contrary"] is True and up["sign_status"].startswith("contrary")
    down = A.h5b_decision(neg, sign=-1)
    assert down["rule_met"] is True and down["contrary"] is False
    assert A.h5b_decision(dict(estimate=0.1, lo=-0.2, hi=0.4), sign=1)["rule_met"] is False
    none = A.h5b_decision(neg) if A.H5B_SIGN is None else A.h5b_decision(neg, sign=0)
    assert none["rule_met"] is None and none["dev_sign_suggestion"] == -1
    assert A.h5b_decision(dict(estimate=1.0, lo=None, hi=None), sign=1)["rule_met"] is None   # no inference
    # a contrary H5b is never a supported primary, whatever Holm says
    res = {"e1": dict(hypotheses=dict(H1a=dict(rule_met=True, p=1e-4))),
           "e2": dict(hypotheses=dict(H2c=dict(rule_met=True, p=1e-4))),
           "e4": dict(hypotheses=dict(H4a=dict(rule_met=True, noninferiority_p=1e-4))),
           "e5": dict(hypotheses=dict(H5b=dict(up, p=1e-4)))}
    v = {r["name"]: r for r in A.holm_summary(res)["tables"]["verdicts"]}
    assert v["H5b"]["holm_reject"] and v["H5b"]["contrary"] and v["H5b"]["supported"] is False


def test_x1_test_analysis_needs_the_preregistered_sign_and_no_redirected_inputs(monkeypatch, tmp_path, capsys):
    from slicemech.data import puffer_fetch
    calls = []
    monkeypatch.setattr(puffer_fetch, "require_test_read_gate", lambda tf=None: calls.append(tf) or dict(path="x"))
    monkeypatch.setattr(A, "H5B_SIGN", None)
    assert A.main(["--split", "test", "--out", str(tmp_path / "never")]) == 3
    assert "H5B_SIGN" in capsys.readouterr().err and not calls          # refused before the TEST gate
    monkeypatch.setattr(A, "H5B_SIGN", 1)
    assert A.main(["--split", "test", "--source", "e2=results/raw/dev/e2", "--out", str(tmp_path / "never")]) == 3
    assert "--source" in capsys.readouterr().err
    assert not (tmp_path / "never").exists()


# ====================================================================================== X2 verdicts need Holm
def test_x2_a_rule_without_a_holm_rejection_is_not_supported():
    """The reviewer's case: H1a's CI excludes 0 (one-sided p 0.024) but Holm with p = 0.3 / 0.4 / 0.6 does not reject."""
    res = {"e1": dict(hypotheses=dict(H1a=dict(rule_met=True, p=0.0241))),
           "e2": dict(hypotheses=dict(H2c=dict(rule_met=False, p=0.3))),
           "e4": dict(hypotheses=dict(H4a=dict(rule_met=True, noninferiority_p=0.4))),
           "e5": dict(hypotheses=dict(H5b=dict(rule_met=None, p=0.6)))}
    h = A.holm_summary(res)
    v = {r["name"]: r for r in h["tables"]["verdicts"]}
    assert v["H1a"]["p_holm"] == pytest.approx(0.0964) and v["H1a"]["holm_reject"] is False
    assert v["H1a"]["rule_met"] is True and v["H1a"]["supported"] is False
    assert h["macros"]["EoneSupported"]["value"] == "no" and v["H5b"]["supported"] is None
    assert "EfiveSupported" not in h["macros"]
    res["e1"]["hypotheses"]["H1a"]["p"] = 1e-4
    for e, k in (("e2", "H2c"), ("e4", "H4a")):
        res[e]["hypotheses"][k].update(rule_met=True, p=1e-4, noninferiority_p=1e-4)
    v = {r["name"]: r for r in A.holm_summary(res)["tables"]["verdicts"]}
    assert v["H1a"]["supported"] and v["H2c"]["supported"] and v["H4a"]["supported"]
    # on TEST the inputs must also be complete and of one provenance
    res["e1"]["inputs_check"] = dict(ok=False)
    for e in ("e2", "e4"):
        res[e]["inputs_check"] = dict(ok=True)
    v = {r["name"]: r for r in A.holm_summary(res, split="test")["tables"]["verdicts"]}
    assert v["H1a"]["supported"] is False and v["H2c"]["supported"] is True
    assert v["H5b"]["supported"] is None and v["H5b"]["reason"] == "rule not evaluable"


# ====================================================================================== X3 Holm over every summary
def test_x3_holm_is_never_rebuilt_from_a_subset(tmp_path):
    src = _synthetic_tree(tmp_path / "raw")
    out = tmp_path / "summary"
    A.run_analysis("dev", out, n_boot=200, src=src, crossed=False, confine=False, verbose=False)
    full = json.loads((out / "holm.json").read_text())
    assert full["hypotheses"]["complete"]
    for exps in (["e1"], ["e7"]):
        A.run_analysis("dev", out, n_boot=200, src=src, crossed=False, confine=False, verbose=False, exps=exps)
        sub = json.loads((out / "holm.json").read_text())
        assert sub["hypotheses"]["complete"] and sub["hypotheses"]["missing"] == []
        assert [r["p_holm"] for r in sub["tables"]["holm"]] == [r["p_holm"] for r in full["tables"]["holm"]]
        assert set(sub["macros"]) >= {"EonePHolm", "EtwoPHolm", "EfourPHolm", "EfivePHolm"}
        assert sub["family_sources"]["e2"] == "existing summary"


# ====================================================================================== X8 minimum dates
def test_x8_no_inference_below_the_minimum_number_of_dates():
    rng = np.random.default_rng(0)
    for nd in (1, 2, A.MIN_DATES - 1):
        p = pd.DataFrame(dict(date=[f"d{j % nd}" for j in range(12)], unit=[f"u{j}" for j in range(12)],
                              a=rng.normal(1.0, 0.1, 12), b=rng.normal(10.0, 0.1, 12)))
        r = A.contrast(p, n_boot=200, alternative="greater", pct_null=1.0, pct_alternative="less")
        assert r["inference"] is False and not math.isfinite(r["p"]) and not math.isfinite(r["lo"])
        assert r["estimate"] == pytest.approx(float((p["a"] - p["b"]).mean()))
        assert not math.isfinite(r["p_pct"]) and "crossed" not in r
        js = A._clean(r)
        assert js["lo"] is None and js["p"] is None
        assert A._rule(r["lo"] > 0, r["lo"]) is None
        sub = A.load_subgroups(p.assign(offered_load=0.9), n_boot=200, alternative="greater")
        assert all("estimate" not in b for b in sub["bins"]) and sub["slope"] is None
    p = pd.DataFrame(dict(date=[f"d{j % 5}" for j in range(10)], unit=[f"u{j}" for j in range(10)],
                          a=rng.normal(1.0, 0.1, 10), b=0.0))
    r = A.one_sample(p, "a", n_boot=200)
    assert r["inference"] is True and r["lo"] > 0


# ====================================================================================== X9 crossed bootstrap
def test_x9_crossed_bootstrap_tests_h4a_on_its_margin_and_covers_h5b(tmp_path):
    rng = np.random.default_rng(3)
    n = 24
    p = pd.DataFrame(dict(date=[f"d{j // 4}" for j in range(n)], unit=[f"u{j}" for j in range(n)],
                          a=rng.normal(0.2, 0.05, n), b=rng.normal(50.0, 1.0, n)))
    p["a"] = p["a"] + p["b"]                                    # a - b ~ 0.2 credits = 0.4 % of b
    mem = {u: {"date": {d: 1.0}, "tractor_urllc": {f"c{j % 3}": 1.0}} for j, (u, d) in enumerate(zip(p["unit"],
                                                                                                    p["date"]))}
    r = A.contrast(p, n_boot=300, alternative="greater", pct_null=A.H4A_MARGIN_PCT, pct_alternative="less", mem=mem)
    c = r["crossed"]
    assert {"pct_upper_one_sided", "p_pct", "pct_lo", "pct_hi"} <= set(c) and c["pct_null"] == A.H4A_MARGIN_PCT
    assert c["pct_upper_one_sided"] < A.H4A_MARGIN_PCT and c["p_pct"] < 0.05      # tests the margin, not 0
    # H5b: analyze_e5 passes the E5 memberships (pool-half compositions) to the crossed bootstrap
    src = _synthetic_tree(tmp_path / "raw")
    T = {k: pd.concat(v, ignore_index=True) for k, v in A.load_e5(src["e5"]).items()}
    later = T["e5_pairs"][T["e5_pairs"]["phase"] == "later"]
    mem5 = {u: {"date": {u.split("|")[0]: 1.0}, "raca_run": {str(i % 4): 1.0}}
            for i, u in enumerate(sorted(later["unit"].unique()))}
    res = A.analyze_e5(T, dict(n_boot=300, memberships_e5=mem5, split="train"))
    assert "crossed" in res["hypotheses"]["H5b"] and res["hypotheses"]["H5b"]["crossed"]["factors"] == [
        "date", "raca_run"]


# ====================================================================================== X10 / X11 E7
def _nr_rows(nr_loss: float, fluid_loss: float, date: str):
    rows = []
    for arm in ("protected_truthful", "unconstrained_truthful", "unconstrained_br"):
        for t, c in enumerate(("video", "urllc", "mmtc")):
            hon = arm == "unconstrained_br" and t != 1
            rows.append(dict(arm=arm, date=date, offset=60, tenant=t, br_tenant=1, cls=c,
                             nr_delivered_pdb_mbit=10.0 - (nr_loss if hon else 0.0),
                             nr_delivered_mbit=10.0 - (nr_loss if hon else 0.0), nr_pdcp_sdu_mbit=11.0,
                             nr_rlc_drop_mbit=0.1, nr_delay_p95_ms_median=20.0,
                             fluid_raca_on_time=10.0 - (fluid_loss if hon else 0.0), corr_delivered_fluid_cal=0.9,
                             corr_delivered_fluid_nr=0.999))
    return rows


def test_x10_h7a_needs_sign_and_magnitude_and_zero_effects_do_not_agree():
    nr = pd.DataFrame(sum((_nr_rows(2.0, 0.4, f"2024-09-0{j + 1}") for j in range(4)), []))   # ratio 5: sign only
    h = A.analyze_e7(dict(nr=nr), dict(n_boot=10))["hypotheses"]["H7a"]
    assert h["sign_criterion"] and not h["magnitude_criterion"] and h["holds"] is False
    nr = pd.DataFrame(sum((_nr_rows(0.0, 0.0, f"2024-09-0{j + 1}") for j in range(4)), []))   # zero = no sign
    h = A.analyze_e7(dict(nr=nr), dict(n_boot=10))["hypotheses"]["H7a"]
    assert h["sign_agrees"] == 0 and h["holds"] is False
    nr = pd.DataFrame(sum((_nr_rows(1.0, 0.8, f"2024-09-0{j + 1}") for j in range(4)), []))
    assert A.analyze_e7(dict(nr=nr), dict(n_boot=10))["hypotheses"]["H7a"]["holds"] is True


def test_x11_h7c_uses_the_out_of_sample_calibration_rate(tmp_path):
    import e7_nr as E7
    from test_exp_e7 import _kpi_for, _toy_window
    from slicemech.sim import fluid
    w = _toy_window()
    q = np.tile(np.array([40, 6, 18]), (w.K, 1))
    kpi = _kpi_for(w, q, w.rate_true, tmp_path)
    paths = E7.Paths(out=tmp_path / "nr", staging=tmp_path / "stage", summary=tmp_path)
    runs = [dict(radius=120, rng_run=r, rate_by_class=dict(video=0.40 + 0.01 * r, urllc=0.30, mmtc=0.20))
            for r in (1, 2)] + [dict(radius=60, rng_run=1, rate_by_class=dict(video=9.0, urllc=9.0, mmtc=9.0))]
    (tmp_path / "nr_calibration.json").write_text(json.dumps(dict(chosen=120, runs=runs)))
    rates = E7.calibrated_rates(paths, 120)
    assert rates == pytest.approx((0.415, 0.30, 0.20)) and E7.calibrated_rates(paths, 200) is None
    t = E7.fluid_table(w, q, q, kpi, rates)
    A_ = fluid.cumulative_arrivals(w.arrivals_ms)
    ref, _, _ = fluid.replay(A_, q.astype(float), np.tile(np.array(rates), (w.K, 1)), w.contract.delay_ms,
                             late_counted=True)
    np.testing.assert_array_equal(t["fluid_cal_served"], ref.served_Mbit.ravel())    # NR outcome never enters
    assert t["fluid_nr_served"].notna().all()
    assert E7.fluid_table(w, q, q, kpi)["fluid_cal_served"].isna().all()
    assert E7.H7C_VARIANT == "fluid_cal" and "corr_delivered_fluid_cal" in E7.case_summary(t)["per_tenant"][0]


# ====================================================================================== X4 TEST protocol and assets
@pytest.fixture
def gate_open(monkeypatch, tmp_path):
    """The TEST gate replaced by a stub that passes (as after the freeze); every TEST reader fails if reached."""
    from slicemech.data import calibrate, compose, puffer, puffer_fetch

    def boom(*a, **k):
        raise AssertionError("a TEST source was touched")
    freeze = tmp_path / "PREREG.md"
    freeze.write_text("# frozen\n")
    monkeypatch.setattr(puffer_fetch, "require_test_read_gate", lambda tf=None: dict(path=str(freeze)))
    for m, name in ((compose, "compose_window"), (compose, "compose_parts"), (calibrate, "counts_for_date"),
                    (puffer, "effective_test_dates")):
        monkeypatch.setattr(m, name, boom)
    return freeze


@pytest.mark.parametrize("argv", [["--dates", "2024-09-01"], ["--max-windows", "3"], ["--seeds", "0", "1"],
                                  ["--ckpt-dir", "x"], ["--train-missing"], ["--tag", "rerun"],
                                  ["--arms", "unc_fixed"], ["--bandit-designs"]])
def test_x4_e2_refuses_protocol_options_on_test(gate_open, argv):
    with pytest.raises(SystemExit, match="refusing --split test with"):
        X.main(["--split", "test", "--plan", *argv])


def test_x4_e0_to_e7_and_analysis_refuse_protocol_options_on_test(gate_open, monkeypatch, tmp_path):
    import e0_setup as E0
    import e1_value as E1
    import e3_protection as E3
    import e4_cost as E4
    import e5_training_channel as E5
    import e6_scaling as E6
    import e7_nr as E7
    for main, argv in ((E3.main, ["--ckpt", "protected=x"]), (E4.main, ["--e2-root", "x"]),
                       (E4.main, ["--kappa-scales", "0", "3"])):
        with pytest.raises(SystemExit, match="refusing --split test"):
            main(["--split", "test", "--plan", *argv])
    for main, argv in ((E5.main, ["--lr", "0.01"]), (E5.main, ["--dates", "2024-09-01"]),
                       (E5.main, ["--models-root", "x"]), (E6.main, ["--br-every", "2"]),
                       (E6.main, ["--ckpt-template", "x"])):
        with pytest.raises(SystemExit, match="refusing --split test"):
            main(["--split", "test", "--plan", *argv])
    for main, argv in ((E0.main, ["--limit", "3"]), (E1.main, ["--dates", "2024-09-01"]),
                       (E1.main, ["--allow-missing"])):
        with pytest.raises(SystemExit, match="refusing --split test"):
            main(["--split", "test", *argv])
    paths = E7.Paths(out=tmp_path / "nr", staging=tmp_path / "st", summary=tmp_path / "sum")
    other_offset = "60" if E7.CASE_OFFSET_S == 43260 else "43260"   # an offset other than the frozen constant
    for argv in (["--disc-radius", "60"], ["--case-offset", other_offset]):
        with pytest.raises(SystemExit, match="refusing --split test"):
            E7.main(["--split", "test", "--stage", "cases", "--binary", sys.executable, *argv], paths=paths)
    monkeypatch.setattr(E7, "CASE_OFFSET_S", 43260)              # the frozen value itself may be passed
    with pytest.raises(PermissionError):                         # ... and then the checkpoint rule applies
        E7.require_split_allowed("test", "cases", None, {"protected": __file__, "unconstrained": __file__})
    assert E7.expected_ckpt("protected") == R.model_dir("default", "protected", 0) / R.MODEL_FILE


def _registry_model(root: Path, kind: str, seed: int = 0) -> Path:
    from slicemech.control import features as F
    from slicemech.control import nets as N
    d = root / kind / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    net = N.make_net(kind, seed=seed)
    net.forecaster = F.OperatorAR.default()
    N.save_net(net, d / "model.pt")
    (d / "config.json").write_text(json.dumps(dict(status="done", result=dict(
        model_sha256=sio.file_sha256(d / "model.pt")))))
    return d / "model.pt"


def _freeze_text(paths) -> str:
    ck = "".join(f"  {p}: {sio.file_sha256(p)}\n" for p in paths)
    return (f"# frozen\n\n```freeze\nanalysis_code_sha256: {'a' * 64}\nconfig_sha256: {'b' * 64}\ncheckpoints:\n"
            f"{ck}date_rule: Sundays and Mondays\ntest_dates: 16\n```\n")


def test_x4_test_assets_must_be_frozen_registry_files(monkeypatch, tmp_path):
    reg, old = tmp_path / "models_default", tmp_path / "checkpoints_default"
    net = _registry_model(reg, "protected")
    (reg / "static").mkdir()
    (reg / "static" / "static.json").write_text('{"w": [0.5, 0.3, 0.2], "status": "done"}')
    monkeypatch.setattr(X, "MODELS_DIR", reg)
    monkeypatch.setattr(X, "CKPT_DIR", old)
    freeze = tmp_path / "PREREG.md"
    record = net.parent / "config.json"                           # pre-freeze audit E5-7: frozen with its model
    freeze.write_text(_freeze_text([net, record]))
    with pytest.raises(PermissionError, match="static"):          # the static fit is not in the freeze block
        X.resolve_assets(["prot_fixed", "static_full"], [0], smoke=False, test_freeze_path=freeze)
    freeze.write_text(_freeze_text([net, reg / "static" / "static.json"]))
    with pytest.raises(PermissionError, match="config.json"):     # the model's job record is not frozen
        X.resolve_assets(["prot_fixed", "static_full"], [0], smoke=False, test_freeze_path=freeze)
    freeze.write_text(_freeze_text([net, record, reg / "static" / "static.json"]))
    specs = X.resolve_assets(["prot_fixed", "static_full"], [0], smoke=False, test_freeze_path=freeze)
    assert specs[0].nets["protected"] == str(net)
    for kw in (dict(ckpt_dir=reg), dict(overrides=["protected=x"]), dict(train_missing=True)):
        with pytest.raises(PermissionError, match="frozen training registry"):
            X.resolve_assets(["prot_fixed"], [0], smoke=False, test_freeze_path=freeze, **kw)
    # the E0/E1 fallback layout is never searched on TEST
    shutil.move(str(reg / "protected"), str(tmp_path / "moved"))
    _registry_model(old, "protected")
    with pytest.raises(FileNotFoundError):
        X.resolve_assets(["prot_fixed"], [0], smoke=False, test_freeze_path=freeze)
    assert X.resolve_assets(["prot_fixed"], [0], smoke=False)[0].nets["protected"].startswith(str(old))
    with pytest.raises(PermissionError, match="not frozen"):
        R.verify_frozen([old / "protected" / "seed0" / "model.pt"], freeze)
    # the usage-tariff caches are part of the freeze block (every TEST asset can be checked)
    monkeypatch.setattr(R, "USAGE_TARIFF_DIR", tmp_path / "tariff")
    (tmp_path / "tariff").mkdir()
    (tmp_path / "tariff" / "nu_protected_seed0_abc.json").write_text("{}")
    assert tmp_path / "tariff" / "nu_protected_seed0_abc.json" in R.checkpoint_paths()


@pytest.fixture
def freeze_env(monkeypatch):
    from slicemech.data import puffer as PU
    from slicemech.data import puffer_fetch as PF
    d = sio.DATA_DIR / f"_pytest_r3_freeze_{os.getpid()}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "prereg").mkdir(parents=True)
    monkeypatch.setattr(PF, "PREREG_DIR", d / "prereg")
    monkeypatch.setattr(PU, "TEST_RAW_DIR", d / "puffer_raw")
    yield d, PF
    shutil.rmtree(d, ignore_errors=True)


def test_x4_test_reads_recheck_the_code_hash_after_the_download(freeze_env, monkeypatch):
    d, PF = freeze_env
    ckpt = d / "ckpt.pt"
    ckpt.write_bytes(b"parameters")
    frozen = d / "prereg" / "PREREG.md"
    frozen.write_text("# Pre-registration (frozen)\n\n" + PF.freeze_block([ckpt], "Sundays and Mondays"),
                      encoding="utf-8")
    rec = PF._require_freeze(frozen)
    sio.write_json(PF.manifest_path(), dict(objects={}, replacements={}, failures={}, freeze=rec))
    assert PF.require_freeze_for_test_reads(frozen)["deviations"] == []
    changed = "c" * 64                                            # code edited after the TEST download
    monkeypatch.setattr(PF, "analysis_code_sha256", lambda: changed)
    with pytest.raises(PermissionError, match="not logged"):
        PF.require_freeze_for_test_reads(frozen)
    with pytest.raises(PermissionError, match="not logged"):
        PF.require_test_read_gate()                              # the recorded freeze, same check
    (d / "prereg" / "DEVIATIONS.md").write_text(f"2026-11-02: fix X; new analysis_code_sha256: {changed}\n")
    r = PF.require_freeze_for_test_reads(frozen)
    assert r["deviations"] == ["analysis_code_sha256"] and r["current_analysis_code_sha256"] == changed


def test_x4_job_outputs_carry_provenance_and_stale_outputs_are_replaced(tmp_path):
    spec_a = X.AssetSpec(seed=0, nets={}, sha256=dict(protected="p" * 64, static="s" * 64))
    spec_b = X.AssetSpec(seed=0, nets={}, sha256=dict(protected="q" * 64, static="s" * 64))
    out = sio.ROOT / "results" / "raw" / f"_pytest_r3_{os.getpid()}"
    try:
        def jobs():
            return [X.Job("prot", "e2_manipulation:job_loads", dict(arm="prot_fixed", seed=0, out=str(out / "a.p")),
                          str(out / "a.p")),
                    X.Job("eq", "e2_manipulation:job_loads", dict(arm="equal_full", seed=0, out=str(out / "b.p")),
                          str(out / "b.p"))]
        ja = X.attach_provenance(jobs(), {0: spec_a}, split="dev", smoke=False, modules=X.E2_MODULES)
        jb = X.attach_provenance(jobs(), {0: spec_b}, split="dev", smoke=False, modules=X.E2_MODULES)
        assert ja[0].fp != jb[0].fp and ja[1].fp == jb[1].fp     # only the assets a job uses enter its fingerprint
        assert ja[0].prov["assets"] == {"protected": "p" * 64}
        j = ja[0]
        assert X.job_status(j) == "pending"
        X.write_parquet(pd.DataFrame({"x": [1]}), Path(j.out))
        assert X.job_status(j) == "stale"                        # an output without provenance is not reused
        sio.write_json(X.prov_path(j.out), dict(fp=j.fp))
        assert X.job_status(j) == "done" and X.job_status(jb[0]) == "stale"
        res = X.run_jobs([jb[0]], workers=1, initargs=(), errors_path=out / "err.jsonl", plan=True)
        assert res["pending"] == 1 and res["stale"] == 1
        a = argparse.Namespace(split="dev", smoke=False)
        X.write_config(out, a, {0: spec_a}, dict(experiment="E2"))
        X.write_config(out, a, {0: spec_b}, dict(experiment="E2"))
        hist = (out / "config_history.jsonl").read_text().splitlines()
        assert len(hist) == 2 and json.loads(hist[0])["assets"]["0"]["sha256"]["protected"] == "p" * 64
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _write_two(out: str, replay_out: str) -> None:
    """Job body of the sidecar test: writes a main output and a second (replay-like) output."""
    X.write_parquet(pd.DataFrame({"x": [1]}), Path(replay_out))
    X.write_parquet(pd.DataFrame({"x": [2]}), Path(out))


def test_x4_every_output_of_a_job_gets_a_sidecar():
    out = sio.ROOT / "results" / "raw" / f"_pytest_r3_two_{os.getpid()}"
    try:
        job = X.Job("two", "test_review_round3:_write_two", dict(out=str(out / "stage" / "s.parquet"),
                                                                 replay_out=str(out / "replay" / "r.parquet")),
                    str(out / "stage" / "s.parquet"), fp="e" * 64, prov=dict(job="two", code="c", args=dict(seed=0)))
        X._call(job)
        for f in ("stage/s.parquet", "replay/r.parquet"):
            assert sio.read_json(X.prov_path(out / f))["fp"] == "e" * 64
        assert X.job_status(job) == "done"
        pv = A.provenance({"e2": out}, {})["e2"]
        assert pv["outputs"] == 2 and pv["outputs_without_sidecar"] == 0 and pv["ok"]
    finally:
        shutil.rmtree(out, ignore_errors=True)


@pytest.mark.skipif(not (sio.DATA_DIR / "calibration" / "n3_default.json").exists(), reason="needs the calibration")
def test_x4_runner_deletes_a_stale_output_and_writes_the_sidecar(tmp_path):
    out = sio.ROOT / "results" / "raw" / f"_pytest_r3_pool_{os.getpid()}"
    try:
        good = out / "loads.parquet"
        X.write_parquet(pd.DataFrame({"stale": [1]}), good)
        sio.write_json(X.prov_path(good), dict(fp="0" * 64))
        job = X.Job("loads", "e2_manipulation:job_loads", dict(date="2024-01-01", offset=60, out=str(good)), str(good))
        job.fp, job.prov = "f" * 64, dict(job="loads")
        spec = X.AssetSpec(seed=0, nets={})
        res = X.run_jobs([job], workers=1, initargs=({0: spec}, "train", False, None), errors_path=out / "e.jsonl")
        assert res["stale"] == 1 and res["failed"] == 0
        assert "offered_load" in pd.read_parquet(good).columns and sio.read_json(X.prov_path(good))["fp"] == "f" * 64
        assert X.job_status(job) == "done"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_x4_e5_e6_resume_only_with_the_same_fingerprint(tmp_path):
    import e5_training_channel as E5
    import e6_scaling as E6
    m = tmp_path / "model.pt"
    m.write_bytes(b"one")
    job = dict(job="2024-03-02__protected__s0", split="dev", date="2024-03-02", seed=0, arm="protected",
               model=dict(path=str(m)), d=2, layout=[[60, "A"]], profiles=[["none", 1.0]], overrides={}, ft_iters=50,
               lr=3e-4, M=4, sigma0=0.5, init_log_sigma=-0.7, cal="n3_default", out=str(tmp_path), test_freeze=None)
    code = E5.code_fingerprint()
    job["fp"] = E5.job_fingerprint(job, code)
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / f"{job['job']}.done.json").write_text(json.dumps(dict(fp=job["fp"])))
    assert E5.job_done(tmp_path, job)
    m.write_bytes(b"two")                                         # another model: the record is stale
    assert E5.job_fingerprint(job, code) != job["fp"]
    assert not E5.job_done(tmp_path, dict(job, fp=E5.job_fingerprint(job, code)))
    (tmp_path / "jobs" / f"{job['job']}.done.json").write_text("{}")   # an old record without a fingerprint
    assert not E5.job_done(tmp_path, job)
    j6 = dict(job="n6__2024-03-02__o00060", n=6, models={"protected|0": dict(path=str(m))}, out="x", fp="abc")
    (tmp_path / "jobs" / f"{j6['job']}.done.json").write_text(json.dumps(dict(fp="abc")))
    assert E6.job_done(tmp_path, j6) and not E6.job_done(tmp_path, dict(j6, fp="abd"))


# ====================================================================================== X5 completeness, provenance
def test_x5_completeness_and_provenance_are_checked(tmp_path):
    src = _synthetic_tree(tmp_path / "raw")
    T = A.load_all("dev", False, [], src)
    comp = A.completeness(T, "dev")
    assert comp["e5"]["jobs"]["expected"] == 21 * 3 * 2 and not comp["e5"]["jobs"]["complete"]
    assert not comp["e1"]["protected"]["complete"]                # 4 of 6 offsets, 2 of 5 seeds
    assert comp["e1"]["oracle"]["present"] == 0                   # an expected arm without any row
    assert comp["e2"]["replay"]["unc_fixed"]["expected"] == 42
    prov = A.provenance(src, T)
    assert prov["e2"]["outputs_without_sidecar"] == prov["e2"]["outputs"] > 0 and not prov["e2"]["ok"]
    # sidecars of one experiment that name two different checkpoints are a provenance conflict
    root = tmp_path / "prov" / "e2"
    for i, sha in enumerate(("a" * 64, "b" * 64)):
        f = root / "stage" / "unc_fixed" / "s0" / f"w{i}.parquet"
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": [i]}).to_parquet(f)
        Path(str(f) + ".prov.json").write_text(json.dumps(dict(code="c", args=dict(seed=0),
                                                               assets=dict(unconstrained=sha))))
    pv = A.provenance({"e2": root}, {})
    assert pv["e2"]["asset_conflicts"] == {"unconstrained|seed0": 2} and not pv["e2"]["ok"]
    chk = A.inputs_check("e2", {"e2": dict(x=dict(expected=1, complete=True))}, pv)
    assert chk["ok"] is False


# ====================================================================================== X6 DEV halves
def test_x6_dev_selection_and_estimation_halves_are_disjoint():
    import train as TR
    from slicemech.data import puffer
    dates = list(puffer.split_dates("dev"))
    sel, est = R.dev_dates("select"), R.dev_dates("estimate")
    assert len(sel) == 11 and len(est) == 10 and not set(sel) & set(est) and set(sel) | set(est) == set(dates)
    assert sel == dates[0::2] and est == dates[1::2]                 # alternating: both halves span Feb-May
    assert len(R.window_keys("dev", role="select")) == 66 and len(R.window_keys("dev", role="estimate")) == 60
    ctx = TR.make_ctx("dev")
    assert TR._selection_keys(ctx) == [[d, o] for d, o in R.window_keys("dev", role="select")]
    assert TR.n_windows("dev", False) == 66
    assert TR._selection_keys(TR.make_ctx("dev", smoke=True)) == [list(k) for k in R.SMOKE_DEV_WINDOWS]
    with pytest.raises(ValueError):
        R.window_keys("test", role="select")
    # the DEV power inputs read the estimation half only
    rows = [dict(date=d, unit=f"{d}|o{o}", a=(10.0 if R.dev_role(d) == "select" else 1.0) + 0.01 * k, b=0.0)
            for d in dates for k, o in enumerate((60, 14460))]
    pw = A.power_inputs(pd.DataFrame(rows), "H1a", n_boot=200)
    assert pw["dates"] == 10 and pw["estimate"] == pytest.approx(1.005) and pw["dates_needed_point"] >= A.MIN_DATES


# ====================================================================================== X7 static DEV choice
def test_x7_static_vector_is_chosen_on_dev_among_the_train_candidates(monkeypatch):
    import train as TR
    from slicemech.control import static
    C = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.6, 0.3, 0.1], [0.3, 0.3, 0.4], [0.2, 0.2, 0.6]])
    sc = np.array([5.0, 4.0, 4.5, 3.0, 1.0])                       # TRAIN order: 0, 2, 1, 3, 4
    fit = static.StaticFit(C[0], 5.0, "grid", C, sc, 0.0, {})
    dev_of = {tuple(C[j]): v for j, v in enumerate((2.0, 2.5, 2.1, 2.9, 9.0))}
    monkeypatch.setattr(static, "rollout_fixed", lambda sel, W, **kw: np.array([dev_of[tuple(w)] for w in W]))
    ch = TR.static_dev_choice(fit, None, {}, top_k=4)
    assert np.allclose(ch["w"], C[3]) and ch["rank"] == 3 and ch["dev_welfare"] == 2.9   # C[4] is not a top-4 cand.
    assert ch["train_welfare"] == 3.0 and ch["dev_welfare_train_best"] == 2.0 and len(ch["table"]) == 4
    dev_of[tuple(C[1])] = 2.9                                       # a tie keeps the better TRAIN rank
    assert TR.static_dev_choice(fit, None, {}, top_k=4)["rank"] == 2
    assert R.STATIC_DEV_CANDIDATES == 72


# ====================================================================================== X12 figure styles
def test_x12_every_plotted_series_takes_its_style_from_controllers_yaml():
    import _common as C
    import fig_e2e3_gain
    import fig_e3_bandits
    import fig_e5
    import fig_worked_example
    keys = set(C.controllers())
    assert set(A.NR_CASES.values()) <= keys and A.NR_CASES["unconstrained_truthful"] == "unprotected"
    assert {k for _, k in fig_e5.PIPES} <= keys and "unprotected" not in {k for _, k in fig_e5.PIPES}
    assert C.style("retrain_stored")["color"] not in (C.style("unprotected")["color"], C.style("protected")["color"])
    assert C.style("unprotected_br")["color"] == C.style("unprotected")["color"]
    assert C.style("unprotected_br")["marker"] != C.style("unprotected")["marker"]
    C.setup()
    gm = [dict(cls="urllc", mult=m, arm=a, mean_gain=g, lo=g - 0.1, hi=g + 0.1)
          for a, g in (("protected", -0.1), ("fixed_price", 0.3)) for m in (0.5, 1.0, 2.0)]
    fig = fig_e2e3_gain.make(dict(tables=dict(gain_by_mult=gm)))
    labels = set(fig.legends[0].texts[i].get_text() for i in range(len(fig.legends[0].texts)))
    assert labels == {C.style("protected")["label"], C.style("fixed_price")["label"]}
    tr = [dict(arm=a, cls="urllc", episode=e, mean_mult=1.0, bin_width=10) for a in ("protected", "unprotected")
          for e in range(0, 400, 10)]
    fig = fig_e3_bandits.make(dict(tables=dict(bandit_trajectories=tr)))
    assert all(ln.get_marker() not in (None, "None", "") for ln in fig.axes[0].get_lines()[1:])
    ex = A.example_summary()
    fig = fig_worked_example.make(ex)
    assert f"{ex['tables']['points'][0]['theta']:g}" in fig.axes[1].get_ylabel()


def test_x8_figures_draw_points_without_a_confidence_interval():
    """Below MIN_DATES the summaries carry null bounds; the figure scripts draw the point without an error bar
    (the smoke chain used to stop in fig_e1_value)."""
    import matplotlib.pyplot as plt
    import _common as C
    import fig_e4_frontier
    C.setup()
    fig, ax = plt.subplots()
    eb = C.errorbar(ax, [1.0, 2.0], [0.5, 0.7], [None, 0.6], [None, 0.9], "protected")
    seg = eb.lines[2][0].get_segments()
    assert len(seg) == 2 and seg[0][0][1] == seg[0][1][1] == 0.5          # zero-length bar at the CI-free point
    plt.close(fig)
    rows = [dict(welfare=1.0, welfare_lo=None, welfare_hi=None), dict(welfare=2.0, welfare_lo=1.5, welfare_hi=2.4)]
    assert fig_e4_frontier._err(rows, "welfare") == [[0.0, 0.5], [0.0, pytest.approx(0.4)]]


# ====================================================================================== X13 setting macros
def test_x13_measured_and_selected_settings_are_computed(tmp_path):
    import yaml
    st = yaml.safe_load(open(A.SPEC / "params.yaml"))["experiment_settings"]
    assert not {"RateMean", "RateMin", "RateMax", "PpoLr", "PpoLrEnd", "PpoSamples"} & set(st)
    cal = _calibration_or_skip()
    from slicemech.phy import RATE_TABLE
    sel = dict(status="done", arms={a: dict(cell="c", params=dict(lr=lr, sigma0=0.3, M=4)) for a, lr in
                                    (("protected", 1e-4), ("unconstrained", 3e-4), ("meas_only", 1e-4),
                                     ("kappa0", 1e-4))})
    (tmp_path / "selection.json").write_text(json.dumps(sel))
    m = A.settings_summary(models_root=tmp_path)["macros"]
    assert m["RateMean"]["value"] == pytest.approx(float(cal.r_bar)) and "{:.2f}".format(m["RateMean"]["value"]) == \
        "{:.2f}".format(float(cal.r_bar))
    assert m["RateMin"]["value"] == pytest.approx(float(RATE_TABLE[1]))
    assert m["RateMax"]["value"] == pytest.approx(float(RATE_TABLE[15]))
    assert m["PpoLrProtected"]["value"] == "$1\\times10^{-4}$" and m["PpoLrUnconstrained"]["value"] == \
        "$3\\times10^{-4}$"
    assert "Unconstrained net" in m["PpoLr"]["value"] and m["PpoSamples"]["value"] == "4"
    assert "PpoLr" not in A.settings_summary(models_root=tmp_path / "none")["macros"]


def test_x13_e1_delay_macro_comes_from_the_run_record(tmp_path):
    import test_exp_analysis as TA
    TA.make_e1(tmp_path, np.random.default_rng(1))
    T = {k: pd.concat(v, ignore_index=True) for k, v in A.load_e1(tmp_path / "e1").items()}
    r = A.analyze_e1(T, dict(n_boot=100, split="train"))
    assert "EoneDelay" not in r["macros"] and any("EoneDelay" in n for n in r["notes"])
    (tmp_path / "e1" / "default" / "manifest.json").write_text(json.dumps(dict(config=dict(d=2, rho=0.9, n=3,
                                                                                          lead_s=4.004))))
    T = {k: pd.concat(v, ignore_index=True) for k, v in A.load_e1(tmp_path / "e1").items()}
    m = A.analyze_e1(T, dict(n_boot=100, split="train"))["macros"]["EoneDelay"]
    assert m["value"] == 2.0 and "manifest" in m["source"]


# ====================================================================================== X14 offered load
def test_x14_offered_load_means_all_intervals_in_every_driver():
    import e5_training_channel as E5
    import e7_nr as E7
    from slicemech.eval.headroom import offered_load
    from test_exp_e2e4 import make_window
    w = make_window(3, K=14, warmup=4)
    w.meta.update(offset_s=60, counts=[1, 1, 1])
    rb = 0.43
    m = np.asarray(w.measured, float)
    allk = float(m.mean(0).sum() / (MECH.B * rb))                  # E0/E1 definition
    scored = float(m[w.warmup:].sum(1).mean() / (MECH.B * rb))
    assert offered_load(w, rb, MECH) == allk
    assert offered_load(w, rb, MECH, scored_only=True) == pytest.approx(scored, rel=1e-12)
    info = E5.window_info(w, rb, MECH)
    assert info["offered_load"] == allk and info["offered_load_scored"] == pytest.approx(scored, rel=1e-12)
    assert E7.offered_load(w, rb, MECH.B) == allk
    assert E7.offered_load(w, rb, MECH.B, scored_only=True) == pytest.approx(scored, rel=1e-12)
    import inspect
    import e6_scaling as E6
    src = inspect.getsource(E6.run_job)
    assert "offered_load=offered_load(w, cal.r_bar, mech)" in src and "offered_load_scored" in src


# ====================================================================================== X15 seeds recorded
def test_x15_e2_to_e4_seed_choice_is_a_module_constant_and_is_reported():
    assert X.CHECKPOINT_SEEDS == (0,)
    ap = X.common_args(argparse.ArgumentParser())
    assert ap.get_default("seeds") == list(X.CHECKPOINT_SEEDS)
    BR = pd.DataFrame(dict(price="full", arm="protected", gain=0.0, cert=0.0, eps=1e-8, exact=True, n=3, cls="video",
                           seed=[0, 0, 0]))
    res, _ = A.h3a(BR)
    assert res["checkpoint_seeds"] == [0] and "seed" in res["seed_note"]


def test_x4_driver_docstrings_record_the_round3_rules():
    for mod in ("e2_manipulation", "e5_training_channel", "e6_scaling", "e7_nr", "train", "registry", "analyze"):
        txt = (CODE / "experiments" / f"{mod}.py").read_text()
        assert "round-3 review" in txt, mod
