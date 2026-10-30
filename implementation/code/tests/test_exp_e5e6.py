"""Tests of the E5 (training channel) and E6 (scaling) drivers in ``code/experiments``.

They run the drivers' own code paths on short real TRAIN windows (skipped when the prepared data or the
calibrations are missing) and write only below pytest's ``tmp_path``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parents[1] / "experiments"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"exp_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e5 = _load("e5_training_channel")
e6 = _load("e6_scaling")


def _cal(name="n3_default"):
    calib = pytest.importorskip("slicemech.data.calibrate")
    try:
        return calib.load_calibration(name)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"calibration missing: {exc}")


def _window(cal, *a, **kw):
    try:
        return cal.window(*a, **kw)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"prepared data missing: {exc}")


# ============================================================================================ TEST gate
@pytest.mark.parametrize("mod", [e5, e6], ids=["e5", "e6"])
def test_test_split_is_refused_before_the_freeze(mod, monkeypatch, tmp_path):
    """--split test raises PermissionError before any TEST source, date list or window is touched."""
    from slicemech.data import calibrate, compose, puffer, puffer_fetch, raca, tractor

    def boom(*a, **k):
        raise AssertionError("a TEST source was touched before the freeze gate")

    monkeypatch.setattr(puffer_fetch, "_recorded_freeze", lambda: None)
    for m, name in ((puffer, "split_dates"), (puffer, "effective_test_dates"), (compose, "compose_window"),
                    (compose, "compose_parts"), (tractor, "pool"), (raca, "pool"), (calibrate, "counts_for_date")):
        monkeypatch.setattr(m, name, boom)
    monkeypatch.setattr(mod, "out_dir", lambda split, smoke: tmp_path)
    with pytest.raises(PermissionError):
        mod.main(["--split", "test", "--plan"])
    draft = puffer_fetch.PREREG_DIR / "PREREG_DRAFT.md"
    if draft.is_file():                                   # the draft is never a freeze
        with pytest.raises(PermissionError):
            mod.main(["--split", "test", "--plan", "--test-freeze", str(draft)])
    assert not any(tmp_path.iterdir())


def test_e5_runs_on_dev_and_test_only_unless_smoke():
    with pytest.raises(SystemExit):
        e5.main(["--split", "train", "--plan"])
    with pytest.raises(SystemExit):
        e5.main(["--smoke", "--split", "dev", "--plan"])


def test_plans_cover_the_preregistered_grid(capsys):
    assert e5.main(["--split", "dev", "--plan"]) == 0
    head = capsys.readouterr().out.splitlines()[0]
    assert "E5 dev: 126 jobs" in head                      # 21 DEV dates x 3 seeds x 2 arms
    assert e6.main(["--split", "dev", "--plan"]) == 0
    head = capsys.readouterr().out.splitlines()[0]
    assert "E6 dev: 252 jobs" in head                      # n = 6, 9 x 21 dates x 6 windows


def test_profiles_and_seeds():
    p = e5.profiles(("video", "urllc"), (0.5, 0.7, 1.0, 1.5, 2.0))
    assert p[0] == e5.REF and len(p) == 9 and all(a != 1.0 for _, a in p[1:])
    assert e5.ft_seed("2024-03-10", 1) == e5.ft_seed("2024-03-10", 1) != e5.ft_seed("2024-03-10", 2)
    from slicemech.tenants import poison
    lay = e5.layout("dev", smoke=False)
    assert lay == poison.e5_layout("dev") and [ph for _, ph in lay] == ["A"] * 4 + ["B"] * 4
    assert e5.layout("train", smoke=True) == ((poison.E5_OFFSETS[0], "A"), (poison.E5_OFFSETS[4], "B"))


# ============================================================================================ E5
def test_played_record_hides_every_log_from_measurement_retraining():
    from slicemech.config import Mechanism
    from slicemech.control.retrain import MeasurementView, ReportAccessError
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Multiplier, Truthful
    cal = _cal()
    w = _window(cal, "train", "2024-01-01", 60, K=12, warmup=3, pool_part="A")
    pols = [Truthful(), Multiplier(2.0), Truthful()]
    df = E.run_window(w, E.FixedWeights(), "full", pols, 2, Mechanism(), cal.shape, record=False,
                      price_columns="billed")
    rec = e5.played_record(w, df)
    z = e5.as_matrix(df, "report", w.K, w.n)
    assert np.array_equal(rec.theta, z) and np.array_equal(rec.meta["stored_reports"], z)
    assert np.array_equal(z[:, 1], np.minimum(2.0 * w.theta[:, 1], w.contract.C[1]))
    v = MeasurementView(rec)
    with pytest.raises(ReportAccessError):
        _ = v.theta
    hidden = {"stored_reports", "quota_log", "price_log", "on_time_log", "utility_log"}
    assert hidden <= set(rec.meta) and not hidden & set(v.meta)
    assert np.array_equal(v.arrivals_ms, w.arrivals_ms) and v.meta["report_lead_s"] == w.meta["report_lead_s"]


@pytest.fixture(scope="module")
def e5_smoke(tmp_path_factory):
    """One smoke run of the E5 driver (both arms, sequential) in a temporary directory."""
    _cal()
    out = tmp_path_factory.mktemp("e5")
    orig = e5.out_dir
    e5.out_dir = lambda split, smoke: out
    try:
        rc = e5.main(["--smoke", "--workers", "1"])
    finally:
        e5.out_dir = orig
    return out, rc


def test_e5_smoke_run_h5a_and_reference_contrasts(e5_smoke):
    out, rc = e5_smoke
    assert rc == 0
    chk = json.loads((out / "checks.json").read_text())
    assert chk["jobs_done"] == 2 and not chk["jobs_error"]
    # pre-freeze audit E5-2: every record names its code version; no error is newer than a record
    assert chk["code_versions"] == 1 and chk["jobs_without_code"] == 0 and chk["jobs_error_after_done"] == []
    for f in (out / "jobs").glob("*.done.json"):
        assert json.loads(f.read_text())["code"] == e5.code_fingerprint()
    assert chk["h5a_all_bit_identical"] and chk["measurement_equals_stored_ref_all"]
    assert chk["stored_log_equals_plan_all"] and chk["attack_ref_truthful_all"]
    nets = pd.read_parquet(out / "nets.parquet")
    for arm, g in nets.groupby("arm"):
        meas, sto = g[g["pipeline"] == "measurement"], g[g["pipeline"] == "stored"]
        assert meas["digest"].nunique() == 1                       # H5a: identical whatever was reported
        assert sto["digest"].nunique() == len(sto)                 # the stored pipeline reads the distortion
        assert (meas["l2_to_ref"] == 0).all() and (sto["l2_to_ref"] > 0).sum() == len(sto) - 1
    pairs = pd.read_parquet(out / "pairs.parquet")
    later_meas = pairs[(pairs["phase"] == "later") & (pairs["pipeline"] == "measurement")]
    dcols = [c for c in pairs.columns if c.startswith("d_")]
    assert len(later_meas) and (later_meas[dcols].to_numpy() == 0).all()   # exactly zero later effect
    att = pairs[(pairs["phase"] == "attack") & (pairs["arm"] == "protected")]
    assert (att["d_utility_declared_target_max_interval"] <= 1e-8).all()  # per-interval DSIC in windows 1-4
    con = pd.read_parquet(out / "contrasts.parquet")
    assert len(con) == 2 * 2 * 4                                    # arms x pipelines x (2 targets x 2 alphas)
    assert {"later_win_offered_load", "attack_win_capacity_load", "net_d_utility_declared_target_total"} <= set(con)
    one = con.iloc[0]
    assert np.isclose(one["net_d_utility_declared_target_total"],
                      pairs[(pairs["arm"] == one["arm"]) & (pairs["target"] == one["target"])
                            & (pairs["alpha"] == one["alpha"])
                            & (pairs["pipeline"].isin(["deploy", one["pipeline"]]))
                            ]["d_utility_declared_target_total"].sum())
    later = pd.read_parquet(out / "later.parquet")
    assert set(later["pipeline"]) == {"base", "measurement", "stored"}


def test_e5_models_come_from_the_registry_layout_and_are_checked(e5_smoke, tmp_path):
    import argparse
    import shutil
    out, _ = e5_smoke
    root = out / "models"
    entry = e5.model_entry("protected", 0, models_root=root)
    assert not entry["missing"] and Path(entry["path"]) == root / "default" / "protected" / "seed0" / "model.pt"
    net, sha = e5.load_checked(entry)
    assert net.kind == "protected" and net.forecaster is not None and sha == entry["sha256"]
    args = argparse.Namespace(lr=None, M=None, sigma0=None)
    hyper, src = e5.resolve_hyper(args, entry)
    assert hyper["init_log_sigma"] == entry["log_sigma"] and src["lr"] == "model record"
    hyper, src = e5.resolve_hyper(argparse.Namespace(lr=1e-4, M=8, sigma0=0.3), entry)
    assert (hyper["lr"], hyper["M"], src["init_log_sigma"]) == (1e-4, 8, "command line")
    bad = tmp_path / "models"
    shutil.copytree(root, bad)
    rec = bad / "default" / "protected" / "seed0" / "config.json"
    body = json.loads(rec.read_text())
    body["result"]["model_sha256"] = "0" * 64
    rec.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="SHA-256"):
        e5.load_checked(e5.model_entry("protected", 0, models_root=bad))
    assert e5.model_entry("protected", 7, models_root=bad)["missing"]


@pytest.mark.parametrize("mod,extra", [(e5, ["--dates", "2024-03-10"]),
                                       (e6, ["--dates", "2024-03-10", "--offsets", "60"])], ids=["e5", "e6"])
def test_missing_trained_models_stop_the_run(mod, extra, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "out_dir", lambda split, smoke: tmp_path / "out")
    with pytest.raises(SystemExit, match="missing trained models"):
        mod.main(["--split", "dev", "--models-root", str(tmp_path / "nothing"), *extra])
    assert not (tmp_path / "out" / "jobs").exists()


def test_e5_is_resumable(e5_smoke, capsys):
    out, _ = e5_smoke
    stamp = {p: p.stat().st_mtime_ns for p in (out / "jobs").glob("*.done.json")}
    orig = e5.out_dir
    e5.out_dir = lambda split, smoke: out
    try:
        assert e5.main(["--smoke", "--workers", "1"]) == 0
    finally:
        e5.out_dir = orig
    assert "(0 to run)" in capsys.readouterr().out
    assert stamp == {p: p.stat().st_mtime_ns for p in (out / "jobs").glob("*.done.json")}


# ============================================================================================ E6
def test_sparse_best_responder_only_on_chosen_scored_intervals():
    from slicemech.config import Mechanism
    from slicemech.sim import episode as E
    cal = _cal("n6_default")
    w = _window(cal, "train", "2024-01-14", 60, K=14, warmup=4)
    pols = [e6.SparseBestResponder("full", w.warmup, every=3) for _ in range(w.n)]
    df = E.run_window(w, E.FixedWeights(), "full", pols, 2, Mechanism(), cal.shape, price_columns="billed")
    has = df.groupby("k")["br_gain"].apply(lambda s: bool(np.isfinite(s).all()))
    want = [k >= 4 and (k - 4) % 3 == 0 for k in range(w.K)]
    assert has.tolist() == want
    assert np.array_equal(df["report"].to_numpy(), df["theta_adm"].to_numpy())
    assert (df["br_gain"].dropna() <= 0).all()


@pytest.fixture(scope="module")
def e6_smoke(tmp_path_factory):
    _cal("n6_default")
    out = tmp_path_factory.mktemp("e6")
    orig = e6.out_dir
    e6.out_dir = lambda split, smoke: out
    try:
        rc = e6.main(["--smoke", "--workers", "1", "--n", "6"])
    finally:
        e6.out_dir = orig
    return out, rc


def test_e6_smoke_certified_gain_timing_and_protection(e6_smoke, capsys):
    out, rc = e6_smoke
    assert rc == 0
    chk = json.loads((out / "checks.json").read_text())
    rec = chk["by_n"]["6"]
    assert chk["jobs_done"] == 1 and rec["h6a_zero_certified_gain"]
    assert chk["code_versions"] == 1 and chk["jobs_without_code"] == 0 and chk["jobs_error_after_done"] == []
    for f in (out / "jobs").glob("*.done.json"):
        assert json.loads(f.read_text())["code"] == e6.code_fingerprint()
    for arm in ("equal", "direct", "protected"):
        a = rec["arms"][arm]
        assert a["br_violations"] == 0 and a["br_exact_share"] == 1.0 and a["br_misreports"] == 0
        assert a["reprice_max_abs_diff"] == 0.0 and a["t_full_ms_p95"] > 0
    runs = pd.read_parquet(out / "runs.parquet")
    assert set(runs["arm"]) == set(e6.ARMS) and runs["br_truth_reused"].dropna().all()
    assert (runs["chk_conservation_max"] < 1e-9).all() and (runs["chk_budget_excess"] <= 0).all()
    prot = runs[runs["arm"] == "protected"].iloc[0]
    assert prot["T_pct_V"] >= 0 and np.isclose(prot["p_full_pct_V"] - prot["p_fix_pct_V"], prot["T_pct_V"])
    pc = pd.read_parquet(out / "protection.parquet")
    unp = runs[runs["arm"] == "unprotected"].iloc[0]
    assert len(pc) == 1 and np.isclose(pc["d_W_unprot_minus_prot"].iloc[0], unp["W_realized"] - prot["W_realized"])
    tm = pd.read_parquet(out / "timing.parquet")
    K = e6.SMOKE["K"]
    assert len(tm) == 3 * K and tm.loc[tm["arm"] == "direct", "t_fixed_ms"].isna().all()
    br = pd.read_parquet(out / "br.parquet")
    assert len(br) == 3 * (K - e6.SMOKE["warmup"]) * 6 and (br["excess"] <= 0).all()
    orig = e6.out_dir
    e6.out_dir = lambda split, smoke: out
    try:
        assert e6.main(["--smoke", "--workers", "1", "--n", "6"]) == 0
    finally:
        e6.out_dir = orig
    assert "(0 to run)" in capsys.readouterr().out
