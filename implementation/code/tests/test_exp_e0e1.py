"""Tests of the E0/E1 experiment drivers (``experiments/e0_setup.py``, ``experiments/e1_value.py``).

Real TRAIN windows are composed short (K = 30 s) so the whole file runs in well under a minute. Every output goes to
``data/_pytest_e0e1_<pid>`` (removed afterwards); the driver globals that point at ``results/`` are monkeypatched.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True
_EXP = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import e0_setup as S  # noqa: E402
import e1_value as E1  # noqa: E402
import registry as R  # noqa: E402
from slicemech import io as sio  # noqa: E402
from slicemech.config import Mechanism  # noqa: E402
from slicemech.control import nets  # noqa: E402
from slicemech.data import calibrate as calib  # noqa: E402

DATE, OFF = "2024-01-14", 60


@pytest.fixture(scope="module")
def tmpdir_local():
    d = sio.DATA_DIR / f"_pytest_e0e1_{os.getpid()}"            # unique: other sessions may run the suite too
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def cal():
    return calib.load_calibration("n3_default")


@pytest.fixture(scope="module")
def win(cal):
    return cal.window("train", DATE, OFF, K=30, warmup=5)


@pytest.fixture(scope="module")
def mech():
    return Mechanism()


def _net(kind, seed, cal, d=2):
    net = nets.make_net(kind, seed=seed)
    net.forecaster = cal.operator_forecaster(d)
    net.eval()
    return net


@pytest.fixture(scope="module")
def ckpt_dir(tmpdir_local, cal):
    d = tmpdir_local / "ckpt"
    d.mkdir()
    for kind in S.NET_KINDS:
        nets.save_net(_net(kind, 0, cal), d / f"{kind}_seed0.pt")
    return d


# ---------------------------------------------------------------------------------------------- configs
def test_configs_come_from_the_training_registry(cal):
    assert set(R.CONFIGS) >= {"default", "d1", "d5", "d15", "rho0.5", "rho0.7", "rho1.1", "lead0", "lead1.05",
                              "n6", "n9"}
    for name, c in R.CONFIGS.items():
        assert S.get_config(name) is c
        path = R.calibration_path(c.calibration)
        if path.is_file():                                      # sweep calibrations exist once train.py made them
            assert c.d in S.load_cal(c).operator_ar, f"{name}: no operator forecaster for d={c.d}"
    assert len(S.get_config("default").seeds) == 5 and len(S.get_config("d5").seeds) == 3
    with pytest.raises(SystemExit):
        S.get_config("nope")
    assert S.SMOKE_WINDOWS == tuple((d, int(o)) for d, o in R.SMOKE_TRAIN_WINDOWS)


def test_model_discovery_and_record_checks(tmpdir_local, cal):
    root = tmpdir_local / "models"
    d0 = root / "default" / "protected" / "seed0"
    d0.mkdir(parents=True)
    net = _net("protected", 0, cal)
    nets.save_net(net, d0 / R.MODEL_FILE)
    rec = dict(status="done", result=dict(model_sha256=sio.file_sha256(d0 / R.MODEL_FILE),
                                          param_digest=nets.param_digest(net)))
    sio.write_json(d0 / R.CONFIG_FILE, rec)
    (root / "default" / "protected" / "seed1").mkdir()                      # unfinished: no record
    nets.save_net(net, root / "default" / "protected" / "seed1" / R.MODEL_FILE)
    tune = root / "default" / "protected" / "tune" / "cellA" / "seed2"
    tune.mkdir(parents=True)
    nets.save_net(net, tune / R.MODEL_FILE)
    sio.write_json(tune / R.CONFIG_FILE, rec)
    assert S.find_models(root, "default", "protected") == {0: d0 / R.MODEL_FILE}
    assert S.find_models(root, "default", "kappa0") == {} and S.find_models(root / "absent", "default", "x") == {}
    assert S.load_checked_net(d0 / R.MODEL_FILE, 2) is not None
    with pytest.raises(ValueError):                                          # wrong delay for its forecaster
        S.load_checked_net(d0 / R.MODEL_FILE, 5)
    sio.write_json(d0 / R.CONFIG_FILE, dict(rec, result=dict(rec["result"], param_digest="0" * 64)))
    with pytest.raises(ValueError):                                          # record mismatch
        S.load_checked_net(d0 / R.MODEL_FILE, 2)
    sio.write_json(d0 / R.CONFIG_FILE, dict(rec, spec=dict(ppo=dict(horizon=0))))
    with pytest.raises(ValueError):                                          # D6: lookahead horizon must be 3
        S.load_checked_net(d0 / R.MODEL_FILE, 2)
    sio.write_json(d0 / R.CONFIG_FILE, dict(rec, spec=dict(ppo=dict(horizon=3))))
    assert S.load_checked_net(d0 / R.MODEL_FILE, 2) is not None
    flat = tmpdir_local / "flat"
    flat.mkdir()
    nets.save_net(net, flat / "kappa0_seed3.pt")
    (flat / "kappa0_final.pt").write_bytes(b"x")
    got = S.model_paths(S.get_config("default"), ("protected", "kappa0"), root, flat)
    assert got == {"protected": {0: d0 / R.MODEL_FILE}, "kappa0": {3: flat / "kappa0_seed3.pt"}}


def test_static_weights_come_from_the_registry(tmpdir_local, cal):
    root = tmpdir_local / "models_static"
    sd = R.static_dir("default", root=root)
    sd.mkdir(parents=True)
    np.savez_compressed(sd / R.STATIC_CANDIDATES, candidates=np.eye(3), scores=np.arange(3.0))
    sio.write_json(sd / R.STATIC_FILE, dict(status="done", w=[0.5, 0.3, 0.2], welfare_train=1.0, method="grid",
                                            seconds=1.0, info={}))
    for name in ("default", "d5"):                                           # the d cells share the default's fit
        w, src = E1.static_weights(cal, S.get_config(name), root, tmpdir_local / "out_static", False)
        assert w == [0.5, 0.3, 0.2] and src == str(sd / R.STATIC_FILE)
    with pytest.raises(SystemExit):                                          # TEST never refits
        E1.static_weights(cal, S.get_config("default"), tmpdir_local / "absent", tmpdir_local / "o", False, test=True)


# ---------------------------------------------------------------------------------------------- TEST gate
def test_test_split_is_refused_before_the_freeze(tmpdir_local, monkeypatch):
    assert S.test_gate("dev") is None and S.test_gate("train", "whatever") is None
    with pytest.raises(PermissionError):
        S.test_gate("test", tmpdir_local / "absent" / "PREREG.md")
    with pytest.raises(PermissionError):                      # the draft is never a freeze
        S.test_gate("test", sio.ROOT / "prereg" / "PREREG_DRAFT.md")

    def boom(*a, **k):
        raise AssertionError("a window was composed before the TEST gate")
    monkeypatch.setattr(S, "compose", boom)
    monkeypatch.setattr(calib.Calibration, "window", boom)
    absent = str(tmpdir_local / "absent" / "PREREG.md")
    with pytest.raises(PermissionError):
        S.main(["--split", "test", "--test-freeze", absent])
    with pytest.raises(PermissionError):
        E1.main(["--split", "test", "--test-freeze", absent])
    if not S.PREREG_FREEZE.exists():                         # the default freeze file does not exist yet
        with pytest.raises(PermissionError):
            E1.main(["--split", "test"])


def test_checkpoints_must_be_frozen_on_test(tmpdir_local):
    from slicemech.data import puffer_fetch as PF
    a, b = tmpdir_local / "fa.pt", tmpdir_local / "fb.pt"
    a.write_bytes(b"frozen parameters")
    b.write_bytes(b"other parameters")
    freeze = tmpdir_local / "FREEZE.md"
    freeze.write_text("# Pre-registration\n\n" + PF.freeze_block([a], "Sundays and Mondays"), encoding="utf-8")
    S.verify_frozen_checkpoints([a], freeze)
    with pytest.raises(PermissionError):
        S.verify_frozen_checkpoints([a, b], freeze)          # b is not listed
    a.write_bytes(b"retrained")
    with pytest.raises(PermissionError):
        S.verify_frozen_checkpoints([a], freeze)              # a changed after the freeze


# ---------------------------------------------------------------------------------------------- E0
def test_vcal_sufficient_statistics_reproduce_the_raw_regression(win, cal, mech):
    from slicemech.model.shapes import stack_shape, value
    df, raw = S.vcal_stats(win, cal.shape, mech, raw=True)
    assert len(df) == win.n * len(S.VCAL_GRID)
    x, y, ok = raw["x"], raw["y"], raw["valid"]
    for i in range(win.n):
        xi, yi = x[..., i][ok[..., i]], y[..., i][ok[..., i]]
        g = df[df["tenant"] == i][["N", "Sx", "Sy", "Sxx", "Syy", "Sxy"]].sum()
        res = S.ols_from_sums(*g.to_numpy(float))
        slope, icpt = np.polyfit(xi, yi, 1)
        assert res["N"] == xi.size
        assert res["slope"] == pytest.approx(slope, rel=1e-9, abs=1e-12)
        assert res["intercept"] == pytest.approx(icpt, rel=1e-7, abs=1e-9)
        assert res["r2"] == pytest.approx(np.corrcoef(xi, yi)[0, 1] ** 2, rel=1e-9)
    # one x entry by hand: V / pi at quota c theta / rhat
    ct = win.contract
    k, j, i = 3, 2, 1
    kk = win.warmup + k
    z = min(float(win.theta[kk, i]), ct.C[i])
    q = S.VCAL_GRID[j] * z / float(win.rate_hat[kk, i])
    beta, rho = stack_shape(cal.shape)
    assert x[k, j, i] == pytest.approx(float(value(min(q, mech.B), z, float(win.rate_hat[kk, i]), 1.0, beta[i],
                                                   rho[i], mech.Delta)), rel=1e-12)
    assert np.all(y >= 0) and np.all(raw["q"] <= mech.B)


def test_regime_pooling_equals_the_concatenated_diagnostic(cal, mech, win):
    from slicemech.data.calibrate import regime_diagnostics
    w2 = cal.window("train", "2024-01-01", 60, K=30, warmup=5)
    rows = [S.window_diagnostics(w, cal, mech)[0] for w in (win, w2)]
    pooled = S.regime_summary(pd.DataFrame(rows))["pooled"]
    z = np.concatenate([np.minimum(w.theta, w.contract.C)[w.warmup:] for w in (win, w2)])
    r = np.concatenate([w.rate_hat[w.warmup:] for w in (win, w2)])
    ref = regime_diagnostics(z, r, win.contract, mech)
    for key in ("frac_sumH_gt_B", "frac_all_caps_fit", "frac_weights_change_quota", "mean_weighted_share",
                "gbr_outage"):
        assert pooled[key] == pytest.approx(ref[key], abs=1e-12)
    assert pooled["intervals"] == ref["intervals"]
    r0 = rows[0]
    assert r0["offered_load"] == pytest.approx(np.asarray(win.measured).mean(0).sum() / (mech.B * cal.r_bar))
    assert [r0[f"count_{i}"] for i in range(3)] == list(cal.counts_for("train", DATE))


def test_worked_interval_matches_the_episode_prices(win, cal, mech):
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    cfg = S.get_config("default")
    prot, unc = _net("protected", 0, cal), _net("unconstrained", 1, cal)
    k = 12
    curves, stairs, meta = S.worked_interval(win, cal, cfg, mech, prot, unc, k=k, n_grid=33)
    i = meta["tenant"]
    assert win.contract.cls[i] == "urllc" and meta["k"] == k
    pc = curves[curves["map"] == "protected"]
    assert np.all(np.diff(pc["quota"].to_numpy()) >= 0)                       # protected map: monotone path
    df = E.run_window(win, prot, "full", [Truthful() for _ in range(3)], cfg.d, mech, cal.shape)
    row = df[(df["k"] == k) & (df["tenant"] == i)].iloc[0]
    truth = pc[pc["is_truth"]].iloc[0]
    assert truth["u"] == pytest.approx(row["theta_adm"])
    assert truth["quota"] == row["quota"]
    assert truth["p_full"] == pytest.approx(row["p"], rel=1e-9, abs=1e-12)
    assert truth["p_fix"] == pytest.approx(row["p_fix"], rel=1e-9, abs=1e-12)
    tol = 1e-8 * float(win.contract.pi[i]) * float(win.contract.C[i])
    assert pc["U_full"].max() <= truth["U_full"] + tol                         # truth is optimal under the full price
    for m, g in stairs.groupby("map"):
        assert np.all(np.diff(g["start"].to_numpy()) > 0) and np.all(np.diff(g["quota"].to_numpy()) != 0)
    assert set(curves["map"]) == {"protected", "unconstrained"}
    # automatic selection picks a scored interval with a positive report
    _, _, meta2 = S.worked_interval(win, cal, cfg, mech, prot, unc, n_grid=17)
    assert win.warmup <= meta2["k"] < win.K and meta2["theta_adm"] > 0


# ---------------------------------------------------------------------------------------------- E1
def _plan(ckpt_dir, out, arms=E1.ARMS, nu=(0.02, 0.01, 0.03)):
    cfg = S.get_config("default")
    ckpts = {arm: {s: (str(p), sio.file_sha256(p)) for s, p in S.find_checkpoints(ckpt_dir, E1.LEARNED[arm]).items()}
             for arm in arms if arm in E1.LEARNED}
    plan = dict(split="train", config="default", cfg=S.config_record(cfg), cal_sha=S.calibration_sha(cfg),
                calib_root=None, smoke=True, test_freeze=None, arms=tuple(arms),
                ckpts=ckpts, static_w=[0.5, 0.4, 0.1], nu=list(nu), oracle=dict(pop=8, iters=2, seed=0, horizon=3),
                envelope=False, intervals=True, code_fp=E1.code_fingerprint(), schema=E1.SCHEMA, out=str(out))
    plan["key"] = E1._digest({k: v for k, v in plan.items() if k != "out"})
    return plan


def test_e1_window_job_pairs_every_arm_and_resumes(tmpdir_local, ckpt_dir, cal, mech, monkeypatch):
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    monkeypatch.setattr(S, "SMOKE_K", 30)
    monkeypatch.setattr(S, "SMOKE_WARMUP", 5)
    out = tmpdir_local / "e1_job"
    plan = _plan(ckpt_dir, out)
    res = E1.window_job((plan, DATE, OFF))
    assert res["ran"] == res["planned"] == len(E1.ARMS)                       # one seed per learned arm
    s = pd.read_parquet(out / "windows" / f"{S.key_of(DATE, OFF)}.parquet")
    A = s[s["tenant"] == -1].set_index("arm")
    assert set(A.index) == set(E1.ARMS)
    assert s["window_id"].nunique() == 1                                         # paired on one composed window
    w = cal.window("train", DATE, OFF, K=30, warmup=5)
    df = E.run_window(w, E.FixedWeights(), "full", [Truthful() for _ in range(3)], 2, mech, cal.shape)
    W_eq = float(df[df["scored"]].groupby("k")["realized_value"].sum().mean())
    assert A.loc["equal", "realized_value"] == pytest.approx(W_eq, rel=1e-12)
    assert A.loc["oracle", "oracle_selfcheck"] < 1e-9
    assert A.loc["kappa0", "T"] == pytest.approx(0.0, abs=1e-9) and A.loc["kappa0", "frac_T_pos"] == 0.0
    assert A.loc["meas_only", "p"] == 0.0 and A.loc["meas_only", "price"] == pytest.approx(A.loc["meas_only", "p_usage"])
    assert A.loc["unprotected", "price_rule"] == "fixed" and np.isnan(A.loc["unprotected", "p"])
    assert np.isnan(A.loc["direct", "frac_T_pos"]) and A.loc["protected", "price_rule"] == "full"
    for c in ("chk_conservation_max", "chk_cert_violation_p", "chk_cert_violation_p_fix"):
        assert np.nanmax(A[c].to_numpy(float)) <= 1e-9
    assert (A["offered_load"] > 0).all() and A["offered_load"].nunique() == 1
    assert set(s["code_fp"]) == {E1.code_fingerprint()} and set(s["schema"]) == {E1.SCHEMA}     # audit F7, HC-4
    iv = pd.read_parquet(out / "intervals" / f"{S.key_of(DATE, OFF)}.parquet")
    assert len(iv) == len(E1.ARMS) * (30 - 5) * 3 and iv["quota"].dtype == np.int16

    # resume: nothing left to do
    assert E1.window_job((plan, DATE, OFF))["ran"] == 0
    assert E1.window_job((dict(plan, force=True), DATE, OFF))["ran"] == len(E1.ARMS)   # --force: every run again
    # a changed checkpoint makes exactly that run stale; every other row is kept unchanged
    before = A.copy()
    nets.save_net(_net("protected", 7, cal), ckpt_dir / "protected_seed0.pt")
    plan2 = _plan(ckpt_dir, out)
    res2 = E1.window_job((plan2, DATE, OFF))
    assert res2["ran"] == 1
    s2 = pd.read_parquet(out / "windows" / f"{S.key_of(DATE, OFF)}.parquet")
    A2 = s2[s2["tenant"] == -1].set_index("arm")
    assert len(s2) == len(s) and set(A2.index) == set(E1.ARMS)
    assert A2.loc["protected", "ckpt_sha"] != before.loc["protected", "ckpt_sha"]
    for arm in ("equal", "static", "oracle", "kappa0", "direct"):
        assert A2.loc[arm, "realized_value"] == before.loc[arm, "realized_value"]
        assert A2.loc[arm, "fp"] == before.loc[arm, "fp"]
    iv2 = pd.read_parquet(out / "intervals" / f"{S.key_of(DATE, OFF)}.parquet")
    assert len(iv2) == len(iv)
    # a run restricted to some arms leaves the other arms' rows in place
    assert E1.window_job((_plan(ckpt_dir, out, arms=("equal",)), DATE, OFF))["ran"] == 0
    nets.save_net(_net("protected", 0, cal), ckpt_dir / "protected_seed0.pt")        # restore for later tests


def test_e1_runs_without_interval_rows_are_redone_when_intervals_are_requested(tmpdir_local, ckpt_dir, monkeypatch):
    monkeypatch.setattr(S, "SMOKE_K", 30)
    monkeypatch.setattr(S, "SMOKE_WARMUP", 5)
    out = tmpdir_local / "e1_iv"
    plan = _plan(ckpt_dir, out, arms=("equal", "direct"))
    plan["intervals"] = False
    assert E1.window_job((plan, DATE, OFF))["ran"] == 2
    assert not (out / "intervals").exists()
    plan["intervals"] = True
    assert E1.window_job((plan, DATE, OFF))["ran"] == 2
    iv = pd.read_parquet(out / "intervals" / f"{S.key_of(DATE, OFF)}.parquet")
    assert set(iv["arm"]) == {"equal", "direct"}
    assert len(pd.read_parquet(out / "windows" / f"{S.key_of(DATE, OFF)}.parquet")) == 2 * 4
    assert E1.window_job((plan, DATE, OFF))["ran"] == 0


def test_quicklook_averages_seeds_within_a_window():
    rows = []
    for d, o, base in (("2024-03-02", 60, 10.0), ("2024-03-04", 60, 12.0), ("2024-03-06", 60, 11.0)):
        for arm, seed, W in (("protected", 0, base + 1), ("protected", 1, base + 3), ("meas_only", 0, base),
                             ("static", -1, base + 0.5)):
            rows.append(dict(date=d, offset_s=o, arm=arm, seed=seed, tenant=-1, realized_value=W,
                             offered_load=base / 10, deadline_miss_pct=1.0, payment_pct_of_V=2.0))
    q = E1.quicklook(pd.DataFrame(rows), n_boot=100)
    pm = q["paired_differences"]["protected-meas_only"]
    assert pm["mean_diff"] == pytest.approx(2.0) and pm["windows"] == 3 and pm["windows_positive"] == 3
    assert q["paired_differences"]["protected-static"]["mean_diff"] == pytest.approx(1.5)
    assert q["seeds"]["protected"] == 2 and q["welfare"]["protected"]["mean"] == pytest.approx(13.0)
    assert "protected-direct" not in q["paired_differences"]


def test_resolve_arms_aliases():
    assert E1.resolve_arms("unconstrained,direct_true_rates,equal") == ("equal", "unprotected", "direct_truerates")
    assert E1.resolve_arms(None) == E1.ARMS
    with pytest.raises(SystemExit):
        E1.resolve_arms("bogus")


def test_smoke_mains_end_to_end(tmpdir_local, monkeypatch):
    """Both drivers in --smoke mode (shortened further), outputs redirected to the test folder; second runs resume."""
    monkeypatch.setattr(S, "RESULTS", tmpdir_local / "results")
    monkeypatch.setattr(S, "SMOKE_K", 30)
    monkeypatch.setattr(S, "SMOKE_WARMUP", 5)
    monkeypatch.setattr(S, "SMOKE_PPO_ITERS", 1)
    monkeypatch.setattr(S, "smoke_models_root", lambda cfg: None)       # standalone: no training-driver models
    monkeypatch.setattr(S, "WORKED_GRID", 17)
    monkeypatch.setattr(E1, "SMOKE_SEEDS", (0,))
    m = E1.main(["--smoke", "--workers", "1", "--arms", "equal,protected,oracle,meas_only", "--n-boot", "50"])
    assert m["split"] == "train" and m["windows"] == 2 and m["runs_computed_this_invocation"] == 8
    assert m["oracle_selfcheck_max"] < 1e-9 and m["usage_nu"] is not None
    assert "protected-meas_only" in m["quicklook"]["paired_differences"]
    assert E1.main(["--smoke", "--workers", "1", "--arms", "equal,protected,oracle,meas_only"])[
        "runs_computed_this_invocation"] == 0
    s = S.main(["--smoke", "--workers", "1"])
    assert s["windows"] == 2 and s["worked_interval"]["status"] == "ok"
    assert set(s["vcal"]) == {"video", "urllc", "mmtc"} and s["regime"]["pooled"]["intervals"] == 2 * 25
    s2 = S.main(["--smoke", "--workers", "1"])
    assert s2["windows_computed_this_run"] == 0 and s2["worked_interval"]["seconds"] == 0.0
    out = tmpdir_local / "results" / "raw" / "smoke"
    assert (out / "e1" / "default" / "windows.parquet").exists() and (out / "e0" / "default" / "summary.json").exists()
    assert not (tmpdir_local / "results" / "summary").exists()                 # smoke never writes results/summary
