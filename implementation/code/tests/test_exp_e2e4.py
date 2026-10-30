"""Tests for the E2-E4 drivers (experiments/e2_manipulation.py, e3_protection.py, e4_cost.py).

Synthetic windows (the real ``data.compose.Window`` dataclass, bursty per-ms arrivals) and untrained nets saved with
``control.nets.save_net`` are injected into the drivers' worker state, so every job body (stage best responses,
replays, non-monotonicity, bandit evaluator, H3a certificates, gain identity, H3d, payments, E4b) and every summary
runs on the real allocator, pricing engines, best responses and fluid kernel without prepared data. Outputs go to
``results/raw/_pytest_e2e4_<pid>/`` and are removed afterwards. The TEST-split guard is checked without any TEST
source: every TEST reader is replaced by a function that fails if called.
"""
from __future__ import annotations

import argparse
import collections
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

torch.set_num_threads(1)
EXP = Path(__file__).resolve().parents[1] / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import e2_manipulation as X  # noqa: E402
import e3_protection as E3  # noqa: E402
import e4_cost as E4  # noqa: E402
from slicemech import io as sio  # noqa: E402
from slicemech.config import PARAMS, Mechanism, Shape, contract_from_params  # noqa: E402
from slicemech.control import features as F  # noqa: E402
from slicemech.control import nets as N  # noqa: E402
from slicemech.data.compose import Window  # noqa: E402
from slicemech.sim import episode as E  # noqa: E402
from slicemech.tenants.strategies import Truthful  # noqa: E402

CLASSES = ("video", "urllc", "mmtc")
MECH = Mechanism()
SHAPE = Shape(beta=(np.array([1.0]), np.array([0.5, 1.5]), np.array([0.25, 0.75, 3.0])),
              rho=(np.array([1.0]), np.array([0.5, 0.5]), np.array([0.4, 0.4, 0.2])))
NU = [0.02, 0.05, 0.01]


def make_window(seed: int, K: int = 14, warmup: int = 4, date: str = "2024-01-01", mu=(20.0, 10.0, 4.0)) -> Window:
    """Bursty synthetic window near full load (as tests/test_episode.py)."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float)
    n = mu.size
    ct = contract_from_params(list(CLASSES[:n]), mu)
    level = mu * rng.gamma(4.0, 0.25, (K, n))
    per_ms = np.repeat(level, 1000, axis=0) / 1000.0 * rng.gamma(0.5, 2.0, (K * 1000, n))
    arrivals = per_ms.astype(np.float32)
    measured = arrivals.astype(float).reshape(K, 1000, n).sum(1)
    theta = measured * rng.uniform(0.85, 1.15, (K, n))
    r = rng.uniform(0.25, 0.6, (K + 1, n))
    return Window(id=f"synth-{seed}", split="train", date=date, n=n, cls=CLASSES[:n], arrivals_ms=arrivals,
                  theta=theta, measured=measured, rate_true=r[1:], rate_hat=r[:-1], warmup=warmup, contract=ct,
                  meta=dict(tod_start_s=3600.0 * seed % 86400, rate_age_s=np.ones((K, n))))


# ============================================================================================ fixtures
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e4_nets")
    paths = {}
    for kind in X.NET_KINDS:
        net = N.make_net(kind, seed=0)
        net.forecaster = F.OperatorAR.default()
        paths[kind] = str(tmp / f"{kind}_seed0.pt")
        N.save_net(net, paths[kind])
    spec = X.AssetSpec(seed=0, nets=paths, static_w=[0.5, 0.3, 0.2], nu=list(NU))
    w0 = make_window(1, date="2024-01-01")
    w1 = make_window(2, date="2024-01-08")
    wins = collections.OrderedDict({("train", w0.date, 0): w0, ("train", w1.date, 1): w1})
    X._W.clear()
    X._W.update(specs={0: spec}, split="train", smoke=False, test_freeze=None,
                cal=types.SimpleNamespace(shape=SHAPE, n=3, r_bar=0.43), mech=MECH, windows=wins)
    out = sio.ROOT / "results" / "raw" / f"_pytest_e2e4_{os.getpid()}"
    yield types.SimpleNamespace(spec=spec, w0=w0, w1=w1, keys=[(w0.date, 0), (w1.date, 1)], out=out)
    shutil.rmtree(out, ignore_errors=True)
    X._W.clear()


def _p(root: Path, *parts) -> str:
    return str(root.joinpath(*parts))


# ============================================================================================ split guard
def test_test_split_is_refused_before_the_freeze_without_reading_test_sources(monkeypatch):
    from slicemech.data import calibrate, compose, puffer, puffer_fetch

    def boom(*a, **k):
        raise AssertionError("a TEST source was touched")

    monkeypatch.setattr(puffer_fetch, "_recorded_freeze", lambda: None)
    for mod, name in ((compose, "compose_window"), (compose, "compose_parts"), (calibrate, "counts_for_date"),
                      (puffer, "effective_test_dates")):
        monkeypatch.setattr(mod, name, boom)
    with pytest.raises(PermissionError):
        X.guard_split("test")
    for main in (X.main, E3.main, E4.main):
        with pytest.raises(SystemExit, match="refusing --split test"):
            main(["--split", "test", "--plan"])
        with pytest.raises(SystemExit, match="refusing --split test"):
            main(["--split", "test", "--plan", "--test-freeze", str(sio.ROOT / "prereg" / "PREREG_DRAFT.md")])


def test_split_is_required_and_smoke_is_train_only():
    with pytest.raises(SystemExit, match="--split is required"):
        X.main(["--plan"])
    with pytest.raises(ValueError, match="TRAIN"):
        X.guard_split("dev", smoke=True)
    with pytest.raises(ValueError):
        X.guard_split("validation")
    X.guard_split("train")
    X.guard_split("dev")


# ============================================================================================ windows / arms
def test_replay_windows_are_spread_over_the_day_and_rotate_over_dates():
    keys = X.split_keys("test")                       # pre-declared date rule only; nothing is read
    assert len(keys) == 16 * 8
    rk = X.replay_keys(keys, 2)
    assert len(rk) == 32 and rk == X.replay_keys(keys, 2)
    by_date = collections.defaultdict(list)
    for d, o in rk:
        by_date[d].append(o)
    assert len(by_date) == 16
    for t, (d, offs) in enumerate(by_date.items()):
        assert len(offs) == 2 and (offs[1] - offs[0]) == 4 * 10800   # half a day apart
    first = [min(o) for o in by_date.values()]
    assert len(set(first)) == 4                       # the time of day rotates over dates
    dev = X.replay_keys(X.split_keys("dev"), 2)
    assert len(dev) == 42 and set(dev) <= set(X.split_keys("dev"))
    assert X.window_keys("train", smoke=True) == [tuple(k) for k in X.SMOKE_WINDOWS]
    assert len(X.window_keys("dev", max_windows=5)) == 5


def test_arm_registry_matches_controllers_yaml_and_prices():
    import yaml
    keys = {c["key"] for c in yaml.safe_load(open(sio.ROOT / "spec" / "controllers.yaml"))}
    arms = set(X.E2_ARMS) | set(E3.CERT_ARMS) | set(E4.PAY_ARMS) | set(E4.e4b_arms())
    for a in arms:
        kind, price, key = X.arm_kind(a)
        assert key in keys or key == "kappa0", a
    for a in E3.CERT_ARMS + E3.H3D_ARMS + E3.IDENT_ARMS:
        assert X.arm_kind(a)[1] == "full"
    assert X.arm_kind(X.scaled_arm(0.5)) == ("protected_scaled:0.5", "full", "protected")
    assert E4.e4b_arms() == ["kappa_x0", "kappa_x0.5", "kappa_x2"]
    assert X.arm_needs("unc_usage") == {"unconstrained", "nu", "protected"}
    assert X.arm_needs("static_full") == {"static"} and X.seed_free("direct_full") and not X.seed_free("prot_full")
    assert X.BR_EVERY == PARAMS["experiment_settings"]["BrEvery"]
    assert X.BANDIT_EPISODES == PARAMS["experiment_settings"]["BanditEpisodes"]
    assert X.BANDIT_RUNS == PARAMS["experiment_settings"]["BanditRuns"]
    assert X.HORIZON == 3 and X.D == PARAMS["information"]["measurement_delay_s"]


# ============================================================================================ consistency
@pytest.mark.parametrize("arm", ["prot_full", "unc_fixed", "direct_full", "directtrue_full", "static_full"])
def test_interval_rules_reproduce_the_episode_quotas(env, arm):
    w = env.w0
    ctrl, pr = X.make_arm(arm, env.spec)
    df = E.run_window(w, ctrl, pr, [Truthful() for _ in range(3)], X.D, MECH, SHAPE, record=False,
                      price_columns="billed")
    Q = df["quota"].to_numpy().reshape(w.K, 3)
    rules = X.interval_rules(w, ctrl, range(w.K), MECH, SHAPE)
    th = np.minimum(w.theta, w.contract.C)
    for k in range(w.K):
        rule, _ = rules[k]
        np.testing.assert_array_equal(np.asarray(rule(th[k][None, :]))[0], Q[k])


@pytest.mark.parametrize("arm", ["prot_fixed", "unc_usage"])
def test_stage_records_equal_the_replay_derivation(env, arm):
    """The direct stage path (non-replay windows) and the replay path give the same per-interval best responses."""
    w = env.w0
    ctrl, pr = X.make_arm(arm, env.spec)
    ks = X.scored_every(w.warmup, w.K, X.BR_EVERY)
    direct = X.stage_records(w, ctrl, pr, ks, MECH, SHAPE).set_index(["k", "tenant"]).sort_index()
    truth = E.run_window(w, ctrl, pr, [Truthful() for _ in range(3)], X.D, MECH, SHAPE, price_columns="billed")
    devs = {}
    for i in range(3):
        pols = [Truthful() for _ in range(3)]
        pols[i] = X.ScoredBestResponder(pr.charge, w.warmup)
        devs[i] = E.run_window(w, ctrl, pr, pols, X.D, MECH, SHAPE, price_columns="billed")
    rep = X._stage_from_tables(truth, devs, ks, w.contract, w.warmup).set_index(["k", "tenant"]).sort_index()
    for c in ("theta", "z_star", "gain", "U_truth", "U_star", "q_truth", "q_br", "dq_dev", "dq_victims"):
        np.testing.assert_allclose(rep[c].to_numpy(), direct[c].to_numpy(), rtol=0, atol=1e-12, err_msg=c)
    np.testing.assert_allclose(rep["dV_victims_decl"], direct["dV_victims_decl"], rtol=0, atol=1e-9)
    assert (rep["exact"] == direct["exact"]).all()


def test_scored_best_responder_is_truthful_in_the_warmup(env):
    w = env.w0
    ctrl, pr = X.make_arm("prot_fixed", env.spec)
    truth = E.run_window(w, ctrl, pr, [Truthful() for _ in range(3)], X.D, MECH, SHAPE)
    pols = [Truthful() for _ in range(3)]
    pols[1] = X.ScoredBestResponder("fixed", w.warmup)
    dev = E.run_window(w, ctrl, pr, pols, X.D, MECH, SHAPE)
    warm = dev["k"] < w.warmup
    pd.testing.assert_frame_equal(dev[warm].drop(columns=["br_price"]), truth[warm].drop(columns=["br_price"]))
    b = dev[(dev["tenant"] == 1)]
    assert b.loc[b["k"] < w.warmup, "br_gain"].isna().all()
    assert np.isfinite(b.loc[b["k"] >= w.warmup, "br_gain"]).all()


def test_scaled_kappa_maps(env):
    net = X._cached_net(env.spec.nets["protected"])
    wf = F.window_features(env.w0, X.D, env.w0.contract, forecaster=net.forecaster, mech=MECH)
    ctx = wf.context(6)
    z = np.minimum(env.w0.theta[6], env.w0.contract.C)
    np.testing.assert_allclose(X.ScaledKappa(net, 1.0).priority_map(ctx).weights(z), net.priority_map(ctx).weights(z),
                               rtol=0, atol=0)
    a, kappa = net.heads_np(ctx)
    m2 = X.ScaledKappa(net, 2.0).priority_map(ctx)
    np.testing.assert_allclose(m2.kappa, 2 * kappa)
    m0 = X.ScaledKappa(net, 0.0).priority_map(ctx)
    assert m0.monotone_by_construction and np.all(m0.kappa == 0)
    z2 = z.copy()
    z2[0] *= 3
    np.testing.assert_allclose(m0.weights(z), m0.weights(z2))          # reports no longer move the weights
    with pytest.raises(ValueError):
        X.ScaledKappa(net, -1.0)


def test_bandit_evaluator_matches_the_episode_adapter_and_memoises(env):
    memo = X.PersistentMemo(env.out / "memo_test.sqlite")
    ev = X.bandit_evaluator("prot_fixed", env.spec, memo)
    ctrl, pr = X.make_arm("prot_fixed", env.spec)
    ref = E.make_bandit_evaluator([env.w0], ctrl, pr, X.D, MECH, SHAPE)
    m = np.array([1.25, 1.0, 0.8])
    a = ev(X.wkey(env.w0.date, 0), m)
    b = ref(env.w0.id, m)
    for k in ("utility", "utility_realized", "on_time_Mbit", "price"):
        np.testing.assert_array_equal(a[k], b[k])
    assert a["W_realized"] == b["W_realized"] and memo.misses == 1
    c = ev(X.wkey(env.w0.date, 0), m)                    # from SQLite, exact after the JSON round trip
    assert memo.hits == 1 and memo.count() == 1
    for k in ("utility", "realized_value"):
        np.testing.assert_array_equal(c[k], a[k])


# ============================================================================================ job bodies + summaries
def test_jobs_and_summaries_end_to_end(env):
    out = env.out
    r2, r3, r4 = out / "e2", out / "e3", out / "e4"
    (d0, o0), (d1, o1) = env.keys
    # ---- E2: replay window (w0) and stage window (w1)
    for arm in ("prot_fixed", "unc_usage", "unc_env"):
        X.job_e2_window(arm, 0, d0, o0, True, _p(r2, "stage", arm, "s0", f"{X.wkey(d0, o0)}.parquet"),
                        _p(r2, "replay", arm, "s0", f"{X.wkey(d0, o0)}.parquet"))
        X.job_e2_window(arm, 0, d1, o1, False, _p(r2, "stage", arm, "s0", f"{X.wkey(d1, o1)}.parquet"), None)
    X.job_e2_window("unc_fixed", 0, d1, o1, False, _p(r2, "stage", "unc_fixed", "s0", f"{X.wkey(d1, o1)}.parquet"),
                    None, stage_limit=1)
    for kind in ("unconstrained", "protected"):
        X.job_e2_nonmono(kind, 0, d0, o0, _p(r2, "nonmono", kind, "s0", f"{X.wkey(d0, o0)}.parquet"))
    pool = [X.wkey(d0, o0), X.wkey(d1, o1)]
    for algo in X.BANDIT_ALGOS:
        X.job_e2_bandit("prot_fixed", 0, algo, 0, 4, pool, _p(r2, "bandit", "prot_fixed", "s0", "memo.sqlite"),
                        _p(r2, "bandit", "prot_fixed", "s0", f"{algo}_run0.npz"))
    a2 = argparse.Namespace(split="train", smoke=False)
    s2 = X.e2_summary(r2, a2)
    stage = pd.read_parquet(r2 / "stage_all.parquet")
    assert set(stage["source"]) == {"replay", "stage"}
    assert (stage["gain"] >= 0).all() and (stage.loc[stage["arm"] == "prot_fixed", "exact"]).all()
    assert "prot_fixed|urllc" in s2["H2a_stage_best_response"]
    assert any(k.startswith("unc_usage|deviator") for k in s2["H2c_victims"])
    nm = s2["H2b_nonmonotonicity"]
    assert all(nm[f"protected|{c}"]["nonmono_count"] == 0 for c in CLASSES)     # negative control
    vic = pd.read_parquet(r2 / "victims.parquet")
    # a deviator that never misreports leaves the honest tenants' outcomes exactly unchanged
    still = vic[vic["share_misreport"] == 0]
    assert (still[["d_on_time_victims", "d_value_victims"]].abs().to_numpy() == 0).all()
    assert len(s2["bandits"]) == 2
    # ---- E3
    for arm in ("prot_full", "equal_full", "direct_full"):
        E3.job_e3_cert(arm, 0, d0, o0, _p(r3, "cert", arm, "s0", f"{X.wkey(d0, o0)}.parquet"))
    E3.job_e3_ident("prot_full", 0, d0, o0, _p(r3, "ident", "prot_full", "s0", f"{X.wkey(d0, o0)}.parquet"), limit=1)
    E3.job_e3_h3d("prot_full", 0, d0, o0, _p(r3, "h3d", "prot_full", "s0", f"{X.wkey(d0, o0)}.parquet"))
    s3 = E3.e3_summary(r3, argparse.Namespace(split="train", smoke=False))
    assert s3["H3a_all_arms"]["violations"] == 0 and s3["H3a_all_arms"]["violations_strict"] == 0
    assert s3["H3a_certified_gain"]["prot_full"]["exact_share"] == 1.0
    assert s3["H3c_premium"]["equal_full"]["max_T"] == 0.0
    ident = s3["H3c_identity"]
    assert ident["max_abs_resid_same"] < 1e-9 and ident["n_resid_indep_above_bound"] == 0
    assert ident["fixed_price_br"]["max_full_gain_at_fixed_br"] <= 1e-9
    h3d = pd.read_parquet(next((r3 / "h3d").rglob("*.parquet")))
    assert len(h3d) == 3 * (1 + len(E3.H3D_MULTS))
    # declared utility never gains under the full price (the constant multipliers are particular reports)
    base = h3d[h3d["m"] == 1.0].set_index("tenant")["U_declared"]
    dev = h3d[h3d["m"] != 1.0]
    assert (dev["U_declared"].to_numpy() <= base.loc[dev["tenant"]].to_numpy() + 1e-7).all()
    # ---- E4
    for arm in ("prot_full", "unc_truth", "equal_full", "kappa_x0", "kappa_x2"):
        for d, o in env.keys:
            E4.job_e4_pay(arm, 0, d, o, _p(r4, "pay", arm, "s0", f"{X.wkey(d, o)}.parquet"))
    E4.job_e4_learn("prot_fixed", 0, [1.0, 1.25, 1.0], [[d0, o0]],
                    _p(r4, "learn", "prot_fixed", "s0", "m1_1.25_1.parquet"), n_runs=2)
    a4 = argparse.Namespace(split="train", smoke=False, kappa_scales=[0.0, 0.5, 1.0, 2.0], h4b_arms=list(E4.H4B_ARMS),
                            e2_root=str(r2), tag=None)
    s4 = E4.e4_summary(r4, a4)
    pay = pd.read_parquet(r4 / "pay_windows.parquet")
    k0 = pay[pay["arm"] == "kappa_x0"]
    assert np.allclose(k0["T_mean"], 0.0) and np.allclose(k0["pay_pct_V_p"], k0["pay_pct_V_p_fix"])
    assert {"s=0", "s=1", "s=2", "equal_full"} <= set(s4["E4b_frontier"])
    h4b = s4["H4b_best_response"]
    assert h4b["check_prot_fixed_truthful_equals_protected"] == 0.0     # E2's protected truthful run = E4's
    assert any(k.startswith("unc_usage|deviator") for k in h4b)
    lr = s4["H4b_learning"]["prot_fixed|s0"]["m1_1.25_1"]
    assert lr["honest_tenants"] == [0, 2] and lr["n_runs"] == 2
    assert "H4a_descriptive_unc_minus_prot" in s4


def test_e4_learning_profiles_come_from_bandit_end_states(env, tmp_path):
    root = env.out / "learnprof"
    for r, last in enumerate(([1.0, 1.25, 1.0], [1.0, 1.25, 1.0], [0.8, 1.0, 1.0])):
        X.write_npz(root / "bandit" / "unc_fixed" / "s0" / f"exp3_run{r}.npz",
                    greedy_m=np.array([[1.0, 1.0, 1.0], last]))
    prof = E4.learn_profiles(root, 0, ["unc_fixed", "unc_usage"])
    assert prof == {"unc_fixed": {(0.8, 1.0, 1.0): 1, (1.0, 1.25, 1.0): 2}}


# ============================================================================================ job runner
def test_job_runner_plans_resumes_and_logs_failures(env):
    root = env.out / "runner"
    done = root / "done.parquet"
    X.write_parquet(pd.DataFrame({"a": [1]}), done)
    jobs = [X.Job("done", "e2_manipulation:job_loads", {}, str(done), 5.0),
            X.Job("todo", "e2_manipulation:job_loads", {}, str(root / "todo.parquet"), 7.0)]
    res = X.run_jobs(jobs, workers=2, initargs=(), errors_path=root / "errors.jsonl", plan=True)
    assert res["pending"] == 1 and abs(res["est_core_hours"] - 7.0 / 3600) < 1e-12


@pytest.mark.skipif(not (sio.DATA_DIR / "calibration" / f"{X.CAL_NAME}.json").exists(),
                    reason="needs the stored calibration and TRAIN data")
def test_job_runner_pool_runs_skips_and_logs(env):
    """Real spawn pool: one good job (real TRAIN window loads), one failing job; a rerun skips the finished one."""
    root = env.out / "pool"
    good = root / "loads.parquet"
    jobs = [X.Job("good", "e2_manipulation:job_loads", dict(date="2024-01-01", offset=60, out=str(good)), str(good)),
            X.Job("bad", "e2_manipulation:job_loads", dict(date="2024-01-01", offset=60), str(root / "bad.parquet"))]
    init = ({0: env.spec}, "train", False, None)
    res = X.run_jobs(jobs, workers=2, initargs=init, errors_path=root / "errors.jsonl")
    assert res["failed"] == 1 and good.exists() and (root / "errors.jsonl").exists()
    loads = pd.read_parquet(good)
    assert 0.3 < float(loads["offered_load_scored"].iloc[0]) < 2.0
    res2 = X.run_jobs(jobs, workers=2, initargs=init, errors_path=root / "errors.jsonl")
    assert res2["pending"] == 1 and res2["failed"] == 1 and (root / "errors.prev.jsonl").exists()


def test_atomic_writers_refuse_paths_outside_the_project(tmp_path):
    with pytest.raises(ValueError):
        X.write_parquet(pd.DataFrame({"a": [1]}), tmp_path / "x.parquet")
    with pytest.raises(ValueError):
        X.write_npz(tmp_path / "x.npz", a=np.zeros(1))


def test_asset_resolution_reports_missing_checkpoints(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing inputs"):
        X.resolve_assets(["prot_fixed", "static_full"], [0], smoke=False, ckpt_dir=tmp_path)
    net = N.make_net("protected", seed=0)
    net.forecaster = F.OperatorAR.default()
    N.save_net(net, tmp_path / "protected_seed0.pt")
    (tmp_path / "static.json").write_text('{"w": [0.5, 0.3, 0.2]}')
    specs = X.resolve_assets(["prot_fixed", "static_full"], [0], smoke=False, ckpt_dir=tmp_path)
    assert specs[0].nets["protected"] == str(tmp_path / "protected_seed0.pt")
    assert specs[0].static_w == [0.5, 0.3, 0.2] and len(specs[0].sha256["protected"]) == 64

