"""Regression tests for the round-1 adversarial review (CHANGELOG.md, "fixer, round 1").

One section per confirmed finding:

* F1  exogenous pools reused inside a split -> disjoint pool parts for E5, crossed (capture/run) bootstrap;
* F2  TEST statistics in the data report -> withheld before the pre-registration freeze;
* F3  myopic PPO reward / oracle objective -> lookahead replay (``sim.fluid.lookahead_window``);
* F4  welfare-headroom regime diagnostic (``eval.headroom``);
* F5  operator AR never fitted / silently defaulted -> fitted per delay, carried by nets, required;
* F6  water-fill cancellation at large weight spreads -> direct evaluation at breakpoints;
* F7  full-map prices ignored ``monotone_by_construction`` -> refused unless ``assume_monotone``;
* F8  no split guards in PPO / static fitting;
* F9  measurement-mode retraining used z = m -> rebuilt truthful-equivalent reports;
* F10 unequal initial report priors of the protected and unconstrained nets.

Tests on real data (TRAIN/DEV only; TEST is never touched) skip when the prepared data are missing.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

from slicemech.config import Contract, Mechanism, Shape, contract_from_params  # noqa: E402
from slicemech.control import features as F  # noqa: E402
from slicemech.control import nets as N  # noqa: E402
from slicemech.control import oracle as O  # noqa: E402
from slicemech.control import ppo as P  # noqa: E402
from slicemech.control import retrain as RT  # noqa: E402
from slicemech.control import static as S  # noqa: E402
from slicemech.model import allocation as AL  # noqa: E402
from slicemech.sim import fluid  # noqa: E402

CLASSES = ("video", "urllc", "mmtc")
FC = F.OperatorAR.default()


@dataclass
class Win:
    """Minimal window with the fields of ``data.Window``."""

    id: str
    split: str
    date: str
    n: int
    cls: tuple
    arrivals_ms: np.ndarray
    theta: np.ndarray
    measured: np.ndarray
    rate_true: np.ndarray
    rate_hat: np.ndarray
    warmup: int
    contract: object
    meta: dict


def synth_window(seed, K=10, split="train", mu=(12.0, 6.0, 2.0), warmup=2):
    """Bursty per-ms arrivals near full load for the real allocator and fluid kernel."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float)
    ct = contract_from_params(list(CLASSES), mu)
    per_ms = np.repeat(mu * rng.gamma(4.0, 0.25, (K, 3)), 1000, axis=0) / 1000.0
    per_ms *= rng.gamma(2.0, 0.5, per_ms.shape)
    arr = per_ms.astype(np.float32)
    m = arr.astype(float).reshape(K, 1000, 3).sum(1)
    r = rng.uniform(0.12, 0.25, (K + 1, 3))
    return Win(f"s{seed}", split, "2024-01-01", 3, CLASSES, arr, np.minimum(m * rng.uniform(0.8, 1.2, m.shape), ct.C),
               m, r[1:], r[:-1], warmup, ct, dict(tod_start_s=0.0))


def _load_cal():
    calib = pytest.importorskip("slicemech.data.calibrate")
    try:
        return calib.load_calibration("n3_default")
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"calibration missing: {exc}")


def _real_window(cal, split, date, offset, **kw):
    try:
        return cal.window(split, date, offset, **kw)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"prepared data missing: {exc}")


# ============================================================================================ F6 water-fill
def _ref_fill(L, U, w, T):
    """Bisection reference for x = clip(tau w, L, U), sum x = T (independent of the package)."""
    lo, hi = 0.0, 1.0
    while np.clip(hi * w, L, U).sum() < T:
        hi *= 2.0
    for _ in range(2000):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if np.clip(mid * w, L, U).sum() < T:
            lo = mid
        else:
            hi = mid
    return np.clip(hi * w, L, U)


def test_f6_fill_is_exact_for_extreme_weight_spreads():
    """Zero floors and a capped top-weight tenant: the slope-accumulating fill lost the small weights and raised
    'water-fill total mismatch' from a log-spread of 20 on; now exact up to one rounding for spreads up to 1e30."""
    ct = Contract(cls=("video",) * 3, g=np.zeros(3), M=np.array([10.0, 60, 60]), C=np.array([20.0, 80, 80]),
                  pi=np.ones(3), delay_ms=np.full(3, 100.0), mu=np.full(3, 10.0))
    mech = Mechanism(B=50, gamma=0.0, eta=1.0, Delta=1.0, integer=True)
    z, r = np.array([10.0, 60, 60]), np.ones(3)
    expect = np.array([10.0, 40 / (1 + np.exp(-2.0)), 40 * np.exp(-2.0) / (1 + np.exp(-2.0))])
    for y in ([0.0, -10, -12], [0.0, -15, -17], [0.0, -20, -22], [0.0, -30, -32], [0.0, -60, -62],
              [0.0, -69, -71]):
        x = AL.fractional(z, np.exp(np.array(y)), r, ct, mech)
        np.testing.assert_allclose(x, expect, rtol=0, atol=1e-12)
    # random contracts (zero floors likely), weight spreads up to 1e30, against the bisection reference
    rng = np.random.default_rng(6)
    for _ in range(300):
        n = int(rng.integers(3, 10))
        L = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0, 3, n))
        U = L + rng.uniform(0.0, 40.0, n)
        w = 10.0 ** rng.uniform(-30, 0, n)
        w /= w.max()
        T = L.sum() + rng.uniform(0.01, 0.99) * (U - L).sum()
        x = AL._fill(L, U, w, np.array(T))
        assert abs(x.sum() - T) <= 1e-9 * max(1.0, T)
        np.testing.assert_allclose(x, _ref_fill(L, U, w, T), rtol=0, atol=1e-9)


# ============================================================================================ F7 price guard
def _bump_rule():
    """n = 2, need stage, zero floors; the opponent weight has a narrow bump between two 65-grid points."""
    from slicemech.model.scores import CallableMap
    ct = Contract(cls=("video",) * 2, g=np.zeros(2), M=np.full(2, 40.0), C=np.full(2, 40.0), pi=np.ones(2),
                  delay_ms=np.full(2, 100.0), mu=np.full(2, 5.0))
    mech = Mechanism(B=40, gamma=0.0, eta=4.0, Delta=1.0, integer=True)
    c, eps = 10.0 + 0.5 * 40.0 / 64, 0.05

    def fn(z):
        z = np.asarray(z, float)
        return np.stack([1.0 + z[..., 0] / 40.0, 1.0 + 3.0 * np.exp(-((z[..., 0] - c) / eps) ** 2)], -1)

    rule = AL.QuotaRule(ct, mech, np.ones(2), CallableMap(fn, monotone=False))
    return ct, mech, rule, Shape(beta=(np.ones(1),) * 2, rho=(np.ones(1),) * 2)


def test_f7_full_price_refuses_rules_not_monotone_by_construction():
    from slicemech.pricing import envelope as EV
    from slicemech.pricing.thresholds import NonMonotoneRuleError, max_drop, own_path, own_thresholds
    ct, mech, rule, unit = _bump_rule()
    z, r = np.array([40.0, 20.0]), np.ones(2)
    drop, _ = max_drop(own_path(0, z, rule, np.linspace(0, 40, 400001)))
    assert drop >= 1                                                   # the rule really is non-monotone ...
    for call in (lambda: EV.full_price(0, z, rule, ct, unit, mech, r),
                 lambda: EV.full_prices(z, rule, ct, unit, mech, r),
                 lambda: EV.premium(0, z, rule, ct, unit, mech, r),
                 lambda: own_thresholds(0, z, rule, 5)):
        with pytest.raises(NonMonotoneRuleError, match="not monotone by construction"):
            call()                                                     # ... and is refused before the grid check
    # explicit opt-in documents the failure mode: the 65-point grid misses the drop and certifies a wrong price
    p = EV.full_price(0, z, rule, ct, unit, mech, r, assume_monotone=True)
    assert p.p_hi - p.p_lo < 1e-6
    assert EV.price_nonmonotone(0, z, rule, ct, unit, mech, r).p > p.p_hi + 1e-3
    # the fixed-priority price freezes the weights (monotone) and needs no flag
    assert np.isfinite(EV.fixed_priority_price(0, z, rule, ct, unit, mech, r).p)
    with pytest.raises(TypeError):
        EV.full_price(0, z, rule, ct, unit, mech, r, assume_monotone="yes")


# ============================================================================================ F3 lookahead
def test_f3_lookahead_window_equals_replay_of_modified_path():
    w = synth_window(1, K=8)
    mech = Mechanism()
    A = fluid.cumulative_arrivals(w.arrivals_ms)
    Q = AL.integer(w.theta, np.ones(3), w.rate_hat, w.contract, mech).astype(float)
    out, c0, _ = fluid.replay(A, Q, w.rate_true, w.contract.delay_ms)
    ks = np.array([1, 4, 6])
    h = 3
    rng = np.random.default_rng(0)
    Qs = np.stack([np.stack([rng.multinomial(int(Q[k].sum()), [0.5, 0.3, 0.2]) for _ in range(3)]) for k in ks])
    steps = np.minimum(h, w.measured.shape[0] - 1 - ks)
    nxt = np.minimum(ks[:, None] + np.arange(1, h + 1)[None, :], len(Q) - 1)
    la = fluid.lookahead_window(A, ks * 1000, c0[ks], Qs, w.rate_true[ks], Q[nxt], w.rate_true[nxt],
                                w.contract.delay_ms, steps=steps)
    assert la.shape == (3, 3, h + 1, 3)
    for s, k in enumerate(ks):
        for m in range(3):
            Qp = Q.copy()
            Qp[k] = Qs[s, m]
            ref, _, _ = fluid.replay(A, Qp, w.rate_true, w.contract.delay_ms)
            for j in range(h + 1):
                if j <= steps[s]:
                    assert np.array_equal(la[s, m, j], ref.on_time_Mbit[k + j])      # bit for bit
                else:
                    assert np.all(la[s, m, j] == 0)
    # step 0 equals the one-interval counterfactual
    cf = fluid.counterfactual_window(A, ks * 1000, c0[ks], Qs, w.rate_true[ks], w.contract.delay_ms)
    assert np.array_equal(la[:, :, 0], cf.on_time_Mbit)
    # the stepwise version from a state equals the window version
    st = fluid.state_at(A, 4000, c0[4], w.contract.delay_ms)
    cp = fluid.counterfactual_path(st, A[4000:8000], Qs[1], w.rate_true[4], Q[5:8], w.rate_true[5:8],
                                   w.contract.delay_ms)
    assert np.array_equal(cp, la[1])
    with pytest.raises(ValueError):
        fluid.lookahead_window(A, ks * 1000, c0[ks], Qs, w.rate_true[ks], Q[nxt], w.rate_true[nxt],
                               w.contract.delay_ms)                    # steps default h runs past the window end


def _carryover_window(K=5):
    """Tenant 0 (deadline 2002 ms, pi 1.5) gets a 10-Mbit burst at the start of interval 1; tenant 1 (deadline
    100 ms, pi 1) offers a steady 5 Mbit/s; r = 1 Mbit/s per PRB, B = 10. Giving tenant 0 the whole cell in
    interval 1 wins that interval (15 > 12.5 credits) but loses the window: the burst is served on time in
    interval 2 anyway, while tenant 1's dropped bits are gone."""
    ct = Contract(cls=("video", "urllc"), g=np.zeros(2), M=np.full(2, 50.0), C=np.full(2, 50.0),
                  pi=np.array([1.5, 1.0]), delay_ms=np.array([2002.0, 100.0]), mu=np.array([2.0, 5.0]))
    arr = np.zeros((K * 1000, 2), np.float32)
    arr[1000, 0] = 10.0
    arr[:, 1] = 5e-3
    m = arr.astype(float).reshape(K, 1000, 2).sum(1)
    r = np.ones((K, 2))
    return Win("carry", "train", "2024-01-01", 2, ct.cls, arr, m.copy(), m, r, r.copy(), 1, ct, dict(tod_start_s=0.0))


def _split_quota(z, w, r, ct, mech):
    """Pure weight split of B (PRB), floored: makes the weights the action."""
    return np.floor(mech.B * w / w.sum(-1, keepdims=True) + 1e-9).astype(np.int64)


def test_f3_lookahead_reward_has_the_sign_of_the_window_welfare_change():
    w = _carryover_window()
    mech = Mechanism(B=10)
    env = P.FluidEnv(w, mech)
    path = np.full((w.measured.shape[0], 2), 5.0)
    env.replay(path)
    Qs = np.array([[[5.0, 5.0], [10.0, 0.0]]])                                      # (S=1, M=2, n)
    pi = w.contract.pi
    myopic = env.counterfactual_at(np.array([1]), Qs)[0] @ pi
    look = env.lookahead_at(np.array([1]), Qs, P.auto_horizon(w.contract, mech))[0].sum(1) @ pi
    full = []                                                          # window welfare from interval 1 on
    for q in Qs[0]:
        p2 = path.copy()
        p2[1] = q
        out, _, _ = fluid.replay(env.A, p2, w.rate_true, w.contract.delay_ms)
        full.append(float((out.on_time_Mbit[1:] @ pi).sum()))
    assert myopic[1] > myopic[0]                                       # one interval: the burst move looks better
    assert full[1] < full[0] - 1.0                                     # ... but the window loses > 1 credit
    assert look[1] < look[0]                                           # the lookahead return has the right sign
    assert look[1] - look[0] == pytest.approx(full[1] - full[0], abs=1e-9)   # h = 3 covers the rest of the window


def test_f3_ppo_rollout_rewards_are_lookahead_returns():
    ws = [synth_window(s, K=8) for s in range(2)]
    net = N.make_net("protected", seed=0)
    mech = Mechanism()
    data = P.prepare(ws, mech=mech, net=net, d=2, forecaster=FC, horizon=None)
    assert all(p.horizon == 3 for p in data)                          # ceil(2002 ms / 1 s)
    gen = torch.Generator().manual_seed(0)
    ls = torch.tensor(np.log(0.5), dtype=N.DTYPE)
    roll = P.rollout(net, data, mech=mech, env_factory=P.fluid_env_factory, quota_fn=P.default_quota_fn, M=3,
                     log_sigma=ls, gen=gen)
    # recompute the rewards of the first window from the sampled scores
    p0 = data[0]
    ks = p0.scored
    y = roll.y[:len(ks)]
    Qs = P.quotas(P.default_quota_fn, p0.z_alloc[ks][:, None, :], N.softmax_np(y), p0.r_hat[ks][:, None, :],
                  p0.contract, mech)
    with torch.no_grad():
        Yd = net(torch.as_tensor(p0.feats.c_tenant), torch.as_tensor(p0.feats.c_global),
                 torch.as_tensor(p0.zn)).numpy()
    env = P.FluidEnv(p0.window, mech)
    env.replay(P.quotas(P.default_quota_fn, p0.z_alloc, N.softmax_np(Yd), p0.r_hat, p0.contract, mech))
    R = env.lookahead_at(ks, Qs, 3).sum(2) @ p0.pi / p0.norm
    np.testing.assert_array_equal(R, roll.R[:len(ks)])
    # horizon 0 keeps the one-interval reward of INTERFACES.md §9
    d0 = P.prepare(ws[:1], mech=mech, net=net, d=2, forecaster=FC, horizon=0)
    r0 = P.rollout(net, d0, mech=mech, env_factory=P.fluid_env_factory, quota_fn=P.default_quota_fn, M=3,
                   log_sigma=ls, gen=torch.Generator().manual_seed(0))
    np.testing.assert_array_equal(r0.R, env.counterfactual_at(ks, Qs) @ p0.pi / p0.norm)
    with pytest.raises(ValueError):
        P.PPOConfig(horizon=-1)


def test_f3_oracle_lookahead_beats_the_myopic_oracle_on_carryover():
    w = _carryover_window()
    mech = Mechanism(B=10)
    kw = dict(pop=16, iters=4, elite=4, seed=0, quota_fn=_split_quota)
    myo = O.oracle_window(w, mech, horizon=0, **kw)
    look = O.oracle_window(w, mech, **kw)
    assert look.horizon == 3 and myo.horizon == 0
    assert myo.quotas[1, 0] >= 9                                       # myopic: (nearly) the whole cell to the burst
    assert look.welfare[w.warmup:].sum() > myo.welfare[w.warmup:].sum() + 1.0
    # replaying the lookahead oracle reproduces its own path
    env = P.FluidEnv(w, mech)
    assert np.array_equal(env.replay(look.quotas), look.on_time)
    with pytest.raises(ValueError):
        O.oracle_window(w, mech, base_weights=[1.0, -1.0], **kw)


# ============================================================================================ F5 forecaster
def test_f5_no_silent_default_forecaster():
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    w = synth_window(3, K=6)
    net = N.make_net("protected", seed=1)
    with pytest.raises(ValueError, match="forecaster"):
        F.context(w, 3, 2, w.contract)
    with pytest.raises(ValueError, match="forecaster"):
        F.window_features(w, 2, w.contract)
    with pytest.raises(ValueError, match="forecaster"):
        P.prepare([w], mech=Mechanism(), net=net, d=2)
    with pytest.raises(ValueError, match="forecaster"):
        P.evaluate(net, [w], d=2)
    shape = Shape(beta=(np.ones(1),) * 3, rho=(np.ones(1),) * 3)
    pols = [Truthful() for _ in range(3)]
    with pytest.raises(ValueError, match="forecaster"):
        E.run_window(w, net, "full", pols, 2, Mechanism(), shape)
    with pytest.raises(ValueError, match="forecaster"):
        E.run_window(w, E.MapController(lambda ctx: E.FixedMap(np.ones(3))), "full", pols, 2, Mechanism(), shape)
    E.run_window(w, E.FixedWeights(), "full", pols, 2, Mechanism(), shape)           # context-free: fine
    with pytest.raises(ValueError, match="horizon"):
        F.context(w, 3, 2, w.contract, forecaster=dataclasses.replace(FC, horizon=1))


def test_f5_train_attaches_forecaster_and_save_load_roundtrip(tmp_path):
    ws = [synth_window(s, K=8) for s in range(2)]
    fit = F.fit_operator_ar(ws, 2)
    cfg = P.PPOConfig(iterations=1, windows_per_iter=2, M=2, minibatch=16, eval_every=0)
    res = P.train(N.make_net("unconstrained", seed=0), ws, d=2, mech=Mechanism(), config=cfg, forecaster=fit)
    assert res.net.forecaster is fit
    np.testing.assert_array_equal(res.final_net.forecaster.coef["urllc"], fit.coef["urllc"])
    path = tmp_path / "net.pt"
    N.save_net(res.net, path)
    back = N.load_net(path)
    assert back.forecaster.horizon == 2
    for c in fit.coef:
        np.testing.assert_array_equal(back.forecaster.coef[c], fit.coef[c])
    assert N.param_digest(back) == N.param_digest(res.net)
    a = P.evaluate(back, ws, d=2)                                      # uses the carried forecaster
    b = P.evaluate(res.net, ws, d=2, forecaster=fit)
    assert a["welfare"] == b["welfare"]
    with pytest.raises(ValueError, match="horizon"):
        P.evaluate(back, ws, d=1)                                      # fitted for d = 2


def test_f5_calibration_carries_fitted_operator_forecasters():
    from slicemech.data.calibrate import Calibration, OPERATOR_DELAYS, fit_operator_forecasters
    cal = _load_cal()
    if not cal.operator_ar:
        pytest.skip("calibration predates operator_ar (run python -m slicemech.data.calibrate --operator-ar-only)")
    assert set(cal.operator_ar) == set(OPERATOR_DELAYS) == {1, 2, 5, 15}
    for d in OPERATOR_DELAYS:
        ar = cal.operator_forecaster(d)
        assert ar.horizon == d and set(ar.coef) == set(CLASSES)
        assert all(np.all(np.isfinite(v)) for v in ar.coef.values())
    with pytest.raises(ValueError, match="no operator forecaster"):
        cal.operator_forecaster(3)
    back = Calibration.from_dict(cal.to_dict())
    np.testing.assert_array_equal(back.operator_forecaster(2).coef["urllc"], cal.operator_forecaster(2).coef["urllc"])
    # the helper equals fit_operator_ar on TRAIN-split views
    ws = [synth_window(s, K=12) for s in range(3)]
    fits = fit_operator_forecasters([w.measured for w in ws], ws[0].contract, ds=(1, 2),
                                    keys=[("2024-01-01", 60 + 7200 * s) for s in range(3)])   # TRAIN keys
    np.testing.assert_array_equal(fits[2].coef["video"], F.fit_operator_ar(ws, 2).coef["video"])


# ============================================================================================ F8 split guards
def test_f8_split_guards():
    tr = [synth_window(s, K=6) for s in range(2)]
    te = [synth_window(10 + s, K=6, split="test") for s in range(2)]
    dv = [synth_window(20, K=6, split="dev")]
    net = N.make_net("protected", seed=0)
    cfg = P.PPOConfig(iterations=1, windows_per_iter=2, M=2, minibatch=16, eval_every=1)
    kw = dict(d=2, mech=Mechanism(), config=cfg, forecaster=FC)
    with pytest.raises(ValueError, match="training windows"):
        P.train(net, te, **kw)
    with pytest.raises(ValueError, match="development"):
        P.train(net, tr, tr, **kw)                                     # TRAIN windows as checkpoint selection
    with pytest.raises(ValueError, match="development"):
        P.train(net, tr, te, **kw)                                     # TEST windows as checkpoint selection
    P.train(net, tr, dv, **kw)
    P.train(net, te, allow_splits=("test",), **kw)                     # explicit, pre-registered exception
    with pytest.raises(ValueError, match="static-fit"):
        S.fit_static(te, Mechanism(), steps=2)
    with pytest.raises(ValueError, match="static-fit"):
        S.fit_static(dv, Mechanism(), steps=2)
    assert S.fit_static(te, Mechanism(), steps=2, allow_splits=("test",)).w.shape == (3,)
    rr = RT.retrain(net, te, "measurement", d=2, mech=Mechanism(), forecaster=FC, report_rebuild="measured",
                    config=P.PPOConfig(iterations=1, windows_per_iter=2, M=2, minibatch=16, eval_every=0))
    assert rr.report_rebuild == "measured"
    with pytest.raises(ValueError, match="allowed splits"):
        P.check_splits(tr, ("holdout",), "training")


# ============================================================================================ F9 report rebuild
def test_f9_rebuilt_reports_equal_truthful_theta_on_composed_windows():
    cal = _load_cal()
    for split, date, off in (("train", "2024-01-14", 60), ("dev", "2024-03-10", 60)):
        w = _real_window(cal, split, date, off, K=30, warmup=5)
        v = RT.MeasurementView(w)
        h, exact = RT.own_history_from_ingress(v.arrivals_ms, w.K, v.meta["report_lead_s"])
        assert exact
        np.testing.assert_array_equal(h, w.meta["own_history"])
        z, info = RT.rebuild_reports(v, cal.forecaster)
        assert info["exact_bytes"] is True
        np.testing.assert_array_equal(z, w.theta)                      # bit for bit, without reading theta
    short = _real_window(cal, "train", "2024-01-14", 60, K=12, warmup=2, lead_s=1.0)
    with pytest.raises(ValueError, match="persistence"):
        RT.rebuild_reports(RT.MeasurementView(short), cal.forecaster)
    with pytest.raises(ValueError, match="tenant AR"):
        RT.rebuild_reports(RT.MeasurementView(w), None)


def test_f9_measurement_retraining_trains_on_deployment_like_reports():
    """With report_rebuild='tenant_ar' the measurement pipeline sees exactly the truthful reports of deployment
    (bit-identical parameters to training on theta), still without reading any report."""
    cal = _load_cal()
    ws = [_real_window(cal, "dev", d, 60, K=12, warmup=3) for d in ("2024-03-02", "2024-03-04")]
    fit = cal.operator_forecaster(2)
    base = N.make_net("protected", seed=0)
    cfg = P.PPOConfig(iterations=2, windows_per_iter=2, M=2, minibatch=32, eval_every=0)
    kw = dict(d=2, mech=Mechanism(), config=cfg, seed=3, forecaster=fit)
    meas = RT.retrain(base, ws, "measurement", tenant_forecaster=cal.forecaster, **kw)
    ref = P.train(base, ws, report_source="theta", allow_splits=("dev",), **kw)
    assert N.param_digest(meas.net) == N.param_digest(ref.net)
    lied = [dataclasses.replace(w, theta=np.minimum(2.0 * w.theta, w.contract.C)) for w in ws]
    again = RT.retrain(base, lied, "measurement", tenant_forecaster=cal.forecaster, **kw)
    assert N.param_digest(again.net) == N.param_digest(meas.net)       # reports never read
    with pytest.raises(ValueError, match="tenant_forecaster"):
        RT.retrain(base, ws, "measurement", **kw)


# ============================================================================================ F10 report prior
def test_f10_unconstrained_net_starts_with_the_protected_report_prior(tmp_path):
    w = synth_window(5, K=6)
    ctx = F.context(w, 4, 2, w.contract, forecaster=FC)
    un = N.make_net("unconstrained", seed=1)
    old = N.make_net("unconstrained", seed=1, report_prior=None)
    assert float(un.report_skip.detach()) == 1.0 and old.report_skip is None
    assert un.n_parameters() == old.n_parameters() + 1
    sd_old = old.state_dict()
    for k, v in un.state_dict().items():                              # the seeded encoder/head init is unchanged
        if k != "report_skip":
            assert torch.equal(v, sd_old[k])
    c_t = torch.as_tensor(ctx[0])[None].expand(5, -1, -1)
    c_g = torch.as_tensor(ctx[1])[None].expand(5, -1)
    zn = torch.as_tensor(np.linspace(0.1, 3.0, 5)[:, None] * np.ones((1, 3)))
    with torch.no_grad():
        np.testing.assert_allclose((un(c_t, c_g, zn) - old(c_t, c_g, zn)).numpy(), torch.log1p(zn).numpy(),
                                   rtol=0, atol=1e-12)
    # the deployed map snapshot includes the skip term
    Z = w.theta[:4]
    with torch.no_grad():
        y = un(torch.as_tensor(np.repeat(ctx[0][None], 4, 0)), torch.as_tensor(np.repeat(ctx[1][None], 4, 0)),
               torch.as_tensor(Z / w.contract.mu)).numpy()
    np.testing.assert_allclose(un.priority_map(ctx).weights(Z), N.softmax_np(y), rtol=1e-12)
    # report sensitivity at init: same order as the protected net (kappa ~ 1)
    pr = N.make_net("protected", seed=1)
    _, kappa = pr.heads_np(ctx)
    assert np.all(np.abs(kappa - 1.0) < 0.5)
    N.save_net(un, tmp_path / "u.pt")
    assert N.param_digest(N.load_net(tmp_path / "u.pt")) == N.param_digest(un)
    with pytest.raises(ValueError):
        N.make_net("unconstrained", report_prior=float("nan"))


# ============================================================================================ F1 pools and bootstrap
def test_f1_pool_parts_are_disjoint_halves():
    tractor = pytest.importorskip("slicemech.data.tractor")
    raca = pytest.importorskip("slicemech.data.raca")
    try:
        full = tractor.pool("dev", "urllc")
        runs = raca.pool("dev", 201)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"prepared data missing: {exc}")
    a, b = tractor.pool("dev", "urllc", "A"), tractor.pool("dev", "urllc", "B")
    for t, ta, tb in zip(full, a, b):
        assert ta.stem == f"{t.stem}#A" and tb.stem == f"{t.stem}#B"
        assert tractor.capture_of(ta.stem) == t.stem
        # round 2: the halves are interleaved 2-s blocks (tests/test_review_round2.py), still a partition
        assert len(ta) + len(tb) == len(t) and ta.bytes_ms.sum() + tb.bytes_ms.sum() == t.bytes_ms.sum()
        np.testing.assert_array_equal(ta.bytes_ms, t.bytes_ms[tractor.part_mask(len(t), "A")])
    ra, rb = raca.pool("dev", 201, "A"), raca.pool("dev", 201, "B")
    ida, idb = {r.run_id for r in ra}, {r.run_id for r in rb}
    assert not ida & idb and ida | idb == {r.run_id for r in runs} and abs(len(ida) - len(idb)) <= 1
    with pytest.raises(ValueError):
        tractor.pool("dev", "urllc", "C")


def test_f1_e5_windows_from_disjoint_parts_share_no_exogenous_source():
    from slicemech.tenants.poison import PoisonPlan, plan_date
    cal = _load_cal()
    wa = _real_window(cal, "dev", "2024-03-10", 60, K=30, warmup=5, pool_part="A")
    wb = _real_window(cal, "dev", "2024-03-10", 14460, K=30, warmup=5, pool_part="B")
    w0 = _real_window(cal, "dev", "2024-03-10", 60, K=30, warmup=5)
    assert wa.id != w0.id and "-pA-" in wa.id and wa.meta["pool_part"] == "A"
    users_a = {s for t in wa.meta["tenants"] if "users" in t for s, _ in t["users"]}
    users_b = {s for t in wb.meta["tenants"] if "users" in t for s, _ in t["users"]}
    # round 2: the E5 phase maps to a pool half per date (2024-03-10 swaps them); the halves stay disjoint
    ha, hb = wa.meta["pool_half"], wb.meta["pool_half"]
    assert {ha, hb} == {"A", "B"}
    assert all(s.endswith(f"#{ha}") for s in users_a) and all(s.endswith(f"#{hb}") for s in users_b)
    runs_a = {t["cqi_run"]["run"] for t in wa.meta["tenants"]}
    runs_b = {t["cqi_run"]["run"] for t in wb.meta["tenants"]}
    assert not runs_a & runs_b
    plan = plan_date(["a1", "a2", "b1", "b2"], target=1, alpha=2.0, n_train=2, n_eval=2)
    assert (plan.train_part, plan.eval_part) == ("A", "B")
    with pytest.raises(ValueError, match="disjoint pool parts"):
        PoisonPlan(target=1, alpha=2.0, train_ids=("a",), eval_ids=("b",), train_part="A", eval_part="A")


def test_f1_crossed_bootstrap_accounts_for_shared_sources():
    from slicemech.eval.stats import cluster_bootstrap, crossed_bootstrap, window_memberships
    rng = np.random.default_rng(0)
    G, per, L = 16, 8, 3                                               # 16 dates, 8 windows, 3 shared captures
    dates = np.repeat(np.arange(G), per)
    cap = rng.integers(L, size=G * per)
    effect = rng.normal(0, 1.0, L)                                     # a strong shared capture effect
    x = 0.2 + effect[cap] + rng.normal(0, 0.3, G * per)
    A_date = np.eye(G)[dates]
    A_cap = np.eye(L)[cap]
    date_only = cluster_bootstrap(x, dates, n_boot=4000, seed=1)
    crossed = crossed_bootstrap(x, {"date": A_date, "capture": A_cap}, n_boot=4000, seed=1)
    assert crossed.estimate == pytest.approx(x.mean())
    assert (crossed.hi - crossed.lo) > 2.0 * (date_only.hi - date_only.lo)   # dates alone understate the spread
    one = crossed_bootstrap(np.full(G * per, 3.0), {"date": A_date}, n_boot=200, seed=0)
    assert one.lo == one.hi == 3.0
    with pytest.raises(ValueError):
        crossed_bootstrap(x, {"date": A_date * 2.0}, n_boot=10)
    metas = [dict(date="2024-09-01", tenants=[dict(cls="video", cqi_run=dict(run=1)),
                                             dict(cls="urllc", users=[("u1#A", 0), ("u1#A", 5), ("u2#A", 1)],
                                                  cqi_run=dict(run=2))]),
             dict(date="2024-09-02", tenants=[dict(cls="video", cqi_run=dict(run=2)),
                                             dict(cls="urllc", users=[("u2#B", 0)], cqi_run=dict(run=2))])]
    mem = window_memberships(metas)
    labels, A = mem["tractor_urllc"]
    assert labels == ["u1", "u2"]
    np.testing.assert_allclose(A, [[2 / 3, 1 / 3], [0.0, 1.0]])
    np.testing.assert_allclose(mem["raca_run"][1], [[0.5, 0.5], [0.0, 1.0]])
    np.testing.assert_allclose(mem["date"][1], np.eye(2))


# ============================================================================================ F2 TEST gate
def test_f2_data_report_withholds_test_statistics(monkeypatch):
    calib = pytest.importorskip("slicemech.data.calibrate")
    from slicemech import io as sio
    from slicemech.data import raca, tractor
    cal = _load_cal()
    real_r, real_t = raca.pool, tractor.pool

    def no_test_r(split, *a, **k):
        assert split != "test", "TEST Raca statistics were computed"
        return real_r(split, *a, **k)

    def no_test_t(split, *a, **k):
        assert split != "test", "TEST TRACTOR statistics were computed"
        return real_t(split, *a, **k)

    monkeypatch.setattr(raca, "pool", no_test_r)
    monkeypatch.setattr(tractor, "pool", no_test_t)
    out = sio.ROOT / "results" / "raw" / "_pytest_data_report.md"
    try:
        try:
            calib.data_report(cal, path=out)
        except (FileNotFoundError, OSError) as exc:
            pytest.skip(f"prepared data missing: {exc}")
        text = out.read_text()
    finally:
        out.unlink(missing_ok=True)
    assert "| TEST |" in text and "withheld" in text
    assert "urllc_03_03" not in text and "mmtc_04_10" not in text
    with pytest.raises(PermissionError):
        calib.data_report(cal, path=out, test_freeze=sio.ROOT / "prereg" / "PREREG_DRAFT.md")
    assert not out.exists()


# ============================================================================================ F4 headroom
def test_f4_welfare_headroom_diagnostic():
    from slicemech.eval.headroom import summarize, window_headroom, window_load
    w = synth_window(7, K=8)
    shape = Shape(beta=(np.ones(1),) * 3, rho=(np.ones(1),) * 3)
    mech = Mechanism()
    row = window_headroom(w, shape, mech, static_w=np.array([0.2, 0.6, 0.2]), grid_steps=4,
                          oracle_kw=dict(pop=8, iters=2, elite=2, seed=0))
    for k in ("W_equal", "W_static", "W_grid", "W_direct", "W_oracle"):
        assert np.isfinite(row[k]) and row[k] > 0
    assert row["W_grid"] >= row["W_equal"] - 1e-12                     # the grid contains equal weights
    assert row["G4_gap"] == pytest.approx((row["W_oracle"] - max(row["W_static"], row["W_direct"])) / row["W_equal"])
    assert row["load"] == pytest.approx(window_load(w, mech))
    s = summarize([row, dict(row, grid_gain=0.0, load=row["load"] + 0.1)])
    assert s["windows"] == 2 and 0.0 <= s["share_grid_gain_below"] <= 1.0
