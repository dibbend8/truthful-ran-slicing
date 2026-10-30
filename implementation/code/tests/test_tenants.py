"""Tests for slicemech.tenants (strategies, bandits, poison)."""
from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest

from slicemech.tenants import bandits as B
from slicemech.tenants import poison as P
from slicemech.tenants.strategies import (
    BestResponder,
    Multiplier,
    OwnHistory,
    StageGame,
    TenantPolicy,
    Truthful,
    admit,
    multiplier_profile,
    profile_reports,
)

BR = namedtuple("BR", "z_star gain U_truth U_star exact")


# --------------------------------------------------------------------------- strategies
def test_truthful_and_admission_counting():
    t = Truthful()
    assert isinstance(t, TenantPolicy)
    assert t.report(0, 3.5, 10.0) == 3.5 and t.n_clipped == 0
    assert t.report(0, 12.0, 10.0) == 10.0 and t.n_clipped == 1          # contract admission, counted
    with pytest.raises(ValueError):
        t.report(0, -1.0, 10.0)
    with pytest.raises(ValueError):
        t.report(0, np.nan, 10.0)
    with pytest.raises(ValueError):
        t.report(0, 1.0, 0.0)


def test_multiplier_series_matches_per_interval():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 8, 50)
    m = Multiplier(1.5)
    z = m.report_series(theta, 10.0)
    m2 = Multiplier(1.5)
    z2 = np.array([m2.report(0, th, 10.0) for th in theta])
    np.testing.assert_array_equal(z, z2)
    np.testing.assert_allclose(z, np.minimum(1.5 * theta, 10.0))
    assert m.n_clipped == m2.n_clipped == int((1.5 * theta > 10).sum()) > 0
    assert Multiplier(0).report(0, 5.0, 10.0) == 0.0
    for bad in (-0.1, np.inf, np.nan):
        with pytest.raises(ValueError):
            Multiplier(bad)


def test_profile_reports_shapes():
    theta = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    z = profile_reports(multiplier_profile([1.0, 2.0, 0.5]), theta, np.array([10.0, 9.0, 10.0]))
    np.testing.assert_allclose(z, [[1, 4, 1.5], [4, 9, 3]])
    with pytest.raises(ValueError):
        profile_reports([Truthful()], theta, np.array([10.0]))
    with pytest.raises(TypeError):
        profile_reports([Truthful(), BestResponder("full"), Truthful()], theta, np.ones(3) * 10)


def test_truthful_is_multiplier_one():
    t = Truthful()
    assert isinstance(t, Multiplier) and t.m == 1.0 and t.name == "truthful" and repr(t) == "Truthful()"
    theta = np.array([0.0, 1.0, 11.0])
    np.testing.assert_array_equal(t.report_series(theta, 10.0), [0.0, 1.0, 10.0])


def test_admit_rejects_negative():
    with pytest.raises(ValueError):
        admit(np.array([-1e-3]), np.array([1.0]))
    z, c = admit(np.array([0.5, 2.0]), np.array([1.0, 1.0]))
    np.testing.assert_array_equal(z, [0.5, 1.0])
    np.testing.assert_array_equal(c, [False, True])


def _toy_rule(z):
    """Monotone toy quota rule (PRB) for two tenants sharing B = 10 proportionally, floored."""
    z = np.asarray(z, float)
    return np.floor(np.minimum(z, 10.0 * z / np.maximum(z.sum(-1, keepdims=True), 1e-12)) + 1e-9)


def test_best_responder_forwards_game_and_validates():
    calls = []

    def fake_br(i, theta, z_minus_i, rule, price, **kw):
        calls.append((i, theta, tuple(z_minus_i), price, kw))
        return BR(z_star=2 * theta, gain=0.25, U_truth=1.0, U_star=1.25, exact=True)

    game = StageGame(rule=_toy_rule, z_minus_i=np.array([4.0]), kwargs=dict(r=np.ones(2), nu=None))
    br = BestResponder("fixed", br_fn=fake_br)
    assert br.needs_game and isinstance(br, TenantPolicy)
    assert br.report(0, 3.0, 10.0, history=OwnHistory(), game=game) == 6.0
    assert calls == [(0, 3.0, (4.0,), "fixed", dict(r=game.kwargs["r"], nu=None))]
    assert br.log == [(0.25, True)] and br.last.U_star == 1.25
    with pytest.raises(ValueError):
        br.report(0, 3.0, 10.0)                                            # no game
    with pytest.raises(ValueError):
        br.report(0, 6.0, 10.0, game=game)                                 # z* = 12 > C
    with pytest.raises(ValueError):
        BestResponder("vcg")
    br2 = BestResponder("full", br_fn=lambda i, th, zm, rule, price, **kw: BR(th, 0.0, 1.0, 1.0, True))
    assert br2.report(0, 12.0, 10.0, game=game) == 10.0 and br2.n_clipped == 1   # type above C is admitted first
    br.reset()
    assert br.log == [] and br.last is None


def test_best_responder_with_brute_force_stub_truthful_under_full_price():
    """Wrapper check with a brute-force stage best response under the envelope price of a 1-D toy rule
    (r = pi = Delta = 1, unit atom: V(x, t) = min(t, x), D(x, u) = 1{u <= x})."""
    grid = np.linspace(0, 10, 2001)

    def X(u):
        return np.floor(np.minimum(u, 6.0) + 1e-9)

    def full_price(z):
        u = grid[grid <= z]
        return 0.0 if u.size < 2 else min(z, X(z)) - np.trapezoid((u <= X(u)).astype(float), u)

    prices = np.array([full_price(z) for z in grid])

    def br_fn(i, theta, z_minus_i, rule, price, **kw):
        util = np.minimum(theta, X(grid)) - prices
        j = int(np.argmax(util))
        ut = min(theta, X(theta)) - full_price(theta)
        return BR(z_star=grid[j], gain=util[j] - ut, U_truth=ut, U_star=util[j], exact=False)

    br = BestResponder("full", br_fn=br_fn)
    for th in (2.5, 4.2, 7.0):
        br.report(0, th, 10.0, game=StageGame(rule=X, z_minus_i=np.array([])))
        assert br.last.gain <= 2e-2                                        # grid resolution only


def _real_game():
    """A congested n = 3 interval built from the real allocation / scores / pricing modules."""
    from slicemech.config import Mechanism, Shape, contract_from_params
    from slicemech.model.allocation import QuotaRule
    from slicemech.model.scores import ProtectedMap

    mu = np.array([25.0, 12.0, 4.0])
    ct = contract_from_params(["video", "urllc", "mmtc"], mu)
    mech = Mechanism()
    r = np.full(3, 0.43)
    atoms = np.array([0.25, 0.75, 1.25, 1.75])
    shape = Shape(beta=(atoms,) * 3, rho=(np.full(4, 0.25),) * 3)
    rule = QuotaRule(ct, mech, r, ProtectedMap(np.array([0.3, -0.2, 0.0]), np.array([1.0, 0.5, 1.5]), mu))
    return ct, mech, r, shape, rule


@pytest.mark.parametrize("price", ["full", "fixed"])
def test_best_responder_with_real_pricing_engine(price):
    pytest.importorskip("slicemech.pricing.response")
    ct, mech, r, shape, rule = _real_game()
    theta = np.array([24.0, 13.0, 3.5])
    eps = 1e-8 * ct.pi * ct.mu * mech.Delta
    br = BestResponder(price)
    for i in range(3):
        game = StageGame(rule=rule, z_minus_i=np.delete(theta, i), kwargs=dict(ct=ct, shape=shape, mech=mech, r=r))
        z = br.report(i, theta[i], ct.C[i], game=game)
        gain, exact = br.log[-1]
        assert exact and 0.0 <= z <= ct.C[i]
        if price == "full":
            assert gain <= eps[i]                                       # Prop. 2: truthful under the full price
        else:
            assert gain >= -eps[i]                                      # truth is a candidate: gain never negative


# --------------------------------------------------------------------------- bandits
def test_eps_schedule():
    assert B.eps_schedule(0) == 1.0
    assert B.eps_schedule(100) == pytest.approx(0.985 ** 100)
    assert B.eps_schedule(250) == 0.05
    with pytest.raises(ValueError):
        B.eps_schedule(-1)
    assert list(B.MULTIPLIERS) == [0.5, 0.67, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2]
    assert B.MULTIPLIERS[B.TRUTH_ARM] == 1.0


def test_squash_is_strictly_increasing_and_bounded():
    g = np.linspace(-5, 5, 101)
    s = B.squash(g, 0.5)
    assert np.all(np.diff(s) > 0) and s.min() >= 0 and s.max() <= 1 and B.squash(0.0) == 0.5


def test_eps_greedy_learns_deterministic_best_arm():
    rng = np.random.default_rng(1)
    lr = B.EpsGreedy(5, rng)
    means = np.array([0.0, -0.1, 0.3, 0.1, -0.2])
    for e in range(400):
        a = lr.select(e)
        lr.update(a, means[a])
    assert lr.greedy() == 2
    p = lr.probabilities(400)
    assert p.sum() == pytest.approx(1.0) and p[2] == pytest.approx(0.95 + 0.05 / 5)


def test_exp3_probabilities_and_learning():
    rng = np.random.default_rng(2)
    lr = B.EXP3(9, rng, horizon=3000)
    p = lr.probabilities()
    assert p.sum() == pytest.approx(1.0) and np.all(p >= lr.gamma / 9 - 1e-15)
    mean = np.full(9, 0.4)
    mean[6] = 0.9
    for _ in range(3000):
        a = lr.select()
        lr.update(a, float(rng.random() < mean[a]))
    assert lr.greedy() == 6 and lr.probabilities()[6] > 0.5
    with pytest.raises(ValueError):
        lr.update(0, 1.5)


def _game(peak: float, amp: float = 0.05, curv: float = 4.0):
    """Synthetic window game: tenant i's window utility is base * (1 + amp * (1 - curv * ln(m_i / peak)**2)),
    i.e. it peaks at multiplier `peak` (relative gains of a few percent, as expected in E2/E3) and ignores others."""
    calls = []

    def evaluate(wid, m):
        calls.append((wid, tuple(m)))
        base = 10.0 + wid                                     # per-window level (common random numbers)
        u = base * (1.0 + amp * (1.0 - curv * np.log(np.asarray(m) / peak) ** 2))
        return {"utility": u, "welfare": float(u.sum())}
    return evaluate, calls


@pytest.mark.parametrize("algo", ["eps", "exp3"])
def test_bandits_stay_truthful_when_truth_is_optimal(algo):
    ev, calls = _game(peak=1.0, amp=1.0, curv=5.0)          # neighbouring arms lose >= 4.5 % relative utility
    run = B.run_learning_tenants(ev, window_ids=list(range(6)), n=3, algo=algo, episodes=400, seed=3)
    np.testing.assert_array_equal(run.final_greedy(), [1.0, 1.0, 1.0])
    assert np.all(run.gain <= 1e-12)                          # no profile beats truth
    assert run.extras["welfare"].shape == (400,)
    assert len(calls) == len(set(calls))                      # memoised: each (window, profile) simulated once


def test_eps_greedy_finds_profitable_inflation_exactly():
    ev, _ = _game(peak=1.5)
    run = B.run_learning_tenants(ev, window_ids=list(range(4)), n=3, algo="eps", episodes=400, seed=4)
    np.testing.assert_array_equal(run.final_greedy(), [1.5, 1.5, 1.5])
    assert run.final_play_share(50).shape == (3, 9)


def test_exp3_moves_mass_to_profitable_inflation():
    """EXP3's regret over 400 episodes exceeds the gaps between neighbouring inflation arms, so the test checks the
    end-state distribution (what E2/E3 report), not the exact arm: over 20 seeds the observed minima were 0.79
    (mass on m > 1) and 1.22 (mean multiplier of the last 100 episodes)."""
    ev, _ = _game(peak=1.5)
    memo = B.MemoizedEvaluator(ev)
    for seed in range(5):
        run = B.run_learning_tenants(memo, window_ids=list(range(4)), n=3, algo="exp3", episodes=400, seed=seed)
        assert np.all(run.final_prob[:, B.MULTIPLIERS > 1].sum(1) >= 0.7)
        assert np.all(run.m[-100:].mean(0) >= 1.15)
    ev, _ = _game(peak=1.0)
    run = B.run_learning_tenants(ev, window_ids=list(range(4)), n=3, algo="exp3", episodes=400, seed=0)
    assert np.all(np.abs(run.m[-100:].mean(0) - 1.0) <= 0.15)


def test_bandit_determinism_shared_memo_and_errors():
    ev, calls = _game(peak=1.0)
    runs, memo = B.run_design(ev, list(range(3)), n=3, algo="exp3", runs=2, episodes=60, seed0=7)
    again, _ = B.run_design(_game(peak=1.0)[0], list(range(3)), n=3, algo="exp3", runs=2, episodes=60, seed0=7)
    for r1, r2 in zip(runs, again, strict=True):
        np.testing.assert_array_equal(r1.arm, r2.arm)
        np.testing.assert_array_equal(r1.window, r2.window)
    assert memo.calls == len(memo.cache) == len(calls)
    with pytest.raises(ValueError):
        B.run_learning_tenants(lambda w, m: np.zeros(3), [0], n=3, episodes=2)   # zero truthful utility
    with pytest.raises(ValueError):
        B.run_learning_tenants(ev, [0], n=3, arms=[0.5, 2.0])                     # no truthful arm
    with pytest.raises(ValueError):
        B.run_learning_tenants(lambda w, m: np.ones(2), [0], n=3, episodes=1)     # wrong shape


# --------------------------------------------------------------------------- poison
def _windows(n_win=8, K=20, seed=0):
    rng = np.random.default_rng(seed)
    ct = SimpleNamespace(C=np.array([10.0, 3.0, 1.0]))
    return [SimpleNamespace(id=f"d1-w{j}", theta=rng.uniform(0, 4, (K, 3)) * [1, 0.5, 0.1], contract=ct,
                            measured=rng.uniform(0, 4, (K, 3)), cls=("video", "urllc", "mmtc"))
            for j in range(n_win)]


def test_poison_plan_and_policies():
    ws = _windows()
    t = P.target_index(ws[0].cls, "urllc")
    assert t == 1 and P.target_index(ws[0].cls, 0) == 0
    with pytest.raises(ValueError):
        P.target_index(ws[0].cls, "embb")
    plan = P.plan_date([w.id for w in ws], target=t, alpha=2.0)
    assert plan.train_ids == tuple(f"d1-w{j}" for j in range(4))
    assert plan.eval_ids == tuple(f"d1-w{j}" for j in range(4, 8))
    pols = plan.policies("d1-w0", 3)
    assert isinstance(pols[1], Multiplier) and pols[1].m == 2.0
    assert isinstance(pols[0], Truthful) and isinstance(pols[2], Truthful)
    assert all(isinstance(p_, Truthful) for p_ in plan.policies("d1-w5", 3))
    with pytest.raises(ValueError):
        P.plan_date([w.id for w in ws[:5]], target=t, alpha=2.0)
    with pytest.raises(ValueError):
        P.PoisonPlan(target=1, alpha=2.0, train_ids=("a",), eval_ids=("a",))


@pytest.mark.parametrize("alpha", P.ALPHAS)
def test_stored_reports_distort_only_target_in_train_windows(alpha):
    ws = _windows()
    meas_before = [w.measured.copy() for w in ws]
    plan = P.plan_date([w.id for w in ws], target=1, alpha=alpha)
    rep, clips = P.stored_reports(ws, plan)
    assert set(rep) == set(plan.train_ids)
    for w in ws[:4]:
        z = rep[w.id]
        np.testing.assert_array_equal(z[:, [0, 2]], w.theta[:, [0, 2]])
        np.testing.assert_allclose(z[:, 1], np.minimum(alpha * w.theta[:, 1], 3.0))
        np.testing.assert_array_equal(clips[w.id], [0, int((alpha * w.theta[:, 1] > 3.0).sum()), 0])
    for w, m in zip(ws, meas_before, strict=True):          # measurements are never touched (H5a precondition)
        np.testing.assert_array_equal(w.measured, m)
    with pytest.raises(ValueError):
        P.stored_reports(ws[1:], plan)                       # a planned window is missing


def test_distort_counts_non_target_clips():
    theta = np.array([[12.0, 1.0, 0.5], [3.0, 2.5, 0.2]])
    z, clips = P.distort(theta, np.array([10.0, 3.0, 1.0]), target=1, alpha=1.5)
    np.testing.assert_allclose(z, [[10.0, 1.5, 0.5], [3.0, 3.0, 0.2]])
    np.testing.assert_array_equal(clips, [1, 1, 0])                     # the truthful video type above C counts too


def test_distort_validation():
    with pytest.raises(ValueError):
        P.distort(np.ones((3, 3)), np.ones(3), 0, 0.0)
    with pytest.raises(ValueError):
        P.distort(np.ones(3), np.ones(3), 0, 2.0)
    with pytest.raises(ValueError):
        P.distort(-np.ones((2, 3)), np.ones(3), 0, 2.0)
