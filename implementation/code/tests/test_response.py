"""Best responses: exact enumeration vs dense grids, certificates, menu regression, non-monotone manipulation."""
from __future__ import annotations

import numpy as np
import pytest

from _core_fixtures import random_case, random_shape
from slicemech.config import Contract, Mechanism, Shape
from slicemech.model.allocation import QuotaRule
from slicemech.model.scores import GeneralScoreMap, ProtectedMap
from slicemech.pricing.response import PRICES, best_response, utilities
from slicemech.pricing.thresholds import own_thresholds
from test_pricing import _menu_game


def _protected(rng, n, load=None):
    c = random_case(rng, n, zero_prob=0.1, load=load)
    pm = ProtectedMap(rng.uniform(-4, 4, n), rng.uniform(0, 2, n), c["mu"])
    shape = random_shape(rng, n, uniform=bool(rng.random() < 0.5))
    return c, QuotaRule(c["ct"], c["mech"], c["r"], pm), shape


@pytest.mark.parametrize("n", [3, 4])
def test_monotone_best_response_dominates_dense_grid(n):
    """Exact cell enumeration >= a dense 4096-point grid for every price; full-price gain <= certificate."""
    rng = np.random.default_rng(500 + n)
    for _ in range(8):
        c, rule, shape = _protected(rng, n, load=float(rng.uniform(1.0, 3.0)))
        ct = c["ct"]
        i = int(rng.integers(n))
        theta = float(rng.uniform(0, ct.C[i]))
        opp = np.delete(c["z"], i)
        nu = rng.uniform(0.05, 0.5, n)
        grid = np.linspace(0, ct.C[i], 4096)
        for price in PRICES:
            br = best_response(i, theta, opp, rule, price, shape=shape, nu=nu)
            assert br.exact and br.gain >= 0 and br.U_star >= br.U_truth
            dense = utilities(i, theta, opp, rule, grid, price, shape=shape, nu=nu)
            tol = br.cert + 1e-9 * ct.pi[i] * c["mech"].Delta * ct.C[i]
            assert br.U_star >= dense.max() - tol, (price, br, dense.max())
            again = utilities(i, theta, opp, rule, [br.z_star], price, shape=shape, nu=nu)[0]
            assert again == pytest.approx(br.U_star, abs=tol)
            if price == "full":
                assert br.gain <= br.cert + 1e-12                       # DSIC up to the certified price error


def test_full_price_gain_certificate_many():
    rng = np.random.default_rng(42)
    for _ in range(60):
        n = int(rng.integers(3, 8))
        c, rule, shape = _protected(rng, n, load=float(rng.uniform(0.5, 4.0)))
        i = int(rng.integers(n))
        theta = float(rng.uniform(0, c["ct"].C[i]))
        br = best_response(i, theta, np.delete(c["z"], i), rule, "full", shape=shape)
        assert br.exact and 0 <= br.gain <= br.cert + 1e-12
        assert br.cert <= 1e-9 * c["ct"].pi[i] * c["ct"].C[i]


def _e4_rule(scale: int = 10):
    """theory_design.md E4 (n = 2, B = 10, eta = 1.5, atoms 1/4 .. 7/4, z_2 = 10) in integer PRB at ``scale`` x the
    resolution: B = 10 scale PRB with r = 1 / scale Mbit/s per PRB, so quotas are floor(scale * X)."""
    ct = Contract(cls=("video",) * 2, g=np.zeros(2), M=np.full(2, 20.0), C=np.full(2, 20.0), pi=np.ones(2),
                  delay_ms=np.full(2, 100.0), mu=np.full(2, 5.0))
    mech = Mechanism(B=10 * scale, gamma=0.0, eta=1.5, Delta=1.0, integer=True)
    shape = Shape(beta=(np.array([0.25, 0.75, 1.25, 1.75]),) * 2, rho=(np.full(4, 0.25),) * 2)
    rule = QuotaRule(ct, mech, np.full(2, 1.0 / scale), ProtectedMap([0.0, 0.0], [1.0, 1.0], ct.mu))
    return rule, shape


def test_e4_fixed_price_is_manipulable_full_price_is_not():
    """E4 at theta = 8: inflating to the ceiling gains 0.75 under p_fix (Prop. 3(iv): at most T(20) - T(8)),
    0 under the full price (truth kept); without a price the gain is larger still."""
    rule, shape = _e4_rule()
    full = best_response(0, 8.0, [10.0], rule, "full", shape=shape)
    fixed = best_response(0, 8.0, [10.0], rule, "fixed", shape=shape)
    none = best_response(0, 8.0, [10.0], rule, "none", shape=shape)
    assert full.exact and full.gain == 0.0 and full.z_star == 8.0
    assert fixed.exact and fixed.z_star == pytest.approx(20.0) and fixed.gain == pytest.approx(0.75, abs=1e-6)
    assert none.gain > fixed.gain
    from slicemech.pricing.envelope import premium
    T = [premium(0, np.array([u, 10.0]), rule, rule.ct, shape, rule.mech, rule.r).T for u in (8.0, 20.0)]
    assert fixed.gain <= T[1] - T[0] + 2 * fixed.cert


def test_menu_regression_best_response():
    """premium_witnesses.current_regrets at truth (10, 40, 40): full 0, frozen 3, no charge 3."""
    ct, mech, shape, rule = _menu_game()
    for price, gain in (("full", 0.0), ("fixed", 3.0), ("none", 3.0)):
        br = best_response(0, 10.0, [40.0, 40.0], rule, price, shape=shape)
        assert br.exact
        assert br.gain == pytest.approx(gain, abs=1e-6), (price, br)


def _e1_game():
    """Theorem 1 example E1 with integer quotas: y_0 = psi(z_0), y_1 = 2 psi(z_0), psi = log(1 + z / 5).

    B = 20, eta = 4, M = (20, 20, 0), z_1 = 20, bystander z_2 = 1 with M_2 = 0, unit atom: X_0(5) = 6 > X_0(10) = 5.
    """
    ct = Contract(cls=("video",) * 3, g=np.zeros(3), M=np.array([20.0, 20.0, 0.0]), C=np.array([20.0, 20.0, 1.0]),
                  pi=np.ones(3), delay_ms=np.full(3, 100.0), mu=np.full(3, 5.0))
    mech = Mechanism(B=20, gamma=0.0, eta=4.0, Delta=1.0, integer=True)
    pm = GeneralScoreMap(np.zeros(3), [[1, 0, 0], [2, 0, 0], [0, 0, 0]], ct.mu)
    rule = QuotaRule(ct, mech, np.ones(3), pm)
    unit = Shape(beta=(np.ones(1),) * 3, rho=(np.ones(1),) * 3)
    return ct, mech, rule, unit


def test_nonmonotone_rule_is_manipulable_under_every_price():
    """Lemma C: with X(5) = 6 > X(10) = 5, any payment leaves type 5 or type 10 a gain >= 1/2."""
    ct, mech, rule, unit = _e1_game()
    assert not rule.monotone_by_construction
    assert rule(np.array([5.0, 20.0, 1.0]))[0] == 6 and rule(np.array([10.0, 20.0, 1.0]))[0] == 5
    for price in PRICES:
        gains = [best_response(0, t, [20.0, 1.0], rule, price, shape=unit, nu=0.3) for t in (5.0, 10.0)]
        assert all(not g.exact and np.isnan(g.cert) for g in gains)
        assert max(g.gain for g in gains) >= 0.5 - 1e-9, (price, gains)


def test_layouts_and_validation():
    rng = np.random.default_rng(7)
    c, rule, shape = _protected(rng, 4)
    a = best_response(1, 1.0, np.delete(c["z"], 1), rule, "full", shape=shape)
    b = best_response(1, 1.0, c["z"], rule, "full", shape=shape)          # full profile, own entry ignored
    assert a == b
    with pytest.raises(ValueError):
        best_response(1, 1.0, c["z"], rule, "usage", shape=shape)         # needs nu
    with pytest.raises(ValueError):
        best_response(1, 1.0, c["z"], rule, "auction", shape=shape)
    with pytest.raises(ValueError):
        best_response(1, float(c["ct"].C[1]) * 2, c["z"], rule, "full", shape=shape)
    with pytest.raises(ValueError):
        best_response(1, 1.0, c["z"], rule, "full")                        # shape required
    with pytest.raises(ValueError):
        best_response(1, 1.0, c["z"][:2], rule, "full", shape=shape)


def test_fixed_price_never_freezes_weights_at_a_knife_edge():
    """Integration regression (smoke run, TRAIN 2024-01-14 +21660 s, interval 69; values copied bit-exactly).

    At a threshold bracket end of the full map the own fractional quota equals q - floor_eps (33 - 1e-9 PRB) up to
    the bracket width. Freezing the priority there leaves the frozen own-quota path flat at the floor boundary
    (tenant 1 interior, tenant 2 capped), where rounding noise of a few ulp flips the integer level and the
    threshold engine reported a spurious NonMonotoneRuleError. The fixed-price enumeration must avoid such points
    and still dominate a dense grid."""
    ct = Contract(cls=("video", "urllc", "mmtc"),
                  g=np.array([12.954200522222234, 9.69357283128891, 0.8273947429333298]),
                  M=np.array([77.7252031333334, 36.35089811733341, 16.547894858666595]),
                  C=np.array([129.54200522222234, 60.584830195555675, 33.09578971733319]),
                  pi=np.array([1.0, 3.0, 0.5]), delay_ms=np.array([2002.0, 150.0, 300.0]),
                  mu=np.array([25.908401044444467, 12.116966039111135, 4.136973714666649]))
    mech = Mechanism(B=106, gamma=0.2, eta=2.0, Delta=1.0, integer=True)
    r = np.array([0.3243834000000001, 0.39468924, 0.4635932400000001])
    pm = ProtectedMap([-2.388948639379165, -2.260395365248482, 2.5961956241112922],
                      [0.880271324921079, 0.2674902081872601, 0.3720283360764529], ct.mu)
    rule = QuotaRule(ct, mech, r, pm)
    shape = Shape(beta=(np.array([1.0]), np.array([0.5, 1.5]), np.array([0.25, 1.75])),
                  rho=(np.array([1.0]), np.array([0.5, 0.5]), np.array([0.5, 0.5])))
    theta = np.array([40.858872, 15.761158653153476, 4.384928863617392])
    i = 1
    br = best_response(i, theta[i], np.delete(theta, i), rule, "fixed", shape=shape)
    assert br.exact and br.gain >= 0
    grid = np.linspace(0, ct.C[i], 4096)
    dense = utilities(i, theta[i], np.delete(theta, i), rule, grid, "fixed", shape=shape)
    assert br.U_star >= dense.max() - (br.cert + 1e-9 * ct.pi[i] * ct.C[i])
    for price in ("full", "usage", "none"):                      # ties prefer cell interiors over bracket ends
        b = best_response(i, theta[i], np.delete(theta, i), rule, price, shape=shape, nu=np.full(3, 0.1))
        lo, hi = own_thresholds(i, theta, rule, int(rule(np.r_[theta[:i], ct.C[i], theta[i + 1:]][None])[0, i]))
        assert not np.any((b.z_star == lo) | (b.z_star == hi)), price
