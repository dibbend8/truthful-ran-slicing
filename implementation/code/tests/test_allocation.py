"""Contract allocator: exact agreement with the rational reference, stage semantics, feasibility, validation."""
from __future__ import annotations

import numpy as np
import pytest

from _core_fixtures import random_case, rg, rg_contract, to_float
from slicemech.config import Contract, Mechanism
from slicemech.model.allocation import FLOOR_EPS, QuotaRule, fractional, integer, stage
from slicemech.model.scores import FixedMap, ProtectedMap


@pytest.mark.parametrize("n", [3, 4, 6, 9])
def test_matches_rational_general(n):
    """Gate G1: float allocation equals the exact rational allocation to 1e-9 on 2000 random instances."""
    rng = np.random.default_rng(100 + n)
    worst, seen = 0.0, {"overload": 0, "need": 0, "leftover": 0, "zero": 0}
    for _ in range(2000):
        c = random_case(rng, n)
        x = fractional(c["z"], c["w"], c["r"], c["ct"], c["mech"])
        exact = to_float(rg.exact_allocation(tuple(c["z"]), tuple(c["w"]), tuple(c["r"]),
                                             rg_contract(c["ct"], c["mech"])))
        worst = max(worst, float(np.max(np.abs(x - exact))))
        st = stage(c["z"], c["r"], c["ct"], c["mech"])
        key = "overload" if st["overload"] else "leftover" if st["leftover"] else "need"
        seen[key] += 1
        seen["zero"] += int(np.any(c["z"] == 0))
    assert worst <= 1e-9, worst
    assert min(seen.values()) >= 50, seen          # every stage (and zero reports) well represented


def test_batched_equals_rowwise():
    rng = np.random.default_rng(1)
    c = random_case(rng, 5)
    Z = c["ct"].C * rng.uniform(0, 1, (64, 5))
    W = np.exp(rng.normal(0, 1, (64, 5)))
    X = fractional(Z, W, c["r"], c["ct"], c["mech"])
    for k in range(64):
        np.testing.assert_array_equal(X[k], fractional(Z[k], W[k], c["r"], c["ct"], c["mech"]))
    X3 = fractional(Z.reshape(8, 8, 5), W.reshape(8, 8, 5), c["r"], c["ct"], c["mech"])
    np.testing.assert_array_equal(X3.reshape(64, 5), X)


@pytest.mark.parametrize("n", [3, 7])
def test_feasibility_bounds_and_zero_reports(n):
    rng = np.random.default_rng(7 + n)
    for _ in range(500):
        c = random_case(rng, n, zero_prob=0.3)
        ct, mech = c["ct"], c["mech"]
        st = stage(c["z"], c["r"], ct, mech)
        x = fractional(c["z"], c["w"], c["r"], ct, mech)
        assert np.all(x >= st["L"] - 1e-12) and np.all(x <= st["U"] + 1e-12)
        assert abs(x.sum() - st["total"]) <= 1e-9 * max(1.0, st["total"])
        assert x.sum() <= mech.B + 1e-9
        assert np.all(x[c["z"] == 0] == 0.0)
        q = integer(c["z"], c["w"], c["r"], ct, mech)
        assert q.dtype == np.int64 and q.sum() <= mech.B and np.all(q >= 0)
        np.testing.assert_array_equal(q, np.floor(x + FLOOR_EPS).astype(np.int64))


def _contract(n, g, M, C):
    return Contract(cls=("video",) * n, g=np.asarray(g, float), M=np.asarray(M, float), C=np.asarray(C, float),
                    pi=np.ones(n), delay_ms=np.full(n, 100.0), mu=np.ones(n))


def test_stage_formulas_by_hand():
    # overload: reservations 12 + 8 + 4 = 24 > B = 12 -> proportional, weights irrelevant
    ct = _contract(3, [12, 8, 4], [20, 20, 20], [30, 30, 30])
    mech = Mechanism(B=12, gamma=0.2, eta=2.0, Delta=1.0, integer=True)
    z = np.array([20.0, 10.0, 4.0])
    st = stage(z, np.ones(3), ct, mech)
    assert st["overload"] and not st["leftover"]
    np.testing.assert_allclose(fractional(z, [1, 50, 0.01], np.ones(3), ct, mech), [6.0, 4.0, 2.0])
    # leftover: H = min(2z, M) = (2, 2, 0); sum H = 4 < 12 -> fill on [H, M 1{z>0}] = [(2,2,0), (5,5,0)]
    ct = _contract(3, [0, 0, 0], [5, 5, 5], [10, 10, 10])
    z = np.array([1.0, 1.0, 0.0])
    st = stage(z, np.ones(3), ct, mech)
    assert st["leftover"] and st["total"] == 10.0
    np.testing.assert_allclose(fractional(z, [1, 1, 1], np.ones(3), ct, mech), [5.0, 5.0, 0.0])
    # need stage: g = 0, gamma = 0.5, B = 10, z = (10, 10, 10), M = 30 -> H = 20 each, floors 0.5*10/3
    ct = _contract(3, [0, 0, 0], [30, 30, 30], [10, 10, 10])
    mech = Mechanism(B=10, gamma=0.5, eta=2.0, Delta=1.0, integer=True)
    z = np.full(3, 10.0)
    st = stage(z, np.ones(3), ct, mech)
    np.testing.assert_allclose(st["L"], np.full(3, 5 / 3))
    x = fractional(z, [2.0, 1.0, 1.0], np.ones(3), ct, mech)
    # weighted part: 10 - 5 = 5 split... clip(tau w, 5/3, 20): tau*2 + 2*max(tau, 5/3) = 10 -> tau = 2.5
    np.testing.assert_allclose(x, [5.0, 2.5, 2.5])


def test_model_tex_example():
    """theory/model.tex Example (one allocation): l = (4, 3, 1), x = (7, 7, 6) for equal weights and
    (4, 12.8, 3.2) for w = (1, 4, 1) or its normalisation (1/6, 2/3, 1/6)."""
    ct = Contract(cls=("video", "urllc", "mmtc"), g=np.array([1.5, 0.5, 0.0]), M=np.full(3, 5.0),
                  C=np.full(3, 10.0), pi=np.ones(3), delay_ms=np.full(3, 100.0), mu=np.ones(3))
    mech = Mechanism(B=20, gamma=0.2, eta=2.0, Delta=1.0, integer=True)
    r, z = np.array([0.5, 0.25, 0.25]), np.array([2.0, 2.5, 0.75])
    st = stage(z, r, ct, mech)
    np.testing.assert_allclose(st["b"], [3, 2, 0])
    np.testing.assert_allclose(st["H"], [8, 20, 6])
    np.testing.assert_allclose(st["L"], [4, 3, 1])
    assert not st["overload"] and not st["leftover"] and st["total"] == 20
    np.testing.assert_allclose(fractional(z, [1, 1, 1], r, ct, mech), [7, 7, 6], rtol=1e-14)
    for w in ([1, 4, 1], [1 / 6, 2 / 3, 1 / 6]):
        np.testing.assert_allclose(fractional(z, w, r, ct, mech), [4, 12.8, 3.2], rtol=1e-14)
    np.testing.assert_array_equal(integer(z, [1, 4, 1], r, ct, mech), [4, 12, 3])


def test_two_tenants_supported():
    """n = 2 (the theory's E3 setting) works although the rational reference needs n >= 3."""
    ct = Contract(cls=("video",) * 2, g=np.zeros(2), M=np.full(2, 20.0), C=np.full(2, 20.0), pi=np.ones(2),
                  delay_ms=np.full(2, 100.0), mu=np.full(2, 5.0))
    mech = Mechanism(B=10, gamma=0.0, eta=3.0, Delta=1.0, integer=True)
    for u in (0.5, 1.0, 4.0, 8.0, 20.0):
        z = np.array([u, 10.0])
        w = np.array([5 + u, 15.0])                       # softmax of log1p(z / 5): ratio (5 + u) / 15
        sigma = (5 + u) / (20 + u)
        np.testing.assert_allclose(fractional(z, w, np.ones(2), ct, mech)[0], min(3 * u, 10 * sigma), rtol=1e-13)


def test_fill_is_weighted_maxmin_and_scale_invariant():
    """KKT form: all coordinates strictly between their bounds share one water level x / w."""
    rng = np.random.default_rng(3)
    for _ in range(300):
        c = random_case(rng, 6, load=2.0)
        st = stage(c["z"], c["r"], c["ct"], c["mech"])
        if st["overload"]:
            continue
        x = fractional(c["z"], c["w"], c["r"], c["ct"], c["mech"])
        w = c["w"] / c["w"].max()
        free = (x > st["L"] + 1e-9) & (x < st["U"] - 1e-9)
        movable = st["U"] > st["L"] + 1e-9                  # L = U coordinates are pinned (no level information)
        if free.sum() >= 2:
            lv = x[free] / w[free]
            assert np.ptp(lv) <= 1e-9 * lv.max()
            at_lo = movable & ~free & (x <= st["L"] + 1e-9)     # floored: own level L/w at or above the water level
            at_hi = movable & ~free & (x >= st["U"] - 1e-9)     # capped: own level U/w at or below it
            assert np.all(x[at_lo] / w[at_lo] >= lv.min() * (1 - 1e-9))
            assert np.all(x[at_hi] / w[at_hi] <= lv.max() * (1 + 1e-9))
        np.testing.assert_allclose(fractional(c["z"], 7.3 * c["w"], c["r"], c["ct"], c["mech"]), x, atol=1e-11)


def test_invalid_inputs_raise():
    rng = np.random.default_rng(4)
    c = random_case(rng, 3)
    ct, mech, r, z, w = c["ct"], c["mech"], c["r"], c["z"], c["w"]
    with pytest.raises(ValueError):
        fractional(ct.C * 1.01, w, r, ct, mech)                 # above the ceiling
    with pytest.raises(ValueError):
        fractional(-z - 1, w, r, ct, mech)
    with pytest.raises(ValueError):
        fractional(z, w, np.zeros(3), ct, mech)
    with pytest.raises(ValueError):
        fractional(z, np.zeros(3), r, ct, mech)
    with pytest.raises(ValueError):
        fractional(z[:2], w[:2], r[:2], ct, mech)
    with pytest.raises(ValueError):
        fractional(np.full(3, np.nan), w, r, ct, mech)
    with pytest.raises(ValueError):
        stage(z, r, ct, Mechanism(B=10, gamma=1.5, eta=2.0, Delta=1.0, integer=True))


def test_quota_rule_wrapper():
    rng = np.random.default_rng(5)
    c = random_case(rng, 4)
    ct, mech, r, z = c["ct"], c["mech"], c["r"], c["z"]
    pm = ProtectedMap(rng.uniform(-4, 4, 4), rng.uniform(0, 2, 4), c["mu"])
    rule = QuotaRule(ct, mech, r, pm)
    x = fractional(z, pm.weights(z), r, ct, mech)
    np.testing.assert_array_equal(rule.fractional(z), x)
    np.testing.assert_array_equal(rule(z), np.floor(x + FLOOR_EPS).astype(np.int64))
    assert rule.monotone_by_construction and not rule.is_direct
    fr = rule.frozen(z)
    assert isinstance(fr.policy, FixedMap)
    np.testing.assert_array_equal(fr(z), rule(z))
    frac_rule = QuotaRule(ct, mech, r, pm, integer=False)
    np.testing.assert_array_equal(frac_rule(z), x)
    exact_rule = QuotaRule(ct, mech, r, pm, floor_eps=0.0)
    np.testing.assert_array_equal(exact_rule(z), np.floor(x).astype(np.int64))
    with pytest.raises(ValueError):
        QuotaRule(ct, mech, r[:2], pm)
    with pytest.raises(TypeError):
        QuotaRule(ct, mech, r, object())
