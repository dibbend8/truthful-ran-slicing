"""Envelope prices: exact reference agreement, menu regression, price/premium properties, certificates."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from _core_fixtures import random_case, random_shape, rg, rg_contract, rg_shapes, to_float
from slicemech.config import Contract, Mechanism, Shape
from slicemech.model.allocation import QuotaRule
from slicemech.model.scores import CallableMap, FixedMap, GeneralScoreMap, ProtectedMap, necessity_witness
from slicemech.model.shapes import marginal, stack_shape, value
from slicemech.pricing.envelope import (fixed_priority_price, fixed_priority_prices, full_price, full_prices,
                                        premium, price_nonmonotone, usage_price)
from slicemech.pricing.thresholds import NonMonotoneRuleError, own_path, own_thresholds


EXACT_FLOOR = 1e-12      # floor offset that emulates the exact floor of the rational references


# ------------------------------------------------------------------------------------------ reference agreement
@pytest.mark.parametrize("n", [3, 4, 6, 9])
def test_fixed_map_full_price_matches_quote_static(n):
    """Gate G1: full price of a FixedMap equals rational_general.quote_static to 1e-9 relative."""
    rng = np.random.default_rng(300 + n)
    for _ in range(150):
        c = random_case(rng, n)
        ct, mech, r, z, w = c["ct"], c["mech"], c["r"], c["z"], c["w"]
        shape = random_shape(rng, n)
        ref = rg.quote_static(tuple(z), tuple(w), tuple(r), rg_contract(ct, mech), rg_shapes(shape))
        p_ref = to_float(ref["exact_price"])
        # the reference floors exact rationals; 1e-12 emulates that while absorbing float noise on exact-integer
        # plateaus (e.g. a lone active tenant gets exactly B) that would make floor(x) drop a level
        rule = QuotaRule(ct, mech, r, FixedMap(w), floor_eps=EXACT_FLOOR)
        tight = full_prices(z, rule, ct, shape, mech, r, tol=2.0 ** -50 * ct.C)
        loose = full_prices(z, rule, ct, shape, mech, r)                 # default tol = 2**-40 C
        for i in range(n):
            assert tight[i].quota == ref["integer"][i]
            scale = max(abs(p_ref[i]), 1e-3 * ct.pi[i] * mech.Delta * max(z[i], 1e-9))
            assert abs(tight[i].p - p_ref[i]) <= 1e-9 * scale, (i, tight[i], p_ref[i])
            assert loose[i].p_lo <= p_ref[i] <= loose[i].p_hi, (i, loose[i], p_ref[i])   # certificate holds
            assert loose[i].p_hi - loose[i].p_lo <= 2 * loose[i].err + 4e-12 * ct.pi[i] * ct.C[i]
        single = full_price(n - 1, z, rule, ct, shape, mech, r, tol=2.0 ** -50 * ct.C)
        assert single == tight[n - 1]


# ----------------------------------------------------------------------------------------------- menu regression
_LOW = 3 / 20
_HIGH, _MID = 3 - 2 * _LOW, (3 - _LOW) / 2
MENU = np.array([[1, 1, 1], [_LOW, _LOW, _HIGH], [_LOW, _HIGH, _LOW], [_HIGH, _LOW, _LOW],
                 [_LOW, _MID, _MID], [_MID, _LOW, _MID], [_MID, _MID, _LOW]])


def menu_selector(crossing: float):
    """premium_witnesses.prepared(): scores intercept[a] + MENU[a][0] * z_0 (influence (1, 0, 0), alpha = 1);
    intercepts -1000 except action 0 (0) and action 3 (-(MENU[3][0] - 1) * crossing); lowest index wins ties."""
    intercept = np.full(7, -1000.0)
    intercept[0], intercept[3] = 0.0, -(MENU[3, 0] - 1) * crossing

    def fn(z):
        scores = intercept + z[..., :1] * MENU[:, 0]
        return MENU[np.argmax(scores, axis=-1)]
    return fn


def _menu_game(crossing=15.0, B=12, gbr=0.0, eta=3.0, mbr=(40, 40, 40), ceiling=(20, 40, 40), atoms=(0.0, 2.0),
               floor_eps=EXACT_FLOOR, monotone=True):
    ct = Contract(cls=("video",) * 3, g=np.full(3, gbr), M=np.array(mbr, float), C=np.array(ceiling, float),
                  pi=np.ones(3), delay_ms=np.full(3, 100.0), mu=np.ones(3))
    mech = Mechanism(B=B, gamma=0.0, eta=eta, Delta=1.0, integer=True)
    a = np.array(atoms, float)
    shape = Shape(beta=(a / a.mean(),) * 3, rho=(np.full(a.size, 1 / a.size),) * 3)
    rule = QuotaRule(ct, mech, np.ones(3), CallableMap(menu_selector(crossing), monotone=monotone),
                     floor_eps=floor_eps)
    return ct, mech, shape, rule


def test_menu_regression_premium_witness():
    """premium_witnesses.py (B = 12): full price 7/2 vs fixed 1/2 at report 20; both 1/2 at report 10."""
    ct, mech, shape, rule = _menu_game()
    r = np.ones(3)
    for u, q, p_full, p_fix in ((10.0, 4, 0.5, 0.5), (20.0, 10, 3.5, 0.5)):
        z = np.array([u, 40.0, 40.0])
        full = full_price(0, z, rule, ct, shape, mech, r)
        fixed = fixed_priority_price(0, z, rule, ct, shape, mech, r)
        assert full.quota == q and fixed.quota == q
        assert full.p == pytest.approx(p_full, rel=1e-9) and fixed.p == pytest.approx(p_fix, rel=1e-9)
        assert full.p_lo <= p_full <= full.p_hi and fixed.p_lo <= p_fix <= fixed.p_hi
        assert premium(0, z, rule, ct, shape, mech, r).T == pytest.approx(p_full - p_fix, abs=1e-9)
    # the same through a map that does not declare monotonicity: refused unless the caller opts in explicitly
    # (round-1 review, finding 7; the grid check alone cannot certify a map that is not monotone by construction)
    _, _, _, undeclared = _menu_game(monotone=False)
    with pytest.raises(NonMonotoneRuleError, match="not monotone by construction"):
        full_price(0, np.array([20.0, 40, 40]), undeclared, ct, shape, mech, r)
    assert full_price(0, np.array([20.0, 40, 40]), undeclared, ct, shape, mech, r,
                      assume_monotone=True).p == pytest.approx(3.5, rel=1e-9)
    # deployed floor offset (1e-9) moves the price by less than 1e-8
    _, _, _, deployed = _menu_game(floor_eps=1e-9)
    assert full_price(0, np.array([20.0, 40, 40]), deployed, ct, shape, mech, r).p == pytest.approx(3.5, abs=1e-8)


def test_menu_regression_outage_premium():
    """premium_witnesses.py outage case: B = 4, g = 10, eta = 1, atoms (1/2, 3/2): full 1/2, fixed 0 at (4, 1, 1)."""
    ct, mech, shape, rule = _menu_game(crossing=3.0, B=4, gbr=10.0, eta=1.0, mbr=(10, 10, 10),
                                       ceiling=(10, 10, 10), atoms=(0.5, 1.5))
    z = np.array([4.0, 1.0, 1.0])
    pm = premium(0, z, rule, ct, shape, mech, np.ones(3))
    assert pm.full.quota == 2
    assert pm.full.p == pytest.approx(0.5, rel=1e-9) and pm.fixed.p == pytest.approx(0.0, abs=1e-9)
    assert pm.T == pytest.approx(0.5, rel=1e-9)


def test_e3_closed_form_premium():
    """theory_design.md E3 (n = 2, divisible quotas): p_fix = 0 and p = T = [5 sigma(z) - u_c]^+, with
    sigma(u) = (5 + u) / (20 + u) and u_c = (5 sqrt(13) - 15) / 2; T(8) = 0.807 and the fixed-price gain of
    reporting the ceiling from theta = 8 equals T(20) - T(8) (Prop. 3(iv) bound attained)."""
    from slicemech.pricing.response import utilities
    ct = Contract(cls=("video",) * 2, g=np.zeros(2), M=np.full(2, 20.0), C=np.full(2, 20.0), pi=np.ones(2),
                  delay_ms=np.full(2, 100.0), mu=np.full(2, 5.0))
    mech = Mechanism(B=10, gamma=0.0, eta=3.0, Delta=1.0, integer=False)
    shape = Shape(beta=(np.array([0.0, 2.0]),) * 2, rho=(np.array([0.5, 0.5]),) * 2)
    rule = QuotaRule(ct, mech, np.ones(2), ProtectedMap([0.0, 0.0], [1.0, 1.0], ct.mu), quantum=2.0 ** -10)
    u_c = (5 * np.sqrt(13) - 15) / 2
    T = lambda u: max(5 * (5 + u) / (20 + u) - u_c, 0.0)                     # noqa: E731
    for u in (1.0, 2.0, 8.0, 20.0):
        z = np.array([u, 10.0])
        pm = premium(0, z, rule, ct, shape, mech, np.ones(2))
        assert pm.full.p_lo <= T(u) <= pm.full.p_hi and pm.full.p_hi - pm.full.p_lo <= 5e-3
        assert pm.fixed.p_lo <= 0.0 <= pm.fixed.p_hi and pm.fixed.p_hi - pm.fixed.p_lo <= 5e-3
        assert pm.T == pytest.approx(T(u), abs=5e-3)
    assert T(8.0) == pytest.approx(0.8076, abs=1e-4)                        # design text: 0.807
    U_fix = utilities(0, 8.0, [10.0], rule, [8.0, 20.0], "fixed", shape=shape)
    U_full = utilities(0, 8.0, [10.0], rule, [8.0, 20.0], "full", shape=shape)
    assert U_fix[1] - U_fix[0] == pytest.approx(T(20.0) - T(8.0), abs=1e-2)
    assert U_full[1] - U_full[0] <= 1e-2


# ------------------------------------------------------------------------------------------- price properties
def _protected_case(rng, n, **kw):
    c = random_case(rng, n, zero_prob=0.0, **kw)
    pm = ProtectedMap(rng.uniform(-4, 4, n), rng.uniform(0, 2, n), c["mu"])
    shape = random_shape(rng, n, uniform=bool(rng.random() < 0.5))
    return c, QuotaRule(c["ct"], c["mech"], c["r"], pm), shape


@pytest.mark.parametrize("n", [3, 5, 8])
def test_price_bounds_zero_report_and_fixed_below_full(n):
    rng = np.random.default_rng(400 + n)
    for _ in range(60):
        c, rule, shape = _protected_case(rng, n)
        ct, mech, r, z = c["ct"], c["mech"], c["r"], c["z"]
        full = full_prices(z, rule, ct, shape, mech, r)
        fixed = fixed_priority_prices(z, rule, ct, shape, mech, r)
        for i in range(n):
            f, x = full[i], fixed[i]
            assert f.p_lo <= f.p <= f.p_hi and x.p_lo <= x.p <= x.p_hi
            assert -f.err - 1e-12 <= f.p <= f.V + f.err + 1e-12            # 0 <= p <= V
            assert -x.err - 1e-12 <= x.p <= f.p + f.err + x.err + 1e-12     # 0 <= p_fix <= p (T >= 0)
            assert f.V <= ct.pi[i] * mech.Delta * z[i] + 1e-12
        z0 = z.copy()
        z0[0] = 0.0
        p0 = full_price(0, z0, rule, ct, shape, mech, r)
        assert p0.p == 0.0 and p0.V == 0.0 and p0.quota == 0.0
        assert fixed_priority_price(0, z0, rule, ct, shape, mech, r).p == 0.0


def test_price_constant_inside_own_quota_cell():
    rng = np.random.default_rng(11)
    checked = 0
    while checked < 40:
        c, rule, shape = _protected_case(rng, 4, load=2.0)
        ct, mech, r, z = c["ct"], c["mech"], c["r"], c["z"]
        i = int(rng.integers(4))
        q_top = int(own_path(i, z, rule, [ct.C[i]])[0])
        if q_top < 3:
            continue
        lo, hi = own_thresholds(i, z, rule, q_top)
        left, right = hi[:-1], lo[1:]                     # cells of levels 1 .. q_top-1 (certain interiors)
        k = int(np.argmax(right - left))
        if right[k] - left[k] <= 1e-6 * ct.C[i]:
            continue
        a, b = left[k] + 0.25 * (right[k] - left[k]), left[k] + 0.75 * (right[k] - left[k])
        pa = full_price(i, np.where(np.arange(4) == i, a, z), rule, ct, shape, mech, r)
        pb = full_price(i, np.where(np.arange(4) == i, b, z), rule, ct, shape, mech, r)
        assert pa.quota == pb.quota == k + 1
        assert abs(pa.p - pb.p) <= pa.err + pb.err + 1e-12
        checked += 1


def test_premium_nonnegative_nondecreasing_and_gain_identity():
    rng = np.random.default_rng(12)
    for _ in range(25):
        c, rule, shape = _protected_case(rng, 4, load=float(rng.uniform(1.2, 3.0)))
        ct, mech, r, z = c["ct"], c["mech"], c["r"], c["z"]
        i = int(rng.integers(4))
        grid = np.linspace(0, ct.C[i], 21)
        prof = lambda u: np.where(np.arange(4) == i, u, z)          # noqa: E731
        prem = [premium(i, prof(u), rule, ct, shape, mech, r) for u in grid]
        T = np.array([p.T for p in prem])
        e = np.array([p.full.err + p.fixed.err for p in prem]) + 1e-12
        assert np.all(T >= -e)                                         # T >= 0
        assert np.all(np.diff(T) >= -(e[1:] + e[:-1]))                 # T nondecreasing in the own report
        beta, rho = stack_shape(shape)
        args = (r[i], ct.pi[i], beta[i], rho[i], mech.Delta)
        for _ in range(4):
            theta, zz = rng.uniform(0, ct.C[i], 2)
            Pt, Pz = premium(i, prof(theta), rule, ct, shape, mech, r), premium(i, prof(zz), rule, ct, shape, mech, r)
            Vz, Vt = value(Pz.full.quota, theta, *args), value(Pt.full.quota, theta, *args)
            g_full = (Vz - Pz.full.p) - (Vt - Pt.full.p)
            g_fix = (Vz - Pz.fixed.p) - (Vt - Pt.fixed.p)
            assert g_fix == pytest.approx(g_full + Pz.T - Pt.T, abs=1e-9)          # gain identity
            cert = 2 * (Pz.full.err + Pt.full.err + Pz.fixed.err + Pt.fixed.err) + 1e-11
            assert g_full <= cert                                                  # full price is truthful
            if zz <= theta:
                assert g_fix <= cert                                               # under-reporting never pays
            else:
                assert g_fix <= Pz.T - Pt.T + cert                                 # gain <= premium increase


def test_fixed_map_has_zero_premium():
    rng = np.random.default_rng(13)
    for _ in range(20):
        c = random_case(rng, 5, zero_prob=0.0)
        shape = random_shape(rng, 5)
        rule = QuotaRule(c["ct"], c["mech"], c["r"], FixedMap(c["w"]))
        pm = premium(2, c["z"], rule, c["ct"], shape, c["mech"], c["r"])
        assert pm.T == pytest.approx(0.0, abs=1e-12)


def test_full_price_against_fine_integration():
    """Continuous weights: the closed-form envelope agrees with a 2^20-point midpoint integral of D(Q(u), u)."""
    rng = np.random.default_rng(14)
    for _ in range(6):
        c, rule, shape = _protected_case(rng, 3, load=2.0)
        ct, mech, r, z = c["ct"], c["mech"], c["r"], c["z"]
        i = 0
        pr = full_price(i, z, rule, ct, shape, mech, r)
        N = 2 ** 20
        u = (np.arange(N) + 0.5) * z[i] / N
        q = own_path(i, z, rule, u)
        beta, rho = stack_shape(shape)
        D = marginal(q.astype(float), u, r[i], ct.pi[i], beta[i], rho[i], mech.Delta)
        p_num = pr.V - D.sum() * z[i] / N
        jumps = (pr.quota + 1) * (np.count_nonzero(beta[i]) + 1)
        assert abs(pr.p - p_num) <= jumps * ct.pi[i] * mech.Delta * z[i] / N + pr.err


def test_fractional_rule_bracket():
    rng = np.random.default_rng(15)
    c, _, shape = _protected_case(rng, 3, load=2.0)
    ct, mech, r, z = c["ct"], c["mech"], c["r"], c["z"]
    pm = ProtectedMap(rng.uniform(-1, 1, 3), rng.uniform(0, 2, 3), c["mu"])
    N = 2 ** 18
    u = (np.arange(N) + 0.5) * z[0] / N
    exact_rule = QuotaRule(ct, mech, r, pm, integer=False)
    Z = np.repeat(z[None, :], N, axis=0)
    Z[:, 0] = u
    x = exact_rule.fractional(Z)[:, 0]
    beta, rho = stack_shape(shape)
    D = marginal(x, u, r[0], ct.pi[0], beta[0], rho[0], mech.Delta)
    widths = []
    for quantum in (2.0 ** -3, 2.0 ** -6):
        rule = QuotaRule(ct, mech, r, pm, integer=False, quantum=quantum)
        pr = full_price(0, z, rule, ct, shape, mech, r)
        p_num = pr.V - D.sum() * z[0] / N
        slack = 8 * ct.pi[0] * mech.Delta * z[0] / N * (np.count_nonzero(beta[0]) + 1)
        assert pr.p_lo - slack <= p_num <= pr.p_hi + slack
        widths.append(pr.p_hi - pr.p_lo)
    assert widths[1] < widths[0]


# ----------------------------------------------------------------------------------------- non-monotone, usage
def test_price_nonmonotone():
    rng = np.random.default_rng(16)
    c, rule, shape = _protected_case(rng, 4, load=2.0)
    ct, mech, r, z = c["ct"], c["mech"], c["r"], c["z"]
    for i in range(4):
        a, b = full_price(i, z, rule, ct, shape, mech, r), price_nonmonotone(i, z, rule, ct, shape, mech, r)
        assert np.isnan(b.p_lo) and b.p == pytest.approx(a.p, abs=2 * a.err + 1e-9)
    pm = GeneralScoreMap([0.0, 0.3, -0.2], [[1, 0, 0], [2, 1, 0], [0, 0, 1]], np.full(3, 5.0))
    wit = necessity_witness(pm, 0, 1, np.array([0.0, 20.0, 7.0]), 5.0, 30.0)
    bad = QuotaRule(wit["ct"], wit["mech"], wit["r"], pm)
    unit = Shape(beta=(np.ones(1),) * 3, rho=(np.ones(1),) * 3)
    with pytest.raises(NonMonotoneRuleError):
        full_price(0, wit["z_v"], bad, wit["ct"], unit, wit["mech"], wit["r"])
    res = price_nonmonotone(0, wit["z_v"], bad, wit["ct"], unit, wit["mech"], wit["r"])
    assert np.isfinite(res.p) and res.quota == bad(wit["z_v"])[0]


def test_usage_price_and_validation():
    x = np.array([[3, 0, 5], [1, 2, 0]])
    np.testing.assert_allclose(usage_price(x, [0.5, 1.0, 0.2]), [[1.5, 0, 1.0], [0.5, 2.0, 0]])
    with pytest.raises(ValueError):
        usage_price(x, [-1, 0, 0])
    rng = np.random.default_rng(17)
    c, rule, shape = _protected_case(rng, 3)
    ct, mech, r, z = c["ct"], c["mech"], c["r"], c["z"]
    with pytest.raises(ValueError):
        full_price(0, z, rule, ct, shape, mech, r * 1.1)                       # value must use the rule's r
    with pytest.raises(ValueError):
        full_price(0, z, rule, ct, shape, Mechanism(B=mech.B + 1, gamma=mech.gamma, eta=mech.eta,
                                                    Delta=mech.Delta, integer=True), r)
    with pytest.raises(ValueError):
        full_price(0, z, rule, ct, Shape(beta=(np.ones(1),) * 2, rho=(np.ones(1),) * 2), mech, r)
    with pytest.raises(ValueError):
        own_thresholds(0, z, rule, 10 ** 6)                                    # level never reached


def test_thresholds_bracket_semantics():
    rng = np.random.default_rng(18)
    for _ in range(20):
        c, rule, _ = _protected_case(rng, 5, load=2.0)
        z, C = c["z"], c["ct"].C
        i = int(rng.integers(5))
        q_top = int(own_path(i, z, rule, [C[i]])[0])
        if q_top == 0:
            continue
        lo, hi = own_thresholds(i, z, rule, q_top)
        assert lo.size == q_top and np.all(hi - lo <= 2.0 ** -40 * C[i]) and np.all(np.diff(hi) >= 0)
        qlo, qhi = own_path(i, z, rule, lo), own_path(i, z, rule, hi)
        levels = np.arange(1, q_top + 1)
        assert np.all(qhi >= levels) and np.all((qlo < levels) | (lo == 0))
