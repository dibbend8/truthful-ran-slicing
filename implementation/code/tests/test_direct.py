"""Direct declared-welfare rule: optimality (LP cross-check), stage box, tie rule, monotonicity fuzz, pricing."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from _core_fixtures import random_case, random_shape, run_parallel
from slicemech.config import Contract, Mechanism, Shape
from slicemech.model.allocation import FLOOR_EPS, QuotaRule, stage
from slicemech.model.direct import DirectRule, direct_fractional, direct_integer
from slicemech.model.shapes import stack_shape, value
from slicemech.pricing.envelope import fixed_priority_price, full_price
from slicemech.pricing.response import best_response


def _lp_welfare(z, r, ct, shape, mech):
    """Max sum_i V_i(x_i, z_i) over the stage box by linear programming (epigraph of the concave V)."""
    st = stage(z, r, ct, mech)
    n = ct.n
    beta, rho = stack_shape(shape)
    K = beta.shape[1]
    nv = n + n * K                                   # x_i, then t_il (served rate per atom)
    c = np.zeros(nv)
    A, b = [], []
    for i in range(n):
        for l in range(K):
            k = n + i * K + l
            c[k] = -ct.pi[i] * mech.Delta * rho[i, l]
            row = np.zeros(nv)
            row[k], row[i] = 1.0, -r[i]              # t <= r x
            A.append(row)
            b.append(0.0)
    bounds = [(st["L"][i], st["U"][i]) for i in range(n)]
    bounds += [(0.0, beta[i, l] * z[i]) for i in range(n) for l in range(K)]   # t <= beta z
    Aeq = np.zeros((1, nv))
    Aeq[0, :n] = 1.0
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=[st["total"]], bounds=bounds,
                  method="highs")
    assert res.status == 0, res.message
    return -res.fun


def _welfare(x, z, r, ct, shape, mech):
    beta, rho = stack_shape(shape)
    return float(np.sum(value(x, z, r, ct.pi, beta, rho, mech.Delta)))


@pytest.mark.parametrize("n", [3, 5, 8])
def test_direct_is_welfare_optimal_in_stage_box(n):
    rng = np.random.default_rng(20 + n)
    for _ in range(150):
        c = random_case(rng, n)
        shape = random_shape(rng, n, uniform=bool(rng.random() < 0.5))
        ct, mech, r, z = c["ct"], c["mech"], c["r"], c["z"]
        x = direct_fractional(z, r, ct, shape, mech)
        st = stage(z, r, ct, mech)
        assert np.all(x >= st["L"] - 1e-12) and np.all(x <= st["U"] + 1e-12)
        assert abs(x.sum() - st["total"]) <= 1e-9 * max(1.0, st["total"])
        W = _welfare(x, z, r, ct, shape, mech)
        if st["overload"]:
            np.testing.assert_allclose(x, st["L"])
            continue
        assert W >= _lp_welfare(z, r, ct, shape, mech) - 1e-7 * max(1.0, W)


def test_tie_rule_is_lexicographic():
    ct = Contract(cls=("video",) * 3, g=np.zeros(3), M=np.array([20.0, 20.0, 0.0]), C=np.full(3, 20.0),
                  pi=np.ones(3), delay_ms=np.full(3, 100.0), mu=np.ones(3))
    mech = Mechanism(B=10, gamma=0.0, eta=2.0, Delta=1.0, integer=True)
    shape = Shape(beta=(np.array([1.0]),) * 3, rho=(np.array([1.0]),) * 3)
    x = direct_fractional(np.array([10.0, 10.0, 0.0]), np.ones(3), ct, shape, mech)
    np.testing.assert_allclose(x, [10.0, 0.0, 0.0])
    q = direct_integer(np.array([10.0, 10.0, 0.0]), np.ones(3), ct, shape, mech)
    np.testing.assert_array_equal(q, [10, 0, 0])


def _direct_fuzz(args):
    n, seed, count = args
    rng = np.random.default_rng(seed)
    worst_f, worst_i = 0.0, 0
    u01 = np.linspace(0, 1, 2048)
    for _ in range(count):
        c = random_case(rng, n)
        shape = random_shape(rng, n, max_atoms=8, uniform=bool(rng.random() < 0.5))
        rule = DirectRule(c["ct"], c["mech"], c["r"], shape)
        i = int(rng.integers(n))
        Z = np.repeat(c["z"][None, :], u01.size, axis=0)
        Z[:, i] = c["ct"].C[i] * u01
        x = rule.fractional(Z)[:, i]
        q = np.floor(x + FLOOR_EPS)
        worst_f = max(worst_f, float(np.max(np.maximum.accumulate(x) - x)))
        worst_i = max(worst_i, int(np.max(np.maximum.accumulate(q) - q)))
    return worst_f, worst_i


def test_direct_monotone_fuzz():
    """Prop. 6: the direct rule's own quota is nondecreasing (fractional and integer), n = 3..9."""
    tasks = [(n, 31_000 + 100 * n + k, 250) for n in range(3, 10) for k in range(8)]
    res = run_parallel(_direct_fuzz, tasks)
    assert max(r[0] for r in res) <= 1e-9
    assert max(r[1] for r in res) == 0


def test_direct_rule_pricing_and_response():
    rng = np.random.default_rng(9)
    for _ in range(20):
        c = random_case(rng, 4, load=2.0, zero_prob=0.0)
        shape = random_shape(rng, 4)
        d = DirectRule(c["ct"], c["mech"], c["r"], shape)
        rule = QuotaRule(c["ct"], c["mech"], c["r"], d)
        np.testing.assert_array_equal(rule(c["z"]), d.quotas(c["z"]))
        assert rule.monotone_by_construction and rule.is_direct
        pr = full_price(0, c["z"], d, c["ct"], shape, c["mech"], c["r"])       # direct rule accepted as is
        assert -pr.err - 1e-12 <= pr.p <= pr.V + pr.err + 1e-12
        br = best_response(0, c["z"][0], c["z"][1:], d, "full", shape=shape)
        assert br.exact and br.gain <= br.cert + 1e-12
    with pytest.raises(TypeError):
        fixed_priority_price(0, c["z"], rule, c["ct"], shape, c["mech"], c["r"])
    with pytest.raises(ValueError):
        best_response(0, c["z"][0], c["z"][1:], rule, "fixed", shape=shape)
