"""Priority maps: ratio condition, monotonicity fuzz (Gate G1) and the Theorem 1 necessity witness."""
from __future__ import annotations

import numpy as np
import pytest

from _core_fixtures import random_case, random_shape, run_parallel
from slicemech.model.allocation import FLOOR_EPS, QuotaRule, fractional
from slicemech.model.scores import (CallableMap, FixedMap, GeneralScoreMap, ProtectedMap, necessity_witness,
                                    ratio_violation, softmax)
from slicemech.pricing.envelope import full_price
from slicemech.pricing.thresholds import NonMonotoneRuleError, max_drop, own_path

N_FUZZ = 10_000          # instances per n (Gate G1)
GRID = 2048              # own-report grid points


def _drops(x: np.ndarray) -> tuple[float, int]:
    frac = float(np.max(np.maximum.accumulate(x) - x))
    q = np.floor(x + FLOOR_EPS)
    return frac, int(np.max(np.maximum.accumulate(q) - q))


def _fuzz_chunk(args) -> tuple[float, int, int]:
    """Worker: ``count`` random protected-map instances of size n; returns (worst fractional drop, worst integer
    drop, number of instances whose own path is not constant)."""
    n, seed, count, kind = args
    rng = np.random.default_rng(seed)
    worst_f, worst_i, moving = 0.0, 0, 0
    u01 = np.linspace(0.0, 1.0, GRID)
    for _ in range(count):
        c = random_case(rng, n)
        ct, mech, r, z = c["ct"], c["mech"], c["r"], c["z"]
        if kind == "protected":
            kappa = rng.uniform(0, 2, n) * (rng.random(n) > 0.1)
            pm = ProtectedMap(rng.uniform(-4, 4, n), kappa, c["mu"])
        else:                                        # general scores satisfying Lam_ii >= Lam_ji (Corollary 1)
            Lam = rng.uniform(-2, 2, (n, n))
            d = np.diag(Lam).copy()
            Lam = np.minimum(Lam, d[None, :])        # every column at most its diagonal entry
            pm = GeneralScoreMap(rng.uniform(-4, 4, n), Lam, c["mu"])
        i = int(rng.integers(n))
        Z = np.repeat(z[None, :], GRID, axis=0)
        Z[:, i] = ct.C[i] * u01
        x = fractional(Z, pm.weights(Z), r, ct, mech)[:, i]
        f, q = _drops(x)
        worst_f, worst_i = max(worst_f, f), max(worst_i, q)
        moving += int(np.ptp(x) > 1e-6)
    return worst_f, worst_i, moving


def _run_parallel(tasks):
    return run_parallel(_fuzz_chunk, tasks)


def test_monotonicity_fuzz_protected_all_n():
    """Gate G1: 10^4 random instances for each n in 3..12; fractional and integer own paths nondecreasing."""
    chunk = 500
    tasks = [(n, 10_000 * n + k, chunk, "protected") for n in range(3, 13) for k in range(N_FUZZ // chunk)]
    res = _run_parallel(tasks)
    for n in range(3, 13):
        mine = [r for t, r in zip(tasks, res) if t[0] == n]
        worst_f = max(r[0] for r in mine)
        worst_i = max(r[1] for r in mine)
        moving = sum(r[2] for r in mine)
        assert worst_f <= 1e-9, (n, worst_f)
        assert worst_i == 0, (n, worst_i)
        assert moving >= N_FUZZ // 4, (n, moving)   # the fuzz exercises non-trivial own paths


def test_monotonicity_fuzz_general_safe_scores():
    tasks = [(n, 77_000 + 10 * n + k, 250, "general") for n in (3, 5, 8) for k in range(4)]
    for worst_f, worst_i, _ in _run_parallel(tasks):
        assert worst_f <= 1e-9 and worst_i == 0


def test_maps_basic():
    rng = np.random.default_rng(0)
    mu = np.array([1.0, 2.0, 4.0])
    z = rng.uniform(0, 5, (10, 3))
    pm = ProtectedMap([0.5, -1.0, 2.0], [0.0, 1.0, 2.0], mu)
    w = pm.weights(z)
    np.testing.assert_allclose(w.sum(-1), 1.0)
    y = pm.a + pm.kappa * np.log1p(z / mu)
    np.testing.assert_allclose(w, softmax(y))
    with pytest.raises(ValueError):
        ProtectedMap([0, 0, 0], [0.1, -0.1, 0], mu)
    with pytest.raises(ValueError):
        pm.weights(-z)
    fm = FixedMap([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(fm.weights(z), np.broadcast_to([1.0, 2.0, 3.0], z.shape))
    with pytest.raises(ValueError):
        FixedMap([1.0, 0.0, 1.0])
    gm = GeneralScoreMap(pm.a, np.diag(pm.kappa), mu)
    np.testing.assert_allclose(gm.weights(z), w, rtol=1e-14)
    assert gm.satisfies_ratio_condition() and gm.monotone_by_construction
    cm = CallableMap(lambda q: pm.weights(q))
    np.testing.assert_array_equal(cm.weights(z), w)
    assert not cm.monotone_by_construction
    with pytest.raises(ValueError):
        CallableMap(lambda q: q[..., :2] + 1).weights(z)
    with pytest.raises(ValueError):
        CallableMap(lambda q: q - 100).weights(z)


def test_callable_map_torch():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 8, dtype=torch.float64), torch.nn.Tanh(),
                              torch.nn.Linear(8, 3, dtype=torch.float64))
    cm = CallableMap(lambda t: torch.softmax(net(t), dim=-1), torch_input=True, n=3)
    w = cm.weights(np.ones((4, 3)))
    assert w.shape == (4, 3) and np.all(w > 0)


def test_ratio_condition_coefficient_rule():
    mu = np.ones(3)
    # Prop. 1(a): column (1, 2) of tenant 0 is coordinatewise monotone but violates Lam_00 >= Lam_10
    bad = GeneralScoreMap(np.zeros(3), [[1, 0, 0], [2, 1, 0], [0, 0, 1]], mu)
    assert not bad.satisfies_ratio_condition()
    assert bad.violations() == [(0, 1, 1.0)]
    # ... and column (-1, -2) is not monotone but safe
    good = GeneralScoreMap(np.zeros(3), [[-1, 0, 0], [-2, 1, 0], [-3, 0, 1]], mu)
    assert good.satisfies_ratio_condition()


def _check_witness(pm, i, z, u_max):
    found = ratio_violation(pm, i, z, u_max)
    assert found is not None
    j, u, v = found
    wit = necessity_witness(pm, i, j, z, u, v)
    rule = QuotaRule(wit["ct"], wit["mech"], wit["r"], pm)
    xu, xv = rule.fractional(wit["z_u"])[i], rule.fractional(wit["z_v"])[i]
    assert xu == pytest.approx(wit["B"] * wit["sigma_u"], rel=1e-12)
    assert xv == pytest.approx(wit["B"] * wit["sigma_v"], rel=1e-12)
    assert rule(wit["z_u"])[i] > rule(wit["z_v"])[i]          # integer own-quota decrease
    # detected by the generic grid check on the own-report path
    path = own_path(i, wit["z_v"], rule, np.linspace(0, wit["ct"].C[i], GRID))
    drop, _ = max_drop(path)
    assert drop >= 1
    with pytest.raises(NonMonotoneRuleError):
        shape = random_shape(np.random.default_rng(0), len(z))
        full_price(i, wit["z_v"], rule, wit["ct"], shape, wit["mech"], wit["r"], n_grid=GRID)
    return wit


def test_necessity_witness_detects_decrease():
    """Gate G1: a map violating Lam_ii >= Lam_ji yields a detected own-quota decrease under the witness contract."""
    mu = np.full(3, 5.0)
    pm = GeneralScoreMap([0.0, 0.3, -0.2], [[1, 0, 0], [2, 1, 0], [0, 0, 1]], mu)
    z = np.array([0.0, 20.0, 7.0])
    wit = _check_witness(pm, 0, z, 40.0)
    assert wit["ct"].M[2] == 0 and wit["mech"].gamma == 0 and np.all(wit["ct"].g == 0)


def test_necessity_witness_random_violations():
    rng = np.random.default_rng(5)
    hits = 0
    for _ in range(200):
        n = int(rng.integers(3, 8))
        Lam = rng.uniform(-1, 1, (n, n))
        i, j = rng.choice(n, 2, replace=False)
        Lam[j, i] = Lam[i, i] + rng.uniform(0.3, 2.0)        # violation in column i
        mu = rng.uniform(0.5, 3, n)
        pm = GeneralScoreMap(rng.uniform(-2, 2, n), Lam, mu)
        z = rng.uniform(0.5, 6, n)
        _check_witness(pm, int(i), z, 10.0)
        hits += 1
    assert hits == 200
