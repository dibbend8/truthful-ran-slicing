"""Declared value V, marginal D and closed-form cell integrals; shape fitting."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from _core_fixtures import rg
from slicemech.config import Shape
from slicemech.model.shapes import cell_integral, fit_shape, marginal, stack_shape, value


def _atoms(rng, K):
    a = rng.uniform(0.05, 3, K)
    if K > 1 and rng.random() < 0.3:
        a[0] = 0.0                                        # a zero atom (idle state) when another atom carries mass
    rho = np.full(K, 1.0 / K)
    return a / np.sum(rho * a), rho


def test_value_matches_reference_and_definition():
    rng = np.random.default_rng(0)
    for _ in range(300):
        K = int(rng.integers(1, 9))
        beta, rho = _atoms(rng, K)
        x, theta, r, pi = rng.uniform(0, 30), rng.uniform(0, 20), rng.uniform(0.1, 1), rng.uniform(0.2, 3)
        v = value(x, theta, r, pi, beta, rho, 1.0)
        direct = pi * sum(p * min(b * theta, r * x) for b, p in zip(beta, rho))
        assert v == pytest.approx(direct, rel=1e-13, abs=1e-14)
        ref = rg.ExactShape(tuple(beta)).value(Fraction(x), Fraction(theta), Fraction(r), Fraction(pi))
        assert v == pytest.approx(float(ref), rel=1e-12, abs=1e-13)
        assert value(x, theta, r, pi, beta, rho, 2.5) == pytest.approx(2.5 * v)


def test_marginal_is_inclusive_and_bounded():
    beta, rho = np.array([0.5, 1.5]), np.array([0.5, 0.5])
    # kink of atom 1.5 at u = r x / 1.5 = 2 for r = 1, x = 3: inclusive -> both atoms count
    assert marginal(3.0, 2.0, 1.0, 1.0, beta, rho, 1.0) == pytest.approx(1.0)
    assert marginal(3.0, 2.0 + 1e-12, 1.0, 1.0, beta, rho, 1.0) == pytest.approx(0.25)
    assert marginal(3.0, 6.0, 1.0, 1.0, beta, rho, 1.0) == pytest.approx(0.25)
    assert marginal(3.0, 6.0 + 1e-9, 1.0, 1.0, beta, rho, 1.0) == 0.0
    u = np.linspace(0, 10, 101)
    d = marginal(3.0, u, 1.0, 2.0, beta, rho, 1.0)
    assert np.all(d >= 0) and np.all(d <= 2.0) and np.all(np.diff(d) <= 0)


def test_cell_integral_against_reference_quadrature_and_identity():
    rng = np.random.default_rng(1)
    for _ in range(300):
        K = int(rng.integers(1, 9))
        beta, rho = _atoms(rng, K)
        q, r, pi = float(rng.integers(0, 40)), rng.uniform(0.1, 1), rng.uniform(0.2, 3)
        a, b = np.sort(rng.uniform(0, 25, 2))
        if rng.random() < 0.2:
            a = 0.0
        I = cell_integral(q, a, b, r, pi, beta, rho, 1.0)
        ref = rg.ExactShape(tuple(beta)).integral(Fraction(q), Fraction(a), Fraction(b), Fraction(r), Fraction(pi))
        assert I == pytest.approx(float(ref), rel=1e-12, abs=1e-12)
        # additivity and the fixed-quota identity V(x, z) = int_0^z D(x, u) du
        m = 0.5 * (a + b)
        assert I == pytest.approx(cell_integral(q, a, m, r, pi, beta, rho, 1.0)
                                  + cell_integral(q, m, b, r, pi, beta, rho, 1.0), rel=1e-12, abs=1e-12)
        assert cell_integral(q, 0.0, b, r, pi, beta, rho, 1.0) == pytest.approx(
            value(q, b, r, pi, beta, rho, 1.0), rel=1e-12, abs=1e-12)
    # fine midpoint quadrature of the indicator form
    beta, rho = np.array([0.0, 0.7, 2.3]), np.array([0.2, 0.5, 0.3])
    beta = beta / np.sum(rho * beta)
    u = np.linspace(1.0, 9.0, 400001)
    mids = 0.5 * (u[1:] + u[:-1])
    quad = np.sum(marginal(5.0, mids, 0.8, 1.3, beta, rho, 1.0)) * (u[1] - u[0])
    assert cell_integral(5.0, 1.0, 9.0, 0.8, 1.3, beta, rho, 1.0) == pytest.approx(quad, rel=1e-5)


def test_broadcasting_over_tenants():
    rng = np.random.default_rng(2)
    shape = Shape(beta=tuple(_atoms(rng, k)[0] for k in (1, 3, 5)), rho=tuple(np.full(k, 1 / k) for k in (1, 3, 5)))
    beta, rho = stack_shape(shape)
    assert beta.shape == (3, 5) and rho.shape == (3, 5) and np.allclose(rho.sum(1), 1)
    x = rng.uniform(0, 20, (7, 3))
    th = rng.uniform(0, 10, (7, 3))
    r, pi = rng.uniform(0.2, 1, 3), rng.uniform(0.5, 2, 3)
    V = value(x, th, r, pi, beta, rho, 1.0)
    for i in range(3):
        np.testing.assert_allclose(V[:, i], value(x[:, i], th[:, i], r[i], pi[i], shape.beta[i], shape.rho[i], 1.0),
                                   rtol=1e-14)
    a = np.zeros((7, 3))
    np.testing.assert_allclose(cell_integral(x, a, th, r, pi, beta, rho, 1.0), V, rtol=1e-12, atol=1e-12)


def test_fit_shape():
    rng = np.random.default_rng(3)
    samples = rng.gamma(0.7, 3.0, 20000)
    beta, rho = fit_shape(samples, n_atoms=64)
    assert beta.size == 64 and np.all(np.diff(beta) >= 0)
    assert abs(rho.sum() - 1) < 1e-12 and abs(np.sum(rho * beta) - 1) < 1e-12
    Shape(beta=(beta,), rho=(rho,))                       # passes the dataclass check
    np.testing.assert_allclose(beta[[0, 31, 63]], np.quantile(samples / samples.mean(), [0.008, 0.49, 0.99]),
                               rtol=0.2)
    b2, r2 = fit_shape([1.0, 3.0], n_atoms=64)            # fewer samples than atoms
    np.testing.assert_allclose(b2, [0.5, 1.5])
    np.testing.assert_allclose(r2, [0.5, 0.5])
    b3, r3 = fit_shape(np.full(100, 4.2))                 # degenerate (exact advance information)
    np.testing.assert_allclose(b3, 1.0)
    for bad in ([], [0.0, 0.0], [1.0, -1.0], [1.0, np.nan]):
        with pytest.raises(ValueError):
            fit_shape(bad)


def test_invalid_inputs():
    beta, rho = np.array([1.0]), np.array([1.0])
    with pytest.raises(ValueError):
        value(-1.0, 1.0, 1.0, 1.0, beta, rho, 1.0)
    with pytest.raises(ValueError):
        value(1.0, 1.0, 0.0, 1.0, beta, rho, 1.0)
    with pytest.raises(ValueError):
        cell_integral(1.0, 2.0, 1.0, 1.0, 1.0, beta, rho, 1.0)
    with pytest.raises(ValueError):
        value(1.0, 1.0, 1.0, 1.0, np.array([1.0, 2.0]), rho, 1.0)
