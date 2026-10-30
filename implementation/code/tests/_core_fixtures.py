"""Shared fixtures of the "core" builder's tests: random instances and the exact rational reference adapter.

The exact engine ``rational_general.py`` (ICC_rebuild_2026-10-10/revision_20261017/general) is imported read-only as a
test oracle.  It needs n >= 3 and its own ``PublicContract``; :func:`rg_contract` and :func:`rg_shapes` adapt our
``Contract``/``Mechanism``/``Shape`` (binary floats are read as exact rationals, as the reference does).
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
_RG_DIR = Path(__file__).resolve().parent / "reference"
if str(_RG_DIR) not in sys.path:
    sys.path.insert(0, str(_RG_DIR))
import rational_general as rg  # noqa: E402

from slicemech.config import Contract, Mechanism, Shape  # noqa: E402

CLASSES = ("video", "urllc", "mmtc")


def random_case(rng: np.random.Generator, n: int, *, B: int | None = None, zero_prob: float = 0.15,
                load: float | None = None) -> dict:
    """A random contract, mechanism, rates, reports and weights covering overload, need and leftover stages.

    Demands are rescaled so that ``sum M / r = load * B`` with ``load`` log-uniform on [0.3, 5] unless given.
    Reports: uniform on ``[0, C]``, zero with probability ``zero_prob``, exactly ``C`` with probability 0.05.
    """
    B = int(rng.integers(4, 121)) if B is None else int(B)
    r = rng.uniform(0.1, 1.0, n)
    mu = rng.uniform(0.2, 3.0, n)
    M = mu * rng.uniform(1.0, 4.0, n)
    g = M * rng.uniform(0.0, 1.0, n) * (rng.random(n) < 0.75)
    C = mu * rng.uniform(1.0, 6.0, n)
    load = float(np.exp(rng.uniform(np.log(0.3), np.log(5.0)))) if load is None else float(load)
    s = load * B / np.sum(M / r)
    mu, M, g, C = mu * s, M * s, g * s, C * s
    pi = rng.uniform(0.2, 3.0, n)
    gamma = float(rng.choice([0.0, rng.uniform(0, 1), 1.0], p=[0.25, 0.6, 0.15]))
    eta = float(rng.choice([1.0, rng.uniform(1.0, 4.0)], p=[0.15, 0.85]))
    ct = Contract(cls=tuple(rng.choice(CLASSES, n)), g=g, M=M, C=C, pi=pi, delay_ms=np.full(n, 150.0), mu=mu)
    mech = Mechanism(B=B, gamma=gamma, eta=eta, Delta=1.0, integer=True)
    z = C * rng.uniform(0.0, 1.0, n)
    z[rng.random(n) < zero_prob] = 0.0
    top = rng.random(n) < 0.05
    z[top] = C[top]
    w = np.exp(rng.normal(0.0, 1.5, n))
    return dict(ct=ct, mech=mech, r=r, z=z, w=w, mu=mu)


def random_shape(rng: np.random.Generator, n: int, max_atoms: int = 6, uniform: bool = True) -> Shape:
    """Random burst shapes; ``uniform=True`` gives equiprobable atoms (the reference's ExactShape class)."""
    beta, rho = [], []
    for _ in range(n):
        K = int(rng.integers(1, max_atoms + 1))
        a = rng.uniform(0.0, 3.0, K)
        if rng.random() < 0.3:
            a[0] = 0.0
        if a.sum() <= 0:
            a[:] = 1.0
        p = np.full(K, 1.0 / K) if uniform else rng.dirichlet(np.ones(K))
        b = a / np.sum(p * a)
        beta.append(b)
        rho.append(p)
    return Shape(beta=tuple(beta), rho=tuple(rho))


def rg_contract(ct: Contract, mech: Mechanism) -> "rg.PublicContract":
    """Exact reference contract (n >= 3) with the same binary-float coefficients."""
    return rg.PublicContract(int(mech.B), tuple(ct.g), tuple(ct.M), tuple(ct.C), tuple(ct.pi),
                             float(mech.gamma), float(mech.eta))


def rg_shapes(shape: Shape) -> tuple:
    """Reference ExactShape per tenant (equiprobable atoms only)."""
    out = []
    for b, p in zip(shape.beta, shape.rho):
        if not np.allclose(p, p[0]):
            raise ValueError("rational_general.ExactShape needs equiprobable atoms")
        out.append(rg.ExactShape(tuple(float(v) for v in b)))
    return tuple(out)


def to_float(values) -> np.ndarray:
    return np.array([float(Fraction(v)) for v in values])


def run_parallel(fn, tasks: list, max_workers: int = 30) -> list:
    """Map ``fn`` over ``tasks`` in fresh 'spawn' worker processes with single-threaded BLAS (deterministic,
    no fork-with-threads hazard).  ``fn`` must be a module-level function of an importable test module."""
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ.update({k: "1" for k in saved})
    try:
        workers = max(1, min(len(tasks), (os.cpu_count() or 2) - 2, max_workers))
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            return list(ex.map(fn, tasks))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
