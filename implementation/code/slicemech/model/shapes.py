"""Declared service value V, its marginal D and closed-form cell integrals (INTERFACES.md §3).

A tenant's public burst shape is a finite set of atoms: the realised rate in an interval is ``beta_l * theta`` with
probability ``rho_l`` (``sum rho = sum rho * beta = 1``).  With quota ``x`` PRB and per-PRB rate ``r`` (Mbit/s per PRB)
the tariff-weighted expected served volume is

    V(x, theta) = pi * Delta * sum_l rho_l * min(beta_l * theta, r * x)            [credits per interval]

and its (inclusive) derivative in the demand argument is

    D(x, u)     = pi * Delta * sum_l rho_l * beta_l * 1{u * beta_l <= r * x}       [credits per interval per Mbit/s].

Broadcasting convention (used by every function here): the atom axis is the LAST axis of ``beta`` and ``rho``; any
leading axes of ``beta``/``rho`` broadcast against the shape of the quota/demand arguments.  So ``beta`` of shape
``(K,)`` serves one tenant for arguments of any shape, and ``beta`` of shape ``(n, K)`` (see :func:`stack_shape`) serves
``n`` tenants for arguments of shape ``(..., n)``.
"""
from __future__ import annotations

import numpy as np

from slicemech.config import Shape

__all__ = ["value", "marginal", "cell_integral", "fit_shape", "stack_shape"]


def _atoms(beta, rho) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return the atom arrays (multipliers >= 0, probabilities >= 0, same shape)."""
    beta = np.asarray(beta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if beta.ndim == 0 or beta.shape != rho.shape:
        raise ValueError(f"beta and rho must be aligned arrays with an atom axis, got {beta.shape} and {rho.shape}")
    if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(rho))) or np.any(beta < 0) or np.any(rho < 0):
        raise ValueError("burst atoms must be finite with beta >= 0 and rho >= 0")
    return beta, rho


def _positive(name: str, a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"{name} must be finite and > 0")
    return a


def _nonnegative(name: str, a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f"{name} must be finite and >= 0")
    return a


def value(x, theta, r, pi, beta, rho, Delta) -> np.ndarray:
    """Declared service value ``V = pi * Delta * sum_l rho_l * min(beta_l * theta, r * x)``.

    Args:
        x: quota (PRB), any shape broadcastable with ``theta``, ``r``, ``pi`` and the leading axes of ``beta``.
        theta: demand / report (Mbit/s), >= 0.
        r: per-PRB rate (Mbit/s per PRB), > 0.
        pi: tariff (credits per Mbit), > 0.
        beta, rho: burst atoms (atom axis last; see module docstring).
        Delta: interval length (s), > 0.

    Returns:
        V (credits per interval), the broadcast shape of the arguments.
    """
    beta, rho = _atoms(beta, rho)
    x = _nonnegative("quota x", x)
    theta = _nonnegative("demand theta", theta)
    r = _positive("rate r", r)
    pi = _positive("tariff pi", pi)
    Delta = float(_positive("Delta", Delta))
    capacity = (r * x)[..., None]                       # Mbit/s
    demand = theta[..., None] * beta                    # Mbit/s per atom
    return pi * Delta * np.sum(rho * np.minimum(demand, capacity), axis=-1)


def marginal(x, u, r, pi, beta, rho, Delta) -> np.ndarray:
    """Inclusive marginal value of demand ``D = pi * Delta * sum_l rho_l * beta_l * 1{u * beta_l <= r * x}``.

    Units: credits per interval per Mbit/s.  Arguments as in :func:`value` with ``u`` (Mbit/s) the demand argument.
    """
    beta, rho = _atoms(beta, rho)
    x = _nonnegative("quota x", x)
    u = _nonnegative("demand u", u)
    r = _positive("rate r", r)
    pi = _positive("tariff pi", pi)
    Delta = float(_positive("Delta", Delta))
    served = u[..., None] * beta <= (r * x)[..., None]
    return pi * Delta * np.sum(np.where(served, rho * beta, 0.0), axis=-1)


def cell_integral(q, a, b, r, pi, beta, rho, Delta) -> np.ndarray:
    """Closed form of ``int_a^b D(q, u) du`` for a constant quota ``q`` on the demand cell ``[a, b]``.

    For atom ``l`` the indicator holds for ``u <= t_l = r q / beta_l``, so the atom contributes
    ``rho_l * beta_l * (clip(t_l, a, b) - a)`` (zero for ``beta_l = 0``).  The result is exact (no quadrature).

    Args:
        q: constant quota on the cell (PRB), >= 0.
        a, b: cell end points (Mbit/s), ``0 <= a <= b`` (raises otherwise).
        r, pi, beta, rho, Delta: as in :func:`value`.

    Returns:
        Integral (credits per interval), broadcast shape of the arguments.
    """
    beta, rho = _atoms(beta, rho)
    q = _nonnegative("quota q", q)
    a = _nonnegative("cell start a", a)
    b = _nonnegative("cell end b", b)
    if np.any(a > b):
        raise ValueError("cell_integral needs a <= b")
    r = _positive("rate r", r)
    pi = _positive("tariff pi", pi)
    Delta = float(_positive("Delta", Delta))
    capacity = (r * q)[..., None]
    capacity, beta_b = np.broadcast_arrays(capacity, beta)
    t = np.divide(capacity, beta_b, out=np.full(capacity.shape, np.inf), where=beta_b > 0)   # Mbit/s
    a_, b_ = a[..., None], b[..., None]
    covered = np.minimum(np.maximum(t, a_), b_) - a_                                          # Mbit/s
    return pi * Delta * np.sum(rho * beta * covered, axis=-1)


def fit_shape(samples, n_atoms: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Quantile atoms of normalised sub-window rates.

    The samples are divided by their mean (so the shape has mean one), sorted and split into ``n_atoms``
    equal-count quantile bins; each atom is the conditional mean of its bin with probability equal to the bin's
    sample share.  Conditional means preserve the mean exactly, so ``sum rho = sum rho * beta = 1`` (re-normalised
    to remove floating-point drift).  With fewer samples than atoms every sample is its own atom.

    Args:
        samples: nonnegative rates (any units; only ratios matter), at least one positive.
        n_atoms: maximum number of atoms (>= 1).

    Returns:
        ``(beta, rho)`` float arrays of equal length, ``beta`` sorted ascending.
    """
    if int(n_atoms) != n_atoms or n_atoms < 1:
        raise ValueError("n_atoms must be a positive integer")
    u = np.asarray(samples, dtype=float).ravel()
    if u.size == 0 or not np.all(np.isfinite(u)) or np.any(u < 0):
        raise ValueError("fit_shape needs a non-empty array of finite nonnegative samples")
    mean = u.mean()
    if mean <= 0:
        raise ValueError("fit_shape needs a positive sample mean")
    u = np.sort(u / mean)
    N = u.size
    if N <= n_atoms:
        beta, rho = u.copy(), np.full(N, 1.0 / N)
    else:
        edges = np.round(np.linspace(0, N, int(n_atoms) + 1)).astype(np.int64)
        counts = np.diff(edges)
        sums = np.add.reduceat(u, edges[:-1])
        beta, rho = sums / counts, counts / N
    rho = rho / rho.sum()
    beta = beta / np.sum(rho * beta)
    return beta, rho


def stack_shape(shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    """Pad a per-tenant :class:`~slicemech.config.Shape` into arrays ``beta, rho`` of shape ``(n, K_max)``.

    Padding atoms have ``beta = rho = 0`` and therefore contribute nothing to V, D or the cell integrals.
    """
    if len(shape.beta) != len(shape.rho) or len(shape.beta) == 0:
        raise ValueError("Shape must hold one atom set per tenant")
    rows = [_atoms(np.ravel(b), np.ravel(r)) for b, r in zip(shape.beta, shape.rho)]
    K = max(b.size for b, _ in rows)
    beta = np.zeros((len(rows), K))
    rho = np.zeros((len(rows), K))
    for i, (b, r) in enumerate(rows):
        beta[i, : b.size], rho[i, : r.size] = b, r
    return beta, rho
