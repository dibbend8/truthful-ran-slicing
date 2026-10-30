"""Direct declared-welfare optimiser inside the contract stage box (INTERFACES.md §5; theory_design.md Prop. 6).

Within the active stage bounds ``[L, U]`` and stage total of :func:`slicemech.model.allocation.stage`, the rule
maximises ``sum_i V_i(x_i, z_i)`` with ``V_i(x, z) = pi_i Delta sum_l rho_il min(beta_il z, r_i x)``.  ``V_i`` is concave
and piecewise linear in ``x``: between the kinks ``x = beta_(k) z / r`` (atoms sorted ascending) its slope is the
marginal-value band ``pi_i r_i Delta P(beta_i > beta_(k))``.  A greedy fractional knapsack over all bands, in
decreasing slope with the fixed lexicographic tie rule (tenant index, then band index), is therefore exact.  The
band order does not depend on reports, which is what makes the rule monotone in the own report (Prop. 6).
Zero-slope bands are filled too (in tie order) so the stage total is placed exactly as for the weight rules.
Overload rows return the proportional reservation, as the allocator does.
"""
from __future__ import annotations

import numpy as np

from slicemech.config import Contract, Mechanism, Shape
from slicemech.model.allocation import FLOOR_EPS, _check_mech, stage
from slicemech.model.shapes import stack_shape

__all__ = ["direct_fractional", "direct_integer", "DirectRule"]


def _band_plan(r: np.ndarray, ct: Contract, shape: Shape, Delta: float):
    """Report-independent band structure: sorted atoms per tenant, band slopes and the greedy order.

    Returns:
        beta_sorted (n, K): atoms ascending per tenant; slopes (n, K+1) in credits per interval per PRB;
        order (n*(K+1),): flat band indices (tenant-major) in greedy order; inverse (n*(K+1),): the inverse
        permutation of ``order`` (greedy position of each tenant-major band).
    """
    beta, rho = stack_shape(shape)
    n, K = beta.shape
    if n != ct.n:
        raise ValueError(f"shape has {n} tenants, contract has {ct.n}")
    srt = np.argsort(beta, axis=1, kind="stable")
    beta_s = np.take_along_axis(beta, srt, axis=1)
    rho_s = np.take_along_axis(rho, srt, axis=1)
    # tail_k = sum_{l > k} rho_(l) for band k = 0..K (band k lies between atoms k and k+1, 1-based)
    tail = np.concatenate([np.cumsum(rho_s[:, ::-1], axis=1)[:, ::-1], np.zeros((n, 1))], axis=1)
    slopes = (ct.pi * r * Delta)[:, None] * tail                                   # (n, K+1)
    tenant = np.repeat(np.arange(n), K + 1)
    band = np.tile(np.arange(K + 1), n)
    order = np.lexsort((band, tenant, -slopes.ravel()))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return beta_s, slopes, order, inverse


def _direct(z: np.ndarray, r: np.ndarray, ct: Contract, mech: Mechanism, plan) -> np.ndarray:
    """Greedy band fill (PRB). Every row's result is independent of the batch it is evaluated in: the per-tenant
    sum of the used band lengths is a contiguous last-axis reduction, not a BLAS product (round-2 review: the
    former ``used @ onehot`` rounded differently for different batch sizes, so a report within ~1e-14 PRB of a
    floor boundary floored differently in the pricing engine's batches and in the deployed one-row call)."""
    beta_s, _, order, inverse = plan
    n = ct.n
    st = stage(z, r, ct, mech)
    L, U, total = st["L"], st["U"], st["total"]
    kinks = np.clip(z[..., None] * beta_s / r[:, None], L[..., None], U[..., None])        # (..., n, K) PRB
    edges = np.concatenate([L[..., None], kinks, U[..., None]], axis=-1)                    # (..., n, K+2)
    lengths = np.diff(edges, axis=-1).reshape(*z.shape[:-1], -1)[..., order]               # (..., n(K+1))
    before = np.cumsum(lengths, axis=-1) - lengths
    remaining = (total - L.sum(axis=-1))[..., None]
    used = np.minimum(lengths, np.maximum(remaining - before, 0.0))
    per_band = np.ascontiguousarray(used[..., inverse]).reshape(*z.shape[:-1], n, -1)      # (..., n, K+1)
    return L + per_band.sum(axis=-1)


def direct_fractional(z, r, ct: Contract, shape: Shape, mech: Mechanism) -> np.ndarray:
    """Divisible declared-welfare maximiser in the stage box (PRB), vectorised over leading axes of ``z``.

    Args:
        z: reports (Mbit/s), shape ``(..., n)``; r: per-PRB rates (Mbit/s per PRB), shape ``(n,)``;
        ct, shape, mech: contract, public burst shapes, mechanism.
    """
    _check_mech(mech)
    r = np.asarray(r, dtype=float)
    if r.shape != (ct.n,):
        raise ValueError(f"direct rule needs one rate per tenant, shape ({ct.n},)")
    z = np.asarray(z, dtype=float)
    return _direct(z, r, ct, mech, _band_plan(r, ct, shape, mech.Delta))


def direct_integer(z, r, ct: Contract, shape: Shape, mech: Mechanism, floor_eps: float = FLOOR_EPS) -> np.ndarray:
    """Integer direct quotas ``floor(direct_fractional + floor_eps)`` (int64 PRB)."""
    return np.floor(direct_fractional(z, r, ct, shape, mech) + floor_eps).astype(np.int64)


class DirectRule:
    """Direct rule for one interval (not a weight map): ``.fractional(z)`` and ``.quotas(z)``.

    Args:
        ct, mech: contract and mechanism; r: per-PRB rates (Mbit/s per PRB), shape ``(n,)``;
        shape: public burst shapes; floor_eps: integer floor offset (PRB).

    ``quotas(z)`` returns int64 quotas when ``mech.integer`` else fractional quotas.  Wrap it in
    :class:`slicemech.model.allocation.QuotaRule` (or pass it to the pricing functions directly) to price it.
    """

    monotone_by_construction = True

    def __init__(self, ct: Contract, mech: Mechanism, r, shape: Shape, floor_eps: float = FLOOR_EPS):
        _check_mech(mech)
        r = np.array(r, dtype=float)
        if r.shape != (ct.n,) or not np.all(np.isfinite(r)) or np.any(r <= 0):
            raise ValueError(f"r must be a positive finite array of shape ({ct.n},)")
        r.setflags(write=False)
        self.ct, self.mech, self.r, self.shape, self.floor_eps = ct, mech, r, shape, float(floor_eps)
        self._plan = _band_plan(r, ct, shape, mech.Delta)

    @property
    def slopes(self) -> np.ndarray:
        """Marginal-value band slopes (credits per interval per PRB), shape ``(n, K+1)``."""
        return self._plan[1]

    def fractional(self, z) -> np.ndarray:
        return _direct(np.asarray(z, dtype=float), self.r, self.ct, self.mech, self._plan)

    def quotas(self, z) -> np.ndarray:
        x = self.fractional(z)
        return np.floor(x + self.floor_eps).astype(np.int64) if self.mech.integer else x

    def __call__(self, z) -> np.ndarray:
        return self.quotas(z)
