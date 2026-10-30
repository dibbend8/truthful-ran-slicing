"""Contract allocator F(z, w) and integer quota rule (INTERFACES.md §2; theory_design.md §1.5).

Stages for one interval (all arrays have the tenant axis last, shape ``(..., n)``; quotas in PRB, rates in Mbit/s,
per-PRB rates ``r`` in Mbit/s per PRB):

1. reservation ``b = min(z, g) / r``;
2. overload (``sum b > B``): ``x = B b / sum b`` (weights do not matter);
3. otherwise spare ``S = B - sum b``, need cap ``H = min(eta z, M) / r``, fairness floor ``l = min(b + gamma S / n, H)``;
   * need stage (``sum H >= B``): weighted fill on ``[l, H]`` with total ``B``;
   * leftover stage (``sum H < B``): weighted fill on ``[H, (M / r) 1{z > 0}]`` with total ``min(B, sum upper)``.

The weighted fill is ``x = clip(tau w, L, U)`` with ``sum x`` equal to the stage total.  It is solved exactly by
sorting the breakpoints ``L/w`` and ``U/w`` (no iteration): the fill total is piecewise linear in the water level
``tau`` with kinks only at those breakpoints, so the root lies on one known linear piece.  A zero report gets a zero
quota in every stage.

``QuotaRule`` packages (contract, mechanism, rates, priority map or direct rule) into the report-to-quota map
``Q(z)`` consumed by :mod:`slicemech.pricing`.
"""
from __future__ import annotations

import numpy as np

from slicemech.config import Contract, Mechanism

__all__ = ["FLOOR_EPS", "stage", "fractional", "integer", "QuotaRule", "make_rule"]

FLOOR_EPS = 1e-9            # deployed quotas are floor(fractional + FLOOR_EPS) (PRB)
_SUM_RTOL = 1e-9            # tolerated relative mismatch of sum(x) and the stage total (floating rounding only)
#: Largest supported weight ratio ``max(w) / min(w)`` within one row (round-2 review). Below it every weight is a
#: normal float after the row is rescaled to max 1, and the water-fill breakpoints stay finite. Softmax scores must
#: therefore spread by at most ``log(1e300) = 690.8`` (``scores.MAX_SCORE_SPREAD = 690``).
MAX_WEIGHT_RATIO = 1e300


# --------------------------------------------------------------------------------------------- validation helpers
def _check_mech(mech: Mechanism) -> None:
    if not (mech.B > 0 and np.isfinite(mech.B)):
        raise ValueError("Mechanism.B must be a positive PRB count")
    if not 0.0 <= mech.gamma <= 1.0:
        raise ValueError("Mechanism.gamma must lie in [0, 1]")
    if not (mech.eta >= 1.0 and np.isfinite(mech.eta)):
        raise ValueError("Mechanism.eta must be >= 1")


def _reports_rates(z, r, ct: Contract) -> tuple[np.ndarray, np.ndarray]:
    """Validate reports (Mbit/s, in [0, C]) and rates (Mbit/s per PRB, > 0) and broadcast them together."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    n = ct.n
    if z.ndim == 0 or z.shape[-1] != n:
        raise ValueError(f"reports must have the tenant axis last with n={n}, got shape {z.shape}")
    if not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("reports must be finite and >= 0 (Mbit/s)")
    if np.any(z > ct.C):
        raise ValueError("reports must not exceed the contract ceiling C (clip them at admission)")
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("per-PRB rates r must be finite and > 0 (Mbit/s per PRB)")
    z, r = np.broadcast_arrays(z, r)
    return z, r


def _weights(w, shape) -> np.ndarray:
    """Validate positive finite weights, broadcast to ``shape`` and rescale each row to max 1 (scale invariance).

    Raises ValueError for non-positive or non-finite weights and for a row whose ratio ``max(w) / min(w)`` exceeds
    :data:`MAX_WEIGHT_RATIO` (the documented range of the exact water fill).
    """
    w = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("priority weights must be finite and > 0")
    w = np.broadcast_to(w, shape)
    w = w / w.max(axis=-1, keepdims=True)
    w_min = w.min(axis=-1)
    if np.any(w_min * MAX_WEIGHT_RATIO < 1.0):
        raise ValueError(f"priority weight ratio max/min = {1.0 / float(w_min.min()):.3e} exceeds the supported "
                         f"{MAX_WEIGHT_RATIO:.0e} (softmax score spread above {np.log(MAX_WEIGHT_RATIO):.1f})")
    return w


# ---------------------------------------------------------------------------------------------------- stages
def stage(z, r, ct: Contract, mech: Mechanism) -> dict:
    """Active-stage bounds of the contract allocator (weights not needed).

    Args:
        z: reports (Mbit/s), shape ``(..., n)``, each in ``[0, C_i]``.
        r: per-PRB rates (Mbit/s per PRB), broadcastable to ``z``.
        ct, mech: contract and mechanism constants.

    Returns:
        dict with
          ``L``, ``U`` (PRB, shape of ``z``): lower and upper bounds of the weighted fill (``L = U = B b / sum b`` in
          overload, so the fill is the identity there);
          ``total`` (PRB, leading shape): PRBs the fill must place;
          ``overload``, ``leftover`` (bool, leading shape): active stage flags (neither = need stage);
          ``b``, ``H`` (PRB) and ``S`` (PRB, leading shape, negative in overload) for diagnostics.
    """
    _check_mech(mech)
    z, r = _reports_rates(z, r, ct)
    n, B = ct.n, float(mech.B)
    b = np.minimum(z, ct.g) / r
    sb = b.sum(axis=-1)
    overload = sb > B
    S = B - sb
    H = np.minimum(mech.eta * z, ct.M) / r
    floor = np.minimum(b + (mech.gamma * np.maximum(S, 0.0) / n)[..., None], H)
    sH = H.sum(axis=-1)
    leftover = ~overload & (sH < B)
    upper_left = np.where(z > 0, ct.M / r, 0.0)
    x_over = np.divide(B * b, sb[..., None], out=np.zeros_like(b), where=overload[..., None])
    ov, lf = overload[..., None], leftover[..., None]
    L = np.where(ov, x_over, np.where(lf, H, floor))
    U = np.where(ov, x_over, np.where(lf, upper_left, H))
    total = np.where(overload, B, np.where(leftover, np.minimum(B, upper_left.sum(axis=-1)), B))
    return dict(L=L, U=U, total=total, overload=overload, leftover=leftover, b=b, H=H, S=S)


_FILL_BISECT_STEPS = 200     # fallback bisection steps on the bracketing piece (never needed in the tests)


def _fill(L: np.ndarray, U: np.ndarray, w: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Exact weighted water-filling ``x = clip(tau w, L, U)`` with ``sum x = total`` (vectorised over rows).

    Requires ``0 <= L <= U``, ``w > 0`` and ``sum L <= total <= sum U`` up to rounding (guaranteed by :func:`stage`).
    Rows with ``total <= sum L`` return ``L`` and rows with ``total >= sum U`` return ``U``. For the rest:

    1. the fill total ``g(tau) = sum_j clip(tau w_j, L_j, U_j)`` is evaluated *directly* at every sorted breakpoint
       ``L/w``, ``U/w`` (O(n^2) per row). It is never accumulated from slope changes: a running slope such as
       ``1 + w_1 + w_2 - 1`` loses every weight below about 1e-16 relative and then selects the wrong piece when the
       weight spread is large (round-1 review, finding 6);
    2. on the first piece ``[t_{k-1}, t_k]`` with ``g(t_k) >= total`` the active set is read at the piece midpoint
       (floored ``tau w_j <= L_j``, capped ``tau w_j >= U_j``, free otherwise) and the water level is solved in
       closed form, ``tau = (total - sum_floored L - sum_capped U) / sum_free w`` (one rounding, any weight spread);
    3. a row whose total still misses by more than the tolerance (a piece only a few ulps wide) is re-solved by
       bisection on that piece with direct evaluation of ``g``.

    Each row is solved in units of ``s = 2^ceil(log2(max U))`` (bounds and total divided by ``s``). The scale is a
    power of two, so the result is bit-identical to the unscaled solve, and the breakpoints stay below
    ``1 / min(w) <= MAX_WEIGHT_RATIO`` for weights rescaled to max 1: they never overflow (round-2 review; before,
    ``U / w`` overflowed to inf for weight ratios above about ``1e306`` and the fill raised a spurious mismatch).
    """
    lead = L.shape[:-1]
    n = L.shape[-1]
    L2, U2 = L.reshape(-1, n), U.reshape(-1, n)
    w2 = np.broadcast_to(w, L.shape).reshape(-1, n)
    T = np.broadcast_to(total, lead).reshape(-1)
    sL, sU = L2.sum(axis=1), U2.sum(axis=1)
    scale = np.maximum(1.0, np.abs(T))
    if np.any(sL > T + _SUM_RTOL * scale) or np.any(L2 > U2):
        raise ArithmeticError("infeasible fill bounds (sum L > total or L > U)")
    x = np.where((T >= sU)[:, None], U2, L2)
    inner = (T > sL) & (T < sU)
    if np.any(inner):
        # exact power-of-two row scale (T < sum U, so max U > 0 on every inner row)
        unit = np.ldexp(1.0, np.frexp(U2[inner].max(axis=1))[1])[:, None]
        Lm, Um, wm, Tm = L2[inner] / unit, U2[inner] / unit, w2[inner], T[inner] / unit[:, 0]
        ts = np.sort(np.concatenate([Lm / wm, Um / wm], axis=1), axis=1)            # (R, 2n) breakpoints in tau
        if not np.all(np.isfinite(ts)):
            raise ArithmeticError("water-fill breakpoints overflow (weight ratio beyond MAX_WEIGHT_RATIO?)")
        g = np.clip(ts[:, :, None] * wm[:, None, :], Lm[:, None, :], Um[:, None, :]).sum(axis=2)   # g(ts_k)
        hit = g >= Tm[:, None]
        k = np.where(hit.any(axis=1), np.argmax(hit, axis=1), ts.shape[1] - 1)
        k = np.maximum(k, 1)
        rows = np.arange(len(k))
        t0, t1 = ts[rows, k - 1], ts[rows, k]
        tm = 0.5 * (t0 + t1)
        y = tm[:, None] * wm
        floored, capped = y <= Lm, y >= Um
        free = ~floored & ~capped
        fixed = np.where(floored, Lm, 0.0).sum(axis=1) + np.where(capped, Um, 0.0).sum(axis=1)
        sw = np.where(free, wm, 0.0).sum(axis=1)
        tau = np.divide(Tm - fixed, sw, out=t1.copy(), where=sw > 0)
        tau = np.minimum(np.maximum(tau, t0), t1)                        # stay on the bracketing piece
        xm = np.minimum(np.maximum(tau[:, None] * wm, Lm), Um)
        # tolerance in PRB, expressed in the row's units
        miss = np.abs(xm.sum(axis=1) - Tm) > _SUM_RTOL * np.maximum(1.0, np.abs(T[inner])) / unit[:, 0]
        if np.any(miss):                                                 # a piece a few ulps wide: bisect on it
            lo, hi = t0[miss], t1[miss]
            Lb, Ub, wb, Tb = Lm[miss], Um[miss], wm[miss], Tm[miss]
            for _ in range(_FILL_BISECT_STEPS):
                mid = 0.5 * (lo + hi)
                below = np.clip(mid[:, None] * wb, Lb, Ub).sum(axis=1) < Tb
                lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
            xm[miss] = np.clip(hi[:, None] * wb, Lb, Ub)
        x[inner] = xm * unit
    mismatch = np.abs(x.sum(axis=1) - T)
    bad = mismatch > _SUM_RTOL * scale
    if np.any(bad):
        raise ArithmeticError(f"water-fill total mismatch up to {mismatch[bad].max():.3e} PRB")
    return x.reshape(L.shape)


def fractional(z, w, r, ct: Contract, mech: Mechanism) -> np.ndarray:
    """Contract allocator ``F(z, w)`` (divisible quotas, PRB).

    Args:
        z: reports (Mbit/s), shape ``(..., n)``.
        w: positive priority weights, broadcastable to ``z`` (scale-free).
        r: per-PRB rates (Mbit/s per PRB), broadcastable to ``z``.
        ct, mech: contract and mechanism constants.

    Returns:
        Fractional quotas (PRB) with the broadcast shape of ``z, w, r``; ``sum x`` equals the stage total.
    """
    z = np.asarray(z, dtype=float)
    shape = np.broadcast_shapes(z.shape, np.shape(w), np.shape(r))
    z = np.broadcast_to(z, shape)
    st = stage(z, r, ct, mech)
    return _fill(st["L"], st["U"], _weights(w, shape), st["total"])


def integer(z, w, r, ct: Contract, mech: Mechanism, floor_eps: float = FLOOR_EPS) -> np.ndarray:
    """Deployed integer quotas ``floor(F(z, w) + floor_eps)`` (int64 PRB); unused PRBs stay idle."""
    return np.floor(fractional(z, w, r, ct, mech) + floor_eps).astype(np.int64)


# ------------------------------------------------------------------------------------------------- quota rule
def _same_contract(a: Contract, b: Contract) -> bool:
    if a is b:
        return True
    return a.cls == b.cls and all(np.array_equal(getattr(a, f), getattr(b, f))
                                  for f in ("g", "M", "C", "pi", "delay_ms", "mu"))


class QuotaRule:
    """Report-to-quota map ``Q(z)`` for one interval: fixed contract, mechanism, rates and policy.

    The policy is either a priority map (an object with ``weights(z)`` and ``monotone_by_construction``;
    see :mod:`slicemech.model.scores`) or a direct rule (an object with ``fractional(z)``, e.g.
    :class:`slicemech.model.direct.DirectRule`).

    Args:
        ct, mech: contract and mechanism.
        r: per-PRB rate estimates (Mbit/s per PRB), shape ``(n,)``; the same ``r`` enters the declared value.
        policy: priority map or direct rule.
        integer: deployed integer quotas (default ``mech.integer``).
        floor_eps: floor offset of the integer quotas (PRB); ``0`` gives the exact floor.
        quantum: pricing resolution of a fractional rule (PRB); integer rules always use 1.

    Calling the rule on reports of shape ``(..., n)`` returns int64 quotas if ``integer`` else float quotas.
    """

    def __init__(self, ct: Contract, mech: Mechanism, r, policy, *, integer: bool | None = None,
                 floor_eps: float = FLOOR_EPS, quantum: float = 2.0 ** -6):
        _check_mech(mech)
        r = np.array(r, dtype=float)
        if r.shape != (ct.n,) or not np.all(np.isfinite(r)) or np.any(r <= 0):
            raise ValueError(f"r must be a positive finite array of shape ({ct.n},)")
        r.setflags(write=False)
        if not (0.0 <= floor_eps < 0.5):
            raise ValueError("floor_eps must lie in [0, 0.5)")
        self.ct, self.mech, self.r, self.policy = ct, mech, r, policy
        self.integer = bool(mech.integer if integer is None else integer)
        self.floor_eps = float(floor_eps)
        if self.integer:
            self.quantum = 1.0
        else:
            if not (0 < quantum <= 1):
                raise ValueError("quantum must lie in (0, 1] PRB")
            self.quantum = float(quantum)
        self.is_direct = not hasattr(policy, "weights")
        if self.is_direct:
            if not hasattr(policy, "fractional"):
                raise TypeError("policy needs weights(z) (priority map) or fractional(z) (direct rule)")
            for name, mine in (("ct", ct), ("mech", mech)):
                theirs = getattr(policy, name, mine)
                same = _same_contract(theirs, mine) if name == "ct" else theirs == mine
                if not same:
                    raise ValueError(f"direct rule was built with a different {name}")
            if hasattr(policy, "r") and not np.array_equal(np.asarray(policy.r, float), r):
                raise ValueError("direct rule was built with different rates r")

    @property
    def n(self) -> int:
        return self.ct.n

    @property
    def monotone_by_construction(self) -> bool:
        """True when the policy guarantees own-quota monotonicity for every contract (Theorem 1 / Prop. 6)."""
        return bool(getattr(self.policy, "monotone_by_construction", False))

    def weights(self, z) -> np.ndarray:
        """Priority weights ``w(z)`` (raises for a direct rule, which has none)."""
        if self.is_direct:
            raise TypeError("a direct rule has no priority weights")
        return np.asarray(self.policy.weights(np.asarray(z, dtype=float)), dtype=float)

    def fractional(self, z) -> np.ndarray:
        """Fractional quotas ``F(z, w(z))`` (PRB)."""
        z = np.asarray(z, dtype=float)
        if self.is_direct:
            return np.asarray(self.policy.fractional(z), dtype=float)
        return fractional(z, self.weights(z), self.r, self.ct, self.mech)

    def fractional_with_weights(self, z, w) -> np.ndarray:
        """Fractional quotas ``F(z, w)`` with explicitly given weights (one row of ``w`` per row of ``z``)."""
        if self.is_direct:
            raise TypeError("a direct rule has no priority weights")
        return fractional(z, w, self.r, self.ct, self.mech)

    def _deploy(self, x: np.ndarray) -> np.ndarray:
        return np.floor(x + self.floor_eps).astype(np.int64) if self.integer else x

    def __call__(self, z) -> np.ndarray:
        """Deployed quotas ``Q(z)`` (int64 PRB for integer rules, float PRB otherwise)."""
        return self._deploy(self.fractional(z))

    def with_weights(self, z, w) -> np.ndarray:
        """Deployed quotas under explicitly frozen weights ``w`` (rows aligned with ``z``)."""
        return self._deploy(self.fractional_with_weights(z, w))

    def frozen(self, z) -> "QuotaRule":
        """The same rule with the priority vector frozen at ``w(z)`` (used by the fixed-priority price)."""
        from slicemech.model.scores import FixedMap  # local import: scores does not depend on allocation
        z = np.asarray(z, dtype=float)
        if z.shape != (self.n,):
            raise ValueError("frozen() needs one report profile of shape (n,)")
        return QuotaRule(self.ct, self.mech, self.r, FixedMap(self.weights(z)), integer=self.integer,
                         floor_eps=self.floor_eps, quantum=self.quantum)

    def __repr__(self) -> str:
        kind = "direct" if self.is_direct else type(self.policy).__name__
        return f"QuotaRule(n={self.n}, policy={kind}, integer={self.integer})"


def make_rule(ct: Contract, mech: Mechanism, r, policy, **kwargs) -> QuotaRule:
    """Convenience constructor, ``QuotaRule(ct, mech, r, policy, **kwargs)``."""
    return QuotaRule(ct, mech, r, policy, **kwargs)
