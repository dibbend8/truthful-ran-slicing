"""Own-report quota thresholds by vectorised bisection (INTERFACES.md §6).

For tenant ``i`` with opponents' reports fixed, a monotone quota rule gives a nondecreasing staircase
``u -> Q_i(u, z_-i)``.  Its level ``q`` threshold is ``u_q = inf{u in [0, u_max]: Q_i(u, z_-i) >= q}``.  The engine
brackets every requested threshold of many independent problems at once:

1. one batched rule call on an ``n_grid``-point own-report grid per problem (this doubles as the monotonicity
   check: a decrease on the grid raises :class:`NonMonotoneRuleError`), giving initial brackets;
2. bisection of all brackets together, one batched rule call per step, until ``u_hi - u_lo <= tol``
   (default ``tol = 2**-40 * C_i``).

Bracket semantics: ``Q_i(u_lo) < q <= Q_i(u_hi)`` (``u_lo = 0`` also when ``Q_i(0) < q`` is only known from
``Q_i(0) = 0``), hence ``u_q`` lies in ``(u_lo, u_hi]``.  Levels are integer quotas for integer rules; for a
fractional rule they are multiples of ``rule.quantum`` PRB (level ``k`` means ``k * quantum <= x < (k+1) * quantum``).
A problem may carry frozen weights (one row per problem): that is how the fixed-priority price is evaluated for
many report profiles in one pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slicemech.config import Contract, Mechanism
from slicemech.model.allocation import FLOOR_EPS, QuotaRule, _same_contract

__all__ = ["NonMonotoneRuleError", "TOL_REL", "as_rule", "require_monotone", "own_thresholds", "own_path", "max_drop",
           "Brackets"]

TOL_REL = 2.0 ** -40            # default bracket width relative to the own ceiling C_i
_MAX_STEPS = 400                # hard guard on bisection steps (never reached for tol >= 2**-60 C)


class NonMonotoneRuleError(ValueError):
    """Raised when a rule's own quota decreases on the check grid, or when a rule that is not monotone by
    construction is handed to a full-map threshold/price engine (thresholds and envelope prices undefined)."""


def require_monotone(rule, assume_monotone: bool, where: str) -> None:
    """Refuse a rule whose policy is not ``monotone_by_construction`` unless the caller opts in explicitly.

    The grid check of :func:`_bracket` cannot see a drop that starts and ends between two grid points, and such a
    rule would get a tight but invalid certificate. ``assume_monotone=True`` is the caller's declaration that the
    rule is monotone for a reason the flag does not record.
    """
    if not isinstance(assume_monotone, bool):
        raise TypeError("assume_monotone must be a bool")
    if not assume_monotone and not bool(getattr(rule, "monotone_by_construction", False)):
        raise NonMonotoneRuleError(
            f"{where}: the rule is not monotone by construction, so threshold brackets and envelope certificates "
            "are not valid for it; use pricing.envelope.price_nonmonotone (uncertified grid envelope) or pass "
            "assume_monotone=True if monotonicity is known from elsewhere")


class _CallableRule:
    """Adapter giving a bare callable ``Q(z)`` the attributes the pricing engine needs (not monotone by default)."""

    is_direct = True

    def __init__(self, fn, ct: Contract, mech: Mechanism, r, monotone: bool = False):
        if ct is None or mech is None or r is None:
            raise ValueError("a bare callable rule needs ct, mech and r")
        self.fn, self.ct, self.mech = fn, ct, mech
        self.r = np.array(r, dtype=float)
        if self.r.shape != (ct.n,) or np.any(self.r <= 0):
            raise ValueError(f"r must be positive with shape ({ct.n},)")
        self.integer = bool(mech.integer)
        self.quantum = 1.0 if self.integer else 2.0 ** -6
        self.floor_eps = FLOOR_EPS
        self.monotone_by_construction = bool(monotone)

    @property
    def n(self) -> int:
        return self.ct.n

    def __call__(self, z):
        return np.asarray(self.fn(np.asarray(z, dtype=float)))

    fractional = __call__

    def weights(self, z):
        raise TypeError("a bare callable rule has no priority weights")


def as_rule(rule, ct: Contract | None = None, mech: Mechanism | None = None, r=None):
    """Return a pricing-ready rule; check that given ``ct``, ``mech``, ``r`` agree with the rule's own.

    Accepts a :class:`~slicemech.model.allocation.QuotaRule`, a direct rule (``.quotas``/``.fractional``, wrapped
    into a QuotaRule) or a bare callable ``Q(z)`` (then ``ct``, ``mech`` and ``r`` are required).
    """
    if isinstance(rule, (QuotaRule, _CallableRule)):
        out = rule
    elif hasattr(rule, "quotas") and hasattr(rule, "fractional") and hasattr(rule, "ct"):
        out = QuotaRule(rule.ct, rule.mech, rule.r, rule, floor_eps=getattr(rule, "floor_eps", FLOOR_EPS))
    elif callable(rule):
        return _CallableRule(rule, ct, mech, r)
    else:
        raise TypeError("rule must be a QuotaRule, a direct rule or a callable Q(z)")
    if ct is not None and not _same_contract(ct, out.ct):
        raise ValueError("ct differs from the rule's contract")
    if mech is not None and mech != out.mech:
        raise ValueError("mech differs from the rule's mechanism")
    if r is not None and not np.array_equal(np.asarray(r, dtype=float), out.r):
        raise ValueError("r differs from the rule's rates (the declared value must use the allocator's r)")
    return out


def _levels(rule, q) -> np.ndarray:
    """Pricing levels of deployed quotas: the integer quota itself, or ``floor(x / quantum + floor_eps)``."""
    q = np.asarray(q)
    if rule.integer:
        lev = np.asarray(q, dtype=np.int64)
        if not np.array_equal(lev, q):
            raise ValueError("integer rule returned non-integer quotas")
        return lev
    return np.floor(np.asarray(q, dtype=float) / rule.quantum + rule.floor_eps).astype(np.int64)


def _own_eval(rule, base: np.ndarray, own: np.ndarray, weights: np.ndarray | None):
    """Closure ``f(pidx, u)`` = own deployed quota of problem ``pidx`` at own report ``u`` (one batched call)."""

    def f(pidx: np.ndarray, u: np.ndarray) -> np.ndarray:
        Z = base[pidx].copy()
        rows = np.arange(pidx.size)
        cols = own[pidx]
        Z[rows, cols] = u
        Q = rule(Z) if weights is None else rule.with_weights(Z, weights[pidx])
        return np.asarray(Q)[rows, cols]

    return f


@dataclass
class Brackets:
    """Threshold brackets of P problems (rows grouped by problem, levels ascending within a problem).

    Attributes:
        prob, level: problem index and level of each row; lo, hi: bracket ends (Mbit/s);
        top: level at u_max per problem; starts: row offsets (P+1,); u_max: search end per problem (Mbit/s);
        top_quota: deployed own quota at u_max per problem (PRB; float), from the same grid evaluation as ``top``.
    """

    prob: np.ndarray
    level: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    top: np.ndarray
    starts: np.ndarray
    u_max: np.ndarray
    top_quota: np.ndarray | None = None


def _bracket(rule, base, own, u_max, *, weights=None, q_hi=None, tol=None, n_grid: int = 65,
             check: bool = True) -> Brackets:
    """Core engine: brackets of levels ``1..q_hi[p]`` (default: the level at ``u_max[p]``) for every problem ``p``.

    Args:
        rule: pricing-ready rule (see :func:`as_rule`).
        base: (P, n) report profiles (own component ignored); own: (P,) tenant per problem;
        u_max: (P,) search end (Mbit/s, > 0); weights: optional (P, n) frozen weights;
        q_hi: optional (P,) highest level wanted; tol: (n,) bracket width per tenant (Mbit/s);
        n_grid: initial grid points per problem (>= 2); check: raise on a grid decrease.
    """
    base = np.asarray(base, dtype=float)
    own = np.asarray(own, dtype=np.int64)
    u_max = np.asarray(u_max, dtype=float)
    P = own.size
    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")
    if np.any(u_max <= 0):
        raise ValueError("threshold search needs u_max > 0")
    tol = TOL_REL * rule.ct.C if tol is None else np.broadcast_to(np.asarray(tol, dtype=float), (rule.ct.n,))
    if np.any(tol <= 0):
        raise ValueError("tol must be > 0")
    f = _own_eval(rule, base, own, weights)
    grid = u_max[:, None] * np.linspace(0.0, 1.0, n_grid)[None, :]
    grid[:, -1] = u_max
    q_grid = np.asarray(f(np.repeat(np.arange(P), n_grid), grid.ravel()))
    lev = _levels(rule, q_grid).reshape(P, n_grid)
    top_quota = q_grid.reshape(P, n_grid)[:, -1].astype(float)
    drops = np.diff(lev, axis=1) < 0
    if check and np.any(drops):
        p, k = np.argwhere(drops)[0]
        raise NonMonotoneRuleError(
            f"own quota of tenant {own[p]} falls from level {lev[p, k]} to {lev[p, k + 1]} between own reports "
            f"{grid[p, k]:.6g} and {grid[p, k + 1]:.6g} Mbit/s")
    top = lev[:, -1]
    q_hi = top.copy() if q_hi is None else np.asarray(q_hi, dtype=np.int64)
    if np.any(q_hi > top):
        raise ValueError("requested level is not reached on [0, u_max]")
    counts = np.maximum(q_hi, 0)
    starts = np.concatenate([[0], np.cumsum(counts)])
    R = int(starts[-1])
    prob = np.repeat(np.arange(P), counts)
    level = np.arange(R) - starts[prob] + 1
    # first grid index with lev >= level (lev rows are sorted once the check passed)
    lmin = int(lev.min()) if lev.size else 0
    span = int(lev.max() - lmin + 2) if lev.size else 1
    key = (lev - lmin + np.arange(P)[:, None] * span).ravel()
    k = np.searchsorted(key, np.maximum(level - lmin, 0) + prob * span, side="left") - prob * n_grid
    lo = np.where(k > 0, grid[prob, np.maximum(k - 1, 0)], 0.0)
    hi = grid[prob, k].copy()
    tol_row = tol[own[prob]]
    active = (hi - lo) > tol_row
    steps = 0
    while np.any(active):
        idx = np.flatnonzero(active)
        mid = 0.5 * (lo[idx] + hi[idx])
        ge = _levels(rule, f(prob[idx], mid)) >= level[idx]
        hi[idx] = np.where(ge, mid, hi[idx])
        lo[idx] = np.where(ge, lo[idx], mid)
        active[idx] = (hi[idx] - lo[idx]) > tol_row[idx]
        steps += 1
        if steps > _MAX_STEPS:
            raise ArithmeticError("threshold bisection did not converge (tol below floating resolution?)")
    if check and R > 1:
        same = prob[1:] == prob[:-1]
        if np.any(np.diff(hi)[same] < 0) or np.any(np.diff(lo)[same] < 0):
            raise NonMonotoneRuleError("thresholds are not ordered by level: the own quota is not monotone")
    return Brackets(prob=prob, level=level, lo=lo, hi=hi, top=top, starts=starts, u_max=u_max, top_quota=top_quota)


def _profile(z, n: int) -> np.ndarray:
    z = np.array(z, dtype=float)
    if z.shape != (n,):
        raise ValueError(f"report profile must have shape ({n},)")
    return z


def own_thresholds(i: int, z, rule, q_hi: int, tol: float | None = None, *, u_max: float | None = None,
                   n_grid: int = 65, ct: Contract | None = None, mech: Mechanism | None = None, r=None,
                   assume_monotone: bool = False):
    """Brackets of the own-quota thresholds of tenant ``i`` for levels ``1 .. q_hi``.

    Args:
        i: tenant; z: report profile (Mbit/s, shape (n,); own component ignored); rule: monotone quota rule;
        q_hi: highest level (PRB for integer rules); tol: bracket width (Mbit/s, default ``2**-40 C_i``);
        u_max: search end (Mbit/s, default ``C_i``); n_grid: grid of the initial brackets and monotonicity check;
        assume_monotone: accept a rule that is not ``monotone_by_construction`` (see :func:`require_monotone`).

    Returns:
        ``(u_lo, u_hi)`` arrays of length ``q_hi``: ``u_q`` lies in ``(u_lo[q-1], u_hi[q-1]]``.

    Raises:
        NonMonotoneRuleError: the rule is not monotone by construction (and ``assume_monotone`` is False), or the
            own quota decreases on the check grid.
        ValueError: level ``q_hi`` is not reached on ``[0, u_max]``.
    """
    rule = as_rule(rule, ct, mech, r)
    require_monotone(rule, assume_monotone, "own_thresholds")
    z = _profile(z, rule.ct.n)
    C_i = float(rule.ct.C[i])
    u_max = C_i if u_max is None else float(u_max)
    if not 0 < u_max <= C_i:
        raise ValueError("u_max must lie in (0, C_i]")
    tol_v = None if tol is None else np.where(np.arange(rule.ct.n) == i, float(tol), 1.0)
    br = _bracket(rule, z[None, :], np.array([i]), np.array([u_max]), q_hi=np.array([int(q_hi)]),
                  tol=tol_v, n_grid=n_grid)
    return br.lo, br.hi


def own_path(i: int, z, rule, u) -> np.ndarray:
    """Own deployed quotas ``Q_i(u, z_-i)`` at the own reports ``u`` (one batched rule call)."""
    rule = as_rule(rule)
    z = _profile(z, rule.ct.n)
    u = np.asarray(u, dtype=float).ravel()
    Z = np.repeat(z[None, :], u.size, axis=0)
    Z[:, i] = u
    return np.asarray(rule(Z))[:, i]


def max_drop(path) -> tuple[float, int]:
    """Largest decrease ``max_{k < l} (path[k] - path[l])`` of a sequence and the index ``l`` where it ends.

    Returns ``(0.0, -1)`` for a nondecreasing sequence.  Used for monotonicity fuzzing and violation rates.
    """
    path = np.asarray(path, dtype=float)
    if path.size < 2:
        return 0.0, -1
    run_max = np.maximum.accumulate(path)
    drop = run_max - path
    l = int(np.argmax(drop))
    return (float(drop[l]), l) if drop[l] > 0 else (0.0, -1)
