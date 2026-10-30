"""Envelope prices: full-decision, fixed-priority, premium, usage tariff (INTERFACES.md §6; theory Prop. 2-3).

For tenant ``i`` with opponents fixed and a monotone quota rule ``Q``:

    p(z)     = V(Q_i(z), z_i) - int_0^{z_i} D(Q_i(u, z_-i), u) du                  (full-decision price)
    p_fix(z) = V(Q_i(z), z_i) - int_0^{z_i} D(Q_i^{w(z)}(u, z_-i), u) du            (priority frozen at w(z))
    T(z)     = p(z) - p_fix(z)                                                       (priority premium, >= 0)

Units: credits per interval.  The integrand is a staircase in the own quota, so the integral is a sum of closed-form
cell integrals over the own-quota cells between consecutive thresholds (:mod:`slicemech.pricing.thresholds`).
Certificate: putting every threshold at its bracket's lower end over-states the staircase (``D`` is nondecreasing in
the quota) and at its upper end under-states it, so ``p_lo <= p <= p_hi`` holds exactly in real arithmetic for the
evaluated rule, with ``p_hi - p_lo <= pi Delta sum_q (u_hi - u_lo)`` (integer rules).  For fractional rules the
cells are quota bands of width ``rule.quantum`` and the bracket uses the band ends.  A floating-point allowance of
``1e-12 pi Delta C_i`` is added to both ends.

Monotonicity guard: the certificate argument assumes a monotone staircase, and a non-monotone rule has no DSIC
price at all (theory Lemma C). The full-map entry points (:func:`full_price`, :func:`full_prices`, :func:`premium`)
therefore refuse a rule whose policy is not ``monotone_by_construction`` with :class:`NonMonotoneRuleError`, even
when its own quota happens to rise on the 65-point check grid (a drop between grid points would otherwise receive
a tight but invalid certificate; round-1 review, finding 7). Pass ``assume_monotone=True`` only for a rule whose
monotonicity is known from elsewhere (e.g. a menu selector proved monotone by hand); use
:func:`price_nonmonotone` for the uncertified grid envelope. The fixed-priority price evaluates the rule with
frozen weights, which is monotone for every priority map, so it needs no flag.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slicemech.config import Contract, Mechanism, Shape
from slicemech.model.shapes import cell_integral, stack_shape, value
from slicemech.pricing.thresholds import (TOL_REL, NonMonotoneRuleError, _bracket, _levels, _own_eval, _profile,
                                         as_rule, require_monotone)

__all__ = ["NonMonotoneRuleError", "PriceResult", "Premium", "full_price", "full_prices", "fixed_priority_price",
           "fixed_priority_prices", "premium", "usage_price", "price_nonmonotone", "staircase", "staircase_integral"]

_FP_ALLOWANCE = 1e-12


@dataclass(frozen=True)
class PriceResult:
    """One tenant's price (credits per interval).

    Attributes:
        p: price; V: declared value ``V(Q_i(z), z_i)``; [p_lo, p_hi]: certified bracket (NaN when uncertified);
        quota: own deployed quota (PRB); err: ``pi Delta sum_q (u_hi - u_lo)`` (integer rules) or the half-width of
        the bracket (fractional rules); n_levels: number of own-quota thresholds used.
    """

    p: float
    V: float
    p_lo: float
    p_hi: float
    quota: float = 0.0
    err: float = 0.0
    n_levels: int = 0


@dataclass(frozen=True)
class Premium:
    """Priority premium ``T = p - p_fix`` with its bracket and both underlying prices (credits per interval)."""

    T: float
    T_lo: float
    T_hi: float
    full: PriceResult
    fixed: PriceResult


def _setup(rule, ct, shape, mech, r):
    rule = as_rule(rule, ct, mech, r)
    beta, rho = stack_shape(shape)
    if beta.shape[0] != rule.ct.n:
        raise ValueError(f"shape has {beta.shape[0]} tenants, contract has {rule.ct.n}")
    return rule, beta, rho


def _price_batch(rule, beta, rho, Zb, own, weights=None, tol=None, n_grid: int = 65) -> dict:
    """Envelope prices of P problems (profile ``Zb[p]``, own tenant ``own[p]``, optional frozen ``weights[p]``).

    Returns a dict of (P,) arrays: p, V, p_lo, p_hi, quota, err, n_levels.
    """
    Zb = np.asarray(Zb, dtype=float)
    own = np.asarray(own, dtype=np.int64)
    P = own.size
    rows = np.arange(P)
    Delta = float(rule.mech.Delta)
    zown = Zb[rows, own]
    out = {k: np.zeros(P) for k in ("p", "V", "p_lo", "p_hi", "err")}
    out["quota"], out["n_levels"] = np.zeros(P), np.zeros(P, dtype=np.int64)
    act = np.flatnonzero(zown > 0)
    idle = np.flatnonzero(zown <= 0)
    if idle.size:                                     # zero own report: price 0, quota from one rule call
        Zi = Zb[idle]
        X = rule(Zi) if weights is None else rule.with_weights(Zi, np.asarray(weights)[idle])
        out["quota"][idle] = np.asarray(X, dtype=float)[np.arange(idle.size), own[idle]]
    if act.size == 0:
        return out
    own_a, z_a = own[act], zown[act]
    br = _bracket(rule, Zb[act], own_a, z_a, weights=None if weights is None else np.asarray(weights)[act],
                  tol=tol, n_grid=n_grid)
    # the own quota at the report is the last grid point of the threshold engine's own evaluation (u_max = z), so
    # the price, its cells and the reported quota come from one batch (round-2 review: comparing it with a
    # separate P-row call compared two batch layouts)
    x_a = br.top_quota
    out["quota"][act] = x_a
    # cells of levels 0..top per problem; cell k starts at threshold k (0 for k = 0) and ends at k+1 (z for the top)
    ncell = br.top + 1
    offs = np.concatenate([[0], np.cumsum(ncell)])
    cp = np.repeat(np.arange(act.size), ncell)
    ck = np.arange(offs[-1]) - offs[cp]
    thr = np.maximum(br.starts[cp] + ck - 1, 0)
    has = ck >= 1
    lo = br.lo if br.lo.size else np.zeros(1)
    hi = br.hi if br.hi.size else np.zeros(1)
    s_lo = np.where(has, lo[thr], 0.0)
    s_hi = np.where(has, hi[thr], 0.0)
    s_mid = 0.5 * (s_lo + s_hi)
    last = ck == br.top[cp]

    def ends(s):
        nxt = np.empty_like(s)
        nxt[:-1] = s[1:]
        return np.where(last, z_a[cp], nxt)

    tenant = own_a[cp]
    r_c, pi_c = rule.r[tenant], rule.ct.pi[tenant]
    b_c, rho_c = beta[tenant], rho[tenant]

    def integral(q, s):
        vals = cell_integral(q, s, ends(s), r_c, pi_c, b_c, rho_c, Delta)
        return np.bincount(cp, weights=vals, minlength=act.size)

    V = value(x_a, z_a, rule.r[own_a], rule.ct.pi[own_a], beta[own_a], rho[own_a], Delta)
    fp = _FP_ALLOWANCE * rule.ct.pi[own_a] * Delta * rule.ct.C[own_a]
    width = np.bincount(br.prob, weights=br.hi - br.lo, minlength=act.size) if br.prob.size else np.zeros(act.size)
    if rule.integer:
        q = ck.astype(float)
        p = V - integral(q, s_mid)
        p_lo = V - integral(q, s_lo) - fp
        p_hi = V - integral(q, s_hi) + fp
        err = rule.ct.pi[own_a] * Delta * width
    else:
        h = rule.quantum
        q_low = ck * h
        q_up = np.minimum((ck + 1) * h, x_a[cp])
        p_lo = V - integral(q_up, s_lo) - fp
        p_hi = V - integral(q_low, s_hi) + fp
        p = 0.5 * (p_lo + p_hi)
        err = 0.5 * (p_hi - p_lo)
    out["p"][act], out["V"][act], out["p_lo"][act], out["p_hi"][act] = p, V, p_lo, p_hi
    out["err"][act], out["n_levels"][act] = err, br.top
    return out


def _result(out: dict, k: int) -> PriceResult:
    return PriceResult(p=float(out["p"][k]), V=float(out["V"][k]), p_lo=float(out["p_lo"][k]),
                       p_hi=float(out["p_hi"][k]), quota=float(out["quota"][k]), err=float(out["err"][k]),
                       n_levels=int(out["n_levels"][k]))


def full_price(i: int, z, rule, ct: Contract, shape: Shape, mech: Mechanism, r, *, tol=None,
               n_grid: int = 65, assume_monotone: bool = False) -> PriceResult:
    """Full-decision envelope price of tenant ``i`` at the report profile ``z`` (Mbit/s, shape (n,)).

    ``rule`` must be monotone in the own report: its policy must be ``monotone_by_construction`` (or the caller
    passes ``assume_monotone=True``), and the own quota must not fall on the ``n_grid`` check grid; otherwise
    :class:`~slicemech.pricing.thresholds.NonMonotoneRuleError` is raised. ``ct``, ``mech`` and ``r`` must be the
    rule's own (checked).  ``tol`` is the threshold bracket width (Mbit/s, default ``2**-40 C``).
    """
    rule, beta, rho = _setup(rule, ct, shape, mech, r)
    require_monotone(rule, assume_monotone, "full_price")
    z = _profile(z, rule.ct.n)
    return _result(_price_batch(rule, beta, rho, z[None, :], np.array([i]), tol=tol, n_grid=n_grid), 0)


def full_prices(z, rule, ct: Contract, shape: Shape, mech: Mechanism, r, *, tol=None,
                n_grid: int = 65, assume_monotone: bool = False) -> tuple[PriceResult, ...]:
    """Full-decision prices of all tenants at ``z``, computed in one batched threshold pass (same monotonicity
    requirement as :func:`full_price`)."""
    rule, beta, rho = _setup(rule, ct, shape, mech, r)
    require_monotone(rule, assume_monotone, "full_prices")
    z = _profile(z, rule.ct.n)
    n = rule.ct.n
    out = _price_batch(rule, beta, rho, np.repeat(z[None, :], n, axis=0), np.arange(n), tol=tol, n_grid=n_grid)
    return tuple(_result(out, k) for k in range(n))


def fixed_priority_price(i: int, z, rule, ct: Contract, shape: Shape, mech: Mechanism, r, *, tol=None,
                         n_grid: int = 65) -> PriceResult:
    """Fixed-priority price: the envelope of the rule re-evaluated with ``FixedMap(w(z))`` (frozen at the report)."""
    rule, beta, rho = _setup(rule, ct, shape, mech, r)
    z = _profile(z, rule.ct.n)
    w = rule.weights(z)[None, :]
    return _result(_price_batch(rule, beta, rho, z[None, :], np.array([i]), weights=w, tol=tol, n_grid=n_grid), 0)


def fixed_priority_prices(z, rule, ct: Contract, shape: Shape, mech: Mechanism, r, *, tol=None,
                          n_grid: int = 65) -> tuple[PriceResult, ...]:
    """Fixed-priority prices of all tenants at ``z`` (one batched pass)."""
    rule, beta, rho = _setup(rule, ct, shape, mech, r)
    z = _profile(z, rule.ct.n)
    n = rule.ct.n
    W = np.repeat(rule.weights(z)[None, :], n, axis=0)
    out = _price_batch(rule, beta, rho, np.repeat(z[None, :], n, axis=0), np.arange(n), weights=W, tol=tol,
                       n_grid=n_grid)
    return tuple(_result(out, k) for k in range(n))


def premium(i: int, z, rule, ct: Contract, shape: Shape, mech: Mechanism, r, *, tol=None,
            n_grid: int = 65, assume_monotone: bool = False) -> Premium:
    """Priority premium ``T = p - p_fix`` of tenant ``i`` (credits per interval) with bracket ``[T_lo, T_hi]``."""
    full = full_price(i, z, rule, ct, shape, mech, r, tol=tol, n_grid=n_grid, assume_monotone=assume_monotone)
    fixed = fixed_priority_price(i, z, rule, ct, shape, mech, r, tol=tol, n_grid=n_grid)
    return Premium(T=full.p - fixed.p, T_lo=full.p_lo - fixed.p_hi, T_hi=full.p_hi - fixed.p_lo,
                   full=full, fixed=fixed)


def usage_price(x, nu) -> np.ndarray:
    """Usage tariff ``nu_i x_i`` (credits per interval) for quotas ``x`` (PRB, shape (..., n)) and ``nu`` >= 0
    (credits per PRB per interval, set revenue-neutral on training data)."""
    x = np.asarray(x, dtype=float)
    nu = np.asarray(nu, dtype=float)
    if not np.all(np.isfinite(nu)) or np.any(nu < 0):
        raise ValueError("usage tariff nu must be finite and >= 0")
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("quotas must be finite and >= 0")
    return nu * x


# ------------------------------------------------------------------------------- non-monotone (grid) envelope
def staircase(rule, z, i: int, u_max: float, n_grid: int = 4096, tol: float | None = None):
    """Piecewise-constant own-quota path on ``[0, u_max]`` for ANY rule (no monotonicity needed).

    The path is evaluated on an ``n_grid`` grid; every grid interval whose end levels differ is bisected (both
    halves are kept while their end levels differ, so several changes inside one grid interval are all located)
    down to width ``tol`` (default ``2**-40 C_i``).  Changes that start and end at the same level inside one grid
    interval are not detected (grid resolution).

    Returns:
        ``(starts, quotas)``: cell ``k`` is ``[starts[k], starts[k+1])`` (the last cell ends at ``u_max``) with own
        deployed quota ``quotas[k]`` (PRB) evaluated at its start.
    """
    rule = as_rule(rule)
    z = _profile(z, rule.ct.n)
    tol = TOL_REL * float(rule.ct.C[i]) if tol is None else float(tol)
    f = _own_eval(rule, z[None, :], np.array([i]), None)
    u = np.linspace(0.0, float(u_max), int(n_grid))
    u[-1] = u_max
    q = np.asarray(f(np.zeros(u.size, dtype=np.int64), u))
    lev = _levels(rule, q)
    pts, qs = [u], [q]
    ch = np.flatnonzero(lev[:-1] != lev[1:])
    a, b, la, lb = u[ch], u[ch + 1], lev[ch], lev[ch + 1]
    while a.size:
        keep = (b - a) > tol
        a, b, la, lb = a[keep], b[keep], la[keep], lb[keep]
        if not a.size:
            break
        m = 0.5 * (a + b)
        qm = np.asarray(f(np.zeros(m.size, dtype=np.int64), m))
        lm = _levels(rule, qm)
        pts.append(m)
        qs.append(qm)
        left, right = lm != la, lm != lb
        a, b = np.concatenate([a[left], m[right]]), np.concatenate([m[left], b[right]])
        la, lb = np.concatenate([la[left], lm[right]]), np.concatenate([lm[left], lb[right]])
    pts, qs = np.concatenate(pts), np.concatenate(qs).astype(float)
    order = np.argsort(pts, kind="stable")
    pts, qs = pts[order], qs[order]
    keep = np.concatenate([[True], np.diff(pts) > 0])
    return pts[keep], qs[keep]


def staircase_integral(starts, quotas, u, r_i: float, pi_i: float, beta_i, rho_i, Delta: float) -> np.ndarray:
    """``int_0^u D(Q(s), s) ds`` for a staircase path (see :func:`staircase`) at query reports ``u`` (Mbit/s)."""
    starts, quotas = np.asarray(starts, dtype=float), np.asarray(quotas, dtype=float)
    u = np.asarray(u, dtype=float)
    if np.any(u < starts[0]) or np.any(u > starts[-1]):
        raise ValueError("query outside the staircase support")
    ends = np.append(starts[1:], starts[-1])
    full = cell_integral(quotas, starts, ends, r_i, pi_i, beta_i, rho_i, Delta)
    cum = np.concatenate([[0.0], np.cumsum(full)])
    j = np.searchsorted(starts, u, side="right") - 1
    part = cell_integral(quotas[j], starts[j], np.maximum(u, starts[j]), r_i, pi_i, beta_i, rho_i, Delta)
    return cum[j] + part


def price_nonmonotone(i: int, z, rule, ct: Contract, shape: Shape, mech: Mechanism, r, *, n_grid: int = 4096,
                      tol: float | None = None) -> PriceResult:
    """Envelope formula on a grid path (4096 points plus bisection on jumps) for a possibly non-monotone rule.

    Only a demonstration that pricing cannot repair a non-monotone map: the result is NOT a truthful price, can be
    negative and carries no certificate (``p_lo = p_hi = NaN``).
    """
    rule, beta, rho = _setup(rule, ct, shape, mech, r)
    z = _profile(z, rule.ct.n)
    q_z = float(np.asarray(rule(z[None, :]))[0, i])
    if z[i] == 0:
        return PriceResult(p=0.0, V=0.0, p_lo=np.nan, p_hi=np.nan, quota=q_z, err=np.nan)
    args = (rule.r[i], rule.ct.pi[i], beta[i], rho[i], rule.mech.Delta)
    starts, quotas = staircase(rule, z, i, z[i], n_grid=n_grid, tol=tol)
    V = float(value(q_z, z[i], *args))
    I = float(staircase_integral(starts, quotas, z[i], *args))
    return PriceResult(p=V - I, V=V, p_lo=np.nan, p_hi=np.nan, quota=q_z, err=np.nan, n_levels=int(starts.size))
