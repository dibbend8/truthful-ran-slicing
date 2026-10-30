"""Per-interval best response of one tenant against fixed opponents (INTERFACES.md §6; experiment_design.md §6).

Utility of report ``z`` for a tenant of type ``theta``: ``U(z) = V(Q_i(z), theta) - price(z)`` (credits per interval),
with ``price`` one of

* ``'full'``  - full-decision envelope price (constant on every own-quota cell for a monotone rule);
* ``'fixed'`` - fixed-priority price (priority frozen at ``w(z)``; for a rule satisfying the ratio condition it is
  nonincreasing inside each own-quota cell, so a cell's supremum is approached at its right end);
* ``'usage'`` - ``nu_i Q_i(z)``;
* ``'none'``  - zero.

Monotone rules (``rule.monotone_by_construction``): exact enumeration over every own-quota cell of ``[0, C_i]``,
plus the truth and the end points ``0`` and ``C_i``.

* ``'full'``, ``'usage'`` and ``'none'``: the utility is constant on a cell.  Each cell is evaluated at its midpoint,
  and both bracket ends of every threshold are added to cover cells narrower than the bracket width.  Ties are
  resolved in candidate order, so a cell's midpoint wins over its bracket ends.
* ``'fixed'``: the fixed price is nonincreasing inside a cell, so the supremum is approached at the right end.  Each
  cell is evaluated at its midpoint and at its right end minus ``1e-9 C_i``.  Bracket ends are NOT evaluated under
  this price.  At a bracket end the own fractional quota equals ``q - floor_eps`` up to the bracket width.  Freezing
  the priority there can leave the frozen own-quota path flat at the floor boundary, where rounding noise of a few ulp
  flips the integer level and the threshold engine reports a spurious drop (integration regression,
  ``tests/test_response.py``).  Cells narrower than ``2e-9 C_i`` are therefore skipped under ``'fixed'``.

Under every price, cells narrower than the threshold tolerance (``2**-40 C_i``) may be missed.

Fractional monotone rules (``rule.integer`` False; round-2 review): the own-quota "cells" are quantum bands of
``rule.quantum`` PRB, inside which the fractional quota ``X_i(u)`` and hence ``V(X_i(u), theta)`` keep changing, so
the cell argument above does not hold. The result is still exact under ``'full'`` (truth is optimal by the
envelope theorem and is always a candidate) and under ``'none'`` (the utility ``V(X_i(u), theta)`` is
nondecreasing in ``u``, so ``C_i`` attains the maximum). Under ``'usage'`` and ``'fixed'`` the own reports at which
``X_i(u)`` crosses each value kink ``beta_l theta / r_i`` are added as candidates (bisection on the monotone path,
both bracket ends), and the result is reported as a search (``exact=False``, ``cert=NaN``).

Ties (round-2 review): a candidate beats the truth only if its utility gain exceeds
``max(tie_tol pi_i Delta C_i, err(z) + err(theta))``, where ``err`` is each price's own error bound. Under
``'fixed'`` every candidate gets its own threshold brackets, so utilities carry independent errors of up to the
certificate; a "gain" inside them is noise and keeps the truth (before, it could make a DSIC arm misreport).

Non-monotone rules: 512-point grid plus golden-section refinement of the 3 best grid points; a lower bound on the
attainable gain (``exact=False``).  The full price of a non-monotone rule is the grid envelope of
:func:`slicemech.pricing.envelope.staircase` (not truthful; shown only to demonstrate manipulation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slicemech.config import Contract, Mechanism, Shape
from slicemech.model.shapes import stack_shape, value
from slicemech.pricing.envelope import _price_batch, staircase, staircase_integral
from slicemech.pricing.thresholds import _bracket, _own_eval, as_rule

__all__ = ["PRICES", "BR", "best_response", "utilities"]

PRICES = ("full", "fixed", "usage", "none")
_GOLD = (np.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class BR:
    """Best-response result (reports in Mbit/s, utilities in credits per interval).

    Attributes:
        z_star: maximising report; gain: ``U_star - U_truth`` (>= 0); U_truth, U_star: utilities of truth and z_star;
        exact: True for the exact enumeration of a monotone rule (integer quotas, or a fractional rule under
        'full'/'none'); price: price rule used;
        cert: numerical certificate of the gain (the price-bracket allowance, ``2 * max error``; NaN for a search);
        n_candidates: number of evaluated reports.
    """

    z_star: float
    gain: float
    U_truth: float
    U_star: float
    exact: bool
    price: str = "full"
    cert: float = 0.0
    n_candidates: int = 0


def _profile_from_opponents(z_minus_i, i: int, n: int) -> np.ndarray:
    z = np.asarray(z_minus_i, dtype=float).ravel()
    if z.size == n - 1:
        z = np.insert(z, i, 0.0)
    elif z.size == n:
        z = z.copy()
        z[i] = 0.0
    else:
        raise ValueError(f"z_minus_i must have n-1={n - 1} opponents (or a full profile of length n={n})")
    return z


class _Game:
    """One tenant's one-interval game with opponents fixed (shared by best_response and utilities)."""

    def __init__(self, i, theta_i, z_minus_i, rule, price, ct, shape, mech, r, nu, tol, n_grid_path):
        if price not in PRICES:
            raise ValueError(f"price must be one of {PRICES}, got {price!r}")
        if shape is None:
            raise ValueError("best_response needs the public burst shape (shape=...)")
        self.rule = as_rule(rule, ct, mech, r)
        n = self.rule.ct.n
        if not 0 <= i < n:
            raise ValueError("tenant index out of range")
        self.i, self.price, self.tol, self.n_grid_path = int(i), price, tol, int(n_grid_path)
        self.base = _profile_from_opponents(z_minus_i, self.i, n)
        self.C_i = float(self.rule.ct.C[self.i])
        self.theta = float(theta_i)
        if not np.isfinite(self.theta) or not 0 <= self.theta <= self.C_i:
            raise ValueError("theta_i must lie in [0, C_i] (Mbit/s)")
        self.beta, self.rho = stack_shape(shape)
        if self.beta.shape[0] != n:
            raise ValueError("shape and contract disagree on n")
        self.args = (self.rule.r[self.i], self.rule.ct.pi[self.i], self.beta[self.i], self.rho[self.i],
                     self.rule.mech.Delta)
        if price == "usage":
            if nu is None:
                raise ValueError("price='usage' needs the usage tariff nu (credits per PRB per interval)")
            nu = np.broadcast_to(np.asarray(nu, dtype=float), (n,))
            if not np.all(np.isfinite(nu)) or np.any(nu < 0):
                raise ValueError("usage tariff nu must be finite and >= 0")
            self.nu_i = float(nu[self.i])
        if price == "fixed" and self.rule.is_direct:
            raise ValueError("the fixed-priority price needs a weight map (a direct rule has no priority vector)")
        self.monotone = self.rule.monotone_by_construction
        self._f = _own_eval(self.rule, self.base[None, :], np.array([self.i]), None)
        self._path = None

    # ---------------------------------------------------------------- primitives
    def quotas(self, u: np.ndarray) -> np.ndarray:
        """Own deployed quotas (PRB) at own reports ``u``."""
        return np.asarray(self._f(np.zeros(u.size, dtype=np.int64), u), dtype=float)

    def V(self, q: np.ndarray, t) -> np.ndarray:
        return value(q, t, *self.args)

    def brackets(self):
        """Threshold brackets of the full map on ``[0, C_i]`` (monotone rules only)."""
        return _bracket(self.rule, self.base[None, :], np.array([self.i]), np.array([self.C_i]), tol=self.tol)

    def _full_path(self):
        """Cached staircase of the full own-quota path on [0, C_i] and its error bound."""
        if self._path is None:
            if self.monotone:
                br = self.brackets()
                starts = np.concatenate([[0.0], 0.5 * (br.lo + br.hi)])
                quotas = np.arange(br.top[0] + 1, dtype=float)
                err = float(self.rule.ct.pi[self.i] * self.rule.mech.Delta * np.sum(br.hi - br.lo))
            else:
                starts, quotas = staircase(self.rule, self.base, self.i, self.C_i, n_grid=self.n_grid_path,
                                           tol=self.tol)
                err = np.nan
            starts = np.append(starts, self.C_i) if starts[-1] < self.C_i else starts
            if quotas.size < starts.size:
                quotas = np.append(quotas, quotas[-1])
            self._path = (starts, quotas, err)
        return self._path

    def prices(self, u: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prices at own reports ``u`` with own quotas ``q``; returns (price, per-point error bound)."""
        if self.price == "none":
            return np.zeros(u.size), np.zeros(u.size)
        if self.price == "usage":
            return self.nu_i * q, np.zeros(u.size)
        if self.price == "full" and (self.rule.integer or not self.monotone):
            starts, quotas, err = self._full_path()
            p = self.V(q, u) - staircase_integral(starts, quotas, u, *self.args)
            return p, np.full(u.size, err)
        Z = np.repeat(self.base[None, :], u.size, axis=0)
        Z[:, self.i] = u
        W = self.rule.weights(Z) if self.price == "fixed" else None
        out = _price_batch(self.rule, self.beta, self.rho, Z, np.full(u.size, self.i), weights=W, tol=self.tol)
        return out["p"], out["err"]

    def utilities(self, u) -> tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=float).ravel()
        if np.any(u < 0) or np.any(u > self.C_i) or not np.all(np.isfinite(u)):
            raise ValueError("reports must lie in [0, C_i]")
        q = self.quotas(u)
        p, err = self.prices(u, q)
        return self.V(q, self.theta) - p, err


def utilities(i: int, theta_i: float, z_minus_i, rule, reports, price: str = "full", *, ct: Contract | None = None,
              shape: Shape | None = None, mech: Mechanism | None = None, r=None, nu=None, tol=None,
              n_grid_path: int = 4096) -> np.ndarray:
    """Utilities ``U(z) = V(Q_i(z), theta_i) - price(z)`` (credits per interval) at the given own reports."""
    game = _Game(i, theta_i, z_minus_i, rule, price, ct, shape, mech, r, nu, tol, n_grid_path)
    return game.utilities(reports)[0]


def _monotone_candidates(game: _Game) -> np.ndarray:
    """Candidate own reports of the exact enumeration (order matters: ties keep the earliest candidate)."""
    br = game.brackets()
    C = game.C_i
    left = np.concatenate([[0.0], br.hi])               # certainly inside cell k from here ...
    right = np.concatenate([br.lo, [C]])                # ... up to here (k = 0 .. top)
    inside = right > left
    mids = 0.5 * (left + right)[inside]
    ends = np.array([game.theta, 0.0, C])
    if game.price == "fixed":
        # never freeze the priority at a bracket end (knife edge of the floor; see the module docstring)
        eps = 1e-9 * C
        wide = (right - left) > 2.0 * eps
        cands = [ends, 0.5 * (left + right)[wide], (right - eps)[wide]]
    else:
        cands = [ends, mids, br.lo, br.hi]              # bracket ends only cover cells narrower than the brackets
    u = np.concatenate(cands)
    return u[(u >= 0) & (u <= C)]


def _kink_candidates(game: _Game, iters: int = 60) -> np.ndarray:
    """Own reports where a fractional own quota ``X_i(u)`` crosses a value kink ``beta_l theta / r_i`` (PRB).

    For each kink inside ``(X_i(0), X_i(C_i)]`` the first crossing ``inf{u: X_i(u) >= kink}`` is bracketed by a
    batched bisection on the monotone path (``iters`` steps); both bracket ends are returned (Mbit/s).
    """
    r_i, _, beta_i, _, _ = game.args
    kinks = np.unique(np.asarray(beta_i, float) * game.theta / r_i)
    x0, x1 = game.quotas(np.array([0.0, game.C_i]))
    kinks = kinks[(kinks > x0) & (kinks <= x1)]
    if kinks.size == 0:
        return np.zeros(0)
    lo, hi = np.zeros(kinks.size), np.full(kinks.size, game.C_i)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        up = game.quotas(mid) >= kinks
        lo, hi = np.where(up, lo, mid), np.where(up, mid, hi)
    return np.concatenate([lo, hi])


def _golden(game: _Game, a: np.ndarray, b: np.ndarray, iters: int):
    """Vectorised golden-section maximisation of U on the intervals [a, b]; returns all evaluated points, values and
    price error bounds."""
    c = b - _GOLD * (b - a)
    d = a + _GOLD * (b - a)
    Uc, ec = game.utilities(c)
    Ud, ed = game.utilities(d)
    pts, vals, errs = [c, d], [Uc, Ud], [ec, ed]
    for _ in range(iters):
        left = Uc >= Ud                                  # keep [a, d]
        a, b = np.where(left, a, c), np.where(left, d, b)
        new = np.where(left, b - _GOLD * (b - a), a + _GOLD * (b - a))
        Un, en = game.utilities(new)
        c, d = np.where(left, new, d), np.where(left, c, new)
        Uc, Ud = np.where(left, Un, Ud), np.where(left, Uc, Un)
        pts.append(new)
        vals.append(Un)
        errs.append(en)
    return np.concatenate(pts), np.concatenate(vals), np.concatenate(errs)


def best_response(i: int, theta_i: float, z_minus_i, rule, price: str = "full", *, ct: Contract | None = None,
                  shape: Shape | None = None, mech: Mechanism | None = None, r=None, nu=None, tol=None,
                  n_grid: int = 512, n_refine: int = 3, golden_iters: int = 60, n_grid_path: int = 4096,
                  tie_tol: float = 1e-12) -> BR:
    """Best response of tenant ``i`` with type ``theta_i`` (Mbit/s) against fixed opponents ``z_minus_i``.

    Args:
        i: tenant; theta_i: true type (Mbit/s, in [0, C_i]);
        z_minus_i: opponents' reports (Mbit/s), length n-1 (or a full profile whose own entry is ignored);
        rule: quota rule; price: 'full' | 'fixed' | 'usage' | 'none';
        ct, mech, r: optional, checked against the rule; shape: public burst shapes (required);
        nu: usage tariff per tenant (credits per PRB per interval; price='usage');
        tol: threshold bracket width (Mbit/s, default 2**-40 C); n_grid, n_refine, golden_iters: non-monotone search;
        n_grid_path: grid of the non-monotone full-price path;
        tie_tol: utility differences up to ``tie_tol * pi_i * Delta * C_i`` (floating-point noise of the price
        sums) count as ties; so does any difference within the two prices' own error bounds (module docstring).

    Returns:
        :class:`BR`.  Ties keep the truthful report and report ``gain = 0``.
    """
    if not tie_tol >= 0:
        raise ValueError("tie_tol must be >= 0")
    game = _Game(i, theta_i, z_minus_i, rule, price, ct, shape, mech, r, nu, tol, n_grid_path)
    U_truth, err_truth = game.utilities(np.array([game.theta]))
    if game.monotone:
        u = _monotone_candidates(game)
        exact = True
        if not game.rule.integer and price in ("usage", "fixed"):
            u = np.concatenate([u, _kink_candidates(game)])   # utility varies inside a quantum band: a search
            exact = False
        U, err = game.utilities(u)
    else:
        grid = np.linspace(0.0, game.C_i, int(n_grid))
        Ug, eg = game.utilities(grid)
        order = np.argsort(-Ug, kind="stable")[: int(n_refine)]
        a = grid[np.maximum(order - 1, 0)]
        b = grid[np.minimum(order + 1, grid.size - 1)]
        up, Up, ep = _golden(game, a, b, int(golden_iters))
        u, U, err = np.concatenate([grid, up]), np.concatenate([Ug, Up]), np.concatenate([eg, ep])
        exact = False
    u = np.concatenate([[game.theta], u])
    U = np.concatenate([U_truth, U])
    err = np.concatenate([err_truth, err])
    # a candidate beats the truth only beyond the floating-point allowance AND both prices' own error bounds
    noise = np.where(np.isfinite(err), err, 0.0)
    threshold = np.maximum(tie_tol * game.rule.ct.pi[game.i] * game.rule.mech.Delta * game.C_i, noise + noise[0])
    beats = (U - U[0]) > threshold
    beats[0] = False
    if np.any(beats):
        k = int(np.flatnonzero(beats)[np.argmax(U[beats])])   # best certified improvement (first on ties)
    else:
        k = 0                                                 # a tie within noise keeps the truth
    if exact:
        cert = 2.0 * float(np.max(err)) if err.size else 0.0
    else:
        cert = float("nan")                              # search: a lower bound on the gain, no certificate
    return BR(z_star=float(u[k]), gain=float(U[k] - U[0]), U_truth=float(U[0]), U_star=float(U[k]), exact=exact,
              price=price, cert=cert, n_candidates=int(u.size))
