"""Episode runner: one window, one controller, one price rule, one tenant profile (INTERFACES.md §11).

``run_window`` plays a composed :class:`slicemech.data.compose.Window` interval by interval and returns one row per
(interval k, tenant). The order inside interval k is the information order of the mechanism
(theory_design.md §1.2; experiment_design.md §0, conditions C1/C2):

1. **context** from released data only: :func:`slicemech.control.features.context` on a :class:`ReleasedData` view
   that holds the measured counters ``m[j]`` for ``j <= k - d`` and the stale rates ``rate_hat[:k+1]``; unreleased
   rows are zero (counters) or a copy of the latest released row (rates), so a feature that read the future would
   change value (tested). Reports, quotas, prices, deliveries and backlog never enter the context.
2. **priority map** built by the controller from the context alone (``controller.priority_map(ctx)``), or a direct
   rule (``controller.quota_policy(ctx, ...)``). Either is wrapped into the interval's quota rule
   :class:`slicemech.model.allocation.QuotaRule` with the allocator rate ``r`` (``rate_hat[k]``, or
   ``rate_true[k]`` for the diagnostic true-rate arm).
3. **reports**: ``policy.report(i, theta_ik, C_i, history=own history, game=StageGame)``; tenants see only their
   own history (A3). An analytic best responder (``policy.needs_game``) also receives the stage game against
   *truthful* opponents (ex-post best response, experiment_design.md §6).
4. **quotas** ``Q = rule(z)`` (integer PRB unless ``mech.integer`` is False).
5. **prices**: full-decision price ``p`` and fixed-priority price ``p_fix`` of every tenant (envelope, with their
   certified brackets), premium ``T = p - p_fix``, usage price ``nu Q``; the *charged* price is chosen by the price
   rule. The bill is fixed at decision time.
6. **fluid replay** of the interval with TRUE per-ms arrivals and TRUE per-PRB rates
   (:func:`slicemech.sim.fluid.run_interval`; queue state carries over).
7. **release**: after interval k the counters visible at the start of interval k+1 (``j <= k + 1 - d``) are
   released to the operator view.

Units: reports and types in Mbit/s, quotas in PRB, volumes in Mbit per interval, values and prices in credits per
interval, rates in Mbit/s per PRB.

Declared and realized utility (experiment_design.md §3, condition C3):
``V_declared = V_i(Q_i, min(theta_i, C_i))`` is the declared-value model at the tenant's (admitted) type with the
allocator's rate; ``utility_declared = V_declared - price`` is the utility the truthfulness theorem is about.
``realized_value = pi_i * on_time_Mbit_i`` and ``utility_realized = realized_value - price``.
``V_report = V_i(Q_i, z_i)`` is the value the mechanism computes from the report (``0 <= p <= V_report``).

Controller protocol (duck-typed): ``priority_map(ctx, z=None) -> PriorityMap`` built from the context only; or
``quota_policy(ctx, *, ct, mech, shape, r)`` returning a direct-rule-like object (``fractional``/``quotas``).
Optional attributes: ``alloc_input`` ('reports' default, or 'forecast': the allocator receives the operator
forecast ``ctx.f`` and reports never enter, e.g. ``control.nets.MeasOnlyNet``), ``rate_source`` ('hat' default or
'true'), ``forecaster`` (operator AR used to build contexts), ``uses_context_features`` (default True; False for
controllers that read only ``ctx.k``, ``ctx.n`` and ``ctx.mu``). Trained nets (``control.nets``), ``StaticFit``,
``OracleController`` and the adapters below satisfy it.

Operator forecaster (round-1 review, finding 5): a controller that reads context features (every net, and any
controller with ``uses_context_features`` not False) or allocates on the forecast needs the forecaster its
contexts are meant to be built with: ``forecaster=`` or ``controller.forecaster``; otherwise ``run_window``
raises instead of falling back to the untrained default. Context-free controllers (equal/fixed weights, direct
rule, random protected maps, static, oracle) get ``OperatorAR.default()`` for the unused feature columns.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Contract, Mechanism, Shape
from ..control import features as _feat
from ..model import allocation as _alloc
from ..model.direct import DirectRule
from ..model.scores import FixedMap, ProtectedMap
from ..model.shapes import stack_shape, value
from ..pricing import envelope as _env
from ..tenants.strategies import OwnHistory, StageGame
from . import fluid

__all__ = ["PRICE_RULES", "PriceRule", "as_price_rule", "ReleasedData", "FixedWeights", "DirectController",
           "RandomProtected", "MapController", "run_window", "episode_checks", "make_bandit_evaluator",
           "SECTION11_COLUMNS"]

PRICE_RULES = ("full", "fixed", "usage", "none")

#: Columns kept with ``record=False`` (INTERFACES.md §11 plus the charged ``price``).
SECTION11_COLUMNS = ("k", "tenant", "cls", "scored", "theta", "report", "quota", "w", "p", "p_fix", "T", "price",
                     "V_declared", "on_time_Mbit", "dropped_Mbit", "realized_value", "utility_declared",
                     "utility_realized", "overload", "leftover", "need")

_CONSERVATION_RTOL = 1e-9       # conservation residual allowed per Mbit of cumulative arrivals (fluid rounding)


# ============================================================================================ price rule
@dataclass(frozen=True)
class PriceRule:
    """Which price is billed, plus what the usage tariff and the non-monotone envelope need.

    Attributes:
        charge: billed price, one of ``PRICE_RULES``: 'full' (full-decision envelope ``p``), 'fixed'
            (fixed-priority price ``p_fix``), 'usage' (``nu_i Q_i``) or 'none'.
        nu: usage tariff (credits per PRB per interval), shape (n,); required for 'usage', optional otherwise
            (then the ``p_usage`` column is filled).
        nonmonotone_full: for a map that is not monotone by construction, fill the ``p`` column with the envelope
            formula on a 4096-point grid path (:func:`slicemech.pricing.envelope.price_nonmonotone`; NOT a truthful
            price and uncertified). Implied when ``charge == 'full'``. Otherwise ``p`` and ``T`` are NaN there.
    """

    charge: str = "full"
    nu: np.ndarray | None = None
    nonmonotone_full: bool = False

    def __post_init__(self):
        if self.charge not in PRICE_RULES:
            raise ValueError(f"price rule must be one of {PRICE_RULES}, got {self.charge!r}")
        if self.nu is not None:
            nu = np.array(self.nu, dtype=float)
            if nu.ndim != 1 or not np.all(np.isfinite(nu)) or np.any(nu < 0):
                raise ValueError("usage tariff nu must be a finite nonnegative (n,) array (credits per PRB)")
            nu.setflags(write=False)
            object.__setattr__(self, "nu", nu)
        if self.charge == "usage" and self.nu is None:
            raise ValueError("price rule 'usage' needs the usage tariff nu")


def as_price_rule(price_rule) -> PriceRule:
    """Accept a :class:`PriceRule`, a name in ``PRICE_RULES`` or a tuple ``('usage', nu)``."""
    if isinstance(price_rule, PriceRule):
        return price_rule
    if isinstance(price_rule, str):
        return PriceRule(price_rule)
    if isinstance(price_rule, tuple) and len(price_rule) == 2 and isinstance(price_rule[0], str):
        return PriceRule(price_rule[0], nu=price_rule[1])
    raise TypeError("price_rule must be a PriceRule, one of PRICE_RULES, or a tuple (name, nu)")


# ============================================================================================ operator view
class ReleasedData:
    """Report-free operator view of a window (what the controller may know at the start of interval k).

    Holds ``measured`` (K, n) Mbit/s with rows ``j <= k - d`` released and the rest zero, ``rate_hat`` (K, n) Mbit/s
    per PRB with rows ``<= k`` known and later rows copies of row k, and ``meta`` with only the time of day of the
    window start and the CQI sample ages (rows after k masked the same way). It carries no reports, types,
    tenant histories, quotas, prices or deliveries. Call :meth:`advance` before building the context of interval k.
    """

    def __init__(self, window, d: int):
        self.id = getattr(window, "id", None)
        self.split = getattr(window, "split", None)
        m = np.asarray(window.measured, dtype=float)
        rh = np.asarray(window.rate_hat, dtype=float)
        if m.ndim != 2 or rh.shape != m.shape:
            raise ValueError("window.measured and window.rate_hat must be (K, n) arrays of the same shape")
        self._m_src, self._rh_src = m, rh
        meta = getattr(window, "meta", None) or {}
        tod = [meta[key] for key in _feat.TOD_KEYS if key in meta]
        if not tod:
            raise KeyError(f"window.meta needs the time of day of interval 0 under one of {_feat.TOD_KEYS}")
        self._age_src = np.asarray(meta["rate_age_s"], dtype=float) if "rate_age_s" in meta else None
        self.d = int(d)
        self.measured = np.zeros_like(m)
        self.rate_hat = np.empty_like(rh)
        self.meta = {"tod_start_s": float(tod[0])}
        if self._age_src is not None:
            self.meta["rate_age_s"] = np.empty_like(self._age_src)
        self.k = -1

    @property
    def released_through(self) -> int:
        """Index of the newest released counter (``k - d``; negative: nothing released yet)."""
        return self.k - self.d

    def advance(self, k: int) -> None:
        """Release everything the operator knows at the start of interval ``k`` (``k`` never decreases)."""
        K = self._m_src.shape[0]
        if not 0 <= k < K or k < self.k:
            raise ValueError(f"cannot advance the operator view from interval {self.k} to {k}")
        j = k - self.d
        if j >= 0:
            self.measured[: j + 1] = self._m_src[: j + 1]
        self.rate_hat[: k + 1] = self._rh_src[: k + 1]
        self.rate_hat[k + 1:] = self._rh_src[k]
        if self._age_src is not None:
            age = self.meta["rate_age_s"]
            age[: k + 1] = self._age_src[: k + 1]
            age[k + 1:] = self._age_src[k]
        self.k = k


# ============================================================================================ controller adapters
class FixedWeights:
    """Constant priority weights (equal weights when ``w`` is None): ``FixedMap(w)`` in every interval."""

    alloc_input = "reports"
    uses_context_features = False

    def __init__(self, w=None, name: str | None = None):
        self.w = None if w is None else np.asarray(w, dtype=float)
        self.name = name or ("equal" if w is None else "fixed")

    def priority_map(self, ctx, z=None):
        return FixedMap(np.ones(ctx.n) if self.w is None else self.w)


class MapController:
    """Wrap a function ``ctx -> PriorityMap`` (the function must use the context only). Set
    ``uses_context_features=False`` only if ``fn`` reads nothing but ``ctx.k``, ``ctx.n`` and ``ctx.mu``."""

    alloc_input = "reports"

    def __init__(self, fn, name: str = "map", uses_context_features: bool = True):
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.fn, self.name = fn, name
        self.uses_context_features = bool(uses_context_features)

    def priority_map(self, ctx, z=None):
        return self.fn(ctx)


class DirectController:
    """Model-based direct optimizer of declared welfare (:class:`slicemech.model.direct.DirectRule`).

    ``rates='hat'`` uses the stale rate ``rate_hat[k]`` like every controller; ``rates='true'`` is the diagnostic
    arm with the true current rate ``rate_true[k]`` (allocator, declared value and prices then all use it).
    """

    alloc_input = "reports"
    uses_context_features = False

    def __init__(self, rates: str = "hat"):
        if rates not in ("hat", "true"):
            raise ValueError("rates must be 'hat' or 'true'")
        self.rate_source = rates
        self.name = "direct" if rates == "hat" else "direct_true_rates"

    def quota_policy(self, ctx, *, ct: Contract, mech: Mechanism, shape: Shape, r) -> DirectRule:
        return DirectRule(ct, mech, r, shape)


class RandomProtected:
    """Protected map with ``(a, kappa)`` drawn per interval, ``a ~ U[-a_max, a_max]``, ``kappa ~ U[0, kappa_max]``.

    The draw depends only on ``(seed, k)`` (report-free, deterministic). Used by tests and the smoke run.
    """

    alloc_input = "reports"
    uses_context_features = False

    def __init__(self, seed: int = 0, a_max: float = 4.0, kappa_max: float = 2.0):
        if a_max < 0 or kappa_max < 0:
            raise ValueError("a_max and kappa_max must be >= 0")
        self.seed, self.a_max, self.kappa_max = int(seed), float(a_max), float(kappa_max)
        self.name = f"random_protected[{self.seed}]"

    def priority_map(self, ctx, z=None):
        rng = np.random.default_rng([self.seed, int(ctx.k)])
        a = rng.uniform(-self.a_max, self.a_max, ctx.n)
        kappa = rng.uniform(0.0, self.kappa_max, ctx.n)
        return ProtectedMap(a, kappa, ctx.mu)


# ============================================================================================ helpers
def _check_d(d) -> int:
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError(f"measurement delay d must be an integer >= 1 interval, got {d!r}")
    return int(d)


def _controller_name(controller) -> str:
    return str(getattr(controller, "name", None) or getattr(controller, "kind", None) or type(controller).__name__)


def _policy_name(policy) -> str:
    return str(getattr(policy, "name", None) or type(policy).__name__)


def _price_arrays(results, n: int):
    """(p, p_lo, p_hi, quota) arrays from a tuple of PriceResult (NaN when ``results`` is None)."""
    if results is None:
        nan = np.full(n, np.nan)
        return nan, nan, nan, nan
    return (np.array([r.p for r in results]), np.array([r.p_lo for r in results]),
            np.array([r.p_hi for r in results]), np.array([r.quota for r in results]))


# ============================================================================================ episode
def run_window(window, controller, price_rule, tenant_policies, d, mech: Mechanism, shape: Shape, record: bool = True,
               *, forecaster=None, tol: float | None = None, late_counted: bool = False,
               price_columns: str = "all") -> pd.DataFrame:
    """Play one window and return the episode table (one row per interval k and tenant, ordered by (k, tenant)).

    Args:
        window: composed :class:`slicemech.data.compose.Window` (TRUE per-ms arrivals and rates, types ``theta``).
        controller: see the module docstring (priority map or direct rule, built from the context only).
        price_rule: :class:`PriceRule`, a name in ``PRICE_RULES``, or ``('usage', nu)``; selects the billed price.
        tenant_policies: n tenant strategies (``slicemech.tenants.strategies``); ``reset()`` is called first.
        d: measurement delay (intervals, >= 1): counter ``m[j]`` is visible at interval k iff ``j <= k - d``.
        mech: mechanism constants (B, gamma, eta, Delta, integer quotas).
        shape: public burst shapes (one atom set per tenant), used by the declared value and prices.
        record: True -> all diagnostic columns (certificates, stage bounds, rates, backlog, best-response log);
            False -> only ``SECTION11_COLUMNS``.
        forecaster: operator AR forecast for the context (default: ``controller.forecaster``; required for
            controllers that read context features or allocate on the forecast, see the module docstring).
        tol: threshold bracket width for the prices and best responses (Mbit/s; default ``2**-40 C_i``).
        late_counted: fluid variant without expiry (late bits are served and counted in ``late_Mbit``; only for the
            comparison with 5G-LENA, experiment_design.md §3). Default: FIFO with deadline expiry.
        price_columns: 'all' computes both envelope prices every interval (``p``, ``p_fix``, ``T``); 'billed' computes
            only the billed one (the other and ``T`` are NaN), e.g. for bandit evaluations.

    Returns:
        pandas DataFrame; ``df.attrs`` holds window id, split, date, n, K, warmup, d, price rule, controller and
        policy names, wall-clock seconds and a per-step timing breakdown.

    Raises:
        ValueError / TypeError on inconsistent inputs (policy count, price rule without the data it needs, a
        fixed-priority price for a direct rule, a best responder against a report-free controller, reports outside
        [0, C]); ArithmeticError if the replay violates conservation or a price engine disagrees with the deployed
        quota.
    """
    t_start = time.perf_counter()
    pr = as_price_rule(price_rule)
    d = _check_d(d)
    ct: Contract = window.contract
    n, K = ct.n, int(np.asarray(window.theta).shape[0])
    policies = list(tenant_policies)
    if len(policies) != n:
        raise ValueError(f"need one tenant policy per tenant (n={n}), got {len(policies)}")
    for pol in policies:
        if not (hasattr(pol, "report") and hasattr(pol, "reset")):
            raise TypeError(f"{pol!r} does not implement the tenant policy protocol (report, reset)")
    if not isinstance(mech, Mechanism):
        raise TypeError("mech must be a slicemech.config.Mechanism")
    beta, rho = stack_shape(shape)
    if beta.shape[0] != n:
        raise ValueError(f"shape has {beta.shape[0]} tenants, contract has {n}")
    slots = int(round(mech.Delta * fluid.SLOTS_PER_S))
    if slots < 1 or abs(slots - mech.Delta * fluid.SLOTS_PER_S) > 1e-9:
        raise ValueError("mech.Delta must be a positive whole number of 1-ms slots")
    if np.asarray(window.arrivals_ms).shape[0] != K * slots:
        raise ValueError(f"window has {np.asarray(window.arrivals_ms).shape[0]} slots, expected K * {slots}")
    if price_columns not in ("all", "billed"):
        raise ValueError("price_columns must be 'all' or 'billed'")
    want_full = price_columns == "all" or pr.charge == "full"
    want_fixed = price_columns == "all" or pr.charge == "fixed"
    if pr.nu is not None and pr.nu.shape != (n,):
        raise ValueError(f"usage tariff nu must have shape ({n},)")

    direct = hasattr(controller, "quota_policy")
    if not direct and not hasattr(controller, "priority_map"):
        raise TypeError("controller needs priority_map(ctx) or quota_policy(ctx, ...)")
    alloc_input = getattr(controller, "alloc_input", "reports")
    if alloc_input not in ("reports", "forecast"):
        raise ValueError(f"controller.alloc_input must be 'reports' or 'forecast', got {alloc_input!r}")
    reports_enter = alloc_input == "reports"
    rate_source = getattr(controller, "rate_source", "hat")
    if rate_source not in ("hat", "true"):
        raise ValueError(f"controller.rate_source must be 'hat' or 'true', got {rate_source!r}")
    if direct and pr.charge == "fixed":
        raise ValueError("the fixed-priority price needs a priority map; a direct rule has no priority vector")
    if direct and not reports_enter:
        raise ValueError("a direct rule allocates on reports (alloc_input must be 'reports')")
    needs_game = [bool(getattr(pol, "needs_game", False)) for pol in policies]
    if any(needs_game) and not reports_enter:
        raise ValueError("best responses against a report-free controller are trivial (reports never enter); "
                         "use open-loop strategies")
    if forecaster is None:
        forecaster = getattr(controller, "forecaster", None)
    if forecaster is None:
        if getattr(controller, "uses_context_features", True) or not reports_enter:
            raise ValueError(f"controller {_controller_name(controller)!r} reads context features (or allocates on "
                             "the operator forecast) but no operator forecaster was given: pass forecaster= "
                             "(e.g. Calibration.operator_forecaster(d)) or set controller.forecaster")
        forecaster = _feat.OperatorAR.default()          # feature columns unused by this context-free controller

    C, pi = ct.C, ct.pi
    Delta = float(mech.Delta)
    theta_raw = np.asarray(window.theta, dtype=float)
    theta_adm = np.minimum(theta_raw, C)                       # type of admitted traffic (Mbit/s)
    rate_true = np.asarray(window.rate_true, dtype=float)
    rate_hat = np.asarray(window.rate_hat, dtype=float)
    r_alloc_all = rate_hat if rate_source == "hat" else rate_true

    names = ("report", "z_alloc", "r_alloc", "quota", "w", "p", "p_lo", "p_hi", "p_fix", "p_fix_lo", "p_fix_hi",
             "p_usage", "price", "V_declared", "V_report", "arrivals_Mbit", "on_time_Mbit", "dropped_Mbit",
             "late_Mbit", "backlog_start_Mbit", "backlog_Mbit", "unused_Mbit", "L_prb", "U_prb", "br_z_star",
             "br_gain", "br_cert", "br_U_truth")
    col = {name: np.full((K, n), np.nan) for name in names}
    clipped = np.zeros((K, n), dtype=bool)
    br_exact = np.zeros((K, n), dtype=bool)
    br_price = np.full((K, n), "", dtype=object)
    overload = np.zeros(K, dtype=bool)
    leftover = np.zeros(K, dtype=bool)
    timing = dict.fromkeys(("context", "map", "reports", "quotas", "prices", "fluid"), 0.0)

    for pol in policies:
        pol.reset()
    hist = [OwnHistory() for _ in range(n)]
    view = ReleasedData(window, d)
    A = fluid.cumulative_arrivals(window.arrivals_ms)
    state = fluid.initial_state(n, ct.delay_ms)
    view.advance(0)

    for k in range(K):
        # (1) context from released data only
        t0 = time.perf_counter()
        ctx = _feat.context(view, k, d, ct, forecaster=forecaster, mech=mech)
        r = r_alloc_all[k].copy()
        # (2) report-free priority map (or direct rule) -> the interval's quota rule
        t1 = time.perf_counter()
        if direct:
            policy = controller.quota_policy(ctx, ct=ct, mech=mech, shape=shape, r=r)
        else:
            policy = controller.priority_map(ctx)
        rule = _alloc.QuotaRule(ct, mech, r, policy)
        # (3) reports (own history only; the analytic best responder also gets the stage game)
        t2 = time.perf_counter()
        z = np.empty(n)
        for i, pol in enumerate(policies):
            game = None
            if needs_game[i]:
                game = StageGame(rule, np.delete(theta_adm[k], i),
                                 dict(ct=ct, shape=shape, mech=mech, r=r, nu=pr.nu, tol=tol))
            before = int(getattr(pol, "n_clipped", 0))
            z[i] = float(pol.report(i, float(theta_raw[k, i]), float(C[i]), history=hist[i], game=game))
            clipped[k, i] = int(getattr(pol, "n_clipped", 0)) > before
            if needs_game[i]:
                br = getattr(pol, "last", None)
                if br is not None:
                    col["br_z_star"][k, i], col["br_gain"][k, i] = br.z_star, br.gain
                    col["br_cert"][k, i], col["br_U_truth"][k, i] = br.cert, br.U_truth
                    br_exact[k, i] = bool(br.exact)
                    br_price[k, i] = str(getattr(br, "price", getattr(pol, "price", "")))
        if not np.all(np.isfinite(z)) or np.any(z < 0) or np.any(z > C):
            raise ValueError(f"interval {k}: reports {z} outside the report space [0, C]")
        # (4) quotas
        t3 = time.perf_counter()
        if reports_enter:
            z_alloc = z
            Q = np.asarray(rule(z[None, :]))[0]
            w = np.full(n, np.nan) if direct else rule.weights(z)
        else:
            z_alloc = np.asarray(ctx.f, dtype=float)
            w = rule.weights(z_alloc)
            Q = np.asarray(rule(z_alloc[None, :]))[0]
        if np.sum(Q) > mech.B + 1e-9 or np.any(Q < 0):
            raise ArithmeticError(f"interval {k}: quotas {Q} violate the PRB budget B={mech.B}")
        st = _alloc.stage(z_alloc, r, ct, mech)
        # (5) prices (fixed at decision time)
        t4 = time.perf_counter()
        if reports_enter:
            full = fixed = None
            if want_full and rule.monotone_by_construction:
                full = _env.full_prices(z, rule, ct, shape, mech, r, tol=tol)
            elif want_full and (pr.charge == "full" or pr.nonmonotone_full):
                full = tuple(_env.price_nonmonotone(i, z, rule, ct, shape, mech, r, tol=tol) for i in range(n))
            if want_fixed and not direct:
                fixed = _env.fixed_priority_prices(z, rule, ct, shape, mech, r, tol=tol)
            p, p_lo, p_hi, qf = _price_arrays(full, n)
            p_fix, p_fix_lo, p_fix_hi, qx = _price_arrays(fixed, n)
            for q_engine in (qf, qx):
                if np.any(np.isfinite(q_engine) & (q_engine != Q)):
                    raise ArithmeticError(f"interval {k}: price engine quota {q_engine} != deployed quota {Q}")
        else:
            # reports never enter: own quota is constant in the own report, so every envelope price is V(Q, 0) = 0
            p = p_lo = p_hi = p_fix = p_fix_lo = p_fix_hi = np.zeros(n)
        p_usage = pr.nu * Q if pr.nu is not None else np.full(n, np.nan)
        price = {"full": p, "fixed": p_fix, "usage": p_usage, "none": np.zeros(n)}[pr.charge]
        if not np.all(np.isfinite(price)):
            raise ArithmeticError(f"interval {k}: billed price ({pr.charge}) is undefined: {price}")
        Qf = Q.astype(float)
        V_dec = value(Qf, theta_adm[k], r, pi, beta, rho, Delta)
        V_rep = value(Qf, z, r, pi, beta, rho, Delta)
        # (6) fluid replay with TRUE arrivals and TRUE rates
        t5 = time.perf_counter()
        state, out = fluid.run_interval(state, A[k * slots:(k + 1) * slots], Q, rate_true[k], ct.delay_ms,
                                        late_counted=late_counted)
        t6 = time.perf_counter()
        for i in range(n):
            hist[i].append(theta_raw[k, i], z[i], Qf[i], price[i])
        # (7) release the counters visible at the start of interval k + 1
        if k + 1 < K:
            view.advance(k + 1)

        for name, v in (("report", z), ("z_alloc", z_alloc), ("r_alloc", r), ("quota", Qf), ("w", w), ("p", p),
                        ("p_lo", p_lo), ("p_hi", p_hi), ("p_fix", p_fix), ("p_fix_lo", p_fix_lo),
                        ("p_fix_hi", p_fix_hi), ("p_usage", p_usage), ("price", price), ("V_declared", V_dec),
                        ("V_report", V_rep), ("L_prb", st["L"]), ("U_prb", st["U"])):
            col[name][k] = v
        for name in ("arrivals_Mbit", "on_time_Mbit", "dropped_Mbit", "late_Mbit", "backlog_start_Mbit",
                     "backlog_Mbit", "unused_Mbit"):
            col[name][k] = getattr(out, name)
        overload[k], leftover[k] = bool(st["overload"]), bool(st["leftover"])
        for key, a, b in (("context", t0, t1), ("map", t1, t2), ("reports", t2, t3), ("quotas", t3, t4),
                          ("prices", t4, t5), ("fluid", t5, t6)):
            timing[key] += b - a

    # conservation of the whole replay (fluid kernel invariant; cheap guard against misuse of the state hand-over)
    resid = (col["arrivals_Mbit"] - col["on_time_Mbit"] - col["late_Mbit"] - col["dropped_Mbit"]
             - (col["backlog_Mbit"] - col["backlog_start_Mbit"]))
    scale = 1.0 + A[-1] if A.shape[0] else np.ones(n)
    if np.any(np.abs(resid) > _CONSERVATION_RTOL * scale) or np.any(col["backlog_start_Mbit"][1:]
                                                                      != col["backlog_Mbit"][:-1]):
        raise ArithmeticError("fluid replay violated conservation or the interval hand-over")

    realized = pi[None, :] * col["on_time_Mbit"]
    ks = np.repeat(np.arange(K), n)
    data = {
        "k": ks, "tenant": np.tile(np.arange(n), K), "cls": np.tile(np.array(ct.cls, dtype=object), K),
        "scored": ks >= int(window.warmup), "theta": theta_raw.ravel(), "theta_adm": theta_adm.ravel(),
        "report": col["report"].ravel(), "clipped": clipped.ravel(),
    }
    for name in ("z_alloc", "r_alloc"):
        data[name] = col[name].ravel()
    data["r_true"] = rate_true.ravel()
    for name in ("quota", "w", "p", "p_lo", "p_hi", "p_fix", "p_fix_lo", "p_fix_hi"):
        data[name] = col[name].ravel()
    data["T"] = (col["p"] - col["p_fix"]).ravel()
    data["T_lo"] = (col["p_lo"] - col["p_fix_hi"]).ravel()
    data["T_hi"] = (col["p_hi"] - col["p_fix_lo"]).ravel()
    for name in ("p_usage", "price", "V_declared", "V_report", "arrivals_Mbit", "on_time_Mbit", "dropped_Mbit",
                 "late_Mbit", "backlog_start_Mbit", "backlog_Mbit", "unused_Mbit"):
        data[name] = col[name].ravel()
    data["realized_value"] = realized.ravel()
    data["utility_declared"] = (col["V_declared"] - col["price"]).ravel()
    data["utility_realized"] = (realized - col["price"]).ravel()
    data["overload"] = np.repeat(overload, n)
    data["leftover"] = np.repeat(leftover, n)
    data["need"] = ~(data["overload"] | data["leftover"])
    data["L_prb"], data["U_prb"] = col["L_prb"].ravel(), col["U_prb"].ravel()
    data["released_through"] = ks - d
    for name in ("br_z_star", "br_gain", "br_cert", "br_U_truth"):
        data[name] = col[name].ravel()
    data["br_exact"] = br_exact.ravel()
    data["br_price"] = br_price.ravel()
    df = pd.DataFrame(data)
    if not record:
        df = df.loc[:, list(SECTION11_COLUMNS)].copy()
    df.attrs.update(
        window_id=str(getattr(window, "id", "")), split=str(getattr(window, "split", "")),
        date=str(getattr(window, "date", "")), n=n, K=K, warmup=int(window.warmup), d=d, price_rule=pr.charge,
        controller=_controller_name(controller), policies=[_policy_name(p) for p in policies],
        alloc_input=alloc_input, rate_source=rate_source, late_counted=bool(late_counted), price_columns=price_columns,
        seconds=time.perf_counter() - t_start,
        timing_s={key: float(v) for key, v in timing.items()},
    )
    return df


# ============================================================================================ checks
def episode_checks(df: pd.DataFrame, ct: Contract | None = None, mech: Mechanism | None = None,
                   eps_rel: float = 1e-8) -> dict:
    """Invariant diagnostics of an episode table (``record=True``); every ``*_max``/``*excess``/``*violation`` value
    should be <= 0 (or within rounding, about 1e-9).

    Returns dict with
        conservation_max: max |arrivals - on_time - late - dropped - backlog change| (Mbit);
        handover_max: max |backlog_start[k+1] - backlog[k]| (Mbit);
        budget_excess: max_k (sum_i quota - B) (PRB; needs ``mech``);
        cert_violation_p / cert_violation_p_fix: max distance of a price outside its certified bracket (credits);
        price_above_V / price_negative: max(p - V_report) and max(-p) over certified full prices (``0 <= p <= V``);
        premium_negative: max(-T_hi) (credits; the premium is >= 0 for maps satisfying the ratio condition);
        br_rows / br_full_rows: rows with a best-response record / with an exact full-price best response;
        br_full_gain_max: largest full-price best-response gain (credits);
        br_full_excess: max(gain - max(cert, eps_rel pi_i mu_i Delta)) over exact full-price best responses
            (credits; needs ``ct``; NaN if there are none). H3a requires <= 0.
    """
    out = {}
    resid = (df["arrivals_Mbit"] - df["on_time_Mbit"] - df["late_Mbit"] - df["dropped_Mbit"]
             - (df["backlog_Mbit"] - df["backlog_start_Mbit"]))
    out["conservation_max"] = float(np.abs(resid).max())
    piv = df.pivot(index="k", columns="tenant", values=["backlog_start_Mbit", "backlog_Mbit"])
    bs, be = piv["backlog_start_Mbit"].to_numpy(), piv["backlog_Mbit"].to_numpy()
    out["handover_max"] = float(np.abs(bs[1:] - be[:-1]).max()) if len(bs) > 1 else 0.0
    if mech is not None:
        out["budget_excess"] = float((df.groupby("k")["quota"].sum() - mech.B).max())
    for name in ("p", "p_fix"):
        lo, hi, v = df[f"{name}_lo"].to_numpy(), df[f"{name}_hi"].to_numpy(), df[name].to_numpy()
        ok = np.isfinite(lo) & np.isfinite(hi)
        out[f"cert_violation_{name}"] = float(np.max(np.r_[0.0, (lo - v)[ok], (v - hi)[ok]]))
    cert = np.isfinite(df["p_lo"].to_numpy())
    out["price_above_V"] = float(np.max(np.r_[-np.inf, (df["p"] - df["V_report"]).to_numpy()[cert]]))
    out["price_negative"] = float(np.max(np.r_[-np.inf, (-df["p"]).to_numpy()[cert]]))
    th = df["T_hi"].to_numpy()
    out["premium_negative"] = float(np.max(np.r_[-np.inf, -th[np.isfinite(th)]]))
    has = np.isfinite(df["br_gain"].to_numpy())
    out["br_rows"] = int(has.sum())
    full = has & df["br_exact"].to_numpy() & (df["br_price"].to_numpy() == "full")
    out["br_full_rows"] = int(full.sum())
    if full.any():
        gain = df["br_gain"].to_numpy()[full]
        allow = df["br_cert"].to_numpy()[full]
        if ct is not None:
            t = df["tenant"].to_numpy()[full]
            Delta = 1.0 if mech is None else float(mech.Delta)
            allow = np.maximum(allow, eps_rel * ct.pi[t] * ct.mu[t] * Delta)
        out["br_full_gain_max"] = float(gain.max())
        out["br_full_excess"] = float((gain - allow).max())
    else:
        out["br_full_gain_max"] = out["br_full_excess"] = float("nan")
    return out


# ============================================================================================ bandit adapter
def make_bandit_evaluator(windows, controller, price_rule, d, mech: Mechanism, shape: Shape, **kwargs):
    """Adapter for :mod:`slicemech.tenants.bandits`: ``evaluate(window_id, m) -> dict``.

    ``windows``: the episode windows (each with a unique ``id``). For a multiplier profile ``m`` (n,) every tenant
    reports ``min(m_i theta_i, C_i)`` for the whole window (``strategies.multiplier_profile``) and the window is played
    with ``run_window(..., record=False, price_columns='billed', **kwargs)``.

    Returns per call a dict with ``utility`` (n,) = declared utility summed over the scored intervals (credits; the
    bandit reward input), ``utility_realized`` (n,) (credits), ``on_time_Mbit`` (n,) and ``price`` (n,) summed over the
    scored intervals, and ``W_realized`` / ``W_declared`` = mean welfare per scored interval (credits).
    """
    from ..tenants.strategies import multiplier_profile
    by_id = {}
    for w in windows:
        if w.id in by_id:
            raise ValueError(f"duplicate window id {w.id!r}")
        by_id[w.id] = w
    kwargs = dict(kwargs)
    kwargs.setdefault("price_columns", "billed")

    def evaluate(window_id, m) -> dict:
        w = by_id[window_id]
        m = np.asarray(m, dtype=float)
        if m.shape != (w.n,):
            raise ValueError(f"multiplier profile must have shape ({w.n},)")
        df = run_window(w, controller, price_rule, multiplier_profile(m), d, mech, shape, record=False, **kwargs)
        sc = df[df["scored"]]
        tot = sc.groupby("tenant")[["utility_declared", "utility_realized", "on_time_Mbit", "price"]].sum()
        per_k = sc.groupby("k")[["realized_value", "V_declared"]].sum().mean()
        return dict(utility=tot["utility_declared"].to_numpy(), utility_realized=tot["utility_realized"].to_numpy(),
                    on_time_Mbit=tot["on_time_Mbit"].to_numpy(), price=tot["price"].to_numpy(),
                    W_realized=float(per_k["realized_value"]), W_declared=float(per_k["V_declared"]))

    return evaluate
