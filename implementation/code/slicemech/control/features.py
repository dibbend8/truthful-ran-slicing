"""Report-free controller context (INTERFACES.md §9; experiment_design.md §4, condition C2).

The context of interval k is built ONLY from quantities that tenants' reports cannot influence:

* released measured demand ``measured[j]`` (Mbit/s) for ``j <= k - d`` (the counter of interval j is
  released d intervals after it started; d >= 1 because interval k's own demand is measured after it);
* the operator's AR(6) forecast ``f`` (Mbit/s) of interval k built from those released values (the forecaster
  is a required argument, fitted on TRAIN per delay d by ``data.calibrate``; ``OperatorAR.default()`` is an
  explicit opt-in for tests);
* the stale per-PRB rate ``rate_hat`` (Mbit/s per PRB; ``rate_hat[k] = rate_true[k-1]`` is known
  before interval k) and three lags;
* the time of day and the public contract constants.

Reports (``window.theta`` or any stored report), quotas, prices and deliveries are never read. The
guard ``_assert_report_free`` raises :class:`ReportLeakError` when a caller tries to hand any of them in.

Feature layout (per tenant, ``N_TENANT_FEATURES = 22``; names in ``TENANT_FEATURES``):
``onehot(video, urllc, mmtc) | g/mu, M/mu, pi/mean(pi), log(delay_s) | m[k-d-l]/mu for l=0..7 |
counter age | f/mu | rate_hat[k-l]/R_BAR for l=0..3 | rate age``.
Global (``N_GLOBAL_FEATURES = 3``): ``sin(tod), cos(tod), sum_j f_j/(rate_hat_j B)``.
Ages are in seconds, capped at ``AGE_CAP_S`` and divided by ``AGE_SCALE_S`` (documented feature scaling).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..config import Contract, Mechanism

CLASSES = ("video", "urllc", "mmtc")
N_COUNTERS = 8            # released measured-demand lags used as features
N_RATE_LAGS = 4           # rate_hat[k], rate_hat[k-1], rate_hat[k-2], rate_hat[k-3]
AR_LAGS = 6               # operator AR(6) forecast
R_BAR = 0.434             # Mbit/s per PRB; mean Raca 5G rate (experiment_design.md §2), feature scale only
AGE_CAP_S = 30.0          # ages above this are reported as the cap (also: "no measurement released yet")
AGE_SCALE_S = 10.0
SECONDS_PER_DAY = 86400.0

TENANT_FEATURES = (
    ["is_video", "is_urllc", "is_mmtc", "g_over_mu", "M_over_mu", "pi_over_mean_pi", "log_delay_s"]
    + [f"m_lag{l}_over_mu" for l in range(N_COUNTERS)]
    + ["counter_age"]
    + ["forecast_over_mu"]
    + [f"rate_hat_lag{l}_over_rbar" for l in range(N_RATE_LAGS)]
    + ["rate_age"]
)
GLOBAL_FEATURES = ("tod_sin", "tod_cos", "est_load")
N_TENANT_FEATURES = len(TENANT_FEATURES)
N_GLOBAL_FEATURES = len(GLOBAL_FEATURES)
IDX_FORECAST = TENANT_FEATURES.index("forecast_over_mu")
IDX_COUNTER0 = TENANT_FEATURES.index("m_lag0_over_mu")

# Names that identify reports or report-dependent outcomes. Passing any of them (as keyword or as an
# attribute of the "window" object, e.g. an episode log) is refused.
FORBIDDEN_NAMES = frozenset({
    "z", "report", "reports", "stored_reports", "theta", "z_report", "quota", "quotas", "x", "x_int",
    "price", "prices", "p", "p_fix", "premium", "delivery", "deliveries", "on_time", "on_time_Mbit",
    "served", "utility", "w_used", "backlog",
})
_FORBIDDEN_ATTRS = FORBIDDEN_NAMES - {"theta", "x", "p"}   # Window legitimately carries theta (unused here)
TOD_KEYS = ("tod_start_s", "start_tod_s", "t0_tod_s")


class ReportLeakError(RuntimeError):
    """Raised when a report (or a report-dependent quantity) is offered to a report-free component."""


class Context(tuple):
    """Context of one interval: unpacks as ``(c_tenant (n, F), c_global (G,))``.

    Extra attributes (all report-free): ``f`` operator forecast (n,) Mbit/s, ``r_hat`` (n,) Mbit/s per PRB,
    ``mu`` (n,) Mbit/s, ``k`` interval index, ``d`` measurement delay (intervals), ``window_id``.
    """

    def __new__(cls, c_tenant, c_global, *, f, r_hat, mu, k, d, window_id=None):
        self = super().__new__(cls, (c_tenant, c_global))
        self.c_tenant, self.c_global = c_tenant, c_global
        self.f, self.r_hat, self.mu = f, r_hat, mu
        self.k, self.d, self.window_id = int(k), int(d), window_id
        return self

    @property
    def n(self) -> int:
        return self.c_tenant.shape[0]


# ----------------------------------------------------------------------------- operator forecast
@dataclass(frozen=True)
class OperatorAR:
    """Direct-horizon AR(lags) operator forecast on released measured demand, per tenant class.

    ``coef[cls]`` has shape ``(lags + 1,)`` = (intercept, c_0 .. c_{lags-1}); the forecast of interval k is
    ``f_i = mu_i * (c0 + sum_l c_l * m[k-d-l, i]/mu_i)``, projected onto the report space ``[0, C_i]``
    (admitted traffic is policed to the ceiling, config.Contract). ``horizon`` is the delay d the
    coefficients were fitted for (None = valid for any d, used by the untrained default).
    Missing lags (``k-d-l < 0``) are filled with the training mean (normalised value 1).
    """

    coef: dict
    horizon: int | None = None
    lags: int = AR_LAGS
    info: dict = field(default_factory=dict)

    @staticmethod
    def default() -> "OperatorAR":
        """Untrained default: equal-weight mean of the last 6 released values (no intercept)."""
        c = np.r_[0.0, np.full(AR_LAGS, 1.0 / AR_LAGS)]
        return OperatorAR(coef={k: c.copy() for k in CLASSES}, horizon=None,
                          info=dict(kind="mean_of_last_6_released"))

    def to_dict(self) -> dict:
        """JSON-able form (coefficients per class as lists)."""
        return dict(coef={c: np.asarray(v, float).tolist() for c, v in self.coef.items()}, horizon=self.horizon,
                    lags=int(self.lags), info=dict(self.info))

    @classmethod
    def from_dict(cls, d: dict) -> "OperatorAR":
        h = d.get("horizon")
        return cls(coef={c: np.asarray(v, float) for c, v in d["coef"].items()},
                   horizon=None if h is None else int(h), lags=int(d.get("lags", AR_LAGS)),
                   info=dict(d.get("info", {})))

    def matrix(self, cls: tuple) -> np.ndarray:
        """Coefficient matrix (n, lags+1) for the given tenant classes."""
        try:
            return np.stack([np.asarray(self.coef[c], float) for c in cls])
        except KeyError as e:
            raise ValueError(f"OperatorAR has no coefficients for class {e}") from None


def fit_operator_ar(windows, d: int, lags: int = AR_LAGS, ridge: float = 1e-8) -> OperatorAR:
    """Fit the horizon-d operator AR forecast on TRAINING windows only (released measured demand only).

    Target ``m[j+d]/mu`` regressed on ``[1, m[j]/mu, ..., m[j-lags+1]/mu]`` pooled over all tenants of a class
    and all windows; no regression pair crosses a window boundary. Ridge ``ridge`` (normalised units) only
    regularises the normal equations. Raises if a window is not from the training split.
    """
    d = _check_d(d)
    X, Y = {c: [] for c in CLASSES}, {c: [] for c in CLASSES}
    for w in windows:
        if getattr(w, "split", None) != "train":
            raise ValueError(f"fit_operator_ar uses training windows only; got split={getattr(w, 'split', None)!r}")
        m = _measured(w)
        mu = np.asarray(w.contract.mu, float)
        K = m.shape[0]
        if K < lags + d:
            continue
        mn = m / mu
        j = np.arange(lags - 1, K - d)
        lagmat = np.stack([mn[j - l] for l in range(lags)], axis=1)      # (P, lags, n)
        tgt = mn[j + d]                                                    # (P, n)
        for i, c in enumerate(w.cls):
            X[c].append(lagmat[:, :, i]); Y[c].append(tgt[:, i])
    coef, info = {}, {}
    for c in CLASSES:
        if not X[c]:
            continue
        x = np.concatenate(X[c]); y = np.concatenate(Y[c])
        A = np.c_[np.ones(len(x)), x]
        G = A.T @ A + ridge * np.eye(A.shape[1])
        coef[c] = np.linalg.solve(G, A.T @ y)
        info[c] = dict(pairs=int(len(x)), rmse=float(np.sqrt(np.mean((A @ coef[c] - y) ** 2))))
    if not coef:
        raise ValueError("no training pairs: windows too short for the AR history and horizon")
    return OperatorAR(coef=coef, horizon=d, lags=lags, info=info)


# ----------------------------------------------------------------------------- guards and accessors
def _assert_report_free(window, extra: dict):
    bad = sorted(set(extra) & FORBIDDEN_NAMES)
    if bad:
        raise ReportLeakError(f"report-free context refuses report-dependent inputs: {bad}")
    if extra:
        raise TypeError(f"unexpected keyword arguments: {sorted(extra)}")
    if isinstance(window, np.ndarray) or np.isscalar(window):
        raise ReportLeakError("context() expects a Window, not an array (reports are never an input)")
    cols = getattr(window, "columns", None)            # an episode/log DataFrame
    if cols is not None and set(map(str, cols)) & (FORBIDDEN_NAMES | {"report"}):
        raise ReportLeakError("context() was given a log table containing reports/outcomes")
    try:
        names = set(vars(window))
    except TypeError:                                   # objects without __dict__ (slots)
        names = set()
    names |= set(getattr(type(window), "__dataclass_fields__", {}))
    bad = sorted(names & _FORBIDDEN_ATTRS)
    if bad:
        raise ReportLeakError(f"window object carries report-dependent fields {bad}")


def _require_forecaster(forecaster) -> "OperatorAR":
    if forecaster is None:
        raise ValueError("an operator forecaster is required (no silent default): pass a fitted OperatorAR, e.g. "
                         "Calibration.operator_forecaster(d), or OperatorAR.default() explicitly")
    if not isinstance(forecaster, OperatorAR):
        raise TypeError(f"forecaster must be an OperatorAR, got {type(forecaster).__name__}")
    return forecaster


def _check_d(d) -> int:
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError(f"measurement delay d must be an integer >= 1 interval, got {d!r}")
    return int(d)


def _measured(window) -> np.ndarray:
    m = np.asarray(window.measured, float)
    if m.ndim != 2 or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("window.measured must be a finite nonnegative (K, n) array in Mbit/s")
    return m


def _rate_hat(window, K, n) -> np.ndarray:
    r = np.asarray(window.rate_hat, float)
    if r.shape != (K, n):
        raise ValueError(f"window.rate_hat must have shape {(K, n)}, got {r.shape}")
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("window.rate_hat must be finite and > 0 (Mbit/s per PRB)")
    return r


def _tod_start(window) -> float:
    meta = getattr(window, "meta", None) or {}
    for key in TOD_KEYS:
        if key in meta:
            t = float(meta[key])
            if not np.isfinite(t):
                raise ValueError(f"window.meta[{key!r}] must be finite")
            return t
    start = meta.get("start")
    if start is not None and hasattr(start, "hour"):
        return start.hour * 3600.0 + start.minute * 60.0 + start.second + getattr(start, "microsecond", 0) * 1e-6
    raise KeyError(f"window.meta needs the time of day of interval 0 in seconds under one of {TOD_KEYS} "
                   "(or a datetime under 'start')")


def _rate_age(window, ks, K, n, Delta) -> np.ndarray:
    """Age (s) of the rate sample behind ``rate_hat[k]`` for the intervals ``ks``: ``meta['rate_age_s']`` (K, n)
    if the data pipeline provides it, else one interval (``rate_hat[k] = rate_true[k-1]``)."""
    meta = getattr(window, "meta", None) or {}
    if "rate_age_s" in meta:
        a = np.asarray(meta["rate_age_s"], float)
        if a.shape != (K, n) or not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError("window.meta['rate_age_s'] must be a finite nonnegative (K, n) array (s)")
        return a[ks]
    return np.full((len(ks), n), Delta)


def _static_features(ct: Contract) -> np.ndarray:
    n = ct.n
    onehot = np.zeros((n, len(CLASSES)))
    for i, c in enumerate(ct.cls):
        if c not in CLASSES:
            raise ValueError(f"unknown tenant class {c!r}; expected one of {CLASSES}")
        onehot[i, CLASSES.index(c)] = 1.0
    if np.any(ct.mu <= 0) or np.any(ct.delay_ms <= 0):
        raise ValueError("contract.mu and contract.delay_ms must be > 0")
    return np.c_[onehot, ct.g / ct.mu, ct.M / ct.mu, ct.pi / ct.pi.mean(), np.log(ct.delay_ms / 1000.0)]


# ----------------------------------------------------------------------------- feature computation
def _features(window, ks: np.ndarray, d: int, ct: Contract, forecaster: OperatorAR, mech: Mechanism):
    """Vectorised core: features for the intervals ``ks`` (S,). Returns (c_t (S,n,F), c_g (S,G), f (S,n), r (S,n))."""
    m = _measured(window)
    K, n = m.shape
    if ct.n != n:
        raise ValueError(f"contract has n={ct.n} tenants but window.measured has n={n}")
    if np.any(ks < 0) or np.any(ks >= K):
        raise IndexError(f"interval index out of range [0, {K})")
    if forecaster.horizon is not None and forecaster.horizon != d:
        raise ValueError(f"forecaster was fitted for horizon {forecaster.horizon}, used with d={d}")
    r_hat = _rate_hat(window, K, n)
    Delta = float(mech.Delta)
    S = len(ks)

    # released counters m[k-d-l], l = 0..7 (normalised by mu); unreleased (index < 0) -> training mean 1
    J = ks[:, None] - d - np.arange(max(N_COUNTERS, forecaster.lags))[None, :]          # (S, L)
    valid = J >= 0
    lag = np.where(valid[:, :, None], m[np.where(valid, J, 0)] / ct.mu, 1.0)            # (S, L, n)
    counters = lag[:, :N_COUNTERS, :]
    age = np.where(ks - d >= 0, d * Delta, AGE_CAP_S)
    counter_age = np.minimum(age, AGE_CAP_S) / AGE_SCALE_S                              # (S,)

    # operator forecast (direct horizon-d AR), projected onto [0, C]
    coef = forecaster.matrix(ct.cls)                                                     # (n, lags+1)
    pred = coef[:, 0][None, :] + np.einsum("sln,nl->sn", lag[:, :forecaster.lags, :], coef[:, 1:])
    f = np.clip(pred, 0.0, ct.C / ct.mu) * ct.mu                                         # (S, n) Mbit/s

    # stale rates and lags (index < 0 -> earliest available rate_hat[0])
    R = np.maximum(ks[:, None] - np.arange(N_RATE_LAGS)[None, :], 0)
    rates = r_hat[R] / R_BAR                                                             # (S, NR, n)
    rate_age = np.minimum(_rate_age(window, ks, K, n, Delta), AGE_CAP_S) / AGE_SCALE_S   # (S, n)

    stat = _static_features(ct)                                                          # (n, 7)
    c_t = np.concatenate([
        np.broadcast_to(stat[None], (S, n, stat.shape[1])),
        counters.transpose(0, 2, 1),
        np.broadcast_to(counter_age[:, None, None], (S, n, 1)),
        (f / ct.mu)[:, :, None],
        rates.transpose(0, 2, 1),
        rate_age[:, :, None],
    ], axis=2)
    tod = (_tod_start(window) + ks * Delta) % SECONDS_PER_DAY
    ang = 2.0 * np.pi * tod / SECONDS_PER_DAY
    rh = r_hat[ks]
    load = (f / (rh * mech.B)).sum(-1)
    c_g = np.stack([np.sin(ang), np.cos(ang), load], axis=1)
    assert c_t.shape == (S, n, N_TENANT_FEATURES) and c_g.shape == (S, N_GLOBAL_FEATURES)
    return np.ascontiguousarray(c_t), c_g, f, rh


def context(window, k: int, d: int, contract: Contract, *, forecaster: OperatorAR | None = None,
            mech: Mechanism | None = None, **forbidden) -> Context:
    """Report-free context of interval ``k`` (INTERFACES.md §9).

    Args:
        window: a data ``Window`` (or a report-free view of one). Only ``measured``, ``rate_hat``, ``meta``
            (time of day, optional ``rate_age_s``) and ``id`` are read; ``theta`` is never touched.
        k: interval index (0-based).
        d: measurement delay in intervals (>= 1); ``measured[j]`` is visible only for ``j <= k - d``.
        contract: public contract (its ``mu`` normalises demands).
        forecaster: operator AR forecast (required; fitted for horizon ``d``, e.g.
            ``Calibration.operator_forecaster(d)``). The untrained :meth:`OperatorAR.default` must be passed
            explicitly (tests); there is no silent default (round-1 review, finding 5).
        mech: mechanism constants (``B``, ``Delta``); default ``Mechanism()``.

    Returns:
        :class:`Context` = ``(c_tenant (n, F), c_global (G,))`` with attributes ``f, r_hat, mu, k, d``.

    Raises:
        ReportLeakError: if reports or report-dependent quantities are passed (keyword or log table).
    """
    _assert_report_free(window, forbidden)
    d = _check_d(d)
    mech = mech or Mechanism()
    forecaster = _require_forecaster(forecaster)
    c_t, c_g, f, rh = _features(window, np.array([int(k)]), d, contract, forecaster, mech)
    return Context(c_t[0], c_g[0], f=f[0], r_hat=rh[0], mu=np.asarray(contract.mu, float), k=k, d=d,
                   window_id=getattr(window, "id", None))


@dataclass(frozen=True)
class WindowFeatures:
    """All report-free contexts of a window: ``c_tenant (K, n, F)``, ``c_global (K, G)``, ``f (K, n)`` Mbit/s,
    ``r_hat (K, n)`` Mbit/s per PRB. ``.context(k)`` equals ``features.context(window, k, d, contract)``."""

    c_tenant: np.ndarray
    c_global: np.ndarray
    f: np.ndarray
    r_hat: np.ndarray
    mu: np.ndarray
    d: int
    window_id: object = None

    @property
    def K(self) -> int:
        return self.c_tenant.shape[0]

    def context(self, k: int) -> Context:
        return Context(self.c_tenant[k], self.c_global[k], f=self.f[k], r_hat=self.r_hat[k], mu=self.mu,
                       k=k, d=self.d, window_id=self.window_id)


def window_features(window, d: int, contract: Contract, *, forecaster: OperatorAR | None = None,
                    mech: Mechanism | None = None, **forbidden) -> WindowFeatures:
    """Vectorised :func:`context` for every interval of a window (same values, one pass; forecaster required)."""
    _assert_report_free(window, forbidden)
    d = _check_d(d)
    mech = mech or Mechanism()
    forecaster = _require_forecaster(forecaster)
    K = _measured(window).shape[0]
    c_t, c_g, f, rh = _features(window, np.arange(K), d, contract, forecaster, mech)
    return WindowFeatures(c_t, c_g, f, rh, np.asarray(contract.mu, float), d, getattr(window, "id", None))
