"""Retraining pipelines (INTERFACES.md §9; experiment_design.md C1 and E5; theory_design.md Theorem 2, A2).

``retrain(net, windows, mode, stored_reports)`` fine-tunes a copy of a trained priority network with the PPO
contextual bandit on windows that have already been played.

* ``mode='measurement'`` (measurement-only retraining): every window is wrapped in a
  :class:`MeasurementView` that exposes only exogenous data (per-ms ingress, released measured demand, CQI
  rates, contract, clocks). Accessing ``theta`` or any report/outcome raises :class:`ReportAccessError`.
  The report input of the training states is rebuilt from released ingress (``report_rebuild``):

  - ``'tenant_ar'`` (default): what a truthful tenant would have reported, rebuilt without reports
    (:func:`rebuild_reports`). Video tenants: ``z_k = m_k``, the interval's ingress, which equals the truthful
    video report exactly when the video lead time is at least ``Delta + report lead`` (default 4.004 s >= 1.05 s;
    a shorter lead raises, because the persistence-forecast part cannot be rebuilt from ingress). URLLC/mMTC
    tenants: the public TRAIN tenant AR forecast (``Calibration.forecaster``) applied to the own-history series
    ``[j - ell, j + 1 - ell)`` rebuilt from the per-ms ingress, which reproduces ``theta`` bit for bit on composed
    windows. The deployed truthful reports of these tenants are AR forecasts, much smoother than ``m``
    (TRAIN report MAE / mu 0.50 for mMTC, 0.04 for URLLC), so ``z = m`` would train the measurement pipeline on
    inputs distributed unlike deployment and confound the E5 pipeline comparison (round-1 review, finding 9);
  - ``'measured'``: ``z_k = m_k`` for every tenant (the earlier rule; kept as an explicit variant).

  Reports are projected onto ``[0, C]``; quotas and rewards are recomputed by the counterfactual fluid replay
  of the policy's own rollout. Stored reports, logged quotas, prices and deliveries are never read, so the
  result is a deterministic function of exogenous data and the seed: bit-identical whatever tenants reported.
  Retraining is assumed to run after the last interval of ``windows`` plus the measurement delay, so every
  ``m_k`` and every ingress millisecond it uses has been released.
* ``mode='stored'`` (the channel the paper closes): the report input is the stored report log
  ``stored_reports[w] (K, n)`` (Mbit/s); everything else is identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Mechanism
from . import ppo as _ppo

BYTES_PER_MBIT = 125_000            # 1 Mbit = 1e6 bit = 125 000 bytes
REBUILD_MODES = ("tenant_ar", "measured")


class ReportAccessError(RuntimeError):
    """Raised when measurement-only retraining (or any report-free consumer) reaches for a report."""


_HIDDEN_META = ("report", "theta", "quota", "price", "deliver", "on_time", "served", "utility", "premium",
                "backlog", "stored")
#: Public configuration keys that pass the filter although their names match it (lead times are composition
#: constants, not reports).
_PUBLIC_META = frozenset({"report_lead_s", "lead_s"})


class MeasurementView:
    """Read-only, report-free view of a data ``Window``.

    Exposes ``id, split, date, n, cls, arrivals_ms, measured, rate_true, rate_hat, warmup, contract`` and a
    filtered ``meta`` (keys mentioning reports, quotas, prices, deliveries or utilities are removed).
    ``theta`` and every other attribute raise.
    """

    __slots__ = ("_w", "_meta")
    _EXPOSED = ("id", "split", "date", "n", "cls", "arrivals_ms", "measured", "rate_true", "rate_hat",
                "warmup", "contract")

    def __init__(self, window):
        if isinstance(window, MeasurementView):
            window = object.__getattribute__(window, "_w")
        object.__setattr__(self, "_w", window)
        meta = dict(getattr(window, "meta", None) or {})
        object.__setattr__(self, "_meta", {k: v for k, v in meta.items()
                                           if k in _PUBLIC_META
                                           or not any(h in str(k).lower() for h in _HIDDEN_META)})

    def __getattr__(self, name):
        if name in MeasurementView._EXPOSED:
            return getattr(object.__getattribute__(self, "_w"), name)
        if name == "meta":
            return object.__getattribute__(self, "_meta")
        if name == "theta" or "report" in name:
            raise ReportAccessError(f"measurement-only view refuses access to {name!r}")
        raise AttributeError(f"MeasurementView exposes no attribute {name!r}")

    def __setattr__(self, name, value):
        raise AttributeError("MeasurementView is read-only")

    def __repr__(self):
        return f"MeasurementView(id={self.id!r})"


def _ingress_bytes(arrivals_ms, K: int) -> tuple[np.ndarray, bool]:
    """Per-ms ingress (Mbit per ms, float32 as stored in ``Window.arrivals_ms``) -> integer bytes (int64).

    Composed windows store ``float32(bytes * 8e-6)``; for fewer than 8e6 bytes per ms the nearest integer recovers
    the bytes exactly. Returns ``(bytes, exact)``, ``exact`` = the recovered bytes reproduce the stored series.
    """
    a = np.asarray(arrivals_ms)
    if a.ndim != 2 or a.shape[0] % K != 0 or a.shape[0] == 0:
        raise ValueError(f"arrivals_ms must be (K * slots, n) with K = {K}, got {a.shape}")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("arrivals_ms must be finite and nonnegative")
    b = np.rint(a.astype(np.float64) * BYTES_PER_MBIT).astype(np.int64)
    exact = bool(np.array_equal((b * 8e-6).astype(a.dtype), a))
    return b, exact


def own_history_from_ingress(arrivals_ms, K: int, report_lead_s: float) -> tuple[np.ndarray, bool]:
    """Zero-delay own history ``h[j]`` = ingress over ``[j - ell, j + 1 - ell)`` (Mbit/s), j = 1..K-1; NaN at j = 0
    (it needs ``ell`` before the window start), as ``data.compose`` builds it for the truthful TRACTOR reports.

    Returns ``(h (K, n), exact)``; ``exact`` says the integer bytes were recovered exactly (then ``h`` equals the
    composer's own history bit for bit).
    """
    b, exact = _ingress_bytes(arrivals_ms, K)
    T = b.shape[0] // K
    ell = int(round(float(report_lead_s) * T))
    if abs(ell - float(report_lead_s) * T) > 1e-9 or not 0 <= ell < T:
        raise ValueError(f"report lead {report_lead_s} s must be a whole number of slots in [0, {T})")
    c = np.vstack([np.zeros((1, b.shape[1]), np.int64), np.cumsum(b, axis=0)])
    h = np.full((K, b.shape[1]), np.nan)
    j = np.arange(1, K)
    h[1:] = (c[T * (j + 1) - ell] - c[T * j - ell]) * 8e-6
    if ell == 0:
        h[0] = c[T] * 8e-6
    return h, exact


def rebuild_reports(window, tenant_forecaster, *, Delta: float = 1.0) -> tuple[np.ndarray, dict]:
    """Report input ``z (K, n)`` (Mbit/s) for measurement-only retraining, rebuilt from exogenous data only.

    Video tenants: ``z = measured`` (exact truthful report when ``meta['lead_s'] >= Delta + meta['report_lead_s']``;
    raises otherwise). URLLC/mMTC tenants: ``tenant_forecaster.forecast_series(own_history_from_ingress(...))``
    (the public TRAIN tenant AR; horizon 1). Reads ``arrivals_ms``, ``measured``, ``cls`` and ``meta`` (lead times)
    only; never ``theta`` or any report. Returns ``(z, info)`` with the per-class rule and the byte-exactness flag.
    """
    m = np.asarray(window.measured, dtype=float)
    K, n = m.shape
    cls = tuple(window.cls)
    meta = getattr(window, "meta", None) or {}
    if "report_lead_s" not in meta:
        raise KeyError("window.meta['report_lead_s'] (tenant report lead, s) is needed to rebuild reports")
    ell = float(meta["report_lead_s"])
    z = m.copy()
    info = dict(rule={}, exact_bytes=None)
    if "video" in cls:
        lead = meta.get("lead_s")
        if lead is None or float(lead) < Delta + ell - 1e-12:
            raise ValueError(f"video lead {lead} s < Delta + report lead = {Delta + ell} s: the truthful video report "
                             "contains a persistence forecast that ingress cannot rebuild")
        info["rule"]["video"] = "measured demand (exact next-interval ingress)"
    tract = [i for i, c in enumerate(cls) if c != "video"]
    if tract:
        if tenant_forecaster is None:
            raise ValueError("URLLC/mMTC tenants need the public tenant AR (Calibration.forecaster) to rebuild "
                             "their truthful-equivalent reports; or use report_rebuild='measured'")
        if tenant_forecaster.horizon != 1 or tenant_forecaster.n != n:
            raise ValueError("tenant forecaster must be a horizon-1 per-tenant AR over all n tenants")
        h, exact = own_history_from_ingress(window.arrivals_ms, K, ell)
        f = tenant_forecaster.forecast_series(h)
        z[:, tract] = f[:, tract]
        info["exact_bytes"] = exact
        for i in tract:
            info["rule"][cls[i]] = "tenant AR(6) on own history rebuilt from per-ms ingress"
    return z, info


@dataclass
class RetrainResult:
    net: object                  # retrained copy (the input net is unchanged)
    mode: str
    history: list
    log_sigma: float
    n_projected: int             # report entries projected onto [0, C] (contract rule)
    report_rebuild: str | None = None   # measurement mode: 'tenant_ar' or 'measured'


def retrain(net, windows, mode: str = "measurement", stored_reports=None, *, d: int,
            mech: Mechanism | None = None, config: "_ppo.PPOConfig | None" = None, seed: int = 0,
            env_factory=None, quota_fn=None, forecaster=None, init_log_sigma: float | None = None,
            report_rebuild: str = "tenant_ar", tenant_forecaster=None,
            allow_splits=("train", "dev", "test")) -> RetrainResult:
    """Fine-tune a copy of ``net`` on already-played ``windows`` (see module docstring).

    Args:
        net: trained priority network (unchanged; a copy is trained).
        windows: the played windows (data ``Window`` objects); retraining runs on played windows of any split
            (E5 fine-tunes on test dates), restricted by ``allow_splits``.
        mode: 'measurement' or 'stored'.
        stored_reports: per-window stored report logs ``(K, n)`` Mbit/s; required in 'stored' mode and
            NEVER read in 'measurement' mode.
        d: measurement delay (intervals) used for the contexts.
        forecaster: operator AR for the contexts (default ``net.forecaster``; required).
        report_rebuild: measurement mode only, 'tenant_ar' (default; needs ``tenant_forecaster``) or 'measured'.
        tenant_forecaster: the public TRAIN tenant AR (``Calibration.forecaster``) for ``'tenant_ar'``.
        config: PPO hyper-parameters; default = ``PPOConfig(iterations=50, eval_every=0,
            windows_per_iter=len(windows))`` (E5 fine-tuning, final parameters kept).
        seed: training seed (same seed and exogenous data -> bit-identical parameters).
    """
    mech = mech or Mechanism()
    windows = list(windows)
    if not windows:
        raise ValueError("no windows to retrain on")
    cfg = config or _ppo.PPOConfig(iterations=50, eval_every=0, windows_per_iter=len(windows))
    # no development windows are passed below, so the final parameters are always kept
    views = [MeasurementView(w) for w in windows]
    rebuild = None
    if mode == "measurement":
        del stored_reports                        # never read in this mode
        if report_rebuild not in REBUILD_MODES:
            raise ValueError(f"report_rebuild must be one of {REBUILD_MODES}, got {report_rebuild!r}")
        rebuild = report_rebuild
        if report_rebuild == "measured":
            source = "measured"                   # z_k = m_k, see ppo.resolve_reports
        else:
            if tenant_forecaster is None and any(c != "video" for c in views[0].cls):
                raise ValueError("report_rebuild='tenant_ar' needs tenant_forecaster (Calibration.forecaster)")
            Delta = float(mech.Delta)

            def source(view, _fc=tenant_forecaster, _D=Delta):
                return rebuild_reports(view, _fc, Delta=_D)[0]
    elif mode == "stored":
        if stored_reports is None:
            raise ValueError("stored mode needs stored_reports (one (K, n) array per window)")
        if len(stored_reports) != len(windows):
            raise ValueError(f"{len(stored_reports)} stored report logs for {len(windows)} windows")
        source = [np.asarray(s, float) for s in stored_reports]
    else:
        raise ValueError(f"mode must be 'measurement' or 'stored', got {mode!r}")
    res = _ppo.train(net, views, None, d=d, mech=mech, config=cfg, seed=seed, report_source=source,
                     env_factory=env_factory, quota_fn=quota_fn, forecaster=forecaster,
                     init_log_sigma=init_log_sigma, allow_splits=allow_splits)
    return RetrainResult(net=res.net, mode=mode, history=res.history, log_sigma=res.log_sigma,
                         n_projected=res.n_projected, report_rebuild=rebuild)
