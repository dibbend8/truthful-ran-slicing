"""Per-tenant ridge AR(p) forecasters (default p = 6) for any number of tenants.

Port of ``ICC_rebuild_2026-10-10/src/rebuild/forecast.ARForecast/fit_ar/training_pairs`` generalised from 3 tenants
to n, with *direct* h-step regressions and separate input/target series:

    y_k  =  mean + scale * ( c_0 + sum_{l=1..p} c_l * (x_{k-h-l+1} - mean) / scale )      (per tenant)

where ``y`` is the target series (e.g. measured demand m, Mbit/s), ``x`` the input series the forecaster can see
(``x = y`` for the operator's forecast on released counters with horizon h = d; for a tenant's report, ``x`` is its
own zero-delay history and h = 1), lag 1 = newest available input. Pairs are built inside each window only (never
across windows); a pair containing a NaN input is dropped. ``mean``/``scale`` are the TRAIN mean and standard
deviation of the target. Forecasts are truncated below at 0 Mbit/s (demand cannot be negative); there is no upper
clipping - report ceilings are applied by the tenant strategies, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import io as sio


def _as_windows(series, name) -> list[np.ndarray]:
    out = []
    for s in series:
        a = np.asarray(s, float)
        if a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2:
            raise ValueError(f"{name}: each window must be (K,) or (K, n)")
        out.append(a)
    if not out:
        raise ValueError(f"{name}: no windows")
    if len({a.shape[1] for a in out}) != 1:
        raise ValueError(f"{name}: windows disagree on the tenant count")
    return out


def lagged_pairs(targets, inputs=None, history: int = 6, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Training pairs inside each window: ``X (N, p, n)`` inputs oldest-first, ``Y (N, n)`` targets.

    For target index k the lags are ``inputs[k-h-p+1 .. k-h]``. Pairs never cross windows. Rows whose lags or
    target contain NaN are kept with NaN; :func:`fit_ar` drops them per tenant.
    """
    if history < 1 or horizon < 1:
        raise ValueError("history and horizon must be >= 1")
    T = _as_windows(targets, "targets")
    X = T if inputs is None else _as_windows(inputs, "inputs")
    if len(X) != len(T) or any(x.shape != t.shape for x, t in zip(X, T)):
        raise ValueError("inputs and targets must have identical window shapes")
    xs, ys = [], []
    for x, t in zip(X, T):
        K = len(t)
        first = history + horizon - 1          # first target index with a complete lag vector
        if K <= first:
            continue
        win = np.lib.stride_tricks.sliding_window_view(x, history, axis=0)   # (K-p+1, n, p)
        lags = win[: K - first].transpose(0, 2, 1)                             # lags for k = first .. K-1
        xs.append(lags)
        ys.append(t[first:])
    if not xs:
        raise ValueError("no window is longer than history + horizon - 1")
    return np.concatenate(xs), np.concatenate(ys)


@dataclass(frozen=True)
class ARForecast:
    """Fitted per-tenant AR model. ``coef`` (n, p+1): intercept, then lag-1 (newest) .. lag-p coefficients."""

    coef: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    horizon: int
    meta: dict = field(default_factory=dict)

    @property
    def history(self) -> int:
        return self.coef.shape[1] - 1

    @property
    def n(self) -> int:
        return self.coef.shape[0]

    def predict(self, lags) -> np.ndarray:
        """Forecast (Mbit/s, ``(..., n)``) from lags ``(..., p, n)`` ordered oldest first (last row = newest)."""
        x = np.asarray(lags, float)
        if x.shape[-2:] != (self.history, self.n):
            raise ValueError(f"lags must have trailing shape ({self.history}, {self.n}), got {x.shape}")
        z = (x[..., ::-1, :] - self.mean) / self.scale                 # newest first
        y = self.coef[:, 0] + np.einsum("...ln,nl->...n", z, self.coef[:, 1:])
        return np.maximum(self.mean + self.scale * y, 0.0)

    def forecast_series(self, inputs) -> np.ndarray:
        """Forecast for every target index k of one window from ``inputs (K, n)``: uses ``inputs[k-h-p+1..k-h]``.

        Where the lags are unavailable (index < 0 or NaN) the training mean is returned (documented fallback).
        """
        x = np.asarray(inputs, float)
        if x.ndim != 2 or x.shape[1] != self.n:
            raise ValueError(f"inputs must be (K, {self.n})")
        K, p, h = len(x), self.history, self.horizon
        out = np.broadcast_to(self.mean, (K, self.n)).copy()
        first = p + h - 1
        if K > first:
            lags = np.lib.stride_tricks.sliding_window_view(x, p, axis=0)[: K - first].transpose(0, 2, 1)
            pred = self.predict(np.nan_to_num(lags, nan=0.0))
            ok = ~np.isnan(lags).any(axis=1)
            out[first:] = np.where(ok, pred, out[first:])
        return out

    def to_dict(self) -> dict:
        return dict(coef=self.coef.tolist(), mean=self.mean.tolist(), scale=self.scale.tolist(),
                    horizon=self.horizon, meta=self.meta)

    @classmethod
    def from_dict(cls, d: dict) -> "ARForecast":
        return cls(np.asarray(d["coef"], float), np.asarray(d["mean"], float), np.asarray(d["scale"], float),
                   int(d["horizon"]), dict(d.get("meta", {})))

    def save(self, path):
        """Write as JSON to a confined path."""
        return sio.write_json(path, self.to_dict())


def fit_ar(targets, inputs=None, history: int = 6, horizon: int = 1, ridge: float = 1e-3,
           meta: dict | None = None, *, splits) -> ARForecast:
    """Fit independent per-tenant ridge AR(p) regressions (intercept unpenalised) on TRAIN windows.

    ``targets``/``inputs``: lists of ``(K, n)`` arrays (one per window; ``inputs=None`` means inputs = targets).
    ``splits``: the split label of every window (required, keyword-only); every label must be ``'train'``. The
    tenant forecaster produces the URLLC/mMTC truthful reports of every split, so fitting it on DEV or TEST series
    would leak them (round-2 review: bare arrays carried no split and could not be checked).
    """
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and >= 0")
    T = _as_windows(targets, "targets")
    labels = [str(x) for x in splits]
    if len(labels) != len(T):
        raise ValueError(f"need one split label per window ({len(T)}), got {len(labels)}")
    if any(x != "train" for x in labels):
        raise ValueError(f"fit_ar uses TRAIN windows only; got splits {sorted(set(labels))}")
    allt = np.concatenate(T)
    mean = np.nanmean(allt, axis=0)
    scale = np.nanstd(allt, axis=0)
    scale = np.maximum(scale, np.maximum(np.abs(mean) * 1e-6, 1e-12))
    X, Y = lagged_pairs(T, inputs, history, horizon)
    n = Y.shape[1]
    coef = np.zeros((n, history + 1))
    used = []
    for i in range(n):
        ok = np.isfinite(Y[:, i]) & np.isfinite(X[:, :, i]).all(1)
        if ok.sum() <= history + 1:
            raise ValueError(f"tenant {i}: only {int(ok.sum())} complete training pairs")
        a = np.column_stack([np.ones(ok.sum()), (X[ok, ::-1, i] - mean[i]) / scale[i]])
        b = (Y[ok, i] - mean[i]) / scale[i]
        pen = np.diag(np.r_[0.0, np.full(history, ridge)])
        coef[i] = np.linalg.solve(a.T @ a / len(a) + pen, a.T @ b / len(a))
        used.append(int(ok.sum()))
    m = dict(meta or {}, history=history, horizon=horizon, ridge=ridge, pairs=used, windows=len(T))
    return ARForecast(coef, mean, scale, horizon, m)
