"""Static fitted priorities (INTERFACES.md §9; experiment_design.md §4 "Other controllers").

One weight vector ``w`` (n,) is fitted on TRAINING windows to maximise realized welfare
``W = sum_i pi_i on_time_Mbit_i`` (credits per scored interval, pooled over windows) with truthful reports,
quotas from the contract allocator with the stale rate, and the fluid replay with true arrivals and rates.

* n = 3: exhaustive search over a floored simplex grid
  ``w = floor + (1 - n floor) (a, b, c) / steps`` (equal weights evaluated first, so ties keep equal weights);
* n > 3: cross-entropy method over log-weights (population 64, 20 iterations, 8 elites, elitist).

The fitted controller is report-independent (``FixedMap``), hence monotone; it is priced by the envelope.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..config import Mechanism
from .nets import softmax_np
from .ppo import check_splits, default_quota_fn, fluid_env_factory, prepare, quotas


def simplex_grid(n: int = 3, steps: int = 20, floor: float = 0.05) -> np.ndarray:
    """Floored simplex grid (P, n), rows summing to 1, equal weights first, duplicates removed.

    ``floor`` is the smallest weight on the grid (strictly positive weights, as softmax maps require).
    """
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not 0 < floor < 1.0 / n:
        raise ValueError(f"floor must lie in (0, 1/n) = (0, {1.0 / n:.4f})")
    import itertools
    rows = [np.full(n, 1.0 / n)]
    for c in itertools.product(range(steps + 1), repeat=n - 1):
        if sum(c) <= steps:
            v = np.array(c + (steps - sum(c),), float) / steps
            rows.append(floor + (1 - n * floor) * v)
    out, seen = [], set()
    for r in rows:
        key = tuple(np.round(r, 12))
        if key not in seen:
            seen.add(key); out.append(r)
    return np.asarray(out)


def rollout_fixed(data, W: np.ndarray, *, mech: Mechanism, env_factory, quota_fn) -> np.ndarray:
    """Realized welfare (credits per scored interval, pooled over windows) of each fixed weight vector in
    ``W (P, n)``. Each candidate is replayed over every window from empty queues (its own backlog path);
    candidates with identical quota sequences on a window are replayed once."""
    W = np.asarray(W, float)
    if W.ndim != 2 or np.any(~np.isfinite(W)) or np.any(W <= 0):
        raise ValueError("candidate weights must be a finite positive (P, n) array")
    P, n = W.shape
    acc = np.zeros(P); cnt = 0
    for a, p in enumerate(data):
        if p.n != n:
            raise ValueError(f"window {a} has n={p.n}, weights have n={n}")
        env = env_factory(p.window, mech)
        Q = quotas(quota_fn, p.z_alloc[None], W[:, None, :], p.r_hat[None], p.contract, mech)   # (P, K, n)
        uq, inv = np.unique(Q.reshape(P, -1), axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        ks = p.scored
        val = np.array([float((np.asarray(env.replay(q.reshape(p.K, n)), float)[ks] @ p.pi).sum()) for q in uq])
        acc += val[inv]
        cnt += len(ks)
    return acc / cnt


@dataclass
class StaticFit:
    """Fitted static priority vector and its search record."""

    w: np.ndarray                    # (n,) weights, sum 1
    welfare: float                   # TRAIN realized welfare, credits per scored interval
    method: str
    candidates: np.ndarray           # (P, n) every evaluated weight vector
    scores: np.ndarray               # (P,) their welfare
    seconds: float = 0.0
    info: dict = field(default_factory=dict)
    alloc_input = "reports"
    uses_context_features = False

    def priority_map(self, ctx=None, z=None):
        """``FixedMap(w)`` (report- and context-independent)."""
        from ..model.scores import FixedMap
        return FixedMap(self.w.copy())

    def alloc_reports(self, ctx, z):
        return np.asarray(z, float)


def fit_static(windows, mech: Mechanism | None = None, *, method: str = "auto", steps: int = 20,
               floor: float = 0.05, pop: int = 64, iters: int = 20, elite: int = 8, sigma0: float = 1.0,
               sigma_min: float = 0.05, seed: int = 0, report_source="theta", env_factory: Callable | None = None,
               quota_fn: Callable | None = None, allow_splits=("train",)) -> StaticFit:
    """Fit a static priority vector on training windows (see module docstring).

    Args:
        windows: training windows (truthful reports via ``report_source='theta'``); their split must be in
            ``allow_splits`` (default 'train' only; round-1 review, finding 8). An in-sample fit on evaluation
            windows (a diagnostic upper reference, e.g. ``eval.headroom``) must say so with ``allow_splits``.
        method: 'grid' (n = 3 default), 'cem' (n > 3 default) or 'auto'.
        steps, floor: simplex grid resolution and weight floor.
        pop, iters, elite, sigma0, sigma_min: CEM population, iterations, elite count, initial and minimum
            standard deviation of the log-weights.
        seed: CEM sampling seed.
    """
    mech = mech or Mechanism()
    env_factory = env_factory or fluid_env_factory
    quota_fn = quota_fn or default_quota_fn
    windows = list(windows)
    if not windows:
        raise ValueError("no windows to fit on")
    check_splits(windows, allow_splits, "static-fit")
    data = prepare(windows, mech=mech, report_source=report_source)
    n = data[0].n
    if method == "auto":
        method = "grid" if n == 3 else "cem"
    t0 = time.perf_counter()
    if method == "grid":
        C = simplex_grid(n, steps, floor)
        S = rollout_fixed(data, C, mech=mech, env_factory=env_factory, quota_fn=quota_fn)
        best = int(np.argmax(S))                      # first maximum: equal weights win ties
        return StaticFit(C[best], float(S[best]), "grid", C, S, time.perf_counter() - t0,
                         dict(steps=steps, floor=floor))
    if method != "cem":
        raise ValueError(f"unknown method {method!r}")
    if not 1 <= elite < pop:
        raise ValueError("need 1 <= elite < pop")
    rng = np.random.default_rng(int(seed))
    mean, sd = np.zeros(n), np.full(n, float(sigma0))
    cands, scores = [], []
    best_y, best_s = np.zeros(n), -np.inf
    for it in range(iters):
        Y = mean + sd * rng.standard_normal((pop, n))
        Y[0] = mean                                   # current mean (equal weights at iteration 0)
        if it > 0:
            Y[1] = best_y                             # elitism: incumbent is always re-evaluated
        W = softmax_np(Y)
        S = rollout_fixed(data, W, mech=mech, env_factory=env_factory, quota_fn=quota_fn)
        cands.append(W); scores.append(S)
        order = np.argsort(-S, kind="stable")
        if S[order[0]] > best_s:
            best_s, best_y = float(S[order[0]]), Y[order[0]].copy()
        el = Y[order[:elite]]
        mean = el.mean(0)
        sd = np.maximum(el.std(0), sigma_min)
    return StaticFit(softmax_np(best_y), best_s, "cem", np.concatenate(cands), np.concatenate(scores),
                     time.perf_counter() - t0, dict(pop=pop, iters=iters, elite=elite, sigma0=sigma0))
