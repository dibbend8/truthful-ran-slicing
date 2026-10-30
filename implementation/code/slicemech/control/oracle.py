"""Hindsight oracle (INTERFACES.md §9; experiment_design.md §4): per-interval CEM over priority weights.

For every interval k of a window, starting from the oracle's own replay state (true backlog), a
cross-entropy search over log-weights ``y`` maximises the realized welfare ``sum_i pi_i on_time_Mbit_i``
(credits) of a short lookahead: the candidate's quotas in interval k, then the quotas of a *base policy* (fixed
weights ``base_weights``, equal by default) in intervals k+1 .. k+h (``h = horizon``, default
``ppo.auto_horizon``: 3 for the 2002-ms video deadline), all evaluated by the counterfactual fluid replay with
the TRUE arrivals and TRUE rates (``sim.fluid.counterfactual_path``). Quotas come from the same contract
allocator as every other controller (truthful reports, stale rate ``rate_hat`` by default; ``rate='true'``
gives the true-rate diagnostic). The best weights are played and the state advances.

This is a rollout (one-step lookahead with a base-policy continuation): with a full horizon it is never worse
than its base policy, so passing the fitted static weights as ``base_weights`` makes it an improvement of the
static arm. With a truncated horizon that guarantee is approximate, and it is still not a dynamic-programming
optimum. ``horizon=0`` is the earlier myopic (one-interval) objective, which ranks quota changes by the wrong
sign in about half of the moves on light windows (round-1 review, finding 3); report that variant only as a
"myopic reference".

Search per interval: candidate 0 = equal weights, candidate 1 = previous interval's best (warm start),
candidate 2 = the base weights (when given); the rest are Gaussian samples around the running mean. Identical
quota vectors are replayed once. If every candidate of the first iteration yields the same quotas (weights inert,
Prop. 4), the search stops and the equal weights are kept. Ties are resolved towards the earliest candidate
(equal weights first).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..config import Mechanism
from .nets import softmax_np
from .ppo import default_quota_fn, fluid_env_factory, prepare, resolve_horizon
from .ppo import quotas as _quotas


@dataclass
class OracleResult:
    """Oracle trajectory of one window (all arrays over intervals k = 0..K-1)."""

    window_id: object
    weights: np.ndarray       # (K, n)
    quotas: np.ndarray        # (K, n) PRB
    on_time: np.ndarray       # (K, n) Mbit
    welfare: np.ndarray       # (K,) credits
    scored: np.ndarray        # (K,) bool, k >= warmup
    evaluations: np.ndarray   # (K,) distinct quota vectors replayed
    horizon: int = 0          # lookahead intervals of the per-interval objective
    base_weights: np.ndarray | None = None   # (n,) continuation policy of the lookahead

    @property
    def mean_welfare(self) -> float:
        """Credits per scored interval."""
        return float(self.welfare[self.scored].mean())

    def controller(self) -> "OracleController":
        return OracleController(self)


class OracleController:
    """Replays the oracle weights: ``priority_map(ctx)`` -> ``FixedMap(weights[ctx.k])`` (reference only)."""

    alloc_input = "reports"
    uses_context_features = False

    def __init__(self, result: OracleResult):
        self.result = result

    def priority_map(self, ctx, z=None):
        from ..model.scores import FixedMap
        wid = getattr(ctx, "window_id", None)
        if wid is not None and self.result.window_id is not None and wid != self.result.window_id:
            raise ValueError(f"oracle computed for window {self.result.window_id!r}, asked for {wid!r}")
        return FixedMap(self.result.weights[ctx.k].copy())

    def alloc_reports(self, ctx, z):
        return np.asarray(z, float)


def _seed_for(window, seed: int) -> np.random.Generator:
    wid = str(getattr(window, "id", ""))
    return np.random.default_rng([int(seed), zlib.crc32(wid.encode())])


def oracle_window(window, mech: Mechanism | None = None, *, pop: int = 64, iters: int = 8, elite: int = 8,
                  sigma0: float = 1.5, sigma_min: float = 0.05, seed: int = 0, rate: str = "hat",
                  report_source="theta", env_factory: Callable | None = None,
                  quota_fn: Callable | None = None, horizon: int | None = None, base_weights=None) -> OracleResult:
    """Per-interval hindsight CEM on one window (see module docstring).

    Args:
        window: data Window (needs ``theta``, ``arrivals_ms``, ``rate_true``, ``rate_hat``).
        pop, iters, elite: CEM population per iteration, iterations per interval, elite count.
        sigma0, sigma_min: initial and minimum std of the log-weights.
        rate: 'hat' (allocator uses rate_hat, as all controllers) or 'true' (rate_true; diagnostic).
        seed: combined with the window id for a per-window deterministic stream.
        horizon: lookahead intervals (None: ``ppo.auto_horizon`` of the contract; 0: myopic objective).
        base_weights: (n,) positive weights of the continuation policy (default equal weights), e.g. the
            fitted static weights; the env must implement ``lookahead`` when ``horizon > 0``.
    """
    mech = mech or Mechanism()
    env_factory = env_factory or fluid_env_factory
    quota_fn = quota_fn or default_quota_fn
    if not 1 <= elite < pop or pop < 2 or iters < 1:
        raise ValueError("need pop >= 2, 1 <= elite < pop, iters >= 1")
    if not (isinstance(report_source, str) or callable(report_source)) and len(report_source) != 1:
        raise ValueError("oracle_window takes one stored report array (a length-1 sequence); "
                         "use oracle_windows for several windows")
    data = prepare([window], mech=mech, report_source=report_source)
    p = data[0]
    if rate == "true":
        r_alloc = np.asarray(window.rate_true, float)
        if not np.all(np.isfinite(r_alloc)) or np.any(r_alloc <= 0):
            raise ValueError("rate_true must be finite and > 0")
    elif rate == "hat":
        r_alloc = p.r_hat
    else:
        raise ValueError("rate must be 'hat' or 'true'")
    K, n = p.K, p.n
    h = resolve_horizon(horizon, p.contract, mech)
    if base_weights is None:
        y_base = np.zeros(n)
    else:
        bw = np.asarray(base_weights, dtype=float)
        if bw.shape != (n,) or not np.all(np.isfinite(bw)) or np.any(bw <= 0):
            raise ValueError(f"base_weights must be a finite positive (n,) = ({n},) array")
        y_base = np.log(bw / bw.max())
    Q_base = (_quotas(quota_fn, p.z_alloc, softmax_np(y_base)[None, :], r_alloc, p.contract, mech)
              if h > 0 else None)                                            # (K, n) continuation quotas
    rng = _seed_for(window, seed)
    env = env_factory(window, mech); env.reset()
    if h > 0 and not hasattr(env, "lookahead"):
        raise TypeError(f"{type(env).__name__} has no lookahead(); use horizon=0 for the myopic objective")
    weights = np.zeros((K, n)); quotas = np.zeros((K, n)); on_time = np.zeros((K, n))
    welfare = np.zeros(K); evals = np.zeros(K, dtype=np.int64)
    prev_y = np.zeros(n)
    for k in range(K):
        mean, sd = prev_y.copy(), np.full(n, float(sigma0))
        best_y, best_s, best_Q = np.zeros(n), -np.inf, None
        n_eval = 0
        h_k = min(h, K - 1 - k)
        for it in range(iters):
            Y = mean + sd * rng.standard_normal((pop, n))
            Y[0] = 0.0                                # equal weights
            Y[1] = prev_y if it == 0 else best_y      # warm start / incumbent
            if base_weights is not None and pop > 2:
                Y[2] = y_base                         # the continuation policy itself
            W = softmax_np(Y)
            Q = _quotas(quota_fn, p.z_alloc[k], W, r_alloc[k], p.contract, mech)          # (pop, n)
            uq, inv = np.unique(Q, axis=0, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            if h_k > 0:
                on_u = np.asarray(env.lookahead(uq, Q_base[k + 1:k + 1 + h_k]), float).sum(axis=1)   # (U, n)
            else:
                on_u = np.asarray(env.counterfactual(uq), float)      # (U, n)
            s_u = on_u @ p.pi
            n_eval += len(uq)
            S = s_u[inv]
            order = np.argsort(-S, kind="stable")
            if S[order[0]] > best_s:
                best_s, best_y, best_Q = float(S[order[0]]), Y[order[0]].copy(), Q[order[0]].copy()
            if it == 0 and len(uq) == 1:              # weights inert in this interval
                break
            el = Y[order[:elite]]
            mean, sd = el.mean(0), np.maximum(el.std(0), sigma_min)
        on = np.asarray(env.advance(best_Q), float)
        weights[k], quotas[k], on_time[k] = softmax_np(best_y), best_Q, on
        welfare[k], evals[k] = float(on @ p.pi), n_eval
        prev_y = best_y
    scored = np.arange(K) >= p.warmup
    return OracleResult(getattr(window, "id", None), weights, quotas, on_time, welfare, scored, evals, h,
                        None if base_weights is None else softmax_np(y_base))


def oracle_windows(windows, mech: Mechanism | None = None, *, report_source="theta", **kw) -> list[OracleResult]:
    """:func:`oracle_window` for each window (independent per-window seeds). A sequence ``report_source``
    (stored reports aligned with ``windows``) is split so that window i uses entry i."""
    windows = list(windows)
    if isinstance(report_source, str) or callable(report_source):
        sources = [report_source] * len(windows)
    else:
        if len(report_source) != len(windows):
            raise ValueError(f"{len(report_source)} report arrays for {len(windows)} windows")
        sources = [[r] for r in report_source]
    return [oracle_window(w, mech, report_source=s, **kw) for w, s in zip(windows, sources)]
