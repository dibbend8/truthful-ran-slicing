"""PPO as a contextual bandit for priority networks (INTERFACES.md §9; experiment_design.md §4).

One allocation interval is one bandit step (no bootstrapped value). State = (report-free context c_k,
reports z_k); action = score vector ``y ~ N(y_theta(c, z), sigma^2 I)`` (learned scalar log sigma, initial 0.5,
projected onto [0.05, 1] after every step); weights ``w = softmax(y)``; quotas ``Q = F(z, w)`` from the contract
allocator with the stale rate ``rate_hat``; reward (lookahead return of horizon h)

    R = sum_{j=0..h_k} sum_i pi_i * on_time_Mbit_i[k+j]  /  sum_i pi_i mu_i Delta      (dimensionless)

from the fluid replay with TRUE per-ms arrivals and TRUE rates, started from the backlog state of the
policy's own rollout: the sampled quota is played in interval k and the deterministic path's quotas in
intervals k+1 .. k+h_k, ``h_k = min(h, K-1-k)`` (``sim.fluid.lookahead_window``). Because the context is
exogenous, those later quotas are exactly what the deployed policy plays, so ``R`` is the change of the
window's realized welfare truncated after h intervals.

Why a lookahead (round-1 review, finding 3): with carry-over backlog and deadlines longer than one interval
(video 2002 ms), the on-time value of interval k alone ranks quota changes by the wrong sign in about half of
the moves on light windows (sign agreement 0.40-0.49 with the window welfare change at load 0.64-0.71; 0.79-0.90
with h = 3). The default horizon is ``h = ceil(max(delay_ms) / (1000 Delta))`` (3 for the default contracts),
which covers every bit that arrives in interval k. ``PPOConfig(horizon=0)`` restores the one-interval reward of
INTERFACES.md §9 literally (gamma = 0, ``sim.fluid.counterfactual`` semantics). Each iteration:

1. roll out ``windows_per_iter`` training windows with the deterministic mean action (the deployed
   policy), recording the replay state at every interval start;
2. at every scored interval (k >= warmup) draw M action samples and replay that interval (and the h_k
   continuation intervals of the deterministic path) counterfactually from the recorded state;
3. advantage = R - mean_M R (per state), divided by its batch standard deviation; no critic;
4. clipped-surrogate PPO (clip 0.2, 4 passes, minibatch 2048 samples, Adam lr 3e-4 -> 3e-5 linear,
   grad-norm clip 1, entropy 1e-3).

Fast path. The context is exogenous (no backlog, quota, price or delivery features; condition C2), so the
deterministic path's quotas of a whole window are known before replaying it. Step 1 is therefore one
``fluid.replay`` call per window and step 2 one ``fluid.lookahead_window`` call per window (h = 0:
``fluid.counterfactual_window``); both are bit-for-bit equal to chaining ``run_interval`` from each
interval-start state.

Data guards (round-1 review, findings 5 and 8):
* the operator forecaster that builds the contexts must be given explicitly (``forecaster=``) or carried by the
  net (``net.forecaster``, set by :func:`train` and persisted by ``nets.save_net``); there is no silent default.
  ``features.OperatorAR.default()`` passed explicitly is the opt-in for tests;
* :func:`train` accepts training windows from the 'train' split only and development windows from the 'dev'
  split only, unless ``allow_splits`` names other splits (``retrain`` passes it for E5 test-date fine-tuning).

Every ``eval_every`` iterations the deterministic policy is evaluated on development windows (realized
welfare, credits per interval) and the best checkpoint is kept. Deployment is the deterministic mean.

Integration points (everything else is self-contained):
* ``quota_fn(z, w, r, ct, mech) -> Q`` defaults to ``slicemech.model.allocation.integer`` (``fractional`` if
  ``mech.integer`` is False); it is called with fully broadcast arrays of one shape ``(..., n)``;
* ``env_factory(window, mech) -> ReplayEnv`` defaults to :class:`FluidEnv` over ``slicemech.sim.fluid``.

This module also hosts the report-source resolution and the replay helpers used by ``static``, ``oracle``
and ``retrain``.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
import torch
from torch import nn

from ..config import Contract, Mechanism
from . import features as _feat
from .nets import DTYPE, PriorityNet, clone_net, softmax_np

MS_PER_S = 1000   # 1-ms slots per second (numerology 0)
ADV_STD_RTOL = 1e-12   # advantage spread below this (relative to |R|) is rounding noise: skip the update


# ============================================================================= replay environment
class ReplayEnv(Protocol):
    """Replay of one window's TRUE per-ms arrivals and TRUE per-PRB rates under quotas held per interval.

    Window-level (PPO, static, evaluation):
      ``replay(Q (K', n)) -> on_time (K', n)`` Mbit: intervals 0..K'-1 from empty queues; remembers the
      interval-start states of that path.
      ``counterfactual_at(ks (S,), Q (S, P, n)) -> on_time (S, P, n)`` Mbit: interval ks[s] replayed with each
      of P quota vectors from the start state of the last ``replay`` path.
    Stepwise (oracle): ``reset()``; ``counterfactual(Q (P, n)) -> (P, n)`` from the current state (no change);
      ``advance(Q (n,)) -> (n,)`` plays interval ``k`` and moves to ``k + 1``.
    """

    k: int

    def replay(self, Q: np.ndarray) -> np.ndarray: ...
    def counterfactual_at(self, ks: np.ndarray, Q: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...
    def counterfactual(self, Q_batch: np.ndarray) -> np.ndarray: ...
    def advance(self, Q: np.ndarray) -> np.ndarray: ...


class FluidEnv:
    """:class:`ReplayEnv` over ``slicemech.sim.fluid`` (INTERFACES.md §7), FIFO with deadline expiry.

    Lookahead extensions (PPO reward and oracle objective, round-1 review finding 3):
    ``lookahead_at(ks (S,), Q (S, M, n), h) -> on_time (S, M, h+1, n)``: interval ks[s] with each quota vector,
    then the last ``replay`` path's quotas for the next ``min(h, K'-1-ks[s])`` intervals (zeros beyond);
    ``lookahead(Q (P, n), Q_next (h', n)) -> (P, h'+1, n)``: the same from the current stepwise state.
    """

    def __init__(self, window, mech: Mechanism):
        from ..sim import fluid
        self._fluid = fluid
        self.r_true = np.asarray(window.rate_true, float)
        if self.r_true.ndim != 2 or not np.all(np.isfinite(self.r_true)) or np.any(self.r_true < 0):
            raise ValueError("window.rate_true must be a finite nonnegative (K, n) array (Mbit/s per PRB)")
        self.K, self.n = self.r_true.shape
        self.delay_ms = np.asarray(window.contract.delay_ms, float)
        self.slots = int(round(mech.Delta * MS_PER_S))
        if abs(self.slots - mech.Delta * MS_PER_S) > 1e-9:
            raise ValueError("Delta must be a whole number of 1-ms slots")
        self.A = fluid.cumulative_arrivals(window.arrivals_ms)          # (T_ms, n) float64, inclusive
        if self.A.shape != (self.A.shape[0], self.n) or self.A.shape[0] < self.K * self.slots:
            raise ValueError(f"arrivals_ms must cover K * {self.slots} ms for n={self.n} tenants, "
                             f"got shape {self.A.shape}")
        self._c_start = None
        self._Q_path = None
        self.state, self.k = None, 0

    # window level
    def replay(self, Q):
        Q = np.asarray(Q)
        out, c_start, _ = self._fluid.replay(self.A, Q, self.r_true[:len(Q)], self.delay_ms,
                                             slots_per_interval=self.slots)
        self._c_start = c_start
        self._Q_path = np.array(Q, dtype=float)
        return out.on_time_Mbit

    def counterfactual_at(self, ks, Q):
        if self._c_start is None:
            raise RuntimeError("counterfactual_at needs a preceding replay() path")
        ks = np.asarray(ks, dtype=np.int64)
        if ks.ndim != 1 or np.any(ks < 0) or np.any(ks >= len(self._c_start)):
            raise IndexError(f"counterfactual intervals must be a 1-D array in [0, {len(self._c_start)})")
        out = self._fluid.counterfactual_window(self.A, ks * self.slots, self._c_start[ks], np.asarray(Q),
                                                self.r_true[ks], self.delay_ms, slots_per_interval=self.slots)
        return out.on_time_Mbit

    def lookahead_at(self, ks, Q, h: int):
        """On-time Mbit (S, M, h+1, n): interval ks[s] with ``Q[s, m]``, then the replay path for up to h intervals."""
        if self._c_start is None:
            raise RuntimeError("lookahead_at needs a preceding replay() path")
        ks = np.asarray(ks, dtype=np.int64)
        Kp = len(self._c_start)
        if ks.ndim != 1 or np.any(ks < 0) or np.any(ks >= Kp):
            raise IndexError(f"lookahead intervals must be a 1-D array in [0, {Kp})")
        h = int(h)
        if h < 0:
            raise ValueError("lookahead horizon must be >= 0")
        steps = np.minimum(h, Kp - 1 - ks)
        nxt = np.minimum(ks[:, None] + np.arange(1, h + 1)[None, :], Kp - 1)          # (S, h); padded rows unused
        return self._fluid.lookahead_window(self.A, ks * self.slots, self._c_start[ks], np.asarray(Q),
                                            self.r_true[ks], self._Q_path[nxt], self.r_true[nxt], self.delay_ms,
                                            steps=steps, slots_per_interval=self.slots)

    # stepwise
    def reset(self):
        self.state, self.k = self._fluid.initial_state(self.n, self.delay_ms), 0

    def _slice(self):
        if self.k >= self.K:
            raise IndexError("replay finished")
        return self.A[self.k * self.slots:(self.k + 1) * self.slots]

    def counterfactual(self, Q_batch):
        out = self._fluid.counterfactual(self.state, self._slice(), np.asarray(Q_batch), self.r_true[self.k],
                                         self.delay_ms)
        return out.on_time_Mbit

    def lookahead(self, Q_batch, Q_next):
        """On-time Mbit (P, h'+1, n) from the current state: ``Q_batch[p]`` now, then ``Q_next`` (h', n)."""
        Q_next = np.asarray(Q_next, dtype=float)
        h = Q_next.shape[0]
        if self.k + h >= self.K:
            raise IndexError(f"lookahead of {h} intervals from interval {self.k} runs past the window end K={self.K}")
        A = self.A[self.k * self.slots:(self.k + 1 + h) * self.slots]
        return self._fluid.counterfactual_path(self.state, A, np.asarray(Q_batch), self.r_true[self.k], Q_next,
                                               self.r_true[self.k + 1:self.k + 1 + h], self.delay_ms,
                                               slots_per_interval=self.slots)

    def advance(self, Q):
        self.state, out = self._fluid.run_interval(self.state, self._slice(), np.asarray(Q), self.r_true[self.k],
                                                   self.delay_ms)
        self.k += 1
        return out.on_time_Mbit


def auto_horizon(contract: Contract, mech: Mechanism) -> int:
    """Lookahead horizon covering every bit that arrives in the changed interval:
    ``ceil(max(delay_ms) / (1000 Delta))`` intervals (3 for the 2002-ms video deadline at Delta = 1 s)."""
    return int(math.ceil(float(np.max(contract.delay_ms)) / (1000.0 * float(mech.Delta)) - 1e-12))


def resolve_horizon(horizon, contract: Contract, mech: Mechanism) -> int:
    """``horizon`` (int >= 0) or :func:`auto_horizon` when it is None."""
    if horizon is None:
        return auto_horizon(contract, mech)
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError(f"lookahead horizon must be None (auto) or an integer >= 0, got {horizon!r}")
    return int(horizon)


SPLITS = ("train", "dev", "test")


def check_splits(windows, allowed, role: str) -> None:
    """Raise unless every window's ``split`` is in ``allowed`` (round-1 review, finding 8)."""
    allowed = tuple(allowed)
    if not allowed or any(a not in SPLITS for a in allowed):
        raise ValueError(f"allowed splits must be a non-empty subset of {SPLITS}, got {allowed}")
    bad = sorted({str(getattr(w, "split", None)) for w in windows} - set(allowed))
    if bad:
        raise ValueError(f"{role} windows must come from split(s) {allowed}; got split(s) {bad}. Pass allow_splits "
                         "explicitly only for a pre-registered use (e.g. E5 test-date fine-tuning via retrain)")


def resolve_forecaster(forecaster, net, d: int | None = None):
    """The operator forecaster for contexts: the explicit one, else ``net.forecaster``; raise when neither exists.

    There is no silent default (round-1 review, finding 5): pass ``features.OperatorAR.default()`` explicitly to
    use the untrained mean-of-6 forecaster (tests), or a fitted ``features.fit_operator_ar`` (calibration).
    """
    fc = forecaster if forecaster is not None else getattr(net, "forecaster", None)
    if fc is None:
        raise ValueError("no operator forecaster: pass forecaster= (e.g. Calibration.operator_forecaster(d)) or "
                         "use a net that carries one (net.forecaster); features.OperatorAR.default() must be "
                         "passed explicitly")
    if not isinstance(fc, _feat.OperatorAR):
        raise TypeError(f"forecaster must be a features.OperatorAR, got {type(fc).__name__}")
    if d is not None and fc.horizon is not None and fc.horizon != d:
        raise ValueError(f"forecaster was fitted for horizon {fc.horizon}, used with d={d}")
    return fc


def fluid_env_factory(window, mech: Mechanism) -> FluidEnv:
    return FluidEnv(window, mech)


def default_quota_fn(z, w, r, ct: Contract, mech: Mechanism):
    """Contract allocator quotas (PRB): ``allocation.integer`` (or ``fractional`` when ``mech.integer`` is False)."""
    from ..model import allocation
    fn = allocation.integer if mech.integer else allocation.fractional
    return fn(z, w, r, ct, mech)


def quotas(quota_fn, z, w, r, ct: Contract, mech: Mechanism) -> np.ndarray:
    """Broadcast ``z, w, r`` to one shape ``(..., n)``, call ``quota_fn`` and check the output shape."""
    shape = np.broadcast_shapes(np.shape(z), np.shape(w), np.shape(r))
    z, w, r = (np.ascontiguousarray(np.broadcast_to(a, shape)) for a in (z, w, r))
    Q = np.asarray(quota_fn(z, w, r, ct, mech))
    if Q.shape != shape:
        raise ValueError(f"quota_fn returned shape {Q.shape}, expected {shape}")
    return Q


# ============================================================================= report sources
def resolve_reports(window, source, contract: Contract, index: int | None = None):
    """Report matrix ``z (K, n)`` (Mbit/s) used as allocator/network input during training or evaluation.

    ``source``: ``'theta'`` (truthful reports carried by the window), ``'measured'`` (rebuilt from released
    measured demand, ``z_k = m_k``), a callable ``window -> (K, n)``, or a sequence of arrays aligned with the
    window list (stored reports; ``index`` selects the entry). Reports are projected onto the report space
    ``[0, C]`` exactly as the contract prescribes (config.Contract: reports are clipped to [0, C]); the number
    of projected entries is returned so callers can log it. Raises on non-finite or negative entries or on a
    shape mismatch.

    Returns ``(z, n_projected)``.
    """
    if isinstance(source, str):
        if source == "theta":
            z = np.asarray(window.theta, float)
        elif source == "measured":
            z = np.asarray(window.measured, float)
        else:
            raise ValueError(f"unknown report source {source!r}")
    elif callable(source):
        z = np.asarray(source(window), float)
    else:
        if index is None:
            raise ValueError("a sequence of stored reports needs the window index")
        z = np.asarray(source[index], float)
    K = np.asarray(window.measured).shape[0]
    if z.shape != (K, contract.n):
        raise ValueError(f"reports must have shape {(K, contract.n)}, got {z.shape}")
    if not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("reports must be finite and nonnegative (Mbit/s)")
    over = z > contract.C
    return np.minimum(z, contract.C), int(over.sum())


@dataclass
class Prepared:
    """Per-window arrays for replay. ``z_alloc`` is the allocator input (reports, or the forecast ``f``)."""

    window: object
    contract: Contract
    K: int
    n: int
    warmup: int
    pi: np.ndarray
    norm: float                      # sum_i pi_i mu_i Delta (credits per interval)
    z_alloc: np.ndarray              # (K, n) Mbit/s
    zn: np.ndarray | None            # (K, n) z / mu for the network (None if the scores ignore reports)
    feats: object | None             # features.WindowFeatures
    r_hat: np.ndarray                # (K, n) Mbit/s per PRB
    n_projected: int = 0
    horizon: int = 0                 # lookahead intervals of the training reward (see module docstring)

    @property
    def scored(self) -> np.ndarray:
        return np.arange(self.warmup, self.K)


def prepare(windows, *, mech: Mechanism, net: PriorityNet | None = None, d: int | None = None,
            report_source="theta", forecaster=None, horizon: int | None = 0) -> list[Prepared]:
    """Prepare windows for replay. With ``net`` given, report-free features are computed (needs ``d`` and an
    operator forecaster: ``forecaster`` or ``net.forecaster``, see :func:`resolve_forecaster`). Reports are
    resolved only when needed: a measurement-only net never touches any report source. ``horizon`` is the
    lookahead of the training reward (None: :func:`auto_horizon` of each window's contract)."""
    if net is not None:
        forecaster = resolve_forecaster(forecaster, net, d)
    out = []
    for idx, w in enumerate(windows):
        ct = w.contract
        K = np.asarray(w.measured).shape[0]
        warmup = int(w.warmup)
        if not 0 <= warmup < K:
            raise ValueError(f"window {getattr(w, 'id', idx)}: warmup {warmup} outside [0, K={K})")
        feats = None
        if net is not None:
            if d is None:
                raise ValueError("d (measurement delay, intervals) is required to build contexts")
            feats = _feat.window_features(w, d, ct, forecaster=forecaster, mech=mech)
            r_hat = feats.r_hat
        else:
            r_hat = _feat._rate_hat(w, K, ct.n)
        nproj = 0
        if net is not None and net.alloc_input == "forecast":
            z_alloc, zn = feats.f, None
        else:
            z_alloc, nproj = resolve_reports(w, report_source, ct, idx)
            zn = z_alloc / ct.mu if (net is not None and net.uses_reports_in_scores) else None
        out.append(Prepared(window=w, contract=ct, K=K, n=ct.n, warmup=warmup, pi=np.asarray(ct.pi, float),
                            norm=float(np.sum(ct.pi * ct.mu) * mech.Delta), z_alloc=z_alloc, zn=zn,
                            feats=feats, r_hat=r_hat, n_projected=nproj,
                            horizon=resolve_horizon(horizon, ct, mech)))
    return out


# ============================================================================= rollout
@dataclass
class Rollout:
    c_t: np.ndarray | None          # (S, n, F) contexts of the scored states
    c_g: np.ndarray | None          # (S, G)
    zn: np.ndarray | None           # (S, n)
    y: np.ndarray | None            # (S, M, n) sampled scores
    logp: np.ndarray | None         # (S, M) log-probabilities under the rollout policy
    R: np.ndarray | None            # (S, M) normalised counterfactual rewards
    welfare: np.ndarray             # (n_windows,) deterministic-path realized welfare, credits per scored interval
    reward: np.ndarray              # (n_windows,) same, divided by sum pi mu Delta
    scored: np.ndarray              # (n_windows,) number of scored intervals
    quotas: list | None = None      # deterministic-path quotas per window, (K, n) PRB


def gaussian_logp(y, mean, log_sigma):
    """log N(y; mean, sigma^2 I) summed over the last axis (torch)."""
    n = y.shape[-1]
    return (-0.5 * ((y - mean) / log_sigma.exp()).square().sum(-1)
            - n * log_sigma - 0.5 * n * math.log(2 * math.pi))


def _stack(data, attr):
    return torch.as_tensor(np.concatenate([getattr(p.feats, attr) for p in data]), dtype=DTYPE)


def rollout(net: PriorityNet, data: Sequence[Prepared], *, mech: Mechanism, env_factory, quota_fn,
            M: int = 0, log_sigma: torch.Tensor | None = None, gen: torch.Generator | None = None,
            keep_quotas: bool = False) -> Rollout:
    """Replay ``data`` under the deterministic mean policy; with ``M > 0`` also draw M Gaussian score samples
    per scored interval and replay each counterfactually from that interval's start state on the path: the
    sample's quota at k, then the path's quotas for ``min(p.horizon, K-1-k)`` intervals (lookahead return)."""
    if not data:
        raise ValueError("no windows to roll out")
    n = data[0].n
    if any(p.n != n for p in data):
        raise ValueError("all windows in a rollout batch must have the same number of tenants")
    if M > 0 and (log_sigma is None or gen is None):
        raise ValueError("sampling needs log_sigma and a torch Generator")
    with torch.no_grad():                                   # one forward over every interval of every window
        c_t, c_g = _stack(data, "c_tenant"), _stack(data, "c_global")
        zn = (torch.as_tensor(np.concatenate([p.zn for p in data]), dtype=DTYPE)
              if data[0].zn is not None else None)
        Y = net(c_t, c_g, zn)                               # (sum K, n)
    welfare = np.zeros(len(data)); cnt = np.zeros(len(data), dtype=np.int64)
    buf = {k: [] for k in ("idx", "y", "logp", "R")}
    kept = []
    off = 0
    for a, p in enumerate(data):
        y = Y[off:off + p.K]
        env = env_factory(p.window, mech)
        Q = quotas(quota_fn, p.z_alloc, softmax_np(y.numpy()), p.r_hat, p.contract, mech)     # (K, n)
        on = np.asarray(env.replay(Q), float)                                                # (K, n)
        ks = p.scored
        welfare[a], cnt[a] = float((on[ks] @ p.pi).mean()), len(ks)
        if keep_quotas:
            kept.append(Q)
        if M > 0:
            with torch.no_grad():
                eps = torch.randn((len(ks), M, n), generator=gen, dtype=DTYPE)
                mean = y[torch.as_tensor(ks)][:, None, :]
                ys = mean + log_sigma.exp() * eps                                           # (S, M, n)
                lp = gaussian_logp(ys, mean, log_sigma)                                     # (S, M)
            Qs = quotas(quota_fn, p.z_alloc[ks][:, None, :], softmax_np(ys.numpy()), p.r_hat[ks][:, None, :],
                        p.contract, mech)                                                   # (S, M, n)
            if p.horizon == 0:
                on_cf = np.asarray(env.counterfactual_at(ks, Qs), float)                    # (S, M, n)
            else:
                if not hasattr(env, "lookahead_at"):
                    raise TypeError(f"{type(env).__name__} has no lookahead_at(); a lookahead reward (horizon "
                                    f"{p.horizon}) needs it, or use PPOConfig(horizon=0)")
                on_cf = np.asarray(env.lookahead_at(ks, Qs, p.horizon), float).sum(axis=2)  # (S, M, n)
            buf["R"].append(on_cf @ p.pi / p.norm)
            buf["idx"].append(off + ks); buf["y"].append(ys.numpy()); buf["logp"].append(lp.numpy())
        off += p.K
    reward = welfare / np.array([p.norm for p in data])
    if M == 0:
        return Rollout(None, None, None, None, None, None, welfare, reward, cnt, kept or None)
    idx = torch.as_tensor(np.concatenate(buf["idx"]))
    return Rollout(c_t[idx].numpy(), c_g[idx].numpy(), zn[idx].numpy() if zn is not None else None,
                   np.concatenate(buf["y"]), np.concatenate(buf["logp"]), np.concatenate(buf["R"]),
                   welfare, reward, cnt, kept or None)


# ============================================================================= PPO
@dataclass(frozen=True)
class PPOConfig:
    """Hyper-parameters (experiment_design.md §4; spec/params.yaml ``training``)."""

    iterations: int = 400
    windows_per_iter: int = 32
    M: int = 4                       # counterfactual action samples per state
    clip: float = 0.2
    passes: int = 4
    minibatch: int = 2048            # state-action samples per gradient step
    lr_start: float = 3e-4
    lr_end: float = 3e-5             # linear decay over iterations
    grad_clip: float = 1.0
    entropy_coef: float = 1e-3
    sigma0: float = 0.5
    sigma_min: float = 0.05
    sigma_max: float = 1.0
    eval_every: int = 50             # DEV evaluation period (0: never; keep the final parameters)
    single_thread: bool = True       # torch.set_num_threads(1) during training (bit-reproducibility)
    horizon: int | None = None       # reward lookahead (intervals); None: auto_horizon(contract); 0: one interval

    def __post_init__(self):
        if self.iterations < 1 or self.windows_per_iter < 1 or self.M < 2 or self.passes < 1 or self.minibatch < 1:
            raise ValueError("iterations, windows_per_iter, passes, minibatch >= 1 and M >= 2 required")
        if self.horizon is not None and (isinstance(self.horizon, bool) or not isinstance(self.horizon, int)
                                         or self.horizon < 0):
            raise ValueError("horizon must be None (auto) or an integer >= 0")
        if not 0 < self.sigma_min <= self.sigma0 <= self.sigma_max:
            raise ValueError("need 0 < sigma_min <= sigma0 <= sigma_max")
        if self.lr_start <= 0 or self.lr_end <= 0 or self.clip <= 0 or self.grad_clip <= 0:
            raise ValueError("learning rates, clip and grad_clip must be > 0")


@dataclass
class PPOResult:
    net: PriorityNet                 # selected checkpoint (best DEV welfare, else final)
    log_sigma: float                 # exploration log-std at the selected checkpoint
    best_iteration: int              # 1-based iteration of the selected checkpoint
    best_dev_welfare: float | None   # credits per interval
    history: list = field(default_factory=list)
    final_net: PriorityNet | None = None
    n_projected: int = 0             # training report entries projected onto [0, C] (contract rule)


class _Threads:
    def __init__(self, on: bool):
        self.on, self.prev = on, None

    def __enter__(self):
        if self.on:
            self.prev = torch.get_num_threads(); torch.set_num_threads(1)

    def __exit__(self, *exc):
        if self.on and self.prev is not None:
            torch.set_num_threads(self.prev)


def _update(net: PriorityNet, log_sigma: nn.Parameter, opt, roll: Rollout, cfg: PPOConfig, gen) -> dict:
    """Clipped-surrogate passes over one rollout. Returns diagnostics."""
    S, M, n = roll.y.shape
    adv = roll.R - roll.R.mean(axis=1, keepdims=True)
    std = float(adv.std())
    tied = float(np.mean(np.ptp(roll.R, axis=1) == 0))
    if std <= ADV_STD_RTOL * max(1.0, float(np.abs(roll.R).mean())):
        # every interval inert (or differences at rounding level only): no policy-gradient signal
        return dict(skipped=True, adv_std=std, frac_tied=tied, clip_frac=0.0, approx_kl=0.0)
    adv = torch.as_tensor(adv / (std + 1e-12), dtype=DTYPE)
    c_t = torch.as_tensor(roll.c_t, dtype=DTYPE); c_g = torch.as_tensor(roll.c_g, dtype=DTYPE)
    zn = torch.as_tensor(roll.zn, dtype=DTYPE) if roll.zn is not None else None
    ys = torch.as_tensor(roll.y, dtype=DTYPE); old = torch.as_tensor(roll.logp, dtype=DTYPE)
    params = list(net.parameters()) + [log_sigma]
    per = max(1, cfg.minibatch // M)
    lo, hi = math.log(cfg.sigma_min), math.log(cfg.sigma_max)
    clip_frac, kl, steps = 0.0, 0.0, 0
    for _ in range(cfg.passes):
        perm = torch.randperm(S, generator=gen)
        for s0 in range(0, S, per):
            idx = perm[s0:s0 + per]
            mean = net(c_t[idx], c_g[idx], zn[idx] if zn is not None else None)
            lp = gaussian_logp(ys[idx], mean[:, None, :], log_sigma)
            ratio = (lp - old[idx]).exp()
            A = adv[idx]
            surr = torch.minimum(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
            entropy = n * (0.5 + 0.5 * math.log(2 * math.pi) + log_sigma)
            loss = -surr - cfg.entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            with torch.no_grad():
                log_sigma.clamp_(lo, hi)
                clip_frac += float(((ratio - 1).abs() > cfg.clip).double().mean())
                kl += float((old[idx] - lp).mean())
            steps += 1
    return dict(skipped=False, adv_std=std, frac_tied=tied, clip_frac=clip_frac / steps, approx_kl=kl / steps)


def train(net: PriorityNet, windows, dev_windows=None, *, d: int, mech: Mechanism | None = None,
          config: PPOConfig | None = None, seed: int = 0, report_source="theta", dev_report_source="theta",
          env_factory: Callable | None = None, quota_fn: Callable | None = None, forecaster=None,
          init_log_sigma: float | None = None, verbose: bool = False, allow_splits=("train",)) -> PPOResult:
    """Train a copy of ``net`` with PPO on ``windows`` (the input net is not modified).

    Args:
        net: priority network (``nets.make_net``); its parameters are the starting point.
        windows: training windows (or report-free views, see ``retrain``); their split must be in
            ``allow_splits`` (default: 'train' only).
        dev_windows: development windows for checkpoint selection (None: keep the final parameters); split
            'dev' only.
        d: measurement delay in intervals (context uses ``measured[j]``, ``j <= k - d``).
        report_source: report input for training (see :func:`resolve_reports`); ignored by meas-only nets.
        forecaster: operator AR for the contexts (fitted for horizon ``d``); default ``net.forecaster``; raises
            when neither is set. The trained net carries it (``result.net.forecaster``).
        seed: seeds window sampling (numpy), action noise and minibatch order (torch). Same seed and inputs
            give bit-identical parameters.
        init_log_sigma: starting log-std (default ``log(config.sigma0)``).
        allow_splits: splits accepted for ``windows`` (pre-registered exceptions only, e.g. E5 retraining on
            test dates through ``retrain``).

    Returns:
        :class:`PPOResult` with the selected checkpoint and a per-iteration history (train reward, welfare in
        credits per interval, sigma, lr, PPO diagnostics, DEV welfare at evaluation points).
    """
    cfg = config or PPOConfig()
    mech = mech or Mechanism()
    env_factory = env_factory or fluid_env_factory
    quota_fn = quota_fn or default_quota_fn
    windows = list(windows)
    check_splits(windows, allow_splits, "training")
    if dev_windows:
        dev_windows = list(dev_windows)
        check_splits(dev_windows, ("dev",), "development (checkpoint selection)")
    forecaster = resolve_forecaster(forecaster, net, d)
    with _Threads(cfg.single_thread):
        net = clone_net(net)
        net.forecaster = forecaster
        data = prepare(windows, mech=mech, net=net, d=d, report_source=report_source, forecaster=forecaster,
                       horizon=cfg.horizon)
        dev = (prepare(dev_windows, mech=mech, net=net, d=d, report_source=dev_report_source,
                       forecaster=forecaster) if dev_windows else None)
        gen = torch.Generator().manual_seed(int(seed))
        rng = np.random.default_rng(int(seed))
        ls0 = math.log(cfg.sigma0) if init_log_sigma is None else float(init_log_sigma)
        log_sigma = nn.Parameter(torch.tensor(ls0, dtype=DTYPE))
        opt = torch.optim.Adam(list(net.parameters()) + [log_sigma], lr=cfg.lr_start)
        best_state, best_val, best_it, best_ls = None, -math.inf, cfg.iterations, ls0
        history = []
        for it in range(cfg.iterations):
            t0 = time.perf_counter()
            frac = it / (cfg.iterations - 1) if cfg.iterations > 1 else 0.0
            lr = cfg.lr_start + (cfg.lr_end - cfg.lr_start) * frac
            for g in opt.param_groups:
                g["lr"] = lr
            sel = np.sort(rng.choice(len(data), size=min(cfg.windows_per_iter, len(data)), replace=False))
            roll = rollout(net, [data[i] for i in sel], mech=mech, env_factory=env_factory, quota_fn=quota_fn,
                           M=cfg.M, log_sigma=log_sigma.detach(), gen=gen)
            diag = _update(net, log_sigma, opt, roll, cfg, gen)
            rec = dict(iteration=it + 1, lr=lr, sigma=float(log_sigma.detach().exp()), states=int(roll.R.shape[0]),
                       horizon=int(max(data[i].horizon for i in sel)),
                       sample_reward=float(roll.R.mean()), welfare=float(np.average(roll.welfare, weights=roll.scored)),
                       reward=float(np.average(roll.reward, weights=roll.scored)), **diag)
            if dev is not None and cfg.eval_every and ((it + 1) % cfg.eval_every == 0 or it + 1 == cfg.iterations):
                ev = rollout(net, dev, mech=mech, env_factory=env_factory, quota_fn=quota_fn)
                val = float(np.average(ev.welfare, weights=ev.scored))
                rec["dev_welfare"] = val
                if val > best_val:
                    best_val, best_it = val, it + 1
                    best_state = copy.deepcopy(net.state_dict()); best_ls = float(log_sigma.detach())
            rec["seconds"] = time.perf_counter() - t0
            history.append(rec)
            if verbose:
                print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()})
        final = clone_net(net)
        if best_state is not None:
            net.load_state_dict(best_state)
        else:
            best_ls = float(log_sigma.detach())
        net.eval()
    return PPOResult(net=net, log_sigma=best_ls, best_iteration=best_it,
                     best_dev_welfare=(best_val if best_state is not None else None), history=history,
                     final_net=final, n_projected=sum(p.n_projected for p in data))


def evaluate(net: PriorityNet, windows, *, d: int, mech: Mechanism | None = None, report_source="theta",
             env_factory: Callable | None = None, quota_fn: Callable | None = None, forecaster=None) -> dict:
    """Deterministic deployment of ``net`` on ``windows`` (truthful reports by default). The operator forecaster
    is ``forecaster`` or ``net.forecaster`` (required, see :func:`resolve_forecaster`).

    Returns ``dict(welfare=credits per scored interval (pooled), reward=welfare / sum pi mu Delta,
    per_window=(n_windows,) welfare, scored=(n_windows,) counts)``.
    """
    mech = mech or Mechanism()
    data = prepare(windows, mech=mech, net=net, d=d, report_source=report_source, forecaster=forecaster)
    ev = rollout(net, data, mech=mech, env_factory=env_factory or fluid_env_factory,
                 quota_fn=quota_fn or default_quota_fn)
    return dict(welfare=float(np.average(ev.welfare, weights=ev.scored)),
                reward=float(np.average(ev.reward, weights=ev.scored)), per_window=ev.welfare, scored=ev.scored)
