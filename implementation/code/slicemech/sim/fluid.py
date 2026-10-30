"""Fluid FIFO-with-deadline replay of isolated slice queues (INTERFACES.md §7; experiment_design.md §3).

Model
-----
Each tenant ``i`` owns one downlink queue that is served once per 1-ms slot. During a supervisory interval the tenant
holds the quota ``Q_i`` (PRB, constant over the interval) at its true per-PRB rate ``r_i`` (Mbit/s per PRB), so it may
clear ``cap_i = r_i * Q_i / 1000`` Mbit per slot. Capacity a queue cannot use is lost (strict isolation, as with NR RBG
masks); it is reported as *unused capacity*.

The queue is represented by two cumulative curves over absolute slots ``tau = 0, 1, ...`` counted from the window start:

* ``A(tau)``: cumulative arrivals through slot ``tau`` inclusive (Mbit), with ``A(tau) = 0`` for ``tau < 0``;
* ``Cc``: cumulative cleared bits, i.e. served plus dropped (Mbit).

Bits that arrive in slot ``a`` can be served in slot ``a``. A bit that arrives in slot ``a`` and is served in slot ``s``
has delay ``s - a`` ms; it is *on time* iff ``s - a <= d_i``, where ``d_i`` is the delay budget in whole ms. Slot ``t``
does, in this order (FIFO, head first)::

    serve : s = min(A(t) - Cc, cap);    Cc += s        (served, on time)
    expire: e = max(0, A(t - d) - Cc);  Cc += e        (dropped: bits from slots <= t - d cannot be on time any more)

Because service is FIFO and expired bits leave at the head, every served bit is on time.

*Late-counted variant* (``late_counted=True``; only for comparison with 5G-LENA, which delivers late packets instead of
dropping them): nothing expires. The served range ``(Cc, Cc + s]`` is split at ``A(t - d - 1)``: the part at or below it
arrived more than ``d`` slots ago and is counted as *late*, the rest as on time.

Per interval and tenant the replay reports arrivals, on-time, late and dropped Mbit, the backlog at the start and end of
the interval and the unused capacity, all in Mbit. They satisfy

    arrivals = on_time + late + dropped + (backlog - backlog_start)      (to about 1e-12 Mbit; increments telescope)
    on_time + late + unused = cap * T                                     (to T * spacing(A), see below)

The curves are kept in absolute float64 coordinates so that Cc <= A(t), the expiry invariant Cc >= A(t - d) and the
state hand-over between intervals hold exactly; the price is that each slot's service is quantised to the float spacing
of the cumulative curve (about 9e-13 Mbit at A = 5000 Mbit), i.e. at most about 5e-10 Mbit per 1000-slot interval.

State carries over between intervals: ``FluidState`` holds the slot index, ``Cc`` and the last ``H = max(d) + 1`` values
of ``A``, which is all the history the deadline rule needs.

Entry points
------------
``run_interval``          one interval from a state (returns the next state) -- INTERFACES.md §7.
``counterfactual``        many quota vectors from one start state, one numba call -- INTERFACES.md §7.
``counterfactual_window`` many (start state, quota vector) pairs of one window in one numba call (PPO fast path).
``replay``                a whole window with a quota sequence, one numba call; returns every interval-start state.
``trace_interval``        per-slot curves for one interval (diagnostics and tests).
``served_delay_cdf``      served Mbit with delay <= delta, from inverting the cumulative curves (evaluation only).

Cumulative arrivals must be float64 (use ``cumulative_arrivals``): a float32 running sum loses about 1e-3 Mbit over a
200-s window. All kernels are deterministic and serial; numba compiles them on first use without an on-disk cache.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from numba import njit

__all__ = [
    "SLOTS_PER_S", "FluidState", "FluidOut", "delay_slots", "cumulative_arrivals", "initial_state", "state_at",
    "run_interval", "counterfactual", "counterfactual_window", "replay", "trace_interval", "served_delay_cdf",
]

SLOTS_PER_S = 1000          # 1-ms slots (numerology 0); cap per slot = r * Q / 1000 Mbit


# ---------------------------------------------------------------------------------------------------------------------
# numba kernels
#
# Every path (batch, replay, trace) advances a lane with the same two per-slot steps, so all entry points agree bit for
# bit. Increments are differences of the cleared curve, so they telescope and conservation holds to rounding; min/max
# keep Cc <= A(t) and every increment >= 0 exactly. The batch kernel interleaves the M quota vectors that share one
# arrival curve in its innermost loop, which LLVM vectorises.
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=False, inline="always")
def _step_expiry(c, cap, a_now, a_old):
    """One slot, expiry mode. c: cleared (Mbit); cap (Mbit/slot); a_now = A(t); a_old = A(t - d).

    Returns (cleared after the slot, on-time increment, dropped increment, unused-capacity increment), Mbit.
    """
    served = min(cap, a_now - c)              # service the slot can use (the backlog A(t) - c is >= 0)
    c_srv = min(c + cap, a_now)               # serve head first, never past the arrival curve
    c_new = max(c_srv, a_old)                 # expire bits from slots <= t - d
    return c_new, c_srv - c, c_new - c_srv, cap - served


@njit(cache=False, inline="always")
def _step_late(c, cap, a_now, a_thr):
    """One slot, late-counted mode (no expiry). a_thr = A(t - d - 1): served bits at or below it are late.

    Returns (cleared after the slot, on-time increment, late increment, unused-capacity increment), Mbit.
    """
    served = min(cap, a_now - c)
    c_srv = min(c + cap, a_now)
    split = min(max(a_thr, c), c_srv)         # served range (c, c_srv] = late part (c, split] + on-time (split, c_srv]
    return c_srv, c_srv - split, split - c, cap - served


@njit(cache=False)
def _batch_kernel(ext, base, T, c0, Q, r, d, late_counted, on, late, drop, unused, c_end):
    """Evaluate quota vector ``Q[s, m]`` from start state ``s`` for every (s, m).

    ext : (n, L) absolute cumulative arrivals; column ``base[s] + j`` holds A(t0_s + j), and the d_i + 1 columns before
          ``base[s]`` hold A(t0_s - 1) ... A(t0_s - d_i - 1).
    c0  : (S, n) cleared at the start states (Mbit); Q (PRB), r (Mbit/s per PRB): (S, M, n); d: (n,) slots.
    Outputs (S, M, n): on-time, late, dropped, unused (Mbit) and cleared at the interval end.
    """
    S, M, n = Q.shape
    c = np.empty(M)
    cap = np.empty(M)
    o = np.empty(M)
    x = np.empty(M)
    u = np.empty(M)
    for s in range(S):
        b = base[s]
        for i in range(n):
            a = ext[i]
            di = d[i]
            for m in range(M):
                c[m] = c0[s, i]
                cap[m] = r[s, m, i] * Q[s, m, i] / 1000.0
                o[m] = 0.0
                x[m] = 0.0
                u[m] = 0.0
            if late_counted:
                for j in range(T):
                    a_now = a[b + j]
                    a_thr = a[b + j - di - 1]
                    for m in range(M):
                        c_new, on_inc, late_inc, un_inc = _step_late(c[m], cap[m], a_now, a_thr)
                        c[m] = c_new
                        o[m] += on_inc
                        x[m] += late_inc
                        u[m] += un_inc
            else:
                for j in range(T):
                    a_now = a[b + j]
                    a_old = a[b + j - di]
                    for m in range(M):
                        c_new, on_inc, drop_inc, un_inc = _step_expiry(c[m], cap[m], a_now, a_old)
                        c[m] = c_new
                        o[m] += on_inc
                        x[m] += drop_inc
                        u[m] += un_inc
            for m in range(M):
                on[s, m, i] = o[m]
                unused[s, m, i] = u[m]
                c_end[s, m, i] = c[m]
                if late_counted:
                    late[s, m, i] = x[m]
                    drop[s, m, i] = 0.0
                else:
                    late[s, m, i] = 0.0
                    drop[s, m, i] = x[m]


@njit(cache=False)
def _replay_kernel(ext, base0, T, Q, r, d, late_counted, on, late, drop, unused, c_start, c_last):
    """Replay K consecutive intervals (Q, r: (K, n)) from empty queues; interval k starts at column base0 + k*T.

    Tenants are interleaved in the inner loop (independent dependency chains). Outputs (K, n) as in ``_batch_kernel``,
    plus the cleared Mbit at each interval start (c_start) and after the last interval (c_last, (n,)).
    """
    K, n = Q.shape
    c = np.zeros(n)
    cap = np.empty(n)
    o = np.empty(n)
    x = np.empty(n)
    u = np.empty(n)
    for k in range(K):
        b = base0 + k * T
        for i in range(n):
            c_start[k, i] = c[i]
            cap[i] = r[k, i] * Q[k, i] / 1000.0
            o[i] = 0.0
            x[i] = 0.0
            u[i] = 0.0
        for j in range(T):
            for i in range(n):
                if late_counted:
                    c_new, on_inc, x_inc, un_inc = _step_late(c[i], cap[i], ext[i, b + j], ext[i, b + j - d[i] - 1])
                else:
                    c_new, on_inc, x_inc, un_inc = _step_expiry(c[i], cap[i], ext[i, b + j], ext[i, b + j - d[i]])
                c[i] = c_new
                o[i] += on_inc
                x[i] += x_inc
                u[i] += un_inc
        for i in range(n):
            on[k, i] = o[i]
            unused[k, i] = u[i]
            if late_counted:
                late[k, i] = x[i]
                drop[k, i] = 0.0
            else:
                late[k, i] = 0.0
                drop[k, i] = x[i]
    for i in range(n):
        c_last[i] = c[i]


@njit(cache=False)
def _trace_kernel(ext, base, T, c0, Q, r, d, late_counted, tr_on, tr_late, tr_drop, tr_cleared):
    """Per-slot increments of one interval for each tenant (diagnostics): c0, Q, r (n,); traces (T, n), Mbit."""
    n = Q.shape[0]
    for i in range(n):
        c = c0[i]
        cap = r[i] * Q[i] / 1000.0
        for j in range(T):
            if late_counted:
                c, on_inc, late_inc, _ = _step_late(c, cap, ext[i, base + j], ext[i, base + j - d[i] - 1])
                drop_inc = 0.0
            else:
                c, on_inc, drop_inc, _ = _step_expiry(c, cap, ext[i, base + j], ext[i, base + j - d[i]])
                late_inc = 0.0
            tr_on[j, i] = on_inc
            tr_late[j, i] = late_inc
            tr_drop[j, i] = drop_inc
            tr_cleared[j, i] = c


# ---------------------------------------------------------------------------------------------------------------------
# state and outputs
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FluidState:
    """Queue state at the start of slot ``t`` (immutable; arrays are read-only copies).

    t       : absolute index of the next slot to simulate (slots since the window start).
    cleared : (n,) cumulative cleared bits ``Cc`` = served + dropped since the window start (Mbit).
    A_hist  : (H, n) cumulative arrivals ``A(t-H) .. A(t-1)`` (Mbit), zero for slots before the window start.
              ``H >= max(d) + 1`` for every delay budget ``d`` (slots) the state is used with.
    """
    t: int
    cleared: np.ndarray
    A_hist: np.ndarray

    def __post_init__(self):
        t = int(self.t)
        if t != self.t or t < 0:
            raise ValueError(f"FluidState.t must be a nonnegative integer slot index, got {self.t!r}")
        A = np.array(self.A_hist, dtype=np.float64, order="C")
        if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
            raise ValueError(f"FluidState.A_hist must have shape (H >= 1, n >= 1), got {A.shape}")
        H, n = A.shape
        c = np.array(self.cleared, dtype=np.float64)
        if c.shape != (n,):
            raise ValueError(f"FluidState.cleared must have shape ({n},), got {c.shape}")
        if not (np.isfinite(A).all() and np.isfinite(c).all()):
            raise ValueError("FluidState arrays must be finite")
        if np.any(A[0] < 0) or np.any(A[1:] < A[:-1]):
            raise ValueError("FluidState.A_hist must be a nonnegative, nondecreasing cumulative curve")
        if t < H and np.any(A[:H - t] != 0):
            raise ValueError("FluidState.A_hist must be zero for slots before the window start (tau < 0)")
        if np.any(c < 0) or np.any(c > A[-1]):
            raise ValueError("FluidState needs 0 <= cleared <= A(t-1) (nonnegative backlog)")
        A.setflags(write=False)
        c.setflags(write=False)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "A_hist", A)
        object.__setattr__(self, "cleared", c)

    @property
    def n(self) -> int:
        return self.A_hist.shape[1]

    @property
    def H(self) -> int:
        """History length (slots); the state supports delay budgets up to ``H - 1`` ms."""
        return self.A_hist.shape[0]

    @property
    def backlog(self) -> np.ndarray:
        """(n,) queued Mbit at the start of slot ``t``."""
        return self.A_hist[-1] - self.cleared


@dataclass(frozen=True, eq=False)
class FluidOut:
    """Per-interval replay outputs; every field has shape ``(..., n)`` and is in Mbit.

    arrivals_Mbit       bits that arrived during the interval.
    on_time_Mbit        bits served with delay <= d.
    late_Mbit           bits served with delay > d (only with ``late_counted=True``; zero otherwise).
    dropped_Mbit        bits expired at the deadline (zero with ``late_counted=True``).
    backlog_start_Mbit  queue at the interval start (identical across a counterfactual batch).
    backlog_Mbit        queue at the interval end.
    unused_Mbit         capacity ``cap * T`` that found an empty queue.
    """
    arrivals_Mbit: np.ndarray
    on_time_Mbit: np.ndarray
    late_Mbit: np.ndarray
    dropped_Mbit: np.ndarray
    backlog_start_Mbit: np.ndarray
    backlog_Mbit: np.ndarray
    unused_Mbit: np.ndarray

    @property
    def served_Mbit(self) -> np.ndarray:
        return self.on_time_Mbit + self.late_Mbit

    @property
    def capacity_Mbit(self) -> np.ndarray:
        """Capacity granted over the interval, ``r * Q * T / 1000`` (Mbit), up to rounding."""
        return self.on_time_Mbit + self.late_Mbit + self.unused_Mbit

    def conservation_residual(self) -> np.ndarray:
        """``arrivals - on_time - late - dropped - (backlog - backlog_start)``; zero up to rounding."""
        return (self.arrivals_Mbit - self.on_time_Mbit - self.late_Mbit - self.dropped_Mbit
                - (self.backlog_Mbit - self.backlog_start_Mbit))

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __getitem__(self, key: str) -> np.ndarray:
        """Dict-style access by field name, with or without the ``_Mbit`` suffix (e.g. ``out["on_time"]``)."""
        if not isinstance(key, str):
            raise TypeError("FluidOut is indexed by field name; index the field arrays for batch elements")
        name = key if key.endswith("_Mbit") else key + "_Mbit"
        if name not in {f.name for f in fields(self)}:
            raise KeyError(key)
        return getattr(self, name)


# ---------------------------------------------------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------------------------------------------------

def delay_slots(delay_ms, n: int) -> np.ndarray:
    """Delay budgets (ms, shape (n,)) as int64 slot counts; they must be nonnegative whole milliseconds."""
    d = np.asarray(delay_ms, dtype=np.float64)
    if d.shape != (n,):
        raise ValueError(f"delay_ms must have shape ({n},), got {d.shape}")
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("delay_ms must be finite and nonnegative")
    di = np.rint(d)
    if np.any(np.abs(d - di) > 1e-9):
        raise ValueError(f"delay_ms must be whole milliseconds (1-ms slots), got {d}")
    return di.astype(np.int64)


def cumulative_arrivals(arrivals_ms) -> np.ndarray:
    """Per-slot arrivals (T, n) in Mbit per ms (e.g. ``Window.arrivals_ms``) -> float64 cumulative curve A (T, n).

    ``A[t]`` is the cumulative arrivals through slot ``t`` inclusive; slice it per interval as ``A[k*T:(k+1)*T]``.
    """
    x = np.asarray(arrivals_ms)
    if x.ndim != 2:
        raise ValueError(f"arrivals_ms must have shape (T, n), got {x.shape}")
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("arrivals_ms must be finite and nonnegative")
    return np.cumsum(x, axis=0, dtype=np.float64)


def _cumulative(A, n, name) -> np.ndarray:
    """Validate a cumulative arrival curve of shape (T >= 1, n) and return it as float64."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] < 1 or (n is not None and A.shape[1] != n) or A.shape[1] < 1:
        want = f"(T >= 1, {n})" if n is not None else "(T >= 1, n)"
        raise ValueError(f"{name} must have shape {want}, got {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError(f"{name} must be finite")
    if np.any(A[1:] < A[:-1]):
        raise ValueError(f"{name} must be nondecreasing in time (cumulative arrivals)")
    return A


def _nonneg(x, name) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return x


def _quota_rate(Q, r_true, n):
    """Validate quotas Q (..., n) (PRB) and rates broadcastable to Q.shape (Mbit/s per PRB); float64 copies."""
    Q = _nonneg(Q, "Q")
    if Q.ndim < 1 or Q.shape[-1] != n:
        raise ValueError(f"quota array must have shape (..., {n}), got {Q.shape}")
    r = _nonneg(r_true, "r_true")
    try:
        r = np.broadcast_to(r, Q.shape)
    except ValueError:
        raise ValueError(f"r_true with shape {r.shape} does not broadcast to the quota shape {Q.shape}") from None
    return np.ascontiguousarray(Q), np.ascontiguousarray(r)


def _check_horizon(d, H):
    if d.max() + 1 > H:
        raise ValueError(f"delay budget {d.max()} ms needs a state history of {d.max() + 1} slots, state has {H}; "
                         "build the state with initial_state/state_at using the largest delay")


def _check_no_expired(cleared, A_deadline):
    """Expiry mode: at a slot boundary every bit from slots <= t-1-d has left, i.e. cleared >= A(t-1-d)."""
    if np.any(cleared < A_deadline):
        raise ValueError("state still queues bits older than the delay budget; it was produced with "
                         "late_counted=True or a larger delay_ms, which expiry mode cannot continue")


def _run_batch(ext, base, T, c0, Q, r, d, late_counted):
    """Allocate outputs and run ``_batch_kernel``; Q, r are (S, M, n). Returns a dict of (S, M, n) arrays."""
    S, M, n = Q.shape
    res = {k: np.empty((S, M, n)) for k in ("on", "late", "drop", "unused", "c_end")}
    _batch_kernel(ext, np.ascontiguousarray(base, dtype=np.int64), int(T), np.ascontiguousarray(c0), Q, r, d,
                  bool(late_counted), res["on"], res["late"], res["drop"], res["unused"], res["c_end"])
    return res


def _make_out(res, A_prev, A_end, c0):
    """Assemble a FluidOut from kernel results (S, M, n) and per-state A(t0-1), A(t0+T-1), cleared (S, n)."""
    shape = res["on"].shape
    A_prev = A_prev[:, None, :]
    A_end = A_end[:, None, :]
    return FluidOut(
        arrivals_Mbit=np.broadcast_to(A_end - A_prev, shape).copy(),
        on_time_Mbit=res["on"],
        late_Mbit=res["late"],
        dropped_Mbit=res["drop"],
        backlog_start_Mbit=np.broadcast_to(A_prev - c0[:, None, :], shape).copy(),
        backlog_Mbit=A_end - res["c_end"],
        unused_Mbit=res["unused"],
    )


def _reshape_out(out: FluidOut, shape) -> FluidOut:
    return FluidOut(**{k: v.reshape(shape) for k, v in out.as_dict().items()})


def _require_state(state):
    if not isinstance(state, FluidState):
        raise TypeError(f"state must be a FluidState, got {type(state).__name__}")


def _prepare_single(state, A_cum_slice, Q, r_true, delay_ms, late_counted):
    """Validate one start state with its interval inputs; build the (n, H + T) arrival array the kernels read."""
    _require_state(state)
    n, H = state.n, state.H
    A = _cumulative(A_cum_slice, n, "A_cum_slice")
    T = A.shape[0]
    if np.any(A[0] < state.A_hist[-1]):
        raise ValueError("A_cum_slice must continue the state's arrival curve: need A_cum_slice[0] >= A(t-1)")
    d = delay_slots(delay_ms, n)
    _check_horizon(d, H)
    if not late_counted:
        _check_no_expired(state.cleared, state.A_hist[H - 1 - d, np.arange(n)])
    Qf, rf = _quota_rate(Q, r_true, n)
    ext = np.empty((n, H + T))
    ext[:, :H] = state.A_hist.T
    ext[:, H:] = A.T
    return A, d, Qf, rf, ext


def _single_state(state, A_cum_slice, Q, r_true, delay_ms, late_counted):
    """Shared path of run_interval / counterfactual: one start state, a batch Q (..., n) of quota vectors."""
    A, d, Qf, rf, ext = _prepare_single(state, A_cum_slice, Q, r_true, delay_ms, late_counted)
    n, H = state.n, state.H
    res = _run_batch(ext, np.array([H]), A.shape[0], state.cleared[None, :], Qf.reshape(1, -1, n),
                     rf.reshape(1, -1, n), d, late_counted)
    out = _make_out(res, state.A_hist[-1][None, :], A[-1][None, :], state.cleared[None, :])
    return out, res, ext, Qf.shape


# ---------------------------------------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------------------------------------

def initial_state(n: int, delay_ms) -> FluidState:
    """Empty queues at the window start (slot 0), with history for delay budgets up to ``max(delay_ms)`` ms."""
    if int(n) != n or n < 1:
        raise ValueError(f"need a positive integer number of tenants, got {n!r}")
    d = delay_slots(delay_ms, int(n))
    H = int(d.max()) + 1
    return FluidState(0, np.zeros(int(n)), np.zeros((H, int(n))))


def state_at(A_cum, t: int, cleared, delay_ms) -> FluidState:
    """State at the start of slot ``t`` of a window with cumulative arrivals ``A_cum`` (T_w, n) and cleared Mbit.

    Rebuilds the arrival history from the window curve, e.g. from the interval-start ``cleared`` values that ``replay``
    returns, so that ``run_interval``/``counterfactual`` can be started from any interval of a rollout.
    """
    A = _cumulative(A_cum, None, "A_cum")
    T_w, n = A.shape
    d = delay_slots(delay_ms, n)
    H = int(d.max()) + 1
    t = int(t)
    if not 0 <= t <= T_w:
        raise ValueError(f"slot index t={t} outside [0, {T_w}]")
    hist = np.zeros((H, n))
    src = A[max(t - H, 0):t]
    hist[H - len(src):] = src
    return FluidState(t, cleared, hist)


def run_interval(state: FluidState, A_cum_slice, Q, r_true, delay_ms, *, late_counted: bool = False):
    """Replay one interval and return ``(next_state, out)`` (INTERFACES.md §7).

    state        : FluidState at the interval start (not modified).
    A_cum_slice  : (T, n) absolute cumulative arrivals A(t0) .. A(t0+T-1) (Mbit), e.g. ``A_cum[k*1000:(k+1)*1000]``
                   from ``cumulative_arrivals``; T slots of 1 ms (1000 for a 1-s interval).
    Q            : (n,) quota (PRB) held for the interval; integers in deployment, fractional values are read as
                   time-shared average PRBs.
    r_true       : (n,) true per-PRB rate in this interval (Mbit/s per PRB); capacity is r_true * Q / 1000 Mbit/slot.
    delay_ms     : (n,) delay budgets (whole ms), e.g. ``Contract.delay_ms``.
    late_counted : no expiry; late bits are served and counted in ``late_Mbit`` (5G-LENA comparison only).
    Returns the state at slot t0+T and a FluidOut whose fields have shape (n,).
    """
    _require_state(state)
    Q = np.asarray(Q)
    if Q.shape != (state.n,):
        raise ValueError(f"Q must have shape ({state.n},), got {Q.shape}")
    out, res, ext, _ = _single_state(state, A_cum_slice, Q, r_true, delay_ms, late_counted)
    T = ext.shape[1] - state.H
    new_state = FluidState(state.t + T, res["c_end"][0, 0], ext[:, -state.H:].T)
    return new_state, _reshape_out(out, (state.n,))


def counterfactual(state: FluidState, A_cum_slice, Q_batch, r_true, delay_ms, *, late_counted: bool = False) -> FluidOut:
    """Evaluate many quota vectors for the same interval from the same start state, in one numba call.

    Q_batch has shape (..., n) (PRB); r_true (Mbit/s per PRB) broadcasts to it, normally (n,). Every field of the
    returned FluidOut has shape Q_batch.shape. Each element equals ``run_interval`` with that quota vector (bit for bit).
    """
    out, _, _, lead = _single_state(state, A_cum_slice, Q_batch, r_true, delay_ms, late_counted)
    return _reshape_out(out, lead)


def counterfactual_window(A_cum, t0, cleared0, Q_batch, r_true, delay_ms, *, slots_per_interval: int = SLOTS_PER_S,
                          late_counted: bool = False) -> FluidOut:
    """Evaluate S start states of one window, each with its own batch of quota vectors, in one numba call.

    A_cum      : (T_w, n) float64 cumulative arrivals of the whole window (Mbit), from slot 0.
    t0         : (S,) integer start slots of the evaluated intervals (each interval is ``slots_per_interval`` long).
    cleared0   : (S, n) cumulative cleared Mbit at those slots (e.g. rows of the ``cleared_start`` array of ``replay``).
    Q_batch    : (S, ..., n) quotas (PRB) per start state.
    r_true     : (S, n) true rates of each state's interval (Mbit/s per PRB), or the full shape of Q_batch.
    Returns a FluidOut with fields of shape Q_batch.shape; element [s, ...] equals
    ``counterfactual(state_at(A_cum, t0[s], cleared0[s], delay_ms), A_cum[t0[s]:t0[s]+T], Q_batch[s], r_true[s], ...)``.
    """
    A = _cumulative(A_cum, None, "A_cum")
    T_w, n = A.shape
    T = int(slots_per_interval)
    if T != slots_per_interval or T < 1:
        raise ValueError(f"slots_per_interval must be a positive integer, got {slots_per_interval!r}")
    t0 = np.asarray(t0)
    if t0.ndim != 1 or not np.issubdtype(t0.dtype, np.integer):
        raise ValueError(f"t0 must be a 1-D integer array of start slots, got dtype {t0.dtype} shape {t0.shape}")
    S = t0.shape[0]
    if S == 0 or np.any(t0 < 0) or np.any(t0 + T > T_w):
        raise ValueError(f"need S >= 1 and 0 <= t0, t0 + {T} <= {T_w} for every start slot")
    c0 = np.asarray(cleared0, dtype=np.float64)
    if c0.shape != (S, n) or not np.isfinite(c0).all():
        raise ValueError(f"cleared0 must be finite with shape ({S}, {n}), got {c0.shape}")
    Q = np.asarray(Q_batch)
    if Q.ndim < 2 or Q.shape[0] != S or Q.shape[-1] != n:
        raise ValueError(f"Q_batch must have shape ({S}, ..., {n}), got {Q.shape}")
    r = np.asarray(r_true)
    if r.shape == (S, n):
        r = r.reshape((S,) + (1,) * (Q.ndim - 2) + (n,))
    Qf, rf = _quota_rate(Q, r, n)
    d = delay_slots(delay_ms, n)
    H = int(d.max()) + 1
    ext = np.zeros((n, H + T_w))
    ext[:, H:] = A.T
    cols = np.arange(n)[None, :]
    A_prev = ext[cols, (H + t0 - 1)[:, None]]                               # A(t0 - 1), (S, n)
    A_end = ext[cols, (H + t0 + T - 1)[:, None]]                            # A(t0 + T - 1)
    if np.any(c0 < 0) or np.any(c0 > A_prev):
        raise ValueError("need 0 <= cleared0 <= A(t0-1) (nonnegative backlog) for every start state")
    if not late_counted:
        _check_no_expired(c0, ext[cols, (H + t0 - 1)[:, None] - d[None, :]])
    res = _run_batch(ext, H + t0, T, c0, Qf.reshape(S, -1, n), rf.reshape(S, -1, n), d, late_counted)
    return _reshape_out(_make_out(res, A_prev, A_end, c0), Q.shape)


def _continuation(Q_next, r_next, S, n, name):
    """Validate continuation quotas/rates ``(S, h, n)`` (PRB, Mbit/s per PRB); returns float64 copies and h."""
    Qn = np.asarray(Q_next)
    if Qn.ndim != 3 or Qn.shape[0] != S or Qn.shape[2] != n:
        raise ValueError(f"{name}: Q_next must have shape ({S}, h, {n}), got {Qn.shape}")
    Qn, rn = _quota_rate(Qn, r_next, n)
    return Qn, rn, Qn.shape[1]


def _lookahead(ext, base0, T, c0, Qf, rf, Qn, rn, steps, d, late_counted):
    """Shared core of the lookahead replays (all arrays already validated).

    Lane (s, m) plays ``Qf[s, m]`` for the interval starting at column ``base0[s]`` of ``ext``, then the continuation
    ``Qn[s, j-1]`` (rates ``rn[s, j-1]``) in the ``steps[s]`` following intervals. Returns on-time Mbit per lane and
    step, shape (S, M, h+1, n), zero for steps beyond ``steps[s]``. Every step runs the same per-slot kernel as
    ``replay``, so a lane equals the replay of a quota path that differs only in its first interval, bit for bit.
    """
    S, M, n = Qf.shape
    h = Qn.shape[1]
    res = _run_batch(ext, base0, T, c0, Qf, rf, d, late_counted)
    out = np.zeros((S, M, h + 1, n))
    out[:, :, 0] = res["on"]
    c = res["c_end"]
    for j in range(1, h + 1):
        sa = np.flatnonzero(steps >= j)
        if sa.size == 0:
            break
        lanes = sa.size * M
        base = np.repeat(base0[sa] + j * T, M)
        Qj = np.ascontiguousarray(np.broadcast_to(Qn[sa, j - 1][:, None, :], (sa.size, M, n)).reshape(lanes, 1, n))
        rj = np.ascontiguousarray(np.broadcast_to(rn[sa, j - 1][:, None, :], (sa.size, M, n)).reshape(lanes, 1, n))
        r2 = _run_batch(ext, base, T, np.ascontiguousarray(c[sa].reshape(lanes, n)), Qj, rj, d, late_counted)
        out[sa, :, j] = r2["on"].reshape(sa.size, M, n)
        c[sa] = r2["c_end"].reshape(sa.size, M, n)
    return out


def lookahead_window(A_cum, t0, cleared0, Q_batch, r_true, Q_next, r_next, delay_ms, *, steps=None,
                     slots_per_interval: int = SLOTS_PER_S, late_counted: bool = False) -> np.ndarray:
    """Multi-interval counterfactual: play each quota vector at its start interval, then a fixed continuation.

    Used by the PPO lookahead reward and the oracle (round-1 review, finding 3): with carry-over backlog and
    deadlines longer than one interval, the on-time value of the changed interval alone can have the wrong sign
    for the change in window welfare; the continuation captures the carry-over.

    A_cum    : (T_w, n) float64 cumulative arrivals of the whole window (Mbit), from slot 0.
    t0       : (S,) integer start slots of the evaluated intervals; cleared0: (S, n) cleared Mbit there.
    Q_batch  : (S, ..., n) quota vectors (PRB) for the start interval; r_true (S, n) or Q_batch.shape (Mbit/s per PRB).
    Q_next   : (S, h, n) continuation quotas (PRB) for the h intervals after each start (the same for every vector
               of the batch, e.g. the deterministic policy path); r_next (S, h, n) their true rates.
    steps    : (S,) number of continuation intervals actually played per state, 0 <= steps <= h (default h); use it
               to stop at the window end.
    Returns on-time Mbit of shape ``Q_batch.shape[:-1] + (h + 1, n)``: step 0 is the start interval, step j the
    j-th continuation interval (zero beyond ``steps[s]``). Element ``[s, ..., j, :]`` equals ``replay`` of the
    path whose interval ``t0[s] / T`` holds ``Q_batch[s, ...]`` and whose next intervals hold ``Q_next[s]``.
    """
    A = _cumulative(A_cum, None, "A_cum")
    T_w, n = A.shape
    T = int(slots_per_interval)
    if T != slots_per_interval or T < 1:
        raise ValueError(f"slots_per_interval must be a positive integer, got {slots_per_interval!r}")
    t0 = np.asarray(t0)
    if t0.ndim != 1 or not np.issubdtype(t0.dtype, np.integer):
        raise ValueError(f"t0 must be a 1-D integer array of start slots, got dtype {t0.dtype} shape {t0.shape}")
    S = t0.shape[0]
    c0 = np.asarray(cleared0, dtype=np.float64)
    if c0.shape != (S, n) or not np.isfinite(c0).all():
        raise ValueError(f"cleared0 must be finite with shape ({S}, {n}), got {c0.shape}")
    Q = np.asarray(Q_batch)
    if Q.ndim < 2 or Q.shape[0] != S or Q.shape[-1] != n:
        raise ValueError(f"Q_batch must have shape ({S}, ..., {n}), got {Q.shape}")
    r = np.asarray(r_true)
    if r.shape == (S, n):
        r = r.reshape((S,) + (1,) * (Q.ndim - 2) + (n,))
    Qf, rf = _quota_rate(Q, r, n)
    Qn, rn, h = _continuation(Q_next, r_next, S, n, "lookahead_window")
    steps = np.full(S, h, dtype=np.int64) if steps is None else np.asarray(steps)
    if steps.shape != (S,) or not np.issubdtype(steps.dtype, np.integer) or np.any(steps < 0) or np.any(steps > h):
        raise ValueError(f"steps must be an integer array of shape ({S},) with values in [0, {h}]")
    if S == 0 or np.any(t0 < 0) or np.any(t0 + (1 + steps) * T > T_w):
        raise ValueError(f"need S >= 1 and 0 <= t0, t0 + (1 + steps) * {T} <= {T_w} for every start state")
    d = delay_slots(delay_ms, n)
    H = int(d.max()) + 1
    ext = np.zeros((n, H + T_w))
    ext[:, H:] = A.T
    cols = np.arange(n)[None, :]
    A_prev = ext[cols, (H + t0 - 1)[:, None]]
    if np.any(c0 < 0) or np.any(c0 > A_prev):
        raise ValueError("need 0 <= cleared0 <= A(t0-1) (nonnegative backlog) for every start state")
    if not late_counted:
        _check_no_expired(c0, ext[cols, (H + t0 - 1)[:, None] - d[None, :]])
    out = _lookahead(ext, H + t0.astype(np.int64), T, c0, Qf.reshape(S, -1, n), rf.reshape(S, -1, n), Qn, rn,
                     steps.astype(np.int64), d, late_counted)
    return out.reshape(Q.shape[:-1] + (h + 1, n))


def counterfactual_path(state: FluidState, A_cum_slice, Q_batch, r_true, Q_next, r_next, delay_ms, *,
                        slots_per_interval: int = SLOTS_PER_S, late_counted: bool = False) -> np.ndarray:
    """Stepwise version of :func:`lookahead_window` from one :class:`FluidState` (oracle).

    A_cum_slice : ((1 + h) * T, n) absolute cumulative arrivals of the start interval and the h following ones.
    Q_batch     : (..., n) quota vectors (PRB) for the start interval; r_true (n,) its true rates.
    Q_next      : (h, n) continuation quotas (PRB); r_next (h, n) their true rates (Mbit/s per PRB).
    Returns on-time Mbit of shape ``Q_batch.shape[:-1] + (h + 1, n)``; step 0 equals ``counterfactual``.
    """
    _require_state(state)
    n = state.n
    T = int(slots_per_interval)
    if T != slots_per_interval or T < 1:
        raise ValueError(f"slots_per_interval must be a positive integer, got {slots_per_interval!r}")
    Qn = np.asarray(Q_next)
    if Qn.ndim != 2 or Qn.shape[1] != n:
        raise ValueError(f"Q_next must have shape (h, {n}), got {Qn.shape}")
    h = Qn.shape[0]
    A = np.asarray(A_cum_slice)
    if A.ndim != 2 or A.shape[0] != (1 + h) * T:
        raise ValueError(f"A_cum_slice must cover (1 + h) * {T} = {(1 + h) * T} slots, got shape {A.shape}")
    _, d, Qf, rf, ext = _prepare_single(state, A, Q_batch, r_true, delay_ms, late_counted)
    Qn, rn, _ = _continuation(Qn[None], np.asarray(r_next, dtype=np.float64)[None], 1, n, "counterfactual_path")
    lead = Qf.shape[:-1]
    out = _lookahead(ext, np.array([state.H], dtype=np.int64), T, state.cleared[None, :], Qf.reshape(1, -1, n),
                     rf.reshape(1, -1, n), Qn, rn, np.array([h], dtype=np.int64), d, late_counted)
    return out.reshape(lead + (h + 1, n))


def replay(A_cum, Q, r_true, delay_ms, *, slots_per_interval: int = SLOTS_PER_S, late_counted: bool = False):
    """Replay K consecutive intervals of a window from empty queues at slot 0, in one numba call.

    A_cum : (T_w, n) float64 cumulative arrivals (Mbit) with T_w >= K * slots_per_interval.
    Q     : (K, n) quotas (PRB) per interval; r_true: (K, n) true per-PRB rates (Mbit/s per PRB).
    Returns ``(out, cleared_start, state_end)``: FluidOut with (K, n) fields, the (K, n) cumulative cleared Mbit at
    each interval start (interval k starts at slot k * slots_per_interval; feed these to ``counterfactual_window`` or
    ``state_at``), and the FluidState after interval K-1. Equals chaining ``run_interval`` bit for bit.
    """
    A = _cumulative(A_cum, None, "A_cum")
    T_w, n = A.shape
    T = int(slots_per_interval)
    if T != slots_per_interval or T < 1:
        raise ValueError(f"slots_per_interval must be a positive integer, got {slots_per_interval!r}")
    Q = np.asarray(Q)
    if Q.ndim != 2 or Q.shape[1] != n or Q.shape[0] < 1:
        raise ValueError(f"Q must have shape (K >= 1, {n}), got {Q.shape}")
    K = Q.shape[0]
    if K * T > T_w:
        raise ValueError(f"{K} intervals of {T} slots need {K * T} slots of arrivals, A_cum has {T_w}")
    Qf, rf = _quota_rate(Q, r_true, n)
    d = delay_slots(delay_ms, n)
    H = int(d.max()) + 1
    ext = np.zeros((n, H + T_w))
    ext[:, H:] = A.T
    res = {k: np.empty((K, n)) for k in ("on", "late", "drop", "unused", "c_start")}
    c_last = np.empty(n)
    _replay_kernel(ext, H, T, Qf, rf, d, bool(late_counted), res["on"], res["late"], res["drop"],
                   res["unused"], res["c_start"], c_last)
    starts = np.arange(K) * T
    A_prev = ext[:, H + starts - 1].T                                       # (K, n)
    A_end = ext[:, H + starts + T - 1].T
    c_end = np.vstack([res["c_start"][1:], c_last[None, :]])
    out = FluidOut(arrivals_Mbit=A_end - A_prev, on_time_Mbit=res["on"], late_Mbit=res["late"],
                   dropped_Mbit=res["drop"], backlog_start_Mbit=A_prev - res["c_start"], backlog_Mbit=A_end - c_end,
                   unused_Mbit=res["unused"])
    state_end = FluidState(K * T, c_last, ext[:, K * T:H + K * T].T)
    return out, res["c_start"], state_end


def trace_interval(state: FluidState, A_cum_slice, Q, r_true, delay_ms, *, late_counted: bool = False) -> dict:
    """Per-slot curves of one interval (diagnostics): dict of (T, n) arrays in Mbit.

    ``on_time_Mbit``, ``late_Mbit``, ``dropped_Mbit``: per-slot increments; ``cleared_Mbit``: cumulative cleared Cc
    after each slot (absolute, same origin as A); ``cleared_before_Mbit``: Cc at the start of each slot.
    """
    _require_state(state)
    Q = np.asarray(Q)
    if Q.shape != (state.n,):
        raise ValueError(f"Q must have shape ({state.n},), got {Q.shape}")
    A, d, Qf, rf, ext = _prepare_single(state, A_cum_slice, Q, r_true, delay_ms, late_counted)
    tr = {k: np.empty(A.shape) for k in ("on", "late", "drop", "cleared")}  # (T, n)
    _trace_kernel(ext, state.H, A.shape[0], state.cleared, Qf, rf, d, bool(late_counted),
                  tr["on"], tr["late"], tr["drop"], tr["cleared"])
    before = np.vstack([state.cleared[None, :], tr["cleared"][:-1]])
    return {"on_time_Mbit": tr["on"], "late_Mbit": tr["late"], "dropped_Mbit": tr["drop"],
            "cleared_Mbit": tr["cleared"], "cleared_before_Mbit": before}


def served_delay_cdf(state: FluidState, A_cum_slice, Q, r_true, delay_ms, max_delay_ms: int, *,
                     late_counted: bool = False) -> np.ndarray:
    """Served Mbit with delay <= delta, for delta = 0 .. max_delay_ms (array (max_delay_ms + 1, n)), one interval.

    Inverts the cumulative curves: in slot t the served range (Cc_before, Cc_before + s] holds bits that arrived in slot
    a with delay t - a, and the bits with delay <= delta are those above A(t - delta - 1). In expiry mode the value at
    delta = d equals the served total. Needs ``max_delay_ms <= state.H - 1`` (the history the state holds). Evaluation
    only: vectorised over delta and slots per tenant, memory about 3 * (max_delay_ms + 1) * T floats.
    """
    D = int(max_delay_ms)
    if D != max_delay_ms or D < 0:
        raise ValueError(f"max_delay_ms must be a nonnegative integer, got {max_delay_ms!r}")
    if D > state.H - 1:
        raise ValueError(f"max_delay_ms={D} exceeds the state's history ({state.H - 1} ms)")
    tr = trace_interval(state, A_cum_slice, Q, r_true, delay_ms, late_counted=late_counted)
    A = np.asarray(A_cum_slice, dtype=np.float64)
    T = A.shape[0]
    H = state.H
    full = np.vstack([state.A_hist, A])                                     # row H + j is A(t0 + j)
    lo = tr["cleared_before_Mbit"]                                          # (T, n)
    hi = lo + tr["on_time_Mbit"] + tr["late_Mbit"]                          # Cc after service, before expiry
    delta = np.arange(D + 1)
    rows = H + np.arange(T)[None, :] - delta[:, None] - 1                   # (D+1, T), all >= 0 since D <= H - 1
    cdf = np.empty((D + 1, state.n))
    for i in range(state.n):
        thr = full[rows, i]                                                 # (D+1, T): A(t - delta - 1)
        cdf[:, i] = np.maximum(hi[None, :, i] - np.maximum(lo[None, :, i], thr), 0.0).sum(axis=1)
    return cdf
