"""Tests for slicemech.sim.fluid (INTERFACES.md §7).

Covers: hand-computed examples, conservation and capacity identities at realistic magnitudes, the deadline guarantee
(no served bit older than its budget), equality with a slow per-packet FIFO simulation in exact rational arithmetic,
determinism, batch/sequential equality of the counterfactual paths, the late-counted variant, and input validation.
"""
from collections import deque
from fractions import Fraction

import numpy as np
import pytest

from slicemech.sim import fluid as F

CONS_TOL = 1e-9      # Mbit; required conservation accuracy
REF_TOL = 1e-12      # Mbit; fluid vs exact per-packet reference (case magnitudes are 1e-3 .. 1 Mbit)


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------

def chain(A_cum, Q, r, d, T, late_counted=False):
    """Run the window interval by interval with run_interval; returns (list of outs, list of start states)."""
    n = A_cum.shape[1]
    st = F.initial_state(n, d)
    outs, states = [], []
    for k in range(Q.shape[0]):
        states.append(st)
        st, out = F.run_interval(st, A_cum[k * T:(k + 1) * T], Q[k], r[k], d, late_counted=late_counted)
        outs.append(out)
    return outs, states, st


def stack(outs, name):
    return np.stack([getattr(o, name) for o in outs])


def realistic_window(seed, K=200, T=1000):
    """Three tenants at cell-scale magnitudes (tens of Mbit/s, 106 PRB at ~0.43 Mbit/s/PRB), float32 per-ms input."""
    rng = np.random.default_rng(seed)
    n = 3
    mean_rate = np.array([24.0, 12.0, 4.0])                               # Mbit/s
    on = rng.random((K * T, n)) < np.array([0.35, 0.8, 0.2])               # bursty on/off per ms
    arr = np.where(on, rng.gamma(0.7, 1.0, size=(K * T, n)), 0.0)
    arr = arr / arr.mean(axis=0) * mean_rate / 1000.0                      # Mbit per ms
    arr[rng.random((K * T, n)) < 0.002] *= 40.0                            # rare large bursts
    arrivals_ms = arr.astype(np.float32)
    w = rng.dirichlet(np.ones(n), size=K)
    Q = np.floor(106 * w).astype(np.int64)
    Q[rng.random((K, n)) < 0.03] = 0
    r = rng.uniform(0.15, 0.66, size=(K, n))
    d = np.array([2002.0, 150.0, 300.0])
    return F.cumulative_arrivals(arrivals_ms), Q, r, d


def packet_case(seed, n=3, K=5, T=30):
    """Small random case with explicit packets: list over tenants of [(slot, size_Mbit)], plus per-ms sums."""
    rng = np.random.default_rng(seed)
    d = rng.choice(np.array([0, 1, 2, 5, T // 2, T - 1, T, T + 7, 2 * T + 3]), size=n).astype(float)
    Q = rng.integers(0, 12, size=(K, n))
    Q[rng.random((K, n)) < 0.15] = 0
    r = rng.uniform(0.05, 0.7, size=(K, n))
    r[rng.random((K, n)) < 0.05] = 0.0
    mean_cap = np.maximum((r * Q).mean(axis=0) / 1000.0, 1e-4)           # Mbit per slot
    packets, arrivals = [], np.zeros((K * T, n))
    for i in range(n):
        load = rng.uniform(0.3, 3.0)                                       # offered / granted (while on)
        pk_rate = rng.uniform(0.3, 3.0)                                    # packets per slot while on
        size_mean = load * mean_cap[i] / pk_rate
        on, pk = True, []
        for t in range(K * T):
            if rng.random() < 0.1:
                on = not on
            if on:
                for _ in range(rng.poisson(pk_rate)):
                    s = float(rng.uniform(0.2, 1.8) * size_mean)
                    pk.append((t, s))
                    arrivals[t, i] += s
        packets.append(pk)
    return packets, arrivals, Q, r, d


def packet_reference(packets, K, T, Q, r, d, late_counted):
    """Slow per-packet FIFO-with-deadline simulation in exact rational arithmetic (independent of cumulative curves).

    Each slot t: enqueue the packets of slot t; serve head-first up to r*Q/1000 Mbit (fluid: a packet may be split
    across slots); a served piece of a packet from slot a has delay t - a. Expiry mode then drops every packet from a
    slot <= t - d; late-counted mode never drops and classifies pieces with delay > d as late.
    Returns dict of (K, n) float arrays, the per-interval served-delay histograms and the largest served delay.
    """
    n = len(packets)
    keys = ("arrivals", "on_time", "late", "dropped", "backlog_start", "backlog", "unused")
    res = {k: np.zeros((K, n)) for k in keys}
    hist = [[dict() for _ in range(n)] for _ in range(K)]
    max_served_delay = np.full(n, -1)
    for i in range(n):
        queue, ptr, pk = deque(), 0, packets[i]
        for k in range(K):
            cap = Fraction(float(r[k, i])) * Fraction(float(Q[k, i])) / 1000
            acc = {key: Fraction(0) for key in keys}
            acc["backlog_start"] = sum((p[1] for p in queue), Fraction(0))
            for t in range(k * T, (k + 1) * T):
                while ptr < len(pk) and pk[ptr][0] == t:
                    size = Fraction(pk[ptr][1])
                    queue.append([t, size])
                    acc["arrivals"] += size
                    ptr += 1
                budget = cap
                while queue and budget > 0:
                    a, rem = queue[0]
                    take = min(rem, budget)
                    delay = t - a
                    if delay <= d[i]:
                        acc["on_time"] += take
                    else:
                        assert late_counted, "expiry mode served a bit older than its deadline"
                        acc["late"] += take
                    max_served_delay[i] = max(max_served_delay[i], delay)
                    hist[k][i][delay] = hist[k][i].get(delay, Fraction(0)) + take
                    budget -= take
                    if take == rem:
                        queue.popleft()
                    else:
                        queue[0][1] = rem - take
                acc["unused"] += budget
                if not late_counted:
                    while queue and queue[0][0] <= t - d[i]:
                        acc["dropped"] += queue.popleft()[1]
            acc["backlog"] = sum((p[1] for p in queue), Fraction(0))
            for key in keys:
                res[key][k, i] = float(acc[key])
    return res, hist, max_served_delay


# ---------------------------------------------------------------------------------------------------------------------
# hand-computed examples
# ---------------------------------------------------------------------------------------------------------------------

def test_hand_example_expiry_and_late():
    # 1 Mbit arrives in slot 0; cap = 1.0 * 250 / 1000 = 0.25 Mbit/slot; budget d = 2 ms (serve in slots 0, 1, 2).
    A = np.array([[1.0], [1.0], [1.0], [1.0]])
    st = F.initial_state(1, [2])
    s1, out = F.run_interval(st, A, [250], [1.0], [2])
    assert (out.on_time_Mbit[0], out.dropped_Mbit[0], out.late_Mbit[0]) == (0.75, 0.25, 0.0)
    assert (out.unused_Mbit[0], out.backlog_Mbit[0], out.arrivals_Mbit[0]) == (0.25, 0.0, 1.0)
    assert s1.t == 4 and s1.cleared[0] == 1.0
    _, late = F.run_interval(st, A, [250], [1.0], [2], late_counted=True)
    assert (late.on_time_Mbit[0], late.late_Mbit[0], late.dropped_Mbit[0], late.unused_Mbit[0]) == (0.75, 0.25, 0, 0)
    # dict-style access aliases
    assert out["on_time"] is out.on_time_Mbit and out["dropped_Mbit"] is out.dropped_Mbit


def test_zero_delay_budget_serves_only_in_arrival_slot():
    A = np.cumsum(np.array([[0.4], [0.0], [0.3], [0.1]]), axis=0)
    st = F.initial_state(1, [0])
    tr = F.trace_interval(st, A, [100], [2.0], [0])                       # cap 0.2 Mbit/slot
    np.testing.assert_allclose(tr["on_time_Mbit"][:, 0], [0.2, 0.0, 0.2, 0.1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(tr["dropped_Mbit"][:, 0], [0.2, 0.0, 0.1, 0.0], rtol=0, atol=1e-15)


def test_deadline_spans_interval_boundaries_and_backlog_carries_over():
    # 3 Mbit in slot 0 of interval 0 with zero quota; budget 7 slots > interval length 3: interval 1 still serves it.
    T = 3
    arr = np.zeros((4 * T, 1))
    arr[0, 0] = 3.0
    A = F.cumulative_arrivals(arr)
    Q = np.array([[0], [1000], [1000], [1000]])
    r = np.full((4, 1), 0.5)                                               # cap 0.5 Mbit/slot when Q = 1000
    outs, _, _ = chain(A, Q, r, [7.0], T)
    np.testing.assert_allclose(stack(outs, "on_time_Mbit")[:, 0], [0.0, 1.5, 1.0, 0.0])   # slots 3..7 serve, 8 expires
    np.testing.assert_allclose(stack(outs, "dropped_Mbit")[:, 0], [0.0, 0.0, 0.5, 0.0])
    np.testing.assert_allclose(stack(outs, "backlog_Mbit")[:, 0], [3.0, 1.5, 0.0, 0.0])
    np.testing.assert_array_equal(stack(outs, "backlog_start_Mbit")[1:], stack(outs, "backlog_Mbit")[:-1])


# ---------------------------------------------------------------------------------------------------------------------
# conservation, capacity, nonnegativity, deadline guarantee
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("late_counted", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_conservation_realistic_window(seed, late_counted):
    A, Q, r, d = realistic_window(seed)
    T = 1000
    out, c_start, st = F.replay(A, Q, r, d, slots_per_interval=T, late_counted=late_counted)
    assert np.abs(out.conservation_residual()).max() <= CONS_TOL
    # arrivals measured on the absolute input curve, and window totals
    K = Q.shape[0]
    A_end = A[np.arange(K) * T + T - 1]
    A_prev = np.vstack([np.zeros((1, 3)), A_end[:-1]])
    np.testing.assert_allclose(out.arrivals_Mbit, A_end - A_prev, rtol=0, atol=CONS_TOL)
    total = out.on_time_Mbit.sum(0) + out.late_Mbit.sum(0) + out.dropped_Mbit.sum(0) + out.backlog_Mbit[-1]
    np.testing.assert_allclose(total, A[K * T - 1], rtol=0, atol=CONS_TOL)
    np.testing.assert_allclose(st.backlog, out.backlog_Mbit[-1], rtol=0, atol=0)
    # capacity identity (service is quantised to the float spacing of A, at most half a spacing per slot)
    np.testing.assert_allclose(out.capacity_Mbit, r * Q * T / 1000.0, rtol=0, atol=T * np.spacing(A.max()))
    # exact nonnegativity
    for name, v in out.as_dict().items():
        assert (v >= 0).all(), name
    if late_counted:
        assert (out.dropped_Mbit == 0).all()
        assert out.late_Mbit.sum() > 0                                     # the case exercises late service
    else:
        assert (out.late_Mbit == 0).all()
        assert out.dropped_Mbit.sum() > 0 and out.unused_Mbit.sum() > 0    # the case exercises expiry and idling


@pytest.mark.parametrize("seed", [3, 4])
def test_no_served_bit_older_than_deadline(seed):
    """Expiry mode: in every slot the served range lies above A(t - d - 1), i.e. every served bit has delay <= d."""
    A, Q, r, d = realistic_window(seed, K=20)
    T = 1000
    _, c_start, _ = F.replay(A, Q, r, d, slots_per_interval=T)
    dsl = d.astype(int)
    for k in range(Q.shape[0]):
        st = F.state_at(A, k * T, c_start[k], d)
        tr = F.trace_interval(st, A[k * T:(k + 1) * T], Q[k], r[k], d)
        served = tr["on_time_Mbit"] > 0
        t = k * T + np.arange(T)[:, None]
        thr_idx = t - dsl[None, :] - 1                                     # slot index of A(t - d - 1)
        thr = np.where(thr_idx >= 0, A[np.clip(thr_idx, 0, None), np.arange(3)[None, :]], 0.0)
        assert (tr["cleared_before_Mbit"][served] >= thr[served]).all()
        # after each slot nothing older than d slots is queued: Cc >= A(t - d)
        thr2_idx = t - dsl[None, :]
        thr2 = np.where(thr2_idx >= 0, A[np.clip(thr2_idx, 0, None), np.arange(3)[None, :]], 0.0)
        assert (tr["cleared_Mbit"] >= thr2).all()
        cdf = F.served_delay_cdf(st, A[k * T:(k + 1) * T], Q[k], r[k], d, int(dsl.max()))
        for i in range(3):                                                 # all service has delay <= d_i
            assert abs(cdf[dsl[i], i] - tr["on_time_Mbit"][:, i].sum()) <= CONS_TOL
        # the trace kernel runs the same per-slot arithmetic as the batch kernel
        st_next, out = F.run_interval(st, A[k * T:(k + 1) * T], Q[k], r[k], d)
        np.testing.assert_array_equal(tr["cleared_Mbit"][-1], st_next.cleared)
        np.testing.assert_allclose(tr["on_time_Mbit"].sum(0), out.on_time_Mbit, rtol=0, atol=CONS_TOL)
        np.testing.assert_allclose(tr["dropped_Mbit"].sum(0), out.dropped_Mbit, rtol=0, atol=CONS_TOL)


# ---------------------------------------------------------------------------------------------------------------------
# equality with the per-packet reference
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("late_counted", [False, True])
def test_matches_per_packet_reference(late_counted):
    K, T = 5, 30
    seen = {"dropped": 0.0, "late": 0.0, "unused": 0.0, "backlog": 0.0}
    for seed in range(40):
        packets, arrivals, Q, r, d = packet_case(seed, K=K, T=T)
        ref, hist, max_delay = packet_reference(packets, K, T, Q, r, d, late_counted)
        A = F.cumulative_arrivals(arrivals)
        outs, states, _ = chain(A, Q, r, d, T, late_counted)
        for key in ref:
            got = stack(outs, key + "_Mbit")
            np.testing.assert_allclose(got, ref[key], rtol=0, atol=REF_TOL, err_msg=f"seed {seed} field {key}")
        if not late_counted:
            assert (max_delay <= d).all()
        # served-delay distribution from inverting the cumulative curves
        Dmax = states[0].H - 1
        for k in range(K):
            cdf = F.served_delay_cdf(states[k], A[k * T:(k + 1) * T], Q[k], r[k], d, Dmax, late_counted=late_counted)
            for i in range(3):
                ref_cdf = np.cumsum([float(hist[k][i].get(dl, 0)) for dl in range(Dmax + 1)])
                np.testing.assert_allclose(cdf[:, i], ref_cdf, rtol=0, atol=REF_TOL)
        for key in seen:
            seen[key] += ref[key].sum()
    # the random cases must exercise every branch of the kernel
    assert seen["unused"] > 0 and seen["backlog"] > 0
    assert (seen["late"] > 0) if late_counted else (seen["dropped"] > 0)


def test_fractional_quotas_match_reference():
    K, T = 4, 25
    for seed in range(10):
        packets, arrivals, Q, r, d = packet_case(100 + seed, K=K, T=T)
        Qf = Q + np.random.default_rng(seed).random(Q.shape)               # time-shared average PRBs
        ref, _, _ = packet_reference(packets, K, T, Qf, r, d, late_counted=False)
        outs, _, _ = chain(F.cumulative_arrivals(arrivals), Qf, r, d, T)
        for key in ref:
            np.testing.assert_allclose(stack(outs, key + "_Mbit"), ref[key], rtol=0, atol=REF_TOL)


# ---------------------------------------------------------------------------------------------------------------------
# determinism and batched counterfactuals
# ---------------------------------------------------------------------------------------------------------------------

def test_determinism_bitwise():
    A, Q, r, d = realistic_window(5, K=30)
    o1, c1, s1 = F.replay(A, Q, r, d)
    o2, c2, s2 = F.replay(A.copy(), Q.copy(), r.copy(), d.copy())
    for k in o1.as_dict():
        np.testing.assert_array_equal(getattr(o1, k), getattr(o2, k))
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(s1.cleared, s2.cleared)


@pytest.mark.parametrize("late_counted", [False, True])
def test_replay_equals_run_interval_chain(late_counted):
    A, Q, r, d = realistic_window(6, K=25)
    T = 1000
    out, c_start, st_end = F.replay(A, Q, r, d, late_counted=late_counted)
    outs, states, st_chain = chain(A, Q, r, d, T, late_counted)
    for key in out.as_dict():
        np.testing.assert_array_equal(getattr(out, key), stack(outs, key))
    np.testing.assert_array_equal(c_start, np.stack([s.cleared for s in states]))
    for k in (0, 1, 7, 24):
        rebuilt = F.state_at(A, k * T, c_start[k], d)
        assert rebuilt.t == states[k].t
        np.testing.assert_array_equal(rebuilt.A_hist, states[k].A_hist)
    np.testing.assert_array_equal(st_end.A_hist, st_chain.A_hist)
    np.testing.assert_array_equal(st_end.cleared, st_chain.cleared)
    assert st_end.t == st_chain.t == Q.shape[0] * T


@pytest.mark.parametrize("late_counted", [False, True])
def test_counterfactual_batch_equals_sequential(late_counted):
    A, Q, r, d = realistic_window(7, K=12)
    T = 1000
    _, states, _ = chain(A, Q, r, d, T, late_counted)
    rng = np.random.default_rng(0)
    for k in (0, 5, 11):
        st = states[k]
        before = (st.cleared.copy(), st.A_hist.copy())
        Qb = rng.integers(0, 60, size=(2, 8, 3))
        cf = F.counterfactual(st, A[k * T:(k + 1) * T], Qb, r[k], d, late_counted=late_counted)
        assert cf.on_time_Mbit.shape == (2, 8, 3)
        for a in range(2):
            for b in range(8):
                _, seq = F.run_interval(st, A[k * T:(k + 1) * T], Qb[a, b], r[k], d, late_counted=late_counted)
                for key in seq.as_dict():
                    np.testing.assert_array_equal(getattr(cf, key)[a, b], getattr(seq, key))
        np.testing.assert_array_equal(st.cleared, before[0])              # start state untouched
        np.testing.assert_array_equal(st.A_hist, before[1])


@pytest.mark.parametrize("late_counted", [False, True])
def test_counterfactual_window_equals_per_state(late_counted):
    A, Q, r, d = realistic_window(8, K=40)
    T = 1000
    _, c_start, _ = F.replay(A, Q, r, d, late_counted=late_counted)
    ks = np.array([0, 3, 17, 39])
    rng = np.random.default_rng(1)
    Qb = rng.integers(0, 60, size=(len(ks), 4, 3))
    win = F.counterfactual_window(A, ks * T, c_start[ks], Qb, r[ks], d, late_counted=late_counted)
    for s, k in enumerate(ks):
        st = F.state_at(A, k * T, c_start[k], d)
        cf = F.counterfactual(st, A[k * T:(k + 1) * T], Qb[s], r[k], d, late_counted=late_counted)
        for key in cf.as_dict():
            np.testing.assert_array_equal(getattr(win, key)[s], getattr(cf, key))
    # the rollout's own quota reproduces the replay's next start state
    own = F.counterfactual_window(A, ks[:-1] * T, c_start[ks[:-1]], Q[ks[:-1]], r[ks[:-1]], d,
                                  late_counted=late_counted)
    out_full, _, _ = F.replay(A, Q, r, d, late_counted=late_counted)
    np.testing.assert_array_equal(own.on_time_Mbit, out_full.on_time_Mbit[ks[:-1]])
    np.testing.assert_array_equal(own.backlog_Mbit, out_full.backlog_Mbit[ks[:-1]])


# ---------------------------------------------------------------------------------------------------------------------
# late-counted variant
# ---------------------------------------------------------------------------------------------------------------------

def test_late_counted_equals_expiry_without_deadline_pressure():
    A, Q, r, d = realistic_window(9, K=10)
    big = np.full(3, 10_000.0)                                             # longer than the window: nothing expires
    e, _, _ = F.replay(A, Q, r, big)
    l, _, _ = F.replay(A, Q, r, big, late_counted=True)
    for key in e.as_dict():
        np.testing.assert_array_equal(getattr(e, key), getattr(l, key))
    assert (e.dropped_Mbit == 0).all() and (l.late_Mbit == 0).all()


def test_expiry_never_has_less_on_time_service_than_late_counted():
    """Dropping bits that are already late never costs on-time service: cumulative on-time(expiry) >= late-counted."""
    for seed in (10, 11):
        A, Q, r, d = realistic_window(seed, K=60)
        e, ce, _ = F.replay(A, Q, r, d)
        l, cl, _ = F.replay(A, Q, r, d, late_counted=True)
        assert (np.cumsum(e.on_time_Mbit, 0) >= np.cumsum(l.on_time_Mbit, 0) - CONS_TOL).all()
        assert (ce >= cl - CONS_TOL).all()                                 # expiry clears at least as fast
        np.testing.assert_allclose(l.on_time_Mbit + l.late_Mbit + l.unused_Mbit, r * Q, rtol=0,
                                   atol=1000 * np.spacing(A.max()))


# ---------------------------------------------------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------------------------------------------------

def test_invalid_inputs_raise():
    n, T = 3, 10
    d = np.array([5.0, 2.0, 0.0])
    st = F.initial_state(n, d)
    A = np.cumsum(np.full((T, n), 0.01), axis=0)
    Q, r = np.array([3, 4, 5]), np.full(n, 0.4)
    bad_calls = [
        lambda: F.run_interval(st, A, [-1, 4, 5], r, d),                    # negative quota
        lambda: F.run_interval(st, A, [np.nan, 4, 5], r, d),                # NaN quota
        lambda: F.run_interval(st, A, [3, 4], r, d),                        # wrong quota shape
        lambda: F.run_interval(st, A, Q, [0.4, -0.1, 0.4], d),              # negative rate
        lambda: F.run_interval(st, A[::-1], Q, r, d),                       # decreasing cumulative curve
        lambda: F.run_interval(st, A[:, :2], Q, r, d),                      # wrong tenant count
        lambda: F.run_interval(st, A, Q, r, [5.5, 2, 0]),                   # fractional ms
        lambda: F.run_interval(st, A, Q, r, [-1, 2, 0]),                    # negative delay
        lambda: F.run_interval(st, A, Q, r, [50, 2, 0]),                    # history too short for this delay
        lambda: F.counterfactual(st, A, np.ones((4, 2)), r, d),             # batch with wrong tenant axis
        lambda: F.cumulative_arrivals(-np.ones((4, n))),
        lambda: F.FluidState(0, np.zeros(n), np.ones((3, n))),              # history before window start not zero
        lambda: F.FluidState(5, np.full(n, 2.0), np.ones((3, n))),          # cleared above arrivals
        lambda: F.FluidState(1.5, np.zeros(n), np.zeros((3, n))),           # non-integer slot index
        lambda: F.replay(A, np.ones((2, n)), np.ones((2, n)), d, slots_per_interval=T),     # not enough slots
        lambda: F.counterfactual_window(A, np.array([0.0]), np.zeros((1, n)), np.ones((1, n)), r[None], d,
                                        slots_per_interval=5),              # float start slots
        lambda: F.counterfactual_window(A, np.array([6]), np.zeros((1, n)), np.ones((1, n)), r[None], d,
                                        slots_per_interval=5),              # interval runs past the window
        lambda: F.served_delay_cdf(st, A, Q, r, d, 6),                      # delay beyond the state's history
    ]
    for call in bad_calls:
        with pytest.raises(ValueError):
            call()
    # the next slice must continue the state's arrival curve
    st1, _ = F.run_interval(st, A, Q, r, d)
    with pytest.raises(ValueError):
        F.run_interval(st1, A, Q, r, d)
    # a late-counted state that still queues expired bits cannot be continued in expiry mode
    burst = np.zeros((T, n))
    burst[0] = 1.0
    st_late, _ = F.run_interval(st, np.cumsum(burst, 0), [0, 0, 0], r, d, late_counted=True)
    with pytest.raises(ValueError):
        F.run_interval(st_late, np.cumsum(burst, 0) + 1.0, Q, r, d)
    F.run_interval(st_late, np.cumsum(burst, 0) + 1.0, Q, r, d, late_counted=True)   # fine in its own mode
    with pytest.raises(TypeError):
        F.run_interval({"t": 0}, A, Q, r, d)
