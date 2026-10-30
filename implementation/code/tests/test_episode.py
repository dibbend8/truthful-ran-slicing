"""Tests for slicemech.sim.episode.run_window (INTERFACES.md §11).

Synthetic windows use the real ``data.compose.Window`` dataclass (validated shapes) with bursty per-ms arrivals, so
every test runs the real allocator, pricing engines, best responses, features and fluid kernel. One test composes a
short real TRAIN window and is skipped when the prepared data are missing.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

torch.set_num_threads(1)

from slicemech.config import Mechanism, Shape, contract_from_params  # noqa: E402
from slicemech.control import features as F  # noqa: E402
from slicemech.control import nets as N  # noqa: E402
from slicemech.control import oracle as O  # noqa: E402
from slicemech.control import ppo as P  # noqa: E402
from slicemech.control import static as S  # noqa: E402
from slicemech.data.compose import Window  # noqa: E402
from slicemech.eval import metrics as EM  # noqa: E402
from slicemech.model import allocation as AL  # noqa: E402
from slicemech.model.direct import direct_integer  # noqa: E402
from slicemech.sim import episode as E  # noqa: E402
from slicemech.sim import fluid  # noqa: E402
from slicemech.tenants.strategies import BestResponder, Multiplier, Truthful  # noqa: E402

CLASSES = ("video", "urllc", "mmtc")
MECH = Mechanism()
FC = F.OperatorAR.default()      # explicit opt-in to the untrained operator forecaster (no silent default)
SHAPE = Shape(beta=(np.array([1.0]), np.array([0.5, 1.5]), np.array([0.25, 0.75, 3.0])),
              rho=(np.array([1.0]), np.array([0.5, 0.5]), np.array([0.4, 0.4, 0.2])))


def make_window(seed: int, K: int = 12, warmup: int = 3, mu=(20.0, 10.0, 4.0), split: str = "train") -> Window:
    """Bursty synthetic window near full load (B r ~ 42 Mbit/s against sum mu = 34 Mbit/s)."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float)
    n = mu.size
    ct = contract_from_params(list(CLASSES[:n]), mu)
    level = mu * rng.gamma(4.0, 0.25, (K, n))                                   # Mbit/s per interval
    per_ms = np.repeat(level, 1000, axis=0) / 1000.0 * rng.gamma(0.5, 2.0, (K * 1000, n))
    arrivals = per_ms.astype(np.float32)
    measured = arrivals.astype(float).reshape(K, 1000, n).sum(1)               # Mbit/s (Delta = 1 s)
    theta = measured * rng.uniform(0.85, 1.15, (K, n))                          # imperfect truthful information
    r = rng.uniform(0.25, 0.6, (K + 1, n))
    return Window(id=f"synth-{seed}", split=split, date="2024-01-01", n=n, cls=CLASSES[:n], arrivals_ms=arrivals,
                  theta=theta, measured=measured, rate_true=r[1:], rate_hat=r[:-1], warmup=warmup, contract=ct,
                  meta=dict(tod_start_s=3600.0 * seed % 86400, rate_age_s=np.ones((K, n))))


def truthful(n: int = 3):
    return [Truthful() for _ in range(n)]


class Recorder:
    """Controller wrapper recording every context it was given (and refusing report-carrying contexts)."""

    def __init__(self, inner):
        self.inner = inner
        self.alloc_input = getattr(inner, "alloc_input", "reports")
        self.contexts = []
        self.maps = []

    def priority_map(self, ctx, z=None):
        assert z is None
        self.contexts.append((np.array(ctx[0]), np.array(ctx[1]), int(ctx.k)))
        m = self.inner.priority_map(ctx)
        self.maps.append(m)
        return m


# ============================================================================================ structure
def test_columns_order_record_flag_and_attrs():
    w = make_window(1)
    df = E.run_window(w, E.FixedWeights(), "full", truthful(), 2, MECH, SHAPE)
    assert len(df) == w.K * w.n
    assert list(df["k"]) == list(np.repeat(np.arange(w.K), w.n))
    assert list(df["tenant"]) == list(np.tile(np.arange(w.n), w.K))
    assert set(E.SECTION11_COLUMNS) <= set(df.columns)
    assert set(EM.EPISODE_COLUMNS) <= set(df.columns)
    assert (df["scored"] == (df["k"] >= w.warmup)).all()
    assert (df["released_through"] == df["k"] - 2).all()
    small = E.run_window(w, E.FixedWeights(), "full", truthful(), 2, MECH, SHAPE, record=False)
    assert tuple(small.columns) == E.SECTION11_COLUMNS
    pd.testing.assert_frame_equal(small, df.loc[:, list(E.SECTION11_COLUMNS)])
    for key in ("window_id", "split", "date", "warmup", "d", "price_rule", "controller", "policies", "timing_s"):
        assert key in df.attrs
    assert df.attrs["policies"] == ["truthful"] * 3 and df.attrs["controller"] == "equal"


def test_deterministic():
    w = make_window(2)
    a = E.run_window(w, E.RandomProtected(3), "full", truthful(), 2, MECH, SHAPE)
    b = E.run_window(w, E.RandomProtected(3), "full", truthful(), 2, MECH, SHAPE)
    pd.testing.assert_frame_equal(a, b)


# ============================================================================================ allocation + replay
@pytest.mark.parametrize("seed", [3, 4])
def test_quotas_and_delivery_match_allocator_and_fluid(seed):
    w = make_window(seed, K=15)
    df = E.run_window(w, E.RandomProtected(seed), "full", truthful(), 2, MECH, SHAPE)
    K, n = w.K, w.n
    z = df["report"].to_numpy().reshape(K, n)
    W = df["w"].to_numpy().reshape(K, n)
    Q = df["quota"].to_numpy().reshape(K, n)
    np.testing.assert_array_equal(z, np.minimum(w.theta, w.contract.C))
    np.testing.assert_array_equal(Q, AL.integer(z, W, w.rate_hat, w.contract, MECH))
    assert (Q.sum(1) <= MECH.B).all()
    A = fluid.cumulative_arrivals(w.arrivals_ms)
    out, _, _ = fluid.replay(A, Q.astype(np.int64), w.rate_true, w.contract.delay_ms)
    np.testing.assert_array_equal(df["on_time_Mbit"].to_numpy().reshape(K, n), out.on_time_Mbit)
    np.testing.assert_array_equal(df["dropped_Mbit"].to_numpy().reshape(K, n), out.dropped_Mbit)
    chk = E.episode_checks(df, w.contract, MECH)
    assert chk["conservation_max"] < 1e-9 and chk["handover_max"] == 0.0 and chk["budget_excess"] <= 0
    # stage flags agree with the allocator
    st = AL.stage(z, w.rate_hat, w.contract, MECH)
    np.testing.assert_array_equal(df["overload"].to_numpy().reshape(K, n)[:, 0], st["overload"])
    np.testing.assert_array_equal(df["leftover"].to_numpy().reshape(K, n)[:, 0], st["leftover"])


def test_value_and_utility_identities():
    w = make_window(5)
    df = E.run_window(w, E.RandomProtected(5), "full", truthful(), 2, MECH, SHAPE)
    pi = w.contract.pi[df["tenant"].to_numpy()]
    np.testing.assert_array_equal(df["realized_value"], pi * df["on_time_Mbit"])
    np.testing.assert_array_equal(df["utility_declared"], df["V_declared"] - df["price"])
    np.testing.assert_array_equal(df["utility_realized"], df["realized_value"] - df["price"])
    np.testing.assert_array_equal(df["price"], df["p"])
    # truthful reports: declared value at the type equals the value computed from the report
    np.testing.assert_allclose(df["V_declared"], df["V_report"], rtol=0, atol=0)


# ============================================================================================ prices
def test_prices_certified_bounded_and_premium_nonnegative():
    w = make_window(6, K=15)
    df = E.run_window(w, E.RandomProtected(6), "fixed", truthful(), 2, MECH, SHAPE)
    chk = E.episode_checks(df, w.contract, MECH)
    assert chk["cert_violation_p"] <= 0 and chk["cert_violation_p_fix"] <= 0
    tol = 1e-9 * float((w.contract.pi * w.contract.C).max())
    assert chk["price_above_V"] <= tol and chk["price_negative"] <= tol and chk["premium_negative"] <= tol
    np.testing.assert_array_equal(df["price"], df["p_fix"])
    np.testing.assert_allclose(df["T"], df["p"] - df["p_fix"], rtol=0, atol=0)
    assert np.isfinite(df[["p", "p_fix", "T"]].to_numpy()).all()


def test_equal_weights_have_zero_premium():
    w = make_window(7)
    df = E.run_window(w, E.FixedWeights(), "full", truthful(), 2, MECH, SHAPE)
    np.testing.assert_array_equal(df["p"], df["p_fix"])
    assert (df["T"] == 0).all()


def test_usage_rule():
    w = make_window(8)
    nu = np.array([0.1, 0.5, 0.05])
    df = E.run_window(w, E.FixedWeights(), ("usage", nu), truthful(), 2, MECH, SHAPE)
    np.testing.assert_array_equal(df["price"], nu[df["tenant"].to_numpy()] * df["quota"])
    np.testing.assert_array_equal(df["p_usage"], df["price"])
    with pytest.raises(ValueError, match="nu"):
        E.run_window(w, E.FixedWeights(), "usage", truthful(), 2, MECH, SHAPE)


# ============================================================================================ information order
@pytest.mark.parametrize("d", [1, 3])
def test_context_uses_released_data_only(d):
    w = make_window(9, K=14)
    rec = Recorder(E.RandomProtected(9))
    E.run_window(w, rec, "full", truthful(), d, MECH, SHAPE, forecaster=FC)
    full = F.window_features(w, d, w.contract, mech=MECH, forecaster=FC)   # the same from the whole window
    assert [k for _, _, k in rec.contexts] == list(range(w.K))
    for c_t, c_g, k in rec.contexts:
        np.testing.assert_array_equal(c_t, full.c_tenant[k])
        np.testing.assert_array_equal(c_g, full.c_global[k])


def test_released_view_masks_the_future():
    w = make_window(10, K=10)
    v = E.ReleasedData(w, d=2)
    assert not hasattr(v, "theta") and set(v.meta) == {"tod_start_s", "rate_age_s"}
    for k in range(w.K):
        v.advance(k)
        np.testing.assert_array_equal(v.measured[: max(k - 1, 0)], w.measured[: max(k - 1, 0)])
        assert (v.measured[max(k - 1, 0):] == 0).all()
        np.testing.assert_array_equal(v.rate_hat[: k + 1], w.rate_hat[: k + 1])
        assert (v.rate_hat[k + 1:] == w.rate_hat[k]).all()
    with pytest.raises(ValueError):
        v.advance(3)                                             # time never runs backwards
    with pytest.raises(F.ReportLeakError):
        F.context(v, 0, 2, w.contract, reports=np.ones(3))


def test_reports_never_reach_the_controller():
    w = make_window(11)
    w_lie = dataclasses.replace(w, theta=w.theta * 1.7)          # different types and reports, same measurements
    net = N.make_net("protected", seed=4)
    runs = []
    for win, pols in ((w, truthful()), (w, [Multiplier(2.0), Multiplier(0.5), Truthful()]), (w_lie, truthful())):
        rec = Recorder(net)
        E.run_window(win, rec, "full", pols, 2, MECH, SHAPE, forecaster=FC)
        runs.append(rec)
    for other in runs[1:]:
        for (a_t, a_g, _), (b_t, b_g, _) in zip(runs[0].contexts, other.contexts):
            np.testing.assert_array_equal(a_t, b_t)
            np.testing.assert_array_equal(a_g, b_g)
        for m0, m1 in zip(runs[0].maps, other.maps):
            np.testing.assert_array_equal(m0.a, m1.a)
            np.testing.assert_array_equal(m0.kappa, m1.kappa)


class HistorySpy(Truthful):
    """Truthful tenant that records the length and content of the own history it sees."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def report(self, i, theta, C, history=None, game=None):
        assert game is None
        self.seen.append((len(history), list(history.z), list(history.quota)))
        return super().report(i, theta, C, history=history, game=game)


def test_tenants_see_only_their_own_past():
    w = make_window(12)
    spies = [HistorySpy() for _ in range(3)]
    df = E.run_window(w, E.FixedWeights(), "full", spies, 2, MECH, SHAPE)
    for i, spy in enumerate(spies):
        own = df[df["tenant"] == i]
        for k, (length, zs, qs) in enumerate(spy.seen):
            assert length == k
            assert zs == own["report"].tolist()[:k] and qs == own["quota"].tolist()[:k]


# ============================================================================================ best responses
@pytest.mark.parametrize("ctrl", [E.FixedWeights(), E.RandomProtected(13)], ids=["equal", "protected"])
def test_full_price_best_response_gain_within_certificate(ctrl):
    w = make_window(13, K=10)
    base = E.run_window(w, ctrl, "full", truthful(), 2, MECH, SHAPE)
    br = E.run_window(w, ctrl, "full", [Truthful(), BestResponder("full"), Truthful()], 2, MECH, SHAPE)
    chk = E.episode_checks(br, w.contract, MECH)
    assert chk["br_full_rows"] == w.K and chk["br_full_excess"] <= 0
    rows = br[br["tenant"] == 1]
    assert rows["br_exact"].all() and (rows["br_price"] == "full").all()
    # declared utility of the best responder never beats truth by more than the certificate
    u_br = rows["utility_declared"].to_numpy()
    u_tr = base[base["tenant"] == 1]["utility_declared"].to_numpy()
    assert np.all(u_br - u_tr <= rows["br_cert"].to_numpy() + 1e-9)
    if isinstance(ctrl, E.RandomProtected):
        assert (rows["T"] > 1e-3).any()                         # a premium exists, yet truth stays optimal


def test_fixed_price_best_response_gain_equals_episode_difference():
    w = make_window(14, K=10)
    ctrl = E.RandomProtected(14)
    base = E.run_window(w, ctrl, "fixed", truthful(), 2, MECH, SHAPE)
    br = E.run_window(w, ctrl, "fixed", [Truthful(), BestResponder("fixed"), Truthful()], 2, MECH, SHAPE)
    rows, ref = br[br["tenant"] == 1], base[base["tenant"] == 1]
    gain = rows["br_gain"].to_numpy()
    assert (gain >= 0).all()
    assert gain.max() > 1.0                                     # inflation pays under the fixed-priority price
    scale = 1e-9 * float(w.contract.pi[1] * w.contract.C[1])
    # same report-free rule per interval and truthful opponents: the episode realises exactly the computed gain
    np.testing.assert_allclose(rows["br_U_truth"], ref["utility_declared"], rtol=0, atol=scale)
    np.testing.assert_allclose(rows["utility_declared"].to_numpy() - ref["utility_declared"].to_numpy(), gain,
                               rtol=0, atol=scale)
    assert (rows["report"] >= np.minimum(rows["theta"], w.contract.C[1]) - 1e-12).all()


# ============================================================================================ controllers
def test_direct_controller_and_true_rate_diagnostic():
    w = make_window(15)
    df = E.run_window(w, E.DirectController(), "full", truthful(), 2, MECH, SHAPE)
    K, n = w.K, w.n
    z = df["report"].to_numpy().reshape(K, n)
    Q = df["quota"].to_numpy().reshape(K, n)
    for k in range(K):
        np.testing.assert_array_equal(Q[k], direct_integer(z[k], w.rate_hat[k], w.contract, SHAPE, MECH))
    assert df["w"].isna().all() and df["p_fix"].isna().all() and df["T"].isna().all()
    chk = E.episode_checks(df, w.contract, MECH)
    assert chk["cert_violation_p"] <= 0
    dt = E.run_window(w, E.DirectController("true"), "full", truthful(), 2, MECH, SHAPE)
    np.testing.assert_array_equal(dt["r_alloc"], dt["r_true"])
    with pytest.raises(ValueError, match="fixed-priority"):
        E.run_window(w, E.DirectController(), "fixed", truthful(), 2, MECH, SHAPE)


def test_unconstrained_net_fixed_price_and_grid_envelope():
    w = make_window(16, K=5, warmup=1)
    net = N.make_net("unconstrained", seed=2)
    df = E.run_window(w, net, "fixed", truthful(), 2, MECH, SHAPE, forecaster=FC)
    assert df["p"].isna().all() and np.isfinite(df["p_fix"]).all()
    np.testing.assert_array_equal(df["price"], df["p_fix"])
    dg = E.run_window(w, net, E.PriceRule("fixed", nonmonotone_full=True), truthful(), 2, MECH, SHAPE,
                      forecaster=FC)
    assert np.isfinite(dg["p"]).all() and dg["p_lo"].isna().all()
    np.testing.assert_array_equal(dg["quota"], df["quota"])


def test_measurement_only_controller_ignores_reports():
    w = make_window(17)
    net = N.make_net("meas_only", seed=1)
    net.forecaster = FC                                          # carried by the net (as ppo.train sets it)
    nu = np.array([0.2, 0.4, 0.1])
    a = E.run_window(w, net, ("usage", nu), truthful(), 2, MECH, SHAPE)
    b = E.run_window(w, net, ("usage", nu), [Multiplier(2.0), Multiplier(0.5), Truthful()], 2, MECH, SHAPE)
    np.testing.assert_array_equal(a["quota"], b["quota"])
    np.testing.assert_array_equal(a["on_time_Mbit"], b["on_time_Mbit"])
    assert (a["p"] == 0).all() and (a["p_fix"] == 0).all()
    np.testing.assert_array_equal(a["price"], nu[a["tenant"].to_numpy()] * a["quota"])
    with pytest.raises(ValueError, match="report-free"):
        E.run_window(w, net, "full", [BestResponder("full"), Truthful(), Truthful()], 2, MECH, SHAPE)


def test_ppo_evaluate_matches_episode_welfare():
    w = make_window(18, K=14)
    net = N.make_net("protected", seed=7)
    ev = P.evaluate(net, [w], d=2, mech=MECH, forecaster=FC)
    df = E.run_window(w, net, "full", truthful(), 2, MECH, SHAPE, forecaster=FC)
    W = df[df["scored"]].groupby("k")["realized_value"].sum().mean()
    assert W == pytest.approx(ev["welfare"], rel=1e-12)


def test_oracle_and_static_controllers_replay_their_own_paths():
    w = make_window(19, K=8, warmup=2)
    res = O.oracle_window(w, MECH, pop=16, iters=2, elite=4, seed=0)
    df = E.run_window(w, res.controller(), "full", truthful(), 2, MECH, SHAPE)
    np.testing.assert_array_equal(df["quota"].to_numpy().reshape(w.K, w.n), res.quotas)
    np.testing.assert_array_equal(df["on_time_Mbit"].to_numpy().reshape(w.K, w.n), res.on_time)
    fit = S.fit_static([w], MECH, steps=4)
    ds = E.run_window(w, fit, "full", truthful(), 2, MECH, SHAPE)
    W = ds[ds["scored"]].groupby("k")["realized_value"].sum().mean()
    assert W == pytest.approx(fit.welfare, rel=1e-12)


def test_metrics_interop():
    w = make_window(20)
    df = E.run_window(w, E.FixedWeights(), "full", truthful(), 2, MECH, SHAPE)
    m = EM.interval_metrics(df, price_col="price", pi=w.contract.pi)
    assert len(m) == len(df)
    s = EM.window_summary(df, w.warmup, price_col="price", keys=dict(window=w.id, arm="equal"))
    assert set(s["tenant"]) == {-1, 0, 1, 2}
    iw = EM.interval_welfare(df, price_col="price", B=MECH.B)
    assert (iw["idle_prb"] >= 0).all()
    # default price column of an episode table is the billed price (consistent with its utility columns)
    fx = E.run_window(w, E.RandomProtected(20), "fixed", truthful(), 2, MECH, SHAPE)
    sm = EM.window_summary(fx, w.warmup)
    allrow = sm[sm["tenant"] == -1].iloc[0]
    sc = fx[fx["scored"]]
    assert allrow["price"] == pytest.approx(sc.groupby("k")["p_fix"].sum().mean(), rel=1e-12)
    assert allrow["utility_declared"] == pytest.approx(allrow["V_declared"] - allrow["price"], rel=1e-12)


def test_late_counted_variant():
    w = make_window(22, K=10)
    a = E.run_window(w, E.FixedWeights(), "full", truthful(), 2, MECH, SHAPE)
    b = E.run_window(w, E.FixedWeights(), "full", truthful(), 2, MECH, SHAPE, late_counted=True)
    np.testing.assert_array_equal(a["quota"], b["quota"])        # the decision never sees the replay
    assert (b["dropped_Mbit"] == 0).all() and (a["late_Mbit"] == 0).all()
    assert b["on_time_Mbit"].sum() <= a["on_time_Mbit"].sum() + 1e-9   # expiry never delivers less on time
    assert E.episode_checks(b, w.contract, MECH)["conservation_max"] < 1e-9
    assert b.attrs["late_counted"] is True


def test_billed_price_columns_only():
    w = make_window(23, K=8)
    a = E.run_window(w, E.RandomProtected(23), "fixed", truthful(), 2, MECH, SHAPE)
    b = E.run_window(w, E.RandomProtected(23), "fixed", truthful(), 2, MECH, SHAPE, price_columns="billed")
    assert b["p"].isna().all() and b["T"].isna().all()
    np.testing.assert_array_equal(a["price"], b["price"])
    np.testing.assert_array_equal(a["utility_declared"], b["utility_declared"])
    with pytest.raises(ValueError, match="price_columns"):
        E.run_window(w, E.FixedWeights(), "full", truthful(), 2, MECH, SHAPE, price_columns="some")


def test_bandit_evaluator_drives_learning_tenants():
    from slicemech.tenants import bandits as BD
    wins = [make_window(24, K=6, warmup=1), make_window(25, K=6, warmup=1)]
    ev = E.make_bandit_evaluator(wins, E.RandomProtected(1), "fixed", 2, MECH, SHAPE)
    out = ev(wins[0].id, np.ones(3))
    ref = E.run_window(wins[0], E.RandomProtected(1), "fixed", truthful(), 2, MECH, SHAPE)
    sc = ref[ref["scored"]]
    np.testing.assert_allclose(out["utility"], sc.groupby("tenant")["utility_declared"].sum().to_numpy(), rtol=1e-12)
    memo = BD.MemoizedEvaluator(ev)
    run = BD.run_learning_tenants(memo, [w.id for w in wins], n=3, algo="eps", episodes=6, seed=0)
    assert run.final_greedy().shape == (3,)
    assert 1 <= memo.calls <= 6 + len(wins)                     # episodes plus one truthful baseline per window


# ============================================================================================ invalid inputs
class BadPolicy(Truthful):
    def report(self, i, theta, C, history=None, game=None):
        return float(C) * 2.0


def test_invalid_inputs_raise():
    w = make_window(21, K=4, warmup=1)
    with pytest.raises(ValueError, match="one tenant policy"):
        E.run_window(w, E.FixedWeights(), "full", truthful(2), 2, MECH, SHAPE)
    for d in (0, True, 1.5):
        with pytest.raises(ValueError, match="delay"):
            E.run_window(w, E.FixedWeights(), "full", truthful(), d, MECH, SHAPE)
    with pytest.raises(ValueError, match="price rule"):
        E.run_window(w, E.FixedWeights(), "auction", truthful(), 2, MECH, SHAPE)
    with pytest.raises(ValueError, match="report space"):
        E.run_window(w, E.FixedWeights(), "full", [BadPolicy(), Truthful(), Truthful()], 2, MECH, SHAPE)
    with pytest.raises(ValueError, match="slots"):
        E.run_window(w, E.FixedWeights(), "full", truthful(), 2, Mechanism(Delta=0.0005), SHAPE)
    with pytest.raises(TypeError):
        E.run_window(w, object(), "full", truthful(), 2, MECH, SHAPE)
    bad_shape = Shape(beta=(np.array([1.0]),) * 2, rho=(np.array([1.0]),) * 2)
    with pytest.raises(ValueError, match="shape"):
        E.run_window(w, E.FixedWeights(), "full", truthful(), 2, MECH, bad_shape)


# ============================================================================================ real data
def test_real_train_window_smoke():
    calib = pytest.importorskip("slicemech.data.calibrate")
    try:
        cal = calib.load_calibration("n3_default")
        w = cal.window("train", "2024-01-14", 60, K=30, warmup=5)
    except (FileNotFoundError, OSError) as exc:                 # prepared data not available
        pytest.skip(f"prepared TRAIN data missing: {exc}")
    df = E.run_window(w, E.RandomProtected(0), "full", truthful(), 2, MECH, cal.shape)
    chk = E.episode_checks(df, w.contract, MECH)
    assert chk["conservation_max"] < 1e-9 and chk["cert_violation_p"] <= 0 and chk["budget_excess"] <= 0
    assert df["need"].any()
