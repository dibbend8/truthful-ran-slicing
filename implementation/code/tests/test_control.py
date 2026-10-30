"""Tests for slicemech.control (INTERFACES.md §9; experiment_design.md §4 and §8 "Training isolation").

Stubs (local to this file): a ``StubWindow`` with the fields of ``data.Window``, a stateless replay env
(on-time Mbit = min(r Q Delta, demand)) and a simple quota rule. Integration tests against the real
``model.allocation`` / ``sim.fluid`` modules are skipped when those modules are missing.
"""
from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

from slicemech.config import Mechanism, contract_from_params  # noqa: E402
from slicemech.control import features as F  # noqa: E402
from slicemech.control import nets as N  # noqa: E402
from slicemech.control import oracle as O  # noqa: E402
from slicemech.control import ppo as P  # noqa: E402
from slicemech.control import retrain as RT  # noqa: E402
from slicemech.control import static as S  # noqa: E402

CLASSES = ("video", "urllc", "mmtc")


# ============================================================================= stubs
def _install_scores_stub():
    """Minimal stand-in for slicemech.model.scores, used only if the real module is absent."""
    mod = types.ModuleType("slicemech.model.scores")

    def _sm(y):
        e = np.exp(y - y.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

    class ProtectedMap:
        monotone_by_construction = True

        def __init__(self, a, kappa, mu):
            self.a, self.kappa, self.mu = map(lambda v: np.asarray(v, float), (a, kappa, mu))
            assert np.all(self.kappa >= 0)

        def weights(self, z):
            return _sm(self.a + self.kappa * np.log1p(np.asarray(z, float) / self.mu))

    class FixedMap:
        monotone_by_construction = True

        def __init__(self, w):
            self.w = np.asarray(w, float)

        def weights(self, z):
            return np.broadcast_to(self.w, np.shape(z)).copy()

    class CallableMap:
        def __init__(self, fn, monotone=False):
            self.fn, self.monotone_by_construction = fn, monotone

        def weights(self, z):
            return np.asarray(self.fn(np.asarray(z, float)))

    mod.ProtectedMap, mod.FixedMap, mod.CallableMap, mod.softmax = ProtectedMap, FixedMap, CallableMap, _sm
    sys.modules["slicemech.model.scores"] = mod


try:
    from slicemech.model import scores as _scores  # noqa: F401
    HAVE_SCORES = True
except ImportError:
    _install_scores_stub()
    HAVE_SCORES = False


@dataclass
class StubWindow:
    """Fields of data.Window (INTERFACES.md §8)."""

    id: str
    split: str
    date: str
    n: int
    cls: tuple
    arrivals_ms: np.ndarray
    theta: np.ndarray
    measured: np.ndarray
    rate_true: np.ndarray
    rate_hat: np.ndarray
    warmup: int
    contract: object
    meta: dict


class PoisonedWindow:
    """A window whose reports explode on access: proves a consumer never reads them."""

    def __init__(self, w):
        self.__dict__["_w"] = w

    def __getattr__(self, name):
        if name == "theta":
            raise AssertionError("theta (reports) was read")
        return getattr(self.__dict__["_w"], name)


def make_window(seed, K=30, mu=(10.0, 10.0, 10.0), level=2.0, split="train", warmup=3, slots=0, noise=0.1):
    """Stub window: demand ~ level * mu * (1 + noise N(0,1)) per interval (Mbit/s), rate 1 Mbit/s/PRB."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float)
    n = len(mu)
    cls = tuple(CLASSES[i % 3] for i in range(n))
    ct = contract_from_params(list(cls), mu)
    m = np.clip(level * mu * (1 + noise * rng.standard_normal((K, n))), 0, ct.C)
    r = np.ones((K, n))
    arr = (np.repeat(m / 1000.0, slots, axis=0).astype(np.float32) if slots else np.zeros((0, n), np.float32))
    return StubWindow(f"w{seed}", split, "2024-01-01", n, cls, arr, m.copy(), m, r, r.copy(), warmup, ct,
                      dict(tod_start_s=3600.0 * (seed % 24)))


class StubEnv:
    """Stateless replay: on-time Mbit = min(r_true Q Delta, demand Delta) (demand = window.measured).
    The lookahead methods follow ``ppo.FluidEnv`` (without carry-over the continuation is action-independent)."""

    def __init__(self, window, mech):
        self.m = np.asarray(window.measured, float) * mech.Delta
        self.r = np.asarray(window.rate_true, float) * mech.Delta
        self.k = 0
        self._Q = None

    def replay(self, Q):
        K = len(Q)
        self._Q = np.asarray(Q, float)
        return np.minimum(np.asarray(Q) * self.r[:K], self.m[:K])

    def counterfactual_at(self, ks, Q):
        return np.minimum(np.asarray(Q) * self.r[ks][:, None, :], self.m[ks][:, None, :])

    def lookahead_at(self, ks, Q, h):
        S, M, n = np.shape(Q)
        out = np.zeros((S, M, h + 1, n))
        out[:, :, 0] = self.counterfactual_at(ks, Q)
        for j in range(1, h + 1):
            for s, k in enumerate(ks):
                if k + j < len(self._Q):
                    out[s, :, j] = np.minimum(self._Q[k + j] * self.r[k + j], self.m[k + j])
        return out

    def lookahead(self, Q, Q_next):
        Q = np.asarray(Q, float)
        out = np.zeros(Q.shape[:-1] + (len(Q_next) + 1, Q.shape[-1]))
        out[..., 0, :] = self.counterfactual(Q)
        for j, q in enumerate(np.asarray(Q_next, float), start=1):
            out[..., j, :] = np.minimum(q * self.r[self.k + j], self.m[self.k + j])
        return out

    def reset(self):
        self.k = 0

    def counterfactual(self, Q):
        return np.minimum(np.asarray(Q) * self.r[self.k], self.m[self.k])

    def advance(self, Q):
        out = np.minimum(np.asarray(Q) * self.r[self.k], self.m[self.k]); self.k += 1
        return out


def stub_quota(z, w, r, ct, mech):
    """Weighted split of B capped by the need cap min(eta z, M)/r, floored to integers (PRB)."""
    cap = np.minimum(mech.eta * z, ct.M) / r
    x = np.minimum(mech.B * w / w.sum(-1, keepdims=True), cap)
    return np.floor(x + 1e-9).astype(np.int64)


MECH = Mechanism(B=30, gamma=0.2, eta=2.0, Delta=1.0, integer=True)
STUB = dict(env_factory=StubEnv, quota_fn=stub_quota)
FC = F.OperatorAR.default()          # explicit opt-in to the untrained operator forecaster (no silent default)


def small_cfg(**kw):
    base = dict(iterations=3, windows_per_iter=2, M=4, minibatch=64, eval_every=0)
    base.update(kw)
    return P.PPOConfig(**base)


# ============================================================================= features
def test_context_shapes_and_vectorised_equivalence():
    w = make_window(0)
    ctx = F.context(w, 10, 2, w.contract, mech=MECH, forecaster=FC)
    c_t, c_g = ctx
    assert c_t.shape == (3, F.N_TENANT_FEATURES) == (3, len(F.TENANT_FEATURES)) and c_g.shape == (F.N_GLOBAL_FEATURES,)
    assert np.all(np.isfinite(c_t)) and np.all(np.isfinite(c_g))
    wf = F.window_features(w, 2, w.contract, mech=MECH, forecaster=FC)
    for k in (0, 1, 2, 10, 29):
        c = F.context(w, k, 2, w.contract, mech=MECH, forecaster=FC)
        assert np.array_equal(wf.c_tenant[k], c[0]) and np.array_equal(wf.c_global[k], c[1])
        assert np.array_equal(wf.f[k], c.f)


@pytest.mark.parametrize("name", ["reports", "z", "theta", "stored_reports", "quotas", "prices", "deliveries"])
def test_context_refuses_reports(name):
    w = make_window(1)
    with pytest.raises(F.ReportLeakError):
        F.context(w, 5, 2, w.contract, **{name: w.theta[5]})
    with pytest.raises(F.ReportLeakError):
        F.window_features(w, 2, w.contract, **{name: w.theta})


def test_context_refuses_report_carriers():
    w = make_window(2)
    with pytest.raises(F.ReportLeakError):
        F.context(w.theta, 5, 2, w.contract)                     # an array of reports instead of a window
    import pandas as pd
    log = pd.DataFrame(dict(k=[0], tenant=[0], report=[1.0]))
    with pytest.raises(F.ReportLeakError):
        F.context(log, 0, 1, w.contract)

    class Logged(StubWindow):
        pass

    lw = Logged(**vars(w)); lw.quotas = np.zeros((30, 3))
    with pytest.raises(F.ReportLeakError):
        F.context(lw, 5, 2, w.contract)
    with pytest.raises(TypeError):
        F.context(w, 5, 2, w.contract, unknown_kw=1)


def test_context_never_reads_reports_and_is_causal():
    w = make_window(3)
    k, d = 12, 2
    ref = F.context(w, k, d, w.contract, mech=MECH, forecaster=FC)
    pw = F.context(PoisonedWindow(w), k, d, w.contract, mech=MECH, forecaster=FC)      # theta access would raise
    assert np.array_equal(ref[0], pw[0]) and np.array_equal(ref[1], pw[1])
    w2 = replace(w, theta=w.theta * 3.0)
    assert np.array_equal(F.context(w2, k, d, w.contract, mech=MECH, forecaster=FC)[0], ref[0])
    m = w.measured.copy(); m[k - d + 1:] *= 5.0                         # unreleased measurements
    assert np.array_equal(F.context(replace(w, measured=m), k, d, w.contract, mech=MECH, forecaster=FC)[0], ref[0])
    m = w.measured.copy(); m[k - d] *= 5.0                              # the newest released one
    assert not np.array_equal(F.context(replace(w, measured=m), k, d, w.contract, mech=MECH, forecaster=FC)[0], ref[0])
    with pytest.raises(ValueError):
        F.context(w, k, 0, w.contract)                                  # d = 0 would see interval k itself


def test_operator_ar_fit_train_only_and_recovers_ar1():
    rng = np.random.default_rng(0)
    K, n = 4000, 3
    mu = np.array([10.0, 5.0, 1.0])
    x = np.ones((K, n))
    for t in range(1, K):
        x[t] = 1 + 0.8 * (x[t - 1] - 1) + 0.1 * rng.standard_normal(n)
    ct = contract_from_params(list(CLASSES), mu)
    w = StubWindow("ar", "train", "d", n, CLASSES, np.zeros((0, n)), x * mu, x * mu, np.ones((K, n)),
                   np.ones((K, n)), 0, ct, dict(tod_start_s=0.0))
    ar = F.fit_operator_ar([w], d=1)
    for c in CLASSES:   # target 1 + 0.8 (x - 1): intercept 0.2, first lag 0.8
        assert abs(ar.coef[c][1] - 0.8) < 0.05 and abs(ar.coef[c][0] - 0.2) < 0.05
        assert np.all(np.abs(ar.coef[c][2:]) < 0.05)
    with pytest.raises(ValueError):
        F.fit_operator_ar([replace(w, split="dev")], d=1)
    with pytest.raises(ValueError):
        F.context(w, 10, 2, ct, forecaster=ar)                          # fitted for d=1


# ============================================================================= nets
def _ctx_batch(seed, B=16, n=3, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (scale * torch.randn((B, n, F.N_TENANT_FEATURES), generator=g, dtype=N.DTYPE),
            scale * torch.randn((B, F.N_GLOBAL_FEATURES), generator=g, dtype=N.DTYPE))


def test_protected_head_bounds():
    net = N.make_net("protected", seed=0)
    for scale in (1.0, 30.0, 1e3):
        c_t, c_g = _ctx_batch(1, scale=scale)
        a, kappa = net.heads(c_t, c_g)
        assert torch.all(kappa >= 0) and torch.all(kappa <= N.KAPPA_SCALE)
        assert torch.all(a.abs() <= N.A_SCALE)
    k0 = N.make_net("kappa0", seed=0)
    _, kz = k0.heads(*_ctx_batch(2))
    assert torch.all(kz == 0)


def test_protected_map_passes_ratio_condition():
    w = make_window(4)
    rng = np.random.default_rng(0)
    for seed in range(4):
        net = N.make_net("protected", seed=seed)
        for k in (3, 17):
            ctx = F.context(w, k, 2, w.contract, mech=MECH, forecaster=FC)
            pm = net.priority_map(ctx)
            assert pm.monotone_by_construction
            a, kappa = net.heads_np(ctx)
            assert np.all(kappa >= 0)
            ct = w.contract
            for i in range(3):
                z = rng.uniform(0.1, 1.0, 3) * ct.C
                u = np.linspace(0.0, ct.C[i], 513)[1:]
                Z = np.repeat(z[None], len(u), 0); Z[:, i] = u
                Wt = pm.weights(Z)
                for j in range(3):
                    if j != i:
                        lr = np.log(Wt[:, j]) - np.log(Wt[:, i])
                        assert np.all(np.diff(lr) <= 1e-12), "opponent/own ratio rose with the own report"
                if HAVE_SCORES:
                    from slicemech.model.scores import ratio_violation
                    assert ratio_violation(pm, i, z, ct.C[i]) is None
            if HAVE_SCORES:
                from slicemech.model.scores import GeneralScoreMap
                assert GeneralScoreMap(a, np.diag(kappa), ct.mu).satisfies_ratio_condition()
            # deployed map == training forward (softmax of a + kappa log1p(z/mu))
            z = w.theta[k]
            with torch.no_grad():
                y = net(torch.as_tensor(ctx[0])[None], torch.as_tensor(ctx[1])[None],
                        torch.as_tensor(z / ct.mu)[None])[0].numpy()
            assert np.allclose(pm.weights(z), N.softmax_np(y), rtol=1e-12, atol=1e-15)


def test_unconstrained_and_measonly_maps():
    w = make_window(5)
    ctx = F.context(w, 9, 2, w.contract, mech=MECH, forecaster=FC)
    un = N.make_net("unconstrained", seed=1)
    cm = un.priority_map(ctx)
    assert cm.monotone_by_construction is False
    Z = w.theta[5:9]
    with torch.no_grad():
        y = un(torch.as_tensor(np.repeat(ctx[0][None], 4, 0)), torch.as_tensor(np.repeat(ctx[1][None], 4, 0)),
               torch.as_tensor(Z / w.contract.mu)).numpy()
    assert np.allclose(cm.weights(Z), N.softmax_np(y), rtol=1e-12)
    mo = N.make_net("meas_only", seed=1)
    fm = mo.priority_map(ctx)
    W1, W2 = fm.weights(w.theta[3]), fm.weights(2 * w.theta[3])
    assert np.array_equal(W1, W2)
    assert np.array_equal(mo.alloc_reports(ctx, w.theta[3]), ctx.f)
    assert mo.n_parameters() < 25_000 and un.n_parameters() < 25_000


def test_encoder_permutation_equivariance_and_seeding():
    net = N.make_net("protected", seed=3)
    c_t, c_g = _ctx_batch(7, B=5, n=4)
    zn = torch.rand((5, 4), dtype=N.DTYPE)
    perm = torch.tensor([2, 0, 3, 1])
    y = net(c_t, c_g, zn)
    yp = net(c_t[:, perm], c_g, zn[:, perm])
    assert torch.allclose(y[:, perm], yp, atol=1e-12)
    state = torch.get_rng_state()
    a, b = N.make_net("protected", seed=11), N.make_net("protected", seed=11)
    assert torch.equal(torch.get_rng_state(), state)                    # global RNG untouched
    assert N.param_digest(a) == N.param_digest(b) != N.param_digest(N.make_net("protected", seed=12))


# ============================================================================= PPO
def test_ppo_same_seed_same_parameters():
    ws = [make_window(s) for s in range(3)]
    net = N.make_net("protected", seed=0)
    r1 = P.train(net, ws, d=2, mech=MECH, config=small_cfg(), seed=5, forecaster=FC, **STUB)
    r2 = P.train(net, ws, d=2, mech=MECH, config=small_cfg(), seed=5, forecaster=FC, **STUB)
    r3 = P.train(net, ws, d=2, mech=MECH, config=small_cfg(), seed=6, forecaster=FC, **STUB)
    assert N.param_digest(r1.net) == N.param_digest(r2.net)
    assert N.param_digest(r1.net) != N.param_digest(r3.net)
    assert N.param_digest(r1.net) != N.param_digest(net)                # it trained
    assert r1.history == r2.history or all(
        {k: v for k, v in a.items() if k != "seconds"} == {k: v for k, v in b.items() if k != "seconds"}
        for a, b in zip(r1.history, r2.history))


@pytest.mark.parametrize("kind", ["protected", "unconstrained", "meas_only"])
def test_ppo_improves_toy_bandit(kind):
    """Toy bandit: B = 30, demands 20 Mbit/s each, r = 1; tariffs (1, 3, 0.5). Equal weights earn 45 credits
    (normalised reward 1.0); the optimum gives URLLC its 20 PRB and earns 70 (1.56)."""
    ws = [make_window(s, K=40, warmup=2) for s in range(4)]
    dev = [make_window(100 + s, K=40, warmup=2, split="dev") for s in range(2)]
    net = N.make_net(kind, seed=0)
    before = P.evaluate(net, dev, d=1, mech=MECH, forecaster=FC, **STUB)
    cfg = P.PPOConfig(iterations=40, windows_per_iter=4, M=4, minibatch=256, lr_start=3e-3, lr_end=1e-3,
                      eval_every=10)
    res = P.train(net, ws, dev, d=1, mech=MECH, config=cfg, seed=0, forecaster=FC, **STUB)
    after = P.evaluate(res.net, dev, d=1, mech=MECH, forecaster=FC, **STUB)
    assert after["welfare"] > before["welfare"] + 5.0, (before, after)
    assert res.best_dev_welfare == pytest.approx(after["welfare"])
    assert res.best_iteration in (10, 20, 30, 40)
    assert all(math.isfinite(h["sample_reward"]) for h in res.history)
    assert all(0.05 - 1e-12 <= h["sigma"] <= 1.0 + 1e-12 for h in res.history)


def test_ppo_inert_intervals_skip_update():
    """If weights never matter (demand below every cap), all M samples tie and parameters stay put."""
    ws = [make_window(s, level=0.01) for s in range(2)]
    net = N.make_net("protected", seed=0)
    res = P.train(net, ws, d=1, mech=MECH, config=small_cfg(), seed=0, forecaster=FC, **STUB)
    assert all(h["skipped"] for h in res.history)
    assert N.param_digest(res.net) == N.param_digest(net)


# ============================================================================= retraining
def _perturbed(reports, seed, C):
    rng = np.random.default_rng(seed)
    return np.minimum(reports * rng.uniform(0.3, 2.5, reports.shape), C)


@pytest.mark.parametrize("kind", ["protected", "unconstrained", "meas_only"])
def test_measurement_retraining_is_bit_identical_under_report_perturbation(kind):
    ws = [make_window(10 + s, split="test") for s in range(2)]
    base = N.make_net(kind, seed=0)
    cfg = small_cfg(iterations=4)
    stored_a = [w.theta.copy() for w in ws]
    stored_b = [_perturbed(w.theta, 1, w.contract.C) for w in ws]
    ws_b = [replace(w, theta=_perturbed(w.theta, 2, w.contract.C)) for w in ws]
    kw = dict(d=2, mech=MECH, config=cfg, seed=7, forecaster=FC, report_rebuild="measured", **STUB)
    ra = RT.retrain(base, ws, "measurement", stored_a, **kw)
    rb = RT.retrain(base, ws_b, "measurement", stored_b, **kw)
    rc = RT.retrain(base, [PoisonedWindow(w) for w in ws], "measurement", None, **kw)
    assert N.param_digest(ra.net) == N.param_digest(rb.net) == N.param_digest(rc.net)
    assert N.param_digest(ra.net) != N.param_digest(base)              # it did train
    # the stored-report pipeline is report-dependent (the channel the paper closes), except for the
    # measurement-only controller, which never uses reports at all
    sa = RT.retrain(base, ws, "stored", stored_a, d=2, mech=MECH, config=cfg, seed=7, forecaster=FC, **STUB)
    sb = RT.retrain(base, ws, "stored", stored_b, d=2, mech=MECH, config=cfg, seed=7, forecaster=FC, **STUB)
    if kind == "meas_only":
        assert N.param_digest(sa.net) == N.param_digest(sb.net) == N.param_digest(ra.net)
    else:
        assert N.param_digest(sa.net) != N.param_digest(sb.net)
    # the base network is never modified
    assert N.param_digest(base) == N.param_digest(N.make_net(kind, seed=0))


def test_measurement_view_hides_reports():
    w = make_window(20)
    w.meta["stored_reports"] = w.theta
    v = RT.MeasurementView(w)
    with pytest.raises(RT.ReportAccessError):
        _ = v.theta
    assert "stored_reports" not in v.meta and "tod_start_s" in v.meta
    assert np.array_equal(v.measured, w.measured)
    with pytest.raises(AttributeError):
        v.theta = 1
    with pytest.raises(ValueError):
        RT.retrain(N.make_net("protected"), [w], "stored", None, d=1, mech=MECH, forecaster=FC, **STUB)
    with pytest.raises(ValueError):
        RT.retrain(N.make_net("protected"), [w], "bogus", None, d=1, mech=MECH, forecaster=FC, **STUB)


# ============================================================================= static and oracle
def test_simplex_grid():
    G = S.simplex_grid(3, 20, 0.05)
    assert np.allclose(G.sum(1), 1) and np.all(G >= 0.05 - 1e-12)
    assert np.allclose(G[0], 1 / 3) and len(G) == 1 + 231
    with pytest.raises(ValueError):
        S.simplex_grid(3, 20, 0.4)


def test_fit_static_grid_and_cem():
    ws = [make_window(s, K=20, warmup=0) for s in range(3)]
    fit = S.fit_static(ws, MECH, **STUB)
    assert fit.method == "grid" and int(np.argmax(fit.w)) == 1          # URLLC carries the highest tariff
    eq = S.rollout_fixed(P.prepare(ws, mech=MECH), np.full((1, 3), 1 / 3), mech=MECH, **STUB)[0]
    assert fit.welfare == fit.scores.max() and fit.welfare > eq + 15.0          # equal weights earn ~45
    assert np.allclose(fit.priority_map().weights(ws[0].theta[0]), fit.w)
    ws4 = [make_window(s, K=10, warmup=0, mu=(10.0, 10.0, 10.0, 10.0)) for s in range(2)]
    cem = S.fit_static(ws4, MECH, pop=16, iters=5, elite=4, seed=0, **STUB)
    eq4 = S.rollout_fixed(P.prepare(ws4, mech=MECH), np.full((1, 4), 0.25), mech=MECH, **STUB)[0]
    assert cem.method == "cem" and cem.welfare >= eq4


def test_oracle_dominates_equal_weights_per_interval():
    w = make_window(30, K=15, warmup=2)
    res = O.oracle_window(w, MECH, pop=16, iters=4, elite=4, seed=0, **STUB)
    eq = StubEnv(w, MECH).replay(stub_quota(w.theta, np.full((15, 3), 1 / 3), w.rate_hat, w.contract, MECH))
    assert np.all(res.welfare >= eq @ w.contract.pi - 1e-9)
    assert res.mean_welfare > 60.0                                      # equal weights earn ~45
    ctrl = res.controller()
    ctx = F.context(w, 4, 1, w.contract, mech=MECH, forecaster=FC)
    assert np.allclose(ctrl.priority_map(ctx).weights(w.theta[4]), res.weights[4])
    res2 = O.oracle_window(w, MECH, pop=16, iters=4, elite=4, seed=0, **STUB)
    assert np.array_equal(res.weights, res2.weights)


# ============================================================================= integration (real modules)
def _real_modules():
    alloc = pytest.importorskip("slicemech.model.allocation")
    fluid = pytest.importorskip("slicemech.sim.fluid")
    return alloc, fluid


def _real_window(seed, K=12, split="train"):
    rng = np.random.default_rng(seed)
    mu = np.array([12.0, 6.0, 2.0])
    ct = contract_from_params(list(CLASSES), mu)
    per_ms = np.repeat(mu * rng.gamma(4.0, 0.25, (K, 3)), 1000, axis=0) / 1000.0
    per_ms *= rng.gamma(2.0, 0.5, per_ms.shape)                        # bursty ms-level arrivals
    m = per_ms.reshape(K, 1000, 3).sum(1)
    r = rng.uniform(0.15, 0.3, (K, 3))
    return StubWindow(f"r{seed}", split, "2024-01-01", 3, CLASSES, per_ms.astype(np.float32),
                      np.minimum(m, ct.C), m, r, np.r_[r[:1], r[:-1]], 2, ct, dict(tod_start_s=0.0))


def test_fluid_env_window_path_equals_stepwise():
    alloc, _ = _real_modules()
    w = _real_window(0)
    mech = Mechanism()
    Q = alloc.integer(w.theta, np.ones(3), w.rate_hat, w.contract, mech)
    env = P.FluidEnv(w, mech)
    on_path = env.replay(Q)
    Qs = np.stack([Q[3:9] + 1, np.maximum(Q[3:9] - 2, 0)], axis=1)        # (6, 2, n)
    cf = env.counterfactual_at(np.arange(3, 9), Qs)
    env.reset()
    for k in range(w.measured.shape[0]):
        if 3 <= k < 9:
            assert np.array_equal(env.counterfactual(Qs[k - 3]), cf[k - 3])
        assert np.array_equal(env.advance(Q[k]), on_path[k])


def test_ppo_smoke_with_real_allocator_and_fluid():
    _real_modules()
    ws = [_real_window(s) for s in range(2)]
    dev = [_real_window(9, split="dev")]
    cfg = P.PPOConfig(iterations=2, windows_per_iter=2, M=4, minibatch=32, eval_every=1)
    res = P.train(N.make_net("protected", seed=0), ws, dev, d=2, mech=Mechanism(), config=cfg, seed=0,
                  forecaster=FC)
    assert len(res.history) == 2 and all(math.isfinite(h["welfare"]) for h in res.history)
    assert res.best_dev_welfare is not None and res.best_iteration in (1, 2)
    ev = P.evaluate(res.net, dev, d=2, mech=Mechanism())                 # the trained net carries FC
    assert ev["welfare"] == pytest.approx(res.best_dev_welfare)
    fit = S.fit_static(ws, Mechanism(), steps=4)
    assert fit.w.shape == (3,) and math.isfinite(fit.welfare)
    orc = O.oracle_window(ws[0], Mechanism(), pop=8, iters=2, elite=2)
    assert np.all(np.isfinite(orc.welfare))


def test_measurement_retraining_bit_identical_with_real_modules():
    _real_modules()
    ws = [_real_window(s, split="test") for s in range(2)]
    base = N.make_net("protected", seed=0)
    cfg = P.PPOConfig(iterations=2, windows_per_iter=2, M=4, minibatch=32, eval_every=0)
    kw = dict(d=2, mech=Mechanism(), config=cfg, seed=1, forecaster=FC)
    a = RT.retrain(base, ws, "measurement", [w.theta for w in ws], report_rebuild="measured", **kw)
    lied = [PoisonedWindow(replace(w, theta=np.minimum(2.0 * w.theta, w.contract.C))) for w in ws]
    b = RT.retrain(base, lied, "measurement", [0.5 * w.theta for w in ws], report_rebuild="measured", **kw)
    assert N.param_digest(a.net) == N.param_digest(b.net) != N.param_digest(base)
    s1 = RT.retrain(base, ws, "stored", [w.theta for w in ws], **kw)
    s2 = RT.retrain(base, ws, "stored", [0.5 * w.theta for w in ws], **kw)
    assert N.param_digest(s1.net) != N.param_digest(s2.net)


# ============================================================================= regressions (code review)
def test_unconstrained_map_is_a_snapshot():
    w = make_window(40)
    ctx = F.context(w, 6, 1, w.contract, mech=MECH, forecaster=FC)
    net = N.make_net("unconstrained", seed=2)
    pm = net.priority_map(ctx)
    before = pm.weights(w.theta[6])
    with torch.no_grad():
        for prm in net.parameters():
            prm.add_(0.5)
    assert np.array_equal(pm.weights(w.theta[6]), before)
    assert not np.allclose(net.priority_map(ctx).weights(w.theta[6]), before)


def test_oracle_windows_uses_each_windows_stored_reports():
    ws = [make_window(50 + s, K=8, warmup=0) for s in range(2)]
    stored = [w.theta.copy() for w in ws]
    stored[1] = np.minimum(stored[1] * 0.25, ws[1].contract.C)          # window 1 under-reports
    res = O.oracle_windows(ws, MECH, report_source=stored, pop=8, iters=2, elite=2, **STUB)
    one = O.oracle_window(ws[1], MECH, report_source=[stored[1]], pop=8, iters=2, elite=2, **STUB)
    assert np.array_equal(res[1].quotas, one.quotas)
    with pytest.raises(ValueError):
        O.oracle_window(ws[0], MECH, report_source=stored, **STUB)
    with pytest.raises(ValueError):
        O.oracle_windows(ws, MECH, report_source=stored[:1], **STUB)


def test_fluid_env_rejects_bad_interval_indices():
    alloc, _ = _real_modules()
    w = _real_window(1)
    env = P.FluidEnv(w, Mechanism())
    with pytest.raises(RuntimeError):
        env.counterfactual_at(np.array([0]), np.zeros((1, 1, 3)))      # no replay path yet
    env.replay(alloc.integer(w.theta, np.ones(3), w.rate_hat, w.contract, Mechanism()))
    with pytest.raises(IndexError):
        env.counterfactual_at(np.array([-1]), np.zeros((1, 1, 3)))
