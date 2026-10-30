"""Regression tests for the round-2 adversarial review (CHANGELOG.md, "fixer, round 2").

One section per confirmed finding:

* R1  E5 pool parts were time halves of nonstationary captures -> interleaved 2-s block halves, per-date phase flip,
      pre-registered part-balance check, alpha = 1 reference;
* R2  direct rule / general score map depended on the batch layout (BLAS products) -> batch-independent sums; the
      price engine takes the own quota from its own threshold grid;
* R3  fixed-price best response misreported for a noise "gain" -> ties within the prices' own error bounds;
* R4  water fill overflowed for weight ratios above ~1e306 -> exact power-of-two row scaling, documented boundary;
* R5  fractional rules claimed exact best responses under 'usage'/'fixed' -> kink candidates, ``exact=False``;
* R6  ``Window.released`` accepted d = 0;
* R7  forecaster fits could be run on DEV/TEST series without an error;
* R8  ``compose_parts`` did not check the date's split;
* R9  the freeze gate accepted a renamed draft;
* R10 the E5 protocol (4 + 4 windows) could not run on DEV (6 windows per date);
* R11 manipulation harm measured by tariff-weighted total welfare -> honest tenants' own outcomes.

Tests on real data (TRAIN/DEV only; TEST is never touched) skip when the prepared data are missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _core_fixtures import random_case, random_shape
from slicemech.config import Contract, Mechanism, Shape
from slicemech.model import allocation as AL
from slicemech.model.direct import DirectRule
from slicemech.model.scores import (MAX_SCORE_SPREAD, FixedMap, GeneralScoreMap, ProtectedMap, softmax)
from slicemech.pricing.envelope import fixed_priority_prices, full_prices
from slicemech.pricing.response import best_response, utilities


def _load_cal():
    calib = pytest.importorskip("slicemech.data.calibrate")
    try:
        return calib.load_calibration("n3_default")
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"calibration missing: {exc}")


def _real_window(cal, split, date, offset, **kw):
    try:
        return cal.window(split, date, offset, **kw)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"prepared data missing: {exc}")


# ====================================================================================== R1 interleaved E5 halves
def test_r1_pool_halves_are_interleaved_partitions():
    from slicemech.data import tractor
    try:
        full = tractor.pool("dev", "urllc") + tractor.pool("dev", "mmtc")
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"prepared data missing: {exc}")
    for t in full:
        a = tractor.part_trace(t.stem, "A")
        b = tractor.part_trace(t.stem, "B")
        ma = tractor.part_mask(len(t), "A")
        mb = tractor.part_mask(len(t), "B")
        assert not np.any(ma & mb) and np.all(ma | mb)                       # disjoint and exhaustive
        np.testing.assert_array_equal(a.bytes_ms, t.bytes_ms[ma])
        np.testing.assert_array_equal(b.bytes_ms, t.bytes_ms[mb])
        blocks = np.flatnonzero(np.diff(ma.astype(int)) != 0) + 1               # 2-s alternation over the whole trace
        assert blocks.size >= len(t) // 2000 - 1 and np.all(np.diff(blocks) == 1000 * tractor.PART_BLOCK_S)
        assert a.stem == f"{t.stem}#A" and tractor.capture_of(b.stem) == t.stem
    with pytest.raises(ValueError):
        tractor.part_mask(10, "C")
    with pytest.raises(ValueError):
        tractor.part_mask(10, "A", block_s=0)


def test_r1_halves_are_balanced_on_train_and_dev():
    """The pre-registered check: every class mean of each half within 10 % of the pooled mean on TRAIN and DEV.
    The round-1 time halves failed it badly (DEV mMTC 0.06 / 1.94, TRAIN URLLC 0.72 / 1.28)."""
    from slicemech.data import raca, tractor
    try:
        bal = {s: tractor.part_balance(s) for s in ("train", "dev")}
        rbal = {s: raca.part_balance(s) for s in ("train", "dev")}
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"prepared data missing: {exc}")
    for s, b in bal.items():
        assert b["balanced"] and b["block_s"] == tractor.PART_BLOCK_S == 2, (s, b)
        for c, row in b["classes"].items():
            for p in ("A", "B"):
                assert abs(row[f"{p}_ratio"] - 1.0) <= tractor.PART_BALANCE_TOL, (s, c, p, row)
                assert row[f"{p}_wrap_prob"] >= row["wrap_prob"]                 # halves are shorter: more wraps
        assert rbal[s]["balanced"], rbal[s]
    # the selection rule (smallest of {1, 2, 5, 10} s passing on TRAIN and DEV) gives 2 s: 1 s fails DEV mMTC
    one = tractor.part_balance("dev", block_s=1)
    assert not one["balanced"] and not one["classes"]["mmtc"]["balanced"]
    # the old time halves, recomputed: the shift the review found
    old = [tr.bytes_ms for tr in tractor.pool("dev", "mmtc")]
    first = sum(x[: len(x) // 2].sum() for x in old) / sum(len(x) // 2 for x in old)
    pooled = sum(x.sum() for x in old) / sum(len(x) for x in old)
    assert first / pooled < 0.1


def test_r1_part_balance_on_test_needs_the_freeze(monkeypatch):
    from slicemech.data import raca, tractor
    real_t, real_r = tractor.pool, raca.pool

    def no_test_t(split, *a, **k):
        assert split != "test", "TEST TRACTOR statistics were computed"
        return real_t(split, *a, **k)

    def no_test_r(split, *a, **k):
        assert split != "test", "TEST Raca statistics were computed"
        return real_r(split, *a, **k)

    monkeypatch.setattr(tractor, "pool", no_test_t)
    monkeypatch.setattr(raca, "pool", no_test_r)
    with pytest.raises(PermissionError):
        tractor.part_balance("test")
    with pytest.raises(PermissionError):
        raca.part_balance("test")


def test_r1_e5_phases_use_disjoint_halves_that_alternate_with_the_date():
    from slicemech.data import compose, puffer
    assert compose.physical_part("2024-02-28", None) is None
    with pytest.raises(ValueError):
        compose.physical_part("2024-02-28", "C")
    for day in ("2024-02-28", "2024-03-02"):
        a, b = compose.physical_part(day, "A"), compose.physical_part(day, "B")
        assert {a, b} == {"A", "B"} and (a == "B") == compose.part_flip(day)
    tests = puffer.test_dates()
    assert sum(compose.part_flip(d) for d in tests) == len(tests) // 2         # 8 of 16 TEST dates swap
    cal = _load_cal()
    for day in ("2024-02-28", "2024-03-02"):                                   # no flip / flip
        wa = _real_window(cal, "dev", day, 60, K=30, warmup=5, pool_part="A")
        wb = _real_window(cal, "dev", day, 43260, K=30, warmup=5, pool_part="B")
        ha, hb = wa.meta["pool_half"], wb.meta["pool_half"]
        assert {ha, hb} == {"A", "B"} and ha == compose.physical_part(day, "A")
        assert "-pA-" in wa.id and wa.meta["pool_part"] == "A" and wa.meta["part_block_s"] == 2
        users_a = {s for t in wa.meta["tenants"] if "users" in t for s, _ in t["users"]}
        users_b = {s for t in wb.meta["tenants"] if "users" in t for s, _ in t["users"]}
        assert users_a and all(s.endswith(f"#{ha}") for s in users_a)
        assert users_b and all(s.endswith(f"#{hb}") for s in users_b)
        runs_a = {t["cqi_run"]["run"] for t in wa.meta["tenants"]}
        runs_b = {t["cqi_run"]["run"] for t in wb.meta["tenants"]}
        assert not runs_a & runs_b


# ====================================================================================== R2 batch independence
@pytest.mark.parametrize("n", [3, 6, 9, 12])
def test_r2_rules_are_bitwise_independent_of_the_batch(n):
    """A profile gets the same quotas whether it is evaluated alone (1-D or one row) or inside a large batch."""
    rng = np.random.default_rng(40 + n)
    for _ in range(4):
        c = random_case(rng, n, zero_prob=0.05, load=float(rng.uniform(0.5, 3.0)))
        ct, mech, r = c["ct"], c["mech"], c["r"]
        shape = random_shape(rng, n, max_atoms=64, uniform=False)
        Lam = rng.normal(0.0, 1.0, (n, n))
        np.fill_diagonal(Lam, np.abs(np.diag(Lam)) + np.abs(Lam).max())            # ratio condition holds
        policies = (DirectRule(ct, mech, r, shape), GeneralScoreMap(rng.uniform(-3, 3, n), Lam, c["mu"]),
                    ProtectedMap(rng.uniform(-3, 3, n), rng.uniform(0, 2, n), c["mu"]), FixedMap(c["w"]))
        Z = ct.C * rng.uniform(0.0, 1.0, (65 * n, n))
        for pol in policies:
            rule = AL.QuotaRule(ct, mech, r, pol, integer=False)
            big = rule.fractional(Z)
            for k in range(0, Z.shape[0], 7):
                np.testing.assert_array_equal(rule.fractional(Z[k:k + 1])[0], big[k])
                np.testing.assert_array_equal(rule.fractional(Z[k]), big[k])
            if hasattr(pol, "scores"):
                np.testing.assert_array_equal(pol.scores(Z[:1])[0], pol.scores(Z)[0])


def _flip16():
    """The review's knife-edge direct-rule instance (scratchpad r2/dump_flip.py, seed 11, iteration 16)."""
    from slicemech.model.allocation import FLOOR_EPS
    rng = np.random.default_rng(11)
    for it in range(17):
        n = int(rng.integers(3, 8))
        c = random_case(rng, n, zero_prob=0.0, load=float(rng.uniform(1.0, 3)))
        ct, mech, r = c["ct"], c["mech"], c["r"]
        shape = random_shape(rng, n, max_atoms=64, uniform=False)
        d = DirectRule(ct, mech, r, shape)
        z = c["z"].copy()
        i = int(rng.integers(n))

        def xi(u):
            zz = z.copy()
            zz[i] = u
            return d.fractional(zz[None, :])[0, i]
        lo, hi = 0.0, ct.C[i]
        q = int(np.floor(xi(hi) + FLOOR_EPS))
        if q < 2:
            continue
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if xi(mid) >= q - FLOOR_EPS:
                hi = mid
            else:
                lo = mid
        if it == 16:
            z[i] = 11.031198319785181
            return ct, mech, r, shape, z, i
    raise AssertionError("instance not reproduced")


def test_r2_direct_rule_knife_edge_prices_without_error():
    """Before: x_i = 14.999999998999998 in a 1-row call and 14.999999999 in a 325-row call, so the floors were 14
    and 15 and full_prices raised 'own quota at the report disagrees with the threshold grid'."""
    ct, mech, r, shape, z, i = _flip16()
    d = DirectRule(ct, mech, r, shape)
    one = d.fractional(z[None, :])[0]
    many = d.fractional(np.repeat(z[None, :], 65 * ct.n, axis=0))
    assert np.all(many == one)
    assert abs(one[i] - 15.0) < 1e-8                                             # a genuine knife edge
    rule = AL.QuotaRule(ct, mech, r, d)
    res = full_prices(z, rule, ct, shape, mech, r)
    np.testing.assert_array_equal([p.quota for p in res], rule(z[None, :])[0])


def test_r2_price_engine_quota_is_the_deployed_quota():
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(3, 7))
        c = random_case(rng, n, zero_prob=0.2)
        ct, mech, r = c["ct"], c["mech"], c["r"]
        shape = random_shape(rng, n)
        rule = AL.QuotaRule(ct, mech, r, ProtectedMap(rng.uniform(-3, 3, n), rng.uniform(0, 2, n), c["mu"]))
        Q = rule(c["z"][None, :])[0]
        for res in (full_prices(c["z"], rule, ct, shape, mech, r), fixed_priority_prices(c["z"], rule, ct, shape,
                                                                                         mech, r)):
            np.testing.assert_array_equal([p.quota for p in res], Q)
            assert all(p.p == 0.0 and p.V == 0.0 for p, zi in zip(res, c["z"]) if zi == 0)


# ====================================================================================== R3 noise ties (fixed price)
def _noise119():
    """The review's instance (scratchpad r2/repro_noise.py: seed 31, iteration 119)."""
    rng = np.random.default_rng(31)
    for it in range(120):
        n = int(rng.integers(3, 6))
        c = random_case(rng, n, zero_prob=0.0, load=float(rng.uniform(1.0, 3.0)))
        shape = random_shape(rng, n, uniform=False, max_atoms=16)
        i = int(rng.integers(n))
        theta = float(rng.uniform(0.05, 1) * c["ct"].C[i])
        a = rng.uniform(-4, 4, n)
        if it == 119:
            return c, shape, i, theta, a
    raise AssertionError("unreachable")


def test_r3_fixed_price_best_response_keeps_truth_on_a_dsic_map():
    """kappa = 0: weights are constant, T = 0 and p_fix = p, so the mechanism is DSIC. Before the fix the
    fixed-price best response reported z* = 6.776 against theta = 9.840 for a 'gain' of 1.3e-11 (below its
    certificate 2.2e-10) and moved the quotas from [0, 44, 23] to [6, 45, 16]."""
    c, shape, i, theta, a = _noise119()
    n = c["ct"].n
    rule = AL.QuotaRule(c["ct"], c["mech"], c["r"], ProtectedMap(a, np.zeros(n), c["mu"]))
    opp = np.delete(c["z"], i)
    for price in ("fixed", "full"):
        br = best_response(i, theta, opp, rule, price, shape=shape)
        assert br.z_star == theta and br.gain == 0.0, (price, br)


def test_r3_no_noise_misreports_on_dsic_maps():
    rng = np.random.default_rng(31)
    for _ in range(60):
        n = int(rng.integers(3, 6))
        c = random_case(rng, n, zero_prob=0.0, load=float(rng.uniform(1.0, 3.0)))
        shape = random_shape(rng, n, uniform=False, max_atoms=16)
        i = int(rng.integers(n))
        theta = float(rng.uniform(0.05, 1) * c["ct"].C[i])
        for pm in (FixedMap(c["w"]), ProtectedMap(rng.uniform(-4, 4, n), np.zeros(n), c["mu"])):
            br = best_response(i, theta, np.delete(c["z"], i), AL.QuotaRule(c["ct"], c["mech"], c["r"], pm), "fixed",
                               shape=shape)
            assert br.z_star == theta and br.gain == 0.0


# ====================================================================================== R4 weight-ratio boundary
def _r4_case():
    ct = Contract(cls=("video",) * 3, g=np.zeros(3), M=np.array([5.0, 60.0, 60.0]), C=np.full(3, 60.0),
                  pi=np.ones(3), delay_ms=np.full(3, 100.0), mu=np.ones(3))
    mech = Mechanism(B=100, gamma=0.2, eta=2.0, Delta=1.0, integer=False)
    return ct, mech, np.full(3, 30.0), np.full(3, 0.5)


def test_r4_fill_is_exact_up_to_the_documented_weight_ratio():
    """Before: weight ratios above ~1e306 (score spread ~709) overflowed the breakpoints and raised 'water-fill
    total mismatch up to 1.5e2 PRB'. Expected quotas: tenant 0 capped at 10 PRB, the rest split 1 : e."""
    ct, mech, z, r = _r4_case()
    expected = np.array([10.0, 90.0 / (1.0 + np.e), 90.0 * np.e / (1.0 + np.e)])
    for tiny in (1e-200, 1e-299, 1.0 / AL.MAX_WEIGHT_RATIO):
        x = AL.fractional(z, np.array([1.0, tiny, np.e * tiny]), r, ct, mech)
        np.testing.assert_allclose(x, expected, rtol=0, atol=1e-9)
    for s in (600.0, MAX_SCORE_SPREAD - 1.0):                                    # through softmax
        x = AL.fractional(z, softmax(np.array([0.0, -s, -s + 1.0])), r, ct, mech)
        np.testing.assert_allclose(x, expected, rtol=0, atol=1e-9)
    with pytest.raises(ArithmeticError, match="MAX_SCORE_SPREAD"):
        softmax(np.array([0.0, -710.0, -709.0]))
    with pytest.raises(ValueError, match="weight ratio"):
        AL.fractional(z, np.array([1.0, 1e-305, 1e-305]), r, ct, mech)


def test_r4_scaling_keeps_allocations_bit_identical():
    """The power-of-two row scale changes no bit of ordinary allocations (compared with a scale-free solve)."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(3, 9))
        c = random_case(rng, n)
        st = AL.stage(c["z"], c["r"], c["ct"], c["mech"])
        w = AL._weights(c["w"], c["z"].shape)
        x = AL._fill(st["L"], st["U"], w, st["total"])
        k = 2.0 ** int(rng.integers(-30, 30))                                     # exact rescaling of the bounds
        np.testing.assert_array_equal(AL._fill(st["L"] * k, st["U"] * k, w, st["total"] * k), x * k)


# ====================================================================================== R5 fractional best response
def _r5_game():
    ct = Contract(cls=("video",) * 2, g=np.zeros(2), M=np.full(2, 20.0), C=np.full(2, 20.0), pi=np.ones(2),
                  delay_ms=np.full(2, 100.0), mu=np.full(2, 5.0))
    mech = Mechanism(B=10, gamma=0.0, eta=1.5, Delta=1.0, integer=False)
    shape = Shape(beta=(np.array([0.25, 0.75, 1.25, 1.75]),) * 2, rho=(np.full(4, 0.25),) * 2)
    rule = AL.QuotaRule(ct, mech, np.ones(2), ProtectedMap([0.0, 0.0], [1.0, 1.0], ct.mu), integer=False)
    return rule, shape


def test_r5_fractional_usage_best_response_is_a_search_and_finds_the_kink():
    """Before: exact=True, z* = 4.902766, U* = 0.927343, cert = 0, while u = 4.895 gives 0.927454."""
    rule, shape = _r5_game()
    nu = np.full(2, 0.6)
    br = best_response(0, 5.3, [10.0], rule, "usage", shape=shape, nu=nu)
    assert not br.exact and np.isnan(br.cert)
    grid = np.linspace(0, 20, 4001)
    dense = utilities(0, 5.3, [10.0], rule, grid, "usage", shape=shape, nu=nu)
    assert br.U_star >= dense.max() - 1e-9
    # the optimum sits where X_0(u) reaches the value kink beta theta / r = 0.75 * 5.3 = 3.975 PRB: beyond it the
    # marginal value P(beta > 0.75) = 0.5 credits per PRB is below the tariff nu = 0.6
    assert abs(rule(np.array([br.z_star, 10.0]))[0] - 3.975) < 1e-9
    fixed = best_response(0, 5.3, [10.0], rule, "fixed", shape=shape)
    assert not fixed.exact and np.isnan(fixed.cert)
    for price in ("full", "none"):                                                 # provably exact
        b = best_response(0, 5.3, [10.0], rule, price, shape=shape)
        assert b.exact and np.isfinite(b.cert)
        d = utilities(0, 5.3, [10.0], rule, grid[::10], price, shape=shape)       # fractional full prices are slow
        assert b.U_star >= d.max() - b.cert - 1e-9


# ====================================================================================== R6 released d >= 1
def test_r6_released_requires_a_positive_delay():
    cal = _load_cal()
    w = _real_window(cal, "train", "2024-01-01", 60, K=12, warmup=2)
    assert len(w.released(5, 1)) == 5
    for bad in (0, -1, True, 1.0):
        with pytest.raises(ValueError, match="d must be an integer >= 1"):
            w.released(5, bad)


# ====================================================================================== R7 forecaster split guards
def test_r7_forecaster_fits_refuse_non_train_series():
    from slicemech.data.calibrate import fit_operator_forecasters
    from slicemech.forecast import fit_ar
    rng = np.random.default_rng(0)
    series = [rng.gamma(2.0, 1.0, (40, 3)) for _ in range(3)]
    f = fit_ar(series, history=2, horizon=1, splits=["train"] * 3)
    assert f.coef.shape == (3, 3)
    with pytest.raises(TypeError):
        fit_ar(series, history=2, horizon=1)                                      # split labels are required
    with pytest.raises(ValueError, match="TRAIN windows only"):
        fit_ar(series, history=2, horizon=1, splits=["train", "dev", "train"])
    with pytest.raises(ValueError, match="one split label per window"):
        fit_ar(series, history=2, horizon=1, splits=["train"])
    from slicemech.config import contract_from_params
    ct = contract_from_params(["video", "urllc", "mmtc"], np.array([12.0, 6.0, 2.0]))
    train_keys = [("2024-01-01", 60), ("2024-01-01", 7260), ("2024-01-08", 60)]
    fits = fit_operator_forecasters(series, ct, ds=(1,), keys=train_keys)
    assert fits[1].horizon == 1
    with pytest.raises(TypeError):
        fit_operator_forecasters(series, ct, ds=(1,))                             # keys are required
    with pytest.raises(ValueError, match="training windows only"):
        fit_operator_forecasters(series, ct, ds=(1,), keys=train_keys[:2] + [("2024-03-02", 60)])   # a DEV date
    with pytest.raises(ValueError, match="one \\(date, offset\\) key"):
        fit_operator_forecasters(series, ct, ds=(1,), keys=train_keys[:2])


# ====================================================================================== R8 compose_parts split check
def test_r8_compose_parts_checks_the_date_split_before_reading_pools(monkeypatch):
    from slicemech.data import compose, raca, tractor

    def boom(*a, **k):
        raise AssertionError("a pool was read before the split check")

    monkeypatch.setattr(raca, "pool", boom)
    monkeypatch.setattr(tractor, "pool", boom)
    for split, date in (("test", "2024-01-08"), ("train", "2024-03-02"), ("dev", "2024-01-01")):
        with pytest.raises(ValueError, match="is not in split"):
            compose.compose_parts(split, date, 60, ("video", "urllc", "mmtc"), (3, 3, 3), K=5)


# ====================================================================================== R9 freeze gate
@pytest.fixture
def freeze_env(monkeypatch):
    import shutil

    from slicemech import io as sio
    from slicemech.data import puffer as PU
    from slicemech.data import puffer_fetch as PF
    d = sio.DATA_DIR / "_pytest_r2_freeze"
    shutil.rmtree(d, ignore_errors=True)
    (d / "prereg").mkdir(parents=True)
    monkeypatch.setattr(PF, "PREREG_DIR", d / "prereg")
    monkeypatch.setattr(PU, "TEST_RAW_DIR", d / "puffer_raw")
    yield d, PF
    shutil.rmtree(d, ignore_errors=True)


def test_r9_freeze_gate_refuses_drafts_and_checks_hashes(freeze_env):
    from slicemech import io as sio
    d, PF = freeze_env
    real_draft = sio.ROOT / "prereg" / "PREREG_DRAFT.md"
    if not real_draft.is_file():
        pytest.skip("draft pre-registration missing")
    draft_bytes = real_draft.read_bytes()
    (d / "prereg" / "PREREG_DRAFT.md").write_bytes(draft_bytes)
    renamed = d / "prereg" / "PREREG.md"
    renamed.write_bytes(draft_bytes)                                              # the review's bypass
    with pytest.raises(PermissionError):
        PF._require_freeze(renamed)
    renamed.write_bytes(draft_bytes + b"\n")                                      # not byte-equal, still a draft
    with pytest.raises(PermissionError, match="draft"):
        PF._require_freeze(renamed)
    ckpt = d / "ckpt.pt"
    ckpt.write_bytes(b"parameters")
    block = PF.freeze_block([ckpt], "Sundays and Mondays 2024-09-01..10-21")
    frozen = d / "prereg" / "PREREG.md"
    frozen.write_text("# Pre-registration (frozen)\n\nBody.\n\n" + block, encoding="utf-8")
    rec = PF._require_freeze(frozen)
    assert rec["analysis_code_sha256"] == PF.analysis_code_sha256() and rec["config_sha256"] == PF.config_sha256()
    # a missing block, a stale code hash or a changed checkpoint is refused
    frozen.write_text("# Pre-registration (frozen)\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="freeze block"):
        PF._require_freeze(frozen)
    stale = block.replace(PF.analysis_code_sha256(), "f" * 64)
    frozen.write_text("# Pre-registration (frozen)\n\n" + stale, encoding="utf-8")
    with pytest.raises(PermissionError, match="analysis code"):
        PF._require_freeze(frozen)
    assert PF._require_freeze(frozen, check_code=False)["sha256"]
    frozen.write_text("# Pre-registration (frozen)\n\n" + block, encoding="utf-8")
    ckpt.write_bytes(b"retrained")
    with pytest.raises(PermissionError, match="checkpoint"):
        PF._require_freeze(frozen)
    ckpt.write_bytes(b"parameters")
    # once a download recorded the freeze, a different freeze file is refused (even with valid hashes)
    rec = PF._require_freeze(frozen)
    sio.write_json(PF.manifest_path(), dict(objects={}, replacements={}, failures={}, freeze=rec))
    assert PF.require_freeze_for_test_reads(frozen)["sha256"] == rec["sha256"]
    frozen.write_text("# Pre-registration (frozen, edited)\n\n" + block, encoding="utf-8")
    with pytest.raises(PermissionError, match="recorded"):
        PF.require_freeze_for_test_reads(frozen)


def test_r9_data_report_refuses_a_renamed_draft(freeze_env, monkeypatch):
    calib = pytest.importorskip("slicemech.data.calibrate")
    from slicemech import io as sio
    d, PF = freeze_env
    real_draft = sio.ROOT / "prereg" / "PREREG_DRAFT.md"
    if not real_draft.is_file():
        pytest.skip("draft pre-registration missing")
    renamed = d / "prereg" / "PREREG.md"
    renamed.write_bytes(real_draft.read_bytes())
    cal = _load_cal()
    with pytest.raises(PermissionError):
        calib.data_report(cal, path=d / "report.md", test_freeze=renamed)
    assert not (d / "report.md").exists()


# ====================================================================================== R10 E5 protocol on DEV
def test_r10_e5_protocol_runs_on_the_dev_pilot_and_test():
    from slicemech.data import compose, puffer
    from slicemech.tenants import poison
    assert poison.E5_OFFSETS == compose.window_offsets("test")
    assert poison.e5_alphas()[0] == poison.REFERENCE_ALPHA == 1.0 and set(poison.ALPHAS) < set(poison.e5_alphas())
    with pytest.raises(ValueError):
        poison.e5_layout("train")
    for split, day in (("dev", puffer.DEV_DATES[0]), ("test", puffer.test_dates()[0])):
        lay = poison.e5_layout(split)
        assert [p for _, p in lay] == ["A"] * 4 + ["B"] * 4
        ids = [compose.window_id(split, day, o, ("video", "urllc", "mmtc"), (18, 14, 236), 4.004, 1, p)
               for o, p in lay]
        plan = poison.plan_date(ids, target=1, alpha=2.0)                          # pre-registered 4 + 4
        assert plan.train_ids == tuple(ids[:4]) and plan.eval_ids == tuple(ids[4:])
        assert all("-pA-" in w for w in plan.train_ids) and all("-pB-" in w for w in plan.eval_ids)
    # the 6 standard DEV windows are too few: that is why the pilot uses the E5 offsets
    std = [compose.window_id("dev", puffer.DEV_DATES[0], o, ("video",), (1,), 4.004, 1)
           for o in compose.window_offsets("dev")]
    with pytest.raises(ValueError, match="need 8"):
        poison.plan_date(std, target=1, alpha=2.0)


# ====================================================================================== R11 victim harm metric
def _episode(on_time, dropped, pi, k0=0):
    K, n = on_time.shape
    return pd.DataFrame(dict(k=np.repeat(np.arange(k0, k0 + K), n), tenant=np.tile(np.arange(n), K),
                             on_time_Mbit=on_time.ravel(), dropped_Mbit=dropped.ravel(),
                             realized_value=(on_time * pi).ravel()))


def test_r11_victim_effects_see_harm_that_total_welfare_hides():
    """A high-tariff deviator (pi = 3) takes service from two pi = 1 / 0.5 tenants: they lose on-time Mbit and
    value while total realized welfare rises - the smoke-run pattern of the review (2024-01-01 +60)."""
    from slicemech.eval.metrics import victim_effects
    pi = np.array([1.0, 3.0, 0.5])
    on = np.tile([10.0, 4.0, 2.0], (6, 1))
    dr = np.tile([0.5, 0.2, 0.1], (6, 1))
    on_dev = on + np.array([-0.04, 0.04, -0.02])
    dr_dev = dr + np.array([0.04, 0.0, 0.02])
    truth, dev = _episode(on, dr, pi), _episode(on_dev, dr_dev, pi)
    ve = victim_effects(truth, dev, deviator=1, warmup=2)
    assert ve["n_intervals"] == 4
    assert ve["d_on_time_victims"] == pytest.approx(-0.06)
    assert ve["d_value_victims"] == pytest.approx(-0.04 * 1.0 - 0.02 * 0.5)
    assert ve["d_value_deviator"] == pytest.approx(0.12)
    assert ve["dW_realized"] > 0 > ve["d_value_victims"]                          # welfare hides the harm
    assert ve["d_miss_pct_victims"] > 0
    with pytest.raises(ValueError, match="same"):
        victim_effects(truth, dev.iloc[:-1], deviator=1, warmup=0)
    with pytest.raises(ValueError, match="deviator"):
        victim_effects(truth, dev, deviator=5, warmup=0)
