"""Tests for orchestrator decision D2: offered-load normalization per date (calibration owner, 2026-10-19).

Rule (``data.calibrate.counts_for_date``): for every date of every split, each tenant's integer count of viewers /
users / devices is set from the date's offered traffic only, so that the date's mean offered traffic of each class
is closest to ``share_c * rho * B * r_bar_TRAIN`` (class shares split equally among the tenants of a class). TEST
counts may be computed only after the pre-registration freeze.

Real-data tests read local TRAIN/DEV files only; nothing is downloaded and no TEST source is opened (the TEST tests
replace every TEST reader by a stub that fails, or by synthetic offered bytes).
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from slicemech import io as sio
from slicemech.data import calibrate as CA
from slicemech.data import compose as CO
from slicemech.data import puffer as PU
from slicemech.data import puffer_fetch as PF
from slicemech.data import raca as RA
from slicemech.data import tractor as TR

DATE = "2024-01-08"            # a TRAIN date
RHO = 0.9
TOL = 0.03                     # pre-registered check: date offered load within +-3 % of rho on TRAIN dates


def _cal():
    try:
        cal = CA.load_calibration("n3_default")
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"calibration missing: {exc}")
    assert cal.load_rule == "per_date", "n3_default must be calibrated with the D2 per-date rule"
    return cal


def _clear_caches():
    CA._date_load.cache_clear()
    CO._UNITS.clear()


# ------------------------------------------------------------------------------------------------ exactness
def test_d2_unit_bytes_equal_composed_bytes():
    """Per-unit offered bytes use the composition's own draws: the first U units sum to the composed tenant bytes,
    exactly (integer bytes), for video viewers, URLLC users and mMTC devices, with and without a pool half."""
    classes, groups = ("video", "urllc", "mmtc", "video"), (PU.CHANNELS, None, None, ("cw", "fox"))
    counts = (7, 5, 40, 3)
    for part in (None, "B"):
        p = CO.compose_parts("train", DATE, 7260, classes, counts, groups, K=30, pool_part=part)
        for i, c in enumerate(classes):
            u = CO.unit_offered_bytes("train", DATE, 7260, i, c, 0, counts[i], groups[i], K=30, pool_part=part)
            assert u.dtype == np.int64 and u.shape == (counts[i],)
            assert int(u.sum()) == int(p["bytes_ms"][:, i].sum()), (part, i, c)
    # incremental draws are nested and the cache returns prefixes of the same series
    a = CO.unit_offered_bytes("train", DATE, 60, 1, "urllc", 0, 9)
    b = np.concatenate([CO.unit_offered_bytes("train", DATE, 60, 1, "urllc", 0, 4),
                        CO.unit_offered_bytes("train", DATE, 60, 1, "urllc", 4, 9)])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(CO.offered_units("train", DATE, 60, 1, "urllc", 9), a)
    np.testing.assert_array_equal(CO.offered_units("train", DATE, 60, 1, "urllc", 4), a[:4])
    # wrapped users (a DEV URLLC capture is shorter than a window when halved) are summed circularly
    w = CO.unit_offered_bytes("dev", PU.DEV_DATES[0], 60, 1, "urllc", 0, 6, pool_part="A")
    pw = CO.compose_parts("dev", PU.DEV_DATES[0], 60, ("video", "urllc"), (1, 6), K=200, pool_part="A")
    assert pw["meta"]["tenants"][1]["wrapped_users"] > 0 and int(w.sum()) == int(pw["bytes_ms"][:, 1].sum())
    with pytest.raises(ValueError, match="is not in split"):
        CO.unit_offered_bytes("dev", DATE, 60, 0, "video", 0, 2)


# ------------------------------------------------------------------------------------------------ the rule
def test_d2_counts_are_deterministic_and_closest():
    dl = CA.counts_for_date("train", DATE, RHO, 3, info=True)
    _clear_caches()
    dl2 = CA.counts_for_date("train", DATE, RHO, 3, info=True)
    assert dl2.counts == dl.counts and all(isinstance(c, int) and c >= 1 for c in dl.counts)
    np.testing.assert_array_equal(dl2.window_offered_mbps, dl.window_offered_mbps)
    assert CA.counts_for_date("train", DATE, RHO, 3) == dl.counts            # cached and fresh calls agree
    assert dl.r_bar == RA.mean_rate("train", CO.K_DEFAULT + 1)               # the TRAIN per-PRB rate
    np.testing.assert_allclose(dl.target_mbps, RHO * 106 * dl.r_bar * np.array([0.6, 0.3, 0.1]), rtol=1e-12)
    # n = 3: every count is the integer closest to its target (no neighbour is closer)
    offs = CO.window_offsets("train")
    classes = ("video", "urllc", "mmtc")
    for i, c in enumerate(classes):
        U = dl.counts[i]
        tot = np.sum([CO.offered_units("train", DATE, o, i, c, U + 1) for o in offs], axis=0).cumsum()
        rate = tot * 8e-6 / (len(offs) * CO.K_DEFAULT)                       # date mean offered with 1..U+1 units
        err = np.abs(rate - dl.target_mbps[i])
        assert err[U - 1] <= err[U] and (U == 1 or err[U - 1] <= err[U - 2])
        assert dl.offered_mbps[i] == pytest.approx(rate[U - 1], rel=1e-12)
    # the rule's per-window offered rate is the composed window's mean measured demand (all K intervals)
    p = CO.compose_parts("train", DATE, offs[3], classes, dl.counts, K=CO.K_DEFAULT)
    np.testing.assert_allclose(dl.window_offered_mbps[3], p["measured"].mean(0), rtol=1e-12)
    # deterministic in the other split too, and independent of the order of calls
    d_dev = CA.counts_for_date("dev", PU.DEV_DATES[4], RHO, 3)
    _clear_caches()
    assert CA.counts_for_date("dev", PU.DEV_DATES[4], RHO, 3) == d_dev


@pytest.mark.parametrize("n", [3, 6, 9])
def test_d2_offered_load_within_3pct_on_train_dates(n):
    """Pre-registered check: on every TRAIN date the date's offered load is within +-3 % of rho; with one tenant
    per class each tenant is within half a unit of its target, otherwise each class total is."""
    loads = []
    for d in PU.TRAIN_DATES:
        dl = CA.counts_for_date("train", d, RHO, n, info=True)
        loads.append(dl.load)
        assert abs(dl.load / RHO - 1) <= TOL, (n, d, dl.load, dl.counts)
        classes, _ = CO.tenant_layout(n)
        for c in dict.fromkeys(classes):
            idx = [i for i, x in enumerate(classes) if x == c]
            cls_err = dl.offered_mbps[idx].sum() / dl.target_mbps[idx].sum() - 1
            assert abs(cls_err) <= 0.06, (n, d, c, cls_err)                  # half a unit of the class total
    assert np.ptp(loads) > 0                                                 # realized loads are reported, not forced


def test_d2_load_sweep_is_nested():
    cs = [CA.counts_for_date("train", DATE, r, 3, info=True) for r in (0.5, 0.7, 0.9, 1.1)]
    for lo, hi in zip(cs, cs[1:]):
        assert all(a <= b for a, b in zip(lo.counts, hi.counts)) and lo.load < hi.load
    classes = ("video", "urllc", "mmtc")
    small = CO.compose_parts("train", DATE, 60, classes, cs[0].counts, K=20)
    big = CO.compose_parts("train", DATE, 60, classes, cs[-1].counts, K=20)
    assert np.all(big["bytes_ms"] >= small["bytes_ms"])                      # the low load is a subset of the high
    np.testing.assert_array_equal(big["rate_true"], small["rate_true"])      # CQI draws unchanged


def test_d2_lead_sweep_keeps_the_date_counts():
    """A lead-time sweep calibration keeps the default per-date counts (count lead 4.004 s): only the information
    changes, not the traffic volume."""
    keys = [("2024-01-01", 60), ("2024-01-14", 21660)]
    a = CA.calibrate(train_windows=keys)
    b = CA.calibrate(train_windows=keys, lead_s=0.0)
    assert a.date_counts == b.date_counts and b.count_lead_s == CO.LEAD_DEFAULT and b.lead_s == 0.0
    assert b.meta["report_rel_mae"][0] > 0 == a.meta["report_rel_mae"][0]  # lead 0: the video report is a forecast


def test_d2_reads_only_offered_traffic_and_the_train_rate(monkeypatch):
    """No CQI run of the date's split, no composed window (reports, CQI, outcomes) enters the rule."""
    real_pool, real_tpool = RA.pool, TR.pool
    raca_splits, tractor_splits = [], []

    def rec_pool(split, *a, **k):
        raca_splits.append(split)
        return real_pool(split, *a, **k)

    def rec_tpool(split, *a, **k):
        tractor_splits.append(split)
        return real_tpool(split, *a, **k)

    def no_compose(*a, **k):
        raise AssertionError("the count rule composed a window")

    monkeypatch.setattr(RA, "pool", rec_pool)
    monkeypatch.setattr(TR, "pool", rec_tpool)
    monkeypatch.setattr(CO, "compose_parts", no_compose)
    monkeypatch.setattr(CO, "compose_window", no_compose)
    _clear_caches()
    try:
        dl = CA.counts_for_date("dev", PU.DEV_DATES[2], RHO, 3, info=True)
    finally:
        _clear_caches()
    assert set(raca_splits) <= {"train"} and set(tractor_splits) == {"dev"}
    assert dl.r_bar == RA.mean_rate("train", CO.K_DEFAULT + 1)


# ------------------------------------------------------------------------------------------------ calibration
def test_d2_calibration_composes_with_the_date_counts():
    cal = _cal()
    assert set(cal.date_counts) == {"train", "dev"}                          # TEST counts are never stored
    assert set(cal.date_counts["train"]) == set(PU.TRAIN_DATES) and set(cal.date_counts["dev"]) == set(PU.DEV_DATES)
    for split, d in (("train", DATE), ("dev", PU.DEV_DATES[0]), ("dev", PU.DEV_DATES[-1])):
        stored = cal.date_counts[split][d]
        assert stored == CA.counts_for_date(split, d, cal.rho, cal.classes, cal.video_groups, r_bar=cal.r_bar,
                                            shares=cal.shares, K=cal.K, seed=cal.seed, lead_s=cal.count_lead_s)
        w = cal.window(split, d, CO.window_offsets(split)[1], K=12, warmup=2)
        assert tuple(w.meta["counts"]) == tuple(stored) == cal.counts_for(split, d)
    w = cal.window("train", DATE, 60, K=12, warmup=2, counts=(2, 2, 2))      # explicit counts still win
    assert tuple(w.meta["counts"]) == (2, 2, 2)
    assert all(a <= b for a, b in zip(cal.counts_for("train", DATE, rho=0.5), cal.counts_for("train", DATE)))
    back = CA.Calibration.from_dict(cal.to_dict())
    assert back.load_rule == "per_date" and back.date_counts == cal.date_counts
    assert cal.meta["counts_rule"] == "per_date" and cal.meta["gbr_x_mu"] == {"mmtc": 0.16, "urllc": 0.64,
                                                                                "video": 0.4}
    np.testing.assert_allclose(cal.contract.g / cal.contract.mu, [0.40, 0.64, 0.16], rtol=1e-12)
    # every TRAIN date's stored offered load is within +-3 % of rho
    bb = 106 * cal.r_bar
    for d, off in cal.meta["date_offered_mbps"]["train"].items():
        assert abs(np.sum(off) / bb / cal.rho - 1) <= TOL, d
    with pytest.raises(ValueError):
        CA.attach_date_counts(cal, ("test",))


# ------------------------------------------------------------------------------------------------ TEST gate
@pytest.fixture
def no_freeze(monkeypatch):
    """No frozen pre-registration and no fetch manifest; every TEST reader fails loudly if touched."""
    d = sio.DATA_DIR / "_pytest_d2"
    shutil.rmtree(d, ignore_errors=True)
    (d / "prereg").mkdir(parents=True)
    monkeypatch.setattr(PF, "PREREG_DIR", d / "prereg")
    monkeypatch.setattr(PU, "TEST_RAW_DIR", d / "puffer_raw")
    _clear_caches()
    yield d, monkeypatch
    _clear_caches()
    shutil.rmtree(d, ignore_errors=True)


def _forbid_test_reads(monkeypatch):
    real_t, real_r = TR.pool, RA.pool

    def guard_t(split, *a, **k):
        assert split != "test", "a TEST TRACTOR capture was read"
        return real_t(split, *a, **k)

    def guard_r(split, *a, **k):
        assert split != "test", "a TEST Raca run was read"
        return real_r(split, *a, **k)

    def no_units(*a, **k):
        raise AssertionError("TEST offered traffic was read")

    monkeypatch.setattr(TR, "pool", guard_t)
    monkeypatch.setattr(RA, "pool", guard_r)
    monkeypatch.setattr(CO, "offered_units", no_units)
    monkeypatch.setattr(CO, "unit_offered_bytes", no_units)
    monkeypatch.setattr(PU, "load_chunks", no_units)


def test_d2_test_counts_raise_before_the_freeze(no_freeze):
    _, mp = no_freeze
    _forbid_test_reads(mp)
    day = PU.test_dates()[0]
    with pytest.raises(PermissionError, match="freeze"):
        CA.counts_for_date("test", day, RHO, 3)
    with pytest.raises(PermissionError):
        CA.counts_for_date("test", day, RHO, 3, test_freeze=sio.ROOT / "prereg" / "PREREG_DRAFT.md")
    with pytest.raises(PermissionError):
        PF.require_test_read_gate()
    cal = _cal()
    with pytest.raises(PermissionError):
        cal.counts_for("test", day)
    with pytest.raises(PermissionError):
        cal.window("test", day, 60)
    with pytest.raises(PermissionError):                                     # the gate comes before any split check
        CA.counts_for_date("test", DATE, RHO, 3)


def test_d2_test_counts_pass_the_gate_after_the_freeze(no_freeze):
    """After a valid freeze (explicit file, or the one recorded at the TEST download) the rule proceeds; synthetic
    offered bytes stand in for the TEST sources so no TEST statistic is computed here."""
    d, mp = no_freeze
    ckpt = d / "prereg" / "ckpt.pt"
    ckpt.write_bytes(b"checkpoint")
    freeze = d / "prereg" / "PREREG.md"
    freeze.write_text("# Pre-registration (frozen)\n\n" + PF.freeze_block([ckpt], "Sundays and Mondays"),
                      encoding="utf-8")
    calls = []

    def fake_units(split, date, offset_s, tenant, cls, n_units, video_group=None, **kw):
        calls.append(split)
        return np.full(int(n_units), 1_000_000, np.int64)                    # 0.04 Mbit/s per unit over 200 s

    mp.setattr(CO, "offered_units", fake_units)
    day = PU.test_dates()[0]
    dl = CA.counts_for_date("test", day, RHO, 3, test_freeze=freeze, info=True)
    assert calls and set(calls) == {"test"}
    assert dl.counts == tuple(int(round(t / 0.04)) for t in dl.target_mbps)
    with pytest.raises(PermissionError):                                     # nothing recorded yet: needs the file
        CA.counts_for_date("test", day, RHO, 3)
    (d / "puffer_raw").mkdir()
    sio.write_json(d / "puffer_raw" / "manifest.json", dict(freeze=PF._require_freeze(freeze), objects={},
                                                            replacements={}, failures={}, extended=False))
    assert CA.counts_for_date("test", day, RHO, 3) == dl.counts               # the recorded freeze opens the gate
    freeze.write_text(freeze.read_text() + "\nedited after the download\n", encoding="utf-8")
    with pytest.raises(PermissionError):                                     # a changed freeze file is refused
        CA.counts_for_date("test", day, RHO, 3)
