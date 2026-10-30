"""Tests for the data pipeline: slicemech.phy, io, forecast and slicemech.data.* (builder "data").

Real-data tests read only local files (Puffer TRAIN/DEV objects, TRACTOR, Raca); nothing is downloaded.
Temporary files go to ``data/_pytest`` inside the rewrite folder and are removed afterwards.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import shutil

import numpy as np
import pytest

from slicemech import io as sio
from slicemech import phy
from slicemech.config import Mechanism, contract_from_params
from slicemech.forecast import ARForecast, fit_ar, lagged_pairs
from slicemech.data import calibrate as CA
from slicemech.data import compose as CO
from slicemech.data import puffer as PU
from slicemech.data import puffer_fetch as PF
from slicemech.data import raca as RA
from slicemech.data import tractor as TR

MBIT_PER_BYTE = 8e-6
DATE = "2024-01-08"            # a TRAIN date with local data


@pytest.fixture
def local_tmp():
    d = sio.DATA_DIR / "_pytest"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def small_cal():
    """A cheap TRAIN calibration on 6 windows (default layout and load)."""
    keys = CA.train_window_keys()[::30]
    return CA.calibrate(train_windows=keys)


# ------------------------------------------------------------------------------------------------ phy
def test_cqi_table_matches_ts38214():
    # TS 38.214 Table 5.2.2.1-2 (CQI, Qm, code rate x 1024, efficiency), typed independently of phy.py
    spec = [(1, 2, 78, 0.1523), (2, 2, 120, 0.2344), (3, 2, 193, 0.3770), (4, 2, 308, 0.6016), (5, 2, 449, 0.8770),
            (6, 2, 602, 1.1758), (7, 4, 378, 1.4766), (8, 4, 490, 1.9141), (9, 4, 616, 2.4063),
            (10, 6, 466, 2.7305), (11, 6, 567, 3.3223), (12, 6, 666, 3.9023), (13, 6, 772, 4.5234),
            (14, 6, 873, 5.1152), (15, 6, 948, 5.5547)]
    for cqi, qm, rate, eff in spec:
        assert phy.CQI_QM[cqi] == qm and phy.CQI_CODE_RATE_X1024[cqi] == rate
        assert phy.CQI_EFFICIENCY[cqi] == eff
        assert abs(qm * rate / 1024 - eff) < 5e-5          # printed efficiency is Qm*R/1024 to 4 decimals
        assert phy.spectral_efficiency(cqi) == eff
        assert phy.cqi_to_rate(cqi) == pytest.approx(eff * 132 * 1000 * 0.9 / 1e6, rel=1e-15)
    assert phy.RE_PER_PRB == 132 and phy.BLER == 0.1 and phy.SLOTS_PER_S == 1000
    assert phy.cqi_to_rate(1) == pytest.approx(0.0180932, abs=1e-6)
    assert phy.cqi_to_rate(15) == pytest.approx(0.6598984, abs=1e-6)
    assert np.all(np.diff(phy.RATE_TABLE[1:]) > 0)
    for bad in (0, 16, 3.5, np.nan, -1):
        with pytest.raises(ValueError):
            phy.cqi_to_rate(bad)


# ------------------------------------------------------------------------------------------------ io
def test_io_confine_and_disk_guard(local_tmp):
    assert sio.confine(local_tmp / "x.json").is_relative_to(sio.ROOT)
    with pytest.raises(ValueError):
        sio.confine("/tmp/outside.json")
    assert sio.require_free_gb(0.001) > 0
    with pytest.raises(RuntimeError):
        sio.require_free_gb(1e9)
    p = sio.write_json(local_tmp / "a.json", dict(x=np.arange(3), y=np.float64(1.5)))
    assert sio.read_json(p) == dict(x=[0, 1, 2], y=1.5)
    raw = b"hello"
    (local_tmp / "b.bin").write_bytes(raw)
    assert sio.file_sha256(local_tmp / "b.bin") == hashlib.sha256(raw).hexdigest()
    assert sio.file_md5_b64(local_tmp / "b.bin") == base64.b64encode(hashlib.md5(raw).digest()).decode()


# ------------------------------------------------------------------------------------------------ pacing
def test_pacing_conserves_bytes():
    rng = np.random.default_rng(0)
    t0 = 10 * PU.NS
    T_ms = 30_000
    rel = t0 + rng.integers(0, (T_ms - 2100) * PU.MS, 400)          # fully inside the horizon
    size = rng.integers(0, 3_000_000, 400)
    per_ms = PU.pace_ms(rel, size, t0, T_ms)
    assert per_ms.dtype == np.int64 and np.all(per_ms >= 0)
    assert per_ms.sum() == size.sum()                                # every byte, exactly
    # chunks crossing the horizon edges: the window holds exactly cum(end) - cum(start)
    rel2 = t0 + rng.integers(-3 * PU.NS, (T_ms + 1000) * PU.MS, 400)
    per_ms2 = PU.pace_ms(rel2, size, t0, T_ms)
    assert per_ms2.sum() == PU.bytes_released(rel2, size, t0, t0 + T_ms * PU.MS).sum()
    # 1-s block sums equal the closed-form release over each interval
    blocks = per_ms2.reshape(-1, 1000).sum(1)
    a = t0 + np.arange(T_ms // 1000) * PU.NS
    closed = np.array([PU.bytes_released(rel2, size, x, x + PU.NS).sum() for x in a])
    assert np.array_equal(blocks, closed)
    # a single chunk is spread uniformly over 2.002 s (each full ms gets size/2002 bytes up to rounding)
    one = PU.pace_ms(np.array([t0]), np.array([2_002_000]), t0, 3000)
    assert one.sum() == 2_002_000 and one[:2002].min() == 1000 and one[2002:].sum() == 0
    with pytest.raises(ValueError):
        PU.pace_ms(rel, -size - 1, t0, T_ms)


def test_video_tenant_conserves_bytes():
    viewers = PU.viewer_draws(6, PU.CHANNELS, [np.random.default_rng(v) for v in range(6)])
    d0, _ = PU.day_bounds_ns(DATE)
    vt = PU.video_tenant(DATE, d0 + 3600 * PU.NS, 60, viewers, lead_ns=4_004_000_000)
    assert vt["arrivals_bytes"].sum() == vt["interval_bytes"].sum()
    assert np.array_equal(vt["arrivals_bytes"].reshape(60, 1000).sum(1), vt["interval_bytes"])


# ------------------------------------------------------------------------------------------------ video reports
@pytest.mark.parametrize("lead_s", [4.004, 1.05])
def test_video_report_equals_next_interval_ingress(lead_s, small_cal):
    w = small_cal.window("train", DATE, 7260, lead_s=lead_s)
    i = w.cls.index("video")
    # to the byte: the report and the measured ingress come from the same integer bytes
    assert np.max(np.abs(w.theta[:, i] - w.measured[:, i])) < MBIT_PER_BYTE
    assert np.array_equal(w.theta[:, i], w.measured[:, i])
    # and the per-ms arrivals (float32 Mbit) sum to the same interval volume within one byte
    vol = w.arrivals_ms[:, i].astype(np.float64).reshape(w.K, 1000).sum(1)
    assert np.max(np.abs(vol - w.measured[:, i])) < MBIT_PER_BYTE
    assert w.meta["tenants"][i]["exact_reports"]


def test_video_report_not_exact_below_lead_threshold(small_cal):
    w = small_cal.window("train", DATE, 7260, lead_s=0.0)
    i = w.cls.index("video")
    assert not w.meta["tenants"][i]["exact_reports"]
    assert np.max(np.abs(w.theta[:, i] - w.measured[:, i])) > 0.1     # persistence forecast is not exact


@pytest.mark.parametrize("lead_ns", [0, 500_000_000, 4_004_000_000])
def test_tenant_never_uses_chunk_after_report_time(lead_ns, monkeypatch):
    """The report for interval k is unchanged when every chunk with t_j > k - 0.05 s is removed or resized."""
    viewers = PU.viewer_draws(5, PU.CHANNELS, [np.random.default_rng(100 + v) for v in range(5)])
    d0, _ = PU.day_bounds_ns(DATE)
    start, K, ell = d0 + 7200 * PU.NS, 30, 50 * PU.MS
    full = PU.load_chunks(DATE)
    base = PU.video_tenant(DATE, start, K, viewers, lead_ns, ell)
    for k in (0, 3, 11, 17, 29):
        cutoff = start + k * PU.NS - ell
        trunc = {key: (t[t <= cutoff], s[t <= cutoff]) for key, (t, s) in full.items()}
        resized = {key: (t, np.where(t > cutoff, 3 * s + 7, s)) for key, (t, s) in full.items()}
        for table in (trunc, resized):
            monkeypatch.setattr(PU, "load_chunks", lambda day, table=table: table)
            assert PU.video_tenant(DATE, start, K, viewers, lead_ns, ell)["report_bytes"][k] == base["report_bytes"][k]
        # sanity: resizing chunks reported before the cutoff does change the report (the test is not vacuous)
        early = {key: (t, np.where(t <= cutoff, 2 * s + 1, s)) for key, (t, s) in full.items()}
        monkeypatch.setattr(PU, "load_chunks", lambda day, table=early: table)
        assert PU.video_tenant(DATE, start, K, viewers, lead_ns, ell)["report_bytes"][k] != base["report_bytes"][k]
        monkeypatch.setattr(PU, "load_chunks", lambda day: full)


def test_tractor_report_uses_only_history_before_report_time(small_cal):
    """theta_k of a TRACTOR tenant depends on own arrivals before k - 0.05 s only."""
    p = CO.compose_parts("train", DATE, 7260, small_cal.classes, small_cal.counts, K=60)
    i = small_cal.classes.index("urllc")
    b = p["bytes_ms"][:, i].copy()
    f = small_cal.forecaster

    def theta(bytes_i):
        own = p["own_history"].copy()
        own[:, i] = CO._shifted_history(bytes_i, 60, 50)
        return f.forecast_series(own)[:, i]

    th = theta(b)
    for k in (8, 20, 45):
        later = b.copy(); later[1000 * k - 50:] *= 3
        assert theta(later)[k] == th[k]
        earlier = b.copy(); earlier[1000 * k - 1000: 1000 * k - 50] += 10_000
        assert theta(earlier)[k] != th[k]


# ------------------------------------------------------------------------------------------------ operator side
def test_released_measurements_respect_delay(small_cal):
    w = small_cal.window("train", DATE, 60)
    for d in (1, 2, 5, 15):
        for k in range(w.K):
            rel = w.released(k, d)
            assert len(rel) == max(0, k - d + 1)             # never interval j > k - d
            assert np.array_equal(rel, w.measured[: max(0, k - d + 1)])
    with pytest.raises(ValueError):
        w.released(w.K, 2)
    with pytest.raises(ValueError):
        w.released(0, -1)
    with pytest.raises(ValueError):
        w.released(5, 0)                                       # d = 0 would release interval k itself (round 2)


def test_window_fields(small_cal):
    w = small_cal.window("train", DATE, 60)
    K, n = 200, 3
    assert (w.K, w.n, w.T_ms, w.warmup) == (K, n, K * 1000, 40)
    assert w.arrivals_ms.dtype == np.float32 and w.arrivals_ms.shape == (K * 1000, n)
    for a in (w.theta, w.measured, w.rate_true, w.rate_hat):
        assert a.shape == (K, n) and not a.flags.writeable
    assert np.array_equal(w.rate_hat[1:], w.rate_true[:-1])               # one-interval lag
    rates = phy.RATE_TABLE[1:]
    assert np.all(np.isin(w.rate_true, rates)) and np.all(np.isin(w.rate_hat, rates))
    assert w.meta["rate_age_s"].shape == (K, n) and np.all(w.meta["rate_age_s"] >= 1)
    assert 0 <= w.meta["tod_start_s"] < 86400
    assert w.contract.cls == w.cls
    assert np.allclose(w.measured, w.arrivals_ms.astype(float).reshape(K, 1000, n).sum(1), atol=1e-5)


def test_nested_composition_draws():
    classes = ("video", "urllc", "mmtc")
    small = CO.compose_parts("train", DATE, 60, classes, (6, 4, 30), K=20)
    big = CO.compose_parts("train", DATE, 60, classes, (9, 7, 50), K=20)
    ts, tb = small["meta"]["tenants"], big["meta"]["tenants"]
    assert tb[0]["viewers"][:6] == ts[0]["viewers"]
    assert tb[1]["users"][:4] == ts[1]["users"] and tb[2]["users"][:30] == ts[2]["users"]
    assert np.all(big["bytes_ms"] >= small["bytes_ms"])                   # extra units only add traffic
    assert np.array_equal(big["rate_true"], small["rate_true"])           # CQI draws independent of counts


# ------------------------------------------------------------------------------------------------ splits
def test_split_disjointness():
    tr, dv, te, te24 = set(PU.TRAIN_DATES), set(PU.DEV_DATES), set(PU.test_dates()), set(PU.test_dates(True))
    assert (len(tr), len(dv), len(te), len(te24)) == (15, 21, 16, 24)
    assert not (tr & dv) and not (tr & te24) and not (dv & te24) and te <= te24
    assert max(tr) < min(dv) and max(dv) < min(te24)
    assert min(tr) == "2024-01-01" and max(tr) == "2024-02-25" and min(dv) == "2024-02-28" and max(dv) == "2024-05-06"
    import datetime as dt
    assert all(dt.date.fromisoformat(d).weekday() in (6, 0) for d in te24)
    assert min(te) == "2024-09-01" and max(te) == "2024-10-21"
    for d in tr | dv:
        assert PU.local_source(d).exists()
    # TRACTOR: campaign stems disjoint, every URLLC/mMTC capture assigned to exactly one split
    stems = [s for sp in TR.SPLITS.values() for v in sp.values() for s in v]
    assert len(stems) == len(set(stems))
    on_disk = {p.stem for p in TR.RAW_DIR.glob("*.csv") if p.stem.startswith(("urll", "mmtc"))}
    assert on_disk == set(stems)
    for s in stems:
        assert TR.class_of_stem(s) in [c for c, v in TR.SPLITS[TR.split_of_stem(s)].items() if s in v]
    # Raca: date ranges disjoint; pools disjoint; every run inside its split's range
    rngs = sorted(RA.SPLITS.values())
    assert all(a[1] < b[0] for a, b in zip(rngs, rngs[1:]))
    pools = {s: {r.run_id for r in RA.pool(s)} for s in RA.SPLITS}
    assert not (pools["train"] & pools["dev"]) and not (pools["train"] & pools["test"]) and not (pools["dev"] & pools["test"])
    for s in RA.SPLITS:
        a, b = RA.SPLITS[s]
        assert all(a <= r.date <= b for r in RA.pool(s))


def test_window_draws_stay_in_split(small_cal):
    w = small_cal.window("dev", PU.DEV_DATES[0], 60)
    dev_stems = {s for v in TR.SPLITS["dev"].values() for s in v}
    for t in w.meta["tenants"]:
        assert RA.split_of_date(t["cqi_run"]["date"]) == "dev"
        for stem, _ in t.get("users", []):
            assert stem in dev_stems
    with pytest.raises(ValueError):
        small_cal.window("train", PU.DEV_DATES[0], 60)            # date/split mismatch


# ------------------------------------------------------------------------------------------------ forecast
def test_forecast_pairs_do_not_cross_windows():
    a = np.full((10, 2), 1.0); b = np.full((12, 2), 5.0)
    X, Y = lagged_pairs([a, b], history=3, horizon=2)
    assert X.shape == (6 + 8, 3, 2) and Y.shape == (14, 2)
    for x, y in zip(X, Y):
        assert np.all(x == y[None, :])                       # lags and target from the same window


def test_ar_fit_recovers_coefficients_and_is_causal():
    rng = np.random.default_rng(1)
    wins = []
    for _ in range(20):
        x = np.zeros((400, 2)); e = rng.normal(size=(400, 2))
        for t in range(1, 400):
            x[t] = 0.7 * x[t - 1] + e[t]
        wins.append(x + 10.0)
    f = fit_ar(wins, history=2, horizon=1, ridge=0.0, splits=["train"] * len(wins))
    lag1 = f.coef[:, 1] * 1.0                                  # normalised units equal raw units for AR(1)
    assert np.allclose(lag1, 0.7, atol=0.03) and np.allclose(f.coef[:, 2], 0.0, atol=0.03)
    g = ARForecast.from_dict(f.to_dict())
    s = g.forecast_series(wins[0])
    assert np.allclose(s[:2], f.mean)                          # no history yet -> training mean
    k = 50
    pert = wins[0].copy(); pert[k:] += 100.0                   # data at index >= k - h + 1 = k must not matter
    assert np.array_equal(g.forecast_series(pert)[k], s[k])
    assert np.all(g.forecast_series(-wins[0] * 10) >= 0)       # truncation at 0 Mbit/s


# ------------------------------------------------------------------------------------------------ readers
def test_tractor_reader(local_tmp):
    p = local_tmp / "urllc_x.csv"
    p.write_text("App name,No.,Time,Source,Destination,Protocol,Length\n"
                 "A,1,0.0,172.30.1.1,172.30.1.250,UDP,999\n"
                 "A,2,0.0004,172.30.1.250,172.30.1.1,UDP,100\n"
                 "A,3,0.0019,172.30.1.250,172.30.1.1,UDP,50\n"
                 "A,4,2.5,172.30.1.250,172.30.1.1,UDP,7\n")
    b, info = TR.read_trace(p)
    assert len(b) == 2501 and b[0] == 100 and b[1] == 50 and b[2500] == 7 and b.sum() == 157
    assert info["dl_packets"] == 3 and info["cls"] == "urllc"
    p.write_text("App name,No.,Time,Source,Destination,Protocol,Length\nA,1,0.0,172.30.1.250,x,UDP,1.5\n")
    with pytest.raises(ValueError):
        TR.read_trace(p)
    z = np.zeros(20_000, np.int64); z[:1000] = 1; z[9000:10_000] = 1; z[19_000:] = 1
    t, removed = TR.trim_idle(z, gap_s=5)
    assert removed == 17 and t.sum() == 3000 and len(t) == 3000
    tr = TR.Trace("x", "urllc", np.arange(10, dtype=np.int64), {})
    s, wrapped = TR.user_slice(tr, 7, 6)
    assert wrapped and s.tolist() == [7, 8, 9, 0, 1, 2]


def test_raca_reader(local_tmp):
    p = local_tmp / "B_2019.11.21_10.00.00.csv"
    rows = ["Timestamp,NetworkMode,CQI",
            "2019.11.21_10.00.00,5G,10", "2019.11.21_10.00.01,5G,11",
            "2019.11.21_10.00.04,5G,12",              # 2 missing seconds -> held with ages 1, 2
            "2019.11.21_10.00.05,5G,9", "2019.11.21_10.00.05,5G,8",   # conflicting duplicate -> last kept
            "2019.11.21_10.00.06,LTE,9",              # non-5G breaks the run
            "2019.11.21_10.00.07,5G,7", "2019.11.21_10.00.11,5G,7"]   # gap of 3 missing seconds breaks
    p.write_text("\n".join(rows) + "\n")
    runs, info = RA.read_runs(p)
    assert [r["cqi"].tolist() for r in runs] == [[10, 11, 11, 11, 12, 8], [7], [7]]
    assert runs[0]["age"].tolist() == [0, 0, 1, 2, 0, 0]
    assert info["held_seconds"] == 2 and info["duplicate_conflicting_seconds"] == 1
    assert RA.split_of_date("2019-11-21") == "train" and RA.split_of_date("2019-12-10") is None


def _gz_csv(path, lines):
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def test_puffer_parser_validation(local_tmp):
    d0, d1 = PU.day_bounds_ns("2024-01-01")
    assert PU.bucket_object("2024-01-01") == "2024-01-01T11_2024-01-02T11/video_size_2024-01-01T11_2024-01-02T11.csv"
    head = "time (ns GMT),format,channel,video_ts,size"
    good = [head, f"{d0 + 1},640x360-24,abc,0,100", f"{d0 + 2_002_000_000},640x360-24,abc,180180,120",
            f"{d0 + 5},1280x720-24,abc,0,300"]
    p = local_tmp / "good.csv.gz"
    _gz_csv(p, good)
    d, info = PU.parse_video_size(p, "2024-01-01")
    assert len(d) == 3 and info["series"] == 2
    assert info["md5_b64"] == sio.file_md5_b64(p) and info["sha256"] == sio.file_sha256(p)
    bad_sets = {"header": ["time,format,channel,video_ts,size"] + good[1:],
                "reversed": good[:2] + [f"{d0},640x360-24,abc,180180,120"],
                "outside": good + [f"{d1},640x360-24,abc,360360,1"],
                "negative": good + [f"{d0 + 9_000_000_000},640x360-24,abc,360360,-4"],
                "format": good + [f"{d0 + 9},HD,abc,0,4"]}
    for name, lines in bad_sets.items():
        q = local_tmp / f"{name}.csv.gz"
        _gz_csv(q, lines)
        with pytest.raises(Exception):
            PU.parse_video_size(q, "2024-01-01")


def test_prepared_chunk_tables():
    info = PU.date_info(DATE)
    assert info["formats"] == list(PU.FORMATS) and set(info["channels"]) == set(PU.CHANNELS)
    assert info["verified_hashes"]                               # md5/sha256 checked against recorded metadata
    ch = PU.load_chunks(DATE)
    assert set(ch) == {(c, f) for c in PU.CHANNELS for f in PU.FORMATS}
    d0, d1 = PU.day_bounds_ns(DATE)
    for t, s in ch.values():
        assert np.all(np.diff(t) > 0) and t[0] >= d0 and t[-1] < d1 and np.all(s >= 0)


# ------------------------------------------------------------------------------------------------ fetch (no network)
def test_puffer_fetch_urls_and_guards(local_tmp):
    day = "2024-09-01"
    assert PF.metadata_url(day) == ("https://storage.googleapis.com/storage/v1/b/puffer-data-release/o/"
                                    "2024-09-01T11_2024-09-02T11%2Fvideo_size_2024-09-01T11_2024-09-02T11.csv"
                                    "?fields=bucket%2Cname%2Csize%2Cgeneration%2Cmetageneration%2Cmd5Hash%2Ccrc32c"
                                    "%2CcontentType%2CcontentEncoding")
    assert PF.media_url(day, "1725000000000001", "1") == (
        "https://storage.googleapis.com/download/storage/v1/b/puffer-data-release/o/"
        "2024-09-01T11_2024-09-02T11%2Fvideo_size_2024-09-01T11_2024-09-02T11.csv"
        "?alt=media&generation=1725000000000001&ifGenerationMatch=1725000000000001&ifMetagenerationMatch=1")
    with pytest.raises(ValueError):
        PF.media_url(day, "01", "1")
    meta = dict(bucket="puffer-data-release", name=PU.bucket_object(day), size="38000000",
                generation="1725000000000001", metageneration="1", contentType="text/csv",
                contentEncoding="gzip", md5Hash=base64.b64encode(b"0123456789abcdef").decode())
    assert PF.check_metadata(meta, day) is meta
    for key, val in (("name", "other.csv"), ("size", "0"), ("md5Hash", "xx"), ("contentEncoding", "br")):
        with pytest.raises(ValueError):
            PF.check_metadata(dict(meta, **{key: val}), day)
    assert PF.target_path(day, meta) == PU.TEST_RAW_DIR / day / "video_size.csv.gz"
    taken = set(PU.test_dates(True))
    assert PF.replacement_date("2024-09-01", taken) == "2024-09-03"   # Monday 09-02 is already a TEST date

    def no_network(*a, **k):
        raise AssertionError("network must not be touched")

    draft = sio.ROOT / "prereg" / "PREREG_DRAFT.md"
    with pytest.raises(PermissionError):
        PF.fetch_date(day, freeze_file=draft, opener=no_network)
    with pytest.raises(PermissionError):
        PF.fetch_date(day, freeze_file=local_tmp / "x.md", opener=no_network)
    with pytest.raises(PermissionError):
        PF.fetch_all(draft, opener=no_network)
    assert PU.split_dates("test") == PU.test_dates()                    # nothing fetched yet


class _FakeResponse:
    """In-memory HTTP response (status, headers, read) used instead of the network."""

    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        import io as _io
        self._b, self.status, self.headers = _io.BytesIO(body), status, dict(headers or {})

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_gcs(objects: dict, missing=(), tamper=None, header_override=None):
    """Fake opener serving ``objects`` {date: stored bytes} through the metadata and pinned-media URLs."""
    import json as _json
    from urllib.error import HTTPError

    metas = {d: dict(bucket=PF.BUCKET, name=PU.bucket_object(d), size=str(len(b)), generation=str(1000 + i),
                     metageneration="1", contentType="text/csv", contentEncoding="gzip",
                     md5Hash=base64.b64encode(hashlib.md5(b).digest()).decode())
             for i, (d, b) in enumerate(sorted(objects.items()))}
    calls = []

    def opener(req, timeout=None):
        url = req.full_url
        calls.append(url)
        for d, m in metas.items():
            if url == PF.metadata_url(d):
                if d in missing:
                    raise HTTPError(url, 404, "Not Found", {}, None)
                return _FakeResponse(_json.dumps(m).encode())
            if url == PF.media_url(d, m["generation"], m["metageneration"]):
                assert req.get_header("Accept-encoding") == "gzip"
                body = objects[d] if tamper is None else tamper(objects[d])
                h = {"Content-Encoding": "gzip", "x-goog-generation": m["generation"],
                     "x-goog-hash": f"crc32c=AAAAAA==,md5={m['md5Hash']}", "Content-Length": m["size"]}
                h.update(header_override or {})
                return _FakeResponse(body, headers=h)
        raise AssertionError(f"unexpected URL {url}")

    return opener, metas, calls


def _video_size_body(day: str) -> bytes:
    """A small parseable stored video_size object of ``day`` (all 7 channels x 4 formats, 2 chunks each): since the
    pre-freeze audit (F3) the download prepares every object, so a non-parseable body counts as corrupt."""
    d0, _ = PU.day_bounds_ns(day)
    lines = [",".join(PU.SCHEMA)] + [f"{d0 + 1000 + k * PU.PACING_NS},{f},{c},{k * PU.VIDEO_TS_STEP},{900 + k}"
                                     for c in PU.CHANNELS for f in PU.FORMATS for k in range(2)]
    return gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0)


@pytest.fixture
def fake_fetch_env(local_tmp, monkeypatch):
    """TEST download directory, chunk cache and frozen pre-registration file redirected into data/_pytest."""
    raw = local_tmp / "puffer_raw"
    monkeypatch.setattr(PU, "TEST_RAW_DIR", raw)
    monkeypatch.setattr(PU, "CACHE_DIR", local_tmp / "puffer_cache")
    monkeypatch.setattr(PF, "PREREG_DIR", local_tmp)
    ckpt = local_tmp / "ckpt_protected_seed0.pt"
    ckpt.write_bytes(b"trained parameters")
    freeze = local_tmp / "PREREG.md"            # a valid freeze (round-2 gate): hashes of the current tree
    block = PF.freeze_block([ckpt], "Sundays and Mondays 2024-09-01..10-21")
    freeze.write_text("# Pre-registration (frozen)\n\n" + block, encoding="utf-8")
    return raw, freeze


def test_puffer_fetch_pinned_download_offline(fake_fetch_env):
    """End-to-end pinned download through a fake transport (no network): md5/size/EOF/header checks."""
    raw, freeze = fake_fetch_env
    day = "2024-09-01"
    body = _video_size_body(day)
    opener, metas, _ = _fake_gcs({day: body})
    rec = PF.fetch_date(day, freeze_file=freeze, opener=opener)
    target = raw / day / "video_size.csv.gz"
    assert target.read_bytes() == body and rec["sha256"] == hashlib.sha256(body).hexdigest()
    assert rec["freeze"]["sha256"] == sio.file_sha256(freeze)
    assert PU._recorded_hashes(target) == {"md5_b64": metas[day]["md5Hash"]}     # prepare_date re-checks it
    assert PU.local_source(day) == target
    with pytest.raises(FileExistsError):                                       # never overwritten
        PF.fetch_date(day, freeze_file=freeze, opener=opener)
    # corrupt bytes (same size), truncated and over-long bodies are refused; the partial file is kept
    for i, tamper in enumerate((lambda b: bytes([b[0] ^ 1]) + b[1:], lambda b: b[:-1], lambda b: b + b"x")):
        d = PU.test_dates()[1 + i]
        op, _, _ = _fake_gcs({d: body}, tamper=tamper)
        with pytest.raises(PF.CorruptObject):
            PF.fetch_date(d, freeze_file=freeze, opener=op)
        assert not (raw / d / "video_size.csv.gz").exists() and (raw / d / "video_size.csv.gz.partial").exists()
    # headers contradicting the pinned metadata are refused before any byte is written
    for override in ({"x-goog-generation": "999"}, {"Content-Encoding": "identity"}, {"Content-Range": "bytes 0-1/9"},
                     {"x-goog-hash": "md5=" + base64.b64encode(b"x" * 16).decode()}):
        d = PU.test_dates()[5]
        op, _, _ = _fake_gcs({d: body}, header_override=override)
        with pytest.raises(ValueError):
            PF.fetch_date(d, freeze_file=freeze, opener=op)
        assert not (raw / d / "video_size.csv.gz.partial").exists()
    with pytest.raises(ValueError):
        PF.fetch_date("2024-09-04", freeze_file=freeze, opener=opener)        # not a pre-registered date


def test_puffer_fetch_all_replacement_rule(fake_fetch_env):
    """A missing (404) TEST date is replaced by the next free calendar day and recorded in the manifest."""
    raw, freeze = fake_fetch_env
    dates = PU.test_dates()
    objs = {d: _video_size_body(d) for d in dates}             # parseable: the download checks the content (F3)
    objs["2024-09-03"] = _video_size_body("2024-09-03")        # replacement for the missing 2024-09-02
    opener, _, _ = _fake_gcs(objs, missing={"2024-09-02"})
    m = PF.fetch_all(freeze, opener=opener)
    assert m["replacements"] == {"2024-09-02": "2024-09-03"} and "2024-09-02" in m["failures"]
    assert set(m["objects"]) == (set(dates) - {"2024-09-02"}) | {"2024-09-03"}
    eff = PU.effective_test_dates()
    assert len(eff) == 16 and "2024-09-03" in eff and "2024-09-02" not in eff
    assert PU.split_of_date("2024-09-03") == "test"
    assert PF.fetch_all(freeze, opener=opener) == m                            # resuming refetches nothing
    with pytest.raises(PermissionError):
        PF.fetch_all(raw / "x.md", opener=opener)


def test_redirects_are_refused():
    from urllib.request import Request
    h = PF._NoRedirect()
    with pytest.raises(ValueError):
        h.redirect_request(Request("https://storage.googleapis.com/x"), None, 302, "Found", {}, "https://evil/x")


def test_video_group_fallback(small_cal, monkeypatch):
    """A video group with no channel present on a date falls back to every present channel of the fixed set."""
    monkeypatch.setattr(PU, "channels_present", lambda day: ("cbs", "cw", "kron"))
    p = CO.compose_parts("train", DATE, 60, ("video", "video", "urllc"), (4, 4, 2),
                         (("abc",), ("cw", "abc"), None), K=10)
    t0, t1 = p["meta"]["tenants"][:2]
    assert t0["group_fallback"] and t0["group_present"] == ["cbs", "cw"]
    assert {v["channel"] for v in t0["viewers"]} <= {"cbs", "cw"}
    assert not t1["group_fallback"] and t1["group_present"] == ["cw"]
    assert {v["channel"] for v in t1["viewers"]} == {"cw"}


# ------------------------------------------------------------------------------------------------ calibration
def test_tenant_counts_and_contract(small_cal):
    c = small_cal
    assert c.classes == ("video", "urllc", "mmtc") and len(c.counts) == 3 and all(x >= 1 for x in c.counts)
    target = 0.9 * 106 * c.r_bar * np.array([0.6, 0.3, 0.1])
    assert np.all(np.abs(np.array(c.counts) * c.unit_mbps - target) <= 0.5 * c.unit_mbps + 1e-9)
    ref = contract_from_params(list(c.classes), c.mu)
    for f in ("g", "M", "C", "pi", "delay_ms", "mu"):
        assert np.array_equal(getattr(c.contract, f), getattr(ref, f))
    assert 0.40 < c.r_bar < 0.47
    # video shape is degenerate at 1 with exact advance information and w = Delta
    assert np.allclose(c.shape.beta[0], 1.0)
    for b, r in zip(c.shape.beta, c.shape.rho):
        assert abs(r.sum() - 1) < 1e-9 and abs((r * b).sum() - 1) < 1e-9
    assert set(c.diagnostics["targets"]) == {"frac_sumH_gt_B", "frac_all_caps_fit", "frac_weights_change_quota",
                                             "mean_weighted_share", "gbr_outage"}
    c2 = CA.Calibration.from_dict(c.to_dict())
    assert c2.counts == c.counts and np.array_equal(c2.forecaster.coef, c.forecaster.coef)
    assert c2.load_rule == c.load_rule == "per_date" and c2.date_counts == c.date_counts       # D2 default
    assert set(c.date_counts["train"]) == {d for d, _ in CA.train_window_keys()[::30]}
    with pytest.raises(ValueError):
        CA.calibrate(train_windows=[(PU.DEV_DATES[0], 60)])
    # fixed counts on every date
    c0 = CA.calibrate(train_windows=CA.train_window_keys()[::60], lead_s=0.0, counts=c.counts)
    assert c0.counts == c.counts and c0.meta["counts_rule"] == "fixed" and c.meta["counts_rule"] == "per_date"
    assert c0.counts_for("train", DATE) == c.counts and not c0.date_counts
    assert c0.meta["report_rel_mae"][0] > 0                       # lead 0: video report is a forecast
    with pytest.raises(ValueError):
        CA.calibrate(train_windows=CA.train_window_keys()[:1], counts=(1, 2))


def test_burst_samples():
    K = 4
    b = np.zeros((K * 1000, 2), np.int64)
    b[:, 0] = 125                        # 1 Mbit per s, flat
    b[0:150, 1] = 6000                   # 7.2 Mbit burst in the first 150 ms of interval 0 (urllc-like)
    theta = np.array([[1.0, 1.0]] * K)
    sub = CA.subwindow_volumes(b, np.array([2002.0, 150.0]))
    assert sub[0].shape == (K, 1) and sub[1].shape == (K, 6)
    u = CA.burst_samples(sub, theta, np.array([2002.0, 150.0]), warmup=0)
    assert np.allclose(u[0], 1.0)
    assert u[1][0] == pytest.approx(150 * 6000 * 8e-6 / (0.15 * 1.0)) and np.all(u[1][1:] == 0)


def test_regime_diagnostics_definitions():
    ct = contract_from_params(["video", "urllc", "mmtc"], np.array([10.0, 5.0, 2.0]))
    mech = Mechanism()
    r = np.full((3, 3), 0.4)
    z = np.array([[10.0, 5.0, 2.0],       # sum H = (20+10+4)/0.4 = 85 < 106; caps (30+15+8)/0.4 = 132.5 > 106
                  [30.0, 15.0, 6.0],      # need stage
                  [50.0, 25.0, 16.0]])    # sum b = (5+4+0.4)/0.4 = 23.5 < 106: no overload
    d = CA.regime_diagnostics(z, r, ct, mech)
    assert d["frac_sumH_gt_B"] == pytest.approx(2 / 3) and d["frac_all_caps_fit"] == 0.0
    assert d["gbr_outage"] == 0.0 and d["intervals"] == 3
    t = CA.check_targets(d)
    assert t["frac_sumH_gt_B"]["ok"] and t["gbr_outage"]["ok"]
