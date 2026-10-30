"""Regression tests for the pre-freeze audit fixes of the data group (review/prefreeze_audit.md, checklist §7).

* F2  (item 1)  concurrent TEST date preparation: process-unique temp names, a per-day flock with a re-check, an
                atomic TRACTOR cache;
* F1  (item 2)  the frozen code hash ignores dot-directories (lint caches);
* F3  (item 3)  content-invalid objects are corrupt at download time; a size/MD5 mismatch is retried once;
* F5  (item 15) the number of TEST dates is a freeze field, enforced by the download and the TEST read gate;
* F6  (item 16) E7 on TEST needs the frozen NR calibration, its radius and its ns-3 build; --dry-run refused;
* E5-7/F8 (item 17) model config.json, static candidates, sweep tariffs and the NR calibration are frozen assets;
* F12 (item 27) a stale .partial does not block a resume; the manifest must cover exactly the TEST dates;
* E5-3/F4 (items 38/40) the date rule strings and their replacement clause.

No TEST data is read and nothing is downloaded: every object is synthetic, served by a fake GCS opener, and every
directory the code writes is redirected below ``data/_pytest_prefreeze_<pid>``.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import sys
import threading
import time
from urllib.error import HTTPError

import numpy as np
import pyarrow.parquet as pq
import pytest

CODE = pathlib.Path(__file__).resolve().parents[1]
for _p in (CODE, CODE / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import registry as R  # noqa: E402
from slicemech import io as sio  # noqa: E402
from slicemech.data import puffer as PU  # noqa: E402
from slicemech.data import puffer_fetch as PF  # noqa: E402
from slicemech.data import tractor as TR  # noqa: E402

DAY = "2024-09-01"                       # a TEST date NAME; every object below is synthetic


# ------------------------------------------------------------------------------------------------ helpers
def video_size_body(day: str, channels=PU.CHANNELS, formats=PU.FORMATS, rows: int = 2) -> bytes:
    """A small valid stored ``video_size`` object (gzip CSV) for ``day``."""
    d0, _ = PU.day_bounds_ns(day)
    lines = [",".join(PU.SCHEMA)]
    for c in channels:
        for f in formats:
            for k in range(rows):
                lines.append(f"{d0 + 1_000 + k * PU.PACING_NS},{f},{c},{k * PU.VIDEO_TS_STEP},{1000 + 7 * k}")
    return gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0)


class _Resp:
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        import io as _io
        self._b, self.status, self.headers = _io.BytesIO(body), status, dict(headers or {})

    def read(self, n=-1):
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_gcs(objects: dict, missing=(), tamper=None):
    """Opener serving ``objects`` {date: stored bytes}; ``tamper(day, attempt, body)`` may alter a media body."""
    metas = {d: dict(bucket=PF.BUCKET, name=PU.bucket_object(d), size=str(len(b)), generation=str(1000 + i),
                     metageneration="1", contentType="text/csv", contentEncoding="gzip",
                     md5Hash=base64.b64encode(hashlib.md5(b).digest()).decode())
             for i, (d, b) in enumerate(sorted(objects.items()))}
    media_calls: dict = {}

    def opener(req, timeout=None):
        url = req.full_url
        for d, m in metas.items():
            if url == PF.metadata_url(d):
                if d in missing:
                    raise HTTPError(url, 404, "Not Found", {}, None)
                return _Resp(json.dumps(m).encode())
            if url == PF.media_url(d, m["generation"], m["metageneration"]):
                media_calls[d] = media_calls.get(d, 0) + 1
                body = objects[d] if tamper is None else tamper(d, media_calls[d], objects[d])
                return _Resp(body, headers={"Content-Encoding": "gzip", "x-goog-generation": m["generation"],
                                            "Content-Length": m["size"]})
        if url.startswith("https://storage.googleapis.com/storage/v1/b/"):
            raise HTTPError(url, 404, "Not Found", {}, None)            # an object the bucket does not hold
        raise AssertionError(f"unexpected URL {url}")

    return opener, metas, media_calls


@pytest.fixture
def base():
    d = sio.DATA_DIR / f"_pytest_prefreeze_{os.getpid()}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    PU.load_chunks.cache_clear()


@pytest.fixture
def dirs(base, monkeypatch):
    """Puffer cache, TEST download directory and prereg/ redirected below ``base``."""
    monkeypatch.setattr(PU, "CACHE_DIR", base / "puffer")
    monkeypatch.setattr(PU, "TEST_RAW_DIR", base / "puffer_raw")
    monkeypatch.setattr(PF, "PREREG_DIR", base / "prereg")
    (base / "prereg").mkdir()
    return base


def write_freeze(base: pathlib.Path, test_dates: int = 16, name: str = "PREREG.md") -> pathlib.Path:
    ckpt = base / "ckpt.pt"
    if not ckpt.exists():
        ckpt.write_bytes(b"trained parameters")
    f = base / "prereg" / name
    f.write_text("# Pre-registration (frozen)\n\n" + PF.freeze_block([ckpt], test_dates=test_dates),
                 encoding="utf-8")
    return f


def manual_freeze(entries: dict, *, test_dates=16, rule="Sundays and Mondays") -> str:
    """A freeze block written by hand (dummy code/config hashes): ``entries`` maps rel path -> SHA-256."""
    ck = "".join(f"  {k}: {v}\n" for k, v in entries.items())
    td = "" if test_dates is None else f"test_dates: {test_dates}\n"
    return (f"# frozen\n\n```freeze\nanalysis_code_sha256: {'a' * 64}\nconfig_sha256: {'b' * 64}\ncheckpoints:\n"
            f"{ck}date_rule: {json.dumps(rule)}\n{td}```\n")


def put_raw(base: pathlib.Path, day: str, body: bytes) -> pathlib.Path:
    p = base / "puffer_raw" / day / "video_size.csv.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return p


# ------------------------------------------------------------------------------------------------ F2 (item 1)
def test_write_json_uses_process_unique_temp_names(base, monkeypatch):
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(pathlib.Path(a).name), real(a, b)))
    sio.write_json(base / "x.json", dict(a=1))
    assert sio.read_json(base / "x.json") == dict(a=1)
    assert seen and seen[0].startswith("x.json.") and f".{os.getpid()}." in seen[0] and seen[0].endswith(".tmp")
    assert not list(base.glob("*.tmp"))
    # a failing writer leaves no temp file behind
    with pytest.raises(TypeError):
        sio.write_json(base / "y.json", dict(a=object()))
    assert not list(base.glob("*.tmp")) and not (base / "y.json").exists()


def test_prepare_date_writes_summary_first_and_no_temp_files(dirs, monkeypatch):
    put_raw(dirs, DAY, video_size_body(DAY))
    order = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append((pathlib.Path(a).name, pathlib.Path(b).name)),
                                                    real(a, b)))
    out = PU.prepare_date(DAY)
    assert [b for _, b in order] == [f"{DAY}.json", f"{DAY}.parquet"]          # the parquet marks a full cache
    assert all(f".{os.getpid()}." in a for a, _ in order)
    assert out == PU.cache_paths(DAY)[0] and pq.read_table(out).num_rows == 7 * 4 * 2
    assert sio.read_json(PU.cache_paths(DAY)[1])["channels"] == sorted(PU.CHANNELS)
    assert not list((dirs / "puffer").glob("*.tmp")) and (dirs / "puffer" / f"{DAY}.lock").exists()


def _slow_counting_parse(log: pathlib.Path, real, delay: float):
    def parse(source, day):
        with open(log, "a") as f:
            f.write(f"{os.getpid()}:{threading.get_ident()}\n")
        time.sleep(delay)
        return real(source, day)
    return parse


def test_prepare_date_is_serialised_per_day_threads(dirs, monkeypatch):
    """Concurrent first users (separate open file descriptions, so flock conflicts): one parse, no error."""
    put_raw(dirs, DAY, video_size_body(DAY))
    log = dirs / "parses.log"
    monkeypatch.setattr(PU, "parse_video_size", _slow_counting_parse(log, PU.parse_video_size, 0.3))
    barrier, errors, results = threading.Barrier(4), [], []

    def run():
        barrier.wait()
        try:
            results.append(PU.prepare_date(DAY))
        except Exception as exc:                       # noqa: BLE001 - any error fails the test below
            errors.append(repr(exc))
    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert errors == [] and len(results) == 4 and len(set(results)) == 1
    assert len(log.read_text().splitlines()) == 1                             # the re-check inside the lock
    assert pq.read_table(results[0]).num_rows == 56 and not list((dirs / "puffer").glob("*.tmp"))


def _prepare_in_child(day, cache_dir, raw_dir, log, barrier, errq):
    """Spawned worker (module level so that it can be pickled): the audit's race, one E0 worker per process."""
    from slicemech.data import puffer as P
    P.CACHE_DIR, P.TEST_RAW_DIR = pathlib.Path(cache_dir), pathlib.Path(raw_dir)
    P.parse_video_size = _slow_counting_parse(pathlib.Path(log), P.parse_video_size, 0.5)
    barrier.wait(timeout=120)
    try:
        P.prepare_date(day)
        errq.put(None)
    except Exception as exc:                           # noqa: BLE001
        errq.put(repr(exc))


def test_prepare_date_is_serialised_per_day_processes(dirs):
    put_raw(dirs, DAY, video_size_body(DAY))
    log = dirs / "parses.log"
    ctx = mp.get_context("spawn")
    n = 3
    barrier, errq = ctx.Barrier(n), ctx.Queue()
    procs = [ctx.Process(target=_prepare_in_child,
                         args=(DAY, str(dirs / "puffer"), str(dirs / "puffer_raw"), str(log), barrier, errq))
             for _ in range(n)]
    for p in procs:
        p.start()
    errs = [errq.get(timeout=180) for _ in range(n)]
    for p in procs:
        p.join(60)
    assert errs == [None] * n and all(p.exitcode == 0 for p in procs)
    assert len(log.read_text().splitlines()) == 1
    out, summary = PU.cache_paths(DAY)
    assert pq.read_table(out).num_rows == 56 and sio.read_json(summary)["rows"] == 56
    assert not list((dirs / "puffer").glob("*.tmp"))


def test_prepare_date_force_and_missing_summary_rebuild(dirs, monkeypatch):
    put_raw(dirs, DAY, video_size_body(DAY))
    out, summary = PU.cache_paths(DAY)
    PU.prepare_date(DAY)
    summary.unlink()                                  # a parquet without its summary is not a complete cache
    PU.prepare_date(DAY)
    assert summary.exists()
    calls = []
    real = PU.parse_video_size
    monkeypatch.setattr(PU, "parse_video_size", lambda s, d: (calls.append(d), real(s, d))[1])
    PU.prepare_date(DAY)
    assert calls == []
    PU.prepare_date(DAY, force=True)
    assert calls == [DAY]


def test_tractor_cache_is_written_atomically(base, monkeypatch):
    raw, cache = base / "tractor_raw", base / "tractor"
    raw.mkdir()
    (raw / "urllc_pytest.csv").write_text("App name,No.,Time,Source,Destination,Protocol,Length\n"
                                          "A,1,0.0,172.30.1.250,172.30.1.1,UDP,100\n"
                                          "A,2,0.0019,172.30.1.250,172.30.1.1,UDP,50\n"
                                          "A,3,2.5,172.30.1.250,172.30.1.1,UDP,7\n")
    monkeypatch.setattr(TR, "RAW_DIR", raw)
    monkeypatch.setattr(TR, "CACHE_DIR", cache)
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append((pathlib.Path(a).name, pathlib.Path(b).name)),
                                                    real(a, b)))
    TR.load_trace.cache_clear()
    try:
        t = TR.load_trace("urllc_pytest")
        assert [b for _, b in seen] == ["urllc_pytest.json", "urllc_pytest.npz"]
        assert all(f".{os.getpid()}." in a and a.endswith(".tmp") for a, _ in seen)
        assert not list(cache.glob("*.tmp")) and int(t.bytes_ms.sum()) == 157
        TR.load_trace.cache_clear()
        np.testing.assert_array_equal(TR.load_trace("urllc_pytest").bytes_ms, t.bytes_ms)   # read back from cache
    finally:
        TR.load_trace.cache_clear()


# ------------------------------------------------------------------------------------------------ F1 (item 2)
def test_code_hashes_ignore_dot_directories(base, monkeypatch):
    code = base / "code"
    for sub in ("slicemech", "experiments"):
        (code / sub / "pkg").mkdir(parents=True)
        (code / sub / "a.py").write_text("x = 1\n")
        (code / sub / "pkg" / "b.py").write_text("y = 2\n")
    monkeypatch.setattr(PF, "CODE_DIRS", (code / "slicemech", code / "experiments"))
    monkeypatch.setattr(R, "CODE_DIR", code)
    h0, p0 = PF.analysis_code_sha256(), R.package_sha256()
    for junk in ("experiments/.ruff_cache/0.6/123", "experiments/.ruff_cache/CACHEDIR.TAG",
                 "slicemech/.mypy_cache/3.13/y.json", "slicemech/pkg/.pytest_cache/v/x",
                 "experiments/__pycache__/a.cpython-313.pyc", "slicemech/.hidden"):
        (code / junk).parent.mkdir(parents=True, exist_ok=True)
        (code / junk).write_text("cache")
    assert PF.analysis_code_sha256() == h0 and R.package_sha256() == p0
    (code / "experiments" / "a.py").write_text("x = 3\n")            # a real code change still counts
    assert PF.analysis_code_sha256() != h0
    (code / "slicemech" / "pkg" / "b.py").write_text("y = 3\n")
    assert R.package_sha256() != p0


# ------------------------------------------------------------------------------------------------ F3 (item 3)
def test_fetch_validates_content_and_retries_a_mismatch_once(dirs):
    freeze = write_freeze(dirs)
    days = PU.test_dates()[:6]
    good = {d: video_size_body(d) for d in days}
    objs = dict(good)
    objs[days[1]] = gzip.compress(b"x\n", mtime=0)                            # MD5-valid, not a video_size table
    objs[days[2]] = video_size_body(days[2], formats=PU.FORMATS[:3])         # no channel with all four formats
    objs[days[3]] = good[days[3]][:-9]                                        # truncated gzip (MD5 of the stored bytes)
    flip = {days[4]: 1, days[5]: 99}                                          # media corrupted on attempts <= k

    def tamper(d, attempt, b):
        return bytes([b[0] ^ 1]) + b[1:] if attempt <= flip.get(d, 0) else b
    opener, metas, calls = fake_gcs(objs, tamper=tamper)
    raw = dirs / "puffer_raw"

    rec = PF.fetch_date(days[0], freeze_file=freeze, opener=opener)
    assert rec["content"]["channels"] == sorted(PU.CHANNELS) and rec["failed_attempts"] == []
    assert PU.cache_paths(days[0])[0].exists()                                # prepared during the download
    for d in days[1:4]:
        with pytest.raises(PF.CorruptObject, match="content check failed"):
            PF.fetch_date(d, freeze_file=freeze, opener=opener)
        t = raw / d / "video_size.csv.gz"
        assert not t.exists() and (raw / d / "video_size.csv.gz.invalid").exists()
        assert not (raw / d / "object_metadata.json").exists()
        assert not any(p.exists() for p in PU.cache_paths(d))
    rec = PF.fetch_date(days[4], freeze_file=freeze, opener=opener)          # transient mismatch: one retry
    assert calls[days[4]] == 2 and len(rec["failed_attempts"]) == 1
    assert (raw / days[4] / "video_size.csv.gz").read_bytes() == good[days[4]]
    with pytest.raises(PF.CorruptObject, match="2 attempts"):
        PF.fetch_date(days[5], freeze_file=freeze, opener=opener)
    assert calls[days[5]] == 2 and (raw / days[5] / "video_size.csv.gz.partial").exists()
    assert not (raw / days[5] / "video_size.csv.gz").exists()


def test_fetch_keeps_a_date_with_one_missing_channel(dirs):
    freeze = write_freeze(dirs)
    body = video_size_body(DAY, channels=PU.CHANNELS[:-1])
    opener, _, _ = fake_gcs({DAY: body})
    rec = PF.fetch_date(DAY, freeze_file=freeze, opener=opener)              # §2 reassigns the missing channel
    assert rec["content"]["channels_missing"] == [PU.CHANNELS[-1]]


def test_fetch_all_replaces_a_content_invalid_date(dirs):
    freeze = write_freeze(dirs)
    dates = PU.test_dates()
    objs = {d: video_size_body(d) for d in dates}
    objs["2024-09-08"] = gzip.compress(b"time (ns GMT),format,channel,video_ts,size\n", mtime=0)   # empty table
    objs["2024-09-10"] = video_size_body("2024-09-10")       # 09-09 is a TEST Monday -> next free day is 09-10
    opener, _, _ = fake_gcs(objs)
    m = PF.fetch_all(freeze, opener=opener)
    assert m["replacements"] == {"2024-09-08": "2024-09-10"} and "2024-09-08" in m["failures"]
    assert set(m["objects"]) == (set(dates) - {"2024-09-08"}) | {"2024-09-10"}
    assert len(PU.split_dates("test")) == 16 and "2024-09-10" in PU.split_dates("test")
    assert all(PU.cache_paths(d)[0].exists() for d in m["objects"])


def test_fetch_all_stops_after_seven_failed_candidates(dirs):
    freeze = write_freeze(dirs)
    dates = PU.test_dates()
    opener, _, _ = fake_gcs({d: video_size_body(d) for d in dates}, missing={dates[0]})
    with pytest.raises(RuntimeError, match="7 failed candidates.*deviation"):
        PF.fetch_all(freeze, opener=opener)        # 09-01 and its 7 candidates (09-03..09-07, 09-10, 09-11) 404
    m = sio.read_json(PF.manifest_path())
    assert dates[0] in m["failures"] and len(m["failures"]) == 1 + PF.MAX_REPLACEMENTS


# ------------------------------------------------------------------------------------------------ E5-3/F4 (38, 40)
def test_date_rules_state_the_replacement_clause_and_match_the_code():
    clause = ("a missing or corrupt date is replaced by the next calendar day that is neither a TEST date "
              "(including the 24-date extension) nor an earlier replacement; after 7 failed candidates the fetch "
              "stops and the failure is logged as a deviation")
    assert PF.REPLACEMENT_CLAUSE == clause and PF.MAX_REPLACEMENTS == 7
    assert PF.DATE_RULES[16].startswith("Sundays and Mondays 2024-09-01..2024-10-21 (16 dates); " + clause)
    assert PF.DATE_RULES[24].startswith("Sundays and Mondays 2024-09-01..2024-11-18 (24 dates); " + clause)
    assert len(PU.test_dates()) == 16 and len(PU.test_dates(True)) == 24 and PU.test_dates(True)[-1] == "2024-11-18"
    taken = set(PU.test_dates(True))
    assert PF.replacement_date("2024-09-01", taken) == "2024-09-03"         # 09-02 is a TEST Monday
    assert PF.replacement_date("2024-09-03", taken | {"2024-09-03"}) == "2024-09-04"   # an earlier replacement
    assert PF.replacement_date("2024-10-21", taken) == "2024-10-22"
    assert PF.replacement_date("2024-10-26", taken) == "2024-10-29"         # skips the extension's 10-27/10-28
    txt = " ".join((CODE / "EXPERIMENTS.md").read_text(encoding="utf-8").split())
    for n in (16, 24):
        assert PF.DATE_RULES[n] in txt, n                                    # the run book quotes the exact rule


# ------------------------------------------------------------------------------------------------ F5 (item 15)
def test_test_dates_is_a_required_and_consistent_freeze_field(dirs):
    rule16, rule24 = PF.DATE_RULES[16], PF.DATE_RULES[24]
    assert PF.parse_freeze(manual_freeze({"a.pt": "c" * 64}, test_dates=16))["test_dates"] == 16
    assert PF.parse_freeze(manual_freeze({"a.pt": "c" * 64}, test_dates=24, rule=rule24))["test_dates"] == 24
    for kw in (dict(test_dates=None), dict(test_dates=20), dict(test_dates='"16"'), dict(test_dates="true"),
               dict(test_dates=16, rule=rule24), dict(test_dates=24, rule=rule16)):
        with pytest.raises(PermissionError):
            PF.parse_freeze(manual_freeze({"a.pt": "c" * 64}, **kw))
    ckpt = dirs / "c.pt"
    ckpt.write_bytes(b"p")
    assert "test_dates: 24" in PF.freeze_block([ckpt], test_dates=24)
    assert rule24 in PF.freeze_block([ckpt], test_dates=24)                  # the default rule, on one line
    for kw in (dict(test_dates=20), dict(date_rule=rule24, test_dates=16), dict(test_dates=True)):
        with pytest.raises(ValueError):
            PF.freeze_block([ckpt], **kw)
    with pytest.raises(SystemExit):
        PF.main(["--freeze-block", "--checkpoint", str(ckpt)])               # --test-dates is required


def test_fetch_all_takes_the_date_count_from_the_freeze(dirs):
    freeze = write_freeze(dirs, test_dates=24)
    dates = PU.test_dates(True)
    opener, _, _ = fake_gcs({d: video_size_body(d) for d in dates})
    with pytest.raises(ValueError, match="contradicts"):
        PF.fetch_all(freeze, extended=False, opener=opener)
    m = PF.fetch_all(freeze, opener=opener)                                   # no flag: 24 from the freeze
    assert m["extended"] and set(m["objects"]) == set(dates) and len(PU.split_dates("test")) == 24
    assert PF.require_freeze_for_test_reads(freeze)["test_dates"] == 24
    man = sio.read_json(PF.manifest_path())
    sio.write_json(PF.manifest_path(), dict(man, extended=False))            # a manifest that disagrees
    with pytest.raises(PermissionError, match="test_dates = 24"):
        PF.require_freeze_for_test_reads(freeze)
    with pytest.raises(PermissionError):
        PF.require_test_read_gate()
    with pytest.raises(ValueError, match="contradicts"):
        PF.fetch_all(freeze, opener=opener)


def test_a_sixteen_date_freeze_never_fetches_the_extension(dirs):
    freeze = write_freeze(dirs, test_dates=16)
    opener, _, calls = fake_gcs({d: video_size_body(d) for d in PU.test_dates(True)})
    with pytest.raises(ValueError, match="contradicts"):
        PF.fetch_all(freeze, extended=True, opener=opener)                   # a stray --extended is refused
    m = PF.fetch_all(freeze, opener=opener)
    assert not m["extended"] and set(m["objects"]) == set(PU.test_dates()) and set(calls) == set(PU.test_dates())
    assert PF.require_freeze_for_test_reads(freeze)["deviations"] == []


# ------------------------------------------------------------------------------------------------ F12 (item 27)
def test_a_stale_partial_does_not_block_a_resume(dirs):
    freeze = write_freeze(dirs)
    stale = dirs / "puffer_raw" / DAY / "video_size.csv.gz.partial"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"interrupted download")
    opener, _, _ = fake_gcs({DAY: video_size_body(DAY)})
    rec = PF.fetch_date(DAY, freeze_file=freeze, opener=opener)
    assert pathlib.Path(rec["path"]).read_bytes() == video_size_body(DAY) and not stale.exists()


def test_fetch_all_checks_that_the_manifest_covers_the_test_dates(dirs):
    freeze = write_freeze(dirs)
    dates = PU.test_dates()
    opener, _, _ = fake_gcs({d: video_size_body(d) for d in dates})
    m = PF.fetch_all(freeze, opener=opener)
    sio.write_json(PF.manifest_path(), dict(m, objects=dict(m["objects"], **{"2024-09-03": {}})))
    with pytest.raises(RuntimeError, match="does not cover exactly"):
        PF.fetch_all(freeze, opener=opener)


# ------------------------------------------------------------------------------------------------ E5-7/F8 (item 17)
def _registry(root: pathlib.Path) -> dict:
    out = {}
    for cfg, arm in (("default", "protected"), ("n6", "unconstrained")):
        d = root / cfg / arm / "seed0"
        d.mkdir(parents=True)
        (d / R.MODEL_FILE).write_bytes(f"net {cfg} {arm}".encode())
        (d / R.CONFIG_FILE).write_text(json.dumps(dict(status="done", hyperparameters=dict(lr=3e-4))))
        out[(cfg, arm)] = d
    tune = root / "default" / "protected" / "tune" / "cell" / "seed0"
    tune.mkdir(parents=True)
    (tune / R.MODEL_FILE).write_bytes(b"tuning run")
    (tune / R.CONFIG_FILE).write_text("{}")
    st = root / "default" / "static"
    st.mkdir(parents=True)
    (st / R.STATIC_FILE).write_text(json.dumps(dict(status="done", w=[0.5, 0.3, 0.2])))
    (st / R.STATIC_CANDIDATES).write_bytes(b"candidates")
    (root / R.SELECTION_FILE).write_text(json.dumps(dict(status="done")))
    return out


def test_checkpoint_paths_freeze_every_asset_a_driver_reads(base, monkeypatch):
    root = base / "models"
    d = _registry(root)
    paths = R.checkpoint_paths(root=root)
    rel = sorted(p.relative_to(root).as_posix() for p in paths)
    assert rel == sorted(["default/protected/seed0/model.pt", "default/protected/seed0/config.json",
                          "n6/unconstrained/seed0/model.pt", "n6/unconstrained/seed0/config.json",
                          "default/static/static.json", "default/static/candidates.npz", "selection.json"])
    # the full registry adds the TRAIN assets outside the models root
    monkeypatch.setattr(R, "MODELS_DIR", root)
    monkeypatch.setattr(R, "USAGE_TARIFF_DIR", base / "tariff")
    monkeypatch.setattr(R, "SWEEP_TARIFF_DIR", base / "e1")
    monkeypatch.setattr(R, "NR_CALIBRATION_FILE", base / "e7" / "nr_calibration.json")
    for p in (base / "tariff" / "nu_protected_seed0_abc.json", R.sweep_tariff_path("rho0.5"),
              R.sweep_tariff_path("n9"), base / "e7" / "nr_calibration.json"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    full = R.checkpoint_paths()
    assert {base / "tariff" / "nu_protected_seed0_abc.json", base / "e1" / "rho0.5" / "usage_nu.json",
            base / "e1" / "n9" / "usage_nu.json", base / "e7" / "nr_calibration.json"} <= set(full)
    assert set(paths) <= set(full) and len(full) == len(set(full))
    with pytest.raises(ValueError):
        R.sweep_tariff_path("default")


def test_verify_frozen_checks_a_models_record_and_a_static_fits_candidates(base):
    root = base / "models"
    d = _registry(root)
    model = d[("default", "protected")] / R.MODEL_FILE
    static = root / "default" / "static" / R.STATIC_FILE

    def freeze_of(paths):
        f = base / "FREEZE.md"
        f.write_text(manual_freeze({p.relative_to(sio.ROOT).as_posix(): sio.file_sha256(p) for p in paths}))
        return f
    f = freeze_of([model, static])                                            # the old freeze: no companions
    with pytest.raises(PermissionError, match="config.json"):
        R.verify_frozen([model], f)
    with pytest.raises(PermissionError, match="candidates.npz"):
        R.verify_frozen([static], f)
    f = freeze_of(R.checkpoint_paths(root=root))
    checked = R.verify_frozen([model, static], f)
    assert any(c.endswith("seed0/config.json") for c in checked) and any(c.endswith("candidates.npz")
                                                                         for c in checked)
    rec = model.parent / R.CONFIG_FILE                                        # E5 reads lr/M/log_sigma from it
    rec.write_text(json.dumps(dict(status="done", hyperparameters=dict(lr=1e-2))))
    with pytest.raises(PermissionError, match="config.json"):
        R.verify_frozen([model], f)
    rec.unlink()                                                              # a listed companion must exist
    with pytest.raises(PermissionError, match="config.json"):
        R.verify_frozen([model], f)
    plain = base / "fa.pt"                                                    # not a registry model: no companion
    plain.write_bytes(b"x")
    assert R.verify_frozen([plain], freeze_of([plain])) == [plain.relative_to(sio.ROOT).as_posix()]


# ------------------------------------------------------------------------------------------------ F6 (item 16), E7
@pytest.fixture
def e7(base):
    import e7_nr as E7
    paths = E7.Paths(out=base / "nr", staging=base / "staging", summary=base / "summary", min_free_root_gb=0.0,
                     min_free_out_gb=0.0)
    return E7, paths


def _nr_calibration(paths, chosen, binary_sha) -> pathlib.Path:
    p = pathlib.Path(paths.summary) / "nr_calibration.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(chosen=chosen, binary_sha256=binary_sha, runs=[])))
    return p


def test_e7_test_cases_need_the_frozen_nr_calibration(e7, base, monkeypatch):
    E7, paths = e7
    ctx = E7.Ctx(paths=paths, binary=sys.executable, binary_sha="b" * 64)
    freeze = base / "FREEZE.md"
    freeze.write_text(manual_freeze({"x.pt": "c" * 64}))
    with pytest.raises(PermissionError, match="frozen NR calibration"):
        E7.check_frozen_calibration(paths, ctx, freeze)                       # no calibration at all
    cal = _nr_calibration(paths, E7.E7_RADIUS_M, "b" * 64)
    with pytest.raises(PermissionError, match="not frozen"):
        E7.check_frozen_calibration(paths, ctx, freeze)                       # not in the freeze block

    def frozen():
        freeze.write_text(manual_freeze({cal.relative_to(sio.ROOT).as_posix(): sio.file_sha256(cal)}))
    frozen()
    assert E7.check_frozen_calibration(paths, ctx, freeze)["chosen"] == 250 == E7.E7_RADIUS_M
    with pytest.raises(PermissionError, match="ns-3 binary"):
        E7.check_frozen_calibration(paths, E7.Ctx(paths=paths, binary_sha="d" * 64), freeze)   # rebuilt ns-3
    cal = _nr_calibration(paths, 200, "b" * 64)
    with pytest.raises(PermissionError, match="not frozen"):
        E7.check_frozen_calibration(paths, ctx, freeze)                       # recalibrated after the freeze
    frozen()
    with pytest.raises(PermissionError, match="E7_RADIUS_M"):
        E7.check_frozen_calibration(paths, ctx, freeze)


def test_e7_main_on_test_refuses_dry_run_and_an_unfrozen_calibration(e7, base, monkeypatch):
    E7, paths = e7
    from slicemech.data import calibrate, puffer_fetch

    def boom(*a, **k):
        raise AssertionError("a TEST source was touched")
    monkeypatch.setattr(calibrate.Calibration, "window", boom)
    monkeypatch.setattr(E7, "compose", boom)
    with pytest.raises(SystemExit, match="--dry-run"):                        # refused before the freeze gate
        E7.main(["--split", "test", "--stage", "cases", "--dry-run", "--binary", sys.executable], paths=paths)
    ck = {k: base / f"{k}.pt" for k in E7.KINDS}
    for k, p in ck.items():
        p.write_bytes(k.encode())
    freeze = base / "FREEZE.md"
    freeze.write_text(manual_freeze({p.relative_to(sio.ROOT).as_posix(): sio.file_sha256(p) for p in ck.values()}))
    monkeypatch.setattr(puffer_fetch, "require_test_read_gate", lambda tf=None: dict(path=str(freeze)))
    monkeypatch.setattr(E7, "expected_ckpt", lambda kind: ck[kind])
    _nr_calibration(paths, E7.E7_RADIUS_M, "irrelevant")
    with pytest.raises(PermissionError, match="not frozen"):
        E7.main(["--split", "test", "--stage", "cases", "--binary", sys.executable], paths=paths)
    assert not pathlib.Path(paths.out).exists()


def test_e7_case_offset_help_is_current():
    import e7_nr as E7
    from slicemech.data import compose
    src = (CODE / "experiments" / "e7_nr.py").read_text()
    assert "default: the first window of each date" not in src and "23:01 UTC" in src
    offs = compose.window_offsets("test")                                     # pure: no data read
    assert E7.CASE_OFFSET_S == 43260 and list(offs).index(E7.CASE_OFFSET_S) == 4      # the 5th window
