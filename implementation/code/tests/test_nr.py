"""Tests for slicemech.nr (schedule / budget / KPI formats of vslice-nr-5g.cc). ns-3 is never run here."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from slicemech.config import ROOT
from slicemech.nr import run as R
from slicemech.nr import schedule as S

CLS = ("urllc", "mmtc", "video")              # deliberately not in ns-3 slice order (eMBB, URLLC, mMTC)
ORDER = np.array([2, 0, 1])                   # tenant carried by slice eMBB, URLLC, mMTC


@pytest.fixture
def work(request):
    """Scratch directory inside the project tree (tests must not write outside rewrite_2026-10-18/)."""
    d = ROOT / "data" / "_pytest_nr" / request.node.name.replace("[", "_").replace("]", "")
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
    try:
        d.parent.rmdir()
    except OSError:
        pass


def _arrivals(K=3, seed=0):
    rng = np.random.default_rng(seed)
    rate = np.array([1.5, 0.3, 6.0])                        # Mbit/s per tenant (tenant order)
    a = rng.exponential(1.0, (K * 1000, 3)) * rate / 1000.0  # Mbit per ms
    return a.astype(np.float32)


# --------------------------------------------------------------------------- schedule
def test_slice_order():
    np.testing.assert_array_equal(S.slice_order(CLS), ORDER)
    np.testing.assert_array_equal(S.slice_order(("video", "urllc", "mmtc")), [0, 1, 2])
    for bad in (("video", "video", "mmtc"), ("video", "urllc"), ("video", "urllc", "mmtc", "mmtc")):
        with pytest.raises(ValueError):
            S.slice_order(bad)


def test_build_schedule_reorders_and_conserves_volume():
    a = _arrivals()
    s = S.build_schedule(a, CLS)
    b = s["bytes"]
    assert b.dtype == np.int64 and b.shape == (3000, 3) and not s["per_ue"]
    exact = np.cumsum(a.astype(np.float64)[:, ORDER] * S.BYTES_PER_MBIT, axis=0)
    assert np.max(np.abs(np.cumsum(b, axis=0) - exact)) <= 0.5 + 1e-6   # every prefix within half a byte
    per_epoch = a.astype(np.float64).reshape(3, 1000, 3).sum(1)[:, ORDER]
    np.testing.assert_allclose(s["interval_mbit"], per_epoch, rtol=0, atol=1.0 / S.BYTES_PER_MBIT)  # <= 1 B
    with pytest.raises(ValueError):
        S.build_schedule(a[:1500], CLS)                                    # not whole epochs
    with pytest.raises(ValueError):
        S.build_schedule(-a, CLS)


def test_build_schedule_per_ue():
    a = _arrivals(K=1)
    split = np.random.default_rng(1).dirichlet(np.ones(4), size=(1000, 3))
    ue = a[:, :, None].astype(np.float64) * split
    s = S.build_schedule(a, CLS, ue_arrivals_ms=ue, n_ue=4)
    assert s["per_ue"] and s["bytes"].shape == (1000, 12)
    # slice-major: columns 0..3 are the eMBB (video = tenant 2) UEs
    np.testing.assert_allclose(s["bytes"][:, :4].sum(), ue[:, 2, :].sum() * S.BYTES_PER_MBIT, atol=4)
    with pytest.raises(ValueError):
        S.build_schedule(a, CLS, ue_arrivals_ms=ue * 1.1, n_ue=4)


def test_write_schedule_format(work):
    s = S.build_schedule(_arrivals(K=2), CLS)
    p = S.write_schedule(s, work / "sched.csv")
    raw = p.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode().splitlines()
    assert lines[0] == "bin,c0,c1,c2" and len(lines) == 2001
    first = lines[1].split(",")
    assert first[0] == "0" and all(t.lstrip("-").isdigit() for t in first)   # integer bytes, as ReadCsv expects
    np.testing.assert_array_equal(S.read_schedule(p), s["bytes"])
    p10 = S.write_schedule(s, work / "sched10.csv", bin_ms=10)
    back = S.read_schedule(p10)
    assert back.shape == (200, 3)
    np.testing.assert_array_equal(back.sum(0), s["bytes"].sum(0))
    with pytest.raises(ValueError):
        S.write_schedule(s, work / "x.csv", bin_ms=7)


# --------------------------------------------------------------------------- budgets
def test_raise_min_prb_takes_from_largest_and_logs(caplog):
    q = np.array([[50, 2, 54],     # slice 1 short by 2 -> from slice 2 (largest)
                  [0, 0, 106],     # two short slices, both served by the largest
                  [30, 30, 30],    # untouched (sum 90 < B, idle PRBs stay idle)
                  [10, 3, 10]])    # tie between 0 and 2 -> lowest index donates
    with caplog.at_level(logging.INFO, logger="slicemech.nr.schedule"):
        adj, log = S.raise_min_prb(q, B=106, min_prb=4, names=S.NR_SLICES)
    np.testing.assert_array_equal(adj, [[50, 4, 52], [4, 4, 98], [30, 30, 30], [9, 4, 10]])
    np.testing.assert_array_equal(adj.sum(1), q.sum(1))                     # cell total unchanged
    assert log[["k", "recipient", "from_quota", "to_quota", "donor", "prb_moved"]].values.tolist() == [
        [0, "URLLC", 2, 4, "mMTC", 2], [1, "eMBB", 0, 4, "mMTC", 4], [1, "URLLC", 0, 4, "mMTC", 4],
        [3, "URLLC", 3, 4, "eMBB", 1]]
    assert sum("min-PRB adjustment: interval" in r.message for r in caplog.records) == 4


def test_raise_min_prb_prefer_idle():
    q = np.array([[60, 0, 20], [60, 2, 43]])                                  # 26 idle in row 0, 1 idle in row 1
    adj, log = S.raise_min_prb(q, B=106, prefer_idle=True)
    np.testing.assert_array_equal(adj, [[60, 4, 20], [59, 4, 43]])           # idle first, then the largest
    assert log[["k", "donor", "prb_moved"]].values.tolist() == [[0, "idle", 4], [1, "idle", 1], [1, "c0", 1]]
    adj_default, _ = S.raise_min_prb(q, B=106)
    np.testing.assert_array_equal(adj_default, [[56, 4, 20], [58, 4, 43]])   # contract rule: largest donates


def test_raise_min_prb_idle_fallback_and_errors():
    adj, log = S.raise_min_prb(np.array([[4, 1, 4]]), B=106)                 # no donor above 4 -> idle PRBs
    np.testing.assert_array_equal(adj, [[4, 4, 4]])
    assert log["donor"].tolist() == ["idle"] and log["prb_moved"].tolist() == [3]
    with pytest.raises(ValueError):
        S.raise_min_prb(np.array([[4, 1, 4]]), B=10)                         # 12 PRB needed, only 10
    with pytest.raises(ValueError):
        S.raise_min_prb(np.array([[50.5, 4, 4]]))                            # fractional quota
    with pytest.raises(ValueError):
        S.raise_min_prb(np.array([[100, 4, 4]]))                             # exceeds B
    with pytest.raises(ValueError):
        S.raise_min_prb(np.array([[-1, 4, 4]]))


def test_write_budget_slice_order(work):
    q = np.array([[2, 60, 40], [10, 10, 80]])                                 # tenant order (urllc, mmtc, video)
    p, q_adj, log = S.write_budget(q, CLS, work / "budget.csv")
    text = p.read_text()
    assert "\r" not in text and text.splitlines() == ["40,4,58", "80,10,10"]  # eMBB, URLLC, mMTC
    np.testing.assert_array_equal(q_adj, [[4, 58, 40], [10, 10, 80]])        # back in tenant order
    np.testing.assert_array_equal(S.read_budget(p), [[40, 4, 58], [80, 10, 10]])
    assert log["recipient"].tolist() == ["URLLC"] and log["donor"].tolist() == ["mMTC"]


# --------------------------------------------------------------------------- KPIs
def _fake_kpi_rows(K, budget_slice, sched_slice_bytes, min_prb=4, quota_bias=0):
    """Rows as vslice-nr-5g.cc EndEpoch() writes them (setprecision(10), -1 sentinels)."""
    rows = []
    for k in range(K):
        for s, name in enumerate(S.NR_SLICES):
            req = float(budget_slice[min(k, len(budget_slice) - 1), s])
            quota = max(min_prb, int(round(req))) + quota_bias
            sb = float(sched_slice_bytes[k, s])
            no_rx = (k == 0 and s == 2)
            used = 0.0 if no_rx else 0.8 * quota
            tb = 0.0 if no_rx else 0.95 * sb
            rows.append([k, name, req, quota, used, sb, sb, sb, 0.0 if no_rx else 0.9 * sb,
                         0.0 if no_rx else 0.85 * sb, 0.0, 10.0, 0.0 if no_rx else 9.0,
                         -1 if no_rx else 12.5, -1 if no_rx else 10.0, -1 if no_rx else 40.0,
                         -1 if no_rx else 80.0, -1 if no_rx else 120.0, -1 if no_rx else 0.94,
                         0 if k == 0 else 1, 0 if k == 0 else 10, tb, -1 if no_rx else 17.3])
    return rows


def _write_fake_kpi(path, rows):
    with open(path, "w", newline="\n") as f:
        f.write(",".join(R.KPI_COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(x if isinstance(x, str) else f"{x:.10g}" for x in r) + "\n")


def test_read_kpi_fake_file(work):
    K = 3
    sched = np.full((K, 3), 1.25e6)                                           # 10 Mbit per epoch per slice
    budget = np.array([[50, 30, 26]] * K)
    p = work / "kpi.csv"
    _write_fake_kpi(p, _fake_kpi_rows(K, budget, sched))
    df = R.read_kpi(p, cls=CLS)
    assert len(df) == 9 and df["epoch"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert df["slice"].tolist()[:3] == ["eMBB", "URLLC", "mMTC"]
    assert df["tenant"].tolist()[:3] == ORDER.tolist() and df["cls"].tolist()[:3] == ["video", "urllc", "mmtc"]
    row = df.iloc[3]
    assert row.sched_mbit == pytest.approx(10.0) and row.measured_mbps == pytest.approx(10.0)
    assert row.rate_mbps_per_prb == pytest.approx(9.5 / (0.8 * 50))
    assert row.loss_prev == pytest.approx(0.1) and row.quota_used_frac == pytest.approx(0.8)
    first_mmtc = df.iloc[2]
    assert np.isnan(first_mmtc.delay_p95_ms) and np.isnan(first_mmtc.mcs_mean)
    assert np.isnan(first_mmtc.rate_mbps_per_prb)
    assert np.isnan(df.iloc[0].loss_prev)
    rates = R.nr_rates(df)
    assert rates.shape == (3, 3) and np.isnan(rates[0, ORDER[2]])
    np.testing.assert_allclose(R.kpi_matrix(df, "prb_quota")[1], [30, 26, 50])   # tenant order
    bad = work / "bad.csv"
    pd.read_csv(p).rename(columns={"mcs_mean": "mcs"}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        R.read_kpi(bad)
    pd.read_csv(p).iloc[:-1].to_csv(bad, index=False)                        # missing slice row
    with pytest.raises(ValueError):
        R.read_kpi(bad)


def test_kpi_frame_json_matches_csv(work):
    K = 2
    rows = _fake_kpi_rows(K, np.array([[50, 30, 26]]), np.full((K, 3), 1e5))
    p = work / "kpi.csv"
    _write_fake_kpi(p, rows)
    msgs = []
    for k in range(K):
        block = [r for r in rows if r[0] == k]
        m = {"type": "kpi", "epoch": k}
        for j, key in enumerate(R.KPI_COLUMNS[2:], start=2):
            m[key] = [b[j] for b in block]
        msgs.append(json.loads(json.dumps(m)))
    a = R.read_kpi(p, cls=CLS)
    b = R.kpi_frame(msgs, cls=CLS)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


# --------------------------------------------------------------------------- command and open-loop run
def test_build_command_e7_flags(work):
    cmd = R.build_command(budget_file=work / "b.csv", kpi_file=work / "k.csv", schedule_file=work / "s.csv",
                          epochs=200, pdb_ms=[2002.0, 150.0, 300.0], rng_run=7, disc_radius=60.0)
    assert cmd[0] == str(R.NS3_BIN) and R.NS3_BIN.name == "ns3.46-vslice-nr-5g-default"
    for flag in ("--port=0", "--epochs=200", "--uePerSlice=4", "--pdb=2002,150,300",
                 "--rlcBuf=4194304,1048576,1048576", "--mcsTable=1", "--minPrb=4", "--rngRun=7", "--binMs=1",
                 "--discRadius=60", f"--budgetFile={work / 'b.csv'}", f"--kpiFile={work / 'k.csv'}",
                 f"--scheduleFile={work / 's.csv'}"):
        assert flag in cmd, flag
    with pytest.raises(ValueError):
        R.build_command(budget_file="b", kpi_file="k", schedule_file=None, epochs=5, pdb_ms=[1, 2])
    with pytest.raises(ValueError):
        R.build_command(budget_file="b", kpi_file="k", schedule_file=None, epochs=5, pdb_ms=[1, 2, 3],
                        extra=["verbose"])
    env = R.ns3_env()
    assert env["LD_LIBRARY_PATH"].startswith(str(R.NS3_LIB))


def _fake_runner(quota_bias=0, rc=0):
    """Stand-in for subprocess.run: reads the budget and schedule named on the command line and writes a KPI file
    in the scenario's format (ns-3 is not run)."""
    seen = {}

    def run(cmd, cwd, env, capture_output, timeout):
        args = dict(a[2:].split("=", 1) for a in cmd[1:])
        seen.update(args=args, cwd=cwd, env=env)
        K = int(args["epochs"])
        budget = np.loadtxt(args["budgetFile"], delimiter=",", ndmin=2)
        sch = S.read_schedule(args["scheduleFile"])
        sched_slice = sch.reshape(K, 1000, 3, -1).sum((1, 3)).astype(float)
        _write_fake_kpi(args["kpiFile"], _fake_kpi_rows(K, budget, sched_slice, int(args["minPrb"]), quota_bias))
        return subprocess.CompletedProcess(cmd, rc, b"[ns-3] fake run\n", b"")
    return run, seen


def _window(K=4):
    a = _arrivals(K=K)
    ct = SimpleNamespace(delay_ms=np.array([150.0, 300.0, 2002.0]))         # tenant order (urllc, mmtc, video)
    return SimpleNamespace(id="test-w0", arrivals_ms=a, cls=CLS, contract=ct)


def test_run_open_loop_with_fake_ns3(work):
    w = _window()
    quotas = np.array([[30, 2, 70], [30, 10, 60], [0, 50, 50], [40, 40, 26]])  # tenant order; row 2 has tied donors
    runner, seen = _fake_runner()
    df, meta = R.run_window(w, quotas, work / "case", runner=runner, binary=sys.executable, rng_run=3)
    args = seen["args"]
    assert args["pdb"] == "2002,150,300" and args["epochs"] == "4" and args["port"] == "0"
    assert Path(seen["cwd"]) == (work / "case").resolve()
    np.testing.assert_array_equal(meta["quotas_adjusted"], [[30, 4, 68], [30, 10, 60], [4, 48, 48], [40, 40, 26]])
    assert meta["n_adjusted_intervals"] == 2
    adj = pd.read_csv(Path(meta["workdir"]) / "budget_adjustments_run3.csv")
    assert adj["k"].tolist() == [0, 2, 2]                                     # two donors in interval 2
    assert (Path(meta["workdir"]) / "cmd_run3.txt").read_text().startswith(sys.executable)
    assert (Path(meta["workdir"]) / "ns3_run3.log").read_bytes().startswith(b"[ns-3] fake run")
    np.testing.assert_array_equal(R.kpi_matrix(df, "prb_quota"), meta["quotas_adjusted"])
    assert meta["pdcp_matches_schedule"]
    np.testing.assert_allclose(R.kpi_matrix(df, "pdcp_sdu_mbit"),
                               w.arrivals_ms.astype(np.float64).reshape(4, 1000, 3).sum(1), atol=1e-5)


def test_run_open_loop_detects_quota_mismatch_failure_and_dry_run(work):
    w = _window()
    quotas = np.array([[30, 20, 50]] * 4)
    runner, _ = _fake_runner(quota_bias=1)                                     # ns-3 applied something else
    with pytest.raises(ValueError, match="applied"):
        R.run_window(w, quotas, work / "a", runner=runner, binary=sys.executable)
    runner, _ = _fake_runner(rc=1)
    with pytest.raises(RuntimeError, match="rc=1"):
        R.run_window(w, quotas, work / "b", runner=runner, binary=sys.executable)
    df, meta = R.run_window(w, quotas, work / "c", dry_run=True)
    assert df is None and Path(meta["budget"]).exists() and Path(meta["schedule"]).exists()
    with pytest.raises(FileNotFoundError):
        R.run_window(w, quotas, work / "d", runner=_fake_runner()[0], binary=work / "no-such-binary")

    def hang(cmd, cwd, env, capture_output, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout, output=b"[ns-3] stuck at epoch 3\n", stderr=b"")
    with pytest.raises(RuntimeError, match="timed out"):
        R.run_window(w, quotas, work / "f", runner=hang, binary=sys.executable, timeout=5)
    assert (work / "f" / "ns3_run1.log").read_bytes() == b"[ns-3] stuck at epoch 3\n"   # partial output kept
    with pytest.raises(ValueError, match="outside"):
        R.run_window(w, quotas, Path("/tmp") / "slicemech_nr_test")
    with pytest.raises(ValueError):
        R.run_window(w, quotas[:3], work / "e", dry_run=True)                 # K mismatch
