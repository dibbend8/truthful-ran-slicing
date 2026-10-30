"""Tests for ``experiments/e7_nr.py`` (E7 NR quota replay driver). ns-3 is never run: a fake runner reads the budget
and schedule named on the command line and writes a KPI file in the scenario's format."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from slicemech import io as sio
from slicemech.config import ROOT, Contract
from slicemech.nr import run as R
from slicemech.nr import schedule as S
from slicemech.sim import fluid

# the driver next to these tests (code_next/ before the rename, code/ after it), not always ROOT/code
_spec = importlib.util.spec_from_file_location("e7_nr",
                                               Path(__file__).resolve().parents[1] / "experiments" / "e7_nr.py")
E7 = importlib.util.module_from_spec(_spec)
sys.modules["e7_nr"] = E7
_spec.loader.exec_module(E7)


@pytest.fixture
def paths(request):
    """Output roots inside the project tree (tests never write outside rewrite_2026-10-18/); one directory per test
    and process, so concurrent test sessions cannot delete each other's files."""
    base = ROOT / "data" / "_pytest_e7" / f"{request.node.name.replace('[', '_').replace(']', '')}_{os.getpid()}"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    yield E7.Paths(out=base / "out", staging=base / "staging", summary=base / "summary", min_free_root_gb=0.0,
                   min_free_out_gb=0.0)
    shutil.rmtree(base, ignore_errors=True)
    try:
        base.parent.rmdir()
    except OSError:
        pass


def fake_runner(rate=0.45, used_frac=(0.95, 0.5, 0.5), calls=None, rate_of_radius=None):
    """Stand-in for subprocess.run (vslice-nr-5g.cc offline mode): KPI rows with quota = max(minPrb, budget),
    prb_used_mean = used_frac x quota, PDCP = schedule bytes, new-TB bytes = rate x used PRB x 1 s."""
    def run(cmd, cwd, env, capture_output, timeout):
        args = dict(a[2:].split("=", 1) for a in cmd[1:])
        if calls is not None:
            calls.append(args)
        K, minp = int(args["epochs"]), int(args["minPrb"])
        budget = np.loadtxt(args["budgetFile"], delimiter=",", ndmin=2)
        sb = S.read_schedule(args["scheduleFile"]).reshape(K, 1000, 3, -1).sum((1, 3)).astype(float)
        r = rate if rate_of_radius is None else rate_of_radius[float(args["discRadius"])]
        rows = []
        for k in range(K):
            for s, name in enumerate(S.NR_SLICES):
                req = float(budget[min(k, len(budget) - 1), s])
                quota = max(minp, int(round(req)))
                used = used_frac[s] * quota
                b = sb[k, s]
                rows.append([k, name, req, quota, used, b, b, b, 0.9 * b, 0.85 * b, 0.0, 10.0, 9.0, 12.5, 10.0, 40.0,
                             80.0, 120.0, 0.94, 0 if k == 0 else 1, 0 if k == 0 else 10, r * used * S.BYTES_PER_MBIT,
                             17.3])
        with open(args["kpiFile"], "w", newline="\n") as f:
            f.write(",".join(R.KPI_COLUMNS) + "\n")
            for row in rows:
                f.write(",".join(x if isinstance(x, str) else f"{x:.10g}" for x in row) + "\n")
        return subprocess.CompletedProcess(cmd, 0, b"[ns-3] fake run\n", b"")
    return run


# ------------------------------------------------------------------------------------------------ definitions, gates
def test_case_definitions_are_pure_and_fixed(monkeypatch):
    from slicemech.data import calibrate, compose, puffer

    def boom(*a, **k):
        raise AssertionError("defining cases must not read data")
    monkeypatch.setattr(calibrate.Calibration, "window", boom)
    monkeypatch.setattr(compose, "compose_parts", boom)
    cases = E7.define_cases("test")
    assert len(cases) == 12 and len({c.tag for c in cases}) == 12
    dates = puffer.split_dates("test")[:4]
    assert [c.date for c in cases[::3]] == list(dates)
    assert {c.offset for c in cases} == {E7.CASE_OFFSET_S if E7.CASE_OFFSET_S is not None else 60}
    assert [c.arm for c in cases[:3]] == ["protected_truthful", "unconstrained_truthful", "unconstrained_br"]
    for j in range(4):                                       # common random numbers across the arms of a window
        assert {c.rng_run for c in cases[3 * j:3 * j + 3]} == {1 + j}
    assert sorted({c.rng_run for c in cases}) == list(E7.CAL_RNG_RUNS)
    pilot = E7.define_pilot()
    off = E7.CASE_OFFSET_S if E7.CASE_OFFSET_S is not None else 60   # orchestrator decision 2026-10-19
    assert [(c.split, c.date, c.offset, c.arm) for c in pilot] == [
        ("train", "2024-01-01", off, "protected_truthful"), ("train", "2024-01-08", off, "protected_truthful")]
    out = E7.main(["--split", "test", "--list-cases"])
    assert len(out["cases"]) == 12
    alt = E7.define_cases("test", 43260)                     # pre-declared on every split
    assert {c.offset for c in alt} == {43260} and [c.date for c in alt[::3]] == list(dates)
    assert {c.offset for c in E7.define_pilot(43260)} == {43260}
    with pytest.raises(ValueError, match="pre-declared"):
        E7.define_cases("test", 7260)                        # a TRAIN-only offset
    out = E7.main(["--split", "train", "--list-cases", "--case-offset", "43260"])
    assert {c["offset"] for c in out["cases"] + out["pilot"]} == {43260}
    with pytest.raises(ValueError):
        cases[2].episode()                                   # the BR case needs its deviating tenant
    assert cases[2].episode(1).br_tenant == 1 and cases[1].episode(1).br_tenant == -1


def test_test_split_is_refused_before_the_freeze(paths, monkeypatch):
    from slicemech.data import calibrate, puffer_fetch
    monkeypatch.setattr(puffer_fetch, "_recorded_freeze", lambda: None)

    def boom(*a, **k):
        raise AssertionError("TEST data touched before the freeze")
    monkeypatch.setattr(calibrate.Calibration, "window", boom)
    monkeypatch.setattr(E7, "compose", boom)
    with pytest.raises(PermissionError):
        E7.main(["--split", "test", "--stage", "cases", "--binary", sys.executable], runner=fake_runner(),
                paths=paths)
    assert not Path(paths.out).exists()
    with pytest.raises(PermissionError):
        E7.require_split_allowed("test", "cases")
    for stage in ("calibrate", "pilot", "smoke"):            # TRAIN-only stages
        for split in ("dev", "test"):
            with pytest.raises(ValueError):
                E7.require_split_allowed(split, stage)
    assert E7.require_split_allowed("dev", "cases") is None


def test_test_split_needs_frozen_checkpoints(paths, monkeypatch):
    from slicemech.data import puffer_fetch
    base = Path(paths.out).parent
    ck_a, ck_b = base / "a.pt", base / "b.pt"
    ck_a.write_bytes(b"protected")
    ck_b.write_bytes(b"unconstrained")
    freeze = base / "PREREG.md"
    block = (f"```freeze\nanalysis_code_sha256: {'a' * 64}\nconfig_sha256: {'b' * 64}\n"
             f"checkpoints:\n  a.pt: {sio.file_sha256(ck_a)}\ndate_rule: first window of the first 4 dates\n"
             "test_dates: 16\n```\n")
    freeze.write_text("# Pre-registration\n\n" + block)
    monkeypatch.setattr(puffer_fetch, "require_test_read_gate", lambda tf=None: dict(path=str(freeze)))
    monkeypatch.setattr(E7, "expected_ckpt", lambda kind: {"protected": ck_a, "unconstrained": ck_b}[kind])
    with pytest.raises(PermissionError, match="not among the frozen"):
        E7.require_split_allowed("test", "cases", freeze, {"protected": ck_a, "unconstrained": ck_b})
    with pytest.raises(PermissionError, match="frozen unconstrained"):
        E7.require_split_allowed("test", "cases", freeze, {"protected": ck_a})
    freeze.write_text("# Pre-registration\n\n" + block.replace(
        "date_rule", f"  b.pt: {sio.file_sha256(ck_b)}\ndate_rule"))
    assert E7.require_split_allowed("test", "cases", freeze, {"protected": ck_a, "unconstrained": ck_b})["path"]


# ------------------------------------------------------------------------------------------------ small rules
def test_select_br_tenant_rule():
    def rec(kind, i, g):
        return dict(episode=dict(kind=kind, br_tenant=i), br_gain_mean=g)
    recs = [rec("unconstrained", 0, 0.1), rec("unconstrained", 1, 0.3), rec("unconstrained", 2, 0.3),
            rec("protected", 0, 9.0), dict(episode=dict(kind="unconstrained", br_tenant=-1))]
    assert E7.select_br_tenant(recs) == 1                     # tie between 1 and 2 -> lowest index
    with pytest.raises(ValueError):
        E7.select_br_tenant(recs[:2])


def test_fill_rates():
    r = np.array([[np.nan, 0.4, np.nan], [0.5, np.nan, np.nan], [np.nan, 0.3, np.nan], [0.6, 0.0, np.nan]])
    fb = np.full((4, 3), 0.42)
    out, filled = E7.fill_rates(r, fb)
    np.testing.assert_allclose(out[:, 0], [0.5, 0.5, 0.5, 0.6])      # leading NaN: nearest later value
    np.testing.assert_allclose(out[:, 1], [0.4, 0.4, 0.3, 0.3])      # zero is not a rate
    np.testing.assert_allclose(out[:, 2], fb[:, 2])                   # never used a PRB -> fallback
    assert filled.sum() == 8


def test_choose_radius():
    def rows(rates):
        return [dict(radius=R, used_prb_s=100.0, tb_new_mbit=100.0 * v, saturated=True) for R, v in rates.items()]
    rb = 0.4372
    res = E7.choose_radius(rows({30: 0.62, 60: 0.50, 120: 0.40, 200: 0.30}), rb)
    assert res["chosen"] == 120                               # 60 (x1.14) and 120 (x0.91) qualify; 120 is closer
    assert [t["within"] for t in res["table"]] == [False, True, True, False]
    assert E7.choose_radius(rows({30: 0.9, 200: 0.1}), rb)["chosen"] is None
    tie = E7.choose_radius(rows({60: rb * 1.1, 120: rb / 1.1}), rb)
    assert tie["chosen"] == 60
    pooled = [dict(radius=60, used_prb_s=300.0, tb_new_mbit=300.0 * 0.40, saturated=True),
              dict(radius=60, used_prb_s=100.0, tb_new_mbit=100.0 * 0.60, saturated=False)]
    t = E7.choose_radius(pooled, rb)["table"][0]
    assert t["rate"] == pytest.approx(0.45) and not t["saturated"]   # pooled by used PRB-seconds


def test_binding_gate():
    K = 10
    u = np.full((K, 3), 0.5)
    u[:5, 1] = 0.9                                            # exactly 90 % counts as binding
    u[5:7, 2] = 0.95
    t = pd.DataFrame(dict(k=np.repeat(np.arange(K), 3), tenant=np.tile(np.arange(3), K),
                          scored=np.repeat(np.arange(K) >= 2, 3), nr_quota_used_frac=u.ravel()))
    b = E7.binding(t)
    assert b["n_intervals"] == 8 and b["share_any"] == pytest.approx(5 / 8) and b["passes"]
    np.testing.assert_allclose(b["share_by_tenant"], [0, 3 / 8, 2 / 8])
    u[:, :] = 0.2
    t["nr_quota_used_frac"] = u.ravel()
    assert not E7.binding(t)["passes"]


def test_collect_run_refuses_unexpected_files(paths):
    stage = Path(paths.staging) / "x"
    stage.mkdir(parents=True)
    for f in E7.EXPECTED_FILES:
        (stage / f.format(r=3)).write_text("bin,c0,c1,c2\n0,1,2,3\n")
    (stage / "trace.pcap").write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="unexpected"):
        E7._collect_run(stage, Path(paths.out) / "x", 3)
    assert not (stage / "trace.pcap").exists()
    (stage / "trace.pcap").unlink(missing_ok=True)
    info = E7._collect_run(stage, Path(paths.out) / "y", 3)
    assert not stage.exists() and (Path(paths.out) / "y" / "schedule_run3.csv.gz").exists()
    assert not (Path(paths.out) / "y" / "schedule_run3.csv").exists() and len(info["schedule_sha256"]) == 64


def test_disk_guard(paths, monkeypatch):
    E7.disk_guard(paths)
    monkeypatch.setattr(sio, "free_gb", lambda p=None: 1.0)
    with pytest.raises(RuntimeError, match="disk guard"):
        E7.disk_guard(E7.Paths(out=paths.out, staging=paths.staging, summary=paths.summary))


# ------------------------------------------------------------------------------------------------ fluid vs NR
def _toy_window(K=6, warmup=2, seed=0):
    rng = np.random.default_rng(seed)
    cls = ("video", "urllc", "mmtc")
    mu = np.array([20.0, 8.0, 3.0])
    a = (rng.exponential(1.0, (K * 1000, 3)) * mu / 1000.0).astype(np.float32)
    ct = Contract(cls=cls, g=0.4 * mu, M=3 * mu, C=5 * mu, pi=np.array([1.0, 3.0, 0.5]),
                  delay_ms=np.array([2002.0, 150.0, 300.0]), mu=mu)
    return SimpleNamespace(arrivals_ms=a, rate_true=rng.uniform(0.3, 0.6, (K, 3)), contract=ct, cls=cls,
                           warmup=warmup, K=K)


def _kpi_for(window, q_adj, rate_tenant, work):
    """KPI table whose NR per-PRB rate equals ``rate_tenant`` (K, 3, tenant order) in every interval."""
    K = q_adj.shape[0]
    order = S.slice_order(window.cls)
    sb = np.asarray(window.arrivals_ms, float).reshape(K, 1000, 3).sum(1) * S.BYTES_PER_MBIT
    rows = []
    for k in range(K):
        for s, name in enumerate(S.NR_SLICES):
            t = order[s]
            q = int(q_adj[k, t])
            used = 0.8 * q
            rows.append([k, name, q, q, used, sb[k, t], sb[k, t], sb[k, t], 0.9 * sb[k, t], 0.8 * sb[k, t], 0.0, 1.0,
                         1.0, 5.0, 5.0, 9.0, 9.0, 9.0, 1.0, 0, 0, rate_tenant[k, t] * used * S.BYTES_PER_MBIT, 10.0])
    p = work / "kpi.csv"
    with open(p, "w") as f:
        f.write(",".join(R.KPI_COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(x if isinstance(x, str) else f"{x:.17g}" for x in r) + "\n")
    return R.read_kpi(p, cls=window.cls)


def test_fluid_table_matches_direct_replays(paths):
    w = _toy_window()
    q_ep = np.tile(np.array([40, 2, 20]), (w.K, 1))
    q_adj = np.tile(np.array([40, 4, 18]), (w.K, 1))
    kpi = _kpi_for(w, q_adj, w.rate_true, Path(paths.out).parent)
    t = E7.fluid_table(w, q_ep, q_adj, kpi)
    assert len(t) == 3 * w.K and t["adjusted"].sum() == 2 * w.K and t["scored"].sum() == 3 * (w.K - w.warmup)
    A = fluid.cumulative_arrivals(w.arrivals_ms)
    late, _, _ = fluid.replay(A, q_adj.astype(float), w.rate_true, w.contract.delay_ms, late_counted=True)
    exp, _, _ = fluid.replay(A, q_adj.astype(float), w.rate_true, w.contract.delay_ms)
    np.testing.assert_array_equal(t["fluid_raca_on_time"], late.on_time_Mbit.ravel())
    np.testing.assert_array_equal(t["fluid_raca_late"], late.late_Mbit.ravel())
    np.testing.assert_array_equal(t["fluid_raca_expiry_dropped"], exp.dropped_Mbit.ravel())
    assert (t["fluid_raca_dropped"] == 0).all() and (t["fluid_raca_expiry_late"] == 0).all()
    # NR rate == Raca rate in every interval -> the calibrated fluid equals the Raca fluid (to KPI print precision)
    np.testing.assert_allclose(t["r_nr"], t["r_raca"], rtol=1e-12)
    np.testing.assert_allclose(t["fluid_nr_served"], t["fluid_raca_served"], rtol=1e-9, atol=1e-9)
    res = t["fluid_raca_arrivals"] - t["fluid_raca_served"] - t.groupby("tenant")["fluid_raca_backlog"].diff().fillna(
        t["fluid_raca_backlog"])
    assert np.abs(res).max() < 1e-9                                       # late-counted conservation
    np.testing.assert_allclose(t["nr_pdcp_sdu_mbit"], t["fluid_raca_arrivals"], rtol=1e-9)
    summ = E7.case_summary(t)
    assert set(summ["per_tenant"]) == {0, 1, 2} and summ["adjusted_intervals"] == w.K
    assert summ["per_tenant"][0]["nr_delivered_mbit"] == pytest.approx(
        0.9 * t[(t.tenant == 0) & t.scored]["fluid_raca_arrivals"].mean())


def _rec(arm, date, br, per):
    return dict(tag=f"{date}__{arm}", case=dict(date=date, offset=60, arm=arm), br_tenant=br,
                summary=dict(per_tenant={str(i): d for i, d in per.items()}, binding=dict(share_any=1.0, passes=True)))


def _per(nr_pdb, fl, nr_all=None, corr=0.9):
    out = {}
    for i in range(3):
        d = dict(nr_delivered_pdb_mbit=nr_pdb[i], nr_delivered_mbit=(nr_all or nr_pdb)[i],
                 corr_delivered_fluid_cal=corr, corr_delivered_fluid_nr=0.9999)
        for tag, _, _ in E7.FLUID_VARIANTS:
            d[f"{tag}_on_time"] = fl[i]
        out[i] = d
    return out


def test_h7_summary():
    recs = []
    for j, (nr_ub, fl_ub) in enumerate([((10, 7, 3), (10, 8, 3)),       # NR -3, fluid -2: agree, 1.5x
                                        ((10, 9, 3), (10, 6, 3)),       # NR -1, fluid -4: agree, 0.25x
                                        ((10, 12, 3), (10, 8, 3)),      # NR +2, fluid -2: disagree
                                        ((10, 7, 3), (10, 7, 3))]):     # NR -3, fluid -3: agree, 1x
        d = f"2024-09-0{j + 1}"
        recs += [_rec("protected_truthful", d, -1, _per((10, 11, 3), (10, 11, 3))),
                 _rec("unconstrained_truthful", d, -1, _per((10, 10, 3), (10, 10, 3))),
                 _rec("unconstrained_br", d, 0, _per(nr_ub, fl_ub, corr=0.7 if j == 3 else 0.9))]
    recs.append(_rec("protected_truthful", "2024-10-01", -1, _per((1, 1, 1), (1, 1, 1))))   # incomplete window
    h = E7.h7_summary(recs)
    assert len(h["windows"]) == 4 and h["n_cases"] == 13
    w0 = h["windows"][0]
    assert w0["victims"] == [1, 2] and w0["victim_effect"]["nr"] == pytest.approx(-3)
    assert w0["h7a"]["fluid_raca"]["ratio"] == pytest.approx(1.5)
    assert h["h7a"]["fluid_raca"]["sign_agree"] == 3 and h["h7a"]["fluid_raca"]["sign_criterion"]
    assert h["h7a"]["fluid_raca"]["within_2x"] == 2 and not h["h7a"]["fluid_raca"]["magnitude_criterion"]
    assert not h["h7a"]["fluid_raca"]["holds"]                # round 3: both criteria are needed
    assert h["h7b"]["holds"] == 3                             # protected 11+3 beats BR except in window 2 (12+3)
    # round 3: H7c is tested on the out-of-sample calibrated fluid; the in-sample fluid_nr is a decomposition only
    assert h["h7c"]["variant"] == "fluid_cal" and h["h7c"]["min"] == pytest.approx(0.7)
    assert h["h7c"]["share_at_least"] < 1 and not h["h7c"]["holds"]
    assert h["h7c"]["in_sample_fluid_nr"]["min"] == pytest.approx(0.9999)


# ------------------------------------------------------------------------------------------------ end to end
def test_calibration_stage_with_fake_ns3(paths):
    ctx = E7.Ctx(paths=paths, binary=sys.executable, binary_sha="fake")
    calls = []
    run = fake_runner(calls=calls, used_frac=(1.0, 1.0, 0.95),
                      rate_of_radius={30.0: 0.62, 60.0: 0.50, 120.0: 0.40, 200.0: 0.30,
                                      250.0: 0.27, 300.0: 0.24, 400.0: 0.20})
    res = E7.stage_calibrate(ctx, 1, runner=run)
    assert res["chosen"] == 120 and len(calls) == len(E7.RADII) * len(E7.CAL_RNG_RUNS)
    assert all(c["epochs"] == str(E7.CAL_EPOCHS) and c["mcsTable"] == "1" for c in calls)
    t = {r["radius"]: r for r in res["table"]}
    assert t[120]["rate"] == pytest.approx(0.40) and t[120]["saturated"]
    assert E7.calibrated_radius(paths) == 120.0
    assert (Path(paths.out) / "calibration" / "r120_run4" / "kpi_run4.csv").exists()
    n = len(calls)
    E7.stage_calibrate(ctx, 1, runner=run)                    # resumable: nothing rerun
    assert len(calls) == n
    bad = fake_runner(rate_of_radius={float(r): 0.9 for r in E7.RADII})
    with pytest.raises(RuntimeError, match="no radius"):
        E7.stage_calibrate(E7.Ctx(paths=paths, binary=sys.executable, binary_sha="other"), 1, runner=bad)


def test_smoke_pipeline_with_fake_ns3(paths, monkeypatch):
    """Real TRAIN window (short composition), real episodes, fake ns-3: outputs, quotas, fluid and resumability."""
    monkeypatch.setattr(E7, "SMOKE_K", 6)
    monkeypatch.setattr(E7, "SMOKE_WARMUP", 2)
    calls = []
    argv = ["--smoke", "--workers", "1", "--binary", sys.executable, "--disc-radius", "60"]
    out = E7.main(argv, runner=fake_runner(calls=calls), paths=paths)
    assert len(calls) == 1 and calls[0]["epochs"] == "6" and calls[0]["discRadius"] == "60"
    assert calls[0]["pdb"] == "2002,150,300" and calls[0]["uePerSlice"] == "4" and calls[0]["minPrb"] == "4"
    case = E7.define_smoke()[0]
    d = E7.case_dir(case, paths)
    names = {p.name for p in d.iterdir()}
    assert {"schedule_run1.csv.gz", "budget_run1.csv", "budget_adjustments_run1.csv", "kpi_run1.csv", "ns3_run1.log",
            "cmd_run1.txt", "table.parquet", "case.json", "done.json"} <= names
    assert "schedule_run1.csv" not in names
    assert not any(Path(paths.staging).rglob("*.csv"))        # staging emptied
    rec = E7.load_json(d / "case.json")
    assert rec["pdcp_matches_schedule"] and rec["K"] == 6 and rec["disc_radius"] == 60.0
    assert rec["binary_sha256"] == sio.file_sha256(sys.executable)          # pre-freeze audit F6
    q_ep, _ = E7.episode_quotas(case.episode(), paths)
    q_adj, _ = S.raise_min_prb(q_ep[:, S.slice_order(("video", "urllc", "mmtc"))], B=106)
    np.testing.assert_array_equal(S.read_budget(d / "budget_run1.csv"), q_adj)     # slice order == tenant order here
    t = pd.read_parquet(d / "table.parquet")
    np.testing.assert_array_equal(t["quota"].to_numpy().reshape(6, 3), q_adj)
    assert np.allclose(t["r_nr"], 0.45) and not t["r_nr_filled"].any()
    sel = E7.load_json(Path(paths.out) / "smoke_train" / "summary" / "br_selection.json")
    assert sel["2024-01-01_o00060"]["protected_br_equals_truth"]
    assert sel["2024-01-01_o00060"]["br_tenant"] in (0, 1, 2)
    assert (Path(paths.summary) / "smoke_train" / "h7.json").exists()
    assert (Path(paths.summary) / "smoke_train" / "smoke.json").exists()
    assert len(list((Path(paths.out) / "smoke_train" / "episodes").glob("*.parquet"))) == 8
    assert out["n_cases"] == 1 and rec["summary"]["binding"]["passes"]       # fake eMBB uses 95 % of its quota
    # resumable: a second call reruns nothing; --force reruns the NR case
    mtimes = {p: p.stat().st_mtime_ns for p in (Path(paths.out) / "smoke_train" / "episodes").glob("*.parquet")}
    E7.main(argv, runner=fake_runner(calls=calls), paths=paths)
    assert len(calls) == 1
    assert all(p.stat().st_mtime_ns == m for p, m in mtimes.items())
    (d / "done.json").unlink()                                # an interrupted case is rerun from scratch
    E7.main(argv, runner=fake_runner(calls=calls), paths=paths)
    assert len(calls) == 2 and (d / "done.json").exists()
    assert all(p.stat().st_mtime_ns == m for p, m in mtimes.items())
