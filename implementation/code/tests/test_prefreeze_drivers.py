"""Pre-freeze audit fixes of the experiment drivers (review/prefreeze_audit.md §7, items 6, 7, 8, 13, 18, 20, 22, 23,
26 and the driver part of 35).

Real TRAIN windows are composed short (K = 30 s). Every output goes below ``data/_pytest_prefreeze_drivers_<pid>``
(removed afterwards); nothing is written to ``results/`` and no TEST source is read (the TEST gate is stubbed where a
test needs to get past the option check).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True
_EXP = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import e0_setup as S  # noqa: E402
import e1_value as E1  # noqa: E402
import e2_manipulation as X  # noqa: E402
import e3_protection as E3  # noqa: E402
import e4_cost as E4  # noqa: E402
import e5_training_channel as E5  # noqa: E402
import e6_scaling as E6  # noqa: E402
from slicemech import io as sio  # noqa: E402
from slicemech.config import Mechanism  # noqa: E402
from slicemech.data import calibrate as calib  # noqa: E402

DATE, OFF = "2024-01-14", 60
K, WARM = 30, 5


@pytest.fixture(scope="module")
def tmproot():
    d = sio.DATA_DIR / f"_pytest_prefreeze_drivers_{os.getpid()}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def cal():
    try:
        return calib.load_calibration("n3_default")
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"calibration missing: {exc}")


@pytest.fixture(scope="module")
def win(cal):
    try:
        return cal.window("train", DATE, OFF, K=K, warmup=WARM)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"prepared TRAIN data missing: {exc}")


@pytest.fixture(scope="module")
def episode(win, cal):
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    return E.run_window(win, E.FixedWeights(), "full", [Truthful() for _ in range(win.n)], 2, Mechanism(), cal.shape)


# ============================================================================================ archive helper (item 7)
def test_archive_files_moves_or_copies_next_to_the_experiment_folders_and_never_overwrites(tmproot):
    root = tmproot / "results" / "raw" / "dev" / "e2"
    f = root / "stage" / "unc_fixed" / "s0" / "w.parquet"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"one")
    side = Path(str(f) + ".prov.json")
    side.write_text("{}")
    assert S.archive_anchor(root) == (tmproot / "results" / "raw" / "dev").resolve()
    base = S.archive_files([f, side, root / "missing.parquet"], root, stamp="T1", reason="test")
    assert base == (tmproot / "results" / "raw" / "dev" / "archive" / "T1").resolve()
    dest = base / "e2" / "stage" / "unc_fixed" / "s0" / "w.parquet"
    assert dest.read_bytes() == b"one" and not f.exists() and not side.exists()       # moved, not deleted
    log = [json.loads(x) for x in (base / "archive_log.jsonl").read_text().splitlines()]
    assert len(log) == 2 and log[0]["reason"] == "test" and log[0]["dst"] == str(dest)
    f.write_bytes(b"two")                                     # the same path again, same stamp: never overwritten
    S.archive_files([f], root, stamp="T1", copy=True)
    assert dest.read_bytes() == b"one" and (dest.parent / "w.dup1.parquet").read_bytes() == b"two"
    assert f.read_bytes() == b"two"                           # copy=True keeps the source
    assert S.archive_files([root / "nothing"], root, stamp="T2") is None
    other = tmproot / "elsewhere" / "out"                     # outside results/raw: next to the output folder
    assert S.archive_anchor(other) == (tmproot / "elsewhere").resolve()
    assert len(S.archive_stamp()) == len("20261020T120000Z")


# ============================================================================================ E1 (items 7, 8, 13)
def test_e1_stores_prb_inputs_and_on_time_plus_unused_equals_capacity(episode, win):
    mech = Mechanism()
    sc = E1.prb_columns(episode[episode["scored"]], mech.Delta)
    late = sc["late_Mbit"].to_numpy(float)
    assert np.all(late == 0)                                  # deadline expiry: no late service
    cap = sc["quota"].to_numpy(float) * sc["r_true"].to_numpy(float) * mech.Delta
    np.testing.assert_allclose(sc["cap_Mbit"], cap, rtol=0, atol=0)
    np.testing.assert_allclose(sc["on_time_Mbit"] + sc["unused_Mbit"], sc["cap_Mbit"], rtol=1e-9, atol=1e-7)
    assert sc["unused_Mbit"].sum() > 0 and sc["on_time_Mbit"].sum() > 0          # a non-trivial case
    assert np.all(sc["prb_busy"] <= sc["quota"] + 1e-6) and np.all(sc["prb_busy"] >= 0)   # fluid rounding
    s = E1.summarize_run(episode, win, mech, dict(arm="equal"))
    for c in ("cap_Mbit", "unused_Mbit", "prb_busy", "r_true"):
        assert c in s.columns
    np.testing.assert_allclose(s["on_time_Mbit"] + s["unused_Mbit"], s["cap_Mbit"], rtol=1e-9, atol=1e-7)
    t, a = s[s["tenant"] >= 0], s[s["tenant"] == -1].iloc[0]
    for c in ("cap_Mbit", "unused_Mbit", "prb_busy"):                           # tenant -1 = per-interval sums
        assert a[c] == pytest.approx(t[c].sum(), rel=1e-12)
    assert np.isnan(a["r_true"]) and (t["r_true"] > 0).all()
    assert a["prb_busy"] <= mech.B + 1e-6
    sct = episode[episode["scored"]]
    assert t.set_index("tenant")["unused_Mbit"].to_dict() == pytest.approx(
        sct.groupby("tenant")["unused_Mbit"].mean().to_dict(), rel=1e-12)
    iv = E1.interval_rows(episode, dict(arm="equal", seed=-1))
    assert {"unused_Mbit", "r_true"} <= set(iv.columns) and len(iv) == (K - WARM) * win.n
    assert "unused_Mbit" in E1.INTERVAL_COLS and "r_true" in E1.INTERVAL_COLS


def _plan(out: Path, code_fp: str, arms=("equal",)) -> dict:
    cfg = S.get_config("default")
    plan = dict(split="train", config="default", cfg=S.config_record(cfg), cal_sha=S.calibration_sha(cfg),
                calib_root=None, smoke=True, test_freeze=None, arms=tuple(arms), ckpts={}, static_w=[0.5, 0.4, 0.1],
                nu=[0.02, 0.01, 0.03], oracle=dict(pop=8, iters=2, seed=0, horizon=3), envelope=False,
                intervals=True, code_fp=code_fp, schema=E1.SCHEMA, out=str(out))
    plan["key"] = E1._digest({k: v for k, v in plan.items() if k != "out"})
    return plan


def test_e1_run_fingerprint_covers_schema_and_code(monkeypatch):
    p = dict(config="default", cfg={}, cal_sha="c", smoke=False, nu=None, arms=("equal", "static"), ckpts={},
             static_w=[1, 1, 1], envelope=False, oracle={}, code_fp="a")
    assert E1.SCHEMA >= 2
    fp = {r["arm"]: r["fp"] for r in E1.planned_runs(p)}
    fp_code = {r["arm"]: r["fp"] for r in E1.planned_runs(dict(p, code_fp="b"))}
    assert all(fp[a] != fp_code[a] for a in fp)
    monkeypatch.setattr(E1, "SCHEMA", E1.SCHEMA + 1)
    assert all(fp[a] != r["fp"] for r in E1.planned_runs(p) for a in (r["arm"],))
    assert set(E1.CODE_MODULES) == {"e1_value.py", "e0_setup.py", "registry.py"}
    c = E1.code_fingerprint()
    assert c == E1.code_fingerprint() and len(c) == 64


def test_e1_window_job_writes_code_fp_archives_replaced_rows_and_forces(tmproot, cal, monkeypatch):
    monkeypatch.setattr(S, "SMOKE_K", K)
    monkeypatch.setattr(S, "SMOKE_WARMUP", WARM)
    out = tmproot / "results" / "raw" / "smoke" / "e1" / "default"
    key = S.key_of(DATE, OFF)
    plan = dict(_plan(out, "code-1"), archive_stamp="A1")
    res = E1.window_job((plan, DATE, OFF))
    assert res["ran"] == 1 and res["archived"] is None
    s = pd.read_parquet(out / "windows" / f"{key}.parquet")
    assert set(s["code_fp"]) == {"code-1"} and set(s["schema"]) == {E1.SCHEMA}
    assert {"cap_Mbit", "unused_Mbit", "prb_busy", "r_true"} <= set(s.columns)
    iv = pd.read_parquet(out / "intervals" / f"{key}.parquet")
    assert {"unused_Mbit", "r_true"} <= set(iv.columns)
    assert E1.window_job((plan, DATE, OFF))["ran"] == 0                             # resumed: nothing to do
    plan2 = dict(_plan(out, "code-2"), archive_stamp="A2")                      # a code change: every run is stale
    res2 = E1.window_job((plan2, DATE, OFF))
    assert res2["ran"] == 1
    arch = tmproot / "results" / "raw" / "smoke" / "archive" / "A2" / "e1" / "default"
    assert Path(res2["archived"]) == (tmproot / "results" / "raw" / "smoke" / "archive" / "A2").resolve()
    old = pd.read_parquet(arch / "windows" / f"{key}.parquet")
    assert set(old["code_fp"]) == {"code-1"} and (arch / "intervals" / f"{key}.parquet").is_file()
    s2 = pd.read_parquet(out / "windows" / f"{key}.parquet")
    assert set(s2["code_fp"]) == {"code-2"} and len(s2) == len(s)
    assert s2["realized_value"].tolist() == s["realized_value"].tolist()           # same numbers, new code record
    assert E1.window_job((plan2, DATE, OFF))["ran"] == 0
    res3 = E1.window_job((dict(plan2, force=True, archive_stamp="A3"), DATE, OFF))  # --force recomputes
    assert res3["ran"] == 1 and (Path(res3["archived"]) / "e1" / "default" / "windows" / f"{key}.parquet").is_file()


def test_e1_test_split_refuses_protocol_options_but_allows_force(monkeypatch):
    def gate(*a, **k):
        raise RuntimeError("TEST gate reached")

    monkeypatch.setattr(S, "test_gate", gate)
    for extra in (["--intervals", "off"], ["--intervals", "on"], ["--unconstrained-envelope"]):
        with pytest.raises(SystemExit, match="refusing --split test"):
            E1.main(["--split", "test", *extra])
    with pytest.raises(RuntimeError, match="TEST gate reached"):                 # --force is not a protocol option
        E1.main(["--split", "test", "--force"])
    ns = argparse.Namespace(fit_nu=False, smoke=False)
    assert E1.fit_nu_required(ns, "test") and not E1.fit_nu_required(ns, "dev")
    assert E1.fit_nu_required(argparse.Namespace(fit_nu=True, smoke=False), "dev")


def test_e1_tariffs_live_at_the_frozen_train_files_and_are_never_refitted_on_test(tmproot, monkeypatch):
    """Coordinator item (data group, audit E5-7/F8): sweep tariffs at registry.sweep_tariff_path, default at the
    E2-E4 cache; on TEST loaded (and passed to verify_frozen by main), never refitted."""
    import registry as R
    raw = tmproot / "results" / "raw"
    monkeypatch.setattr(R, "SWEEP_TARIFF_DIR", raw / "train" / "e1")
    monkeypatch.setattr(X, "RAW", raw)
    calls = []

    def fake_pool(fn, jobs, workers):
        calls.append(len(jobs))
        yield [1.0, 2.0, 3.0], [2.0, 4.0, 12.0]

    monkeypatch.setattr(S, "run_pool", fake_pool)
    cfg, out = S.get_config("rho0.7"), tmproot / "split_out" / "e1" / "rho0.7"
    small = types.SimpleNamespace(n=3)
    models = {"protected": {0: (str(tmproot / "m.pt"), "a" * 64)}}
    nu, src, path = E1.usage_tariff(small, cfg, out, False, True, models)          # DEV --fit-nu: TRAIN fit
    assert path == R.sweep_tariff_path("rho0.7") == raw / "train" / "e1" / "rho0.7" / "usage_nu.json"
    assert nu == [0.5, 0.5, 0.25] and calls and path.is_file() and not out.exists()   # nothing below the split
    assert sio.read_json(path)["inputs"]["protected_seed0_sha256"] == "a" * 64

    def no_fit(*a, **k):
        raise AssertionError("refitted")

    monkeypatch.setattr(S, "run_pool", no_fit)
    assert E1.usage_tariff(small, cfg, out, False, False, models)[0] == nu             # read, not refitted
    assert E1.usage_tariff(small, cfg, out, False, True, models, test=True)[2] == path  # TEST: the frozen file
    other = {"protected": {0: (str(tmproot / "m.pt"), "b" * 64)}}
    with pytest.raises(SystemExit, match="fitted for other inputs"):
        E1.usage_tariff(small, cfg, out, False, True, other, test=True)
    with pytest.raises(SystemExit, match="never refits"):
        E1.usage_tariff(small, S.get_config("rho0.5"), out, False, True, models, test=True)
    monkeypatch.setattr(S, "run_pool", fake_pool)                                     # DEV refit: old file archived
    E1.usage_tariff(small, cfg, out, False, True, other)
    assert sio.read_json(path)["inputs"]["protected_seed0_sha256"] == "b" * 64
    assert len(list((raw / "train" / "archive").glob("*/e1/rho0.7/usage_nu.json"))) == 1
    # default configuration: the E2-E4 cache of the protected seed-0 model (frozen via USAGE_TARIFF_DIR)
    m = tmproot / "m.pt"
    m.write_bytes(b"model")
    dflt = S.get_config("default")
    want = X.usage_tariff_path(0, m, False)
    assert E1.tariff_file(dflt, out, False, m) == want and want.parent == raw / "train" / "e2" / "usage_tariff"
    mods = {"protected": {0: (str(m), sio.file_sha256(m))}}
    with pytest.raises(SystemExit, match="never refits"):
        E1.usage_tariff(small, dflt, out, False, True, mods, test=True)
    want.parent.mkdir(parents=True)
    want.write_text(json.dumps(dict(nu=[0.1, 0.2, 0.3])))
    nu_d, src_d, path_d = E1.usage_tariff(small, dflt, out, False, True, mods, test=True)
    assert nu_d == [0.1, 0.2, 0.3] and path_d == want and "shared with E2-E4" in src_d
    assert E1.tariff_file(cfg, out, True) == out / "usage_nu.json"                    # smoke stays local
    assert "static_src, nu_path" in inspect.getsource(E1.main)                        # verify_frozen on TEST


def test_e2_to_e4_assets_never_refit_the_tariff_on_test(tmproot, monkeypatch):
    md = tmproot / "models_e2"
    for k in ("protected", "unconstrained"):
        (md / k / "seed0").mkdir(parents=True)
        (md / k / "seed0" / "model.pt").write_bytes(k.encode())
    monkeypatch.setattr(X, "MODELS_DIR", md)
    monkeypatch.setattr(X, "RAW", tmproot / "results" / "raw")

    def no_fit(*a, **k):
        raise AssertionError("refitted on TEST")

    monkeypatch.setattr(X, "usage_tariff", no_fit)
    with pytest.raises(PermissionError, match="never refits"):
        X.resolve_assets(["unc_usage"], [0], smoke=False, test_freeze_path="frozen.md")


# ============================================================================================ E2 (items 7, 18, 22)
def test_e2_stage_allowance_is_scaled_by_delta():
    df = pd.DataFrame(dict(tenant=[0, 1, 2], theta=[1.0, 2.0, 0.0], z_star=[1.0, 3.0, 0.0]))
    ct = types.SimpleNamespace(pi=np.array([1.0, 3.0, 0.5]), mu=np.array([2.0, 1.0, 4.0]))
    w = types.SimpleNamespace(contract=ct, date="2024-01-14", id="w")
    out = X._finish_stage(df.copy(), w, "unc_fixed", 0, types.SimpleNamespace(Delta=2.0))
    np.testing.assert_allclose(out["eps"], X.EPS_REL * ct.pi * ct.mu * 2.0, rtol=1e-15)
    out1 = X._finish_stage(df.copy(), w, "unc_fixed", 0, Mechanism())
    np.testing.assert_allclose(out1["eps"], X.EPS_REL * ct.pi * ct.mu * Mechanism().Delta, rtol=1e-15)


def test_e2_npz_temp_file_does_not_end_in_npz(tmproot, monkeypatch):
    seen = []
    real = os.replace

    def rec(a, b):
        seen.append((str(a), str(b)))
        return real(a, b)

    monkeypatch.setattr(X.os, "replace", rec)
    p = tmproot / "bandit" / "prot_full" / "s0" / "exp3_run0.npz"
    X.write_npz(p, m=np.arange(3.0))
    assert len(seen) == 1 and seen[0][1] == str(p.resolve())
    assert seen[0][0].endswith(f".tmp{os.getpid()}") and not seen[0][0].endswith(".npz")
    assert np.array_equal(np.load(p)["m"], np.arange(3.0))
    assert sorted(x.name for x in p.parent.iterdir()) == ["exp3_run0.npz"]


def test_e2_runner_archives_stale_and_partial_outputs_and_removes_temp_files(tmproot):
    root = tmproot / "results" / "raw" / "dev" / "e2r"
    stale = root / "stage" / "unc_fixed" / "s0" / "w.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    Path(str(stale) + ".prov.json").write_text(json.dumps(dict(fp="old")))
    part_stage = root / "stage" / "unc_fixed" / "s0" / "v.parquet"            # interrupted replay job: only the
    part_rep = root / "replay" / "unc_fixed" / "s0" / "v.parquet"             # replay output exists
    part_rep.parent.mkdir(parents=True)
    part_rep.write_bytes(b"partial")
    bnd = root / "bandit" / "prot_full" / "s0" / "exp3_run0.npz"
    bnd.parent.mkdir(parents=True)
    (bnd.parent / "exp3_run0.npz.tmp4242.npz").write_bytes(b"x")              # old-style leftover temp file
    (stale.parent / "w.parquet.tmp17").write_bytes(b"x")
    (stale.parent / "notes.tmpfile").write_bytes(b"keep")                     # not a temp of a job output
    jobs = [X.Job("w", "m:f", dict(out=str(stale)), str(stale), fp="new"),
            X.Job("v", "m:f", dict(stage_out=str(part_stage), replay_out=str(part_rep)), str(part_stage), fp="new"),
            X.Job("b", "m:f", dict(out=str(bnd)), str(bnd), fp="new")]
    assert [X.job_status(j) for j in jobs] == ["stale", "pending", "pending"]
    assert X.clean_temp_files(jobs) == 2
    assert (stale.parent / "notes.tmpfile").exists() and not list(bnd.parent.glob("*.tmp*"))
    base = X.retire_outputs(jobs, root, stamp="R1")
    assert base == (tmproot / "results" / "raw" / "dev" / "archive" / "R1").resolve()
    a = base / "e2r"
    assert (a / "stage" / "unc_fixed" / "s0" / "w.parquet").read_bytes() == b"old"
    assert (a / "stage" / "unc_fixed" / "s0" / "w.parquet.prov.json").is_file()
    assert (a / "replay" / "unc_fixed" / "s0" / "v.parquet").read_bytes() == b"partial"
    assert not stale.exists() and not part_rep.exists()                      # nothing deleted, nothing left behind
    src = inspect.getsource(X.run_jobs)
    assert "retire_outputs" in src and ".unlink(" not in src


# ============================================================================================ E3 (items 20, 23)
def test_scored_multiplier_is_truthful_in_the_warmup(win, cal):
    from slicemech.sim import episode as E
    from slicemech.tenants.strategies import Truthful
    pol = X.ScoredMultiplier(2.0, 3)
    assert pol.report(1, 1.5, 10.0, history=[0, 0]) == 1.5                       # k = 2 < warm-up
    assert pol.report(1, 1.5, 10.0, history=[0, 0, 0]) == 3.0                    # k = 3: scored
    assert pol.report(1, 8.0, 10.0, history=[0] * 5) == 10.0 and pol.n_clipped == 1
    with pytest.raises(NotImplementedError):
        pol.report_series(np.ones(4), 10.0)
    pols = [Truthful(), X.ScoredMultiplier(2.0, win.warmup), Truthful()]
    df = E.run_window(win, E.FixedWeights(), "full", pols, 2, Mechanism(), cal.shape, record=False,
                      price_columns="billed")
    d1 = df[df["tenant"] == 1].sort_values("k")
    th = np.minimum(np.asarray(win.theta, float)[:, 1], win.contract.C[1])
    want = np.where(np.arange(win.K) < win.warmup, th,
                    np.minimum(2.0 * np.asarray(win.theta, float)[:, 1], win.contract.C[1]))
    np.testing.assert_array_equal(d1["report"].to_numpy(float), want)
    src = inspect.getsource(E3.job_e3_h3d)
    assert "ScoredMultiplier(m, w.warmup)" in src and "Multiplier(m)" not in src.replace("ScoredMultiplier(m", "")


def test_e3_certifies_the_kappa_scaled_arms():
    assert set(E4.e4b_arms()) <= set(E3.CERT_ARMS) and len(set(E3.CERT_ARMS)) == len(E3.CERT_ARMS)
    for a in E3.CERT_ARMS:
        assert X.arm_kind(a)[1] == "full"
    args = argparse.Namespace(split="train", smoke=True, smoke_windows=2, dates=None, max_windows=None, seeds=[0],
                              cert_arms=list(E3.CERT_ARMS), ident_arms=[], h3d_arms=[], ident_limit=None)
    names = {j.name.split(" ")[1] for j in E3.e3_jobs(args, Path("/nonexistent"))}
    assert {"kappa_x0", "kappa_x0.5", "kappa_x2"} <= names


# ============================================================================================ E4 gate (item 35)
def test_e2_completeness_requires_every_predeclared_job_under_the_current_fingerprint(tmproot):
    specs = {0: X.AssetSpec(seed=0, nets={}, sha256=dict(protected="p" * 64, unconstrained="u" * 64, nu="n" * 64))}
    root = tmproot / "e2c"
    st = X.e2_completeness("train", True, specs, root=root)
    assert st["jobs"] > 0 and st["done"] == 0 and not st["complete"]
    a = X.e2_defaults(X.e2_parser().parse_args(["--smoke"]))
    a.split = "train"
    jobs = X.attach_provenance(X.e2_jobs(a, root, specs), specs, split="train", smoke=True, modules=X.E2_MODULES)
    assert len(jobs) == st["jobs"]
    for j in jobs:
        Path(j.out).parent.mkdir(parents=True, exist_ok=True)
        Path(j.out).write_bytes(b"x")
        sio.write_json(X.prov_path(j.out), dict(fp=j.fp, code=j.prov["code"]))
    st = X.e2_completeness("train", True, specs, root=root)
    assert st["complete"] and st["done"] == st["jobs"] and st["code_versions"] == 1
    sio.write_json(X.prov_path(jobs[0].out), dict(fp="other", code="old-code"))      # one job of another version
    st = X.e2_completeness("train", True, specs, root=root)
    assert not st["complete"] and st["stale"] == 1 and st["code_versions"] == 2


def test_e4_refuses_to_start_on_test_until_e2_is_complete(monkeypatch, capsys):
    inc = dict(complete=False, done=3, jobs=10, pending=7, stale=0, code_versions=1, root="r")
    monkeypatch.setattr(X, "e2_completeness", lambda *a, **k: inc)
    ns = argparse.Namespace(split="test", smoke=False, tag=None, e2_root=None)
    with pytest.raises(SystemExit, match="refusing to start E4 on TEST"):
        E4.require_e2_complete(ns, {})
    assert E4.require_e2_complete(argparse.Namespace(split="dev", smoke=False, tag=None, e2_root=None), {}) == inc
    assert "warning" in capsys.readouterr().out
    monkeypatch.setattr(X, "e2_completeness", lambda *a, **k: dict(inc, complete=True))
    assert E4.require_e2_complete(ns, {})["complete"]
    src = inspect.getsource(E4.main)
    assert src.index("require_e2_complete(args, specs)") < src.index("X.write_config(") < src.rindex("X.run_jobs(")


# ============================================================================================ E5/E6 (items 6, 7, 26)
def _fake_e5_done(path: Path, job: str, arm: str = "protected") -> None:
    path.write_text(json.dumps(dict(job=job, arm=arm, fp="f", code="c", timings=dict(job_s=1.0), checks=dict(
        h5a_bit_identical=True, measurement_equals_stored_ref=True, stored_log_equals_plan=True,
        attack_ref_truthful=True, n_stored_digests=3))))


@pytest.mark.parametrize("mod,jid,other", [(E5, "2024-03-10__protected__s0", "2024-03-10__protected__s1"),
                                           (E6, "n6__2024-03-10__o00060", "n6__2024-03-10__o10860")],
                         ids=["e5", "e6"])
def test_e5_e6_recompute_archives_the_old_record_first_and_a_newer_error_counts(mod, jid, other, tmproot):
    out = tmproot / "results" / "raw" / "dev" / mod.__name__
    jdir = out / "jobs"
    jdir.mkdir(parents=True)
    _fake_e5_done(jdir / f"{jid}.done.json", jid)
    (jdir / f"{jid}.later.parquet").write_bytes(b"old numbers")
    (jdir / f"{jid}.later.parquet.tmp").write_bytes(b"partial")
    _fake_e5_done(jdir / f"{other}.done.json", other)
    # the recompute fails right after its start (an incomplete spec): the old record must not survive it
    r = mod._run_safe(dict(out=str(out), job=jid, archive_stamp="S1"))
    assert r["status"] == "error"
    a = tmproot / "results" / "raw" / "dev" / "archive" / "S1" / mod.__name__ / "jobs"
    assert (a / f"{jid}.done.json").is_file() and (a / f"{jid}.later.parquet").read_bytes() == b"old numbers"
    assert not (jdir / f"{jid}.done.json").exists() and not (jdir / f"{jid}.later.parquet").exists()
    assert not (jdir / f"{jid}.later.parquet.tmp").exists() and (jdir / f"{jid}.error.txt").is_file()
    assert (jdir / f"{other}.done.json").is_file()                            # other jobs untouched
    chk = mod.aggregate(out)
    assert chk["jobs_error"] == [jid] and chk["jobs_done"] == 1 and chk["jobs_error_after_done"] == []
    # an error.txt newer than an existing done.json (latest attempt failed) is an error, and the job is left out
    err = jdir / f"{other}.error.txt"
    err.write_text("Traceback ...")
    t = (jdir / f"{other}.done.json").stat().st_mtime
    os.utime(err, (t + 5, t + 5))
    chk = mod.aggregate(out)
    assert set(chk["jobs_error"]) == {jid, other} and chk["jobs_error_after_done"] == [other]
    assert chk["jobs_done"] == 0


def test_e5_e6_fingerprint_ignores_the_archive_stamp_and_records_code(tmproot):
    m = tmproot / "model.pt"
    m.write_bytes(b"one")
    job = dict(job="2024-03-02__protected__s0", split="dev", date="2024-03-02", seed=0, arm="protected",
               model=dict(path=str(m)), d=2, layout=[[60, "A"]], profiles=[["none", 1.0]], overrides={}, ft_iters=50,
               lr=3e-4, M=4, sigma0=0.5, init_log_sigma=-0.7, cal="n3_default", out=str(tmproot), test_freeze=None)
    code = E5.code_fingerprint()
    assert E5.job_fingerprint(job, code) == E5.job_fingerprint(dict(job, archive_stamp="X"), code)
    j6 = dict(job="n6__2024-03-02__o00060", split="dev", n=6, date="2024-03-02", offset=60, seeds=[0], arms=["equal"],
              models={}, d=2, br_every=1, overrides={}, out="x", smoke=False, test_freeze=None)
    assert E6.job_fingerprint(j6, "c") == E6.job_fingerprint(dict(j6, archive_stamp="X"), "c")
    for mod in (E5, E6):
        src = inspect.getsource(mod.run_job)
        assert "code=code_fingerprint()" in src
        assert src.index("retire_job_files(") < src.index("import torch")


def test_e5_e6_build_jobs_use_seeds_default(tmproot, monkeypatch):
    monkeypatch.setattr(E5, "SEEDS_DEFAULT", (4, 7))
    a5 = argparse.Namespace(smoke=False, split="dev", dates="2024-03-10", seeds=None, alphas=None, targets=None,
                            arms="protected", ft_iters=50, models_root=str(tmproot / "no_models"), ckpt_template=None,
                            plan=True, d=2, lr=None, M=None, sigma0=None, keep_episodes=False, test_freeze=None)
    assert sorted({j["seed"] for j in E5.build_jobs(a5, tmproot / "o5")}) == [4, 7]
    assert sorted({j["seed"] for j in E5.build_jobs(argparse.Namespace(**{**vars(a5), "seeds": "1"}),
                                                    tmproot / "o5")}) == [1]
    monkeypatch.setattr(E6, "SEEDS_DEFAULT", (5,))
    a6 = argparse.Namespace(smoke=False, split="dev", n="6", seeds=None, arms="equal", dates="2024-03-10",
                            offsets="60", models_root=None, ckpt_template=None, plan=True, d=2, br_every=1,
                            test_freeze=None)
    assert [j["seeds"] for j in E6.build_jobs(a6, tmproot / "o6")] == [[5]]
    for mod in (E5, E6):
        assert '"0,1,2"' not in inspect.getsource(mod.build_jobs)
