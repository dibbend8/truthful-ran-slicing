"""Tests of the training driver ``experiments/train.py`` and ``experiments/registry.py``.

Fast checks of the job list, the hyper-parameter grid (D6 horizon), the split rules (TEST refused before and after
the freeze), filters and the selection rule, plus one end-to-end run on the smoke windows (2 TRAIN windows, 2 PPO
iterations, a 2-cell grid, forked workers): models, selection, copies, resume, staleness and bit-reproducibility.
Outputs go below ``results/raw/_pytest_train_*`` (``slicemech.io`` refuses to write outside the project) and are
removed afterwards.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pytest

sys.dont_write_bytecode = True
_EXP = pathlib.Path(__file__).resolve().parents[1] / "experiments"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import registry as R  # noqa: E402


def _load_train():
    if "exp_train" in sys.modules:
        return sys.modules["exp_train"]
    spec = importlib.util.spec_from_file_location("exp_train", _EXP / "train.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp_train"] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load_train()


@pytest.fixture
def root():
    d = R.RESULTS_DIR / "raw" / f"_pytest_train_{uuid.uuid4().hex[:8]}"
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def no_freeze(monkeypatch, tmp_path):
    """No frozen pre-registration recorded (the TEST download manifest does not exist)."""
    from slicemech.data import puffer_fetch
    monkeypatch.setattr(puffer_fetch, "manifest_path", lambda: tmp_path / "no_manifest.json")


# ------------------------------------------------------------------------------------------------- job list
def test_job_list_matches_the_design():
    ctx = T.make_ctx("dev")
    jobs = T.build_jobs(ctx)
    kinds = {k: [j for j in jobs if j.kind == k] for k in T.KINDS}
    assert list(R.CONFIGS) == ["default", "d1", "d5", "d15", "rho0.5", "rho0.7", "rho1.1", "lead0", "lead1.05",
                               "n6", "n9"]
    c = R.CONFIGS
    assert (c["default"].n, c["default"].d, c["default"].rho, c["default"].lead_s) == (3, 2, 0.9, 4.004)
    assert c["default"].seeds == (0, 1, 2, 3, 4) and all(c[k].seeds == (0, 1, 2) for k in c if k != "default")
    assert [c[k].d for k in ("d1", "d5", "d15")] == [1, 5, 15]
    assert [c[k].rho for k in ("rho0.5", "rho0.7", "rho1.1")] == [0.5, 0.7, 1.1]
    assert [c[k].lead_s for k in ("lead0", "lead1.05")] == [0.0, 1.05]
    assert (c["n6"].n, c["n6"].calibration, c["n9"].n, c["n9"].calibration) == (6, "n6_default", 9, "n9_default")
    # 5 sweep calibrations, one static fit per distinct window set (the d cells share the default's)
    assert sorted(j.calib for j in kinds["calib"]) == sorted(R.CALIB_SPECS) == sorted(
        ["n3_rho0.5", "n3_rho0.7", "n3_rho1.1", "n3_lead0", "n3_lead1.05"])
    assert R.CALIB_SPECS["n3_lead0"] == dict(n=3, rho=0.9, lead_s=0.0)
    assert R.CALIB_SPECS["n3_rho1.1"] == dict(n=3, rho=1.1, lead_s=4.004)
    assert sorted(j.config for j in kinds["static"]) == sorted(["default", "rho0.5", "rho0.7", "rho1.1", "lead0",
                                                                "lead1.05", "n6", "n9"])
    assert R.static_dir("d5") == R.static_dir("default")
    # tuning: 4 arms x 8 cells x 5 seeds on the default config only; default models are copies of the selected cell
    assert len(kinds["tune"]) == 4 * 8 * 5 and {j.config for j in kinds["tune"]} == {"default"}
    assert len(kinds["copy"]) == 4 * 5 and {j.config for j in kinds["copy"]} == {"default"}
    assert len(kinds["main"]) == 4 * 10 * 3 and "default" not in {j.config for j in kinds["main"]}
    assert {j.arm for j in jobs if j.arm} == set(R.ARMS) == {"protected", "unconstrained", "meas_only", "kappa0"}
    assert len({j.id for j in jobs}) == len(jobs)
    # fewer tuning seeds: the other default seeds become main jobs with the selected cell
    ctx2 = T.make_ctx("dev", tune_seeds=(0, 1))
    j2 = T.build_jobs(ctx2)
    assert sum(j.kind == "tune" for j in j2) == 64
    assert sorted({j.seed for j in j2 if j.kind == "main" and j.config == "default"}) == [2, 3, 4]


def test_tuning_grid_and_ppo_config_follow_the_decisions():
    cells = R.tuning_cells()
    assert len(cells) == 8 and len({c.id for c in cells}) == 8
    assert {c.lr for c in cells} == {1e-4, 3e-4} and {c.sigma0 for c in cells} == {0.3, 0.6}
    assert {c.M for c in cells} == {4, 8}
    for c in cells:
        p = R.ppo_config(c)
        assert p.horizon == 3                                # D6: lookahead return, set explicitly
        assert (p.iterations, p.windows_per_iter, p.eval_every, p.clip) == (400, 32, 50, 0.2)
        assert (p.M, p.sigma0, p.lr_start) == (c.M, c.sigma0, c.lr) and p.lr_end == pytest.approx(c.lr / 10)
        s = R.ppo_config(c, smoke=True)
        assert (s.iterations, s.windows_per_iter, s.eval_every, s.horizon) == (2, 2, 1, 3)
        assert R.cell_by_id(c.id) == c


def test_filters():
    f = T.parse_filters(["kind=tune,main", "arm=protected", "seed=0"])
    assert f == {"kind": {"tune", "main"}, "arm": {"protected"}, "seed": {"0"}}
    assert T.matches(T.Job("tune", "default", "protected", 0, "lr3e-04_s0.3_M4"), f)
    assert not T.matches(T.Job("tune", "default", "kappa0", 0, "lr3e-04_s0.3_M4"), f)
    assert not T.matches(T.Job("static", "default"), f)                 # no arm: excluded by an arm filter
    with pytest.raises(ValueError):
        T.parse_filters(["colour=red"])
    with pytest.raises(ValueError):
        T.parse_filters(["kind"])


# ------------------------------------------------------------------------------------------------- split rules
def test_split_test_is_refused_before_the_freeze(no_freeze, capsys):
    with pytest.raises(PermissionError, match="freeze"):
        T.check_split("test", smoke=False)
    with pytest.raises(PermissionError):
        R.require_split("test")
    assert T.main(["--split", "test", "--plan"]) == 2
    assert "refused" in capsys.readouterr().err
    with pytest.raises(PermissionError):
        T.run(T.make_ctx("test"))
    cal = R.load_calibration("n3_default")
    with pytest.raises(PermissionError):                 # no TEST window is composed without the freeze
        R.compose_windows(cal, "test", [("2024-09-01", 60)])


def test_split_test_is_refused_after_the_freeze_too(monkeypatch):
    from slicemech.data import puffer_fetch
    monkeypatch.setattr(puffer_fetch, "require_test_read_gate", lambda test_freeze=None: {})
    with pytest.raises(PermissionError, match="never use TEST"):
        T.check_split("test", smoke=False)


def test_selection_on_train_needs_smoke():
    with pytest.raises(ValueError, match="smoke"):
        T.check_split("train", smoke=False)
    T.check_split("train", smoke=True)
    T.check_split("dev", smoke=False)
    with pytest.raises(ValueError):
        T.check_split("validation", smoke=False)


# ------------------------------------------------------------------------------------------------- selection rule
def _fake_tune(res, arm, cell_id, seed, value):
    job = T.Job("tune", R.DEFAULT, arm, seed, cell_id)
    spec = res.spec(job)
    p = res.marker(job)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(status="done", spec_digest=R.spec_digest(spec), spec=spec,
                                 result=dict(selection_welfare=value))))


def test_selection_rule_mean_over_seeds_and_ties(root, monkeypatch):
    monkeypatch.setattr(R, "TUNING_GRID", {"lr": (1e-4, 3e-4), "sigma0": (0.3,), "M": (4,)})
    ctx = T.make_ctx("dev", tune_seeds=(0, 1), root=root)
    res = T.Resolver(ctx)
    log = T.Log(ctx, "pytest")
    a, b = [c.id for c in R.tuning_cells()]
    assert T.update_selection(res, log) is None                       # nothing tuned yet
    # protected: cell b has the higher mean (seed 1 decides); kappa0: exact tie -> the earlier cell a
    for s, (va, vb) in enumerate([(50.0, 49.0), (50.0, 52.0)]):
        _fake_tune(res, "protected", a, s, va)
        _fake_tune(res, "protected", b, s, vb)
        _fake_tune(res, "kappa0", a, s, 51.0)
        _fake_tune(res, "kappa0", b, s, 51.0)
    _fake_tune(res, "meas_only", a, 0, 60.0)                          # incomplete: no selection for this arm
    sel = T.update_selection(res, log)
    assert sel["arms"]["protected"]["cell"] == b
    assert sel["arms"]["protected"]["mean_selection_welfare"] == pytest.approx(50.5)
    assert sel["arms"]["protected"]["per_cell"][a]["per_seed"] == {"0": 50.0, "1": 50.0}
    assert sel["arms"]["kappa0"]["cell"] == a and sel["arms"]["kappa0"]["margin_over_second"] == 0.0
    assert "meas_only" not in sel["arms"] and sel["split"] == "dev" and sel["tune_seeds"] == [0, 1]
    # main jobs resolve with the arm's cell; arms without a selection stay blocked
    main = T.Job("main", "rho0.5", "protected", 0)
    if res.calib_sha("n3_rho0.5") is not None:
        assert res.spec(main)["cell"]["id"] == b
    m6 = res.spec(T.Job("main", "n6", "protected", 1))
    assert m6["cell"]["id"] == b and m6["ppo"]["M"] == 4 and m6["ppo"]["horizon"] == 3
    assert m6["hyperparameter_source"].startswith("DEV selection")
    assert res.status(T.Job("main", "n6", "meas_only", 0))[0] == "blocked:selection"
    # a changed tuning record makes the arm's selection disappear (no silent reuse of an old cell)
    p = res.marker(T.Job("tune", R.DEFAULT, "kappa0", 1, b))
    rec = json.loads(p.read_text())
    rec["spec_digest"] = "0" * 64
    p.write_text(json.dumps(rec))
    sel = T.update_selection(res, log)
    assert "kappa0" not in sel["arms"] and "protected" in sel["arms"]


def test_simulation_waits_for_tuning_and_calibration():
    ctx = T.make_ctx("dev")
    jobs = [j for j in T.build_jobs(ctx) if j.arm in (None, "protected")]
    t = T.simulate(jobs, ctx, 22)
    tune = [j for j in jobs if j.kind == "tune"]
    longest_main = max(T.est_seconds(j, ctx) for j in jobs if j.kind == "main")
    assert t >= (sum(T.est_seconds(j, ctx) for j in tune) / 22 + longest_main) * T.CONTENTION * 0.99
    assert T.simulate([], ctx, 4) == 0.0


# ------------------------------------------------------------------------------------------------- end to end
def test_end_to_end_smoke_resume_and_reproducibility(root, monkeypatch):
    """2-cell grid, 2 arms, smoke windows (TRAIN only), forked workers; then resume, staleness and a rerun of one job
    in a fresh root, which must give bit-identical parameters."""
    from slicemech.control import nets, ppo
    monkeypatch.setattr(R, "TUNING_GRID", {"lr": (3e-4,), "sigma0": (0.3, 0.6), "M": (4,)})
    ctx = T.make_ctx("train", smoke=True, workers=3, root=root)
    flt = T.parse_filters(["config=default", "arm=protected,meas_only"])
    assert T.run(ctx, flt) == 0
    res = T.Resolver(ctx)
    sel = R.load_selection(root=ctx.models_root)
    assert set(sel["arms"]) == {"protected", "meas_only"} and sel["split"] == "train"
    for arm in ("protected", "meas_only"):
        best = sel["arms"][arm]
        assert best["mean_selection_welfare"] == max(v["mean"] for v in best["per_cell"].values())
        net, rec = R.load_model("default", arm, 0, root=ctx.models_root)
        assert rec["job"] == f"copy/default/{arm}/seed0" and rec["hyperparameters"]["id"] == best["cell"]
        src = R.read_done(R.tune_dir(arm, best["cell"], 0, root=ctx.models_root) / R.CONFIG_FILE)
        assert src["result"]["model_sha256"] == rec["result"]["model_sha256"]
        assert rec["spec"]["kind"] == "copy" and rec["copied_from"]["spec"]["ppo"]["horizon"] == 3
        assert net.forecaster is not None and net.forecaster.horizon == R.CONFIGS["default"].d
        assert net.kind == arm and rec["decisions"].keys() >= {"D1", "D2", "D6"}
        # deployment reproduces the recorded selection welfare exactly (deterministic replay)
        cal = R.load_calibration("n3_default")
        wins = R.compose_windows(cal, "train", R.window_keys("train", smoke=True))
        ev = ppo.evaluate(net, wins, d=2, mech=R.mechanism(cal))
        assert ev["welfare"] == rec["result"]["selection_welfare"]
        hist = (R.model_dir("default", arm, 0, root=ctx.models_root) / R.HISTORY_FILE).read_text().splitlines()
        assert len(hist) == 1 + 2 and hist[0].startswith("iteration,")
    # two deployed models with their config.json records (pre-freeze audit E5-7) + selection.json
    assert len(R.checkpoint_paths(root=ctx.models_root)) == 2 * 2 + 1
    assert all("tune" not in p.parts for p in R.checkpoint_paths(root=ctx.models_root))
    idx = (pathlib.Path(ctx.models_root) / R.INDEX_FILE).read_text().splitlines()
    assert len(idx) == 1 + 4 + 2                                         # 4 tuning runs + 2 copies

    # resume: nothing reruns, nothing is rewritten
    tune_job = T.Job("tune", R.DEFAULT, "protected", 0, R.tuning_cells()[0].id)
    marker = res.marker(tune_job)
    mtime = marker.stat().st_mtime_ns
    assert T.run(ctx, flt) == 0
    assert marker.stat().st_mtime_ns == mtime

    # a record whose spec changed is stale and is not retrained without --redo-stale
    rec = json.loads(marker.read_text())
    good = rec["spec_digest"]
    rec["spec_digest"] = "f" * 64
    marker.write_text(json.dumps(rec))
    assert res.status(tune_job)[0] == "stale"
    sched = T.Scheduler(res, T.Log(ctx, "pytest"), [tune_job], force=False, redo_stale=False)
    assert sched.todo == [] and list(sched.skipped.values()) == [[tune_job.id]]
    assert T.run(ctx, T.parse_filters([f"cell={tune_job.cell}", "arm=protected", "kind=tune"]),
                 redo_stale=True) == 0
    rec2 = json.loads(marker.read_text())
    assert rec2["spec_digest"] == good

    # static fit through the registry loader
    assert T.run(ctx, T.parse_filters(["kind=static", "config=default"]), ) == 0
    fit = R.load_static("d15", root=ctx.models_root)                      # the d cells share the default's fit
    assert fit.w.shape == (3,) and fit.w.sum() == pytest.approx(1.0) and fit.method == "grid"
    st = R.read_done(R.static_dir("default", root=ctx.models_root) / R.STATIC_FILE)
    assert st["used_by_configs"] == ["default", "d1", "d5", "d15"]

    # bit-reproducibility: the same job in a fresh root (in-process this time) gives identical parameters
    root2 = pathlib.Path(str(root) + "_b")
    try:
        ctx2 = T.make_ctx("train", smoke=True, workers=1, root=root2)
        assert T.run(ctx2, T.parse_filters([f"cell={tune_job.cell}", "arm=protected", "kind=tune"])) == 0
        a = R.read_done(T.Resolver(ctx2).marker(tune_job))
        assert a["result"]["param_digest"] == rec2["result"]["param_digest"]
        assert a["spec_digest"] == rec2["spec_digest"]
        n1 = nets.load_net(marker.parent / R.MODEL_FILE)
        n2 = nets.load_net(T.Resolver(ctx2).marker(tune_job).parent / R.MODEL_FILE)
        assert np.array_equal(nets.param_vector(n1), nets.param_vector(n2))
    finally:
        shutil.rmtree(root2, ignore_errors=True)


def test_cli_runs_forked_workers_in_a_fresh_process(root):
    """The CLI in a fresh interpreter (the way full runs start): forked workers train (no fallback), the selection is
    written and the default models are copied; ``--split test`` is refused with exit code 2."""
    code = pathlib.Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(code / "experiments" / "train.py"), "--smoke", "--split", "train", "--workers", "2",
           "--root", str(root), "--only", "config=default", "--only", "arm=kappa0", "--only", "kind=tune,copy"]
    env = dict(__import__("os").environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run(cmd, cwd=code, env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert "on up to 2 worker(s)" in out.stdout and "cannot train in this process" not in out.stdout
    assert "finished: 9/9 selected jobs done" in out.stdout                    # 8 tuning runs + 1 copy
    sel = R.load_selection(root=root / "models")
    assert list(sel["arms"]) == ["kappa0"] and len(sel["arms"]["kappa0"]["per_cell"]) == 8
    net, rec = R.load_model("default", "kappa0", 0, root=root / "models")
    assert rec["hyperparameters"]["id"] == sel["arms"]["kappa0"]["cell"]
    bad = subprocess.run([sys.executable, str(code / "experiments" / "train.py"), "--split", "test", "--plan"],
                         cwd=code, env=env, capture_output=True, text=True, timeout=300)
    if not (R.ROOT / "data" / "puffer_raw" / "manifest.json").exists():   # no freeze recorded yet
        assert bad.returncode == 2 and "refused" in bad.stderr and "freeze" in bad.stderr
