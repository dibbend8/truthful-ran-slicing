"""Cross-driver agreement (experiment integrator, 2026-10-19): the drivers, the training registry, the analysis and
the figure scripts must name the same files, configurations, arms and summary tables.

Fast (no episode is played): every check reads module constants or calls path/lookup helpers, except the E1
usage-tariff test, which replaces the expensive calls by stand-ins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
EXP = CODE / "experiments"
for _p in (str(EXP), str(EXP / "figures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analyze as A  # noqa: E402
import e0_setup as S  # noqa: E402
import e1_value as E1  # noqa: E402
import e2_manipulation as X  # noqa: E402
import e5_training_channel as E5  # noqa: E402
import e6_scaling as E6  # noqa: E402
import registry as R  # noqa: E402

CONTROLLER_KEYS = {c["key"] for c in yaml.safe_load(open(CODE.parent / "spec" / "controllers.yaml"))}


def test_model_files_resolve_to_the_registry_layout(tmp_path):
    """E0/E1, E2-E4, E5 and E6 look for the deployed models exactly where train.py writes them."""
    for kind in ("protected", "unconstrained", "kappa0"):
        assert X._net_candidates(kind, 3, None)[0] == R.model_dir("default", kind, 3) / R.MODEL_FILE
    assert X._static_candidates(None)[0] == R.static_dir("default") / R.STATIC_FILE
    for kind in ("protected", "unconstrained"):
        e = E5.model_entry(kind, 2, models_root=tmp_path)
        assert Path(e["path"]) == R.model_dir("default", kind, 2, root=tmp_path) / R.MODEL_FILE and e["missing"]
        for n in (6, 9):
            e = E6.model_entry(n, kind, 1, models_root=tmp_path)
            assert Path(e["path"]) == R.model_dir(f"n{n}", kind, 1, root=tmp_path) / R.MODEL_FILE
    # E0/E1 discovery: only finished registry records count, tuning runs never
    d = R.model_dir("rho0.5", "protected", 4, root=tmp_path)
    d.mkdir(parents=True)
    (d / R.MODEL_FILE).write_bytes(b"x")
    assert S.find_models(tmp_path, "rho0.5", "protected") == {}
    (d / R.CONFIG_FILE).write_text('{"status": "done"}')
    assert S.find_models(tmp_path, "rho0.5", "protected") == {4: d / R.MODEL_FILE}
    t = R.tune_dir("protected", R.tuning_cells()[0].id, 0, root=tmp_path)
    t.mkdir(parents=True)
    (t / R.MODEL_FILE).write_bytes(b"x")
    (t / R.CONFIG_FILE).write_text('{"status": "done"}')
    assert S.find_models(tmp_path, "default", "protected") == {}


def test_every_registry_configuration_maps_to_its_analysis_cell():
    for name, cfg in R.CONFIGS.items():
        cell = A.config_cell(name)
        assert (cell["n"], cell["d"], cell["rho"], cell["lead_s"]) == (cfg.n, float(cfg.d), cfg.rho, cfg.lead_s), name
    sweep = {c.d for c in R.CONFIGS.values()} | {R.DEFAULT_D}
    assert sweep == {int(x) for x in A.DELAY_SWEEP}
    assert {c.rho for c in R.CONFIGS.values()} == set(A.LOAD_SWEEP)
    assert {c.lead_s for c in R.CONFIGS.values()} == set(A.LEAD_SWEEP)
    assert {c.n for c in R.CONFIGS.values()} == set(A.N_SWEEP)


def test_every_driver_arm_has_a_controllers_yaml_key():
    """Figures and tables take labels from spec/controllers.yaml; no driver arm may fall back to its raw name."""
    for arm in list(X.ARMS) + list(E1.ARMS) + list(E5.ARMS) + list(E6.ARMS) + list(R.ARMS):
        key = A.map_arm(E1.ALIASES.get(arm, arm))[0]
        if arm == "unconstrained":
            key = "unprotected"
        assert key in CONTROLLER_KEYS, (arm, key)
    for keys in R.ARM_CONTROLLER_KEYS.values():
        assert keys and set(keys) <= CONTROLLER_KEYS
    assert A.map_arm(X.scaled_arm(0.5)) == ("protected", "full", 0.5)


def test_analysis_reads_the_drivers_smoke_and_full_locations():
    for split in ("train", "dev"):
        full = A.sources(split, False)
        assert full["e0"] == S.out_dir("e0", split, "default").parent
        assert full["e1"] == S.out_dir("e1", split, "default").parent
        assert full["e2"] == X.out_root("e2", split, False)
        assert full["e5"] == E5.out_dir(split, False) and full["e6"] == E6.out_dir(split, False)
    smoke = A.sources("train", True)                       # driver smoke runs are TRAIN-only
    assert smoke["e0"] == S.out_dir("e0", "train", "default", smoke=True).parent
    assert smoke["e1"] == S.out_dir("e1", "train", "default", smoke=True).parent
    for e in ("e2", "e3", "e4"):
        assert smoke[e] == X.out_root(e, "train", True)
    assert smoke["e5"] == E5.out_dir("train", True) and smoke["e6"] == E6.out_dir("train", True)


def test_drivers_share_split_settings_with_the_registry():
    assert X.D == E5.D_DEFAULT == R.DEFAULT_D == R.CONFIGS["default"].d
    assert X.HORIZON == E5.HORIZON == S.PPO_HORIZON == R.HORIZON == 3                    # D6
    assert X.CAL_NAME == E5.CAL_NAME == R.CONFIGS["default"].calibration
    assert tuple(S.SMOKE_WINDOWS) == tuple(R.SMOKE_TRAIN_WINDOWS) == tuple(X.SMOKE_WINDOWS)
    for split in ("train", "dev"):
        assert X.split_keys(split) == R.window_keys(split) == S.window_keys(split, False)


def test_e1_usage_tariff_is_e2s_tariff_on_the_default_configuration(tmp_path, monkeypatch):
    """--fit-nu: default delegates to e2_manipulation.usage_tariff (one tariff for E1's meas_only bill and E2-E4);
    other configurations fit on all their TRAIN windows (not the 8 report windows)."""
    from slicemech.data.calibrate import train_window_keys
    calls = {}

    def fake_e2(seed, ckpt, smoke, workers, n_smoke=2):
        calls["e2"] = (seed, str(ckpt), smoke, workers)
        return [1.0, 2.0, 3.0], tmp_path / "e2nu.json"

    def fake_pool(fn, jobs, workers):
        calls["pool"] = (fn, len(jobs), workers, jobs[0])
        for _ in jobs:
            yield [2.0, 4.0, 6.0], [1.0, 2.0, 3.0]

    monkeypatch.setattr(X, "usage_tariff", fake_e2)
    monkeypatch.setattr(X, "usage_tariff_path", lambda seed, ckpt, smoke: tmp_path / "e2nu.json")
    monkeypatch.setattr(R, "SWEEP_TARIFF_DIR", tmp_path / "train_e1")      # pre-freeze audit E5-7/F8 location
    monkeypatch.setattr(S, "run_pool", fake_pool)
    monkeypatch.setattr(E1.sio, "confine", lambda p, **k: Path(p))
    cal = type("Cal", (), {"n": 3})()
    models = {"protected": {0: ("/m/protected/seed0/model.pt", "abc")}}
    nu, src, path = E1.usage_tariff(cal, R.CONFIGS["default"], tmp_path / "d", False, True, models, workers=7)
    assert nu == [1.0, 2.0, 3.0] and calls["e2"] == (0, "/m/protected/seed0/model.pt", False, 7) and "e2" in src
    assert path == tmp_path / "e2nu.json"
    nu, _, path = E1.usage_tariff(cal, R.CONFIGS["rho0.5"], tmp_path / "r", False, True, models, workers=5,
                                  calib_root="/c")
    fn, n_jobs, workers, job0 = calls["pool"]
    assert fn is E1._nu_window_job and n_jobs == len(train_window_keys()) == 180 and workers == 5
    assert job0[:3] == ("rho0.5", "/c", "/m/protected/seed0/model.pt")
    assert np.allclose(nu, [2.0, 2.0, 2.0])
    assert path == R.sweep_tariff_path("rho0.5") == tmp_path / "train_e1" / "rho0.5" / "usage_nu.json"
    assert not (tmp_path / "r").exists()                                     # nothing below the split's folder
    rec = E1.sio.read_json(path)
    assert rec["inputs"]["windows"] == "all TRAIN windows" and len(rec["windows"]) == 180


def test_e0_writes_nothing_into_the_summary_directory():
    """The figure scripts read only analyze.py's summaries; E0 used to drop a DEV file into results/summary/."""
    src = (EXP / "e0_setup.py").read_text()
    assert '"summary" / f"e0_' not in src and "RESULTS / \"summary\"" not in src


def test_make_all_renders_every_paper_figure_script():
    import make_all
    scripts = {p.stem for p in (EXP / "figures").glob("fig_*.py")}
    assert scripts == set(make_all.FIGURES)
