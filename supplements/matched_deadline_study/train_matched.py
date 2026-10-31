#!/usr/bin/env python3
"""Isolated matched-deadline training. Never reads TEST windows or writes assets."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
CONFIGS = ("rho0.7", "default", "rho1.1")
PROFILES = ("video300", "interactive50", "both")
ARMS = ("protected", "unconstrained", "meas_only", "kappa0")
SEEDS = (0, 1, 2)
DEADLINES = {"video300": (300, 150, 300), "interactive50": (2002, 50, 300), "both": (300, 50, 300)}
TRAINING_PROTOCOL = {
    "version": 1,
    "purpose": "Matched-deadline training from scratch; fixed previously selected hyperparameters",
    "training_split": "TRAIN only: all 180 original windows",
    "selection_split": "DEV selection half only: 66 windows across 11 dates",
    "excluded_split": "TEST is never composed, evaluated, or used for choices by this runner",
    "excluded_dev": "60 DEV estimation windows are not used for choices",
    "iterations": 400, "windows_per_iteration": 32, "gaussian_samples_per_state": 8,
    "ppo_passes": 4, "minibatch_samples": 2048, "clip": 0.2,
    "lr_start": 0.0003, "lr_end": 0.00003, "sigma0": 0.6,
    "entropy_coefficient": 0.001, "gradient_clip": 1.0,
    "reward_lookahead_intervals": 3, "measurement_delay_intervals": 2,
    "checkpoint_iterations": list(range(50, 401, 50)),
    "checkpoint_rule": "highest truthful DEV-selection delivery welfare; ties retain earlier checkpoint",
    "initialization": "fresh nets.make_net(kind, seed); never a historical checkpoint",
    "reports": "truthful theta during TRAIN and DEV",
    "forecaster": "unchanged original TRAIN-fitted operator forecaster for delay 2",
    "deadline_change": "same changed contract in TRAIN and DEV, including features and fluid expiry",
    "shape_fit": "reuse prior study's TRAIN-only deadline shape; PPO reward does not consume Shape",
    "seed_roles": "same seed for network initialization, window samples, Gaussian actions and minibatch order",
    "settings": {"configs": CONFIGS, "profiles": PROFILES, "arms": ARMS, "seeds": SEEDS},
    "scientific_jobs": 108,
    "limitation": "Local service-deadline experiment, not a 3GPP QoS compliance test",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if dataclasses.is_dataclass(value):
        return clean(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite number in training record")
    return value


def canonical(value):
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != clean(value):
            raise RuntimeError(f"Refusing to replace a different existing record: {path}")
    else:
        save_json(path, value)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT.parent / "deadline_load_study/upstream")
    p.add_argument("--frozen-study", type=Path, default=ROOT.parent / "deadline_load_study")
    p.add_argument("--output", type=Path, default=ROOT, help="separate matched-study root; models go below models/")
    p.add_argument("--protocol", type=Path, help="study protocol; defaults to OUTPUT/PROTOCOL.txt; required for scientific training")
    p.add_argument("--job", action="append", help="CONFIG/PROFILE/ARM/SEED, repeatable; use --all for the complete grid")
    p.add_argument("--all", action="store_true", help="run all 108 scientific jobs sequentially")
    p.add_argument("--stage", choices=("A", "B", "C"), help="A: default/both 12 jobs; B: other loads/both 24; C: isolated profiles72")
    p.add_argument("--plan", action="store_true", help="show job IDs and budgets; no writes or simulation")
    p.add_argument("--prepare-only", action="store_true", help="verify inputs and save specifications; no composition or training")
    p.add_argument("--workers", type=int, default=1, choices=(1,), help="one process only; orchestrate independent --job processes externally")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="2 iterations, first 2 TRAIN/DEV-selection windows, separate smoke/ output")
    mode.add_argument("--benchmark", action="store_true", help="3 iterations and full 32-window batches, separate benchmark/ output")
    p.add_argument("--verbose", action="store_true", help="print native PPO iteration records")
    return p


def jobs_for(args):
    grid = [(c, p, a, s) for c in CONFIGS for p in PROFILES for a in ARMS for s in SEEDS]
    if sum((bool(args.all), bool(args.job), bool(args.stage))) > 1:
        raise ValueError("Choose one of --all, --stage, or --job")
    if args.stage:
        return [j for j in grid if (j[1] == "both" and j[0] == "default") == (args.stage == "A")
                and ((j[1] == "both" and j[0] != "default") == (args.stage == "B"))
                and ((j[1] != "both") == (args.stage == "C"))]
    if args.job:
        jobs = []
        for text in args.job:
            parts = text.split("/")
            if len(parts) != 4:
                raise ValueError("--job must be CONFIG/PROFILE/ARM/SEED")
            job = (*parts[:3], int(parts[3].removeprefix("seed")))
            if job not in grid:
                raise ValueError(f"Job outside fixed 108-job grid: {text}")
            if job in jobs:
                raise ValueError(f"Duplicate job: {text}")
            jobs.append(job)
        return jobs
    if args.plan or args.prepare_only or args.all:
        return grid
    if args.smoke or args.benchmark:
        return [("default", "both", "protected", 0)]
    raise ValueError("Choose an explicit --job or --all; --plan is the safe inventory mode")


def setup(args):
    args.assets = args.assets.expanduser().resolve()
    args.frozen_study = args.frozen_study.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.is_relative_to(args.assets) or args.output.is_relative_to(args.frozen_study):
        raise ValueError("Output must be outside original assets and the frozen study")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["NUMBA_CACHE_DIR"] = str(args.output / "runtime/numba_cache")
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(args.assets / "code"), str(args.assets / "code/experiments"), str(args.frozen_study)]
    import numpy as np
    import torch
    import registry as R
    from slicemech.control import nets, ppo
    from slicemech.data import puffer
    from slicemech.data.calibrate import Calibration
    from study_support import DeadlineFit, safe_load_net, with_deadlines
    torch.set_num_threads(1)
    return locals()


def source_inventory(args):
    paths = {"runner/train_matched.py": Path(__file__).resolve(),
             "frozen-study/study_support.py": args.frozen_study / "study_support.py",
             "frozen-study/evaluate.py": args.frozen_study / "evaluate.py"}
    for base in (args.assets / "code/slicemech", args.assets / "code/experiments", args.assets / "spec"):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix != ".pyc" and not any(x.startswith(".") or x == "__pycache__" for x in path.relative_to(base).parts):
                paths["assets/" + path.relative_to(args.assets).as_posix()] = path
    return {name: sha(path) for name, path in paths.items()}


def input_inventory(args, api, keys):
    """Hash only needed TRAIN/DEV data, and verify against recovered receipts."""
    git_manifest_path = args.frozen_study / "upstream_asset_manifest.json"
    puffer_manifest_path = args.frozen_study / "puffer_archive_recovery_manifest.json"
    git = json.loads(git_manifest_path.read_text())
    archive = json.loads(puffer_manifest_path.read_text())
    git_expected = {row["source_path"].split("ICC/rewrite_2026-10-18/", 1)[1]: row["sha256"] for row in git["files"]}
    prepared_expected = {row["relative_output"].removeprefix("upstream/"): row["sha256"] for row in archive["tables"]}
    needed = {"results/models/selection.json"}
    for config in CONFIGS:
        needed.add(f"data/calibration/{api['R'].CONFIGS[config].calibration}.json")
    for folder in ("tractor", "raca"):
        needed.update(p.relative_to(args.assets).as_posix() for p in (args.assets / "data" / folder).iterdir() if p.is_file() and p.suffix in (".npz", ".json", ".parquet"))
    for day in sorted({day for split in keys.values() for day, _ in split}):
        if api["puffer"].split_of_date(day) not in ("train", "dev"):
            raise ValueError("Refusing any TEST data in training input inventory")
        needed.update((f"data/puffer/{day}.parquet", f"data/puffer/{day}.json"))
    hashes = {}
    for rel in sorted(needed):
        expected = prepared_expected.get(rel, git_expected.get(rel))
        if expected is None:
            raise RuntimeError(f"Missing recovered source hash for {rel}")
        actual = sha(args.assets / rel)
        if actual != expected:
            raise RuntimeError(f"Recovered asset changed: {rel}")
        hashes[rel] = actual
    return {"files": hashes, "repository": git["repository"], "revision": git["revision"],
            "recovery_manifest_sha256": sha(git_manifest_path),
            "puffer_archive_manifest_sha256": sha(puffer_manifest_path)}


def mode_for(args):
    return "smoke" if args.smoke else "benchmark" if args.benchmark else "scientific"


def build_specs(args, api, jobs):
    R = api["R"]
    keys = {"train": R.window_keys("train"), "dev_selection": R.window_keys("dev", role="select")}
    if (len(keys["train"]), len(keys["dev_selection"])) != (180, 66):
        raise RuntimeError("Original TRAIN/DEV-selection window counts changed")
    if args.smoke:
        keys = {name: value[:2] for name, value in keys.items()}
    sources = source_inventory(args)
    inputs = input_inventory(args, api, keys)
    protocol = (args.protocol or args.output / "PROTOCOL.txt").expanduser().resolve()
    scientific = mode_for(args) == "scientific"
    if scientific and not protocol.is_file():
        raise FileNotFoundError(f"Scientific preparation/training requires the frozen study protocol: {protocol}")
    protocol_sha = sha(protocol) if protocol.is_file() else None
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "torch", "numba", "scipy", "PyYAML")}
    versions["python"] = platform.python_version()
    base = {"training_protocol": TRAINING_PROTOCOL, "mode": mode_for(args), "scientific_result": scientific,
            "source_sha256": sources, "input_inventory": inputs, "study_protocol_sha256": protocol_sha,
            "window_keys": keys, "dependencies": versions, "device": "cpu", "dtype": "float64"}
    out = []
    for config, profile, arm, seed in jobs:
        cell = R.selected_cell(arm, root=args.assets / "results/models")
        if cell.as_dict() != {"lr": 0.0003, "sigma0": 0.6, "M": 8}:
            raise RuntimeError(f"Historical selected cell unexpectedly differs for {arm}")
        cfg = R.ppo_config(cell)
        if args.smoke:
            cfg = dataclasses.replace(cfg, iterations=2, windows_per_iter=2, eval_every=1)
        elif args.benchmark:
            cfg = dataclasses.replace(cfg, iterations=3, eval_every=3)
        if cfg.horizon != 3:
            raise RuntimeError("Reward lookahead must remain explicitly three intervals")
        shape_path = args.frozen_study / "shapes" / config / f"{profile}.json"
        fit = api["DeadlineFit"].from_dict(json.loads(shape_path.read_text()))
        cal_rel = f"data/calibration/{R.CONFIGS[config].calibration}.json"
        if list(fit.contract.delay_ms) != list(DEADLINES[profile]):
            raise ValueError(f"Unexpected deadline profile: {shape_path}")
        if fit.provenance.get("fit_split") != "train" or fit.provenance.get("training_window_count") != 180:
            raise ValueError("Deadline shape must have been fitted on all180 TRAIN windows")
        if fit.provenance.get("source_calibration_file_sha256") != inputs["files"][cal_rel]:
            raise ValueError("Deadline shape uses a different source calibration")
        job = {"config": config, "profile": profile, "arm": arm, "seed": seed}
        spec = {**base, "job": job, "ppo": dataclasses.asdict(cfg), "hyperparameters": cell.as_dict(),
                "calibration_relative_path": cal_rel, "shape_file_sha256": sha(shape_path),
                "deadlines_ms": DEADLINES[profile], "forecaster_delay": 2,
                "benchmark_note": "Diagnostic runs compress the learning-rate schedule and never produce scientific results" if not scientific else None}
        out.append((spec, cfg, fit))
    return out


def destination(args, spec):
    j = spec["job"]
    category = "models" if spec["mode"] == "scientific" else spec["mode"]
    return args.output / category / j["config"] / j["profile"] / j["arm"] / f"seed{j['seed']}"


def existing_result(path, spec):
    if not path.exists():
        return None
    marker = path / "COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError(f"Existing output is incomplete; refusing overwrite: {path}")
    done = json.loads(marker.read_text())
    if done.get("spec_sha256") != digest(spec) or not done.get("complete"):
        raise RuntimeError(f"Existing result has a different specification: {path}")
    for name, expected in done["files_sha256"].items():
        if sha(path / name) != expected:
            raise RuntimeError(f"Existing completed result is corrupt: {path / name}")
    config = json.loads((path / "config.json").read_text())
    if digest(config["spec"]) != digest(spec):
        raise RuntimeError(f"Existing configuration has changed: {path}")
    return {"job": spec["job"], "cached": True, "path": str(path), "spec_sha256": digest(spec)}


def compose_bucket(args, api, spec, fit):
    cal = api["Calibration"].from_dict(json.loads((args.assets / spec["calibration_relative_path"]).read_text()))
    windows = {}
    records = {}
    start = time.perf_counter()
    for role, split in (("train", "train"), ("dev_selection", "dev")):
        windows[role] = []
        records[role] = []
        for index, (day, offset) in enumerate(spec["window_keys"][role], 1):
            if api["puffer"].split_of_date(day) != split:
                raise ValueError("Window key is outside its declared TRAIN/DEV split")
            original = cal.window(split, day, offset)
            window = api["with_deadlines"](original, fit)
            if window.split != split:
                raise ValueError("Composed window has an unexpected split")
            windows[role].append(window)
            arrays = {name: array_hash(api["np"], getattr(window, name)) for name in
                      ("arrivals_ms", "theta", "measured", "rate_true", "rate_hat")}
            records[role].append({"id": window.id, "date": day, "offset": offset,
                                  "K": window.K, "warmup": window.warmup, "arrays": arrays})
            if index % 30 == 0:
                print(f"Composed {role}: {index}/{len(spec['window_keys'][role])}", flush=True)
    return cal, windows, time.perf_counter() - start, records


def array_hash(np, array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(canonical({"shape": a.shape, "dtype": a.dtype.str}).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def prepare_audit(api, windows, net, cal, mech, cfg):
    """Hash deterministically rebuilt inputs, without modifying native PPO."""
    records = {}
    for role, data in windows.items():
        prepared = api["ppo"].prepare(data, net=net, d=2, mech=mech, report_source="theta",
                                       forecaster=cal.operator_forecaster(2), horizon=cfg.horizon)
        rows = []
        for p in prepared:
            arrays = {name: array_hash(api["np"], value) for name, value in {
                "z_alloc": p.z_alloc, "r_hat": p.r_hat, "c_tenant": p.feats.c_tenant,
                "c_global": p.feats.c_global, "forecast": p.feats.f,
                **({"zn": p.zn} if p.zn is not None else {}),
            }.items()}
            rows.append({"id": p.window.id, "horizon": p.horizon, "arrays": arrays})
        records[role] = rows
    return records


def run_job(args, api, spec, cfg, fit, bucket):
    out = destination(args, spec)
    cached = existing_result(out, spec)
    if cached:
        return cached
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_name(out.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Job lock exists; inspect its process before retrying: {lock}") from None
    with os.fdopen(fd, "w") as handle:
        handle.write(canonical({"pid": os.getpid(), "created_utc": now(), "spec_sha256": digest(spec)}))
    stage = Path(tempfile.mkdtemp(prefix=out.name + ".pending-", dir=out.parent))
    try:
        cal, windows, compose_seconds, window_records = bucket
        mech = api["R"].mechanism(cal)
        j = spec["job"]
        net = api["nets"].make_net(api["R"].ARM_NET_KIND[j["arm"]], seed=j["seed"])
        initial_digest = api["nets"].param_digest(net)
        save_json(stage / "window_inputs.json", window_records)
        save_json(stage / "prepared_inputs.json", prepare_audit(api, windows, net, cal, mech, cfg))
        rng = api["np"].random.default_rng(j["seed"])
        generator = api["torch"].Generator().manual_seed(j["seed"])
        save_json(stage / "initial_rng.json", {"seed": j["seed"], "numpy_window_generator": rng.bit_generator.state,
                  "torch_action_and_minibatch_generator": generator.get_state().tolist(),
                  "network_initial_param_digest": initial_digest,
                  "note": "Independent generators reproduced from native PPO initial seeds; no RNG state injected or changed"})
        schedule = api["np"].stack([api["np"].sort(rng.choice(len(windows["train"]),
                    size=min(cfg.windows_per_iter, len(windows["train"])), replace=False)) for _ in range(cfg.iterations)])
        api["np"].savez_compressed(stage / "window_sampling_schedule.npz", train_window_indices=schedule)
        start = time.perf_counter()
        result = api["ppo"].train(net, windows["train"], windows["dev_selection"], d=2, mech=mech,
                                  config=cfg, seed=j["seed"], forecaster=cal.operator_forecaster(2),
                                  report_source="theta", dev_report_source="theta", verbose=args.verbose,
                                  allow_splits=("train",))
        train_wall_seconds = time.perf_counter() - start
        selected = api["ppo"].evaluate(result.net, windows["dev_selection"], d=2, mech=mech)
        expected = result.best_dev_welfare
        error = abs(selected["welfare"] - expected)
        tolerance = 1e-9 * max(1.0, abs(expected))
        if error > tolerance:
            raise ArithmeticError("Selected checkpoint disagrees with independent DEV re-evaluation")
        api["nets"].save_net(result.net, stage / "model.pt")
        api["nets"].save_net(result.final_net, stage / "final_model.pt")
        reloaded = api["safe_load_net"](stage / "model.pt", expected_param_digest=api["nets"].param_digest(result.net))
        loaded_eval = api["ppo"].evaluate(reloaded, windows["dev_selection"], d=2, mech=mech)
        reload_error = abs(loaded_eval["welfare"] - selected["welfare"])
        if reload_error > tolerance:
            raise ArithmeticError("Serialized checkpoint disagrees with selected checkpoint")
        from evaluate import evaluate_window
        first = windows["train"][0]
        native_first = api["ppo"].evaluate(reloaded, [first], d=2, mech=mech)["welfare"]
        standalone = evaluate_window(first, reloaded, "none", fit.shape, mech, d=2, price_tenants=[])
        standalone_welfare = sum(standalone["summary"]["mean_realized_value"])
        engine_error = abs(native_first - standalone_welfare)
        if engine_error > 1e-9 * max(1.0, abs(native_first)):
            raise ArithmeticError("Native training replay disagrees with standalone TRAIN evaluator")
        fields = list(dict.fromkeys(k for row in result.history for k in row))
        with (stage / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(clean(result.history))
        if source_inventory(args) != spec["source_sha256"]:
            raise RuntimeError("Training source changed while job was running")
        protocol = args.protocol or args.output / "PROTOCOL.txt"
        if spec["study_protocol_sha256"] is not None and sha(protocol) != spec["study_protocol_sha256"]:
            raise RuntimeError("Study protocol changed while job was running")
        record = {"status": "done", "created_utc": now(), "spec": spec, "spec_sha256": digest(spec),
                  "initial_param_digest": initial_digest, "net_spec": result.net.spec(),
                  "result": {"best_iteration": result.best_iteration, "best_dev_welfare": expected,
                             "selection_welfare": selected["welfare"], "selection_per_window": selected["per_window"],
                             "log_sigma": result.log_sigma, "sigma": math.exp(result.log_sigma),
                             "iterations": len(result.history), "n_projected": result.n_projected,
                             "dev_curve": {h["iteration"]: h["dev_welfare"] for h in result.history if "dev_welfare" in h},
                             "param_digest": api["nets"].param_digest(result.net), "model_sha256": sha(stage / "model.pt"),
                             "history_sha256": sha(stage / "history.csv"), "train_seconds": sum(h["seconds"] for h in result.history),
                             "train_wall_seconds": train_wall_seconds, "compose_seconds": compose_seconds},
                  "validation": {"dev_replay_absolute_error": error, "reload_dev_absolute_error": reload_error,
                                 "standalone_train_welfare_absolute_error": engine_error,
                                 "tolerance": tolerance, "no_test_windows": True,
                                 "loaded_param_digest_matches": True}}
        save_json(stage / "config.json", record)
        save_json(stage / "COMPLETE.json", {"complete": True, "spec_sha256": digest(spec),
                   "scientific_result": spec["scientific_result"], "created_utc": now(),
                   "files_sha256": {n: sha(stage / n) for n in ("model.pt", "final_model.pt", "config.json", "history.csv",
                       "initial_rng.json", "window_sampling_schedule.npz", "window_inputs.json", "prepared_inputs.json")}})
        if out.exists():
            raise RuntimeError(f"Destination appeared during training; refusing replacement: {out}")
        os.rename(stage, out)
        return {"job": j, "cached": False, "path": str(out), "spec_sha256": digest(spec),
                "best_iteration": result.best_iteration, "train_wall_seconds": train_wall_seconds,
                "validation": record["validation"]}
    except BaseException as exc:
        save_json(stage / "FAILED.json", {"created_utc": now(), "error": repr(exc), "spec_sha256": digest(spec)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    args = parser().parse_args()
    jobs = jobs_for(args)
    if args.plan:
        print(json.dumps({"training_protocol": TRAINING_PROTOCOL, "mode": mode_for(args), "jobs": jobs,
                          "job_count": len(jobs), "writes": False, "composition": False}, indent=2))
        return
    if shutil.disk_usage(args.output if args.output.exists() else args.output.parent).free < 3_000_000_000:
        raise RuntimeError("Require at least 3 GB free at output destination")
    api = setup(args)
    specs = build_specs(args, api, jobs)
    for spec, _, _ in specs:
        immutable_json(args.output / "specifications" / f"{digest(spec)}.json", spec)
    if args.prepare_only:
        print(json.dumps({"prepared": len(specs), "mode": mode_for(args), "training_started": False,
                          "job_ids": [s[0]["job"] for s in specs]}, indent=2))
        return
    bucket_key, bucket = None, None
    for spec, cfg, fit in specs:
        cached = existing_result(destination(args, spec), spec)
        if cached:
            print(json.dumps(cached), flush=True)
            continue
        key = (spec["job"]["config"], spec["job"]["profile"])
        if bucket_key != key:
            bucket = None
            bucket = compose_bucket(args, api, spec, fit)
            bucket_key = key
        print(json.dumps(run_job(args, api, spec, cfg, fit, bucket)), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        sys.exit(2)
