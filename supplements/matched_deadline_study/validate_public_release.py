#!/usr/bin/env python3
"""Validate a restored public research payload without traffic, replay or network.

Writes a new JSON report and adjacent artifacts OUTSIDE the immutable payload.
Raw-evidence hashes are cross-referenced, never presented as raw replay checks.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

MODEL_FILES = {"model.pt", "final_model.pt", "config.json", "history.csv", "initial_rng.json",
               "window_sampling_schedule.npz", "window_inputs.json", "prepared_inputs.json"}
STATIC_FILES = {"static.json", "candidates.npz", "candidates_train.csv", "candidates_selection.csv",
                "selected_weights.json", "train_best_weights.json", "window_inputs.json"}
REDUNDANT_CSV = {n + ".csv" for n in (
    "truth_observations", "truth_seed_averaged_windows", "truth_comparison_windows", "truth_invariants",
    "strategic_observations", "strategic_paired_observations", "strategic_seed_averaged_pairs", "strategic_invariants")}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def relative(name):
    require(isinstance(name, str) and "\\" not in name, "Nonportable manifest path")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == name and name not in ("", "."),
            "Unsafe manifest path: " + str(name))
    return Path(*p.parts)


def relative_directory(name):
    """Historical model directory locators may have one trailing slash."""
    require(isinstance(name, str), "Nonportable directory path")
    return relative(name[:-1] if name.endswith("/") else name)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def external_index(rows):
    index = {}
    for row in rows:
        name = relative(row["path"]).as_posix()
        require(name not in index and valid_hash(row["sha256"]) and isinstance(row["bytes"], int)
                and row["bytes"] >= 0, "Invalid or duplicate external record: " + name)
        index[name] = row
    return index


def inventory(root, manifest):
    require(manifest.get("version") == 1 and manifest.get("scientific_result") is True
            and manifest.get("prepared_release") == "v0.2.0", "Not the supported scientific release manifest")
    require(valid_hash(manifest["source_private_scope_sha256"]) and valid_hash(manifest["source_private_sha256sums_sha256"]),
            "Missing private-source integrity anchors")
    files = manifest["files"]
    require(files and "PUBLIC_RELEASE_MANIFEST.json" not in files, "Invalid manifest inventory")
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Linked paths are forbidden in release payload")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    extras = actual - set(files) - {"PUBLIC_RELEASE_MANIFEST.json"}
    require(extras <= {"RELEASE_VALIDATION.json", "SHA256SUMS"}, "Unlisted payload files: " + str(sorted(extras)))
    require(set(files) <= actual, "Missing manifest members")
    for name, row in files.items():
        path = root / relative(name)
        require(valid_hash(row["sha256"]) and row["provenance"] in
                ("unchanged original", "new release utility", "generated release metadata"), "Invalid file metadata: " + name)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "File hash/size mismatch: " + name)
    if "RELEASE_VALIDATION.json" in extras:
        require("SHA256SUMS" in extras, "Previous validation record is not covered by SHA256SUMS")
        previous = load(root / "RELEASE_VALIDATION.json")
        require(previous.get("complete") is True and previous.get("public_release_manifest_sha256") ==
                sha(root / "PUBLIC_RELEASE_MANIFEST.json"), "Previous validation refers to different release bytes")
    if "SHA256SUMS" in extras:
        sums = {}
        for line in (root / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            relative(name)
            require(valid_hash(expected) and name not in sums, "Malformed SHA256SUMS")
            sums[name] = expected
        require(set(sums) == actual - {"SHA256SUMS"}, "Incomplete SHA256SUMS inventory")
        for name, expected in sums.items():
            require(sha(root / name) == expected, "SHA256SUMS mismatch: " + name)
    return files


def verify_metadata(root, manifest, files):
    raw = external_index(manifest["external_raw_evidence"])
    original = external_index(manifest["external_original_inputs"])
    require(not (set(raw) & set(original)) and not ((set(raw) | set(original)) & set(files)),
            "External inputs overlap included files")
    require(load(root / "RAW_EVIDENCE_REQUIREMENTS.json")["files"] == manifest["external_raw_evidence"],
            "Raw-evidence declarations disagree")
    require(load(root / "EXTERNAL_REQUIREMENTS.json")["files"] == manifest["external_original_inputs"],
            "External-input declarations disagree")
    days, day = [], date(2024, 9, 1)
    while day <= date(2024, 10, 21):
        if day.weekday() in (6, 0):
            days.append(day.isoformat())
        day += timedelta(days=1)
    expected_cases = {f"{day}__{offset}" for day in days for offset in (60, 43260)}
    cells, used_raw, analyses, model_markers = set(), set(), {}, []
    for item in manifest["selected_cells"]:
        key = (item["mode"], item["config"], item["profile"])
        mode, config, profile = key
        require(key not in cells and mode in ("truth", "strategic") and config in ("default", "rho0.7", "rho1.1")
                and profile in ("both", "video300", "interactive50") and (mode != "strategic" or profile == "both")
                and item["cases"] == 32 and item["scientific_result"] is True, "Invalid selected cell")
        cells.add(key)
        folder = root / "matched_deadline_study/evaluation" / mode / config / profile
        markers = sorted(folder.glob("*/COMPLETE.json"))
        require({p.parent.name for p in markers} == expected_cases, "Evaluation case grid is incomplete")
        for marker in markers:
            done = load(marker)
            require(done.get("complete") is True and done.get("scientific_result") is True,
                    "Non-scientific/incomplete evaluation marker")
            spec_path = root / "matched_deadline_study/evaluation_specifications" / (done["spec_sha256"] + ".json")
            spec = load(spec_path)
            require(digest(spec) == done["spec_sha256"] and
                    (spec["mode"], spec["config"], spec["profile"], spec["split"], spec["scientific_result"]) ==
                    (mode, config, profile, "test", True) and marker.parent.name == f"{spec['date']}__{spec['offset']}",
                    "Evaluation specification/marker mismatch")
            require(done["trajectory_count"] == spec["trajectories_per_case"] == (28 if mode == "truth" else 300),
                    "Evaluation trajectory grid changed")
            refs = done["files_sha256"]
            require("result.json" in refs and len(refs) == done["trajectory_count"] + 1,
                    "Incomplete evaluation file references")
            for name, expected in refs.items():
                target = (marker.parent / relative(name)).relative_to(root).as_posix()
                require(target in raw and raw[target]["sha256"] == expected,
                        "External raw-evidence hash differs from original COMPLETE: " + target)
                used_raw.add(target)
            for entry in spec["models"].values():
                if entry["state"] == "reference":
                    require(entry.get("no_fitted_parameters") is True, "Reference has unexpected fitted model")
                    continue
                base = (root / "deadline_load_study/upstream" / relative_directory(entry["asset_relative_dir"])) if entry["state"] == "frozen" else (
                    root / "matched_deadline_study/models" / relative_directory(entry["model_relative_dir"]))
                for name, expected in entry["files_sha256"].items():
                    rel = (base / relative(name)).relative_to(root).as_posix()
                    require(files.get(rel, {}).get("sha256") == expected, "Evaluation references a different model: " + rel)
                if entry["state"] != "frozen":
                    rel = (base / "COMPLETE.json").relative_to(root).as_posix()
                    require(files.get(rel, {}).get("sha256") == entry["completion_sha256"], "Model marker mismatch")
        chosen = None
        for variant in ("all", mode):
            candidate = root / "matched_deadline_study/analysis_matched" / config / profile / variant
            if not (candidate / "findings.json").is_file():
                continue
            f = load(candidate / "findings.json")
            if f["scope"].get("config") == config and f["scope"].get("profile") == profile and mode in f["scope"].get("modes", []):
                chosen = candidate
                break
        require(chosen is not None, "No matching complete analysis")
        f = load(chosen / "findings.json")
        status = [s for s in f["completion"] if s["mode"] == mode]
        require(f.get("scientific_result") is True and len(status) == 1 and status[0]["complete"] is True and
                status[0]["validated_cases"] == status[0]["expected_cases"] == 32, "Incomplete analysis scope")
        require(f["analysis_sha256"] == files["matched_deadline_study/analyze_matched.py"]["sha256"] and
                f["protocol_sha256"] == files["matched_deadline_study/PROTOCOL.txt"]["sha256"], "Analysis source/protocol mismatch")
        omitted = []
        for name, expected in f["table_files_sha256"].items():
            require(len(relative(name).parts) == 1, "Unsafe analysis table reference")
            target = (chosen / name).relative_to(root).as_posix()
            if target in files:
                require(files[target]["sha256"] == expected, "Analysis table hash mismatch: " + target)
            else:
                counterpart = (chosen / Path(name).with_suffix(".parquet")).relative_to(root).as_posix()
                require(name in REDUNDANT_CSV and counterpart in files, "Required analysis table missing: " + target)
                omitted.append(name)
        with (chosen / f"{mode}_cases.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 32 and {f"{r['date']}__{r['offset']}" for r in rows} == expected_cases,
                "Analysis case table has an incomplete grid")
        for row in rows:
            marker = folder / f"{row['date']}__{row['offset']}" / "COMPLETE.json"
            raw_result = (marker.parent / "result.json").relative_to(root).as_posix()
            require(row["completion_sha256"] == sha(marker) and row["result_sha256"] == raw[raw_result]["sha256"]
                    and row["case_spec_sha256"] == load(marker)["spec_sha256"], "Analysis/evaluation case cross-reference mismatch")
        record = analyses.setdefault(chosen.relative_to(root).as_posix(), {"modes": [], "omitted_redundant_csv": omitted})
        record["modes"].append(mode)
    require(cells and used_raw == set(raw), "Raw-evidence declarations include an unselected or unreferenced case")
    actual_markers = {p.relative_to(root).as_posix() for p in (root / "matched_deadline_study/evaluation").rglob("COMPLETE.json")}
    require(len(actual_markers) == 32 * len(cells), "Unexpected evaluation markers outside selected cells")
    for marker in sorted((root / "matched_deadline_study/models").rglob("COMPLETE.json")):
        done = load(marker)
        expected_files = STATIC_FILES if (marker.parent / "static.json").exists() else MODEL_FILES
        require(done.get("complete") is True and done.get("scientific_result") is True and set(done["files_sha256"]) == expected_files,
                "Incorrect model completion/inventory")
        require({p.name for p in marker.parent.iterdir()} == expected_files | {"COMPLETE.json"}, "Extra/missing completed model artifact")
        for name, expected in done["files_sha256"].items():
            rel = (marker.parent / name).relative_to(root).as_posix()
            require(files.get(rel, {}).get("sha256") == expected, "Model COMPLETE hash mismatch: " + rel)
        record = load(marker.parent / ("static.json" if expected_files == STATIC_FILES else "config.json"))
        require(digest(record["spec"]) == record["spec_sha256"] == done["spec_sha256"] and
                record["spec"]["scientific_result"] is True and record["validation"].get("no_test_windows") is True,
                "Model specification or no-TEST fitting validation mismatch")
        model_markers.append(marker.relative_to(root).as_posix())
    return {"selected_cells": [dict(mode=m, config=c, profile=p) for m,c,p in sorted(cells)],
            "evaluation_cases": len(actual_markers), "raw_evidence_hash_cross_references": len(used_raw),
            "raw_replay_verified": False, "model_completion_markers": model_markers,
            "analyses": analyses, "external_original_input_count": len(original)}


# Fresh workers prohibit network access and mutation of the restored payload.
# This is a guard against accidental I/O, not an OS sandbox for hostile code.
WORKER = r'''
import importlib, json, os, pathlib, runpy, sys
root = pathlib.Path(sys.argv[1]).resolve()
job = json.loads(pathlib.Path(sys.argv[2]).read_text())
external = set(job.get('external_paths', []))
def audit(event, args):
    if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.gethostbyname', 'socket.sendto'):
        raise PermissionError('Offline validation forbids network access')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = pathlib.Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(root):
            flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise PermissionError('Release payload is immutable')
            if p.relative_to(root).as_posix() in external:
                raise PermissionError('Raw traffic/replay input reads are forbidden')
    if event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.rename', 'os.chmod'):
        for value in args[:2] if event == 'os.rename' else args[:1]:
            if isinstance(value, (str, bytes, os.PathLike)) and pathlib.Path(os.fsdecode(value)).resolve().is_relative_to(root):
                raise PermissionError('Release payload is immutable')
sys.addaudithook(audit)
sys.dont_write_bytecode = True
if job['kind'] == 'models':
    code = root/'deadline_load_study/upstream/code'
    sys.path[:0] = [str(code), str(code/'experiments'), str(root/'deadline_load_study')]
    modules = {}
    for name in ('slicemech.model.allocation','slicemech.pricing.envelope','slicemech.control.nets','registry'):
        module = importlib.import_module(name)
        location = pathlib.Path(module.__file__).resolve()
        if not location.is_relative_to(code):
            raise ValueError('Module imported outside restored source: '+name)
        modules[name] = location.relative_to(root).as_posix()
    import torch
    torch.set_num_threads(1)
    from study_support import safe_load_net
    import study_support
    if pathlib.Path(study_support.__file__).resolve() != root/'deadline_load_study/study_support.py':
        raise ValueError('Wrong checkpoint loader imported')
    models = []
    for row in job['models']:
        path = root/row['path']
        config = path.parent/'config.json'
        net = safe_load_net(path, expected_file_sha256=row['sha256'], config_path=config if path.name=='model.pt' else None)
        if not all(torch.isfinite(p).all().item() for p in net.parameters()):
            raise ValueError('Nonfinite model parameters')
        record = json.loads(config.read_text())
        if net.spec()['kind'] != record['net_spec']['kind']:
            raise ValueError('Checkpoint/config network-kind mismatch')
        models.append(dict(path=row['path'], sha256=row['sha256'], kind=net.spec()['kind'],
                           param_digest=net.study_checkpoint['param_digest'], weights_only=True))
    from slicemech.data import puffer_fetch
    gate = puffer_fetch.require_test_read_gate(root/'deadline_load_study/upstream/prereg/PREREG.md')
    result = dict(complete=True, imported_modules=modules, safely_loaded_models=models,
                  native_frozen_test_gate=gate, traffic_read=False, simulation_run=False, network_access=False)
    with pathlib.Path(job['output']).open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
else:
    script = pathlib.Path(job['script'])
    sys.path.insert(0,str(script.parent))
    sys.argv = [str(script)] + job['arguments']
    runpy.run_path(str(script),run_name='__main__')
'''


def run_worker(root, directory, job, externals):
    directory.mkdir(parents=True, exist_ok=False)
    job["external_paths"] = sorted(externals)
    job_path = directory / "job.json"
    save_new(job_path, job)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", MPLCONFIGDIR=str(directory / "matplotlib_cache"),
               NUMBA_CACHE_DIR=str(directory / "numba_cache"), XDG_CACHE_HOME=str(directory / "cache"))
    process = subprocess.run([sys.executable, "-I", "-B", "-c", WORKER, str(root), str(job_path)],
                             cwd=directory, env=env, capture_output=True, text=True)
    (directory / "stdout.txt").write_text(process.stdout)
    (directory / "stderr.txt").write_text(process.stderr)
    require(process.returncode == 0, f"Offline worker failed ({job['kind']}): {process.stderr[-2500:]}")


def validate(root, output, artifacts):
    manifest_path = root / "PUBLIC_RELEASE_MANIFEST.json"
    manifest_sha256 = sha(manifest_path)
    manifest = load(manifest_path)
    files = inventory(root, manifest)
    require(sha(__file__) == files["matched_deadline_study/validate_public_release.py"]["sha256"],
            "Validator bytes differ from the release's declared utility")
    metadata = verify_metadata(root, manifest, files)
    external = {r["path"] for key in ("external_original_inputs", "external_raw_evidence") for r in manifest[key]}
    models = [dict(path=name, sha256=row["sha256"]) for name,row in sorted(files.items())
              if Path(name).name in ("model.pt", "final_model.pt")]
    require(models, "Release has no model checkpoints")
    model_output = artifacts / "model_validation.json"
    run_worker(root, artifacts / "models", dict(kind="models", models=models, output=str(model_output)), external)
    reanalyses = []
    adapter = root / "matched_deadline_study/reanalyze_released.py"
    require(adapter.is_file(), "Offline reaggregation adapter is missing")
    for i, (name, row) in enumerate(sorted(metadata["analyses"].items())):
        mode = "all" if set(row["modes"]) == {"truth", "strategic"} else row["modes"][0]
        folder, destination = root / name, artifacts / f"reanalysis_{i:02d}"
        run_worker(root, artifacts / f"reanalysis_worker_{i:02d}", dict(kind="reanalysis", script=str(adapter), arguments=[
            "--release-analysis", str(folder), "--expected-findings-sha256", sha(folder / "findings.json"),
            "--mode", mode, "--output", str(destination)]), external)
        record = load(destination / "DERIVED_ANALYSIS_VERIFICATION.json")
        require(record.get("complete") is True and record.get("raw_replay_verified") is False,
                "Derived-table reaggregation did not complete with the stated limited scope")
        reanalyses.append(dict(analysis=name, mode=mode, verification_report=str(destination / "DERIVED_ANALYSIS_VERIFICATION.json"),
                              sha256=sha(destination / "DERIVED_ANALYSIS_VERIFICATION.json")))
    plots = []
    plot_jobs = [("welfare", c["config"]) for c in metadata["selected_cells"] if c["mode"] == "truth" and c["profile"] == "both"]
    if any(c == dict(mode="strategic", config="default", profile="both") for c in metadata["selected_cells"]):
        plot_jobs.append(("pricing", "default"))
    for which, config in plot_jobs:
        plot_root = artifacts / ("plot_" + which + "_" + config)
        plot_root.mkdir()
        for filename in ("plot_matched.py", "analyze_matched.py"):
            shutil.copyfile(root / "matched_deadline_study" / filename, plot_root / filename)
        run_worker(root, plot_root / "worker", dict(kind="plot", script=str(plot_root / "plot_matched.py"), arguments=[
            "--analysis-root", str(root / "matched_deadline_study/analysis_matched"), "--protocol",
            str(root / "matched_deadline_study/PROTOCOL.txt"), "--which", which, "--config", config]), external)
        figures = []
        for path in sorted((plot_root / "figures").glob("*")):
            if path.is_file():
                figures.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        require(len(figures) == 3, "Expected PDF, PNG and provenance for regenerated plot")
        plots.append(dict(which=which, config=config, artifacts=figures))
    # Recheck bytes and inventory after all subprocesses: nothing may alter the source payload.
    require(sha(manifest_path) == manifest_sha256, "Manifest changed")
    inventory(root, manifest)
    report = dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        validator_sha256=sha(__file__), public_release_manifest_sha256=sha(manifest_path),
        release_root=str(root), prepared_release=manifest["prepared_release"], published=False,
        validation_scope="Offline integrity, restored imports, safe checkpoints, native freeze gate, derived-table reaggregation and plots",
        raw_traffic_read=False, raw_replay_verified=False, simulation_run=False, training_run=False, network_access=False,
        original_scientific_bytes_preserved=True, included_files_verified=len(files),
        metadata=metadata, model_validation=load(model_output), derived_reanalyses=reanalyses, regenerated_plots=plots,
        limitations=["Raw traffic and per-interval replay outputs are external; their original hashes are cross-referenced, not re-read.",
                     "Reaggregation validates retained derived observations; it does not rerun training, replay, or scientific inference from raw traffic.",
                     "The Python audit guard prevents accidental network/file writes; it is not a sandbox for hostile source code."])
    save_new(output, report)
    print(json.dumps(dict(complete=True, output=str(output), models=len(models), analyses=len(reanalyses), plots=len(plots))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="NEW JSON file outside the immutable payload")
    args = parser.parse_args()
    root, output = args.release_root.expanduser().resolve(), args.output.expanduser().resolve()
    require(root.is_dir() and not output.is_relative_to(root), "Output must be outside an existing release payload")
    require(not output.exists(), "Refusing to overwrite validation output")
    artifacts = output.parent / (output.stem + "_artifacts")
    require(not artifacts.exists() and not artifacts.is_relative_to(root), "Refusing to overwrite validation artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    try:
        validate(root, output, artifacts)
    except Exception as exc:
        if not output.exists():
            save_new(output, dict(complete=False, error_type=type(exc).__name__, error=str(exc),
                                 created_utc=datetime.now(timezone.utc).isoformat(), artifacts=str(artifacts)))
        raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, PermissionError) as error:
        raise SystemExit("PUBLIC RELEASE VALIDATION FAILED: " + str(error))
