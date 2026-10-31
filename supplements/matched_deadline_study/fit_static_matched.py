#!/usr/bin/env python3
"""Native static-weight refitting on TRAIN, with DEV-only final selection.

Scientific jobs retain the original 232-vector grid and stable TRAIN-top-72
DEV rule. --smoke uses two short, disjoint TRAIN windows only and writes a
separate diagnostic result. No TEST window is composed or evaluated.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

import train_matched as shared

ROOT = Path(__file__).resolve().parent
CONFIGS = shared.CONFIGS
PROFILES = shared.PROFILES
DEADLINES = shared.DEADLINES
OUTPUT_FILES = ("static.json", "candidates.npz", "candidates_train.csv",
                "candidates_selection.csv", "selected_weights.json",
                "train_best_weights.json", "window_inputs.json")
SMOKE_INTERVALS = 12
SMOKE_WARMUP = 4


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT.parent / "deadline_load_study/upstream")
    p.add_argument("--frozen-study", type=Path, default=ROOT.parent / "deadline_load_study")
    p.add_argument("--output", type=Path, default=ROOT)
    p.add_argument("--protocol", type=Path, help="defaults to OUTPUT/PROTOCOL.txt; required for scientific jobs")
    p.add_argument("--job", action="append", help="CONFIG/PROFILE, repeatable")
    p.add_argument("--all", action="store_true", help="all nine static cells, sequentially")
    p.add_argument("--plan", action="store_true", help="print job inventory; no writes or simulation")
    p.add_argument("--prepare-only", action="store_true", help="verify inputs and save specs; no composition or fitting")
    p.add_argument("--smoke", action="store_true", help="one fit window and one selection window, both short TRAIN-only diagnostics")
    return p


def jobs_for(args):
    grid = [(c, p) for c in CONFIGS for p in PROFILES]
    if args.all and args.job:
        raise ValueError("Choose --all or --job")
    if args.job:
        jobs = [tuple(text.split("/")) for text in args.job]
        if any(job not in grid for job in jobs) or len(set(jobs)) != len(jobs):
            raise ValueError("Jobs must be distinct CONFIG/PROFILE pairs within the fixed nine-cell grid")
        return jobs
    if args.all or args.plan or args.prepare_only:
        return grid
    if args.smoke:
        return [("default", "both")]
    raise ValueError("Choose --job CONFIG/PROFILE or --all")


def setup(args):
    api = shared.setup(args)
    import train as native_train
    from slicemech.control import static
    if Path(native_train.__file__).resolve() != args.assets / "code/experiments/train.py":
        raise RuntimeError("Unexpected native training module")
    api.update(native_train=native_train, static=static)
    return api


def sources(args):
    return {**shared.source_inventory(args), "runner/fit_static_matched.py": shared.sha(__file__)}


def build_specs(args, api, jobs):
    R = api["R"]
    train = R.window_keys("train")
    dev = R.window_keys("dev", role="select")
    if len(train) != 180 or len(dev) != 66:
        raise RuntimeError("Original TRAIN/DEV-selection window counts changed")
    keys = {"train": train, "selection": dev}
    if args.smoke:
        keys = {"train": train[:1], "selection": train[1:2]}
    input_inventory = shared.input_inventory(args, api, keys)
    source_inventory = sources(args)
    protocol = (args.protocol or args.output / "PROTOCOL.txt").expanduser().resolve()
    if not args.smoke and not protocol.is_file():
        raise FileNotFoundError(f"Scientific static fitting requires frozen protocol: {protocol}")
    method = api["native_train"]._static_kwargs(3, args.smoke)
    if method != {"method": "grid", "seed": 0, "steps": 20, "floor": 0.05}:
        raise RuntimeError("Native n=3 static-grid specification changed")
    top_k = int(R.STATIC_DEV_CANDIDATES)
    if top_k != 72:
        raise RuntimeError("Native static DEV shortlist changed")
    grid = api["static"].simplex_grid(3, method["steps"], method["floor"])
    if grid.shape != (232, 3):
        raise RuntimeError("Native simplex grid must contain 232 distinct vectors")
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "pyarrow", "torch", "numba", "scipy", "PyYAML")}
    versions["python"] = platform.python_version()
    base = {"version": 1, "mode": "smoke" if args.smoke else "scientific",
            "scientific_result": not args.smoke, "source_sha256": source_inventory,
            "input_inventory": input_inventory, "study_protocol_sha256": shared.sha(protocol) if protocol.is_file() else None,
            "window_keys": keys, "selection_split": "train_diagnostic" if args.smoke else "dev_selection",
            "method": method, "report_source": "theta", "candidate_count": len(grid),
            "grid_sha256": shared.array_hash(api["np"], grid), "dev_top_k": top_k,
            "train_ties": "native grid order; equal weights first",
            "selection_rule": api["native_train"].STATIC_CHOICE_RULE,
            "selection_ties": "better stable TRAIN rank (first selection maximum)",
            "static_has_no_iterative_final_model": "final deployed weights are DEV-selected; TRAIN-best weights are saved separately",
            "dependencies": versions, "device": "cpu", "dtype": "float64",
            "smoke_prefix": {"intervals": SMOKE_INTERVALS, "warmup": SMOKE_WARMUP} if args.smoke else None}
    specs = []
    for config, profile in jobs:
        shape_path = args.frozen_study / "shapes" / config / f"{profile}.json"
        fit = api["DeadlineFit"].from_dict(json.loads(shape_path.read_text()))
        cal_rel = f"data/calibration/{R.CONFIGS[config].calibration}.json"
        if list(fit.contract.delay_ms) != list(DEADLINES[profile]) or fit.contract.n != 3:
            raise ValueError("Unexpected deadline/tenant layout")
        if fit.provenance.get("fit_split") != "train" or fit.provenance.get("training_window_count") != 180:
            raise ValueError("Deadline shape must come from all 180 original TRAIN windows")
        if fit.provenance.get("source_calibration_file_sha256") != input_inventory["files"][cal_rel]:
            raise ValueError("Deadline fit uses a different calibration")
        spec = {**base, "job": {"config": config, "profile": profile, "arm": "static"},
                "calibration_relative_path": cal_rel, "shape_file_sha256": shared.sha(shape_path),
                "deadlines_ms": DEADLINES[profile],
                "shape_role": "TRAIN provenance and contract consistency; static objective uses actual delivery replay, not contract Shape"}
        specs.append((spec, fit))
    return specs


def destination(args, spec):
    j = spec["job"]
    return args.output / ("smoke" if args.smoke else "models") / j["config"] / j["profile"] / "static"


def existing_result(out, spec):
    if not out.exists():
        return None
    marker = out / "COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError(f"Existing output is incomplete; refusing overwrite: {out}")
    done = json.loads(marker.read_text())
    if done.get("complete") is not True or done.get("spec_sha256") != shared.digest(spec):
        raise RuntimeError(f"Completed static result has a different specification: {out}")
    if set(done.get("files_sha256", {})) != set(OUTPUT_FILES):
        raise RuntimeError(f"Unexpected completed static artifact manifest: {out}")
    if {p.name for p in out.iterdir()} != set(OUTPUT_FILES) | {"COMPLETE.json"}:
        raise RuntimeError(f"Unexpected/missing files in completed static result: {out}")
    for name, expected in done["files_sha256"].items():
        if shared.sha(out / name) != expected:
            raise RuntimeError(f"Corrupt static output: {out / name}")
    record = json.loads((out / "static.json").read_text())
    if shared.digest(record["spec"]) != shared.digest(spec):
        raise RuntimeError(f"Stored static specification changed: {out}")
    return {"job": spec["job"], "cached": True, "path": str(out), "spec_sha256": shared.digest(spec)}


def smoke_prefix(window):
    K = SMOKE_INTERVALS
    slots = len(window.arrivals_ms) // window.K
    changed = {name: getattr(window, name)[:K].copy() for name in ("theta", "measured", "rate_true", "rate_hat")}
    changed.update(arrivals_ms=window.arrivals_ms[:K*slots].copy(), warmup=SMOKE_WARMUP)
    if hasattr(window, "meta"):
        changed["meta"] = dict(window.meta)
        if "rate_age_s" in changed["meta"]:
            changed["meta"]["rate_age_s"] = changed["meta"]["rate_age_s"][:K]
    return dataclasses.replace(window, **changed)


def compose(args, api, spec, fit):
    cal = api["Calibration"].from_dict(json.loads((args.assets / spec["calibration_relative_path"]).read_text()))
    data, records = {}, {}
    start = time.perf_counter()
    for role in ("train", "selection"):
        expected_split = "train" if role == "train" or args.smoke else "dev"
        data[role], records[role] = [], []
        for day, offset in spec["window_keys"][role]:
            if api["puffer"].split_of_date(day) != expected_split:
                raise ValueError("A static window is outside its declared TRAIN/DEV split")
            window = api["with_deadlines"](cal.window(expected_split, day, offset), fit)
            if args.smoke:
                window = smoke_prefix(window)
            if window.split != expected_split:
                raise RuntimeError("Composed static window has wrong split")
            data[role].append(window)
            records[role].append({"id": window.id, "date": day, "offset": offset,
                "split": expected_split, "K": window.K, "warmup": window.warmup,
                "arrays": {name: shared.array_hash(api["np"], getattr(window, name)) for name in
                           ("arrivals_ms", "theta", "measured", "rate_true", "rate_hat")}})
    return cal, data, records, time.perf_counter()-start


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(shared.clean(rows))


def run_job(args, api, spec, fit):
    out = destination(args, spec)
    cached = existing_result(out, spec)
    if cached:
        return cached
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_name(out.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Static job lock exists; inspect its process before retrying: {lock}") from None
    with os.fdopen(fd, "w") as handle:
        handle.write(shared.canonical({"pid": os.getpid(), "created_utc": shared.now(), "spec_sha256": shared.digest(spec)}))
    stage = Path(tempfile.mkdtemp(prefix=out.name+".pending-", dir=out.parent))
    try:
        np, ppo, static = api["np"], api["ppo"], api["static"]
        cal, windows, window_records, compose_seconds = compose(args, api, spec, fit)
        mech = api["R"].mechanism(cal)
        started = time.perf_counter()
        fitted = static.fit_static(windows["train"], mech, report_source="theta",
                                   allow_splits=("train",), **spec["method"])
        C, scores = np.asarray(fitted.candidates, float), np.asarray(fitted.scores, float)
        if C.shape != (spec["candidate_count"], 3) or scores.shape != (len(C),) or not np.isfinite(scores).all():
            raise ArithmeticError("Native static candidate result has invalid dimensions/scores")
        if shared.array_hash(np, C) != spec["grid_sha256"]:
            raise ArithmeticError("Native fitter evaluated a different candidate grid")
        sel = ppo.prepare(windows["selection"], mech=mech, report_source="theta")
        roll = dict(mech=mech, env_factory=ppo.fluid_env_factory, quota_fn=ppo.default_quota_fn)
        choice = api["native_train"].static_dev_choice(fitted, sel, roll, spec["dev_top_k"])
        order = np.argsort(-scores, kind="stable")
        shortlist = order[:spec["dev_top_k"]]
        if len(choice["table"]) != len(shortlist):
            raise ArithmeticError("Native DEV shortlist has wrong length")
        for rank, entry in enumerate(choice["table"]):
            if not np.array_equal(np.asarray(entry["w"]), C[shortlist[rank]]):
                raise ArithmeticError("Native DEV shortlist differs from stable TRAIN top-72")
        chosen_rank = int(choice["rank"])
        dev_scores = np.asarray([r["dev_welfare"] for r in choice["table"]])
        if not np.isfinite(dev_scores).all() or chosen_rank != int(np.argmax(dev_scores)):
            raise ArithmeticError("Native static selection/tie rule failed")
        selected = np.asarray(choice["w"], float)
        reevaluated = float(static.rollout_fixed(sel, selected[None], **roll)[0])
        error = abs(reevaluated-float(choice["dev_welfare"]))
        tolerance = 1e-9*max(1.0, abs(reevaluated))
        if error > tolerance:
            raise ArithmeticError("Selected static vector does not reproduce its selection score")
        # Check the actual replay engine on TRAIN only, never on TEST.
        from evaluate import evaluate_window
        first = windows["train"][0]
        prepared_first = ppo.prepare([first], mech=mech, report_source="theta")
        native_first = float(static.rollout_fixed(prepared_first, selected[None], **roll)[0])
        controller = dataclasses.replace(fitted, w=selected.copy())
        replay = evaluate_window(first, controller, "none", fit.shape, mech, d=2, price_tenants=[])
        engine_error = abs(native_first-sum(replay["summary"]["mean_realized_value"]))
        if engine_error > 1e-9*max(1.0, abs(native_first)):
            raise ArithmeticError("Native static replay differs from standalone TRAIN evaluator")
        ranks = np.empty(len(C), int); ranks[order] = np.arange(len(C))
        np.savez_compressed(stage / "candidates.npz", candidates=C, scores=scores,
                            stable_train_order=order, selection_candidate_ids=shortlist,
                            selection_scores=dev_scores)
        write_csv(stage / "candidates_train.csv", [dict(candidate_id=j, w0=C[j,0], w1=C[j,1], w2=C[j,2],
            train_welfare=scores[j], stable_train_rank=int(ranks[j]), selected_for_dev=bool(ranks[j] < len(shortlist))) for j in range(len(C))])
        write_csv(stage / "candidates_selection.csv", [dict(train_rank=rank, candidate_id=int(shortlist[rank]),
            w0=row["w"][0], w1=row["w"][1], w2=row["w"][2], train_welfare=row["train_welfare"],
            selection_welfare=row["dev_welfare"], selected=rank == chosen_rank) for rank, row in enumerate(choice["table"])])
        selected_record = {"w": selected.tolist(), "selected_weights": selected.tolist(),
            "final_weights": selected.tolist(), "selection_rank": chosen_rank,
            "candidate_id": int(shortlist[chosen_rank]), "selection_split": spec["selection_split"],
            "welfare_train": choice["train_welfare"], "welfare_selection": choice["dev_welfare"]}
        shared.save_json(stage / "selected_weights.json", selected_record)
        shared.save_json(stage / "train_best_weights.json", {"w": fitted.w.tolist(),
            "candidate_id": int(order[0]), "welfare_train": fitted.welfare,
            "welfare_selection": choice["dev_welfare_train_best"]})
        shared.save_json(stage / "window_inputs.json", window_records)
        if sources(args) != spec["source_sha256"]:
            raise RuntimeError("Static-fitting source changed during the job")
        protocol = args.protocol or args.output / "PROTOCOL.txt"
        if spec["study_protocol_sha256"] is not None and shared.sha(protocol) != spec["study_protocol_sha256"]:
            raise RuntimeError("Protocol changed during static fitting")
        record = {"status": "done", "created_utc": shared.now(), "spec": spec,
            "spec_sha256": shared.digest(spec), **selected_record, "method": fitted.method,
            "w_train_best": fitted.w.tolist(), "welfare_train_best": fitted.welfare,
            "welfare_selection_train_best": choice["dev_welfare_train_best"],
            "dev_choice": choice["table"], "dev_choice_rank": chosen_rank, "n_candidates": len(C),
            "info": fitted.info, "seconds": fitted.seconds,
            "timing": {"composition_seconds": compose_seconds, "fit_selection_validation_seconds": time.perf_counter()-started},
            "validation": {"no_test_windows": True, "smoke_uses_train_only": bool(args.smoke),
                "grid_matches_native": True, "shortlist_matches_stable_train_top72": True,
                "selection_replay_absolute_error": error, "standalone_train_replay_absolute_error": engine_error,
                "selection_tolerance": tolerance}}
        shared.save_json(stage / "static.json", record)
        shared.save_json(stage / "COMPLETE.json", {"complete": True, "scientific_result": spec["scientific_result"],
            "created_utc": shared.now(), "spec_sha256": shared.digest(spec),
            "files_sha256": {name: shared.sha(stage / name) for name in OUTPUT_FILES}})
        if out.exists():
            raise RuntimeError(f"Destination appeared during fitting: {out}")
        os.rename(stage, out)
        return {"job": spec["job"], "cached": False, "path": str(out),
                "weights": selected.tolist(), "selection_welfare": reevaluated,
                "validation": record["validation"]}
    except BaseException as exc:
        shared.save_json(stage / "FAILED.json", {"created_utc": shared.now(), "error": repr(exc),
                                                "spec_sha256": shared.digest(spec)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    args = parser().parse_args()
    jobs = jobs_for(args)
    if args.plan:
        print(json.dumps({"jobs": jobs, "job_count": len(jobs), "scientific_result": not args.smoke,
                          "native_grid_candidates": 232, "selection_top_k": 72,
                          "writes": False, "composition": False}, indent=2))
        return
    free_path = args.output if args.output.exists() else args.output.parent
    if shutil.disk_usage(free_path).free < 3_000_000_000:
        raise RuntimeError("Require at least 3 GB free at output destination")
    api = setup(args)
    specs = build_specs(args, api, jobs)
    for spec, _ in specs:
        shared.immutable_json(args.output / "static_specifications" / f"{shared.digest(spec)}.json", spec)
    if args.prepare_only:
        print(json.dumps({"prepared": len(specs), "fitting_started": False,
                          "scientific_result": not args.smoke, "jobs": jobs}, indent=2))
        return
    for spec, fit in specs:
        print(json.dumps(run_job(args, api, spec, fit)), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        sys.exit(2)
