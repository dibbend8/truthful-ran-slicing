#!/usr/bin/env python3
"""Matched-deadline replay: verified inputs, atomic cases, no model selection."""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
from datetime import date, timedelta
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

import train_matched as shared

ROOT = Path(__file__).resolve().parent
MULTIPLIERS = (0.5, 0.67, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0)
OFFSETS = (60, 43260)
TRAFFIC = ("video", "interactive", "background")
MODEL_FILES = {"model.pt", "final_model.pt", "config.json", "history.csv", "initial_rng.json",
               "window_sampling_schedule.npz", "window_inputs.json", "prepared_inputs.json"}
STATIC_FILES = {"static.json", "candidates.npz", "candidates_train.csv", "candidates_selection.csv",
                "selected_weights.json", "train_best_weights.json", "window_inputs.json"}
_ARGS = None
_API = None
_CONTROLLERS = {}


def original_dates():
    day, end, values = date(2024, 9, 1), date(2024, 10, 21), []
    while day <= end:
        if day.weekday() in (6, 0):
            values.append(day.isoformat())
        day += timedelta(days=1)
    return values


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=ROOT.parent / "deadline_load_study/upstream")
    p.add_argument("--frozen-study", type=Path, default=ROOT.parent / "deadline_load_study")
    p.add_argument("--output", type=Path, default=ROOT)
    p.add_argument("--model-root", type=Path, help="defaults to OUTPUT/models; diagnostics use OUTPUT/smoke")
    p.add_argument("--protocol", type=Path, help="defaults to OUTPUT/PROTOCOL.txt")
    p.add_argument("--mode", choices=("truth", "strategic"), required=True)
    p.add_argument("--config", choices=shared.CONFIGS, action="append", help="repeatable; default default")
    p.add_argument("--profile", choices=shared.PROFILES, action="append", help="repeatable; default both")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--plan", action="store_true", help="read-only grid/trajectory inventory, no imports or TEST reads")
    p.add_argument("--prepare-only", action="store_true", help="verify files/gates and save plans; no window composition or replay")
    p.add_argument("--smoke", action="store_true", help="TRAIN-only interface test: first48 intervals, warmup40, reduced model grid")
    return p


def settings(args):
    configs, profiles = args.config or ["default"], args.profile or ["both"]
    if len(set(configs)) != len(configs) or len(set(profiles)) != len(profiles):
        raise ValueError("Duplicate load/deadline settings are not allowed")
    if not 1 <= args.workers <= 16:
        raise ValueError("--workers must be between1 and16")
    if args.smoke and (configs != ["default"] or profiles != ["both"]):
        raise ValueError("Smoke is fixed to default/both; it is not a scientific setting sweep")
    return [(c, p) for c in configs for p in profiles]


def labels(mode, smoke):
    seeds = (0,) if smoke else shared.SEEDS
    arms = shared.ARMS if mode == "truth" else ("protected", "unconstrained")
    out = [(state, arm, seed) for state in ("frozen", "retrained") for arm in arms for seed in seeds
           if not (smoke and state == "retrained" and arm != "protected")]
    if mode == "truth":
        out += [("frozen", "static", -1), ("retrained", "static", -1), ("reference", "equal", -1), ("reference", "direct", -1)]
    return out


def model_key(state, arm, seed):
    return f"{state}__{arm}__s{seed}"


def source_inventory(args):
    return {**shared.source_inventory(args), "runner/evaluate_matched.py": shared.sha(__file__),
            "runner/fit_static_matched.py": shared.sha(ROOT / "fit_static_matched.py")}


def validate_completion(folder, record_name, expected_files, scientific, protocol_sha, job, shape_sha, current_sources):
    marker = folder / "COMPLETE.json"
    done = json.loads(marker.read_text())
    if done.get("complete") is not True or done.get("scientific_result") is not scientific:
        raise ValueError(f"Wrong completion/scientific status: {marker}")
    if set(done.get("files_sha256", {})) != expected_files:
        raise ValueError(f"Unexpected completed artifact set: {folder}")
    if {f.name for f in folder.iterdir()} != expected_files | {"COMPLETE.json"}:
        raise ValueError(f"Incomplete or extra files in completed model folder: {folder}")
    for name, expected in done["files_sha256"].items():
        if shared.sha(folder / name) != expected:
            raise ValueError(f"Completed artifact hash mismatch: {folder / name}")
    record = json.loads((folder / record_name).read_text())
    spec = record["spec"]
    if shared.digest(spec) != done["spec_sha256"] or record.get("spec_sha256") != done["spec_sha256"]:
        raise ValueError(f"Completion/specification mismatch: {folder}")
    if spec["job"] != job or spec["shape_file_sha256"] != shape_sha:
        raise ValueError(f"Model is from a different job or deadline shape: {folder}")
    if scientific and (spec.get("study_protocol_sha256") != protocol_sha or not spec.get("scientific_result")):
        raise ValueError(f"Model uses a different scientific protocol: {folder}")
    for name, expected in spec["source_sha256"].items():
        if current_sources.get(name) != expected:
            raise ValueError(f"Training/static source changed since fitting: {name}")
    validation = record.get("validation", {})
    if validation.get("no_test_windows") is not True:
        raise ValueError(f"Missing TRAIN/DEV-only fit validation: {folder}")
    return record, {"completion_sha256": shared.sha(marker), "spec_sha256": done["spec_sha256"],
                    "files_sha256": done["files_sha256"]}


def recovered_hashes(args):
    git_path = args.frozen_study / "upstream_asset_manifest.json"
    par_path = args.frozen_study / "puffer_archive_recovery_manifest.json"
    git = json.loads(git_path.read_text())
    par = json.loads(par_path.read_text())
    expected = {r["source_path"].split("ICC/rewrite_2026-10-18/", 1)[1]: r["sha256"] for r in git["files"]}
    expected.update({r["relative_output"].removeprefix("upstream/"): r["sha256"] for r in par["tables"]})
    return expected, {"repository": git["repository"], "revision": git["revision"],
                      "github_manifest_sha256": shared.sha(git_path), "puffer_manifest_sha256": shared.sha(par_path)}


def verified_file(args, relative, expected):
    actual = shared.sha(args.assets / relative)
    if expected.get(relative) != actual:
        raise ValueError(f"Recovered input differs from recorded source hash: {relative}")
    return actual


def inventory_models(args, api, config, profile, shape_sha, protocol_sha, sources, expected):
    inventory = {}
    for state, arm, seed in labels(args.mode, args.smoke):
        key = model_key(state, arm, seed)
        if state == "reference":
            inventory[key] = {"state": state, "arm": arm, "seed": seed, "no_fitted_parameters": True}
            continue
        if state == "frozen":
            rel = f"results/models/{config}/{arm}/" + ("" if arm == "static" else f"seed{seed}/")
            names = ("static.json", "candidates.npz") if arm == "static" else ("config.json", "model.pt")
            files = {name: verified_file(args, rel + name, expected) for name in names}
            record = json.loads((args.assets / rel / ("static.json" if arm == "static" else "config.json")).read_text())
            inventory[key] = {"state": state, "arm": arm, "seed": seed, "asset_relative_dir": rel,
                              "files_sha256": files, "param_digest": None if arm == "static" else record["result"]["param_digest"]}
            continue
        rel = f"{config}/{profile}/{arm}" + ("" if arm == "static" else f"/seed{seed}")
        folder = args.model_root / rel
        job = {"config": config, "profile": profile, "arm": arm, **({"seed": seed} if arm != "static" else {})}
        record, provenance = validate_completion(folder, "static.json" if arm == "static" else "config.json",
            STATIC_FILES if arm == "static" else MODEL_FILES, not args.smoke, protocol_sha, job, shape_sha, sources)
        spec = record["spec"]
        if not args.smoke:
            if arm == "static":
                if spec["candidate_count"] != 232 or spec["dev_top_k"] != 72 or spec["selection_split"] != "dev_selection":
                    raise ValueError("Refitted static search/selection budget mismatch")
            elif (spec["ppo"]["iterations"], spec["ppo"]["windows_per_iter"], spec["ppo"]["M"], spec["ppo"]["horizon"]) != (400, 32, 8, 3):
                raise ValueError("Retrained model is not from the complete matched budget")
        inventory[key] = {"state": state, "arm": arm, "seed": seed, "model_relative_dir": rel, **provenance,
                          "param_digest": None if arm == "static" else record["result"]["param_digest"]}
    return inventory


def build_specs(args, api, selected):
    from slicemech.data import puffer_fetch
    protocol = (args.protocol or args.output / "PROTOCOL.txt").expanduser().resolve()
    if not protocol.is_file():
        raise FileNotFoundError(f"Frozen matched-study protocol missing: {protocol}")
    args.protocol = protocol
    protocol_sha = shared.sha(protocol)
    sources = source_inventory(args)
    expected, origin = recovered_hashes(args)
    if args.smoke:
        split, keys, gate = "train", [api["R"].window_keys("train")[0]], None
    else:
        gate = puffer_fetch.require_test_read_gate(args.assets / "prereg/PREREG.md")
        gate = {**gate, "path": "prereg/PREREG.md"}
        dates = list(api["puffer"].effective_test_dates())
        if len(dates) != 16 or len(set(dates)) != 16:
            raise ValueError("This protocol requires exactly the16 fixed TEST dates")
        split, keys = "test", [(d, o) for d in dates for o in OFFSETS]
    files = {}
    for day in sorted({d for d, _ in keys}):
        for suffix in ("parquet", "json"):
            rel = f"data/puffer/{day}.{suffix}"
            files[rel] = verified_file(args, rel, expected)
    for folder in ("tractor", "raca"):
        for path in sorted((args.assets / "data" / folder).iterdir()):
            if path.is_file() and path.suffix in (".json", ".npz", ".parquet"):
                rel = path.relative_to(args.assets).as_posix()
                files[rel] = verified_file(args, rel, expected)
    if not args.smoke:
        for rel in ("prereg/PREREG.md", "prereg/DEVIATIONS.md", "data/puffer_raw/manifest.json"):
            files[rel] = shared.sha(args.assets / rel)
    versions = {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow", "torch", "numba", "scipy", "PyYAML")}
    versions["python"] = platform.python_version()
    specifications = []
    for config, profile in selected:
        cal_rel = f"data/calibration/{api['R'].CONFIGS[config].calibration}.json"
        cal_sha = verified_file(args, cal_rel, expected)
        shape_path = args.frozen_study / "shapes" / config / f"{profile}.json"
        fit = api["DeadlineFit"].from_dict(json.loads(shape_path.read_text()))
        shape_sha = shared.sha(shape_path)
        if (list(fit.contract.delay_ms) != list(shared.DEADLINES[profile]) or
                fit.provenance.get("fit_split") != "train" or fit.provenance.get("training_window_count") != 180 or
                fit.provenance.get("source_calibration_file_sha256") != cal_sha):
            raise ValueError("Deadline shape/calibration provenance mismatch")
        models = inventory_models(args, api, config, profile, shape_sha, protocol_sha, sources, expected)
        scientific = not args.smoke
        base = {"version": 1, "mode": args.mode, "scientific_result": scientific, "split": split,
                "config": config, "profile": profile, "protocol_sha256": protocol_sha,
                "source_sha256": sources, "input_sha256": {**files, cal_rel: cal_sha}, "origin": origin,
                "shape_sha256": shape_sha, "calibration_relative_path": cal_rel,
                "models": models, "window_keys": keys, "native_test_gate": gate,
                "dependencies": versions, "measurement_delay_intervals": 2,
                "multipliers": MULTIPLIERS if args.mode == "strategic" else (),
                "warmup_report_rule": "truthful before interval40; deviation from40 onward",
                "pricing": "Protected full plus same-path fixed; Unconstrained fixed; truthful all3, deviating tenant only",
                "missing_payments": "NaN arrays/null summaries, never interpreted as zero utility",
                "smoke_prefix_intervals": 48 if args.smoke else None,
                "smoke_grid": "four frozen arms seed0, retrained Protected seed0, four deterministic references for truth; three strategic policies seed0" if args.smoke else None,
                "trajectories_per_case": len(models) * (25 if args.mode == "strategic" else 1)}
        for day, offset in keys:
            specifications.append({**base, "date": day, "offset": offset})
    return specifications


def case_path(args, spec):
    category = "smoke_evaluation" if args.smoke else "evaluation"
    return args.output / category / spec["mode"] / spec["config"] / spec["profile"] / f"{spec['date']}__{spec['offset']}"


def existing_case(out, spec):
    if not out.exists():
        return None
    done = json.loads((out / "COMPLETE.json").read_text())
    if done.get("complete") is not True or done.get("spec_sha256") != shared.digest(spec):
        raise ValueError(f"Refusing different/incomplete existing case: {out}")
    actual_names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} - {"COMPLETE.json"}
    if actual_names != set(done.get("files_sha256", {})):
        raise ValueError(f"Completed case file inventory mismatch: {out}")
    for name, expected in done["files_sha256"].items():
        if shared.sha(out / name) != expected:
            raise ValueError(f"Completed case file hash mismatch: {out / name}")
    return {"case": out.name, "config": spec["config"], "profile": spec["profile"], "cached": True,
            "path": str(out), "spec_sha256": shared.digest(spec)}


def worker_init(args):
    global _ARGS, _API
    _ARGS = args
    _API = shared.setup(args)
    from slicemech.sim import episode
    from slicemech.control.static import StaticFit
    from evaluate import evaluate_window, _prefix_window
    _API.update(episode=episode, StaticFit=StaticFit, evaluate_window=evaluate_window, prefix_window=_prefix_window)


def controller(spec, entry):
    key = shared.digest(entry)
    if key in _CONTROLLERS:
        return _CONTROLLERS[key]
    api, args = _API, _ARGS
    state, arm = entry["state"], entry["arm"]
    if state == "reference":
        result = api["episode"].FixedWeights() if arm == "equal" else api["episode"].DirectController("hat")
    else:
        folder = args.assets / entry["asset_relative_dir"] if state == "frozen" else args.model_root / entry["model_relative_dir"]
        for name, expected in entry["files_sha256"].items():
            if shared.sha(folder / name) != expected:
                raise ValueError(f"Model artifact changed after preparation: {folder / name}")
        if arm == "static":
            r = json.loads((folder / "static.json").read_text())
            with api["np"].load(folder / "candidates.npz", allow_pickle=False) as candidates:
                result = api["StaticFit"](api["np"].asarray(r["w"], float), float(r["welfare_train"]), r["method"],
                    candidates["candidates"].copy(), candidates["scores"].copy(), float(r["seconds"]), dict(r["info"]))
        else:
            result = api["safe_load_net"](folder / "model.pt", expected_file_sha256=entry["files_sha256"]["model.pt"],
                                          expected_param_digest=entry["param_digest"])
            if result.spec()["kind"] != arm:
                raise ValueError("Checkpoint architecture differs from its declared arm")
    _CONTROLLERS[key] = result
    return result


def require_prices(result, selected, warmup, fixed):
    np = _API["np"]
    arrays = result["arrays"]
    for prefix in ("price", "fixed_price") if fixed else ("price",):
        paid, low, high = [arrays[prefix + suffix][warmup:, selected] for suffix in ("", "_lo", "_hi")]
        if not all(np.isfinite(a).all() for a in (paid, low, high)):
            raise ArithmeticError("Required scored payments or certified brackets are missing")
        if np.any(low > paid + 1e-10) or np.any(paid > high + 1e-10):
            raise ArithmeticError("Payment lies outside its certified bracket")
    if result["checks"].get("pricing_quota_matches") is not True:
        raise ArithmeticError("Missing price/allocation quota equality check")


def summary_rows(spec, key, entry, result, deviator, multiplier):
    s, rows = result["summary"], []
    for i, traffic in enumerate(TRAFFIC):
        row = {"mode": spec["mode"], "config": spec["config"], "profile": spec["profile"],
               "date": spec["date"], "offset": spec["offset"], "policy": key, "state": entry["state"],
               "arm": entry["arm"], "seed": entry["seed"], "deviator": deviator, "multiplier": multiplier,
               "tenant": i, "traffic": traffic, "scored_intervals": s["scored_intervals"],
               "total_delivery_welfare": s["delivery_welfare"]}
        mapping = {"welfare": "mean_realized_value", "contract_value": "mean_contract_value", "price": "mean_price",
                   "delivery_utility": "mean_delivery_utility", "contract_utility": "mean_contract_utility",
                   "fixed_price": "mean_fixed_price", "fixed_delivery_utility": "mean_fixed_delivery_utility",
                   "fixed_contract_utility": "mean_fixed_contract_utility", "on_time_Mbit": "on_time_Mbit",
                   "expired_Mbit": "dropped_Mbit", "arrivals_Mbit": "arrivals_Mbit", "late_Mbit": "late_Mbit",
                   "initial_backlog_Mbit": "initial_backlog_Mbit", "final_backlog_Mbit": "final_backlog_Mbit",
                   "resolved_byte_expiry_share": "resolved_byte_expiry_share"}
        row.update({name: s[source][i] for name, source in mapping.items()})
        rows.append(row)
    return rows


def paired_gains(spec, key, entry, truth, deviation, tenant, multiplier, warmup):
    np = _API["np"]
    ta, da, s = truth["arrays"], deviation["arrays"], slice(warmup, None)
    for name in ("quota", "on_time_Mbit", "dropped_Mbit", "backlog_Mbit", "report"):
        if not np.array_equal(ta[name][:warmup], da[name][:warmup]):
            raise ArithmeticError(f"Deviation does not share truthful warm-up {name}")
    if not np.array_equal(ta["backlog_start_Mbit"][warmup], da["backlog_start_Mbit"][warmup]):
        raise ArithmeticError("Truth/deviation scored initial queues differ")
    rows = []
    prefixes = ("price", "fixed_price") if entry["arm"] == "protected" else ("price",)
    for prefix in prefixes:
        row = {"policy": key, "state": entry["state"], "arm": entry["arm"], "seed": entry["seed"],
               "deviator": tenant, "multiplier": multiplier, "bill": "fixed_ablation" if prefix == "fixed_price" else "primary",
               "shared_truthful_warmup": True, "initial_queue_equal": True}
        for name, field in (("delivery", "realized_value"), ("contract", "V_declared")):
            tv, dv = ta[field][s, tenant], da[field][s, tenant]
            tp, dp = ta[prefix][s, tenant], da[prefix][s, tenant]
            low = (dv - da[prefix + "_hi"][s, tenant]) - (tv - ta[prefix + "_lo"][s, tenant])
            high = (dv - da[prefix + "_lo"][s, tenant]) - (tv - ta[prefix + "_hi"][s, tenant])
            gain = (dv - dp) - (tv - tp)
            row.update({f"{name}_gain": float(gain.mean()), f"{name}_gain_lo": float(low.mean()),
                        f"{name}_gain_hi": float(high.mean()), f"truth_{name}_utility": float((tv - tp).mean()),
                        f"deviation_{name}_utility": float((dv - dp).mean()),
                        f"{name}_gain_max_interval": float(gain.max())})
            if name == "contract" and entry["arm"] == "protected" and prefix == "price":
                tolerance = 1e-8 * max(1.0, float(np.max(np.abs(tv))), float(np.max(np.abs(dv))))
                row["full_contract_max_certified_lower_gain"] = float(low.max())
                row["full_contract_numeric_tolerance"] = tolerance
                if np.any(low > tolerance):
                    raise ArithmeticError("Full Protected price has a positive contract gain beyond certified uncertainty")
        honest = [i for i in range(3) if i != tenant]
        row["honest_on_time_harm_Mbit_per_interval"] = (ta["on_time_Mbit"][s][:, honest] - da["on_time_Mbit"][s][:, honest]).mean(axis=0).tolist()
        row["honest_welfare_harm_per_interval"] = (ta["realized_value"][s][:, honest] - da["realized_value"][s][:, honest]).mean(axis=0).tolist()
        row["honest_tenants"] = honest
        rows.append(row)
    return rows


def run_case(spec):
    args, api, np = _ARGS, _API, _API["np"]
    out = case_path(args, spec)
    cached = existing_case(out, spec)
    if cached:
        return cached
    if source_inventory(args) != spec["source_sha256"] or shared.sha(args.protocol) != spec["protocol_sha256"]:
        raise ValueError("Evaluation source or protocol changed after preparation")
    # Recheck inputs that can be consumed by this case, without reading another date's data.
    for name, expected in spec["input_sha256"].items():
        if name.startswith("data/puffer/") and spec["date"] not in name:
            continue
        if shared.sha(args.assets / name) != expected:
            raise ValueError(f"Input changed after preparation: {name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_name(out.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Case lock exists; inspect its process before retrying: {lock}") from None
    with os.fdopen(fd, "w") as handle:
        handle.write(shared.canonical({"pid": os.getpid(), "created_utc": shared.now(), "spec_sha256": shared.digest(spec)}))
    stage = Path(tempfile.mkdtemp(prefix=out.name + ".pending-", dir=out.parent))
    start = time.perf_counter()
    try:
        cal = api["Calibration"].from_dict(json.loads((args.assets / spec["calibration_relative_path"]).read_text()))
        shape_path = args.frozen_study / "shapes" / spec["config"] / f"{spec['profile']}.json"
        if shared.sha(shape_path) != spec["shape_sha256"]:
            raise ValueError("Deadline shape changed after preparation")
        fit = api["DeadlineFit"].from_dict(json.loads(shape_path.read_text()))
        extra = {} if args.smoke else {"test_freeze": args.assets / "prereg/PREREG.md"}
        window = api["with_deadlines"](cal.window(spec["split"], spec["date"], spec["offset"], **extra), fit)
        if args.smoke:
            window = api["prefix_window"](window, 48)
        if window.split != spec["split"] or window.warmup != 40 or window.K != (48 if args.smoke else 200):
            raise ValueError("Unexpected replay split, length or warm-up")
        mech = api["R"].mechanism(cal)
        input_arrays = {n: shared.array_hash(np, getattr(window, n)) for n in
                        ("arrivals_ms", "theta", "measured", "rate_true", "rate_hat")}
        actual_load = float(window.measured[window.warmup:].sum(axis=1).mean() / (mech.B * cal.r_bar))
        window_record = {"id": window.id, "split": window.split, "K": window.K, "warmup": window.warmup,
                         "arrays_sha256": input_arrays, "contract_sha256": shared.digest(dataclasses.asdict(window.contract)),
                         "target_load": cal.rho, "actual_scored_offered_load": actual_load,
                         "load_denominator": "B times original TRAIN mean per-PRB rate",
                         "rbar_train": cal.r_bar, "B": mech.B, "counts": window.meta.get("counts"),
                         "units": "values/prices/utilities:credits per scored interval; traffic:scored-window Mbit"}
        rows, records, gains = [], [], []
        (stage / "intervals").mkdir()

        def replay(key, entry, deviator=-1, multiplier=1.0):
            mult = np.ones(3)
            if deviator >= 0:
                mult[deviator] = multiplier
            strategic = spec["mode"] == "strategic"
            selected = ([] if not strategic else list(range(3)) if deviator < 0 else [deviator])
            rule = "full" if entry["arm"] == "protected" else "fixed" if entry["arm"] == "unconstrained" else "none"
            fixed = strategic and entry["arm"] == "protected"
            result = api["evaluate_window"](window, controller(spec, entry), rule, fit.shape, mech, d=2,
                multipliers=mult, price_tenants=selected, price_start_interval=window.warmup,
                deviation_start_interval=window.warmup, calculate_fixed=fixed)
            if strategic:
                require_prices(result, selected, window.warmup, fixed)
            name = f"{key}__i{deviator}__m{multiplier:g}"
            np.savez_compressed(stage / "intervals" / f"{name}.npz", **result["arrays"])
            rows.extend(summary_rows(spec, key, entry, result, deviator, multiplier))
            records.append({"id": name, "policy": key, "deviator": deviator, "multiplier": multiplier,
                            "checks": result["checks"], "metadata": result["metadata"], "timing_s": result["timing_s"]})
            return result

        for key, entry in spec["models"].items():
            truth = replay(key, entry)
            if spec["mode"] == "strategic":
                for tenant in range(3):
                    for multiplier in MULTIPLIERS:
                        deviation = replay(key, entry, tenant, multiplier)
                        gains.extend(paired_gains(spec, key, entry, truth, deviation, tenant, multiplier, window.warmup))
            print(json.dumps({"case_progress": out.name, "policy": key, "trajectories": len(records),
                              "total": spec["trajectories_per_case"]}), flush=True)
        if len(records) != spec["trajectories_per_case"] or len(rows) != 3 * len(records):
            raise ArithmeticError("Incomplete case trajectory grid")
        if source_inventory(args) != spec["source_sha256"] or shared.sha(args.protocol) != spec["protocol_sha256"]:
            raise ValueError("Evaluation source or protocol changed while the case was running")
        shared.save_json(stage / "result.json", {"complete": True, "spec": spec, "spec_sha256": shared.digest(spec),
            "window": window_record, "rows": rows, "paired_gains": gains, "trajectories": records,
            "seconds": time.perf_counter() - start, "created_utc": shared.now()})
        file_hashes = {p.relative_to(stage).as_posix(): shared.sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        shared.save_json(stage / "COMPLETE.json", {"complete": True, "scientific_result": spec["scientific_result"],
            "spec_sha256": shared.digest(spec), "trajectory_count": len(records), "files_sha256": file_hashes})
        if out.exists():
            raise RuntimeError("Case destination appeared while replay was running")
        os.rename(stage, out)
        return {"case": out.name, "config": spec["config"], "profile": spec["profile"], "cached": False,
                "path": str(out), "trajectories": len(records), "seconds": time.perf_counter() - start}
    except BaseException as exc:
        shared.save_json(stage / "FAILED.json", {"error": repr(exc), "spec_sha256": shared.digest(spec), "created_utc": shared.now()})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    args = parser().parse_args()
    selected = settings(args)
    if args.plan:
        n_cases = len(selected) * (1 if args.smoke else 32)
        per_case = len(labels(args.mode, args.smoke)) * (25 if args.mode == "strategic" else 1)
        print(json.dumps({"mode": args.mode, "settings": selected, "cases": n_cases,
            "trajectories_per_case": per_case, "total_trajectories": n_cases * per_case,
            "dates": ["first original TRAIN date"] if args.smoke else original_dates(),
            "offsets": [60] if args.smoke else OFFSETS, "multipliers": MULTIPLIERS if args.mode == "strategic" else [],
            "scientific_result": not args.smoke, "writes": False, "replay_started": False}, indent=2))
        return
    api = shared.setup(args)
    args.model_root = (args.model_root or args.output / ("smoke" if args.smoke else "models")).expanduser().resolve()
    if shutil.disk_usage(args.output if args.output.exists() else args.output.parent).free < 3_000_000_000:
        raise RuntimeError("Require at least3 GB free before evaluation")
    specs = build_specs(args, api, selected)
    for spec in specs:
        shared.immutable_json(args.output / "evaluation_specifications" / f"{shared.digest(spec)}.json", spec)
    manifest = {"mode": args.mode, "scientific_result": not args.smoke, "workers": args.workers,
                "case_spec_sha256": [shared.digest(s) for s in specs], "complete_grid": selected,
                "case_count": len(specs), "trajectories": sum(s["trajectories_per_case"] for s in specs)}
    run_id = shared.digest(manifest)
    shared.immutable_json(args.output / "evaluation_runs" / f"{run_id}.json", manifest)
    if args.prepare_only:
        print(json.dumps({"prepared_cases": len(specs), "manifest_sha256": run_id, "replay_started": False}))
        return
    errors, results = [], []
    if args.workers == 1:
        worker_init(args)
        for spec in specs:
            try:
                result = run_case(spec)
                results.append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                errors.append({"case": [spec["config"], spec["profile"], spec["date"], spec["offset"]], "error": repr(exc)})
                break
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                                     initializer=worker_init, initargs=(args,)) as pool:
            futures = {pool.submit(run_case, spec): spec for spec in specs}
            for future in concurrent.futures.as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(json.dumps(result), flush=True)
                except Exception as exc:
                    errors.append({"case": [spec["config"], spec["profile"], spec["date"], spec["offset"]], "error": repr(exc)})
                    print(json.dumps({"error": errors[-1]}), flush=True)
    completion = {"complete": not errors and len(results) == len(specs), "run_manifest_sha256": run_id,
                  "completed_cases": len(results), "expected_cases": len(specs), "errors": errors, "created_utc": shared.now()}
    # Attempt records are unique; historical failed/completed attempts are retained.
    record = args.output / "evaluation_runs" / f"{run_id}.attempt-{time.time_ns()}.json"
    shared.save_json(record, completion)
    if errors:
        raise RuntimeError(f"Evaluation stopped with{len(errors)} failed case(s); inspect {record}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, PermissionError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        sys.exit(2)
