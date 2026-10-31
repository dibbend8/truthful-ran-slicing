#!/usr/bin/env python3
"""Reproduce published matched-study aggregates from released derived observations.

This verifies derived tables and their inventory, NOT raw traffic/replay arrays.
The historical analyzer is hash-checked, imported unchanged, and used directly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ATOL, RTOL = 1e-10, 1e-10
ARRAY_COLUMNS = ("honest_tenants", "honest_on_time_harm_Mbit_per_interval",
                 "honest_welfare_harm_per_interval")
REDUNDANT_CSV = {name+".csv" for name in (
    "truth_observations", "truth_seed_averaged_windows", "truth_comparison_windows",
    "truth_invariants", "strategic_observations", "strategic_paired_observations",
    "strategic_seed_averaged_pairs", "strategic_invariants")}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def decode(value):
    if isinstance(value, str) and value.lstrip().startswith(("[", "{")):
        return json.loads(value)
    return value


def structured_equal(a, b):
    a, b = decode(a), decode(b)
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(
            structured_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
        return isinstance(a, (list, tuple, np.ndarray)) and isinstance(b, (list, tuple, np.ndarray)) and len(a) == len(b) and all(
            structured_equal(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return (a is None or pd.isna(a)) and (b is None or pd.isna(b))
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return bool(np.isclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=True))
    return bool(a == b)


def compare_frames(actual, expected, label):
    require(list(actual.columns) == list(expected.columns), f"{label}: column/order mismatch")
    require(len(actual) == len(expected), f"{label}: row count mismatch")
    largest = 0.0
    for name in actual:
        a, b = actual[name], expected[name]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            exact = (name.startswith("n_") or name.endswith("_instances") or name in
                     ("seed", "tenant", "deviator", "offset", "scored_intervals",
                      "finite_bootstrap_replicates", "trajectories", "paired_rows"))
            require(np.allclose(av, bv, atol=0 if exact else ATOL,
                                rtol=0 if exact else RTOL, equal_nan=True),
                    f"{label}: numerical mismatch in {name}")
            finite = np.isfinite(av) & np.isfinite(bv)
            if finite.any():
                largest = max(largest, float(np.max(np.abs(av[finite]-bv[finite]))))
        else:
            require(all(structured_equal(x, y) for x, y in zip(a, b)),
                    f"{label}: value mismatch in {name}")
    return largest


def exact_grid(frame, columns, expected, label):
    require(not frame.duplicated(columns).any(), f"{label}: duplicate observations")
    require(set(frame[columns].itertuples(index=False, name=None)) == expected,
            f"{label}: missing, extra or changed grid observations")


def common_scope(frame, config, profile, analyzer, label):
    require(not frame.empty, f"{label}: empty input")
    for column, value in (("config", config), ("profile", profile),
                          ("target_load", analyzer.CONFIGS[config])):
        require(frame[column].eq(value).all(), f"{label}: mixed {column}")


def validate_cases(frame, mode, config, profile, analyzer):
    common_scope(frame, config, profile, analyzer, mode+" cases")
    exact_grid(frame, analyzer.WINDOW, {(d, o) for d in analyzer.dates() for o in analyzer.OFFSETS},
               mode+" cases")
    require(frame["mode"].eq(mode).all() and frame.scientific_result.eq(True).all(),
            "case inventory is not exclusively completed scientific TEST cases")
    require(frame.trajectories.eq(len(analyzer.runs(mode))).all(), "wrong trajectory inventory")
    require(frame.paired_rows.eq(0 if mode == "truth" else 432).all(), "wrong paired-row inventory")
    require(np.isfinite(frame.actual_scored_offered_load.to_numpy(float)).all(), "invalid offered load")
    for column in ("case_spec_sha256", "completion_sha256", "result_sha256"):
        require(frame[column].map(lambda x: isinstance(x, str) and re.fullmatch(r"[a-f0-9]{64}", x) is not None).all(),
                f"invalid inventory digest: {column}")
        require(frame[column].is_unique, f"duplicate inventory digest: {column}")


def validate_truth(frame, config, profile, analyzer):
    common_scope(frame, config, profile, analyzer, "truth")
    expected = {(d, o, policy, i) for d in analyzer.dates() for o in analyzer.OFFSETS
                for policy in analyzer.policies("truth") for i in range(3)}
    exact_grid(frame, analyzer.WINDOW+["policy", "tenant"], expected, "truth")
    require(frame["mode"].eq("truth").all() and frame.deviator.eq(-1).all() and
            frame.multiplier.eq(1).all() and frame.scored_intervals.eq(160).all(), "truth scope mismatch")
    for policy, (state, arm, seed) in analyzer.policies("truth").items():
        rows = frame[frame.policy.eq(policy)]
        require(rows.state.eq(state).all() and rows.arm.eq(arm).all() and rows.seed.eq(seed).all(),
                "truth policy/seed identity mismatch")
    require(all(analyzer.TRAFFIC[int(i)] == t for i, t in zip(frame.tenant, frame.traffic)), "truth tenant label mismatch")
    metrics = ["welfare", "on_time_Mbit", "expired_Mbit", "arrivals_Mbit", "late_Mbit",
               "initial_backlog_Mbit", "final_backlog_Mbit", "total_delivery_welfare"]
    require(np.isfinite(frame[metrics].to_numpy(float)).all(), "nonfinite truth observations")
    grouped = frame.groupby(analyzer.WINDOW+["policy"], sort=False)
    for _, g in grouped:
        require(g.total_delivery_welfare.nunique() == 1 and
                np.isclose(g.welfare.sum(), g.total_delivery_welfare.iloc[0], atol=ATOL, rtol=RTOL),
                "truth welfare total differs from tenant sum")


def validate_strategic(frame, config, profile, analyzer):
    common_scope(frame, config, profile, analyzer, "strategic")
    expected = {(d, o, policy, i, m, bill) for d in analyzer.dates() for o in analyzer.OFFSETS
                for policy, (_, arm, _) in analyzer.policies("strategic").items()
                for i in range(3) for m in analyzer.MULTIPLIERS
                for bill in (("primary", "fixed_ablation") if arm == "protected" else ("primary",))}
    exact_grid(frame, analyzer.WINDOW+["policy", "deviator", "multiplier", "bill"], expected, "strategic")
    require(frame.shared_truthful_warmup.eq(True).all() and frame.initial_queue_equal.eq(True).all(),
            "released shared-warmup/initial-queue assertion failed")
    for policy, (state, arm, seed) in analyzer.policies("strategic").items():
        rows = frame[frame.policy.eq(policy)]
        require(rows.state.eq(state).all() and rows.arm.eq(arm).all() and rows.seed.eq(seed).all(),
                "strategic policy/seed identity mismatch")
    for column in ARRAY_COLUMNS:
        frame[column] = frame[column].map(decode)
        require(frame[column].map(lambda x:isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2).all(),
                f"invalid paired vector {column}")
    require(all(analyzer.TRAFFIC[int(i)] == t for i, t in zip(frame.deviator, frame.traffic)), "deviator label mismatch")
    require(all(list(v) == [j for j in range(3) if j != i]
                for i, v in zip(frame.deviator, frame.honest_tenants)), "honest-tenant identity mismatch")
    for column in ARRAY_COLUMNS[1:]:
        require(np.isfinite(np.asarray(frame[column].tolist(), float)).all(), "nonfinite harm vector")
    for kind in ("delivery", "contract"):
        fields = [f"{kind}_gain", f"{kind}_gain_lo", f"{kind}_gain_hi",
                  f"truth_{kind}_utility", f"deviation_{kind}_utility"]
        require(np.isfinite(frame[fields].to_numpy(float)).all(), f"nonfinite {kind} observations")
        require(np.allclose(frame[f"{kind}_gain"], frame[f"deviation_{kind}_utility"]-frame[f"truth_{kind}_utility"],
                            atol=ATOL, rtol=RTOL), f"{kind} paired-gain identity failed")
        require((frame[f"{kind}_gain_lo"] <= frame[f"{kind}_gain"]+ATOL).all() and
                (frame[f"{kind}_gain"] <= frame[f"{kind}_gain_hi"]+ATOL).all(), f"{kind} bracket order failed")


def add_per_seed(tables):
    # Same expansion as the unchanged historical write_report, without its raw
    # verifier or findings that would incorrectly imply a new raw replay audit.
    for name, frame in list(tables.items()):
        if "seed_aggregate_estimates" not in frame:
            continue
        excluded = {"seed_aggregate_estimates", "estimate", "ci_low", "ci_high",
                    "seed_mean_min", "seed_mean_max", "finite_bootstrap_replicates"}
        records = []
        for row in frame.to_dict("records"):
            for seed, value in row["seed_aggregate_estimates"].items():
                records.append({**{k:v for k,v in row.items() if k not in excluded},
                                "seed":int(seed), "estimate":value})
        tables[name+"_per_seed"] = pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-analysis", type=Path, required=True, help="one released CONFIG/PROFILE/MODE directory")
    parser.add_argument("--expected-findings-sha256", required=True, help="trusted findings.json digest from release inventory")
    parser.add_argument("--output", type=Path, required=True, help="new, nonexistent output directory")
    parser.add_argument("--mode", choices=("all", "truth", "strategic"), default="all")
    parser.add_argument("--analyzer", type=Path, default=ROOT/"analyze_matched.py")
    parser.add_argument("--protocol", type=Path, default=ROOT/"PROTOCOL.txt")
    args = parser.parse_args()
    for name in ("release_analysis", "output", "analyzer", "protocol"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    require(not args.output.exists(), "output already exists; never overwrite an analysis")
    require(not args.output.is_relative_to(args.release_analysis), "output must be outside released input directory")
    source = args.release_analysis
    findings_path = source/"findings.json"
    require(re.fullmatch(r"[a-f0-9]{64}", args.expected_findings_sha256) is not None, "invalid expected findings digest")
    require(sha(findings_path) == args.expected_findings_sha256, "original findings hash mismatch")
    findings = json.loads(findings_path.read_text())
    require(findings.get("scientific_result") is True, "diagnostic/smoke findings cannot be reanalyzed")
    require(sha(args.analyzer) == findings["analysis_sha256"], "historical analyzer hash mismatch")
    require(sha(args.protocol) == findings["protocol_sha256"], "protocol hash mismatch")
    spec = importlib.util.spec_from_file_location("released_matched_analyzer", args.analyzer)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    modes = ("truth", "strategic") if args.mode == "all" else (args.mode,)
    scope = findings["scope"]
    config, profile = scope["config"], scope["profile"]
    require(config in analyzer.CONFIGS and profile in analyzer.PROFILES, "unplanned cell")
    require(set(modes) <= set(scope["modes"]), "requested mode absent from original report")
    for mode in modes:
        status = [s for s in findings["completion"] if s["mode"] == mode]
        require(len(status) == 1 and status[0]["config"] == config and status[0]["profile"] == profile
                and status[0]["complete"] is True and status[0]["scientific_result"] is True
                and status[0]["expected_cases"] == status[0]["validated_cases"] == 32
                and status[0]["missing_cases"] == [], "incomplete original cell")
    verified, omitted_formats = {}, {}
    for filename, expected in findings["table_files_sha256"].items():
        require(Path(filename).name == filename and Path(filename).suffix in (".csv", ".parquet"),
                "invalid released table filename")
        if not (source/filename).exists() and filename in REDUNDANT_CSV:
            omitted_formats[filename] = expected
            continue
        require(sha(source/filename) == expected, f"released table hash mismatch: {filename}")
        verified[filename] = expected
    def read(name):
        require(name+".parquet" in verified and
                (name+".csv" in verified or name+".csv" in omitted_formats),
                f"unmanifested required table: {name}")
        frame = pd.read_parquet(source/(name+".parquet"))
        if name+".csv" in verified:
            compare_frames(frame, pd.read_csv(source/(name+".csv")), name+" source CSV/Parquet")
        return frame
    bootstrap = findings["bootstrap"]
    require(bootstrap["seed"] == analyzer.BOOTSTRAP_SEED and bootstrap["replicates"] >= 2000
            and bootstrap["conditional_on_three_fitted_seeds"] is True, "bootstrap specification mismatch")
    boot = analyzer.Bootstrap(bootstrap["replicates"])
    tables, inventories = {}, {}
    for mode in modes:
        cases = read(mode+"_cases")
        validate_cases(cases, mode, config, profile, analyzer)
        inventories[mode] = cases
        if mode == "truth":
            observations = read("truth_observations")
            validate_truth(observations, config, profile, analyzer)
            tables.update(analyzer.truth_tables(observations, boot))
        else:
            observations = read("strategic_paired_observations")
            validate_strategic(observations, config, profile, analyzer)
            tables.update(analyzer.strategic_tables(observations, boot))
    if len(inventories) == 2:
        a, b = (inventories[m].sort_values(analyzer.WINDOW) for m in modes)
        require(np.array_equal(a.actual_scored_offered_load.to_numpy(), b.actual_scored_offered_load.to_numpy()),
                "truth/strategic inventories have different offered loads")
    add_per_seed(tables)
    differences = {}
    for name, frame in tables.items():
        expected = read(name)
        differences[name] = compare_frames(frame, expected, name+" regenerated/source Parquet")
        if name+".csv" in verified:
            differences[name] = max(differences[name], compare_frames(frame, pd.read_csv(source/(name+".csv")),
                                                                    name+" regenerated/source CSV"))
    # Only now create the fresh output. A concurrent creator causes refusal.
    args.output.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    for name, frame in tables.items():
        serial = frame.copy()
        for column in serial:
            if serial[column].map(lambda x:isinstance(x, (list, dict, np.ndarray))).any():
                serial[column] = serial[column].map(lambda x:json.dumps(analyzer.clean(x))
                    if isinstance(x, (list, dict, np.ndarray)) else x)
        serial.to_csv(args.output/(name+".csv"), index=False)
        serial.to_parquet(args.output/(name+".parquet"), index=False)
        for ext in ("csv", "parquet"):
            filename = name+"."+ext
            output_hashes[filename] = sha(args.output/filename)
    record = {
        "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "verification_scope": "reaggregation of released derived observations only",
        "raw_replay_verified": False, "raw_traffic_or_interval_arrays_read": False,
        "models_or_simulation_executed": False,
        "limits": ["Does not independently verify omitted traffic, replay arrays, model outputs, payment calculations or raw provenance.",
                   "Inventory digests are checked for structure and linkage to the anchored findings; omitted raw files are not rehashed.",
                   "Hash agreement establishes identity with the supplied trust anchor, not independent authenticity of that anchor.",
                   "Retains the original exploratory, conditional-on-three-fitted-seeds statistical scope."],
        "config": config, "profile": profile, "modes": modes,
        "findings_sha256": args.expected_findings_sha256,
        "analysis_sha256": sha(args.analyzer), "protocol_sha256": sha(args.protocol),
        "wrapper_sha256": sha(__file__), "verified_input_tables_sha256": verified,
        "omitted_redundant_csv_sha256_from_original_findings": omitted_formats,
        "output_tables_sha256": output_hashes, "table_rows": {n:len(f) for n,f in tables.items()},
        "case_inventory_rows": {n:len(f) for n,f in inventories.items()},
        "comparison_atol": ATOL, "comparison_rtol": RTOL,
        "maximum_absolute_numeric_difference_by_table": differences,
        "bootstrap": bootstrap, "versions": {"python":sys.version, "numpy":np.__version__, "pandas":pd.__version__},
    }
    (args.output/"DERIVED_ANALYSIS_VERIFICATION.json").write_text(
        json.dumps(analyzer.clean(record), indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"complete":True, "derived_aggregates_verified":True, "raw_replay_verified":False,
                      "output":str(args.output), "tables":len(tables),
                      "maximum_absolute_numeric_difference":max(differences.values(), default=0)}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        print(f"RELEASED REANALYSIS REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
