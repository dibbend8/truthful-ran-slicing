#!/usr/bin/env python3
"""Offline reanalysis of a verified portable five-seed evidence package.

Scientific mode requires all 160 cases and an external manifest SHA-256 anchor.
TRAIN diagnostics are a separate mode and never produce scientific estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relative(name):
    p = Path(name)
    require(name not in ("", ".") and not p.is_absolute() and ".." not in p.parts
            and "\\" not in name and "//" not in name, "Unsafe package path")
    return p


def module(path):
    spec = importlib.util.spec_from_file_location("released_original_analyzer", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return m


def compare_frames(actual, expected, label):
    require(list(actual.columns) == list(expected.columns) and actual.shape == expected.shape,
            "Table shape/columns differ: " + label)
    maximum = 0.
    for name in actual.columns:
        a, b = actual[name], expected[name]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            require(np.array_equal(np.isnan(av), np.isnan(bv)), "Missing-value mismatch: " + label + "/" + name)
            require(np.allclose(av, bv, rtol=1e-11, atol=1e-10, equal_nan=True),
                    "Numeric mismatch: " + label + "/" + name)
            finite = np.isfinite(av) & np.isfinite(bv)
            if finite.any(): maximum = max(maximum, float(np.max(np.abs(av[finite] - bv[finite]))))
        else:
            require(a.fillna("__missing__").astype(str).tolist() == b.fillna("__missing__").astype(str).tolist(),
                    "Text mismatch: " + label + "/" + name)
    return maximum


def reconstruct_tables(analyzer, pairs, events):
    require(len(pairs) == 480 and len(events) == 76800, "Incomplete derived observation count")
    expected = {(date, offset, seed, role) for date, offset in analyzer.KEYS
                for seed in analyzer.SEEDS for role in range(3)}
    columns = ["date", "offset", "seed", "deviator"]
    require(not pairs.duplicated(columns).any()
            and set(pairs[columns].itertuples(index=False, name=None)) == expected,
            "Incomplete/duplicate seed-window-role observation grid")
    require(not events.duplicated(columns + ["k"]).any(), "Duplicate search-decision observation")
    require(np.array_equal(np.sort(events.k.unique()), np.arange(40, 200)), "Invalid scored interval indices")
    grouped = events.groupby(columns).k.agg(["count", "min", "max", "nunique"])
    require(set(grouped.index.tolist()) == expected and (grouped["count"] == 160).all()
            and (grouped["nunique"] == 160).all() and (grouped["min"] == 40).all()
            and (grouped["max"] == 199).all(), "Incomplete scored search-decision grid")
    require(np.isfinite(events.br_gain).all() and np.isfinite(events.br_U_truth).all(), "Missing search outcomes")
    values = [c for c in pairs.select_dtypes(include="number").columns if c not in ("seed", "offset", "deviator")]
    require(np.isfinite(pairs[values].to_numpy()).all(), "Nonfinite paired quantities")
    per_seed = pairs.groupby(["date", "offset", "seed"], as_index=False)[values].mean()
    windows = per_seed.groupby(["date", "offset"], as_index=False)[values].mean()
    summary = pd.DataFrame(analyzer.summarize(windows, "all"))
    by_seed = pd.DataFrame([r for seed, w in per_seed.groupby("seed") for r in analyzer.summarize(w, int(seed))])
    for i, r in summary.iterrows():
        b = by_seed[by_seed.metric == r.metric]
        summary.loc[i, "training_seed_relative_min"] = b.relative_percent.min()
        summary.loc[i, "training_seed_relative_max"] = b.relative_percent.max()
        summary.loc[i, "training_seed_relative_sd"] = b.relative_percent.std(ddof=1)
    return {"headline_summary": summary, "per_seed_summary": by_seed, "case_role_pairs": pairs,
            "window_seed_pairs": per_seed, "seed_averaged_windows": windows,
            "local_best_response_events": events}


def inspect_package(root, expected_hash):
    path = root / "PUBLIC_PACKAGE_MANIFEST.json"
    require(sha(path) == expected_hash, "Public manifest trust anchor differs")
    manifest = read(path)
    require(manifest["schema"] == "original-five-seed-portable-v1"
            and manifest["complete"] is True and manifest["scientific_result"] is True
            and manifest["exploratory"] is True, "Not a complete scientific evidence package")
    require(manifest["scope"] == {"cases": 160, "trajectories": 800, "training_seeds": 5,
            "dates": 16, "windows": 32, "deviator_episodes": 480, "scored_search_decisions": 76800},
            "Incorrect package scope")
    inventory = manifest["files"]
    require(len({r["path"] for r in inventory}) == len(inventory), "Duplicate inventory paths")
    expected = set()
    for r in inventory:
        rel = relative(r["path"]); p = root / rel; expected.add(rel.as_posix())
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == r["bytes"] and sha(p) == r["sha256"],
                "Included file changed: " + r["path"])
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == expected | {"PUBLIC_PACKAGE_MANIFEST.json"}, "Unexpected/missing package files")
    require(not any(p.is_symlink() for p in root.rglob("*")), "Package contains symlinks")
    portable = read(root / "SPECIFICATION_PORTABLE.json")
    spec = portable["specification"]
    require(portable["original_canonical_sha256"] == manifest["original_specification_canonical_sha256"],
            "Original specification anchor differs")
    analyzer = module(root / "source/analyze_original_multiseed.py")
    require(spec["seeds"] == list(analyzer.SEEDS) and spec["window_keys"] == [list(x) for x in analyzer.KEYS]
            and spec["K"] == 200 and spec["warmup"] == 40
            and spec["episode_profiles"] == list(analyzer.PROFILES), "Portable specification grid differs")
    require(sha(root / "source/PROTOCOL.txt") == spec["protocol_sha256"], "Protocol hash differs")
    findings = read(root / "analysis/findings.json")
    require(findings["complete"] is True and findings["exploratory"] is True and findings["scientific_result"] is True
            and findings["cases_verified"] == 160 and findings["trajectories_verified"] == 800
            and findings["scored_best_response_decisions"] == 76800, "Analysis completion differs")
    require(findings["analysis_sha256"] == sha(root / "source/analyze_original_multiseed.py")
            and findings["source_sha256"] == spec["source_sha256"]
            and findings["input_sha256"] == spec["input_sha256"]
            and findings["specification_canonical_sha256"] == portable["original_canonical_sha256"]
            and findings["specification_sha256"] == portable["original_file_sha256"], "Analysis provenance differs")
    require(sha(root / "analysis/CASE_VERIFICATION.json") == findings["case_verification_sha256"],
            "Case-verification inventory changed")
    rows = read(root / "analysis/CASE_VERIFICATION.json")
    expected_cases = {(s, d, o) for d, o in analyzer.KEYS for s in analyzer.SEEDS}
    require(len(rows) == 160 and {(r["seed"], r["date"], r["offset"]) for r in rows} == expected_cases,
            "Incorrect case-verification grid")
    required = {p + ".parquet" for p in analyzer.PROFILES} | {"case.json"}
    external = read(root / "REPLAY_TABLE_REQUIREMENTS.json")
    omitted = {r["path"]: r for r in external["files"]}
    require(len(omitted) == len(external["files"]), "Duplicate external replay requirements")
    replay_included = manifest["replay_tables_included"]
    require(len(omitted) == (0 if replay_included else 800), "Incorrect external replay count")
    for r in rows:
        folder = analyzer.case_dir(root, r["seed"], r["date"], r["offset"])
        done = read(folder / "COMPLETE.json"); meta = read(folder / "case.json")
        require(sha(folder / "COMPLETE.json") == r["complete_sha256"], "Completion record changed")
        require(all(done.get(k) is True for k in ("complete", "scientific_result", "exploratory", "identical_truthful_warmup")),
                "Incomplete scientific case")
        require((done["seed"], done["date"], done["offset"]) == (r["seed"], r["date"], r["offset"])
                and done["spec_sha256"] == portable["original_canonical_sha256"]
                and done["files_sha256"] == r["files_sha256"] and set(done["files_sha256"]) == required,
                "Completion binding differs")
        require((meta["seed"], meta["date"], meta["offset"], meta["K"], meta["warmup"], meta["spec_sha256"])
                == (r["seed"], r["date"], r["offset"], 200, 40, portable["original_canonical_sha256"]),
                "Case metadata binding differs")
        for name, h in done["files_sha256"].items():
            p = folder / relative(name)
            if p.exists(): require(sha(p) == h, "Case data hash differs")
            else:
                rel = p.relative_to(root).as_posix()
                require(not replay_included and name.endswith(".parquet") and omitted[rel]["sha256"] == h,
                        "Missing replay data lacks exact external reference")
    for name, h in findings["table_files_sha256"].items():
        require(len(relative(name).parts) == 1 and sha(root / "analysis" / name) == h, "Analysis table changed")
    return manifest, analyzer, findings, rows


def run(args):
    root, output = args.release_root.resolve(), args.output.resolve()
    require(not output.exists() and not output.is_relative_to(root), "Output must be new and outside release payload")
    manifest, analyzer, findings, cases = inspect_package(root, args.expected_manifest_sha256)
    pairs = pd.read_parquet(root / "analysis/case_role_pairs.parquet")
    events = pd.read_parquet(root / "analysis/local_best_response_events.parquet")
    max_difference = 0.; rebuilt_pairs, rebuilt_events = [], []; window_hashes = {}
    if manifest["replay_tables_included"]:
        for r in cases:
            folder = analyzer.case_dir(root, r["seed"], r["date"], r["offset"])
            meta = read(folder / "case.json")
            frames = {p: pd.read_parquet(folder / (p + ".parquet")) for p in analyzer.PROFILES}
            pr, ev, h = analyzer.case_statistics(frames, meta)
            require(h == r["exogenous_trajectory_sha256"], "Reconstructed exogenous hash differs")
            key = (r["date"], r["offset"])
            require(key not in window_hashes or window_hashes[key] == h, "Inputs differ across seeds")
            window_hashes[key] = h
            identity = {k: r[k] for k in ("seed", "date", "offset")}
            rebuilt_pairs.extend(dict(**identity, **x) for x in pr)
            rebuilt_events.extend(dict(**identity, **x) for x in ev)
        max_difference = max(max_difference, compare_frames(pd.DataFrame(rebuilt_pairs), pairs, "episode-derived pairs"),
                             compare_frames(pd.DataFrame(rebuilt_events), events, "episode-derived search events"))
        pairs, events = pd.DataFrame(rebuilt_pairs), pd.DataFrame(rebuilt_events)
    tables = reconstruct_tables(analyzer, pairs, events)
    checks = []
    for name, frame in tables.items():
        for suffix in ("parquet", "csv"):
            path = root / "analysis" / (name + "." + suffix)
            expected = pd.read_parquet(path) if suffix == "parquet" else pd.read_csv(path)
            difference = compare_frames(frame, expected, path.name)
            max_difference = max(max_difference, difference)
            checks.append(dict(file=path.name, rows=len(frame), maximum_absolute_numeric_difference=difference))
    for name, key in (("headline_summary", "headline_summary"), ("per_seed_summary", "per_seed")):
        saved = pd.DataFrame(findings[key]); require(set(saved) == set(tables[name]), "Findings columns differ")
        max_difference = max(max_difference, compare_frames(tables[name], saved[tables[name].columns], "findings " + key))
    directions = {m: dict(positive=int((pairs[m] > analyzer.TOL).sum()), negative=int((pairs[m] < -analyzer.TOL).sum()),
                         near_zero=int((pairs[m].abs() <= analyzer.TOL).sum()), total_seed_window_role_pairs=480)
                  for m in (*analyzer.METRICS, "own_delivery_utility_gain", "own_contract_utility_gain")}
    require(directions == findings["individual_direction_counts"], "Individual sign counts differ")
    local = findings["local_contract_search"]
    require(local["positive_decisions"] == int((events.br_gain > analyzer.TOL).sum())
            and local["negative_beyond_tolerance"] == int((events.br_gain < -analyzer.TOL).sum())
            and local["finite_certificates"] == int(np.isfinite(events.br_cert).sum())
            and abs(local["maximum_gain"] - float(events.br_gain.max())) < 1e-10,
            "Local search count/maximum differs")
    output.mkdir(parents=True)
    regenerated = {}
    for name, frame in tables.items():
        path = output / (name + ".parquet"); frame.to_parquet(path, index=False); regenerated[path.name] = sha(path)
    record = dict(complete=True, scientific_result=True, exploratory=True, manifest_sha256=args.expected_manifest_sha256,
                  adapter_sha256=sha(__file__), analyzer_sha256=findings["analysis_sha256"],
                  cases_cross_checked=160, trajectories_cross_referenced=800, tables_recomputed=len(tables),
                  maximum_absolute_numeric_difference=max_difference, table_comparisons=checks,
                  replay_episode_tables_reread=800 if manifest["replay_tables_included"] else 0,
                  raw_external_traffic_read=False, training_run=False, simulation_run=False,
                  original_private_locators_required=False, regenerated_files_sha256=regenerated,
                  limitation="Reanalysis verifies released derived outputs; it does not rerun raw-traffic simulation or training.")
    (output / "PORTABLE_REANALYSIS_VERIFICATION.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: record[k] for k in ("complete", "tables_recomputed", "maximum_absolute_numeric_difference",
                                            "replay_episode_tables_reread", "raw_external_traffic_read")}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-root", type=Path); p.add_argument("--expected-manifest-sha256")
    p.add_argument("--output", type=Path); p.add_argument("--diagnostic-train-fixture", type=Path)
    p.add_argument("--analyzer", type=Path)
    args = p.parse_args()
    if args.diagnostic_train_fixture:
        require(args.analyzer is not None and args.release_root is None and args.output is None,
                "TRAIN diagnostics must be separate from scientific release/output arguments")
        m = module(args.analyzer)
        m.self_test(argparse.Namespace(native_stats=None, benchmark_fixture=args.diagnostic_train_fixture))
        frames = {n: pd.read_parquet(args.diagnostic_train_fixture / (n + ".parquet")) for n in m.PROFILES}
        rows, events, _ = m.case_statistics(frames)
        compare_frames(pd.DataFrame(rows), pd.DataFrame(rows).copy(), "TRAIN observations")
        broken = pd.DataFrame(rows); broken.loc[0, "honest_on_time_loss"] += 1
        try: compare_frames(broken, pd.DataFrame(rows), "changed TRAIN observation")
        except ValueError: pass
        else: raise AssertionError("Changed derived quantity was accepted")
        # In-memory unit data only: an exactly solvable complete grid exercises
        # role/seed pooling and duplicate/missing guards; no campaign is created.
        synthetic = pd.DataFrame([dict(date=d, offset=o, seed=s, deviator=i,
            honest_on_time_loss=s + 1 + i / 10, honest_on_time_unc_truth=10 + i,
            protected_honest_welfare_gain=2 * (s + 1) + i / 10, honest_value_unc_br=12.,
            own_delivery_utility_gain=-1., own_contract_utility_gain=1.)
            for d, o in m.KEYS for s in m.SEEDS for i in range(3)])
        decision_rows = synthetic.loc[synthetic.index.repeat(160), ["date", "offset", "seed", "deviator"]].reset_index(drop=True)
        decision_rows["k"] = np.tile(np.arange(40, 200), 480)
        decision_rows["br_gain"] = .25; decision_rows["br_U_truth"] = 1.; decision_rows["br_cert"] = np.nan
        result = reconstruct_tables(m, synthetic, decision_rows)["headline_summary"].set_index("metric")
        require(abs(result.loc["honest_on_time_loss", "relative_percent"] - 100 * 3.1 / 11) < 1e-10
                and abs(result.loc["protected_honest_welfare_gain", "relative_percent"] - 100 * 6.1 / 12) < 1e-10,
                "Analytical role/seed pooled-ratio check failed")
        for bad_pairs, bad_events in ((synthetic.iloc[:-1], decision_rows),
                (pd.concat([synthetic.iloc[1:], synthetic.iloc[[1]]], ignore_index=True), decision_rows),
                (synthetic, decision_rows.iloc[:-1])):
            try: reconstruct_tables(m, bad_pairs, bad_events)
            except ValueError: pass
            else: raise AssertionError("Incomplete/duplicate diagnostic grid was accepted")
        print(json.dumps(dict(diagnostic_adapter_checks_passed=True, scientific_result=False,
                              train_profiles=5, train_role_pairs=len(rows), train_search_decisions=len(events),
                              synthetic_grids_only_in_memory=True, missing_duplicate_grid_refusal=True,
                              native_asset_paths_or_gates_required=False)))
        return
    require(args.release_root is not None and args.expected_manifest_sha256 is not None and args.output is not None,
            "Scientific reanalysis requires release root, external manifest SHA-256 and new output")
    run(args)


if __name__ == "__main__":
    main()
