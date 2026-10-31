#!/usr/bin/env python3
"""Independently verify and analyze the complete original-setting five-seed replay.

No training, replay, network access, partial scientific estimates or publication.
The extension is exploratory; it does not change the historical Holm family.
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

DATES = ("2024-09-01", "2024-09-02", "2024-09-08", "2024-09-09",
         "2024-09-15", "2024-09-16", "2024-09-22", "2024-09-23",
         "2024-09-29", "2024-09-30", "2024-10-06", "2024-10-07",
         "2024-10-13", "2024-10-14", "2024-10-20", "2024-10-21")
KEYS = [(d, 60 + 10800 * w) for t, d in enumerate(DATES)
        for w in sorted((t % 8, (t + 4) % 8))]
PROFILES = ("unconstrained_truth", "protected_truth", "unconstrained_br0",
            "unconstrained_br1", "unconstrained_br2")
SEEDS = tuple(range(5))
EXOGENOUS = ("theta", "theta_adm", "arrivals_Mbit", "r_alloc", "r_true")
WARMUP_FIELDS = ("report", "quota", "on_time_Mbit", "dropped_Mbit",
                 "backlog_start_Mbit", "backlog_Mbit")
METRICS = {
    "honest_on_time_loss": ("honest_on_time_loss", "honest_on_time_unc_truth",
                            "Mbit per scored interval", "Unconstrained truthful honest on-time Mbit"),
    "protected_honest_welfare_gain": ("protected_honest_welfare_gain", "honest_value_unc_br",
                                     "credits per scored interval", "Unconstrained best-response honest delivery welfare"),
}
ORIGINAL = {"honest_on_time_loss": 19.44203832884902,
            "protected_honest_welfare_gain": 16.059932511329347}
ORIGINAL_SOURCES = {
    "honest_on_time_loss": dict(file="implementation/results/summary/e2.json", macro="EtwoVictimLossPct",
        sha256="ca2454e7b7aa8122878880bc9925840119cdf532b1aa1fe841be89e15b50e021"),
    "protected_honest_welfare_gain": dict(file="implementation/results/summary/e4.json", macro="EfourHonestPct",
        sha256="277fc6c58b2ebf11b373a176f6898f538091ec9c526edaf36cd033fd131cb4c4"),
}
TOL = 1e-8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def relative(value):
    p = Path(value)
    require(not p.is_absolute() and value not in ("", ".") and ".." not in p.parts,
            "Unsafe relative path: " + value)
    return p


def case_dir(campaign, seed, date, offset):
    return campaign / "cases" / f"seed{seed}" / f"{date}_o{offset}"


def status(campaign):
    expected = {case_dir(campaign, s, d, o) for s in SEEDS for d, o in KEYS}
    complete = {p.parent for p in (campaign / "cases").glob("seed*/*/COMPLETE.json")}
    require(not complete - expected, "Unexpected completed case directories")
    return dict(expected_cases=160, completion_markers=len(complete),
                expected_trajectories=800, complete_grid=complete == expected,
                estimates_computed=False)


def verify_spec(args):
    spec = load(args.campaign / "SPECIFICATION.json")
    require(spec["study"] == "original_setting_five_seed_headline_supplement", "Wrong study")
    require(spec["scientific_result"] is True and spec["exploratory"] is True, "Wrong scientific scope")
    require(spec["seeds"] == list(SEEDS) and spec["window_keys"] == [list(k) for k in KEYS],
            "Seed/window grid differs from the original rotated 32-window rule")
    require(spec["K"] == 200 and spec["warmup"] == 40 and spec["episode_profiles"] == list(PROFILES),
            "Episode length, warm-up or profiles differ")
    require(sha(args.campaign / "PROTOCOL.txt") == spec["protocol_sha256"], "Protocol changed")
    ss = digest(spec)
    pre = load(args.campaign / "PREFLIGHT.json")
    require(pre["complete"] and pre["native_test_gate_passed"] and pre["cases"] == 160
            and pre["trajectories"] == 800 and pre["models_verified"] == 10
            and pre["spec_sha256"] == ss, "Preflight scope/provenance mismatch")
    final = load(args.campaign / "CAMPAIGN_COMPLETE.json")
    require(final["complete"] and not final["errors"] and final["expected_cases"] == 160
            and final["trajectories"] == 800 and final["spec_sha256"] == ss,
            "Campaign is incomplete or contains failures")
    require(status(args.campaign)["complete_grid"], "Incomplete case grid; no scientific estimates produced")
    original = load(args.frozen_support / "upstream_asset_manifest.json")
    recovered = {r["source_path"].split("ICC/rewrite_2026-10-18/", 1)[1]: r["sha256"]
                 for r in original["files"]}
    parquet = load(args.frozen_support / "puffer_archive_recovery_manifest.json")
    recovered.update({r["relative_output"].removeprefix("upstream/"): r["sha256"]
                      for r in parquet["tables"]})
    require(spec["origin"] == {"repository": original["repository"], "revision": original["revision"]},
            "Recovered source revision differs")
    for name, expected in spec["input_sha256"].items():
        p = args.assets / relative(name)
        require(recovered.get(name) == expected and sha(p) == expected, "Changed input: " + name)
    for name, expected in spec["source_sha256"].items():
        if name == "supplemental_runner":
            p = args.runner
        elif name == "study_support.py":
            p = args.frozen_support / name
        else:
            require(name.startswith("code/"), "Unexpected source category")
            p = args.assets / relative(name)
        require(sha(p) == expected, "Changed scientific source: " + name)
    gate = spec["native_test_gate"]
    require(gate["test_dates"] == 16 and sha(args.assets / "prereg/PREREG.md") == gate["sha256"],
            "Original freeze differs")
    expected_models = {f"{a}/seed{s}" for a in ("protected", "unconstrained") for s in SEEDS}
    require(set(spec["models"]) == expected_models, "Wrong model coverage")
    for name, record in spec["models"].items():
        require(record["weights_only"] is True and record["file_sha256"] ==
                spec["input_sha256"][f"results/models/default/{name}/model.pt"],
                "Model load record differs: " + name)
    return spec, ss


def validate_frame(frame, label, metadata=None):
    required = {"k", "tenant", "cls", "scored", "quota", "price", "V_declared", "realized_value",
                "utility_declared", "utility_realized", "br_gain", "br_cert", "br_U_truth",
                "report", "on_time_Mbit", "dropped_Mbit", "late_Mbit", "backlog_start_Mbit",
                "backlog_Mbit", *EXOGENOUS}
    require(required <= set(frame), "Missing episode columns: " + label)
    require(len(frame) == 600 and not frame.duplicated(["k", "tenant"]).any(), "Duplicate/missing interval rows")
    f = frame.sort_values(["k", "tenant"]).reset_index(drop=True)
    require(np.array_equal(f.k.to_numpy(), np.repeat(np.arange(200), 3))
            and np.array_equal(f.tenant.to_numpy(), np.tile(np.arange(3), 200)), "Wrong interval/tenant grid")
    require(np.array_equal(f.scored.to_numpy(), f.k.to_numpy() >= 40), "Wrong scoring mask")
    require(all(f.loc[f.tenant == i, "cls"].eq(c).all() for i, c in enumerate(("video", "urllc", "mmtc"))),
            "Tenant/class correspondence changed")
    if metadata is not None:
        for c, v in (("profile", label), ("date", metadata["date"]), ("seed", metadata["seed"]),
                     ("window", metadata["window_id"]),
                     ("deviator", int(label[-1]) if label.startswith("unconstrained_br") else -1)):
            require(c in f and f[c].eq(v).all(), "Episode identity mismatch: " + c)
    finite = required - {"cls", "br_gain", "br_cert", "br_U_truth"}
    require(np.isfinite(f[list(finite)].to_numpy(dtype=float)).all(), "Nonfinite physical/utility data")
    require((f[["arrivals_Mbit", "on_time_Mbit", "dropped_Mbit", "late_Mbit",
                "backlog_start_Mbit", "backlog_Mbit", "quota"]] >= -TOL).all().all(), "Negative volume/quota")
    require(np.allclose(f.utility_declared, f.V_declared - f.price, rtol=1e-10, atol=TOL)
            and np.allclose(f.utility_realized, f.realized_value - f.price, rtol=1e-10, atol=TOL),
            "Utility identity failed")
    require(np.allclose(f.realized_value, f.on_time_Mbit * np.tile([1., 3., .5], 200), atol=TOL, rtol=1e-10),
            "Delivery value/tariff identity failed")
    require(np.allclose(f.quota, np.floor(f.quota), atol=1e-12, rtol=0)
            and f.groupby("k").quota.sum().max() <= 106 + TOL, "Integer quota/budget invariant failed")
    residual = f.arrivals_Mbit - f.on_time_Mbit - f.late_Mbit - f.dropped_Mbit - (f.backlog_Mbit - f.backlog_start_Mbit)
    scale = np.maximum(1., f.arrivals_Mbit + f.backlog_start_Mbit)
    require((np.abs(residual) <= 1e-8 * scale).all(), "Byte conservation failed")
    for tenant in range(3):
        t = f[f.tenant == tenant]
        require(np.array_equal(t.backlog_start_Mbit.to_numpy()[1:], t.backlog_Mbit.to_numpy()[:-1]),
                "Backlog handover failed")
    return f


def case_statistics(frames, metadata=None):
    frames = {label: validate_frame(frames[label], label, metadata) for label in PROFILES}
    truth = frames["unconstrained_truth"]
    for label, f in frames.items():
        require(np.array_equal(f[list(EXOGENOUS)].to_numpy(), truth[list(EXOGENOUS)].to_numpy()),
                "Different exogenous trajectory: " + label)
        truthful_rows = np.ones(len(f), dtype=bool)
        if label.startswith("unconstrained_br"):
            truthful_rows = (f.tenant.to_numpy() != int(label[-1])) | (f.k.to_numpy() < 40)
        require(np.array_equal(f.loc[truthful_rows, "report"].to_numpy(),
                               f.loc[truthful_rows, "theta_adm"].to_numpy()),
                "Truthful baseline/opponent/warm-up report identity failed: " + label)
        if label.startswith("unconstrained_br"):
            require(np.array_equal(f.loc[f.k < 40, list(WARMUP_FIELDS)].to_numpy(),
                                   truth.loc[truth.k < 40, list(WARMUP_FIELDS)].to_numpy()),
                    "Deviation did not share truthful warm-up")
            require(np.array_equal(f.loc[f.k == 40, "backlog_start_Mbit"].to_numpy(),
                                   truth.loc[truth.k == 40, "backlog_start_Mbit"].to_numpy()),
                    "Different starting backlog at deviation")
    means = {label: f[f.k >= 40].groupby("tenant").mean(numeric_only=True) for label, f in frames.items()}
    rows, br_events = [], []
    t, p = means["unconstrained_truth"], means["protected_truth"]
    for role in range(3):
        b = means[f"unconstrained_br{role}"]; honest = [i for i in range(3) if i != role]
        row = dict(deviator=role, tenant=("video", "interactive", "background")[role],
                   honest_on_time_unc_truth=float(t.loc[honest, "on_time_Mbit"].sum()),
                   honest_on_time_unc_br=float(b.loc[honest, "on_time_Mbit"].sum()),
                   honest_value_protected_truth=float(p.loc[honest, "realized_value"].sum()),
                   honest_value_unc_br=float(b.loc[honest, "realized_value"].sum()),
                   own_delivery_utility_truth=float(t.loc[role, "utility_realized"]),
                   own_delivery_utility_gain=float(b.loc[role, "utility_realized"] - t.loc[role, "utility_realized"]),
                   own_contract_utility_gain=float(b.loc[role, "utility_declared"] - t.loc[role, "utility_declared"]))
        row["honest_on_time_loss"] = row["honest_on_time_unc_truth"] - row["honest_on_time_unc_br"]
        row["protected_honest_welfare_gain"] = row["honest_value_protected_truth"] - row["honest_value_unc_br"]
        rows.append(row)
        f = frames[f"unconstrained_br{role}"]
        f = f[(f.k >= 40) & (f.tenant == role)]
        require(len(f) == 160 and np.isfinite(f.br_gain).all() and np.isfinite(f.br_U_truth).all(),
                "Missing scored best-response results")
        for r in f[["k", "report", "theta_adm", "br_gain", "br_cert", "br_U_truth"]].to_dict("records"):
            br_events.append(dict(deviator=role, **r))
    values = np.ascontiguousarray(truth[list(EXOGENOUS)].to_numpy())
    return rows, br_events, hashlib.sha256(values.tobytes()).hexdigest()


def bootstrap(values, dates, n_boot=10000, seed=0):
    """Same date/window draw algorithm as original stats.cluster_bootstrap_multi."""
    v = np.asarray(values, dtype=float)
    require(v.ndim == 2 and v.shape[1] == 2 and np.isfinite(v).all(), "Bad bootstrap values")
    uniq, inverse = np.unique(np.asarray(dates), return_inverse=True)
    v = v[np.argsort(inverse, kind="stable")]
    counts = np.bincount(inverse); offsets = np.r_[0, np.cumsum(counts)[:-1]]
    require(len(uniq) == 16 and np.array_equal(counts, np.full(16, 2)), "Bootstrap needs 16 dates x2 windows")
    require((v[:, 1] > 0).all(), "Nonpositive headline denominator")
    rng = np.random.default_rng(seed); G = len(uniq); nmax = int(counts.max())
    chunk = max(1, min(n_boot, 4_000_000 // (G * nmax * v.shape[1])))
    out = np.empty((n_boot, 2))
    for start in range(0, n_boot, chunk):
        R = min(chunk, n_boot - start); D = rng.integers(G, size=(R, G)); nsel = counts[D]
        u = rng.random((R, G, nmax)); idx = np.minimum((u * nsel[..., None]).astype(np.int64), nsel[..., None] - 1)
        picked = offsets[D][..., None] + idx
        means = v[picked].sum((1, 2)) / nsel.sum(1)[:, None]
        out[start:start + R, 0] = means[:, 0]
        out[start:start + R, 1] = 100 * means[:, 0] / means[:, 1]
    return out


def summarize(windows, seed_label):
    records = []
    for name, (numerator, denominator, unit, label) in METRICS.items():
        v = windows[[numerator, denominator]].to_numpy(); means = v.mean(0)
        reps = bootstrap(v, windows.date.to_numpy())
        records.append(dict(metric=name, seed=seed_label, estimate=float(means[0]), unit=unit,
                            ci_low=float(np.quantile(reps[:, 0], .025)), ci_high=float(np.quantile(reps[:, 0], .975)),
                            denominator_mean=float(means[1]), denominator=label,
                            relative_percent=float(100 * means[0] / means[1]),
                            relative_ci_low=float(np.quantile(reps[:, 1], .025)),
                            relative_ci_high=float(np.quantile(reps[:, 1], .975)), n_dates=16, n_windows=32,
                            n_training_seeds=5 if seed_label == "all" else 1, bootstrap_replicates=10000,
                            bootstrap_seed=0, ci_conditional_on_fitted_seeds=True))
    return records


def self_test(args):
    d = np.repeat(DATES, 2); v = np.c_[np.arange(32) / 5 + 1, np.arange(32) / 4 + 10]
    result = bootstrap(v, d)
    require(np.array_equal(result, bootstrap(v, d)), "Bootstrap is not deterministic")
    constant = bootstrap(np.tile([2., 10.], (32, 1)), d)
    require(np.array_equal(constant, np.tile([2., 20.], (10000, 1))), "Constant effect/ratio failed")
    pooled = pd.DataFrame({"date": d})
    for numerator, denominator, _, _ in METRICS.values():
        pooled[numerator] = 1.; pooled[denominator] = np.r_[np.ones(16), np.full(16, 9.)]
    require(all(r["relative_percent"] == 20. for r in summarize(pooled, "all")),
            "Headline incorrectly averaged individual percentages (55.56%) instead of pooled quantities (20%)")
    if args.native_stats:
        spec = importlib.util.spec_from_file_location("original_stats_verification", args.native_stats)
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
        native = module.cluster_bootstrap_multi(v, d, {"absolute": lambda x: x[..., 0],
                "relative": lambda x: 100 * x[..., 0] / x[..., 1]}, n_boot=10000, seed=0)
        require(np.array_equal(result[:, 0], native["absolute"].reps)
                and np.array_equal(result[:, 1], native["relative"].reps), "Native bootstrap mismatch")
    if args.benchmark_fixture:
        frames = {n: pd.read_parquet(args.benchmark_fixture / (n + ".parquet")) for n in PROFILES}
        rows, events, _ = case_statistics(frames)
        require(len(rows) == 3 and len(events) == 480, "TRAIN diagnostic extraction failed")
        broken = {k: f.copy() for k, f in frames.items()}; broken["unconstrained_br0"].loc[0, "quota"] += 1
        try:
            case_statistics(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed warm-up was not rejected")
    print(json.dumps(dict(diagnostics_passed=True, scientific_result=False,
                          native_bootstrap_checked=bool(args.native_stats),
                          train_fixture_checked=bool(args.benchmark_fixture))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("campaign", "assets", "frozen-support", "runner", "output", "native-stats", "benchmark-fixture"):
        ap.add_argument("--" + n, type=Path)
    ap.add_argument("--status-only", action="store_true"); ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(args); return
    require(args.campaign is not None, "--campaign is required")
    if args.status_only:
        print(json.dumps(status(args.campaign))); return
    require(all(getattr(args, n) is not None for n in ("assets", "frozen_support", "runner", "output")),
            "Analysis requires --assets --frozen-support --runner --output")
    require(not args.output.exists(), "Refusing to overwrite analysis output")
    spec, ss = verify_spec(args)
    pairs, events, inventories = [], [], []
    windows_hashes = {}; partial_hashes = {}; meta_hashes = {}
    expected_files = {p + ".parquet" for p in PROFILES} | {"case.json"}
    for date, offset in KEYS:
        for seed in SEEDS:
            p = case_dir(args.campaign, seed, date, offset); done = load(p / "COMPLETE.json")
            require(all(done.get(k) is True for k in ("complete", "scientific_result", "exploratory", "identical_truthful_warmup")),
                    "Case flags are invalid")
            require(done["spec_sha256"] == ss and (done["seed"], done["date"], done["offset"]) == (seed, date, offset),
                    "Case completion identity differs")
            require(set(done["files_sha256"]) == expected_files
                    and {x.name for x in p.glob("*.parquet")} == expected_files - {"case.json"}, "Unexpected/missing case data files")
            for name, expected in done["files_sha256"].items():
                require(sha(p / relative(name)) == expected, "Changed case output: " + str(p / name))
            meta = load(p / "case.json")
            require((meta["seed"], meta["date"], meta["offset"], meta["K"], meta["warmup"], meta["spec_sha256"])
                    == (seed, date, offset, 200, 40, ss), "Case metadata differs")
            frames = {name: pd.read_parquet(p / (name + ".parquet")) for name in PROFILES}
            rows, br, exogenous_hash = case_statistics(frames, meta)
            identity = dict(seed=seed, date=date, offset=offset)
            pairs.extend(dict(**identity, **r) for r in rows); events.extend(dict(**identity, **r) for r in br)
            key = (date, offset)
            for seen, value in ((windows_hashes, exogenous_hash), (partial_hashes, meta["window_input_sha256"]),
                                (meta_hashes, meta["window_meta_sha256"])):
                require(key not in seen or seen[key] == value, "Exogenous window differs across training seeds")
                seen[key] = value
            inventories.append(dict(**identity, complete_sha256=sha(p / "COMPLETE.json"),
                                    exogenous_trajectory_sha256=exogenous_hash, files_sha256=done["files_sha256"]))
    pairs = pd.DataFrame(pairs); events = pd.DataFrame(events)
    require(len(pairs) == 480 and len(events) == 76800, "Unexpected paired/event count")
    value_columns = [c for c in pairs.select_dtypes(include="number").columns if c not in ("seed", "offset", "deviator")]
    per_seed = pairs.groupby(["date", "offset", "seed"], as_index=False)[value_columns].mean()
    windows = per_seed.groupby(["date", "offset"], as_index=False)[value_columns].mean()
    summary = pd.DataFrame(summarize(windows, "all"))
    by_seed = pd.DataFrame([r for seed, w in per_seed.groupby("seed") for r in summarize(w, int(seed))])
    seed0 = by_seed[by_seed.seed == 0].set_index("metric")
    checks = {m: dict(archived_percent=v, rerun_seed0_percent=float(seed0.loc[m, "relative_percent"]),
                     difference_percentage_points=float(seed0.loc[m, "relative_percent"] - v),
                     rounded_headline_matches=f'{seed0.loc[m, "relative_percent"]:.2f}' == f'{v:.2f}')
              for m, v in ORIGINAL.items()}
    for i, r in summary.iterrows():
        b = by_seed[by_seed.metric == r.metric]
        summary.loc[i, "training_seed_relative_min"] = b.relative_percent.min()
        summary.loc[i, "training_seed_relative_max"] = b.relative_percent.max()
        summary.loc[i, "training_seed_relative_sd"] = b.relative_percent.std(ddof=1)
    directions = {m: {"positive": int((pairs[m] > TOL).sum()), "negative": int((pairs[m] < -TOL).sum()),
                      "near_zero": int((pairs[m].abs() <= TOL).sum()), "total_seed_window_role_pairs": 480}
                  for m in (*METRICS, "own_delivery_utility_gain", "own_contract_utility_gain")}
    best = events.loc[events.br_gain.idxmax()]
    findings = dict(complete=True, scientific_result=True, exploratory=True, original_holm_family=False,
                    ready_for_manuscript=all(c["rounded_headline_matches"] for c in checks.values()),
                    cases_verified=160, trajectories_verified=800, scored_best_response_decisions=76800,
                    analysis_sha256=sha(__file__), specification_sha256=sha(args.campaign / "SPECIFICATION.json"),
                    specification_canonical_sha256=ss, protocol_sha256=spec["protocol_sha256"],
                    source_sha256=spec["source_sha256"], input_sha256=spec["input_sha256"],
                    headline_summary=summary.to_dict("records"), per_seed=by_seed.to_dict("records"),
                    seed0_replication=checks, original_summary_sources=ORIGINAL_SOURCES,
                    original_reference_version="https://github.com/dibbend8/truthful-ran-slicing/tree/v0.1.0",
                    individual_direction_counts=directions,
                    local_contract_search=dict(positive_decisions=int((events.br_gain > TOL).sum()),
                        negative_beyond_tolerance=int((events.br_gain < -TOL).sum()), total=76800,
                        finite_certificates=int(np.isfinite(events.br_cert).sum()),
                        maximum_gain=float(best.br_gain), maximum_context={k: (int(best[k]) if k in ("seed", "offset", "deviator", "k") else str(best[k]))
                            for k in ("date", "seed", "offset", "deviator", "k")}),
                    numerical_sign_tolerance=TOL,
                    limitations=["Exploratory extension on reused TEST data; no Holm-adjusted or confirmatory claims.",
                        "Date/window confidence intervals condition on the five fitted networks; seed spread is separate.",
                        "Traffic/channel capture reuse can create dependence beyond Puffer dates.",
                        "Unconstrained best response is a numerical search lower bound, not a certified global maximum.",
                        "This experiment does not test Protected deviations or rerun packet simulation.",
                        "The runner window_input_sha256 covers selected Window arrays only; analysis additionally matches all exported exogenous arrays, including arrivals, across trajectories/seeds."])
    args.output.mkdir(parents=True)
    table_hashes = {}
    for name, frame in (("headline_summary", summary), ("per_seed_summary", by_seed), ("case_role_pairs", pairs),
                        ("window_seed_pairs", per_seed), ("seed_averaged_windows", windows), ("local_best_response_events", events)):
        for suffix in ("csv", "parquet"):
            path = args.output / f"{name}.{suffix}"
            if suffix == "csv": frame.to_csv(path, index=False)
            else: frame.to_parquet(path, index=False)
            table_hashes[path.name] = sha(path)
    (args.output / "CASE_VERIFICATION.json").write_text(json.dumps(inventories, indent=2, sort_keys=True) + "\n")
    findings["case_verification_sha256"] = sha(args.output / "CASE_VERIFICATION.json")
    findings["table_files_sha256"] = table_hashes
    (args.output / "findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=True, output=str(args.output), ready_for_manuscript=findings["ready_for_manuscript"],
                          summary=findings["headline_summary"], seed0_replication=checks)))


if __name__ == "__main__":
    main()
