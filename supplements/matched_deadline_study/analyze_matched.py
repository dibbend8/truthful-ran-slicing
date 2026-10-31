#!/usr/bin/env python3
"""Validate and analyze complete matched-training evaluation cells.

No simulator/training code is imported. Scientific reports require every one
of the 32 planned windows per requested cell and mode. --diagnostic-smoke only
validates the separate TRAIN smoke output; it never produces research estimates.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIGS = {"rho0.7": .7, "default": .9, "rho1.1": 1.1}
PROFILES = ("video300", "interactive50", "both")
ARMS = ("protected", "unconstrained", "meas_only", "kappa0")
TRAFFIC = ("video", "interactive", "background")
MULTIPLIERS = (.5, .67, .8, .9, 1.1, 1.25, 1.5, 2.)
SEEDS = (0, 1, 2)
OFFSETS = (60, 43260)
KEY = ["config", "profile", "target_load"]
WINDOW = ["date", "offset"]
NUMERIC_TOL = 1e-9
GAIN_POSITIVE = 1e-8
BOOTSTRAP_SEED = 20261031
MODEL_FILES = {"model.pt", "final_model.pt", "config.json", "history.csv", "initial_rng.json",
               "window_sampling_schedule.npz", "window_inputs.json", "prepared_inputs.json"}
STATIC_FILES = {"static.json", "candidates.npz", "candidates_train.csv", "candidates_selection.csv",
                "selected_weights.json", "train_best_weights.json", "window_inputs.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical(value):
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(canonical({"shape": a.shape, "dtype": a.dtype.str}).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def close(actual, expected, label, atol=NUMERIC_TOL):
    require(np.allclose(np.asarray(actual, float), np.asarray(expected, float), rtol=1e-10,
                        atol=atol, equal_nan=True), label)


def dates():
    current, values = date(2024, 9, 1), []
    while current <= date(2024, 10, 21):
        if current.weekday() in (6, 0):
            values.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(values)


def policies(mode, smoke=False):
    learned = ARMS if mode == "truth" else ARMS[:2]
    seeds = (0,) if smoke else SEEDS
    result = [(state, arm, seed) for state in ("frozen", "retrained") for arm in learned for seed in seeds
              if not (smoke and state == "retrained" and arm != "protected")]
    if mode == "truth":
        result += [(s, a, -1) for s, a in (("frozen", "static"), ("retrained", "static"),
                                            ("reference", "equal"), ("reference", "direct"))]
    return {f"{s}__{a}__s{seed}": (s, a, seed) for s, a, seed in result}


def runs(mode, smoke=False):
    result = []
    for policy in policies(mode, smoke):
        result.append((policy, -1, 1.))
        if mode == "strategic":
            result.extend((policy, i, m) for i in range(3) for m in MULTIPLIERS)
    return result


def interval_name(policy, tenant, multiplier):
    return f"intervals/{policy}__i{tenant}__m{multiplier:g}.npz"


class Verifier:
    def __init__(self, args):
        self.args = args
        self.checked = {}
        self.models = set()
        self.sources = {}
        self.path_translations = {}
        self.cross_mode = {}
        self.case_signatures = {}
        self.manifests = []
        git_path=args.frozen_study/"upstream_asset_manifest.json"
        parquet_path=args.frozen_study/"puffer_archive_recovery_manifest.json"
        git=json.loads(git_path.read_text());parquet=json.loads(parquet_path.read_text())
        self.recovered={r["source_path"].split("ICC/rewrite_2026-10-18/",1)[1]:r["sha256"] for r in git["files"]}
        self.recovered.update({r["relative_output"].removeprefix("upstream/"):r["sha256"] for r in parquet["tables"]})
        self.origin={"repository":git["repository"],"revision":git["revision"],
                     "github_manifest_sha256":sha(git_path),"puffer_manifest_sha256":sha(parquet_path)}
        for path in sorted((args.root / "evaluation_runs").glob("*.json")):
            if ".attempt-" in path.name:
                continue
            m = json.loads(path.read_text())
            require(digest(m) == path.stem, f"run manifest digest mismatch: {path}")
            require(len(m["case_spec_sha256"]) == m["case_count"] and len(set(m["case_spec_sha256"])) == m["case_count"],
                    f"run manifest contains duplicate/missing case specifications: {path}")
            self.manifests.append((path, m))

    def file(self, path, expected):
        path = Path(path)
        st = path.stat()
        key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
        if key not in self.checked:
            self.checked[key] = sha(path)
        require(self.checked[key] == expected, f"artifact hash mismatch: {path}")
        return expected

    def source(self, name, expected):
        roots = {"runner": self.args.root, "frozen-study": self.args.frozen_study, "assets": self.args.assets}
        prefix, relative = name.split("/", 1)
        require(prefix in roots and not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                f"unexpected source path: {name}")
        if name in self.sources:
            require(self.sources[name] == expected, f"mixed source versions: {name}")
        self.sources[name] = expected
        self.file(roots[prefix] / relative, expected)

    def input(self, name, expected):
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                "invalid relative input path")
        path = self.args.assets / name
        self.file(path, expected)
        if name not in self.recovered:
            return
        original = self.recovered[name]
        if name != "data/puffer_raw/manifest.json":
            require(original == expected, "evaluation input differs from recovered original")
            return
        # The replay relocates only this manifest's top-level frozen-document
        # locator. Preserve all nested receipts and every other field exactly.
        historical_path = self.args.frozen_study / "original_puffer_fetch_manifest.json"
        self.file(historical_path, original)
        current = json.loads(path.read_text())
        historical = json.loads(historical_path.read_text())
        current_locator = current["freeze"].pop("path")
        original_locator = historical["freeze"].pop("path")
        require(current == historical, "Puffer manifest differs beyond top-level freeze.path relocation")
        self.file(self.args.assets / "prereg/PREREG.md", current["freeze"]["sha256"])
        self.path_translations[name] = {
            "only_field": "freeze.path", "original_sha256": original,
            "current_sha256": expected, "locator_changed": current_locator != original_locator,
            "frozen_document_verified_relative_path": "prereg/PREREG.md"}

    def model(self, entry, spec, smoke):
        if entry["state"] == "reference":
            require(entry.get("no_fitted_parameters") is True, "reference falsely carries fitted parameters")
            return
        signature = digest({"entry":entry,"config":spec["config"],"profile":spec["profile"],
                            "shape_sha256":spec["shape_sha256"],"protocol_sha256":spec["protocol_sha256"],
                            "scientific_result":not smoke})
        if signature in self.models:
            return
        if entry["state"] == "frozen":
            folder = self.args.assets / entry["asset_relative_dir"]
            for name, expected in entry["files_sha256"].items():
                relative=(Path(entry["asset_relative_dir"])/name).as_posix()
                require(self.recovered.get(relative)==expected,"frozen model differs from recovered original")
                self.file(folder/name, expected)
        else:
            model_root = self.args.model_root or self.args.root / ("smoke" if smoke else "models")
            folder = model_root / entry["model_relative_dir"]
            marker = folder / "COMPLETE.json"
            self.file(marker, entry["completion_sha256"])
            done = json.loads(marker.read_text())
            expected_files = STATIC_FILES if entry["arm"] == "static" else MODEL_FILES
            require(done.get("complete") is True and done.get("scientific_result") is (not smoke),
                    f"incorrect training completion/scientific flag: {folder}")
            require(set(done["files_sha256"]) == expected_files and done["files_sha256"] == entry["files_sha256"],
                    f"model completion inventory mismatch: {folder}")
            require({p.name for p in folder.iterdir()} == expected_files | {"COMPLETE.json"}, f"extra/missing model files: {folder}")
            for name, expected in entry["files_sha256"].items():
                self.file(folder/name, expected)
            record = json.loads((folder/("static.json" if entry["arm"] == "static" else "config.json")).read_text())
            fit = record["spec"]
            require(digest(fit) == entry["spec_sha256"] == done["spec_sha256"] == record["spec_sha256"], "training spec digest mismatch")
            expected_job = {"config": spec["config"], "profile": spec["profile"], "arm": entry["arm"]}
            if entry["arm"] != "static":
                expected_job["seed"] = entry["seed"]
            require(fit["job"] == expected_job and fit["shape_file_sha256"] == spec["shape_sha256"], "wrong fitted model/shape")
            require(record["validation"].get("no_test_windows") is True, "training lacks no-TEST validation")
            if not smoke:
                require(fit["scientific_result"] is True and fit["study_protocol_sha256"] == spec["protocol_sha256"], "model protocol/scientific mismatch")
                if entry["arm"] == "static":
                    require((fit["candidate_count"], fit["dev_top_k"], fit["selection_split"]) == (232, 72, "dev_selection"), "wrong static fitting budget")
                    with np.load(folder/"candidates.npz",allow_pickle=False) as archive:
                        candidates,scores=archive["candidates"],archive["scores"]
                    require(candidates.shape==(232,3) and scores.shape==(232,) and np.isfinite(scores).all(),"static candidate arrays malformed")
                    require(len({tuple(np.round(w,12)) for w in candidates})==232,"duplicate static grid candidates")
                    order=np.argsort(-scores,kind="stable")[:72]
                    choice=record["dev_choice"]
                    require(len(choice)==72,"incomplete static DEV shortlist")
                    for index,row in enumerate(choice):
                        close(row["w"],candidates[order[index]],"static shortlist differs from stable TRAIN ranking")
                        close(row["train_welfare"],scores[order[index]],"static TRAIN candidate score mismatch")
                    chosen=int(np.argmax([row["dev_welfare"] for row in choice]))
                    require(record["dev_choice_rank"]==chosen,"static DEV selection/tie rule changed")
                    close(record["w"],choice[chosen]["w"],"selected static weights mismatch")
                else:
                    pc = fit["ppo"]
                    require((pc["iterations"], pc["windows_per_iter"], pc["M"], pc["horizon"]) == (400,32,8,3), "wrong neural training budget")
                    exact={"passes":4,"minibatch":2048,"eval_every":50,"single_thread":True}
                    require(all(pc.get(k)==v for k,v in exact.items()),"PPO update/checkpoint budget changed")
                    for name,value in {"clip":.2,"grad_clip":1.,"lr_start":.0003,"lr_end":.00003,
                                       "sigma0":.6,"sigma_min":.05,"sigma_max":1.,"entropy_coef":.001}.items():
                        close(pc[name],value,f"PPO configuration changed: {name}",atol=1e-15)
                    history=pd.read_csv(folder/"history.csv")
                    require(record["result"]["iterations"] == 400 and len(history) == 400 and history.iteration.tolist()==list(range(1,401)), "incomplete training history")
                    require(set(map(int, record["result"]["dev_curve"])) == set(range(50,401,50)), "wrong DEV checkpoint grid")
                    curve={int(k):v for k,v in record["result"]["dev_curve"].items()}
                    best=max(sorted(curve),key=curve.get)
                    require(record["result"]["best_iteration"]==best,"checkpoint does not maximize DEV with earliest tie")
                    close(record["result"]["selection_welfare"],curve[best],"selected DEV score mismatch")
                    require(len(record["result"]["selection_per_window"])==66,"selected DEV windows incomplete")
                keys=fit["window_keys"]
                train=keys["train"];selection=keys["selection"] if entry["arm"]=="static" else keys["dev_selection"]
                require(len(train)==len(set(map(tuple,train)))==180 and len(selection)==len(set(map(tuple,selection)))==66,"TRAIN/DEV fitting window grid incomplete")
                require(not({d for d,_ in train}|{d for d,_ in selection})&set(dates()),"TEST date appears in fitted-model specification")
            for name, expected in fit["source_sha256"].items():
                require(spec["source_sha256"].get(name) == expected, f"fitting/evaluation source mismatch: {name}")
        self.models.add(signature)


def validate_arrays(a, rows, spec, window, shape, mechanism, policy, deviator, multiplier):
    K, warm = window["K"], window["warmup"]
    required = ("quota", "theta", "theta_adm", "report", "z_alloc", "r_alloc", "V_declared", "V_report", "realized_value",
                "price", "price_lo", "price_hi", "utility_realized", "utility_declared", "fixed_price", "fixed_price_lo", "fixed_price_hi",
                "fixed_utility_realized", "fixed_utility_declared", "arrivals_Mbit", "on_time_Mbit", "dropped_Mbit", "late_Mbit",
                "backlog_start_Mbit", "backlog_Mbit")
    require(set(required) <= a.keys() and all(a[k].shape == (K,3) for k in required), "interval array inventory/shape mismatch")
    require(np.array_equal(a["scored"], np.arange(K) >= warm), "incorrect scored interval mask")
    state, arm, seed = policies(spec["mode"], not spec["scientific_result"])[policy]
    for name in ("theta", "theta_adm", "report", "quota", "r_alloc", "realized_value", "V_declared", "V_report",
                 "arrivals_Mbit", "on_time_Mbit", "dropped_Mbit", "late_Mbit", "backlog_start_Mbit", "backlog_Mbit"):
        require(np.isfinite(a[name]).all() and np.min(a[name]) >= -1e-9, f"invalid nonnegative interval field: {name}")
    require(np.array_equal(a["quota"], np.floor(a["quota"])) and (a["quota"].sum(axis=1) <= window["B"]).all(), "quota feasibility failure")
    require(array_sha(a["theta"]) == window["arrays_sha256"]["theta"] and array_sha(a["r_alloc"]) == window["arrays_sha256"]["rate_hat"], "recorded window/rate input hash mismatch")
    ct = shape["contract"]
    admitted = np.minimum(a["theta"], np.asarray(ct["C"]))
    reports = a["theta"].copy()
    if deviator >= 0:
        reports[warm:, deviator] *= multiplier
    reports = np.minimum(reports, np.asarray(ct["C"]))
    require(np.array_equal(a["theta_adm"], admitted) and np.array_equal(a["report"], reports), "wrong true demand/report clipping or warm-up")
    if arm != "meas_only":
        require(np.array_equal(a["z_alloc"], reports), "report-aware allocator has different inputs")
    close(a["realized_value"], a["on_time_Mbit"]*np.asarray(ct["pi"]), "delivery values do not equal credited on-time traffic")
    for name, demand in (("V_declared", admitted), ("V_report", reports)):
        expected = np.zeros((K,3))
        for i in range(3):
            atoms = np.asarray(shape["shape"]["beta"][i])
            mass = np.asarray(shape["shape"]["rho"][i])
            expected[:,i] = ct["pi"][i]*mechanism["Delta"]*np.sum(mass*np.minimum(demand[:,i,None]*atoms,
                                              (a["r_alloc"][:,i]*a["quota"][:,i])[:,None]),axis=1)
        close(a[name], expected, f"independent contract-value reconstruction failed: {name}")
    residual = a["arrivals_Mbit"]-a["on_time_Mbit"]-a["dropped_Mbit"]-a["late_Mbit"]-a["backlog_Mbit"]+a["backlog_start_Mbit"]
    require(np.max(np.abs(residual)) <= 1e-7 and np.array_equal(a["backlog_start_Mbit"][1:],a["backlog_Mbit"][:-1]), "queue conservation/handover failure")
    scored = slice(warm,None)
    rr = rows.sort_values("tenant")
    require(rr.scored_intervals.eq(K-warm).all(), "row scored duration mismatch")
    for name, field in (("on_time_Mbit","on_time_Mbit"),("expired_Mbit","dropped_Mbit"),("arrivals_Mbit","arrivals_Mbit"),("late_Mbit","late_Mbit")):
        close(rr[name], a[field][scored].sum(axis=0), f"window traffic summary mismatch: {name}")
    close(rr.initial_backlog_Mbit, a["backlog_start_Mbit"][warm], "initial queue summary mismatch")
    close(rr.final_backlog_Mbit, a["backlog_Mbit"][-1], "terminal queue summary mismatch")
    for name, field in (("welfare","realized_value"),("contract_value","V_declared")):
        close(rr[name], a[field][scored].mean(axis=0), f"value summary mismatch: {name}")
    close(rr.total_delivery_welfare, np.repeat(a["realized_value"][scored].sum(axis=1).mean(),3), "repeated total welfare field mismatch")
    resolved = rr.on_time_Mbit+rr.expired_Mbit
    close(rr.resolved_byte_expiry_share.astype(float), divide(rr.expired_Mbit,resolved), "expiry summary mismatch")
    max_width = 0.
    for prefix, delivery, contract in (("price","utility_realized","utility_declared"),
                                        ("fixed_price","fixed_utility_realized","fixed_utility_declared")):
        expected_paid = np.zeros((K,3),bool)
        if spec["mode"] == "strategic" and (prefix == "price" or arm == "protected"):
            expected_paid[warm:, list(range(3)) if deviator < 0 else [deviator]] = True
        p, lo, hi = (a[prefix+s] for s in ("","_lo","_hi"))
        require(all(np.array_equal(np.isfinite(v),expected_paid) for v in (p,lo,hi)), "payment missingness mismatch")
        require(np.all(lo[expected_paid] <= p[expected_paid]+1e-10) and np.all(p[expected_paid] <= hi[expected_paid]+1e-10), "payment outside certified bracket")
        if expected_paid.any():
            max_width = max(max_width,float((hi-lo)[expected_paid].max()))
        close(a[delivery],a["realized_value"]-p,"interval delivery-utility identity failed")
        close(a[contract],a["V_declared"]-p,"interval contract-utility identity failed")
        names = ("price","delivery_utility","contract_utility") if prefix == "price" else ("fixed_price","fixed_delivery_utility","fixed_contract_utility")
        for name, field in zip(names,(prefix,delivery,contract)):
            close(rr[name].astype(float), a[field][scored].mean(axis=0), f"priced summary mismatch: {name}")
    return {"max_individual_price_bracket": max_width, "max_conservation_residual": float(np.max(np.abs(residual)))}


def reconstruct_gains(truth, deviation, spec, policy, tenant, multiplier):
    warm, scored = 40, slice(40,None)
    state, arm, seed = policies("strategic", not spec["scientific_result"])[policy]
    for field in ("quota","report","on_time_Mbit","dropped_Mbit","backlog_start_Mbit","backlog_Mbit"):
        require(np.array_equal(truth[field][:warm],deviation[field][:warm]), f"shared warm-up fails: {field}")
    require(np.array_equal(truth["backlog_start_Mbit"][warm],deviation["backlog_start_Mbit"][warm]), "unequal initial scored queues")
    rows = []
    for prefix in (("price","fixed_price") if arm == "protected" else ("price",)):
        row = {"policy":policy,"state":state,"arm":arm,"seed":seed,"deviator":tenant,"multiplier":multiplier,
               "bill":"primary" if prefix == "price" else "fixed_ablation", "shared_truthful_warmup":True,"initial_queue_equal":True}
        for kind, field in (("delivery","realized_value"),("contract","V_declared")):
            tv,dv = truth[field][scored,tenant],deviation[field][scored,tenant]
            tu = tv-truth[prefix][scored,tenant]
            du = dv-deviation[prefix][scored,tenant]
            low = dv-deviation[prefix+"_hi"][scored,tenant]-tv+truth[prefix+"_lo"][scored,tenant]
            high = dv-deviation[prefix+"_lo"][scored,tenant]-tv+truth[prefix+"_hi"][scored,tenant]
            gain = du-tu
            row.update({f"{kind}_gain":float(gain.mean()),f"{kind}_gain_lo":float(low.mean()),f"{kind}_gain_hi":float(high.mean()),
                        f"truth_{kind}_utility":float(tu.mean()),f"deviation_{kind}_utility":float(du.mean()),f"{kind}_gain_max_interval":float(gain.max())})
            require(np.all(gain >= low-1e-9) and np.all(gain <= high+1e-9), "gain outside certified bounds")
            if kind == "contract" and arm == "protected" and prefix == "price":
                tolerance = 1e-8*max(1.,float(np.abs(tv).max()),float(np.abs(dv).max()))
                require(low.max() <= tolerance, "positive Protected contract gain beyond certified uncertainty")
                row.update(full_contract_max_certified_lower_gain=float(low.max()),full_contract_numeric_tolerance=tolerance)
        honest = [i for i in range(3) if i != tenant]
        row["honest_tenants"] = honest
        row["honest_on_time_harm_Mbit_per_interval"] = (truth["on_time_Mbit"][scored][:,honest]-deviation["on_time_Mbit"][scored][:,honest]).mean(axis=0).tolist()
        row["honest_welfare_harm_per_interval"] = (truth["realized_value"][scored][:,honest]-deviation["realized_value"][scored][:,honest]).mean(axis=0).tolist()
        rows.append(row)
    return rows


def validate_case(args, verifier, folder, mode, config, profile, smoke):
    marker = json.loads((folder/"COMPLETE.json").read_text())
    result = json.loads((folder/"result.json").read_text())
    spec = result["spec"]
    sig = digest(spec)
    require(marker.get("complete") is True and result.get("complete") is True and marker.get("scientific_result") is (not smoke), "invalid case completion/scientific flag")
    require(sig == marker["spec_sha256"] == result["spec_sha256"], "case spec digest mismatch")
    require((spec["mode"],spec["config"],spec["profile"],spec["scientific_result"],spec["split"]) == (mode,config,profile,not smoke,"train" if smoke else "test"), "wrong case scope")
    require(folder.name == f"{spec['date']}__{spec['offset']}", "case folder/date mismatch")
    expected_keys = [("2024-01-01",60)] if smoke else [(d,o) for d in dates() for o in OFFSETS]
    require(set(map(tuple,spec["window_keys"])) == set(expected_keys) and len(spec["window_keys"]) == len(expected_keys), "case does not declare the full fixed window grid")
    require((spec["date"],spec["offset"]) in expected_keys, "case date outside fixed grid")
    require(tuple(spec["multipliers"]) == (MULTIPLIERS if mode == "strategic" else ()), "strategy grid changed")
    require(spec["measurement_delay_intervals"] == 2, "measurement delay changed")
    verifier.file(args.protocol,spec["protocol_sha256"])
    require(spec["origin"]==verifier.origin,"recovered input origin manifest mismatch")
    for name, expected in spec["source_sha256"].items():
        verifier.source(name,expected)
    for name, expected in spec["input_sha256"].items():
        verifier.input(name,expected)
    specification = args.root/"evaluation_specifications"/f"{sig}.json"
    require(digest(json.loads(specification.read_text())) == sig,"saved evaluation specification mismatch")
    matching = [(path,m) for path,m in verifier.manifests if sig in m["case_spec_sha256"]]
    require(any(m["mode"] == mode and m["scientific_result"] is (not smoke) and [config,profile] in m["complete_grid"] for _,m in matching), "case not covered by a planned evaluation manifest")
    p = policies(mode,smoke)
    require(set(spec["models"]) == set(p),"wrong model grid")
    for policy, (state,arm,seed) in p.items():
        entry = spec["models"][policy]
        require((entry["state"],entry["arm"],entry["seed"]) == (state,arm,seed),"policy/model metadata mismatch")
        verifier.model(entry,spec,smoke)
    expected_runs = runs(mode,smoke)
    expected_files = {"result.json"} | {interval_name(*r) for r in expected_runs}
    require(set(marker["files_sha256"]) == expected_files,"case artifact manifest is incomplete/extra")
    require({f.relative_to(folder).as_posix() for f in folder.rglob("*") if f.is_file()} == expected_files | {"COMPLETE.json"},"case has extra/missing artifacts")
    for name, expected in marker["files_sha256"].items():
        verifier.file(folder/name,expected)
    require(marker["trajectory_count"] == spec["trajectories_per_case"] == len(expected_runs),"trajectory count mismatch")
    base = {k:v for k,v in spec.items() if k not in ("date","offset")}
    cell_key = (mode,config,profile)
    if cell_key in verifier.case_signatures:
        require(verifier.case_signatures[cell_key] == digest(base),"mixed specifications within a cell")
    verifier.case_signatures[cell_key] = digest(base)
    window = result["window"]
    require((window["K"],window["warmup"],window["split"]) == (48 if smoke else 200,40,"train" if smoke else "test"),"wrong window length/split")
    require(window["target_load"] == CONFIGS[config] and np.isfinite(window["actual_scored_offered_load"]),"invalid target/actual load")
    shape_path = args.frozen_study/"shapes"/config/f"{profile}.json"
    verifier.file(shape_path,spec["shape_sha256"])
    shape = json.loads(shape_path.read_text())
    require(shape["provenance"].get("fit_split")=="train" and shape["provenance"].get("training_window_count")==180,"service shape was not fitted on original TRAIN grid")
    require(digest(shape["contract"]) == window["contract_sha256"],"window/shape contract mismatch")
    cal = json.loads((args.assets/spec["calibration_relative_path"]).read_text())
    mechanism = cal["meta"]["mech"]
    require(window["B"] == mechanism["B"],"capacity mismatch")
    rows = pd.DataFrame(result["rows"])
    rowkeys = ["policy","deviator","multiplier","tenant"]
    require(not rows.duplicated(rowkeys).any() and set(rows[rowkeys].itertuples(index=False,name=None)) == {(*r,i) for r in expected_runs for i in range(3)},"tenant row grid incomplete/duplicated")
    for field,value in (("mode",mode),("config",config),("profile",profile),("date",spec["date"]),("offset",spec["offset"])):
        require(rows[field].eq(value).all(),"row/case metadata mismatch")
    require(all(TRAFFIC[i] == name for i,name in zip(rows.tenant,rows.traffic)),"traffic/tenant mismatch")
    records = {(x["policy"],x["deviator"],x["multiplier"]):x for x in result["trajectories"]}
    require(len(records) == len(result["trajectories"]) == len(expected_runs) and set(records) == set(expected_runs),"trajectory metadata grid mismatch")
    saved_gains = {(x["policy"],x["deviator"],x["multiplier"],x["bill"]):x for x in result["paired_gains"]}
    require(len(saved_gains) == len(result["paired_gains"]),"duplicate paired-gain row")
    reconstructed, invariant_rows = [], []
    baseline = {}
    for policy,dev,mult in expected_runs:
        path = folder/interval_name(policy,dev,mult)
        with np.load(path,allow_pickle=False) as archive:
            a = {name:archive[name] for name in archive.files}
        rr = rows[rows.policy.eq(policy)&rows.deviator.eq(dev)&rows.multiplier.eq(mult)]
        state,arm,seed = p[policy]
        require(rr.state.eq(state).all() and rr.arm.eq(arm).all() and rr.seed.eq(seed).all(),"row/policy labels mismatch")
        checks = validate_arrays(a,rr,spec,window,shape,mechanism,policy,dev,mult)
        native = records[(policy,dev,mult)]["checks"]
        require(native["pricing_quota_matches"] is True and native["budget_excess_PRB"] == 0 and native["handover_max_Mbit"] == 0 and native["price_bracket_violation"] <= 1e-10,"native invariant failed")
        close(native["scored_boundary_conservation_Mbit"],np.zeros(3),"native boundary conservation failed",atol=1e-7)
        invariant_rows.append({**{k:spec[k] for k in ("config","profile","date","offset")},"mode":mode,"policy":policy,"deviator":dev,"multiplier":mult,**checks})
        if dev < 0:
            baseline[policy] = a
            ck = (config,profile,spec["date"],spec["offset"],policy)
            comparable = {name:array_sha(a[name]) for name in ("quota","on_time_Mbit","dropped_Mbit","backlog_start_Mbit","backlog_Mbit","theta","report")}
            comparable.update(input_arrays=window["arrays_sha256"],contract=window["contract_sha256"],model=digest(spec["models"][policy]),shape=spec["shape_sha256"])
            if ck in verifier.cross_mode:
                require(verifier.cross_mode[ck] == comparable,"truth/strategic baseline replay or input mismatch")
            verifier.cross_mode[ck] = comparable
        else:
            for gain in reconstruct_gains(baseline[policy],a,spec,policy,dev,mult):
                gk = (policy,dev,mult,gain["bill"])
                require(gk in saved_gains,"missing saved paired gain")
                saved = saved_gains.pop(gk)
                for name,value in gain.items():
                    if isinstance(value,(str,bool)):
                        require(saved.get(name) == value,f"paired gain metadata mismatch: {name}")
                    else:
                        close(saved[name],value,f"paired gain reconstruction mismatch: {name}")
                reconstructed.append({**{k:spec[k] for k in ("config","profile","date","offset")},"target_load":window["target_load"],"traffic":TRAFFIC[dev],**gain})
    require(not saved_gains,"unexpected saved paired gains")
    rows["target_load"] = window["target_load"]
    record = {"mode":mode,"config":config,"profile":profile,"date":spec["date"],"offset":spec["offset"],
              "target_load":window["target_load"],"actual_scored_offered_load":window["actual_scored_offered_load"],
              "case_spec_sha256":sig,"completion_sha256":sha(folder/"COMPLETE.json"),"result_sha256":sha(folder/"result.json"),
              "trajectories":len(expected_runs),"paired_rows":len(reconstructed),"scientific_result":not smoke}
    return rows,pd.DataFrame(reconstructed),pd.DataFrame(invariant_rows),record


def load_cell(args, verifier, config, profile, mode, smoke=False):
    folder = args.root/("smoke_evaluation" if smoke else "evaluation")/mode/config/profile
    keys = [("2024-01-01",60)] if smoke else [(d,o) for d in dates() for o in OFFSETS]
    expected = {f"{d}__{o}" for d,o in keys}
    available = {p.name:p for p in folder.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}__\d+",p.name)} if folder.exists() else {}
    require(set(available) <= expected,f"unplanned completed window in {folder}")
    complete = set(available) == expected
    if not complete and not args.allow_partial:
        raise ValueError(f"{mode}/{config}/{profile}: {len(available)}/{len(expected)} cases; incomplete cells cannot be analyzed")
    collected = [validate_case(args,verifier,available[name],mode,config,profile,smoke) for name in sorted(available)]
    frames = [pd.concat([r[i] for r in collected],ignore_index=True) if collected else pd.DataFrame() for i in range(3)]
    cases = pd.DataFrame([r[3] for r in collected])
    status = {"mode":mode,"config":config,"profile":profile,"complete":complete,"scientific_result":not smoke,
              "expected_cases":len(expected),"validated_cases":len(available),"missing_cases":sorted(expected-set(available))}
    return (*frames,cases,status)


def divide(a,b):
    a,b = np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)


class Bootstrap:
    def __init__(self,repetitions=2000):
        self.repetitions = repetitions
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        self.di = rng.integers(16,size=(repetitions,16,1))
        self.wi = rng.integers(2,size=(repetitions,16,2))

    def means(self,frame,columns):
        require(not frame.duplicated(WINDOW).any() and set(frame[WINDOW].itertuples(index=False,name=None)) == {(d,o) for d in dates() for o in OFFSETS},"bootstrap requires exact 16-date/two-window grid")
        v = frame.sort_values(WINDOW)[columns].to_numpy(float).reshape(16,2,len(columns))
        require(np.isfinite(v).all(),"nonfinite bootstrap input")
        return v.mean(axis=(0,1)),v[self.di,self.wi].mean(axis=(1,2))


def describe(averaged,raw,groups,columns,metrics,boot):
    records = []
    for key,g in averaged.groupby(groups,sort=True,dropna=False):
        key = key if isinstance(key,tuple) else (key,)
        meta = dict(zip(groups,key))
        point,samples = boot.means(g,columns)
        mask = np.ones(len(raw),bool)
        for name,value in meta.items():
            mask &= raw[name].eq(value).to_numpy()
        original = raw.loc[mask]
        seed_means = {int(seed):x[columns].mean().to_numpy(float) for seed,x in original.groupby("seed")}
        for metric,(fn,unit) in metrics.items():
            estimate = float(fn(point))
            draws = np.asarray(fn(samples),float)
            finite = draws[np.isfinite(draws)]
            low,high = np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
            seed_estimates = {str(seed):float(fn(x)) for seed,x in seed_means.items()}
            seeds = [value for value in seed_estimates.values() if np.isfinite(value)]
            records.append({**meta,"metric":metric,"estimate":estimate,"ci_low":low,"ci_high":high,"unit":unit,
                "n_dates":16,"n_windows":32,"n_fitted_seeds":0 if set(original.seed)=={-1} else len(seed_means),
                "seed_mean_min":min(seeds) if seeds else np.nan,"seed_mean_max":max(seeds) if seeds else np.nan,
                "seed_aggregate_estimates":seed_estimates,
                "finite_bootstrap_replicates":len(finite)})
    return pd.DataFrame(records)


def matched_pairs(raw,left,right,keys,columns):
    a = raw[raw.policy_group.eq(left)].drop(columns="policy_group")
    b = raw[raw.policy_group.eq(right)].drop(columns="policy_group")
    ad,bd = set(a.seed)=={-1},set(b.seed)=={-1}
    require(not a.empty and not b.empty,"missing contrast policy")
    if ad and not bd:
        a = a.drop(columns="seed")
    if bd and not ad:
        b = b.drop(columns="seed")
    join = keys+([] if ad != bd else ["seed"])
    out = a.merge(b,on=join,suffixes=("_left","_right"),validate="one_to_many" if ad and not bd else "many_to_one" if bd and not ad else "one_to_one")
    require(len(out) == max(len(a),len(b)),"unmatched contrast rows")
    return out


def truth_tables(rows,boot):
    raw = rows.copy()
    for new,old in (("on_time","on_time_Mbit"),("expired","expired_Mbit"),("arrivals","arrivals_Mbit"),("late","late_Mbit")):
        raw[new] = raw[old]/raw.scored_intervals
    raw["policy_group"] = raw.state+"__"+raw.arm
    columns = ["welfare","on_time","expired","arrivals","late","initial_backlog_Mbit","final_backlog_Mbit"]
    group = KEY+["policy_group","seed"]+WINDOW
    total = raw.groupby(group,as_index=False)[columns].sum();total["traffic"]="all"
    raw = pd.concat([raw[group+["traffic"]+columns],total],ignore_index=True)
    averaged = raw.groupby(KEY+["policy_group","traffic"]+WINDOW,as_index=False)[columns].mean()
    metrics = {"delivery_welfare":(lambda a:a[...,0],"credits/scored interval"),
        "on_time_volume":(lambda a:a[...,1],"Mbit/scored interval"),
        "on_time_window_volume":(lambda a:a[...,1]*160,"Mbit/scored window"),
        "expired_volume":(lambda a:a[...,2],"Mbit/scored interval"),
        "resolved_byte_expiry_share":(lambda a:divide(a[...,2],a[...,1]+a[...,2]),"fraction of pooled resolved bytes"),
        "arrival_volume":(lambda a:a[...,3],"Mbit/scored interval"),
        "late_volume":(lambda a:a[...,4],"Mbit/scored interval"),
        "initial_backlog":(lambda a:a[...,5],"Mbit"),"final_backlog":(lambda a:a[...,6],"Mbit")}
    summary = describe(averaged,raw,KEY+["policy_group","traffic"],columns,metrics,boot)
    contrasts = []
    for arm in (*ARMS,"static"):
        contrasts.append(("within_arm_training",f"retrained__{arm}",f"frozen__{arm}"))
    groups = sorted(raw.policy_group.unique())
    for state in ("frozen","retrained"):
        left = f"{state}__protected"
        contrasts.extend(("protected_vs_baseline",left,right) for right in groups if right != left)
    paired = []
    keys = KEY+["traffic"]+WINDOW
    for kind,left,right in contrasts:
        p = matched_pairs(raw,left,right,keys,columns)
        p["contrast_kind"],p["left"],p["right"] = kind,left,right
        paired.append(p)
    paired = pd.concat(paired,ignore_index=True)
    paircols = [c+s for s in ("_left","_right") for c in columns]
    pairgroups = KEY+["contrast_kind","left","right","traffic"]
    av = paired.groupby(pairgroups+WINDOW,as_index=False)[paircols].mean()
    n = len(columns)
    cm = {"welfare_difference":(lambda a:a[...,0]-a[...,n],"credits/scored interval; left minus right"),
        "relative_welfare_difference":(lambda a:divide(a[...,0]-a[...,n],a[...,n]),"fraction of right mean welfare"),
        "on_time_difference":(lambda a:a[...,1]-a[...,n+1],"Mbit/scored interval; left minus right"),
        "expiry_share_difference":(lambda a:divide(a[...,2],a[...,1]+a[...,2])-divide(a[...,n+2],a[...,n+1]+a[...,n+2]),"pooled expiry-fraction difference; left minus right")}
    comparisons = describe(av,paired,pairgroups,paircols,cm,boot)
    gap = matched_pairs(raw,"retrained__protected","frozen__protected",keys,columns)
    direct = raw[raw.policy_group.eq("reference__direct")][keys+["welfare"]].rename(columns={"welfare":"direct_welfare"})
    gap = gap.merge(direct,on=keys,validate="many_to_one")
    gc = ["welfare_left","welfare_right","direct_welfare"]
    ga = gap.groupby(KEY+["traffic"]+WINDOW,as_index=False)[gc].mean()
    gm = {"protected_minus_direct_before":(lambda a:a[...,1]-a[...,2],"credits/scored interval"),
          "protected_minus_direct_after":(lambda a:a[...,0]-a[...,2],"credits/scored interval"),
          "change_in_protected_minus_direct_gap":(lambda a:a[...,0]-a[...,1],"credits/scored interval; algebraically the Protected retraining effect"),
          "relative_gap_before":(lambda a:divide(a[...,1]-a[...,2],a[...,2]),"fraction of Direct mean welfare"),
          "relative_gap_after":(lambda a:divide(a[...,0]-a[...,2],a[...,2]),"fraction of Direct mean welfare"),
          "relative_gap_change":(lambda a:divide(a[...,0]-a[...,1],a[...,2]),"fraction of Direct mean welfare")}
    return {"truth_observations":rows,"truth_seed_averaged_windows":averaged,"truth_summary":summary,
            "truth_comparison_windows":av,"truth_comparisons":comparisons,
            "direct_gap_before_after":describe(ga,gap,KEY+["traffic"],gc,gm,boot)}


def strategic_tables(pairs,boot):
    pairs = pairs.copy()
    for kind in ("delivery","contract"):
        positive = pairs[f"truth_{kind}_utility"] > 0
        pairs[f"{kind}_positive_fraction"] = positive.astype(float)
        pairs[f"{kind}_eligible_gain"] = pairs[f"{kind}_gain"].where(positive,0.)
        pairs[f"{kind}_eligible_truth"] = pairs[f"truth_{kind}_utility"].where(positive,0.)
    pairs["honest_on_time_loss"] = pairs.honest_on_time_harm_Mbit_per_interval.map(sum)
    pairs["honest_welfare_loss"] = pairs.honest_welfare_harm_per_interval.map(sum)
    columns = [f"{k}_{m}" for k in ("delivery","contract") for m in ("gain","gain_lo","gain_hi","positive_fraction","eligible_gain","eligible_truth")]
    columns += ["truth_delivery_utility","truth_contract_utility","honest_on_time_loss","honest_welfare_loss"]
    groups = KEY+["state","arm","bill","deviator","traffic","multiplier"]
    averaged = pairs.groupby(groups+WINDOW,as_index=False)[columns].mean()
    metrics = {}
    for j,kind in enumerate(("delivery","contract")):
        i = 6*j
        metrics[f"{kind}_utility_gain"] = (lambda a,i=i:a[...,i],"credits/scored interval; positive favors deviation")
        metrics[f"{kind}_gain_numerical_lower"] = (lambda a,i=i:a[...,i+1],"credits/scored interval; certified-payment lower bound")
        metrics[f"{kind}_gain_numerical_upper"] = (lambda a,i=i:a[...,i+2],"credits/scored interval; certified-payment upper bound")
        metrics[f"{kind}_positive_truth_fraction"] = (lambda a,i=i:a[...,i+3],"fraction of positive-truth seed/window pairs")
        metrics[f"{kind}_relative_gain_positive_truth"] = (lambda a,i=i:divide(a[...,i+4],a[...,i+5]),"eligible summed gains / eligible summed truthful utility")
        metrics[f"{kind}_positive_truth_mean_denominator"] = (lambda a,i=i:divide(a[...,i+5],a[...,i+3]),"credits/scored interval among positive-truth pairs")
    metrics.update(delivery_truth_utility=(lambda a:a[...,12],"credits/scored interval"),
        contract_truth_utility=(lambda a:a[...,13],"credits/scored interval"),
        honest_others_on_time_loss=(lambda a:a[...,14],"Mbit/scored interval; positive harms honest tenants"),
        honest_others_welfare_loss=(lambda a:a[...,15],"credits/scored interval; positive harms honest tenants"))
    summary = describe(averaged,pairs,groups,columns,metrics,boot)
    counts = pairs.groupby(groups,as_index=False).agg(n_seed_window_pairs=("seed","size"),
        delivery_positive_truth_pairs=("delivery_positive_fraction","sum"),contract_positive_truth_pairs=("contract_positive_fraction","sum"),
        positive_delivery_instances=("delivery_gain",lambda x:int((x>GAIN_POSITIVE).sum())),
        certified_positive_delivery_instances=("delivery_gain_lo",lambda x:int((x>GAIN_POSITIVE).sum())))
    summary = summary.merge(counts,on=groups,validate="many_to_one")
    comparison_keys = KEY+["arm","bill","deviator","traffic","multiplier","seed"]+WINDOW
    new = pairs[pairs.state.eq("retrained")][comparison_keys+columns]
    old = pairs[pairs.state.eq("frozen")][comparison_keys+columns]
    change = new.merge(old,on=comparison_keys,suffixes=("_new","_old"),validate="one_to_one")
    require(len(change)==len(new)==len(old),"incomplete frozen/retrained strategy pairing")
    cc = [c+s for s in ("_new","_old") for c in columns]
    cg = KEY+["arm","bill","deviator","traffic","multiplier"]
    ca = change.groupby(cg+WINDOW,as_index=False)[cc].mean();n=len(columns)
    cm = {"change_in_delivery_gain":(lambda a:a[...,0]-a[...,n],"credits/scored interval; retrained minus frozen"),
          "change_in_contract_gain":(lambda a:a[...,6]-a[...,n+6],"credits/scored interval; retrained minus frozen"),
          "change_in_delivery_gain_numerical_lower":(lambda a:a[...,1]-a[...,n+2],"credits/scored interval; certified-payment lower bound"),
          "change_in_delivery_gain_numerical_upper":(lambda a:a[...,2]-a[...,n+1],"credits/scored interval; certified-payment upper bound"),
          "change_in_contract_gain_numerical_lower":(lambda a:a[...,7]-a[...,n+8],"credits/scored interval; certified-payment lower bound"),
          "change_in_contract_gain_numerical_upper":(lambda a:a[...,8]-a[...,n+7],"credits/scored interval; certified-payment upper bound"),
          "change_in_relative_delivery_gain":(lambda a:divide(a[...,4],a[...,5])-divide(a[...,n+4],a[...,n+5]),"difference of positive-baseline normalized gains; eligibility can differ"),
          "change_in_honest_welfare_loss":(lambda a:a[...,15]-a[...,n+15],"credits/scored interval; retrained minus frozen")}
    victims = []
    for row in pairs.to_dict("records"):
        for i,ontime,welfare in zip(row["honest_tenants"],row["honest_on_time_harm_Mbit_per_interval"],row["honest_welfare_harm_per_interval"]):
            victims.append({**{k:row[k] for k in groups+["seed"]+WINDOW},"victim":i,"victim_traffic":TRAFFIC[i],"on_time_loss":ontime,"welfare_loss":welfare})
    victims = pd.DataFrame(victims);vg=groups+["victim","victim_traffic"]
    va=victims.groupby(vg+WINDOW,as_index=False)[["on_time_loss","welfare_loss"]].mean()
    vm={"on_time_loss":(lambda a:a[...,0],"Mbit/scored interval; positive harms honest tenant"),
        "welfare_loss":(lambda a:a[...,1],"credits/scored interval; positive harms honest tenant")}
    return {"strategic_paired_observations":pairs,"strategic_seed_averaged_pairs":averaged,
            "strategic_summary":summary,"strategic_training_effects":describe(ca,change,cg,cc,cm,boot),
            "strategic_victim_summary":describe(va,victims,vg,["on_time_loss","welfare_loss"],vm,boot)}


def instance_diagnostics(pairs):
    fields=KEY+WINDOW+["state","arm","bill","seed","deviator","traffic","multiplier","delivery_gain","delivery_gain_lo","delivery_gain_hi","truth_delivery_utility","contract_gain"]
    records=[]
    groups=KEY+["state","arm","bill","deviator","traffic","multiplier"]
    for _,g in pairs.groupby(groups):
        positive=g[g.truth_delivery_utility>0].copy()
        positive["relative_gain"]=positive.delivery_gain/positive.truth_delivery_utility
        records.append({**{k:g.iloc[0][k] for k in groups},"instances":len(g),
            "positive_delivery_instances":int((g.delivery_gain>GAIN_POSITIVE).sum()),
            "certified_positive_delivery_instances":int((g.delivery_gain_lo>GAIN_POSITIVE).sum()),
            "positive_truth_denominators":len(positive),
            "maximum_absolute_instance":g.loc[g.delivery_gain.idxmax(),fields].to_dict(),
            "maximum_relative_instance":None if positive.empty else positive.loc[positive.relative_gain.idxmax(),fields+["relative_gain"]].to_dict()})
    return records


def write_report(args,cell,modes,loaded,boot,verifier):
    config,profile=cell
    tables={};status=[]
    for mode in modes:
        rows,pairs,invariants,cases,s=loaded[(config,profile,mode)]
        status.append(s)
        tables.update(truth_tables(rows,boot) if mode=="truth" else strategic_tables(pairs,boot))
        tables[f"{mode}_cases"]=cases;tables[f"{mode}_invariants"]=invariants
        if mode=="strategic":tables["strategic_observations"]=rows
    for name,frame in list(tables.items()):
        if "seed_aggregate_estimates" not in frame:
            continue
        per_seed = []
        excluded = {"seed_aggregate_estimates", "estimate", "ci_low", "ci_high",
                    "seed_mean_min", "seed_mean_max", "finite_bootstrap_replicates"}
        for row in frame.to_dict("records"):
            for seed,value in row["seed_aggregate_estimates"].items():
                per_seed.append({**{k:v for k,v in row.items() if k not in excluded},
                                 "seed":int(seed),"estimate":value})
        tables[f"{name}_per_seed"] = pd.DataFrame(per_seed)
    findings={"created_utc":datetime.now(timezone.utc).isoformat(),"scientific_result":True,
        "scope":{"config":config,"profile":profile,"modes":modes,"all_nine_cells_complete_claim":False},
        "completion":status,"analysis_sha256":sha(__file__),"protocol_sha256":sha(args.protocol),
        "positive_gain_threshold_credits_per_interval":GAIN_POSITIVE,
        "verified_source_sha256":verifier.sources,"verified_recovered_origin":verifier.origin,
        "verified_manifest_path_translations":verifier.path_translations,
        "bootstrap":{"replicates":boot.repetitions,"seed":BOOTSTRAP_SEED,"interval":"percentile95%",
            "method":"average matched fitted seeds; resample dates then windows within dates","conditional_on_three_fitted_seeds":True},
        "limits":["Reused TEST dates; exploratory and not multiplicity-adjusted.",
            "No best-per-window multiplier or adaptive strategy is constructed.",
            "Instance maxima and positive counts describe correlated observations, not aggregate strategy performance.",
            "Positive numerical gain bounds and statistical confidence intervals are different quantities.",
            "Negative mean delivery gains do not establish practical strategyproofness.",
            "The Direct gap change is algebraically the Protected training effect, not independent evidence.",
            "Expiry is a pooled resolved-byte fraction, not packet reliability or an arrival-cohort rate.",
            "Relative utility includes only positive own-truth seed/window denominators; absolute gains retain all pairs."]}
    addendum=args.root/"SIX_HOUR_ADDENDUM.txt"
    if addendum.is_file():
        findings["resource_addendum_sha256"]=sha(addendum)
    findings["actual_scored_offered_load"]={mode:tables[f"{mode}_cases"].actual_scored_offered_load.agg(["mean","min","max"]).to_dict() for mode in modes}
    findings["invariants"]={mode:{c:float(tables[f"{mode}_invariants"][c].max()) for c in ("max_individual_price_bracket","max_conservation_residual")} for mode in modes}
    if "truth" in modes:
        findings["welfare_means"]=tables["truth_summary"].query("traffic == 'all' and metric == 'delivery_welfare'").to_dict("records")
    if "strategic" in modes:
        pairs=tables["strategic_paired_observations"]
        findings["individual_contexts_not_adaptive_strategies"]=instance_diagnostics(pairs)
        summary=tables["strategic_summary"]
        largest=[]
        for _,g in summary[summary.metric.isin(["delivery_utility_gain","contract_utility_gain"])].groupby(["state","arm","bill","metric"]):
            largest.append(g.loc[g.estimate.idxmax()].to_dict())
        findings["largest_aggregate_fixed_multiplier_gains_descriptive"]=largest
        findings["maximum_mean_gain_bracket_width"]=float((pairs.delivery_gain_hi-pairs.delivery_gain_lo).max())
    output=args.output/config/profile/args.mode
    output.mkdir(parents=True,exist_ok=True)
    table_hashes = {}
    for name,frame in tables.items():
        frame=frame.copy()
        for column in frame.columns:
            if frame[column].map(lambda x:isinstance(x,(list,dict,np.ndarray))).any():
                frame[column]=frame[column].map(lambda x:json.dumps(clean(x)) if isinstance(x,(list,dict,np.ndarray)) else x)
        frame.to_csv(output/f"{name}.csv",index=False)
        frame.to_parquet(output/f"{name}.parquet",index=False)
        for extension in ("csv", "parquet"):
            filename = f"{name}.{extension}"
            table_hashes[filename] = sha(output / filename)
    findings["table_files_sha256"] = table_hashes
    (output/"findings.json").write_text(json.dumps(clean(findings),indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"report":str(output),"complete_cell":True,"modes":modes,"table_rows":{n:len(f) for n,f in tables.items()}}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--assets",type=Path,default=ROOT.parent/"deadline_load_study/upstream")
    p.add_argument("--frozen-study",type=Path,default=ROOT.parent/"deadline_load_study")
    p.add_argument("--protocol",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--model-root",type=Path,help="relocated model root; default ROOT/models (ROOT/smoke for diagnostics)")
    p.add_argument("--mode",choices=("truth","strategic","all"),default="all")
    p.add_argument("--cell",action="append",help="CONFIG/PROFILE, repeatable; defaults to default/both")
    p.add_argument("--all",action="store_true",help="require all nine scientific cells complete")
    p.add_argument("--allow-partial",action="store_true",help="console-only validation/status; never produce research estimates")
    p.add_argument("--diagnostic-smoke",action="store_true",help="validate separate TRAIN smoke only; never produce research estimates")
    p.add_argument("--bootstrap",type=int,default=2000)
    args=p.parse_args()
    for name in ("root","assets","frozen_study"):
        setattr(args,name,getattr(args,name).expanduser().resolve())
    args.protocol=(args.protocol or args.root/"PROTOCOL.txt").expanduser().resolve()
    args.output=(args.output or args.root/"analysis_matched").expanduser().resolve()
    if args.model_root:
        args.model_root=args.model_root.expanduser().resolve()
    require(not any(args.output.is_relative_to(x) for x in (args.assets,args.frozen_study,args.root/"evaluation",args.root/"models")),"analysis output must be separate from source assets, raw evaluation and model directories")
    require(not(args.all and args.cell),"choose --all or --cell")
    require(args.bootstrap>=2000,"at least2000 bootstrap replicates required")
    cells=[(c,pr) for c in CONFIGS for pr in PROFILES] if args.all else [tuple(x.split("/")) for x in (args.cell or ["default/both"])]
    require(len(cells)==len(set(cells)) and all(len(x)==2 and x[0] in CONFIGS and x[1] in PROFILES for x in cells),"invalid/duplicate cells")
    if args.diagnostic_smoke:
        require(cells==[("default","both")],"diagnostic smoke is fixed to default/both")
    modes=("truth","strategic") if args.mode=="all" else (args.mode,)
    verifier=Verifier(args);loaded={}
    # Validate ALL requested cells before emitting any scientific report.
    for cell in cells:
        for mode in modes:
            loaded[(*cell,mode)]=load_cell(args,verifier,*cell,mode,args.diagnostic_smoke)
    statuses=[x[4] for x in loaded.values()]
    if args.allow_partial or args.diagnostic_smoke:
        diagnostic={"scientific_estimates_written":False,"diagnostic_smoke":args.diagnostic_smoke,
                    "status":statuses,"verified_file_count":len(verifier.checked),"analysis_sha256":sha(__file__)}
        if args.diagnostic_smoke:
            output=args.root/"analysis_diagnostics";output.mkdir(parents=True,exist_ok=True)
            (output/"smoke_validation.json").write_text(json.dumps(diagnostic,indent=2)+"\n")
        print(json.dumps(diagnostic,indent=2));return
    boot=Bootstrap(args.bootstrap)
    for cell in cells:
        write_report(args,cell,modes,loaded,boot,verifier)


if __name__=="__main__":
    try:
        main()
    except (ValueError,KeyError,FileNotFoundError) as error:
        print(f"ANALYSIS REFUSED: {error}",file=sys.stderr)
        raise SystemExit(2)
