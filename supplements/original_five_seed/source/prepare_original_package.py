#!/usr/bin/env python3
"""Stage portable evidence for the completed original-setting five-seed extension.

Never uploads or publishes. --skeleton stages source/instructions only.
Scientific packaging requires the complete original campaign, unchanged analysis
and an explicitly supplied SHA-256 of the findings approved for packaging.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCES = ("run_original_supplement.py", "analyze_original_multiseed.py", "reanalyze_original_release.py",
           "prepare_original_package.py", "PROTOCOL.txt", "ANALYSIS_README.txt", "PORTABLE_README.txt")
PRIVATE = {
    "private filesystem locator": rb"/(?:home|Users)/[^\s\"'\\]+",
    "private sharing URL": rb"https?://(?:drive\.google\.com|drive\.usercontent\.google\.com)/[^\s\"']+",
    "private IPv4 address": rb"(?<![0-9])(?:10\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.[0-9]+\.[0-9]+|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+)(?![0-9])",
    "private key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "GitHub credential": rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
    "credential in URL": rb"https?://[^/\s:@]+:[^/\s@]+@",
    "literal credential": rb"(?i)[\"']?(?:password|passwd|api_key|secret_key|access_token|auth_token)[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_+/.=-]{12,}[\"']",
}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def module(path):
    spec = importlib.util.spec_from_file_location("package_original_analyzer", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return m


def privacy_check(path):
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        parts = [str(x).encode() for c in frame.select_dtypes(include=["object", "string"]).columns
                 for x in frame[c].dropna().unique()]
        data = b"\n".join(parts)
    else:
        data = path.read_bytes()
    flags = [name for name, pattern in PRIVATE.items() if re.search(pattern, data)]
    require(not flags, "Private metadata detected in " + path.name + ": " + ", ".join(flags))


def copy_checked(source, dest, expected=None):
    require(source.is_file() and not source.is_symlink() and not dest.exists(), "Unsafe/existing copy destination")
    digest = sha(source); require(expected is None or digest == expected, "Source changed: " + source.name)
    privacy_check(source); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest)
    require(sha(dest) == digest and sha(source) == digest, "File changed while copying")


def stage_sources(output):
    records = []
    for name in SOURCES:
        p = HERE / name; target = output / "source" / name; copy_checked(p, target)
        records.append(dict(path="source/" + name, bytes=p.stat().st_size, sha256=sha(p), transformation="none"))
    write(output / "SOURCE_MANIFEST.json", dict(source_bytes_preserved=True, files=records))


def portable_specification(spec, original_file_sha256, canonical_sha256):
    # Only nonfunctional locators are removed. No grid, code/input/model hash,
    # measured outcome, analysis method or original case record is altered.
    clean = json.loads(json.dumps(spec)); removed = []
    for key in sorted(clean["models"]):
        old = clean["models"][key].pop("path")
        pointer = "/models/" + key.replace("~", "~0").replace("/", "~1") + "/path"
        removed.append(dict(pointer=pointer, reason="private filesystem locator", value_sha256=hashlib.sha256(old.encode()).hexdigest()))
    old = clean["native_test_gate"].pop("path")
    removed.append(dict(pointer="/native_test_gate/path", reason="private filesystem locator",
                        value_sha256=hashlib.sha256(old.encode()).hexdigest()))
    if "repository" in clean["origin"]:
        old = clean["origin"].pop("repository")
        removed.append(dict(pointer="/origin/repository", reason="original repository locator omitted; revision and content hashes retained",
                            value_sha256=hashlib.sha256(old.encode()).hexdigest()))
    return dict(schema="original-five-seed-portable-specification-v1", specification=clean,
                original_file_sha256=original_file_sha256, original_canonical_sha256=canonical_sha256,
                transformations=removed,
                public_baseline="https://github.com/dibbend8/truthful-ran-slicing/releases/tag/v0.2.0",
                note="Explicit derivative of the immutable original specification. Historical case hashes refer to the original canonical digest; deleted locators cannot be reconstructed from this public derivative.")


def skeleton(output):
    require(not output.exists(), "Refusing to overwrite staging directory")
    output.mkdir(parents=True); stage_sources(output)
    (output / "README.md").write_text("# Five-seed original-setting supplement: source staging\n\n"
        "This directory contains portable source and instructions only. No scientific results or completed release are staged. "
        "The immutable analyzer still requires all 160 cases and 800 trajectories. "
        "After completion and review, create a new child or sibling directory with the packaging command in source/PORTABLE_README.txt.\n")
    write(output / "STAGING_STATUS.json", dict(complete=False, scientific_result=False, source_staging_complete=True,
          results_staged=False, source_manifest_sha256=sha(output / "SOURCE_MANIFEST.json"), publication_performed=False))
    print(json.dumps(dict(source_staging_complete=True, scientific_result=False, results_staged=False)))


def archive_payload(payload, archive):
    require(not archive.exists(), "Archive exists")
    entries = {}
    with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|") as tar:
            for p in sorted(payload.rglob("*")):
                if not p.is_file(): continue
                name = p.relative_to(payload).as_posix(); data = p.read_bytes()
                info = tarfile.TarInfo(name); info.size = len(data); info.mtime = 0; info.mode = 0o644
                tar.addfile(info, io.BytesIO(data)); entries[name] = hashlib.sha256(data).hexdigest()
    with tarfile.open(archive, "r:gz") as tar:
        seen = set()
        for member in tar:
            require(member.isfile() and member.name in entries, "Unexpected archive member")
            require(hashlib.sha256(tar.extractfile(member).read()).hexdigest() == entries[member.name],
                    "Archive member readback differs")
            seen.add(member.name)
        require(seen == set(entries), "Archive members missing")
    return dict(bytes=archive.stat().st_size, sha256=sha(archive), members=len(entries), all_members_readback_verified=True)


def package(args):
    require(not args.output.exists(), "Refusing to overwrite package output")
    findings_file = args.analysis / "findings.json"
    require(sha(findings_file) == args.approved_findings_sha256, "Findings differ from the explicitly approved hash")
    findings = read(findings_file)
    require(findings["complete"] and findings["scientific_result"] and findings["exploratory"]
            and findings["ready_for_manuscript"], "Analysis is incomplete or seed0 replication needs review")
    analyzer = module(HERE / "analyze_original_multiseed.py")
    spec, canonical = analyzer.verify_spec(argparse.Namespace(campaign=args.campaign, assets=args.assets,
            frozen_support=args.frozen_support, runner=HERE / "run_original_supplement.py"))
    require(findings["analysis_sha256"] == sha(HERE / "analyze_original_multiseed.py")
            and findings["specification_canonical_sha256"] == canonical
            and findings["specification_sha256"] == sha(args.campaign / "SPECIFICATION.json"), "Analysis/source binding differs")
    require(findings["source_sha256"] == spec["source_sha256"] and findings["input_sha256"] == spec["input_sha256"],
            "Analysis provenance differs")
    payload = args.output / "payload"; payload.mkdir(parents=True); stage_sources(payload)
    require(sha(payload / "source/PROTOCOL.txt") == spec["protocol_sha256"], "Copied protocol differs")
    portable = portable_specification(spec, sha(args.campaign / "SPECIFICATION.json"), canonical)
    write(payload / "SPECIFICATION_PORTABLE.json", portable); privacy_check(payload / "SPECIFICATION_PORTABLE.json")
    copy_checked(findings_file, payload / "analysis/findings.json", args.approved_findings_sha256)
    for name, h in findings["table_files_sha256"].items():
        require(len(analyzer.relative(name).parts) == 1, "Unexpected analysis table path")
        copy_checked(args.analysis / name, payload / "analysis" / name, h)
    copy_checked(args.analysis / "CASE_VERIFICATION.json", payload / "analysis/CASE_VERIFICATION.json",
                 findings["case_verification_sha256"])
    cases = read(args.analysis / "CASE_VERIFICATION.json")
    expected_cases = {(s, d, o) for d, o in analyzer.KEYS for s in analyzer.SEEDS}
    require(len(cases) == 160 and {(r["seed"], r["date"], r["offset"]) for r in cases} == expected_cases,
            "Incomplete/duplicate analyzed case grid")
    omitted = []
    for r in cases:
        src = analyzer.case_dir(args.campaign, r["seed"], r["date"], r["offset"])
        dest = analyzer.case_dir(payload, r["seed"], r["date"], r["offset"])
        copy_checked(src / "COMPLETE.json", dest / "COMPLETE.json", r["complete_sha256"])
        done = read(src / "COMPLETE.json")
        require(done["files_sha256"] == r["files_sha256"] and done["spec_sha256"] == canonical, "Case hash binding differs")
        for name, h in r["files_sha256"].items():
            p = src / analyzer.relative(name); require(sha(p) == h, "Original episode/case changed")
            if name == "case.json" or args.include_replay_tables:
                copy_checked(p, dest / name, h)
            else:
                omitted.append(dict(path=(dest / name).relative_to(payload).as_posix(), bytes=p.stat().st_size, sha256=h))
    write(payload / "REPLAY_TABLE_REQUIREMENTS.json", dict(replay_tables_included=args.include_replay_tables,
          files=omitted, note="Original derived episode files. If omitted, hashes are cross-references only; statistical reanalysis uses the included paired observations."))
    external = [dict(path=name, bytes=(args.assets / name).stat().st_size, sha256=h)
                for name, h in sorted(spec["input_sha256"].items())]
    write(payload / "EXTERNAL_INPUT_REQUIREMENTS.json", dict(files=external, raw_traffic_included=False,
          baseline_release="https://github.com/dibbend8/truthful-ran-slicing/releases/tag/v0.2.0",
          note="Required for full replay, not offline reanalysis. Restore the baseline archive, the public v0.2.0 repository test-source tree and matching prepared traffic caches; traffic access conditions remain applicable."))
    write(payload / "EXTERNAL_SOURCE_REQUIREMENTS.json", dict(
          files=[dict(path=name, sha256=h) for name, h in sorted(spec["source_sha256"].items())],
          public_source="https://github.com/dibbend8/truthful-ran-slicing/tree/v0.2.0",
          note="supplemental_runner is included under source/. study_support.py is in the baseline archive's deadline_load_study directory. Other entries are relative to its upstream directory; the 31 code/tests files are supplied by the public repository's implementation/code/tests tree."))
    (payload / "README.md").write_text("# Original-setting five-seed reporting supplement\n\n"
        "This exploratory extension reuses 16 TEST recording dates, 32 prespecified rotated windows and five original training seeds. "
        "It contains 160 complete cases and 800 trajectories, using original deadlines 2002/150/300 ms. "
        "It does not expand the original Holm family or rerun packet simulation.\n\n"
        "See analysis/headline_summary.csv and analysis/per_seed_summary.csv for effects, denominators, conditional confidence intervals and seed spread. "
        "All individual paired outcomes and search decisions are retained. "
        "The portable specification explicitly records removal of private locator fields; scientific case records retain their original bytes.\n\n"
        "Verify the archive checksum first. From this extracted directory, run:\n\n```bash\n"
        "python -B source/reanalyze_original_release.py --release-root . \\\n"
        "  --expected-manifest-sha256 HASH_FROM_ARCHIVE_MANIFEST \\\n"
        "  --output ../new-reanalysis\n```\n\n"
        "The output path must not exist. No external raw traffic, model loading, simulation or training is required. "
        "Use Python with NumPy, pandas and PyArrow installed; source/PORTABLE_README.txt records the tested analysis environment and installation example. "
        "If replay tables are included, they are independently reread and paired; otherwise their hashes are cross-referenced. "
        "See source/PORTABLE_README.txt for reproduction levels and external input requirements.\n")
    records = []
    for p in sorted(payload.rglob("*")):
        if p.is_file():
            privacy_check(p); records.append(dict(path=p.relative_to(payload).as_posix(), bytes=p.stat().st_size, sha256=sha(p)))
    manifest = dict(schema="original-five-seed-portable-v1", complete=True, scientific_result=True, exploratory=True,
        scope=dict(cases=160, trajectories=800, training_seeds=5, dates=16, windows=32, deviator_episodes=480,
                   scored_search_decisions=76800), original_specification_canonical_sha256=canonical,
        approved_findings_sha256=args.approved_findings_sha256, replay_tables_included=args.include_replay_tables,
        raw_external_traffic_included=False, original_scientific_case_bytes_preserved=True,
        specification_is_explicit_derivative=True, files=records, publication_performed=False)
    write(payload / "PUBLIC_PACKAGE_MANIFEST.json", manifest)
    manifest_sha = sha(payload / "PUBLIC_PACKAGE_MANIFEST.json")
    subprocess.run([sys.executable, "-B", str(payload / "source/reanalyze_original_release.py"),
                    "--release-root", str(payload), "--expected-manifest-sha256", manifest_sha,
                    "--output", str(args.output / "restored_reanalysis")], check=True)
    validation = args.output / "restored_reanalysis/PORTABLE_REANALYSIS_VERIFICATION.json"
    require(read(validation)["complete"] is True, "Restored reanalysis did not complete")
    archive = args.output / "original-five-seed-evidence.tar.gz"
    record = archive_payload(payload, archive)
    record.update(file=archive.name, public_package_manifest_sha256=manifest_sha,
                  portable_reanalysis_verification_sha256=sha(validation), published=False,
                  created_utc=datetime.now(timezone.utc).isoformat())
    write(args.output / "ARCHIVE_MANIFEST.json", record)
    (args.output / "archive.sha256").write_text(record["sha256"] + "  " + archive.name + "\n")
    print(json.dumps(record))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--skeleton", action="store_true")
    p.add_argument("--campaign", type=Path); p.add_argument("--analysis", type=Path)
    p.add_argument("--assets", type=Path); p.add_argument("--frozen-support", type=Path)
    p.add_argument("--approved-findings-sha256"); p.add_argument("--include-replay-tables", action="store_true")
    args = p.parse_args()
    if args.skeleton:
        require(not any((args.campaign, args.analysis, args.assets, args.frozen_support, args.approved_findings_sha256)),
                "Source-only staging must not be mixed with scientific-result arguments")
        skeleton(args.output); return
    require(all((args.campaign, args.analysis, args.assets, args.frozen_support, args.approved_findings_sha256)),
            "Packaging requires complete campaign, analysis, assets, support and explicitly approved findings hash")
    package(args)


if __name__ == "__main__":
    main()
