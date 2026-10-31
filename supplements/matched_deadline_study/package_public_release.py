#!/usr/bin/env python3
"""Prepare a new, validated public release from an already verified private payload.

No publishing, uploading, source mutation, traffic acquisition or replay occurs.
The default scope requires nine truthful and three jointly tightened strategic
cells. --allow-partial is explicit and records every missing requested cell.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
UTILITIES=('package_public_release.py','validate_public_release.py','reanalyze_released.py',
           'PUBLIC_RELEASE_README.txt','RELEASED_ANALYSIS_README.txt',
           'bundle_matched.py','BUNDLING_README.txt')
OPTIONAL_RECORDS=('EXECUTION_PARALLELISM_NOTE.txt',)
EXPECTED={(mode,config,profile) for mode in ('truth','strategic') for config in ('default','rho0.7','rho1.1')
          for profile in (('both','video300','interactive50') if mode=='truth' else ('both',))}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def safe_path(value):
    path=Path(value)
    require(not path.is_absolute() and '..' not in path.parts and value not in ('','.'),'Unsafe manifest path')
    return path


def load_builder(path):
    spec=importlib.util.spec_from_file_location('private_bundle_support',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def parse_sums(payload):
    values={}
    for line in (payload/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);safe_path(name)
        require(len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'Malformed SHA256SUMS digest')
        require(name not in values,'Duplicate SHA256SUMS member')
        values[name]=expected
    actual={p.relative_to(payload).as_posix() for p in payload.rglob('*') if p.is_file()}
    require(actual==set(values)|{'SHA256SUMS'},'Private payload inventory is incomplete or contains extra files')
    for name,expected in values.items():
        path=payload/name
        require(not path.is_symlink() and sha(path)==expected,'Private payload hash mismatch: '+name)
    return values


def selection(name):
    """Return keep/raw/exclude; scientific bytes are never rewritten."""
    path=safe_path(name);parts=path.parts
    if parts[0] not in ('matched_deadline_study','deadline_load_study'):
        return 'exclude'
    if any(x in ('server','runtime','.git','.ssh','.env','.venv','venv','__pycache__','smoke','smoke_evaluation','diagnostic_layout') for x in parts):
        return 'exclude'
    if path.suffix in ('.log','.pyc','.pem','.key') or path.name in ('SERVER_STATUS.txt','drive_archive.json'):
        return 'exclude'
    if parts[:2]==('matched_deadline_study','evaluation'):
        return 'keep' if path.name=='COMPLETE.json' else 'raw'
    if 'analysis_matched' in parts and path.suffix=='.csv':
        if any(token in path.stem for token in ('observations','windows','pairs','invariants')):
            return 'exclude'
    if parts[:2]==('matched_deadline_study','analysis_diagnostics'):
        return 'exclude'
    if path.name in ('training_smoke_validation.json','evaluation_smoke_validation.json','bundling_validation.json'):
        return 'exclude'
    return 'keep'


def validate_scope(payload,scope,hashes,allow_partial):
    require(scope.get('scientific_result') is True and not scope.get('dry_run',False),'A diagnostic or dry-run payload cannot become a public result')
    cells=scope.get('selected_cells',[])
    keys={(r['mode'],r['config'],r['profile']) for r in cells}
    require(len(keys)==len(cells) and keys<=EXPECTED,'Unexpected or duplicated public study cells')
    require(allow_partial or keys==EXPECTED,'Public release requires9 truth and3 both strategic cells; use --allow-partial explicitly otherwise')
    require(cells and all(r['cases']==32 and r.get('scientific_result') is True for r in cells),'Incomplete public cell specification')
    for mode,config,profile in sorted(keys):
        folder=payload/'matched_deadline_study/evaluation'/mode/config/profile
        markers=sorted(folder.glob('*/COMPLETE.json'))
        require(len(markers)==32,'Public source lacks32 completed raw cases')
        for marker in markers:
            complete=load(marker)
            require(complete.get('complete') is True and complete.get('scientific_result') is True,'Incomplete or diagnostic raw case')
            for name,expected in complete['files_sha256'].items():
                rel=(marker.parent/safe_path(name)).relative_to(payload).as_posix()
                require(hashes.get(rel)==expected,'Raw case differs from original completion hash')
        valid=False
        for variant in ('all',mode):
            findings=payload/'matched_deadline_study/analysis_matched'/config/profile/variant/'findings.json'
            if not findings.is_file():continue
            f=load(findings);s=f['scope']
            if s.get('config')!=config or s.get('profile')!=profile or mode not in s.get('modes',[]):continue
            status=[x for x in f['completion'] if x['mode']==mode]
            require(f.get('scientific_result') is True and len(status)==1 and status[0]['complete'] and status[0]['validated_cases']==status[0]['expected_cases']==32,'Analysis completion mismatch')
            require(f['analysis_sha256']==hashes['matched_deadline_study/analyze_matched.py'] and f['protocol_sha256']==hashes['matched_deadline_study/PROTOCOL.txt'],'Analysis source/protocol changed')
            for name,expected in f['table_files_sha256'].items():
                rel=(findings.parent/safe_path(name)).relative_to(payload).as_posix()
                require(hashes.get(rel)==expected,'Analysis table differs from original finding hash')
            valid=True;break
        require(valid,'No complete verified analysis for included raw cell')
    return cells,sorted(EXPECTED-keys)


def manifest_inventory(payload,hashes,builder):
    keep={};raw=[];omitted=[]
    for name,expected in sorted(hashes.items()):
        path=payload/name;action=selection(name)
        if action=='keep':
            issues=builder.sensitive(path)
            require(not issues,'Public privacy scan refused '+name+': '+', '.join(issues))
            keep[name]=dict(source=path,sha256=expected,bytes=path.stat().st_size,provenance='unchanged original')
        elif action=='raw':
            raw.append(dict(path=name,sha256=expected,bytes=path.stat().st_size,reason='raw replay output; excluded from public archive, available in verified private evidence snapshot'))
        else:omitted.append(dict(path=name,sha256=expected,bytes=path.stat().st_size,reason='redundant format, diagnostic or private packaging metadata'))
    return keep,raw,omitted


def readme(version,cells,missing):
    return f'''Truthful RAN slicing: prepared {version} research release

This archive is prepared for review; package preparation does not publish it.
It includes{len(cells)} complete scientific mode/cell combinations. There are
{len(missing)} missing combinations from the planned9 truthful/3 strategic scope.
PUBLIC_RELEASE_MANIFEST.json is authoritative about included cells and files.
The original v0.1.0 source release remains a separate historical version.
When included, matched_deadline_study/EXECUTION_PARALLELISM_NOTE.txt records an
execution-only scheduling change; the frozen scope and training budgets remain
unchanged.

What can be reproduced offline
- Recompute statistical summaries from hashed released experiment observations
  with matched_deadline_study/reanalyze_released.py. This checks the entire
  date/window/seed/strategy grid, recomputes bootstrap estimates and compares
  them with the original results. It does not replay the simulator.
- Regenerate figures with matched_deadline_study/plot_matched.py.
- Inspect and load all included original and matched selected/final checkpoints,
  exact training histories, specifications, calibration and shape parameters.
See matched_deadline_study/RELEASED_ANALYSIS_README.txt and the public validation
record for commands. From this extracted archive's root, run an offline check
with a NEW output path outside the archive directory:
  python -B matched_deadline_study/validate_public_release.py \\
    --release-root . --output ../release-validation.json
The validator also creates ../release-validation_artifacts/; both output paths
must be unused. It imports the restored code, safely loads checkpoints, checks
the native gate, recomputes derived statistics and regenerates plots without
reading raw traffic or using the network.

Runtime and dependencies
Use a compatible Python3.11+ environment with NumPy, pandas, pyarrow, torch,
matplotlib, scipy, PyYAML and numba. Exact training/replay version records are in
matched_deadline_study/evaluation_specifications/*.json under dependencies, and
matched_deadline_study/models/.../config.json (or static.json) under
spec.dependencies. They describe their individual stages; they do not reconstruct
the historical environment of every original model. No unified pip lock file is
supplied, so installing these package names alone does not claim an identical
environment. Consult the version records for the stage being reproduced.

What still needs external inputs
The original raw/prepared traffic is not redistributed. The original per-interval
replay files and result.json records are also excluded from this compact public
archive. EXTERNAL_REQUIREMENTS.json lists data prerequisites, and
RAW_EVIDENCE_REQUIREMENTS.json lists the exact omitted raw-evidence hashes.
Original COMPLETE files and recorded specifications remain unchanged. Restore
these assets at their stated paths to run analyze_matched.py's full raw-data
verification; restore traffic to rerun training or replay. Do not disable hashes
or historical TEST gates to make missing assets pass. Dataset acquisition and
redistribution conditions are independent of this code archive.

Integrity and provenance
All retained scientific source, model, configuration and analysis files are
byte-identical to the validated private evidence snapshot. Historical author
filesystem paths in provenance are retained as recorded; portable commands use
explicit paths or the sibling layout, not those old strings. No credentials,
private Drive sharing URLs, SSH configuration, server logs or environment trees
are included. New release utilities and metadata are identified separately in
the manifest; originals are not silently sanitized or rehashed.

The exact original findings may list CSV files omitted here to avoid duplicating
Parquet observations; retained tables retain their original hashes. Their
Parquet equivalents and statistical CSVs for plotting are included. Every
released file is covered by PUBLIC_RELEASE_MANIFEST.json and SHA256SUMS; the
manifest itself is covered by SHA256SUMS. RELEASE_VALIDATION.json records an
offline check from a fresh restored copy, separate from the research runs.

The full three-load original comparator set comprises78 checkpoint/config/static
files, absent from the v0.1.0 source-only package. This archive includes the
original frozen comparator subset required by its completed cells, alongside
new fitted models. It also includes all309 assets and8 calibration JSONs required
by the unchanged historical TEST-read gate; their inclusion is not evidence of
new experiments outside the stated scope. Dependencies used during each stage are recorded in those
immutable specifications; no historical environment is invented.

No new software/data license or ownership claim is assigned. Existing notices
and applicable third-party conditions remain in force. No DOI is claimed.
Use the actual versioned GitHub release URL only after publication; the expected
location is https://github.com/dibbend8/truthful-ran-slicing/releases/tag/{version}.
Later Zenodo archiving may add a version DOI, but it is not implied here.

The new TEST study is exploratory because these dates were inspected previously.
Observed individual delivery-utility gains are retained even when an average
is negative. Numerical payment bounds and statistical confidence intervals are
different. These trace replays do not establish practical strategyproofness,
packet reliability or end-to-end3GPP QoS compliance.
'''


def build_archive(payload,destination,version):
    with destination.open('xb') as output:
        with gzip.GzipFile(fileobj=output,mode='wb',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(p for p in payload.rglob('*') if p.is_file()):
                    info=tar.gettarinfo(str(path),arcname='truthful-ran-slicing-'+version+'/'+path.relative_to(payload).as_posix())
                    info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                    with path.open('rb') as handle:tar.addfile(info,handle)
    require(destination.stat().st_size<2*(1<<30),'Archive exceeds2GiB release-asset limit; use separately reviewed split assets')
    expected={'truthful-ran-slicing-'+version+'/'+p.relative_to(payload).as_posix():sha(p) for p in payload.rglob('*') if p.is_file()}
    seen=set()
    with tarfile.open(destination,'r:gz') as tar:
        for member in tar:
            require(member.isfile() and member.name in expected and member.name not in seen,'Unexpected public archive member')
            digest=hashlib.sha256();handle=tar.extractfile(member)
            for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
            require(digest.hexdigest()==expected[member.name],'Public archive readback mismatch');seen.add(member.name)
    require(seen==set(expected),'Public archive is missing members')
    return dict(file=destination.name,bytes=destination.stat().st_size,sha256=sha(destination),all_members_readback_verified=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-payload',type=Path,required=True,help='payload/ of completed verified private bundle')
    p.add_argument('--root',type=Path,default=ROOT,help='release utility source directory')
    p.add_argument('--assets',type=Path,help='original recovered upstream for completing historical gate dependencies; defaults to private payload upstream')
    p.add_argument('--output',type=Path,help='NEW output directory; never the existing public repository')
    p.add_argument('--version',default='v0.2.0')
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();args.private_payload=args.private_payload.expanduser().resolve();args.root=args.root.expanduser().resolve()
    args.assets=(args.assets or args.private_payload/'deadline_load_study/upstream').expanduser().resolve()
    require(args.version=='v0.2.0','This reviewed export targets v0.2.0')
    require(args.dry_run or args.output is not None,'Use --dry-run or provide a NEW --output')
    if args.output:
        args.output=args.output.expanduser().resolve()
        require(not args.output.exists(),'Refusing to overwrite existing output')
        require(not args.output.is_relative_to(args.private_payload) and not args.private_payload.is_relative_to(args.output),'Output and source payload must be separate')
    builder=load_builder(args.root/'bundle_matched.py')
    hashes=parse_sums(args.private_payload);scope=load(args.private_payload/'BUNDLE_SCOPE.json')
    cells,missing=validate_scope(args.private_payload,scope,hashes,args.allow_partial)
    files,external_raw,omitted=manifest_inventory(args.private_payload,hashes,builder)
    gate=builder.frozen_gate_inventory(args.assets,args.private_payload/'deadline_load_study')
    require(gate['prereg_sha256']==hashes['deadline_load_study/upstream/prereg/PREREG.md'], 'Historical gate belongs to different preregistration')
    for row in gate['files']:
        source=args.assets/row['asset_relative_path'];name=row['path']
        require(not builder.sensitive(source),'Historical gate asset privacy scan failed: '+name)
        if name in files:
            require(files[name]['sha256']==row['sha256'],'Conflicting frozen gate artifact')
        else:
            files[name]=dict(source=source,sha256=row['sha256'],bytes=row['bytes'],provenance='unchanged original')
    for name in UTILITIES:
        path=args.root/name;require(path.is_file(),'Missing release utility: '+name)
        require(not builder.sensitive(path),'Release utility privacy scan failed: '+name)
        files['matched_deadline_study/'+name]=dict(source=path,sha256=sha(path),bytes=path.stat().st_size,provenance='new release utility')
    for name in OPTIONAL_RECORDS:
        path=args.root/name
        if not path.is_file():continue
        require(not builder.sensitive(path),'Execution-note privacy scan failed: '+name)
        target='matched_deadline_study/'+name
        if target in files:
            require(files[target]['sha256']==sha(path),'Execution note differs from the private snapshot')
        else:
            files[target]=dict(source=path,sha256=sha(path),bytes=path.stat().st_size,provenance='generated release metadata')
    external_inputs=load(args.private_payload/'EXTERNAL_REQUIREMENTS.json')['files']
    summary=dict(scientific_result=True,prepared_release=args.version,published=False,selected_cells=cells,
                 missing_requested_cells=[dict(mode=m,config=c,profile=pr) for m,c,pr in missing],
                 files=len(files),included_bytes=sum(r['bytes'] for r in files.values()),
                 external_traffic_files=len(external_inputs),external_raw_evidence_files=len(external_raw),
                 external_raw_evidence_bytes=sum(r['bytes'] for r in external_raw),dry_run=args.dry_run)
    summary['historical_gate_asset_files']=len(gate['files'])
    if args.dry_run:
        print(json.dumps(summary,indent=2));return
    args.output.mkdir(parents=True,exist_ok=False);payload=args.output/'payload';payload.mkdir()
    for name,row in files.items():
        path=payload/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['source'],path)
        require(sha(path)==row['sha256'],'Source bytes changed during public export')
    metadata={'README.txt':readme(args.version,cells,missing).encode(),
              'EXTERNAL_REQUIREMENTS.json':encoded(dict(files=external_inputs,automatic_download=False)),
              'RAW_EVIDENCE_REQUIREMENTS.json':encoded(dict(files=external_raw,automatic_download=False)),
              'OMITTED_PRIVATE_ARTIFACTS.json':encoded(dict(files=omitted)),
              'FROZEN_GATE_REQUIREMENTS.json':encoded(gate),
              'BASELINE_COVERAGE.json':(args.private_payload/'BASELINE_COVERAGE.json').read_bytes()}
    for name,data in metadata.items():
        path=payload/name;path.write_bytes(data)
        files[name]=dict(source=path,sha256=sha(path),bytes=len(data),provenance='generated release metadata')
    manifest=dict(version=1,scientific_result=True,prepared_release=args.version,published=False,selected_cells=cells,
        missing_requested_cells=summary['missing_requested_cells'],source_private_scope_sha256=hashes['BUNDLE_SCOPE.json'],
        source_private_sha256sums_sha256=sha(args.private_payload/'SHA256SUMS'),
        external_original_inputs=external_inputs,external_raw_evidence=external_raw,
        files={name:{k:v for k,v in row.items() if k!='source'} for name,row in sorted(files.items())},
        preservation='Retained original scientific files have unchanged bytes; new utilities and metadata are labeled.',
        privacy='Explicit whitelist; no server/SSH/env/log/Drive-sharing files; benign recorded author paths retained.')
    (payload/'PUBLIC_RELEASE_MANIFEST.json').write_bytes(encoded(manifest))
    validation=args.output/'RELEASE_VALIDATION.json'
    command=[sys.executable,'-B',str(payload/'matched_deadline_study/validate_public_release.py'),
             '--release-root',str(payload),'--output',str(validation)]
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    outcome=subprocess.run(command,env=environment,capture_output=True,text=True)
    require(outcome.returncode==0,'Offline restored-layout validation failed; new output retained for inspection. '+outcome.stderr[-1500:])
    result=load(validation);require(result.get('complete') is True,'Offline validator did not confirm completion')
    shutil.copyfile(validation,payload/'RELEASE_VALIDATION.json')
    allfiles=sorted(p for p in payload.rglob('*') if p.is_file())
    (payload/'SHA256SUMS').write_text(''.join(sha(path)+'  '+path.relative_to(payload).as_posix()+'\n' for path in allfiles))
    archive=build_archive(payload,args.output/('truthful-ran-slicing-'+args.version+'-evidence.tar.gz'),args.version)
    (args.output/'ARCHIVE_MANIFEST.json').write_bytes(encoded(dict(**archive,published=False,
        public_release_manifest_sha256=sha(payload/'PUBLIC_RELEASE_MANIFEST.json'),validation_sha256=sha(validation),
        created_utc=datetime.now(timezone.utc).isoformat())))
    print(json.dumps(dict(**summary,output=str(args.output),archive=archive,offline_validation='passed'),indent=2))


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,FileNotFoundError) as error:
        raise SystemExit('PUBLIC RELEASE REFUSED: '+str(error))
