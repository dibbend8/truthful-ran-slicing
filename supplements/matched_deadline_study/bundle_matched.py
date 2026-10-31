#!/usr/bin/env python3
"""Build a new local evidence bundle from verified, completed matched-study cells.

No upload, training, evaluation, download or credential access is performed.
Default requested scope: three both-deadline strategic cells plus all nine
truth cells; only complete and analyzed cells are included, with omissions
listed explicitly. Diagnostic smoke is allowed only with --dry-run.
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
import tarfile

ROOT = Path(__file__).resolve().parent
CONFIGS = ('default', 'rho0.7', 'rho1.1')
PROFILES = ('both', 'video300', 'interactive50')
DOCS = ('PROTOCOL.txt', 'SIX_HOUR_ADDENDUM.txt', 'TRAINING_README.txt',
        'EVALUATION_README.txt', 'ANALYSIS_README.txt', 'BUNDLING_README.txt')
OPTIONAL_DOCS = ('EXECUTION_PARALLELISM_NOTE.txt',)
SCRIPTS = ('train_matched.py', 'fit_static_matched.py', 'evaluate_matched.py',
           'analyze_matched.py', 'plot_matched.py', 'bundle_matched.py')
VALIDATIONS = ('training_smoke_validation.json', 'evaluation_smoke_validation.json',
              'analysis_diagnostics/smoke_validation.json', 'bundling_validation.json')
RECOVERY = ('upstream_asset_manifest.json', 'puffer_archive_recovery_manifest.json',
            'original_puffer_fetch_manifest.json')
TEXT_SUFFIXES = {'.py', '.sh', '.txt', '.md', '.json', '.yaml', '.yml', '.csv', '.cff'}
BLOCKED_PATTERNS = {
    'private-key material': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'GitHub access token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'AWS access key': rb'\bAKIA[0-9A-Z]{16}\b',
    'Slack access token': rb'xox[baprs]-[A-Za-z0-9-]{20,}',
    'credential in URL': rb'https?://[^/\s:@]+:[^/\s@]+@',
    'literal credential assignment': rb'(?i)["\']?(?:password|passwd|api_key|secret_key|access_token|auth_token)["\']?\s*[:=]\s*["\'][A-Za-z0-9_+/.=-]{12,}["\']',
    'private archive sharing URL': rb'https?://(?:drive\.google\.com|drive\.usercontent\.google\.com)/[^\s"\']+',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def relative(value):
    p = Path(value)
    require(not p.is_absolute() and '..' not in p.parts and value not in ('', '.'), 'Invalid bundle-relative path')
    return p


def sensitive(path):
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    data = path.read_bytes()
    return [category for category,pattern in BLOCKED_PATTERNS.items() if re.search(pattern,data)]


class Inventory:
    def __init__(self):
        self.files = {}
        self.external = {}
        self.figure_status = []
        self.frozen_gate = None

    def add(self, source, destination, expected=None, reason='verified study artifact'):
        source = Path(source)
        destination = relative(destination).as_posix()
        forbidden = {'server', 'runtime', '.git', '.ssh', '.env', '.venv', 'venv', '__pycache__'}
        require(not forbidden.intersection(Path(destination).parts) and source.suffix not in ('.log','.pyc','.pem','.key'),
                f'Excluded file category: {destination}')
        require(source.is_file() and not source.is_symlink(), f'Missing or linked artifact: {destination}')
        actual = sha(source)
        require(expected is None or actual == expected, f'Artifact hash mismatch: {destination}')
        issues = sensitive(source)
        require(not issues, f'Sensitive content refused in {destination}: {", ".join(issues)}')
        row = dict(source=source, sha256=actual, bytes=source.stat().st_size, reason=reason)
        if destination in self.files:
            require(self.files[destination]['sha256'] == actual, f'Conflicting artifact: {destination}')
            return
        self.files[destination] = row

    def need(self, source, destination, expected=None, reason='external traffic data'):
        source = Path(source);destination=relative(destination).as_posix()
        require(source.is_file() and not source.is_symlink(), f'Missing external prerequisite: {destination}')
        actual=sha(source)
        require(expected is None or actual==expected, f'External prerequisite hash mismatch: {destination}')
        item=dict(path=destination,sha256=actual,bytes=source.stat().st_size,reason=reason)
        if destination in self.external:
            require(self.external[destination]['sha256']==actual, f'Conflicting external prerequisite: {destination}')
        self.external[destination]=item

    def asset(self, assets, name, expected, reason):
        name=relative(name).as_posix();source=assets/name;destination='deadline_load_study/upstream/'+name
        # Prepared traffic caches are intentionally external too. JSON receipts,
        # calibration and checkpoint records are included when safe.
        if name.startswith('data/') and source.suffix in ('.parquet','.npz','.csv','.gz','.zst'):
            self.need(source,destination,expected,reason)
        elif sensitive(source):
            self.need(source,destination,expected,'exact original metadata retained externally to avoid sharing private locator/credential content')
        else:
            self.add(source,destination,expected,reason)


def import_analyzer(path):
    spec=importlib.util.spec_from_file_location('matched_bundle_validator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def frozen_gate_inventory(assets, frozen_study):
    """Original TEST-read gate requires the entire frozen asset set, even when
    the new study compares only a subset of those policies. Never bypass it.
    """
    import yaml
    assets, frozen_study = Path(assets), Path(frozen_study)
    prereg = assets/'prereg/PREREG.md'
    blocks = re.findall(r'^```freeze[ \t]*\n(.*?)^```[ \t]*$', prereg.read_text(), re.S|re.M)
    require(len(blocks)==1, 'Expected one original freeze block')
    frozen = yaml.safe_load(blocks[0])
    recovery_path=frozen_study/'upstream_asset_manifest.json'
    recovery=load(recovery_path)
    recovered={r['source_path'].split('ICC/rewrite_2026-10-18/',1)[1]:r['sha256'] for r in recovery['files']}
    required=dict(frozen['checkpoints'])
    calibrations=sorted((assets/'data/calibration').glob('*.json'))
    require(len(calibrations)==8, 'Full original configuration hash requires all8 calibration JSONs')
    for path in calibrations:
        name=path.relative_to(assets).as_posix()
        require(name in recovered, 'Calibration absent from recovery manifest')
        required[name]=recovered[name]
    records=[]
    for name,expected in sorted(required.items()):
        name=relative(name).as_posix();path=assets/name
        require(recovered.get(name)==expected and path.is_file() and sha(path)==expected,
                'Missing or changed historical TEST gate dependency: '+name)
        records.append(dict(path='deadline_load_study/upstream/'+name,asset_relative_path=name,
                            sha256=expected,bytes=path.stat().st_size,
                            role='original calibration' if name.startswith('data/calibration/') else 'historical frozen gate asset'))
    return dict(prereg_sha256=sha(prereg),recovery_manifest_sha256=sha(recovery_path),
                frozen_checkpoint_assets=len(frozen['checkpoints']),calibration_files=len(calibrations),files=records,
                note='These historical assets satisfy unchanged TEST-read checks; their inclusion is not a claim that their experiments were rerun.')


def requested(args):
    if args.diagnostic_smoke:
        return [('truth','default','both'),('strategic','default','both')]
    values = [tuple(x.split('/')) for x in args.cell] if args.cell else (
        [('strategic',c,'both') for c in CONFIGS]+[('truth',c,p) for c in CONFIGS for p in PROFILES])
    require(len(values)==len(set(values)) and all(len(x)==3 and x[0] in ('truth','strategic') and x[1] in CONFIGS and x[2] in PROFILES for x in values),'Invalid or duplicate --cell; use MODE/CONFIG/PROFILE')
    return values


def analysis_for(args, mode, config, profile, cases, analyzer):
    for variant in ('all',mode):
        folder=args.analysis_root/config/profile/variant
        if not (folder/'findings.json').is_file():
            continue
        f=load(folder/'findings.json');scope=f['scope']
        if scope.get('config')!=config or scope.get('profile')!=profile or mode not in scope.get('modes',[]):
            continue
        require(f.get('scientific_result') is True,'Diagnostic analysis cannot enter scientific bundle')
        status=[s for s in f['completion'] if s['mode']==mode]
        require(len(status)==1 and status[0]['complete'] and status[0]['expected_cases']==status[0]['validated_cases']==32,'Analysis cell completion mismatch')
        require(f['analysis_sha256']==sha(args.root/'analyze_matched.py') and f['protocol_sha256']==sha(args.root/'PROTOCOL.txt'),'Stale analysis source/protocol')
        table_hashes=f.get('table_files_sha256',{})
        require(table_hashes, 'Analysis lacks table SHA-256 inventory')
        for filename,expected in table_hashes.items():
            require(len(relative(filename).parts)==1 and sha(folder/filename)==expected,'Analysis table hash mismatch')
        csv=analyzer.pd.read_csv(folder/f'{mode}_cases.csv')
        expected={r['case_spec_sha256']:r for r in cases.to_dict('records')}
        require(len(csv)==32 and set(csv.case_spec_sha256)==set(expected),'Analysis/raw case inventory mismatch')
        for row in csv.to_dict('records'):
            for key in ('completion_sha256','result_sha256'):
                require(row[key]==expected[row['case_spec_sha256']][key], 'Analysis uses different raw case bytes')
        return folder,f
    raise FileNotFoundError('complete analysis report unavailable')


def add_models(inv,args,spec):
    for entry in spec['models'].values():
        if entry['state']=='reference':
            continue
        if entry['state']=='frozen':
            base=relative(entry['asset_relative_dir'])
            for name,expected in entry['files_sha256'].items():
                inv.asset(args.assets,(base/relative(name)).as_posix(),expected,'frozen comparator model used in selected cell')
            continue
        rel=relative(entry['model_relative_dir']);base=args.model_root/rel
        target='matched_deadline_study/'+('smoke/' if args.diagnostic_smoke else 'models/')+rel.as_posix()
        inv.add(base/'COMPLETE.json',target+'/COMPLETE.json',entry['completion_sha256'])
        for name,expected in entry['files_sha256'].items():
            inv.add(base/relative(name),target+'/'+name,expected,'completed matched model artifact')
        record=load(base/('static.json' if entry['arm']=='static' else 'config.json'))
        fit=record['spec']
        for name,expected in fit['input_inventory']['files'].items():
            inv.asset(args.assets,name,expected,'original TRAIN/DEV prerequisite for matched fitting')
        category='static_specifications' if entry['arm']=='static' else 'specifications'
        candidate=args.root/category/(entry['spec_sha256']+'.json')
        if candidate.is_file():
            require(import_analyzer(args.root/'analyze_matched.py').digest(load(candidate))==entry['spec_sha256'],'Training specification digest mismatch')
            inv.add(candidate,'matched_deadline_study/'+category+'/'+candidate.name)


def original_coverage(args,inv):
    records=[]
    for destination,row in inv.files.items():
        prefix='deadline_load_study/upstream/results/models/'
        if not destination.startswith(prefix) or row['reason']!='frozen comparator model used in selected cell':
            continue
        relative_asset=destination.removeprefix('deadline_load_study/upstream/')
        public=args.original_release/'implementation'/relative_asset
        exists=public.is_file()
        records.append(dict(asset=relative_asset,required_sha256=row['sha256'],present_in_original_v0_1_package=exists,
                            matching_original_v0_1_bytes=exists and sha(public)==row['sha256'],included_in_this_bundle=True))
    complete_grid=[]
    for config in CONFIGS:
        for arm in ('protected','unconstrained','meas_only','kappa0'):
            complete_grid.extend(f'results/models/{config}/{arm}/seed{seed}/{name}' for seed in range(3) for name in ('model.pt','config.json'))
        complete_grid.extend(f'results/models/{config}/static/{name}' for name in ('static.json','candidates.npz'))
    return dict(original_release_label='local source package corresponding to original v0.1.0 release',
                original_release_inspected=args.original_release.is_dir(),required_frozen_files=len(records),
                required_frozen_files_present=sum(r['present_in_original_v0_1_package'] for r in records),files=records,
                full_three_load_frozen_grid_files=78,
                full_grid_files_present_in_original_v0_1_package=sum((args.original_release/'implementation'/name).is_file() for name in complete_grid),
                limitation='This checks local package contents, not the current network repository or release attachments.')


def collect(args,analyzer):
    inv=Inventory();selected=[];excluded=[];scope=requested(args)
    verifier=analyzer.Verifier(argparse.Namespace(root=args.root,assets=args.assets,frozen_study=args.frozen_study,
        model_root=args.model_root,protocol=args.root/'PROTOCOL.txt',allow_partial=False))
    for mode,config,profile in scope:
        category='smoke_evaluation' if args.diagnostic_smoke else 'evaluation'
        folder=args.root/category/mode/config/profile
        keys=[('2024-01-01',60)] if args.diagnostic_smoke else [(d,o) for d in analyzer.dates() for o in analyzer.OFFSETS]
        missing=[f'{d}__{o}' for d,o in keys if not (folder/f'{d}__{o}'/'COMPLETE.json').is_file()]
        if missing:
            excluded.append(dict(mode=mode,config=config,profile=profile,reason='incomplete case grid',missing_cases=missing));continue
        loaded=analyzer.load_cell(verifier.args,verifier,config,profile,mode,args.diagnostic_smoke)
        report=None
        if not args.diagnostic_smoke:
            try:
                report=analysis_for(args,mode,config,profile,loaded[3],analyzer)
            except FileNotFoundError:
                excluded.append(dict(mode=mode,config=config,profile=profile,reason='complete cases await complete analysis'));continue
        specs=[]
        for day,offset in keys:
            case=folder/f'{day}__{offset}';result=load(case/'result.json');spec=result['spec'];specs.append(spec)
            target=f'matched_deadline_study/{category}/{mode}/{config}/{profile}/{case.name}'
            inv.add(case/'COMPLETE.json',target+'/COMPLETE.json')
            for name,expected in load(case/'COMPLETE.json')['files_sha256'].items():
                inv.add(case/relative(name),target+'/'+name,expected,'verified per-case result and raw interval arrays')
            sp=args.root/'evaluation_specifications'/(result['spec_sha256']+'.json')
            inv.add(sp,'matched_deadline_study/evaluation_specifications/'+sp.name)
        first=specs[0]
        for name,expected in first['source_sha256'].items():
            prefix,path=name.split('/',1);path=relative(path)
            if prefix=='runner':source=args.root/path;target='matched_deadline_study/'+path.as_posix()
            elif prefix=='frozen-study':source=args.frozen_study/path;target='deadline_load_study/'+path.as_posix()
            elif prefix=='assets':source=args.assets/path;target='deadline_load_study/upstream/'+path.as_posix()
            else:raise ValueError('Unexpected recorded source namespace')
            inv.add(source,target,expected,'exact source bytes recorded by selected evaluations')
        for name,expected in first['input_sha256'].items():
            inv.asset(args.assets,name,expected,'original evaluation prerequisite')
        add_models(inv,args,first)
        inv.add(args.frozen_study/'shapes'/config/(profile+'.json'),f'deadline_load_study/shapes/{config}/{profile}.json',first['shape_sha256'])
        specids={analyzer.digest(s) for s in specs}
        for path,manifest in verifier.manifests:
            if specids&set(manifest['case_spec_sha256']):
                inv.add(path,'matched_deadline_study/evaluation_runs/'+path.name,reason='planned run; may list additional unbundled cells, not evidence of their completion')
        if report:
            directory,findings=report;target=f'matched_deadline_study/analysis_matched/{config}/{profile}/{directory.name}'
            inv.add(directory/'findings.json',target+'/findings.json')
            # A report may analyze both modes. Include all its hashed tables so
            # it remains verifiable, but scope records only complete raw cells.
            for name,expected in findings['table_files_sha256'].items():
                inv.add(directory/name,target+'/'+name,expected,'verified analysis table')
        selected.append(dict(mode=mode,config=config,profile=profile,cases=len(keys),scientific_result=not args.diagnostic_smoke))
    require(selected,'No complete verified cells are available; nothing will be packaged')
    require(not args.require_all or not excluded,'Requested scope incomplete; --require-all refuses a partial bundle')
    inv.frozen_gate=frozen_gate_inventory(args.assets,args.frozen_study)
    for row in inv.frozen_gate['files']:
        inv.add(args.assets/row['asset_relative_path'],row['path'],row['sha256'],row['role'])
    for name in DOCS+SCRIPTS:
        inv.add(args.root/name,'matched_deadline_study/'+name)
    for name in OPTIONAL_DOCS:
        if (args.root/name).is_file():
            inv.add(args.root/name,'matched_deadline_study/'+name,
                    reason='execution-only scheduling record; frozen scientific scope unchanged')
    for name in VALIDATIONS:
        path=args.root/name
        if path.is_file():inv.add(path,'matched_deadline_study/'+name,reason='diagnostic validation record; not scientific evidence')
    for name in RECOVERY:
        source=args.frozen_study/name;target='deadline_load_study/'+name
        if sensitive(source):inv.need(source,target,reason='exact recovery manifest with private archive locator; supplied separately with original frozen study')
        else:inv.add(source,target,reason='original recovery metadata')
    if not args.diagnostic_smoke:
        included_hashes={row['sha256'] for row in inv.files.values()}
        for stem in ('matched_welfare_gaps','matched_pricing'):
            path=args.root/'figures'/(stem+'_provenance.json')
            if not path.is_file():
                continue
            record=load(path)
            require(record.get('scientific_result') is True and record['script_sha256']==sha(args.root/'plot_matched.py'), 'Stale or diagnostic scientific figure')
            covered=all(source['table_sha256'] in included_hashes and source['findings_sha256'] in included_hashes for source in record['source_tables'])
            inv.figure_status.append(dict(figure=stem,included=covered,reason='all source tables included' if covered else 'figure includes unbundled analysis'))
            if covered:
                for extension in ('pdf','png'):
                    inv.add(path.parent/(stem+'.'+extension),'matched_deadline_study/figures/'+stem+'.'+extension,
                            record[extension+'_sha256'],'scientific figure derived from included verified analysis')
                inv.add(path,'matched_deadline_study/figures/'+path.name)
    # Baseline scripts permit rerunning from the original prior-study assets.
    for name in ('PROTOCOL.txt','STRATEGIC_ADDENDUM.txt','REPRODUCE.txt','reproduce_study.py'):
        source=args.frozen_study/name
        if source.is_file() and not sensitive(source):inv.add(source,'deadline_load_study/'+name,reason='prior frozen-study context and instructions')
    require(not(set(inv.files)&set(inv.external)),'A prerequisite is both included and external')
    return inv,selected,excluded,scope,verifier


class SplitWriter(io.RawIOBase):
    def __init__(self,directory,part_bytes):
        self.directory=directory;self.limit=part_bytes;self.handle=None;self.part_size=0
        self.total=0;self.full_hash=hashlib.sha256();self.part_hash=None;self.parts=[]
    def writable(self):return True
    def write(self,data):
        data=memoryview(data);size=len(data);self.full_hash.update(data);self.total+=size
        while data:
            if self.handle is None:
                self.filename=f'matched-evidence.tar.gz.part{len(self.parts)+1:04d}'
                self.handle=(self.directory/self.filename).open('xb');self.part_size=0;self.part_hash=hashlib.sha256()
            count=min(len(data),self.limit-self.part_size);chunk=data[:count]
            self.handle.write(chunk);self.part_hash.update(chunk);self.part_size+=count;data=data[count:]
            if self.part_size==self.limit:self.finish_part()
        return size
    def finish_part(self):
        if self.handle is not None:
            self.handle.close();self.parts.append(dict(file=self.filename,bytes=self.part_size,sha256=self.part_hash.hexdigest()));self.handle=None
    def finish(self):self.finish_part()


class PartsReader(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.handle=None
    def readable(self):return True
    def readinto(self,buffer):
        while True:
            if self.handle is None:
                path=next(self.paths,None)
                if path is None:return 0
                self.handle=path.open('rb')
            count=self.handle.readinto(buffer)
            if count:return count
            self.handle.close();self.handle=None
    def close(self):
        if self.handle is not None:self.handle.close()
        super().close()


def archive(payload,parts,part_bytes):
    parts.mkdir();writer=SplitWriter(parts,part_bytes)
    with gzip.GzipFile(fileobj=writer,mode='wb',mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(p for p in payload.rglob('*') if p.is_file()):
                name='matched-deadline-evidence/'+path.relative_to(payload).as_posix()
                info=tar.gettarinfo(str(path),arcname=name);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                with path.open('rb') as handle:tar.addfile(info,handle)
    writer.finish()
    expected={('matched-deadline-evidence/'+p.relative_to(payload).as_posix()):sha(p) for p in payload.rglob('*') if p.is_file()}
    seen=set()
    with io.BufferedReader(PartsReader([parts/p['file'] for p in writer.parts])) as stream:
        with tarfile.open(fileobj=stream,mode='r|gz') as tar:
            for member in tar:
                require(member.isfile() and member.name in expected and member.name not in seen,'Unexpected archive member')
                h=hashlib.sha256();handle=tar.extractfile(member)
                for block in iter(lambda:handle.read(1<<20),b''):h.update(block)
                require(h.hexdigest()==expected[member.name],'Archive readback hash mismatch');seen.add(member.name)
    require(seen==set(expected),'Archive readback missing members')
    for part in writer.parts:require(sha(parts/part['file'])==part['sha256'],'Archive part hash mismatch')
    return dict(format='concatenated tar.gz byte stream',archive_bytes=writer.total,archive_sha256=writer.full_hash.hexdigest(),
                part_size_bytes=part_bytes,parts=writer.parts,readback_all_members_verified=True)


def instructions(scientific):
    return f'''Matched-deadline evidence bundle

Scientific result: {str(scientific).lower()}. Read BUNDLE_SCOPE.json first: it
lists exactly the completed cells included and the requested cells omitted.
This is a partial-scope release unless that inventory states otherwise.
Planned manifests can name additional work; they do not establish completion.

Integrity: SHA256SUMS covers every payload file except itself. The external
ARCHIVE_MANIFEST.json gives the ordered64MiB parts (last part may be smaller),
each part hash and the complete archive hash. Concatenate the listed parts in
order into matched-evidence.tar.gz, verify its SHA-256, then extract it. Parts
are consecutive bytes of one archive, not independently extractable archives.

This is an evidence-and-code snapshot, not a self-contained traffic dataset.
Raw/prepared Puffer, TRACTOR and Raca traffic caches are omitted. Restore every
EXTERNAL_REQUIREMENTS.json entry at its stated relative path using the original
frozen study/data distribution; verify its exact hash before any rerun. The
original recovery manifest may be required separately because its private
archive locator is intentionally not shared here. No download is automatic.
Original data licenses and access conditions remain applicable.

The included frozen checkpoints cover the comparisons actually bundled and the
entire historical TEST-read gate dependency set;
BASELINE_COVERAGE.json checks those exact files against the locally inspected
original v0.1.0 source package. Do not assume that original package included
trained checkpoints or data. Historical gate-only models do not represent new
matched-training results. FROZEN_GATE_REQUIREMENTS.json identifies these assets.
No new license or ownership claim is introduced; preserve existing notices.

Keep matched_deadline_study/ and deadline_load_study/ as sibling directories.
After restoring external prerequisites, use a compatible environment with the
exact dependency versions recorded in each evaluation/model specification.
The runners expect NumPy, pandas, pyarrow, torch, numba, scipy and PyYAML; plots
also use matplotlib. The analyzer uses NumPy/pandas/pyarrow and does not train.
Recorded hardware/versions can differ across stages; do not rewrite manifests.

Example completed-cell reanalysis (replace cell/mode with BUNDLE_SCOPE entries):
  python -B matched_deadline_study/analyze_matched.py --mode strategic \\
    --cell default/both --output reanalysis
  python -B matched_deadline_study/evaluate_matched.py --mode strategic \\
    --config default --profile both --output replay_matched \\
    --model-root matched_deadline_study/models \\
    --protocol matched_deadline_study/PROTOCOL.txt
For replay, copy the exact runner files and protocol into a new replay output
root first; see EVALUATION_README.txt. Training and static fitting similarly
require new output roots, unchanged budgets and restored TRAIN/DEV inputs.
See TRAINING_README.txt and ANALYSIS_README.txt for complete commands and limits.

Array files preserve individual gains and certified payment bounds. Analysis
averages matched seeds before a date/window bootstrap; maxima are descriptive,
not adaptive strategies. Later TEST evaluation is exploratory because these
dates were previously inspected. No packet-reliability, 3GPP-compliance or
practical-strategyproofness claim follows from this finite replay study.
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--assets',type=Path,default=ROOT.parent/'deadline_load_study/upstream')
    p.add_argument('--frozen-study',type=Path,default=ROOT.parent/'deadline_load_study')
    p.add_argument('--analysis-root',type=Path)
    p.add_argument('--model-root',type=Path)
    p.add_argument('--original-release',type=Path,default=ROOT.parents[1]/'outputs/icc2027_implementation')
    p.add_argument('--output',type=Path,help='new, nonexistent directory containing payload/ and archive parts/')
    p.add_argument('--cell',action='append',help='MODE/CONFIG/PROFILE, repeatable; overrides default six-hour requested scope')
    p.add_argument('--require-all',action='store_true',help='fail unless every requested cell is complete and analyzed')
    p.add_argument('--dry-run',action='store_true',help='validate inventory without writing files')
    p.add_argument('--diagnostic-smoke',action='store_true',help='TRAIN smoke only; requires --dry-run; never creates an archive')
    p.add_argument('--part-mib',type=int,default=64,choices=(64,),help='fixed archive part size')
    args=p.parse_args()
    for name in ('root','assets','frozen_study','original_release'):setattr(args,name,getattr(args,name).expanduser().resolve())
    args.analysis_root=(args.analysis_root or args.root/'analysis_matched').expanduser().resolve()
    args.model_root=(args.model_root or args.root/('smoke' if args.diagnostic_smoke else 'models')).expanduser().resolve()
    require(not args.diagnostic_smoke or args.dry_run,'Diagnostic smoke is dry-run only')
    require(args.dry_run or args.output is not None,'Provide a new --output directory or --dry-run')
    if args.output:
        args.output=args.output.expanduser().resolve()
        require(not args.output.exists(),'Refusing to overwrite existing output')
        require(not any(args.output.is_relative_to(x) for x in (args.assets,args.frozen_study,args.model_root,args.root/'evaluation')),'Bundle output must be separate from evidence and assets')
    analyzer=import_analyzer(args.root/'analyze_matched.py')
    inv,selected,excluded,scope,verifier=collect(args,analyzer)
    report=dict(scientific_result=not args.diagnostic_smoke,dry_run=args.dry_run,selected_cells=selected,omitted_requested_cells=excluded,
                requested_cells=[dict(mode=m,config=c,profile=pr) for m,c,pr in scope],all_requested_cells_complete=not excluded,
                full_nine_cell_two_mode_study_complete=len(selected)==18,files=len(inv.files),included_bytes=sum(r['bytes'] for r in inv.files.values()),
                external_prerequisite_files=len(inv.external),external_bytes=sum(r['bytes'] for r in inv.external.values()),
                optional_figures=inv.figure_status,
                frozen_gate_asset_files=len(inv.frozen_gate['files']),
                original_baseline_coverage=original_coverage(args,inv),verified_file_count=len(verifier.checked),bundle_script_sha256=sha(__file__))
    if args.dry_run:
        print(json.dumps(report,indent=2));return
    args.output.mkdir(parents=True,exist_ok=False);payload=args.output/'payload';payload.mkdir()
    for destination,row in sorted(inv.files.items()):
        target=payload/destination;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['source'],target)
        require(sha(target)==row['sha256'],'File changed while copying bundle')
    scope_record={k:v for k,v in report.items() if k!='original_baseline_coverage'}
    scope_record['created_utc']=datetime.now(timezone.utc).isoformat()
    generated={'BUNDLE_SCOPE.json':encoded(scope_record),'BASELINE_COVERAGE.json':encoded(report['original_baseline_coverage']),
               'EXTERNAL_REQUIREMENTS.json':encoded(dict(files=list(sorted(inv.external.values(),key=lambda r:r['path'])),automatic_download=False)),
               'BUNDLE_INVENTORY.json':encoded({'files':[{k:v for k,v in dict(path=d,**r).items() if k!='source'} for d,r in sorted(inv.files.items())]}),
               'FROZEN_GATE_REQUIREMENTS.json':encoded(inv.frozen_gate),
               'README.txt':instructions(True).encode()}
    for name,data in generated.items():(payload/name).write_bytes(data)
    files=sorted(p for p in payload.rglob('*') if p.is_file())
    (payload/'SHA256SUMS').write_text(''.join(sha(f)+'  '+f.relative_to(payload).as_posix()+'\n' for f in files))
    archive_record=archive(payload,args.output/'parts',args.part_mib*(1<<20))
    archive_record.update(scientific_result=True,scope_sha256=sha(payload/'BUNDLE_SCOPE.json'),payload_sha256sums_sha256=sha(payload/'SHA256SUMS'))
    (args.output/'ARCHIVE_MANIFEST.json').write_bytes(encoded(archive_record))
    print(json.dumps(dict(output=str(args.output),selected_cells=selected,omitted_requested_cells=excluded,
                         archive_bytes=archive_record['archive_bytes'],parts=len(archive_record['parts']),verified=True),indent=2))


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,FileNotFoundError) as error:
        raise SystemExit('BUNDLE REFUSED: '+str(error))
