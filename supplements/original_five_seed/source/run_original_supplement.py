#!/usr/bin/env python3
"""Original-method five-seed headline replay; separate exploratory protocol/output."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,importlib.metadata,json,multiprocessing,os,platform,sys,time,traceback
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def json_default(x):
 if hasattr(x,'tolist'):return x.tolist()
 if hasattr(x,'item'):return x.item()
 if isinstance(x,Path):return str(x)
 raise TypeError(f'Cannot encode {type(x).__name__}')
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False,default=json_default)+'\n');t.replace(p)
def immutable(p,x):
 if Path(p).exists():
  if json.loads(Path(p).read_text())!=x:raise ValueError(f'Existing specification differs: {p}')
 else:atomic(p,x)
def imports(assets,support):
 sys.path[:0]=[str(Path(assets)/'code'),str(Path(assets)/'code/experiments'),str(support)]
 import numpy as np,torch
 torch.set_num_threads(1)
 from study_support import safe_load_net
 from slicemech.data.calibrate import Calibration
 from slicemech.data import puffer_fetch
 from slicemech.config import Mechanism
 from slicemech.sim import episode as E
 from slicemech.tenants.strategies import Truthful
 import e2_manipulation as X
 return dict(np=np,torch=torch,safe_load_net=safe_load_net,Calibration=Calibration,puffer_fetch=puffer_fetch,Mechanism=Mechanism,E=E,Truthful=Truthful,X=X)
G={}
def init_worker(assets,support,out,spec):
 G.update(imports(assets,support));G.update(assets=Path(assets),out=Path(out),spec=spec,nets={})
 G['cal']=G['Calibration'].from_dict(json.loads((Path(assets)/'data/calibration/n3_default.json').read_text()));G['mech']=G['Mechanism']()
def net(arm,seed):
 k=(arm,seed)
 if k not in G['nets']:
  p=G['assets']/f'results/models/default/{arm}/seed{seed}'
  G['nets'][k]=G['safe_load_net'](p/'model.pt',config_path=p/'config.json')
 return G['nets'][k]
def case_path(out,seed,date,offset):return Path(out)/'cases'/f'seed{seed}'/f'{date}_o{offset}'
def completed(p,spec_sha):
 f=p/'COMPLETE.json'
 if not f.exists():return False
 j=json.loads(f.read_text())
 if j.get('spec_sha256')!=spec_sha:raise ValueError(f'Existing case specification differs: {p}')
 for n,h in j['files_sha256'].items():
  if not (p/n).is_file() or sha(p/n)!=h:raise ValueError(f'Completed output changed: {p/n}')
 return True

def run_case(job):
 seed,date,offset=job;p=case_path(G['out'],seed,date,offset);ss=digest(G['spec'])
 if completed(p,ss):return dict(seed=seed,date=date,offset=offset,resumed=True,seconds=0.)
 p.mkdir(parents=True,exist_ok=True);start=time.perf_counter();np=G['np'];E=G['E'];X=G['X'];Truthful=G['Truthful'];cal=G['cal'];mech=G['mech'];times={}
 atomic(p/'STATUS.json',dict(state='running',pid=os.getpid(),seed=seed,date=date,offset=offset,start_unix=time.time(),phase='compose'))
 w=cal.window('test',date,offset,test_freeze=G['assets']/'prereg/PREREG.md')
 assert w.K==200 and w.warmup==40 and np.array_equal(w.contract.delay_ms,[2002,150,300])
 ah=hashlib.sha256()
 for name in ('arrivals','bytes_ms','theta','rate_hat','rate_true'):
  v=getattr(w,name,None)
  if v is not None:
   z=np.ascontiguousarray(v);ah.update(name.encode());ah.update(str((z.shape,z.dtype.str)).encode());ah.update(z.tobytes())
 metadata=dict(seed=seed,date=date,offset=offset,window_id=w.id,K=w.K,warmup=w.warmup,window_input_sha256=ah.hexdigest(),window_meta_sha256=hashlib.sha256(json.dumps(getattr(w,'meta',{}),sort_keys=True,default=json_default).encode()).hexdigest(),spec_sha256=ss)
 # The original window metadata uses ordinary JSON-compatible types.
 atomic(p/'case.json',metadata)
 truth=None;files={}
 for label,arm,charge,dev in [('unconstrained_truth','unconstrained','fixed',None),('protected_truth','protected','full',None)]+[(f'unconstrained_br{i}','unconstrained','fixed',i) for i in range(w.n)]:
  atomic(p/'STATUS.json',dict(state='running',pid=os.getpid(),seed=seed,date=date,offset=offset,phase=label,elapsed_seconds=time.perf_counter()-start))
  pols=[Truthful() for _ in range(w.n)]
  if dev is not None:pols[dev]=X.ScoredBestResponder(charge,w.warmup)
  t=time.perf_counter();df=E.run_window(w,net(arm,seed),E.PriceRule(charge),pols,2,mech,cal.shape,price_columns='billed');times[label]=time.perf_counter()-t
  if label=='unconstrained_truth':truth=df
  if dev is not None:
   for c in ('report','quota','on_time_Mbit','dropped_Mbit'):
    assert np.array_equal(df.loc[df.k<w.warmup,c].to_numpy(),truth.loc[truth.k<w.warmup,c].to_numpy()),f'Warm-up mismatch: {c}'
  df.attrs={};df.insert(0,'profile',label);df.insert(0,'seed',seed);df.insert(0,'window',w.id);df.insert(0,'date',date);df['deviator']=-1 if dev is None else dev
  target=p/f'{label}.parquet';tmp=p/f'{label}.{os.getpid()}.tmp';df.to_parquet(tmp,index=False);tmp.replace(target);files[target.name]=sha(target)
 files['case.json']=sha(p/'case.json')
 result=dict(complete=True,scientific_result=True,exploratory=True,seed=seed,date=date,offset=offset,spec_sha256=ss,seconds=time.perf_counter()-start,times_seconds=times,files_sha256=files,identical_truthful_warmup=True)
 atomic(p/'COMPLETE.json',result);atomic(p/'STATUS.json',dict(state='complete',seconds=result['seconds']))
 return {k:result[k] for k in ('seed','date','offset','seconds','times_seconds')}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--frozen-support',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=20);ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
 if not 1<=a.workers<=20:raise ValueError('workers must be1..20')
 for n in ('assets','frozen_support','output'):setattr(a,n,getattr(a,n).resolve())
 if a.output==a.assets or a.output.is_relative_to(a.assets):raise ValueError('Supplemental outputs must be outside recovered assets')
 protocol=a.output/'PROTOCOL.txt'
 if not protocol.is_file():raise ValueError('A separate frozen supplemental protocol is required')
 api=imports(a.assets,a.frozen_support)
 gate=api['puffer_fetch'].require_test_read_gate(a.assets/'prereg/PREREG.md')
 git=json.loads((a.frozen_support/'upstream_asset_manifest.json').read_text());par=json.loads((a.frozen_support/'puffer_archive_recovery_manifest.json').read_text());expected={r['source_path'].split('ICC/rewrite_2026-10-18/',1)[1]:r['sha256'] for r in git['files']};expected.update({r['relative_output'].removeprefix('upstream/'):r['sha256'] for r in par['tables']})
 keys=api['X'].replay_keys(api['X'].split_keys('test'),2)
 if len(keys)!=32 or len(set(d for d,o in keys))!=16:raise ValueError('Original replay window grid differs')
 required=['data/calibration/n3_default.json']
 for arm in ('protected','unconstrained'):
  for seed in range(5):required.extend([f'results/models/default/{arm}/seed{seed}/{n}' for n in ('model.pt','config.json')])
 for d in sorted(set(d for d,o in keys)):required.extend([f'data/puffer/{d}.{suffix}' for suffix in ('json','parquet')])
 for folder in ('tractor','raca'):
  required.extend(str(p.relative_to(a.assets)) for p in sorted((a.assets/'data'/folder).iterdir()) if p.is_file() and p.suffix in ('.json','.parquet','.npz'))
 inputs={}
 for rel in required:
  h=sha(a.assets/rel)
  if expected.get(rel)!=h:raise ValueError(f'Recovered source hash mismatch: {rel}')
  inputs[rel]=h
 models={}
 for arm in ('protected','unconstrained'):
  for seed in range(5):
   p=a.assets/f'results/models/default/{arm}/seed{seed}';n=api['safe_load_net'](p/'model.pt',config_path=p/'config.json');models[f'{arm}/seed{seed}']=dict(n.study_checkpoint)
 sources={str(p.relative_to(a.assets)):sha(p) for p in sorted((a.assets/'code').rglob('*')) if p.is_file() and p.suffix!='.pyc' and not any(x.startswith('.') or x=='__pycache__' for x in p.relative_to(a.assets/'code').parts)}
 sources['supplemental_runner']=sha(__file__);sources['study_support.py']=sha(a.frozen_support/'study_support.py')
 spec=dict(study='original_setting_five_seed_headline_supplement',scientific_result=True,exploratory=True,protocol_sha256=sha(protocol),input_sha256=inputs,source_sha256=sources,native_test_gate=gate,origin=dict(repository=git['repository'],revision=git['revision']),models=models,seeds=list(range(5)),window_keys=[list(k) for k in keys],K=200,warmup=40,episode_profiles=['unconstrained_truth','protected_truth','unconstrained_br0','unconstrained_br1','unconstrained_br2'],versions={n:importlib.metadata.version(n) for n in ('numpy','pandas','pyarrow','torch','numba','scipy','PyYAML')},python=platform.python_version())
 immutable(a.output/'SPECIFICATION.json',spec)
 jobs=[(s,d,int(o)) for d,o in keys for s in range(5)]
 atomic(a.output/'PREFLIGHT.json',dict(complete=True,native_test_gate_passed=True,models_verified=len(models),input_files_verified=len(inputs),cases=len(jobs),trajectories=5*len(jobs),workers=a.workers,spec_sha256=digest(spec),utc_unix=time.time()))
 print(json.dumps(dict(preflight='passed',cases=len(jobs),models_verified=len(models),prepare_only=a.prepare_only)),flush=True)
 if a.prepare_only:return
 pending=[j for j in jobs if not completed(case_path(a.output,*j),digest(spec))];done=[];errors=[];start=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork'),initializer=init_worker,initargs=(str(a.assets),str(a.frozen_support),str(a.output),spec)) as pool:
  futures={pool.submit(run_case,j):j for j in pending}
  for f in concurrent.futures.as_completed(futures):
   try:r=f.result();done.append(r);event={'event':'case_complete',**r}
   except Exception as e:
    r=dict(job=futures[f],error=str(e),traceback=traceback.format_exc());errors.append(r);event={'event':'case_failed',**r}
   print(json.dumps(event),flush=True)
   with (a.output/'progress.jsonl').open('a') as h:h.write(json.dumps(event)+'\n')
   atomic(a.output/'CAMPAIGN_STATUS.json',dict(complete=False,expected_cases=160,previously_complete=len(jobs)-len(pending),newly_completed=len(done),errors=errors,elapsed_seconds=time.perf_counter()-start))
 final=dict(complete=not errors and all(completed(case_path(a.output,*j),digest(spec)) for j in jobs),expected_cases=160,trajectories=800,seconds=time.perf_counter()-start,errors=errors,spec_sha256=digest(spec))
 atomic(a.output/'CAMPAIGN_COMPLETE.json',final);print(json.dumps(final),flush=True)
 if not final['complete']:raise SystemExit(2)
if __name__=='__main__':main()
