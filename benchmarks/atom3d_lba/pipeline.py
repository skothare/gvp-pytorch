"""Fresh-start paper reproduction. CLI planning is stdlib-only; jobs are explicit."""
from __future__ import annotations
import argparse
import contextlib
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
COUNTS={'train':3507,'val':466,'test':490}
SPLITS=tuple(COUNTS)
SEEDS=[42,123,7,2026,17,29,53,101,211,307]
ARMS_A=['baseline','ep5','ep20']
ARMS_B=['baseline','ep20','raw_qr','random_ep']
CHECKPOINTS={'ep5':'eee0825fa6646bc2b64bd3352d3f4382f6a85cbe529364ecea62ed1e1e4da255',
 'ep20':'950e4183f32ec959ac2b16b262806a9427015e8666925100c963f0aff4d17748'}
PATHS=['estats_root','gvp_root','lmdb_root','ep5_checkpoint','ep20_checkpoint','output_root','estats_python','gvp_python']
METRICS=['rmse','pearson_r','spearman_r','r2']

def read(p): return json.loads(Path(p).read_text())
def file_stat(p):
    s=Path(p).stat();return [s.st_ino,s.st_size,s.st_mtime_ns]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def config(path):
    path=Path(path).resolve();c=read(path)
    for k in PATHS:
        if not c.get(k): raise ValueError('Missing configuration: '+k)
        p=Path(c[k]).expanduser();c[k]=str((path.parent/p).resolve())
    c['scheduler']=dict(cpu_partition='dept_cpu',gpu_partition='koes_gpu,dept_gpu',exclude='g021',
        controlled_constraint='L40',cpus=4,memory='32G',cpu_time='12:00:00',gpu_time='12:00:00',
        max_parallel=2,pqr_shards=32) | c.get('scheduler',{})
    for k in ['max_parallel','pqr_shards','cpus']:
        if not isinstance(c['scheduler'][k],int) or c['scheduler'][k]<1:raise ValueError('Invalid '+k)
    if c['scheduler']['max_parallel']>2:raise ValueError('At most two array tasks may run')
    out=Path(c['output_root'])
    for k in ['estats_root','gvp_root','lmdb_root']:
        p=Path(c[k])
        if out==p or out in p.parents or p in out.parents:raise ValueError('Output cannot contain source/data inputs')
    c['configuration_file']=str(path)
    return c

def bootstrap(c):
    sys.path.insert(0,str(Path(c['estats_root'])/'benchmarks/atom3d_lba'))
    sys.path.insert(0,c['gvp_root'])

def environment(c):
    env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_')}
    env.update(PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    env['PATH']=str(Path(c['estats_python']).parent)+os.pathsep+env.get('PATH','')
    return env

def stages(c):
    # Sequential dependencies cap total GPU concurrency across arrays at two.
    return [('pqr','cpu',3*c['scheduler']['pqr_shards']),('pqr_audit','cpu',None),
        ('cache_ep5','gpu',3),('audit_ep5','cpu',None),('cache_ep20','gpu',3),('audit_ep20','cpu',None),
        ('control_init','cpu',None),('controls','gpu',6),('controls_audit','cpu',None),
        ('diagnostics','controlled',None),('panel_a','controlled',5),('panel_b','gvp',40),
        ('ridge','cpu',2),('report','cpu',None)]

def task(stage,index):
    if stage=='panel_a': options=SEEDS[:5]
    elif stage=='panel_b': options=list(itertools.product(ARMS_B,SEEDS))
    elif stage=='ridge': options=['ep5','ep20']
    elif stage=='controls': options=list(itertools.product(['raw_qr','random_ep'],SPLITS))
    elif stage.startswith('cache_'): options=list(SPLITS)
    else: raise ValueError('No task mapping for '+stage)
    if index is None or not 0<=index<len(options):raise ValueError('Invalid task index')
    return options[index]

def plan(c):
    return dict(protocol='lba-paper-v1',production_trainings=55,ridge_models=2,
        counts=COUNTS,seeds_a=SEEDS[:5],seeds_b=SEEDS,
        stages=[dict(stage=s,resource=r,tasks=n or 1,after=stages(c)[i-1][0] if i else 'initialize')
                for i,(s,r,n) in enumerate(stages(c))])

def source_files(c):
    result=[]
    for k in ['estats_root','gvp_root']:
        result+=sorted((Path(c[k])/'benchmarks/atom3d_lba').rglob('*.py'))
        result+=sorted((Path(c[k])/'benchmarks/atom3d_lba').glob('*.sh'))
        result+=sorted((Path(c[k])/'benchmarks/atom3d_lba').glob('*.json'))
    result += [Path(c['gvp_root'])/'gvp'/x for x in ['__init__.py','atom3d.py','data.py']]
    return result

def reference_cohort():
    return read(HERE/'COHORT.json')

def preflight(c):
    for k in ['estats_root','gvp_root','lmdb_root']:
        if not Path(c[k]).is_dir():raise ValueError('Missing directory '+k)
    for arm in ['ep5','ep20']:
        if sha(c[arm+'_checkpoint'])!=CHECKPOINTS[arm]:raise ValueError('Wrong published checkpoint: '+arm)
    scripts={
      'estats_python':'import torch,numpy,pandas,scipy,sklearn,atom3d,lmdb,pdb2pqr,pdbfixer,openmm,pytest,matplotlib',
      'gvp_python':'import torch,numpy,pandas,scipy,sklearn,atom3d,lmdb,torch_geometric,torch_scatter,torch_cluster,pytest'}
    for python,imports in scripts.items():
        subprocess.run([c[python],'-s','-c',imports],env=environment(c),check=True)
    for split in SPLITS:
        if not (Path(c['lmdb_root'])/split/'data.mdb').is_file():raise ValueError('Missing LMDB split '+split)
    exe=Path(c['estats_python']).parent/'pdb2pqr30'
    if not exe.is_file():raise ValueError('pdb2pqr30 missing from extraction environment')
    return dict(passed=True,checkpoint_sha256=CHECKPOINTS)

def initialize(c):
    preflight(c);bootstrap(c)
    import v9_lba as v9
    from atom3d.datasets import LMDBDataset
    root=Path(c['output_root'])
    if root.exists():raise ValueError('Initialize requires a new output root; use existing run for stages')
    ids={};targets={};datasets={}
    for split,n in COUNTS.items():
        ds=LMDBDataset(str(Path(c['lmdb_root'])/split));ids[split]=v9.dataset_ids(ds)
        if len(ds)!=n or ids[split]!=list(ds.ids()):raise ValueError('Wrong split counts/order')
        targets[split]={ds[i]['id']:float(ds[i]['scores']['neglog_aff']) for i in range(len(ds))}
        datasets[split]=sha(Path(c['lmdb_root'])/split/'data.mdb')
        ref=reference_cohort()
        if ids[split]!=ref['split_ids'][split] or datasets[split]!=ref['dataset_sha256'][split]:
            raise ValueError('Dataset differs from published LBA-30 split: '+split)
        ds._env.close()
    for arm,epoch in [('ep5',4),('ep20',19)]:
        obj,_=v9.read_checkpoint(c[arm+'_checkpoint'])
        if obj['epoch']!=epoch:raise ValueError('Checkpoint epoch mismatch')
        del obj
        model=v9.load_v9_model(c[arm+'_checkpoint'],'cpu');del model
    packages={k:json.loads(subprocess.check_output([c[k],'-s','-m','pip','list','--format=json'],env=environment(c),text=True)) for k in ['estats_python','gvp_python']}
    info=dict(schema='lba-paper-v1',config=c,split_ids=ids,targets=targets,dataset_sha256=datasets,
        checkpoint_sha256=CHECKPOINTS,sources={str(p):sha(p) for p in source_files(c)},
        dataset_stat={s:file_stat(Path(c['lmdb_root'])/s/'data.mdb') for s in SPLITS},
        packages=packages,created_at=time.time(),epochs=50,batch=8,lr=1e-4,workers=4,
        train_time=0,val_time=0,lmdb_root=c['lmdb_root'],counts=COUNTS,
        caches={a:{'root':str(root/'caches'/a)} for a in ['ep5','ep20']})
    root.mkdir(parents=True);write(root/'run.json',info);write(root/'config.json',c)
    write(root/'test_targets.json',targets['test']);write(root/'PLAN.json',plan(c))
    print('INITIALIZED',root,flush=True)

def run_info(c):
    info=read(Path(c['output_root'])/'run.json')
    # configuration_file may point at the run-local copy after submission.
    a={k:v for k,v in info['config'].items() if k!='configuration_file'}
    b={k:v for k,v in c.items() if k!='configuration_file'}
    if a!=b:raise ValueError('Configuration changed after initialization')
    for p,h in info['sources'].items():
        if sha(p)!=h:raise ValueError('Release source changed: '+p)
    for a,h in CHECKPOINTS.items():
        if sha(c[a+'_checkpoint'])!=h:raise ValueError('Checkpoint changed')
    for split,signature in info['dataset_stat'].items():
        if file_stat(Path(c['lmdb_root'])/split/'data.mdb')!=signature:raise ValueError('Dataset file changed; re-audit required: '+split)
    bootstrap(c)
    return info

@contextlib.contextmanager
def locked(root,name):
    p=Path(root)/'locks'/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def receipt(root,path,files,**extra):
    root=Path(root)
    write(path,dict(run_sha256=sha(root/'run.json'),files={str(Path(p).relative_to(root)):sha(p) for p in files},
        completed_at=time.time(),node=platform.node(),job=os.environ.get('SLURM_JOB_ID'),**extra))

def verify(root,path):
    root=Path(root);r=read(path)
    if r['run_sha256']!=sha(root/'run.json'):raise ValueError('Foreign completion record')
    for p,h in r['files'].items():
        if sha(root/p)!=h:raise ValueError('Completed artifact changed: '+p)
    return r

def cache_audit(c,arm):
    import v9_lba as v9
    from check_v9_lba_cache import audit_split
    from atom3d.datasets import LMDBDataset
    root=Path(c['output_root']);cache=root/'caches'/arm
    rows=[audit_split(LMDBDataset(str(Path(c['lmdb_root'])/s)),cache,root/'pqr',c[arm+'_checkpoint'],s) for s in SPLITS]
    files=[v9.directory(cache,r,s)/v9.MANIFEST for r in v9.REPRESENTATIONS for s in SPLITS]
    receipt(root,cache/'audit.json',files,passed=True,splits=rows,checkpoint_sha256=CHECKPOINTS[arm])

def check_audits(c):
    root=Path(c['output_root'])
    for a in ['ep5','ep20']:verify(root,root/'caches'/a/'audit.json')

def initialize_random(checkpoint):
    import torch
    import v9_lba as v9
    from model_loader import load_model_module
    _,kw=v9.read_checkpoint(checkpoint);has=kw.pop('encoder_final_norm')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1729)
        model=load_model_module('V9','lba_random').create_model_with_distance_features(**kw)
    if not has:model.encoder.final_norm=torch.nn.Identity()
    return model.eval().requires_grad_(False)

def control_init(c):
    import torch
    import v9_lba as v9
    import control_features as f
    from model_loader import load_model_module
    from atom3d.datasets import LMDBDataset
    check_audits(c);root=Path(c['output_root']);dest=root/'controls';dest.mkdir(exist_ok=True)
    if (dest/'initialized.json').exists():raise ValueError('Control initialization already published')
    model=initialize_random(c['ep20_checkpoint'])
    model.eval().requires_grad_(False)
    v9.atomic_write(dest/'random_initialization.pt',model.state_dict(),tensor=True)
    def values():
        ds=LMDBDataset(str(Path(c['lmdb_root'])/'train'))
        for i in range(len(ds)):
            e=ds[i];_,lut,stems=v9.entry_inputs(e,'real',root/'pqr/train')
            yield f.raw_features(e,lut,stems)
    v9.atomic_write(dest/'scaler.json',f.fit_scaler(values()))
    receipt(root,dest/'initialized.json',[dest/'random_initialization.pt',dest/'scaler.json'],random_init_seed=1729)

def control_contract(c,arm,split):
    root=Path(c['output_root'])
    verify(root,root/'controls/initialized.json')
    return dict(schema='lba-paper-control-v1',arm=arm,split=split,width=2 if arm=='raw_qr' else 256,
        run_sha256=sha(root/'run.json'),scaler_sha256=sha(root/'controls/scaler.json'),
        random_state_sha256=sha(root/'controls/random_initialization.pt'))

def control_tensor(path,record,width):
    import torch
    import control_features as f
    x=torch.load(path,map_location='cpu',weights_only=True);f.validate_features(x,record['n_atoms'],width)
    if f.tensor_hash(x)!=record['tensor_sha256']:raise ValueError('Changed control tensor')
    return x

def prepare_control(c,arm,split,device='cuda'):
    import torch
    import v9_lba as v9
    import control_features as f
    from atom3d.datasets import LMDBDataset
    root=Path(c['output_root']);dest=root/'controls'/arm/split
    contract=control_contract(c,arm,split);ds=LMDBDataset(str(Path(c['lmdb_root'])/split));ids=v9.dataset_ids(ds)
    source=v9.read_manifest(v9.directory(root/'caches/ep20','decoder',split),ids=ids)
    with v9.shared.cache_lock(dest):
        p=dest/'_manifest.json'
        m=read(p) if p.exists() else dict(contract=contract,entries={},expected_ids=ids,complete=False,pending=None)
        if m['contract']!=contract or m['expected_ids']!=ids:raise ValueError('Wrong control contract/IDs')
        files={x.name for x in dest.glob('*.pt')};recorded={i+'_pocket.pt' for i in m['entries']}
        pending=m.get('pending');allowed=recorded|({pending['id']+'_pocket.pt'} if pending else set())
        if not recorded<=files or not files<=allowed:raise ValueError('Orphan or missing control tensors')
        model=None
        if arm=='random_ep':
            model=v9.load_v9_model(c['ep20_checkpoint'],device)
            model.load_state_dict(torch.load(root/'controls/random_initialization.pt',weights_only=True,map_location='cpu'),strict=True)
        scaler=read(root/'controls/scaler.json')
        for i,pid in enumerate(ids):
            elem=ds[i];record,lut,stems=v9.entry_inputs(elem,'real',root/'pqr'/split)
            if any(source['entries'][pid][k]!=v for k,v in record.items()):raise ValueError('Control input differs from EP')
            if pid in m['entries']:
                if any(m['entries'][pid][k]!=v for k,v in record.items()):raise ValueError('Changed control input')
                control_tensor(dest/(pid+'_pocket.pt'),m['entries'][pid],contract['width']);continue
            if pending and pending['id']==pid and (dest/(pid+'_pocket.pt')).exists():
                rec=pending['record']
                if any(rec[k]!=v for k,v in record.items()):raise ValueError('Changed pending inputs')
                control_tensor(dest/(pid+'_pocket.pt'),rec,contract['width'])
            else:
                if arm=='raw_qr':x=f.standardize(f.raw_features(elem,lut,stems),scaler)
                else:
                    batch,_=v9.df_to_v8_batch(elem['atoms_pocket'],elem['atoms_protein'],device,'real',lut,stems)
                    x=v9.extract_features(model,batch)['decoder']
                rec=dict(record,tensor_sha256=f.tensor_hash(x))
                m.update(complete=False,pending=dict(id=pid,record=rec));v9.atomic_write(p,m)
                v9.atomic_write(dest/(pid+'_pocket.pt'),x,tensor=True)
            m['entries'][pid]=rec;m['pending']=None;v9.atomic_write(p,m)
        m['complete']=True;v9.atomic_write(p,m)

def controls_audit(c):
    import v9_lba as v9
    from atom3d.datasets import LMDBDataset
    root=Path(c['output_root']);files=[]
    for arm,split in itertools.product(['raw_qr','random_ep'],SPLITS):
        dest=root/'controls'/arm/split;m=read(dest/'_manifest.json');ds=LMDBDataset(str(Path(c['lmdb_root'])/split))
        ids=v9.dataset_ids(ds)
        if (m['contract']!=control_contract(c,arm,split) or not m['complete'] or m['pending'] or
            set(m['entries'])!=set(ids) or m['expected_ids']!=ids or
            {p.name for p in dest.glob('*.pt')}!={i+'_pocket.pt' for i in ids}):raise ValueError('Invalid control cache')
        for i,pid in enumerate(ids):
            rec,_,_=v9.entry_inputs(ds[i],'real',root/'pqr'/split)
            if any(m['entries'][pid][k]!=v for k,v in rec.items()):raise ValueError('Control input changed')
            control_tensor(dest/(pid+'_pocket.pt'),m['entries'][pid],m['contract']['width'])
        files.append(dest/'_manifest.json')
    receipt(root,root/'controls/audit.json',files,passed=True)

def run_child(c,kind,out,**options):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    command=[c['gvp_python'],'-u',str(HERE/'worker.py'),'--config',str(Path(c['output_root'])/'config.json'),
             '--kind',kind,'--out',str(out)]
    for k,v in options.items():command += ['--'+k,str(v)]
    log=out.with_suffix('.'+kind+'.log')
    with log.open('x') as f:
        f.write('COMMAND_JSON: '+json.dumps(command)+'\n');f.flush()
        env=environment(c)
        if kind=='controlled':env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0')
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,env=env,cwd=c['gvp_root'])
    if re.search(r'Skipped batch due to OOM|Traceback \(most recent call last\)',log.read_text()):raise ValueError('Failed/partial training: '+str(log))
    return log

def attempt(root,name):
    suffix=os.environ.get('SLURM_JOB_ID','cpu')+'_r'+os.environ.get('SLURM_RESTART_COUNT','0')+'_'+str(time.time_ns())
    return Path(root)/'attempts'/name/suffix

def same_run(a,b):
    keys=['initial_sha256','shared_initial_sha256','trace','curves','final_sha256','best_state_sha256','metrics','predictions_sha256']
    return {k:a.get(k)==b.get(k) for k in keys}

def diagnostics(c):
    root=Path(c['output_root']);check_audits(c);paths=[];findings={}
    dest=attempt(root,'diagnostics')
    for seed in [42,307]:
        for native in [1,0]:
            records=[]
            for repeat in range(2):
                out=dest/f'baseline_{seed}_{native}_{repeat}'
                opts=dict(arm='baseline',seed=seed,native=native)
                if native:opts['steps']=16
                log=run_child(c,'controlled',out,**opts);records.append(read(out/'result.json'))
                paths += list(out.iterdir())+[log]
            findings[f'{seed}_{native}']=same_run(*records)
            if not native and not all(findings[f'{seed}_{native}'].values()):raise ValueError('Controlled repetition differed')
    for arm in ['ep5','ep20']:
        records=[]
        for repeat in range(2):
            out=dest/f'{arm}_{repeat}';log=run_child(c,'controlled',out,arm=arm,seed=42,steps=2)
            records.append(read(out/'result.json'));paths+=list(out.iterdir())+[log]
        findings[arm]=same_run(*records)
        if not all(findings[arm].values()):raise ValueError('Fusion repeatability failed')
    receipt(root,root/'diagnostics.json',paths,passed=True,findings=findings)

def train_a(c,index):
    root=Path(c['output_root']);verify(root,root/'diagnostics.json');check_audits(c)
    seed=task('panel_a',index);dest=attempt(root,f'a_seed{seed}');records={};paths=[]
    arms=ARMS_A[index%3:]+ARMS_A[:index%3]
    for arm in arms:
        out=dest/arm;log=run_child(c,'controlled',out,arm=arm,seed=seed)
        records[arm]=read(out/'result.json');paths+=list(out.iterdir())+[log]
    base=records['baseline']
    for arm in ['ep5','ep20']:
        if records[arm]['shared_initial_sha256']!=base['shared_initial_sha256']:raise ValueError('Backbone mismatch')
        for a,b in zip(records[arm]['curves'],base['curves']):
            if any(a[k]!=b[k] for k in ['train_order_sha256','val_order_sha256']):raise ValueError('Batch schedule mismatch')
    if records['ep5']['initial_sha256']!=records['ep20']['initial_sha256']:raise ValueError('Fusion initialization mismatch')
    receipt(root,root/'completed'/f'a_{seed}.json',paths,panel='a',seed=seed,
        results={a:str((dest/a/'result.json').relative_to(root)) for a in ARMS_A})

def train_b(c,index):
    root=Path(c['output_root']);check_audits(c);verify(root,root/'controls/audit.json')
    arm,seed=task('panel_b',index);out=attempt(root,f'b_{arm}_{seed}')
    log=run_child(c,'native_train',out,arm=arm,seed=seed)
    best=out/f'LBA_seed{seed}_best.pt'
    if not best.is_file():raise ValueError('Training did not produce best checkpoint')
    testlog=run_child(c,'native_test',out,arm=arm,seed=seed)
    result=validate_predictions(out/'predictions.csv',read(root/'run.json')['targets']['test'])
    write(out/'result.json',dict(arm=arm,seed=seed,metrics=result,curves=parse_curves(log)))
    receipt(root,root/'completed'/f'b_{arm}_{seed}.json',list(out.iterdir())+[log,testlog],panel='b',seed=seed,
        results={arm:str((out/'result.json').relative_to(root))})

def parse_curves(log):
    text=Path(log).read_text();rows={}
    for e,s,v in re.findall(r'EPOCH (\d+) (TRAIN|VAL)\s+loss: ([0-9.eE+-]+)',text):
        rows.setdefault(int(e),{'epoch_1based':int(e)+1})[s.lower()+'_loss']=float(v)
    if len(rows)!=50 or any('train_loss' not in r or 'val_loss' not in r for r in rows.values()):raise ValueError('Incomplete epoch log')
    return [rows[k] for k in sorted(rows)]

def validate_predictions(path,targets):
    import numpy as np
    from metrics import regression_metrics
    rows=list(csv.DictReader(Path(path).open()));ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(targets):raise ValueError('Wrong prediction IDs')
    y=np.array([float(r['target']) for r in rows]);p=np.array([float(r['prediction']) for r in rows])
    if not np.allclose(y,[targets[i] for i in ids],atol=1e-6,rtol=0):raise ValueError('Wrong prediction targets')
    result=regression_metrics(y,p)
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite metrics')
    return result

def ridge(c,index):
    import numpy as np
    import torch
    import v9_lba as v9
    from ridge import pool_residues,fit_probe
    from atom3d.datasets import LMDBDataset
    root=Path(c['output_root']);arm=task('ridge',index);verify(root,root/'caches'/arm/'audit.json')
    arrays={};ids={}
    for split in SPLITS:
        ds=LMDBDataset(str(Path(c['lmdb_root'])/split));x=[];y=[];ids[split]=v9.dataset_ids(ds)
        d=v9.directory(root/'caches'/arm,'decoder',split);m=v9.read_manifest(d,ids=ids[split])
        for i,pid in enumerate(ids[split]):
            elem=ds[i];value=v9.validate_entry(elem,d,m['entries'][pid],'real',root/'pqr'/split)
            x.append(pool_residues(value.numpy(),elem['atoms_pocket']));y.append(float(elem['scores']['neglog_aff']))
        arrays[split+'_x']=np.asarray(x,dtype=np.float64);arrays[split+'_y']=np.asarray(y,dtype=np.float64)
    scaler,model,search=fit_probe(arrays['train_x'],arrays['train_y'],arrays['val_x'],arrays['val_y'])
    out=attempt(root,'ridge_'+arm);out.mkdir(parents=True)
    p=model.predict(scaler.transform(arrays['test_x']))
    with (out/'predictions.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['id','target','prediction']);w.writerows(zip(ids['test'],arrays['test_y'],p))
    np.savez_compressed(out/'features.npz',**arrays)
    np.savez(out/'model.npz',coef=model.coef_,intercept=model.intercept_,mean=scaler.mean_,scale=scaler.scale_)
    write(out/'result.json',dict(arm=arm,alpha=model.alpha,validation_search=search,
        metrics=validate_predictions(out/'predictions.csv',read(root/'run.json')['targets']['test']),
        readout='linear ridge',fitting='train only; validation alpha selection; no train+val refit'))
    receipt(root,root/'completed'/f'ridge_{arm}.json',list(out.iterdir()),panel='c',seed=None,
        results={arm:str((out/'result.json').relative_to(root))})

def execute_stage(c,stage,index):
    info=run_info(c);root=Path(c['output_root'])
    name=stage+('_'+str(index) if index is not None else '')
    with locked(root,name):
        done=root/'stages'/f'{name}.json'
        if done.exists():verify(root,done);raise ValueError('Stage/task already completed')
        start=time.time()
        if stage=='pqr':
            n=c['scheduler']['pqr_shards']
            if index is None or not 0<=index<3*n:raise ValueError('Wrong PQR task index')
            subprocess.run([c['estats_python'],str(Path(c['estats_root'])/'benchmarks/atom3d_lba/build_pqr_cache.py'),
                '--lmdb-root',c['lmdb_root'],'--pqr-root',str(root/'pqr'),'--split',SPLITS[index//n],
                '--shard',str(index%n),'--n-shards',str(n),'--repair-inline'],check=True,env=environment(c))
        elif stage=='pqr_audit':
            import v9_lba as v9
            from atom3d.datasets import LMDBDataset
            statistics={}
            for split in SPLITS:
                subprocess.run([c['estats_python'],str(Path(c['estats_root'])/'benchmarks/atom3d_lba/build_pqr_cache.py'),
                    '--lmdb-root',c['lmdb_root'],'--pqr-root',str(root/'pqr'),'--split',split,'--aggregate-only'],check=True,env=environment(c))
                ds=LMDBDataset(str(Path(c['lmdb_root'])/split));unmatched=0;affected=0;entries={}
                for i in range(len(ds)):
                    record,_,_=v9.entry_inputs(ds[i],'real',root/'pqr'/split)
                    pid=ds[i]['id'];entries[pid]=record
                    repair=root/'pqr'/split/'_provenance'/(pid+'.json')
                    entries[pid]['repair_provenance_sha256']=sha(repair) if repair.exists() else None
                    n=sum(row[-1] for row in record['charge_stats']['unmatched']);unmatched+=n;affected+=int(n>0)
                statistics[split]=dict(count=len(ds),fallback_atoms=unmatched,affected_complexes=affected)
                v9.atomic_write(root/'pqr'/split/'_release_manifest.json',dict(schema='lba-paper-pqr-v1',entries=entries,split=split))
                ds._env.close()
            v9.atomic_write(root/'pqr/audit.json',dict(passed=True,statistics=statistics))
        elif stage.startswith('cache_'):
            import v9_lba as v9
            from precompute_v9_lba import precompute_split
            from atom3d.datasets import LMDBDataset
            arm=stage[6:];split=task(stage,index)
            if not read(root/'pqr/audit.json')['passed']:raise ValueError('PQR audit required')
            contracts={rep:v9.cache_contract(c[arm+'_checkpoint'],split,rep) for rep in v9.REPRESENTATIONS}
            model=v9.load_v9_model(c[arm+'_checkpoint'],'cuda')
            precompute_split(LMDBDataset(str(Path(c['lmdb_root'])/split)),model,root/'caches'/arm,'cuda',contracts,root/'pqr'/split)
        elif stage.startswith('audit_'):cache_audit(c,stage[6:])
        elif stage=='control_init':control_init(c)
        elif stage=='controls':prepare_control(c,*task(stage,index))
        elif stage=='controls_audit':controls_audit(c)
        elif stage=='diagnostics':diagnostics(c)
        elif stage=='panel_a':train_a(c,index)
        elif stage=='panel_b':train_b(c,index)
        elif stage=='ridge':ridge(c,index)
        elif stage=='report':
            from reporting import report
            report(c)
        else:raise ValueError('Unknown stage')
        run_info(c)
        receipt(root,done,[],stage=stage,index=index,seconds=time.time()-start)

def submit(c):
    run_info(c);root=Path(c['output_root']);journal=root/'submissions.jsonl'
    with locked(root,'submit'):
        if journal.exists():raise ValueError('Submission journal exists; do not duplicate a partial submission')
        previous=None;jobs={};s=c['scheduler']
        for stage,resource,count in stages(c):
            gpu=resource!='cpu';python=c['gvp_python'] if resource in ['gvp','controlled'] else c['estats_python']
            cmd=['sbatch','--parsable','--job-name=lba_'+stage,'--partition='+s['gpu_partition' if gpu else 'cpu_partition'],
                '--cpus-per-task='+str(s['cpus']),'--mem='+s['memory'],'--time='+s['gpu_time' if gpu else 'cpu_time'],
                '--output='+str(root/'slurm'/f'{stage}_%A_%a.out'),'--error='+str(root/'slurm'/f'{stage}_%A_%a.err')]
            if gpu:cmd+=['--gres=gpu:1']
            if gpu and s['exclude']:cmd+=['--exclude='+s['exclude']]
            if resource=='controlled':cmd+=['--constraint='+s['controlled_constraint']]
            if count:cmd+=['--array=0-'+str(count-1)+'%'+str(s['max_parallel'])]
            if previous:cmd+=['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes']
            cmd += [str(HERE/'job.sh'),python,str(HERE/'pipeline.py'),str(root/'config.json'),stage]
            (root/'slurm').mkdir(exist_ok=True)
            value=subprocess.check_output(cmd,text=True,env=environment(c)).strip().split(';')[0]
            if not value.isdigit():raise ValueError('Unexpected scheduler reply: '+value)
            with journal.open('a') as f:f.write(json.dumps(dict(stage=stage,job=value,command=cmd,time=time.time()))+'\n');f.flush();os.fsync(f.fileno())
            jobs[stage]=value;previous=value
        write(root/'jobs.json',jobs)
    return jobs

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--stage',required=True)
    action=p.add_mutually_exclusive_group();action.add_argument('--dry-run',action='store_true');action.add_argument('--submit',action='store_true');action.add_argument('--execute',action='store_true')
    p.add_argument('--index',type=int);a=p.parse_args();c=config(a.config)
    if a.stage=='reference-report':
        bootstrap(c)
        from reporting import reference_report
        reference_report(Path(c['output_root'])/'historical_reference_report');return
    if a.stage=='all':
        if a.dry_run: print(json.dumps(plan(c),indent=2));return
        if not a.submit:p.error('all requires --dry-run or explicit --submit')
        if not (Path(c['output_root'])/'run.json').exists():
            subprocess.run([c['estats_python'],str(HERE/'pipeline.py'),'--config',a.config,'--stage','initialize','--execute'],env=environment(c),check=True)
        print(json.dumps(submit(c),indent=2));return
    if a.submit:p.error('--submit is supported only for all; individual stages use --execute in an allocation')
    if a.stage=='preflight':print(json.dumps(preflight(c),indent=2));return
    if a.dry_run:print(json.dumps(plan(c),indent=2));return
    if not a.execute:p.error('A stage requires explicit --execute')
    if a.stage=='initialize':initialize(c);return
    index=a.index if a.index is not None else (int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else None)
    execute_stage(c,a.stage,index)

if __name__=='__main__':main()
