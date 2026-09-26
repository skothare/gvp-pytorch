"""New, isolated workflow. No imports from or mutations to historical runners."""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SEEDS=[42,123,7,2026,17]
ARMS=['baseline','ep5','ep20']
COUNTS={'train':3507,'val':466,'test':490}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def packages():
    return {r['name']:r['version'] for r in json.loads(subprocess.check_output(
        [sys.executable,'-m','pip','list','--format=json'],text=True))}


@contextlib.contextmanager
def lock(root,name):
    (root/'locks').mkdir(exist_ok=True)
    with (root/'locks'/name).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def initialize(root,seeds=None):
    from atom3d.datasets import LMDBDataset
    g=Path(__file__).resolve().parent; w=g.parent
    previous=g/'logs/ep20_lba_20260923T230422Z/run.json'
    historical=read(previous)
    e=w/'Estats/dl_model/evaluate_proteinshake_V2'
    caches={'ep5':e/'v9_lba_precomputed_cache_57359286',
            'ep20':w/historical['cache_root']}
    if root.exists():
        raise ValueError('Choose a new run root')
    seeds=SEEDS if seeds is None else seeds
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    info=dict(schema='unified-lba-v1',created_at=time.time(),seeds=seeds,arms=ARMS,epochs=50,
              batch=8,lr=1e-4,workers=4,train_time=0,val_time=0,
              lmdb_root=str(w/historical['lmdb_root']),counts=COUNTS,
              representation='decoder',charge_mode='real',caches={},input_sha256={},split_ids={},
              packages=packages(),python=sys.version,code_sha256={},
              hardware='NVIDIA L40',diagnostic_seeds=[42,307],
              protocol='independent loader RNGs; train/val shuffled; mean-batch validation MSE; CSR graph pooling; strict deterministic PyTorch; TF32 disabled')
    for name,digest in historical['dataset_sha256'].items():
        print('Checking dataset hash',name,flush=True)
        if sha(w/name)!=digest:
            raise ValueError('Dataset changed since EP20 experiment')
        info['input_sha256'][str(w/name)]=digest
    manifests={}
    for arm,cache in caches.items():
        audit=read(cache/'audit.json')
        if audit.get('passed') is not True or audit.get('charge_mode')!='real':
            raise ValueError('A passed real-mode cache audit is required')
        info['input_sha256'][str(cache/'audit.json')]=sha(cache/'audit.json')
        info['caches'][arm]=dict(root=str(cache),checkpoint_sha256=audit['checkpoint_sha256'])
        for split,count in COUNTS.items():
            p=cache/'decoder/charge_real'/split/'_manifest.json';m=read(p);c=m['contract']
            digest=audit['manifest_sha256'].get('decoder/'+split,
                       audit['manifest_sha256'].get('decoder/real/'+split))
            if sha(p)!=digest or m['complete'] is not True or len(m['entries'])!=count:
                raise ValueError('Cache audit/manifest mismatch')
            if (c['checkpoint_sha256']!=audit['checkpoint_sha256'] or c['width']!=256
                    or c['representation']!='decoder' or c['charge_mode']!='real'
                    or c['checkpoint_epoch']!={'ep5':4,'ep20':19}[arm]):
                raise ValueError('Wrong cache contract')
            info['input_sha256'][str(p)]=sha(p);manifests[arm,split]=m
    targets={}
    for split,count in COUNTS.items():
        ds=LMDBDataset(str(Path(info['lmdb_root'])/split));ids=list(ds.ids())
        if len(ids)!=count or len(set(ids))!=count:
            raise ValueError('Wrong cohort')
        info['split_ids'][split]=ids
        a,b=manifests['ep5',split],manifests['ep20',split]
        if (a['expected_ids']!=b['expected_ids'] or len(a['expected_ids'])!=count
                or set(a['expected_ids'])!=set(ids)):
            raise ValueError('Cache/dataset ID membership differs')
        if {k for k in a['contract'] if a['contract'][k]!=b['contract'][k]}!={'checkpoint_sha256','checkpoint_epoch'}:
            raise ValueError('Feature protocol differs')
        for pid in ids:
            for k in ['input_sha256','pqr_sha256','charge_stats','n_atoms']:
                if a['entries'][pid][k]!=b['entries'][pid][k]:
                    raise ValueError('Cache inputs differ')
        if split=='test':
            targets={ds[i]['id']:float(ds[i]['scores']['neglog_aff']) for i in range(len(ds))}
    files=['unified_lba_runner.py','unified_lba_workflow.py','submit_lba_unified_gpu.sh',
           'submit_lba_unified_cpu.sh','gvp/__init__.py','gvp/atom3d.py','gvp/data.py']
    root.mkdir(parents=True)
    for name in files:
        dest=root/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(g/name,dest);info['code_sha256'][name]=sha(dest)
    write_new(root/'test_targets.json',targets)
    info['targets_sha256']=sha(root/'test_targets.json')
    write_new(root/'run.json',info)
    print('INITIALIZED',root,flush=True)


def config(root,check_environment=True):
    info=read(root/'run.json')
    if info['schema']!='unified-lba-v1':
        raise ValueError('Wrong protocol')
    for name,digest in info['code_sha256'].items():
        if sha(root/'code'/name)!=digest:
            raise ValueError('Private code snapshot changed: '+name)
    # LMDBs were hashed at initialization; do not reread their full bytes per task.
    for name,digest in info['input_sha256'].items():
        if not name.endswith('data.mdb') and sha(name)!=digest:
            raise ValueError('Read-only cache provenance changed: '+name)
    if sha(root/'test_targets.json')!=info['targets_sha256']:
        raise ValueError('Targets changed')
    if check_environment and packages()!=info['packages']:
        raise ValueError('Conda environment changed after initialization')
    return info


def execute(root,arm,seed,out,native=False,steps=None):
    command=[sys.executable,'-u',str(root/'code/unified_lba_runner.py'),
             '--run-root',str(root),'--arm',arm,'--seed',str(seed),'--out',str(out)]
    if native:command+=['--native']
    if steps:command+=['--steps',str(steps)]
    out.parent.mkdir(parents=True,exist_ok=True)
    log=out.with_suffix('.log')
    with log.open('x') as f:
        f.write('COMMAND_JSON: '+json.dumps(command)+'\n');f.flush()
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=root/'code')
    return read(out/'result.json')


def receipt(root,paths,**extra):
    return dict(run_sha256=sha(root/'run.json'),files={str(p.relative_to(root)):sha(p) for p in paths},
                job=os.environ.get('SLURM_JOB_ID'),array=os.environ.get('SLURM_ARRAY_JOB_ID'),
                completed_at=time.time(),**extra)


def verify_receipt(root,path):
    r=read(path)
    if r['run_sha256']!=sha(root/'run.json'):
        raise ValueError('Completion belongs to a different run')
    for name,digest in r['files'].items():
        if sha(root/name)!=digest:
            raise ValueError('Completed artifact changed: '+name)
    return r


def comparison(a,b):
    keys=['initial_sha256','shared_initial_sha256','trace','curves','final_sha256',
          'best_state_sha256','metrics','predictions_sha256']
    return {k:a.get(k)==b.get(k) for k in keys}


def diagnostics(root):
    info=config(root)
    with lock(root,'diagnostics'):
        if (root/'DIAGNOSTICS_PASSED.json').exists():
            raise ValueError('Diagnostics already complete')
        attempt=root/'diagnostics'/str(os.environ['SLURM_JOB_ID'])
        attempt.mkdir(parents=True,exist_ok=False);findings={};paths=[]
        for seed in info['diagnostic_seeds']:
            for mode in ['native','controlled']:
                runs=[]
                for repeat in range(2):
                    out=attempt/f'{mode}_{seed}_repeat{repeat}'
                    r=execute(root,'baseline',seed,out,native=mode=='native',steps=16 if mode=='native' else None)
                    runs.append(r);paths.extend(p for p in out.iterdir() if p.is_file());paths.append(out.with_suffix('.log'))
                checks=comparison(*runs);findings[f'{mode}_{seed}']=checks
                if mode=='controlled' and not all(checks.values()):
                    write_new(attempt/'FAILED.json',findings)
                    raise ValueError('Controlled baseline repetition differed; production is blocked')
        for arm in ['ep5','ep20']:
            runs=[]
            for repeat in range(2):
                out=attempt/f'{arm}_smoke_repeat{repeat}'
                runs.append(execute(root,arm,42,out,steps=2))
                paths.extend(p for p in out.iterdir() if p.is_file());paths.append(out.with_suffix('.log'))
            checks=comparison(*runs);findings[arm+'_smoke']=checks
            if not all(checks.values()):
                write_new(attempt/'FAILED.json',findings)
                raise ValueError('Fusion repeatability failed; production is blocked')
        config(root)
        write_new(root/'DIAGNOSTICS_PASSED.json',receipt(root,paths,passed=True,checks=findings))
        print('DIAGNOSTICS_PASSED',flush=True)


def train(root,index):
    info=config(root);verify_receipt(root,root/'DIAGNOSTICS_PASSED.json')
    if index is None or not 0<=index<len(info['seeds']):
        raise ValueError('Invalid seed index')
    seed=info['seeds'][index]
    with lock(root,f'seed{seed}'):
        completed=root/'completed'/f'seed{seed}.json'
        if completed.exists():raise ValueError('Seed triad already complete')
        attempt=root/'attempts'/f'seed{seed}'/(os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0'))
        attempt.mkdir(parents=True,exist_ok=False);runs={};paths=[]
        # Rotate arm order, while retaining the same allocation for each triad.
        arms=info['arms'][index%3:]+info['arms'][:index%3]
        for arm in arms:
            out=attempt/arm;runs[arm]=execute(root,arm,seed,out)
            paths.extend(p for p in out.iterdir() if p.is_file());paths.append(out.with_suffix('.log'))
        baseline=runs['baseline']
        for arm in ['ep5','ep20']:
            if runs[arm]['shared_initial_sha256']!=baseline['shared_initial_sha256']:
                raise ValueError('Shared backbone initialization differs')
            for a,b in zip(runs[arm]['curves'],baseline['curves']):
                if any(a[k]!=b[k] for k in ['train_order_sha256','val_order_sha256']):
                    raise ValueError('Arm batch schedules differ')
        if runs['ep5']['initial_sha256']!=runs['ep20']['initial_sha256']:
            raise ValueError('Fusion initializations differ')
        config(root)
        write_new(completed,receipt(root,paths,seed=seed,arms=arms,
                  results={arm:str((attempt/arm/'result.json').relative_to(root)) for arm in arms}))
        print('SEED_TRIAD_COMPLETE',seed,flush=True)


def report(root):
    import numpy as np
    from scipy.stats import pearsonr,spearmanr
    def regression(y,p):
        y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError('Nonfinite predictions')
        return dict(rmse=float(np.sqrt(np.mean((y-p)**2))),pearson_r=float(pearsonr(y,p).statistic),
                    spearman_r=float(spearmanr(y,p).statistic),r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)))
    info=config(root,check_environment=False);verify_receipt(root,root/'DIAGNOSTICS_PASSED.json')
    targets=read(root/'test_targets.json');rows=[]
    for seed in info['seeds']:
        rec=verify_receipt(root,root/'completed'/f'seed{seed}.json')
        for arm in info['arms']:
            resultpath=root/rec['results'][arm];r=read(resultpath)
            pred=list(csv.DictReader(resultpath.with_name('predictions.csv').open()))
            ids=[p['id'] for p in pred]
            if ids!=info['split_ids']['test']:raise ValueError('Wrong test cohort')
            y=[float(p['target']) for p in pred];p=[float(p['prediction']) for p in pred]
            if not np.allclose(y,[targets[i] for i in ids],rtol=0,atol=1e-6):raise ValueError('Wrong labels')
            m=regression(y,p)
            if any(abs(m[k]-r['metrics'][k])>1e-12 for k in m):raise ValueError('Metric mismatch')
            rows.append(dict(arm=arm,seed=seed,**m))
    out=root/'report';out.mkdir(exist_ok=False)
    keys=['rmse','pearson_r','spearman_r','r2']
    with (out/'results.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['arm','seed']+keys);writer.writeheader();writer.writerows(rows)
    arrays={arm:np.array([[next(r for r in rows if r['arm']==arm and r['seed']==seed)[k] for k in keys] for seed in info['seeds']]) for arm in info['arms']}
    lines=['# Controlled LBA-30 comparison','',
           f"{len(info['seeds'])} matched seeds; means ± population SD. Seed variability is not a confidence interval. Full cohort; frozen real-mode decoder caches.",
           '', '| Model | RMSE | Pearson | Spearman | R² |','|---|---:|---:|---:|---:|']
    comparisons=dict(arrays)
    for a,b in [('ep5','baseline'),('ep20','baseline'),('ep20','ep5')]:
        comparisons[a+' minus '+b]=arrays[a]-arrays[b]
    for arm,x in comparisons.items():
        lines.append('| '+arm+' | '+' | '.join(f'{x[:,j].mean():.4f} ± {x[:,j].std():.4f}' for j in range(4))+' |')
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(14,4))
    for j,ax in enumerate(axes):
        for i,arm in enumerate(info['arms']):
            values=arrays[arm][:,j]
            ax.bar(i,values.mean(),yerr=values.std(),alpha=.55,capsize=4)
            ax.scatter(np.full(len(values),i),values,color='black',s=15)
        ax.set_xticks(range(3),info['arms']);ax.set_title(keys[j]);ax.set_ylabel('Test metric')
    fig.suptitle(f"Controlled LBA-30: {len(info['seeds'])} matched seeds, mean ± population SD")
    fig.tight_layout();fig.savefig(out/'metrics.png',dpi=200);fig.savefig(out/'metrics.pdf');plt.close(fig)
    write_new(out/'RESULTS_AUDIT.json',dict(passed=True,run_sha256=sha(root/'run.json'),evaluations=len(rows),test_count=len(targets),report_packages=packages()))
    print('UNIFIED_RESULTS_AUDIT_PASSED',out,flush=True)


def submit(root):
    info=config(root)
    if (root/'jobs.json').exists() or (root/'submissions.jsonl').exists():raise ValueError('Already submitted; inspect partial submissions before retrying')
    conda=Path(sys.executable).resolve().parents[3]
    gpu=root/'code/submit_lba_unified_gpu.sh';cpu=root/'code/submit_lba_unified_cpu.sh'
    def sbatch(arguments):
        output=subprocess.check_output(['sbatch','--parsable',*arguments],text=True).strip()
        job=output.split(';')[0]
        if not job.isdigit():raise ValueError('Unexpected Slurm response: '+output)
        with (root/'submissions.jsonl').open('a') as f:f.write(json.dumps(dict(job=job,args=arguments,time=time.time()))+'\n')
        return job
    diagnostic=sbatch(['--job-name=lba_repeatability','--time=12:00:00',
                        '--output='+str(root/'diagnostics_%j.out'),'--error='+str(root/'diagnostics_%j.err'),
                        str(gpu),'diagnostics',str(root),str(conda)])
    production=sbatch(['--job-name=lba_unified','--array=0-'+str(len(info['seeds'])-1)+'%2',
                       '--time=12:00:00','--dependency=afterok:'+diagnostic,'--kill-on-invalid-dep=yes',
                       '--output='+str(root/'production_%A_%a.out'),'--error='+str(root/'production_%A_%a.err'),
                       str(gpu),'train',str(root),str(conda)])
    reporting=sbatch(['--job-name=lba_unified_report','--time=01:00:00',
                      '--dependency=afterok:'+production,'--kill-on-invalid-dep=yes',
                      '--output='+str(root/'report_%j.out'),'--error='+str(root/'report_%j.err'),
                      str(cpu),str(root),str(conda)])
    jobs=dict(diagnostics=diagnostic,production=production,report=reporting)
    write_new(root/'jobs.json',jobs);print(json.dumps(jobs),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','diagnostics','train','report','submit'])
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--index',type=int)
    p.add_argument('--seeds',type=int,nargs='+')
    a=p.parse_args();root=a.run_root.resolve()
    if a.stage=='initialize':
        initialize(root,a.seeds)
    elif a.stage=='train':
        index=a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
        train(root,index)
    else:globals()[a.stage](root)


if __name__=='__main__':main()
