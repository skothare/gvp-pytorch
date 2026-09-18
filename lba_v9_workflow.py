"""V9 decoder GVP provenance, completion validation and paired reporting. No submissions."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
EVAL = ROOT.parent / 'Estats/dl_model/evaluate_proteinshake_V2'
sys.path.insert(0, str(EVAL))
COUNTS = dict(train=3507, val=466, test=490)
SEEDS = [42, 123, 7, 2026, 17, 29, 53, 101, 211, 307]
ARMS = ['v9_real', 'baseline']
SOURCES = ['run_atom3d.py', 'lba_v9_workflow.py', 'submit_lba_v9_array.sh',
           'gvp/atom3d.py', 'gvp/__init__.py', 'summarize_lba_v8.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def validate_cache(root):
    root = Path(root).resolve()
    audit = json.loads((root / 'audit.json').read_text())
    if (audit.get('passed') is not True or audit.get('model_family') != 'v9'
        or audit.get('cache_root') != str(root) or audit.get('charge_mode') != 'real'
        or {s['split']: s['count'] for s in audit['splits']} != COUNTS):
        raise ValueError('A complete V9 real-mode audit is required')
    checkpoint = audit['checkpoint_sha256']
    for rep in ('encoder', 'decoder'):
        for split, count in COUNTS.items():
            path = root / rep / 'charge_real' / split / '_manifest.json'
            if sha(path) != audit['manifest_sha256'][f'{rep}/real/{split}']:
                raise ValueError(f'Stale cache audit: {path}')
            m = json.loads(path.read_text()); c = m['contract']
            if (not m['complete'] or m.get('pending') or len(m['entries']) != count
                or set(m['entries']) != set(m['expected_ids']) or len(set(m['expected_ids'])) != count
                or c['checkpoint_sha256'] != checkpoint or c['representation'] != rep
                or c['split'] != split or c['charge_mode'] != 'real' or c['width'] != 256):
                raise ValueError('Invalid V9 manifest')
    # Confirm the sources underlying the audited extraction are still available unchanged.
    for name, digest in c['source_sha256'].items():
        if sha(ROOT.parent / 'Estats' / name) != digest:
            raise ValueError(f'Extraction source changed: {name}')
    return audit


def initialize(root, cache, phase):
    root = Path(root).resolve(); cache = Path(cache).resolve()
    audit = validate_cache(cache)
    info = dict(schema=1, phase=phase, representation='decoder', charge_mode='real',
                cache_root=str(cache), arms=ARMS, seeds=SEEDS if phase == 'production' else [42],
                epochs=50 if phase == 'production' else 1, batch=8, lr=1e-4, workers=4,
                train_time=0, val_time=0, audit_sha256=sha(cache/'audit.json'),
                source_sha256={s:sha(ROOT/s) for s in SOURCES},
                revisions={r:subprocess.check_output(['git','-C',str(ROOT.parent/r),'rev-parse','HEAD'],text=True).strip()
                           for r in ('Estats','gvp-pytorch')},
                git_status={r:subprocess.check_output(['git','-C',str(ROOT.parent/r),'status','--short'],text=True)
                            for r in ('Estats','gvp-pytorch')},
                environment=subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True),
                python=sys.executable, created_at=time.time())
    write_new(root/'run.json', info)
    print(f'Initialized {root}; tasks 0-{len(info["seeds"])*2-1}')


def config(root):
    root = Path(root); info=json.loads((root/'run.json').read_text())
    for s,h in info['source_sha256'].items():
        if sha(ROOT/s)!=h: raise ValueError(f'Source changed since initialization: {s}')
    if sha(Path(info['cache_root'])/'audit.json')!=info['audit_sha256']:
        raise ValueError('Audit changed since initialization')
    validate_cache(info['cache_root'])
    return info


def task(info, index):
    n=len(info['seeds'])
    if not 0 <= index < len(info['arms'])*n: raise ValueError('Task index out of range')
    return info['arms'][index//n], info['seeds'][index%n]


def validate_predictions(path, cache):
    from metrics import regression_metrics
    rows=list(csv.DictReader(Path(path).open()))
    ids=[r['id'] for r in rows]
    expected=json.loads((Path(cache)/'decoder/charge_real/test/_manifest.json').read_text())['expected_ids']
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected): raise ValueError('Prediction ID coverage mismatch')
    y=np.array([float(r['target']) for r in rows]); p=np.array([float(r['prediction']) for r in rows])
    if not np.isfinite(y).all() or not np.isfinite(p).all(): raise ValueError('Nonfinite predictions')
    metrics=regression_metrics(y,p)
    if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('Nonfinite metrics')
    return metrics


def finish(root, index, attempt):
    from summarize_lba_v8 import parse_log
    root=Path(root); attempt=Path(attempt).resolve(); info=config(root); arm,seed=task(info,index)
    logged=parse_log(attempt/'run.log',info['epochs'])
    metrics=validate_predictions(attempt/'predictions.csv',info['cache_root'])
    if any(abs(metrics[k]-logged[k])>0.00011 for k in logged): raise ValueError('Log/prediction metrics disagree')
    files=[attempt/'run.log',attempt/'predictions.csv',attempt/'attempt.json',
           attempt/'models'/f'LBA_seed{seed}_best.pt',attempt/'models'/f'LBA_seed{seed}_last.pt']
    for p in files:
        if not p.is_file() or p.stat().st_size==0: raise ValueError(f'Missing output {p}')
    write_new(root/'completed'/f'{arm}_{seed}.json', dict(arm=arm,seed=seed,metrics=metrics,
        run_sha256=sha(root/'run.json'), files={str(p):sha(p) for p in files}, completed_at=time.time()))


def summarize(root):
    root=Path(root);info=config(root); rows=[]
    for i in range(len(info['arms'])*len(info['seeds'])):
        arm,seed=task(info,i); rec=json.loads((root/'completed'/f'{arm}_{seed}.json').read_text())
        if rec['arm']!=arm or rec['seed']!=seed or rec['run_sha256']!=sha(root/'run.json'):
            raise ValueError('Wrong completion record')
        for p,h in rec['files'].items():
            if sha(p)!=h: raise ValueError(f'Changed result: {p}')
        pred=next(p for p in rec['files'] if p.endswith('/predictions.csv'))
        values=validate_predictions(pred,info['cache_root'])
        if values!=rec['metrics']: raise ValueError('Changed metrics')
        rows.append(dict(arm=arm,seed=seed,**values))
    keys=['rmse','pearson_r','spearman_r','r2']
    with (root/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['arm','seed']+keys);w.writeheader();w.writerows(rows)
    lines=['# V9 decoder real-mode GVP comparison', '', 'Means ± population SD; matched seeds. No significance claim.', '',
           '| Arm | RMSE | Pearson | Spearman | R² |','|---|---:|---:|---:|---:|']
    arrays={a:np.array([[r[k] for k in keys] for r in rows if r['arm']==a]) for a in ARMS}
    for a,x in arrays.items():
        lines.append('| '+a+' | '+' | '.join(f'{m:.4f} ± {s:.4f}' for m,s in zip(x.mean(0),x.std(0)))+' |')
    x=arrays['v9_real']-arrays['baseline']
    lines.append('| v9_real − baseline | '+' | '.join(f'{m:+.4f} ± {s:.4f}' for m,s in zip(x.mean(0),x.std(0)))+' |')
    (root/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','describe','finish','summarize'])
    p.add_argument('--run-root',required=True);p.add_argument('--cache-root');p.add_argument('--phase',choices=['rehearsal','production'],default='production')
    p.add_argument('--index',type=int);p.add_argument('--attempt');a=p.parse_args()
    if a.action=='initialize': initialize(a.run_root,a.cache_root,a.phase)
    elif a.action=='describe':
        c=config(a.run_root);arm,seed=task(c,a.index)
        print('\n'.join(map(str,[arm,seed,c['epochs'],c['cache_root']])))
    elif a.action=='finish': finish(a.run_root,a.index,a.attempt)
    else: summarize(a.run_root)

if __name__=='__main__': main()
