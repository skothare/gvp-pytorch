"""CPU-only final artifact audit and paper figure; existing runs are read-only."""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from atom3d.datasets import LMDBDataset

P=argparse.ArgumentParser();P.add_argument('--out',type=Path,required=True);args=P.parse_args()
G=Path(__file__).resolve().parent;W=G.parent
roots={k:G/'logs'/v for k,v in dict(parent='ep20_lba_20260923T230422Z',
    controls='lba_controls_20260924T040248Z',controlled='lba_unified_20260924T175340Z').items()}
keys=['rmse','pearson_r','spearman_r','r2']
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def met(y,p):
    y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    assert np.isfinite(y).all() and np.isfinite(p).all()
    return dict(rmse=float(np.sqrt(np.mean((y-p)**2))),pearson_r=float(pearsonr(y,p).statistic),
                spearman_r=float(spearmanr(y,p).statistic),r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)))
ds=LMDBDataset(str(W/'Estats/dl_model/evaluate_proteinshake_V2/data/atom3D/split-by-sequence-identity-30/data/test'))
targets={ds[i]['id']:float(ds[i]['scores']['neglog_aff']) for i in range(len(ds))}
assert len(targets)==490
rows=[];files_checked=0;warnings={};fatals=[];logs_checked=set();receipts=[]
fatal=re.compile(r'Traceback \(most recent call last\)|CUDA out of memory|Skipped batch due to OOM|slurmstepd: error|DUE TO TIME LIMIT|Segmentation fault|Killed process|(?:RuntimeError|ValueError|FileNotFoundError):',re.I)
def scan(p):
    if p in logs_checked:return
    logs_checked.add(p);text=p.read_text(errors='replace')
    for m in fatal.finditer(text):fatals.append(dict(path=str(p.relative_to(W)),text=text[max(0,m.start()-50):m.end()+180]))
    for line in text.splitlines():
        if re.search(r'\b(?:UserWarning|DeprecationWarning|FutureWarning|WARNING):',line):
            warnings.setdefault(line.strip(),set()).add(str(p.relative_to(W)))
def verify_receipt(root,p):
    global files_checked
    rec=read(p);assert rec['run_sha256']==sha(root/'run.json')
    for f,h in rec['files'].items():
        f=Path(f) if Path(f).is_absolute() else root/f
        assert sha(f)==h,str(f);files_checked+=1
        if f.suffix=='.log':scan(f)
    receipts.append(dict(path=str(p.relative_to(W)),sha256=sha(p)))
    return rec
def prediction_row(group,arm,seed,p,expected):
    data=list(csv.DictReader(p.open()));ids=[r['id'] for r in data]
    assert len(ids)==len(set(ids))==490 and set(ids)==set(targets)
    y=[float(r['target']) for r in data];z=[float(r['prediction']) for r in data]
    assert np.allclose(y,[targets[i] for i in ids],rtol=0,atol=1e-6)
    m=met(y,z);assert all(abs(m[k]-expected[k])<1e-12 for k in keys)
    row=dict(group=group,arm=arm,seed=seed,**m,prediction_file=str(p.relative_to(W)))
    rows.append(row);return row

for group in ['parent','controls']:
    root=roots[group];info=read(root/'run.json');count=0
    for p in sorted((root/'completed').glob('*.json')):
        rec=verify_receipt(root,p)
        pred=root/next(f for f in rec['files'] if f.endswith('/predictions.csv'))
        prediction_row(group,rec['arm'],rec['seed'],pred,rec['metrics']);count+=1
        if rec['seed'] is not None:
            text=pred.with_name('run.log').read_text()
            assert 'TRAIN_OK' in text and 'TEST_OK' in text
            for stage in ['TRAIN','VAL']:
                curve=re.findall(r'EPOCH (\d+) '+stage+r'\s+loss:\s*(\S+)',text)
                assert [int(i) for i,v in curve]==list(range(50))
                assert np.isfinite([float(v) for i,v in curve]).all()
    assert count=={'parent':21,'controls':30}[group]
    for arm in info['arms']:
        assert sorted(r['seed'] for r in rows if r['group']==group and r['arm']==arm)==sorted(info['seeds'])
    print('VERIFIED',group,count,flush=True)

root=roots['controlled'];info=read(root/'run.json')
diag=verify_receipt(root,root/'DIAGNOSTICS_PASSED.json');assert diag['passed']
for k,v in diag['checks'].items():
    if not k.startswith('native'):assert all(v.values())
for seed in info['seeds']:
    rec=verify_receipt(root,root/'completed'/f'seed{seed}.json');assert rec['seed']==seed
    results={}
    for arm in info['arms']:
        p=root/rec['results'][arm];r=read(p);results[arm]=r
        assert r['seed']==seed and r['arm']==arm and r['strict'] and r['deterministic_algorithms']
        assert r['gpu']=='NVIDIA L40' and len(r['curves'])==50
        assert [c['epoch_1based'] for c in r['curves']]==list(range(1,51))
        for c in r['curves']:
            assert c['train_count']==3507 and c['val_count']==466
            assert np.isfinite([c['train_loss'],c['val_loss']]).all()
        assert r['best_epoch_1based']==int(np.argmin([c['val_loss'] for c in r['curves']]))+1
        prediction_row('controlled',arm,seed,p.with_name('predictions.csv'),r['metrics'])
    for arm in ['ep5','ep20']:
        assert results[arm]['shared_initial_sha256']==results['baseline']['shared_initial_sha256']
        for a,b in zip(results[arm]['curves'],results['baseline']['curves']):
            assert all(a[k]==b[k] for k in ['train_order_sha256','val_order_sha256'])
    assert results['ep5']['initial_sha256']==results['ep20']['initial_sha256']
assert len(rows)==66
print('VERIFIED controlled 15 plus diagnostic receipts',flush=True)

for root in roots.values():
    for p in root.iterdir():
        if p.suffix in ['.err','.out']:scan(p)
assert not fatals,fatals
# Check published CSVs against independently recomputed predictions.
for group in ['controls','controlled']:
    for r in csv.DictReader((roots[group]/'report/results.csv').open()):
        seed=int(r['seed']) if r['seed'] else None
        source=group if group=='controlled' or r['arm'] in ['raw_qr','random_ep','shuffled_ep'] else 'parent'
        v=next(v for v in rows if v['group']==source and v['arm']==r['arm'] and v['seed']==seed)
        assert all(abs(float(r[k])-v[k])<1e-12 for k in keys)
aggregates={};paired={}
for group,root in roots.items():
    for arm in sorted({r['arm'] for r in rows if r['group']==group}):
        selected=[r for r in rows if r['group']==group and r['arm']==arm]
        a=np.array([[r[k] for k in keys] for r in selected])
        aggregates[group+'/'+arm]=dict(n=len(selected),metrics={k:dict(mean=float(a[:,j].mean()),sd_ddof0=float(a[:,j].std())) for j,k in enumerate(keys)})
def paired_summary(ga,a,gb,b,seeds):
    aa=np.array([[next(r for r in rows if r['group']==ga and r['arm']==a and r['seed']==s)[k] for k in keys] for s in seeds])
    bb=np.array([[next(r for r in rows if r['group']==gb and r['arm']==b and r['seed']==s)[k] for k in keys] for s in seeds])
    delta=aa-bb
    return dict(seeds=seeds,per_seed=delta.tolist(),metrics={k:dict(mean=float(delta[:,j].mean()),sd_ddof0=float(delta[:,j].std()),wins=int(((delta[:,j]<0) if k=='rmse' else (delta[:,j]>0)).sum())) for j,k in enumerate(keys)})
for a,b in [('ep5','baseline'),('ep20','baseline'),('ep20','ep5')]:paired['controlled/'+a+'-'+b]=paired_summary('controlled',a,'controlled',b,info['seeds'])
ten=read(roots['controls']/'run.json')['seeds']
for a in ['raw_qr','random_ep','shuffled_ep']:
    paired['controls/'+a+'-ep20']=paired_summary('controls',a,'parent','ep20',ten)
report=dict(passed=True,created_at=datetime.now(timezone.utc).isoformat(),test_records=490,evaluations=66,
            artifact_hashes_checked=files_checked,log_files_scanned=len(logs_checked),fatal_notices=fatals,
            warnings={k:sorted(v) for k,v in warnings.items()},source_receipts=receipts,aggregates=aggregates,paired=paired,
            scope='Saved artifact/checkpoint hashes, completed training curves, held-out labels from LMDB, recomputed predictions/metrics, diagnostic receipt and controlled initialization/batch schedules. No GPU checkpoint replay or repeat cache extraction.',
            scheduler_limit='sacct database connection refused; scontrol no longer retains the final report jobs. Scheduler exit codes could not be independently retrieved.')
args.out.mkdir(parents=True,exist_ok=False)
(args.out/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
with (args.out/'verified_results.csv').open('w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(8.2,3.6));arms=['baseline','ep5','ep20']
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
for j,k in enumerate(['rmse','pearson_r']):
    ax=axes[j]
    for seed,color in zip(info['seeds'],colors):
        values=[next(r[k] for r in rows if r['group']=='controlled' and r['seed']==seed and r['arm']==a) for a in arms]
        ax.plot(range(3),values,'o-',color=color,alpha=.8,lw=1,markersize=4,label=str(seed))
    means=[aggregates['controlled/'+a]['metrics'][k]['mean'] for a in arms]
    ax.scatter(range(3),means,color='black',marker='D',s=35,zorder=5,label='Mean')
    ax.set_xticks(range(3),['GVP','GVP + EP-5','GVP + EP-20'])
    ax.set_ylabel('Test RMSE (lower is better)' if k=='rmse' else 'Test Pearson r (higher is better)')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,title='Matched seed',loc='upper center',ncol=6,bbox_to_anchor=(.5,1.02),frameon=False)
fig.tight_layout(rect=(0,0,1,.87))
for ext in ['pdf','png','svg']:fig.savefig(args.out/('controlled_paired_seeds.'+ext),dpi=250,bbox_inches='tight')
plt.close(fig)
lines=['# Completed LBA experiment audit','',f'All 66 reported evaluations verified against 490 canonical test labels. {files_checked} artifact hashes checked; {len(logs_checked)} logs scanned; no matched fatal failure notices.',
       '', 'Slurm accounting is unavailable from this session; completed report jobs have expired from scontrol. Completion claims are based on saved receipts, artifacts and logs.',
       '', '| Group / arm | n | RMSE | Pearson | Spearman | R² |','|---|---:|---:|---:|---:|---:|']
for name,a in aggregates.items():
    lines.append('| '+name+' | '+str(a['n'])+' | '+' | '.join(f"{a['metrics'][k]['mean']:.4f} ± {a['metrics'][k]['sd_ddof0']:.4f}" if a['n']>1 else f"{a['metrics'][k]['mean']:.4f}" for k in keys)+' |')
lines+=['','`verified_results.csv` contains all per-seed values and prediction-file references. `AUDIT.json` records checks, warnings, paired differences and evidence hashes.',
        '', 'Use the controlled three-arm table as a distinct result block. Show the ten-seed EP-20 feature controls separately; do not combine their baselines or substitute EP-5 into EP-20 control comparisons.',
        '', 'The paired-seed figure displays all five seeds rather than hiding baseline seed 7. Black diamonds are means; there are no confidence intervals or significance markings.',
        '', 'Interpretation: EP-5 improves all four average metrics over its controlled baseline (4/5 paired seeds), and outperforms EP-20 in all five seeds. EP-20 does not improve mean RMSE versus baseline. Random EP is worse than pretrained EP-20; one random initialization does not characterize the entire random-model distribution. Within-pocket shuffle preserves the vector multiset and most performance; these results do not demonstrate that correct atom-vector assignment is necessary. Pretraining-membership/resume differences preclude an epoch-only causal claim.']
(args.out/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:report[k] for k in ['passed','evaluations','artifact_hashes_checked','log_files_scanned','fatal_notices','aggregates','paired']},indent=2))
print('WARNING_CLASSES',json.dumps({k:len(v) for k,v in report['warnings'].items()}))
print('OUTPUT',args.out)
