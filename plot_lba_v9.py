"""Audit completed V9 runs and plot their saved test metrics/learning curves on CPU."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lba_v9_workflow import summarize, validate_predictions
from summarize_lba_v8 import parse_log


def main(root):
    root=Path(root).resolve()
    summarize(root)  # Revalidate source, cache audit and completion-bound artifact hashes.
    info=json.loads((root/'run.json').read_text()); seeds=info['seeds']
    keys=['rmse','pearson_r','spearman_r','r2']
    titles=['RMSE ↓','Pearson r ↑','Spearman ρ ↑','R² ↑']
    arms=['baseline','v9_real'];labels=['GVP','GVP + V9 decoder (real)'];colors=['#526677','#00897b']
    metrics={};curves={};warnings=Counter();records=[];targets=None
    failure=re.compile(r'Traceback \(most recent call last\)|Skipped batch due to OOM|CUDA out of memory|DUE TO TIME LIMIT|CANCELLED|Segmentation fault')
    for arm in arms:
        for seed in seeds:
            receipt=json.loads((root/'completed'/f'{arm}_{seed}.json').read_text())
            log=next(Path(p) for p in receipt['files'] if p.endswith('/run.log'))
            text=log.read_text(); logged=parse_log(log,info['epochs'])
            pred=log.parent/'predictions.csv'; values=validate_predictions(pred,info['cache_root'])
            if any(abs(values[k]-logged[k])>0.00011 for k in keys):raise ValueError('Metrics/log mismatch')
            rows=list(csv.DictReader(pred.open())); ys={r['id']:float(r['target']) for r in rows}
            if targets is None:targets=ys
            elif ys!=targets:raise ValueError('Test targets disagree between runs')
            metrics[arm,seed]=[values[k] for k in keys]
            for line in text.splitlines():
                if re.search(r'\b\w*Warning:',line):warnings[line]+=1
            for stage in ['TRAIN','VAL']:
                pairs=re.findall(rf'EPOCH (\d+) {stage}\s+loss:\s*(\S+)',text)
                values=np.array([float(v) for _,v in pairs])
                if len(values)!=info['epochs'] or not np.isfinite(values).all():raise ValueError('Invalid learning curve')
                curves[arm,seed,stage]=values
            attempt=json.loads((log.parent/'attempt.json').read_text())
            for suffix in ['out','err']:
                path=root.parent/f"v9_train_{attempt['array']}_{attempt['task']}.{suffix}"
                content=path.read_text()
                if failure.search(content):raise ValueError(f'Failure in {path}')
                if suffix=='out' and f'TASK_COMPLETE arm={arm} seed={seed}' not in content:
                    raise ValueError(f'Missing scheduler completion marker: {path}')
                if suffix=='err' and content.strip():raise ValueError(f'Scheduler stderr requires review: {path}')
            records.append(dict(arm=arm,seed=seed,**attempt,
                                best_logged_epoch_1based=int(np.argmin(curves[arm,seed,'VAL']))+1))
    audit=dict(passed=True,runs=records,test_records=len(targets),warnings=dict(warnings),
               scope='Artifact/log audit; no new inference, chemical audit, or Slurm accounting query.')
    (root/'RESULTS_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (root/'RESULTS_AUDIT.md').write_text('# V9 results audit\n\n'
        f'Passed: {len(records)} evaluations; {len(targets)} matching test IDs/targets per run.\n\n'
        'Verified source/cache/receipt hashes, 50 train/validation epochs, train/test completion, '
        'finite predictions and losses, metric agreement, scheduler completion and empty scheduler stderr. '
        'No recorded skipped-OOM batches or traceback markers. No new model inference.\n\n'
        'Warnings in training/test logs:\n\n'+'\n'.join(f'- {n} occurrences: `{w}`' for w,n in warnings.items())+'\n')
    plots=root/'plots';plots.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(plots/f'{name}.png',dpi=180,bbox_inches='tight')
        fig.savefig(plots/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    a=np.array([metrics['baseline',s] for s in seeds]);b=np.array([metrics['v9_real',s] for s in seeds])
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for j,ax in enumerate(axes.flat):
        for i,s in enumerate(seeds):
            ax.plot([0,1],[a[i,j],b[i,j]],color='0.8',lw=1,zorder=0)
        for i,x in enumerate([a,b]):
            ax.scatter(np.full(len(seeds),i),x[:,j],color=colors[i],s=25)
            ax.errorbar(i+0.13,x[:,j].mean(),yerr=x[:,j].std(ddof=0),fmt='s',color='black',capsize=5)
        ax.set_xticks([0,1],labels);ax.set_title(titles[j]);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Held-out ATOM3D LBA-30 test: 10 matched seeds\nDots = seeds; black squares/bars = mean ± population SD (not confidence intervals)')
    fig.tight_layout(rect=[0,0,1,.92]);save(fig,'test_metrics')
    fig,axes=plt.subplots(2,2,figsize=(11,8));d=b-a
    for j,ax in enumerate(axes.flat):
        ax.bar(np.arange(len(seeds)),d[:,j],color=['#00897b' if v*(1 if j==0 else -1)<0 else '#c05a47' for v in d[:,j]])
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(np.arange(len(seeds)),seeds,rotation=45)
        ax.set_title(titles[j]);ax.set_xlabel('Matched seed');ax.set_ylabel('V9 − GVP');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Paired test differences: green favors V9; red favors baseline')
    fig.tight_layout(rect=[0,0,1,.95]);save(fig,'paired_differences')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,stage in zip(axes,['TRAIN','VAL']):
        for arm,label,color in zip(arms,labels,colors):
            x=np.stack([curves[arm,s,stage] for s in seeds]);epochs=np.arange(1,info['epochs']+1)
            for row in x:ax.plot(epochs,row,color=color,alpha=.13,lw=.7)
            ax.plot(epochs,x.mean(0),color=color,label=label,lw=2)
        ax.set_title(stage.title());ax.set_xlabel('Epoch (1-based)');ax.set_ylabel('Logged mean batch MSE')
        ax.set_yscale('log');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Thin lines = individual seeds; thick lines = seed mean; log y-axis\nValidation selects checkpoints; test metrics are not used in these curves')
    fig.tight_layout(rect=[0,0,1,.88]);save(fig,'learning_curves')
    with (root/'paired_differences.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed']+keys);w.writerows([s,*row] for s,row in zip(seeds,d))
    print('Wins for V9:',dict(zip(keys,[int((d[:,0]<0).sum())]+[int((d[:,i]>0).sum()) for i in range(1,4)])))
    print(f'Plots and audit saved in {root}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);main(p.parse_args().run_root)
