"""CPU-only comparison of audited V8/V9 decoder experiments, preserving seed counts."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_lba_v8 import parse_log,sha
from lba_v9_workflow import summarize
ROOT=Path(__file__).resolve().parent
V8=ROOT/'logs/v8_decoder_production_20260917T044644Z'
V9=ROOT/'logs/v9_decoder_production_20260918T161209Z'
KEYS=['rmse','pearson_r','spearman_r','r2']
TITLES=['RMSE (lower better)','Pearson r (higher better)','Spearman ρ (higher better)','R² (higher better)']


def main():
    summarize(V9)
    out=V9/'v8_comparison';out.mkdir(exist_ok=True)
    old=json.loads((V8/'run.json').read_text());new=json.loads((V9/'run.json').read_text())
    rows=[]
    for arm in old['arms']:
        for seed in old['seeds']:
            receipt=json.loads((V8/arm/f'seed{seed}.complete.json').read_text());log=V8/arm/f'seed{seed}.log'
            checkpoint=Path(old['model_root'])/arm/f'LBA_seed{seed}_best.pt'
            assert sha(log)==receipt['log_sha256'] and sha(checkpoint)==receipt['checkpoint_sha256']
            assert sha(V8/'run.json')==receipt['run_sha256']
            rows.append(dict(round='V8 decoder',arm=arm,seed=seed,**parse_log(log,old['epochs'])))
    for row in csv.DictReader((V9/'results.csv').open()):
        rows.append(dict(round='V9 decoder',arm=row['arm'],seed=int(row['seed']),**{k:float(row[k]) for k in KEYS}))
    with (out/'per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['round','arm','seed']+KEYS);w.writeheader();w.writerows(rows)
    groups=[('V9 decoder','baseline','GVP, V9 round'),('V9 decoder','v9_real','GVP + V9 real'),
            ('V8 decoder','v8_real','GVP + V8 real'),('V8 decoder','v8_zero','GVP + V8 zero/unit'),
            ('V8 decoder','baseline','GVP, V8 round')]
    colors=['#526677','#00897b','#527bbd','#db9b3f','#9da6ad']
    def arr(group,seeds=None):
        return np.array([[r[k] for k in KEYS] for r in rows if (r['round'],r['arm'])==group[:2] and (seeds is None or r['seed'] in seeds)])
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    def aggregate(seeds,name,title):
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for j,ax in enumerate(axes.flat):
            for i,g in enumerate(groups):
                x=arr(g,seeds)[:,j];ax.bar(i,x.mean(),yerr=x.std(ddof=0),capsize=4,color=colors[i],alpha=.8)
                ax.scatter(i+np.linspace(-.16,.16,len(x)),x,color='black',s=15,zorder=3)
            ax.set_xticks(range(5),[g[2]+f'\n(n={len(arr(g,seeds))})' for g in groups],rotation=18,ha='right')
            ax.set_title(TITLES[j]);ax.axhline(0,color='black',lw=.5);ax.grid(axis='y',alpha=.2)
        fig.suptitle(title+'\nBars = seed means; error bars = population SD, not CI; dots = individual seeds')
        fig.tight_layout(rect=[0,0,1,.93]);save(fig,name)
    aggregate(None,'all_seeds_bars','Decoder experiments: all available seeds (V9: 10; V8: 3, not 5)')
    common=set(old['seeds'])&set(new['seeds'])
    aggregate(common,'shared_seeds_bars','Decoder experiments: identical seed set 42, 123, 7')
    fig,axes=plt.subplots(2,2,figsize=(14,8));seeds=new['seeds'];x=np.arange(len(seeds));width=.2
    for j,ax in enumerate(axes.flat):
        for i,g in enumerate(groups[:4]):
            lookup={r['seed']:r[KEYS[j]] for r in rows if (r['round'],r['arm'])==g[:2]}
            ax.bar(x+(i-1.5)*width,[lookup.get(s,np.nan) for s in seeds],width,label=g[2],color=colors[i])
        ax.set_xticks(x,seeds,rotation=40);ax.set_xlabel('Seed');ax.set_title(TITLES[j]);ax.axhline(0,color='black',lw=.5)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,.96))
    fig.suptitle('Individual test results: V8 bars absent where no run exists')
    fig.tight_layout(rect=[0,0,1,.9]);save(fig,'per_seed_bars')
    lines=['# V8 versus V9 decoder comparison','','V8 has 3 verified seeds, not 5. V8 metrics have logged four-decimal precision; V9 uses prediction-derived metrics.','']
    for title,subset in [('All available seeds',None),('Shared seeds 42, 123, 7',common)]:
        lines += ['## '+title,'','| Group | n | RMSE | Pearson | Spearman | R² |','|---|---:|---:|---:|---:|---:|']
        for g in groups:
            a=arr(g,subset);lines.append('| '+g[2]+f' | {len(a)} | '+' | '.join(f'{m:.4f} ± {s:.4f}' for m,s in zip(a.mean(0),a.std(0)))+' |')
        lines.append('')
    env=lambda d:{r['name']:r['version'] for r in json.loads(d['environment'])}
    audit=dict(v8_seeds=old['seeds'],v9_seeds=new['seeds'],same_packages=env(old)==env(new),
      model_hashes_match={k:old['source_sha256'][k]==new['source_sha256'][k] for k in ['gvp/atom3d.py','gvp/__init__.py']},
      matched_settings={k:old[k]==new[k] for k in ['epochs','batch','lr','train_time','val_time']},
      workers_match=old['num_workers']==new['workers'],v8_receipt_hashes_checked=True,
      scope='Descriptive across runs; shared seed numbers do not ensure bitwise identical GPU training. No significance test.')
    (out/'comparison_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines+=['Baseline model-source hashes, package versions and recorded training settings match across rounds. Same seed does not guarantee GPU bitwise reproducibility. This comparison does not isolate a particular pretraining change.']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
