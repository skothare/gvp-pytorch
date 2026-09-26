"""Recompute metrics from identified predictions; keep paper cohorts separate."""
import csv
import itertools
from pathlib import Path
import numpy as np
import pipeline as p

LABELS={'baseline':'GVP','ep5':'GVP + EP-5','ep20':'GVP + EP-20','raw_qr':'GVP + charges/radii','random_ep':'GVP + random EP'}
COLORS={'baseline':'#435D7D','ep5':'#C3313D','ep20':'#2879B5','raw_qr':'#6D96B5','random_ep':'#173C66'}

def aggregate(rows):
    result=[]
    for panel in ['a','b','c']:
        for arm in dict.fromkeys(r['arm'] for r in rows if r['panel']==panel):
            subset=[r for r in rows if r['panel']==panel and r['arm']==arm]
            x=np.array([[r[k] for k in p.METRICS] for r in subset])
            row=dict(panel=panel,arm=arm,n=len(subset))
            for j,k in enumerate(p.METRICS):row[k]=float(x[:,j].mean());row[k+'_sd']=None if panel=='c' else float(x[:,j].std(ddof=0))
            result.append(row)
    return result

def paired(rows):
    out=[]
    for panel in ['a','b']:
        arms=list(dict.fromkeys(r['arm'] for r in rows if r['panel']==panel))
        for arm,ref in itertools.combinations(arms,2):
            a={r['seed']:r for r in rows if r['panel']==panel and r['arm']==arm}
            b={r['seed']:r for r in rows if r['panel']==panel and r['arm']==ref}
            if a.keys()!=b.keys():raise ValueError('Unpaired seed sets')
            for seed in a:out.append(dict(panel=panel,comparison=arm+' minus '+ref,seed=seed,**{k:a[seed][k]-b[seed][k] for k in p.METRICS}))
    return out

def csv_write(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def collect(root,info):
    root=Path(root);rows=[];curves=[];records=[]
    p.verify(root,root/'diagnostics.json')
    expected=[('a',s,root/'completed'/f'a_{s}.json') for s in p.SEEDS[:5]]
    expected += [('b',s,root/'completed'/f'b_{a}_{s}.json') for a,s in itertools.product(p.ARMS_B,p.SEEDS)]
    expected += [('c',None,root/'completed'/f'ridge_{a}.json') for a in ['ep5','ep20']]
    for panel,seed,path in expected:
        rec=p.verify(root,path);records.append(path)
        if rec['panel']!=panel or rec['seed']!=seed:raise ValueError('Wrong panel/seed receipt')
        for arm,relative in rec['results'].items():
            resultpath=root/relative;r=p.read(resultpath)
            m=p.validate_predictions(resultpath.with_name('predictions.csv'),info['targets']['test'])
            if any(abs(m[k]-r['metrics'][k])>1e-10 for k in p.METRICS):raise ValueError('Reported metrics differ from predictions')
            rows.append(dict(panel=panel,arm=arm,seed=seed,**m,result=relative))
            for curve in r.get('curves',[]):
                curves.append(dict(panel=panel,arm=arm,seed=seed,epoch=curve['epoch_1based'],train_loss=curve['train_loss'],val_loss=curve['val_loss']))
    expected_keys={('a',a,s) for a,s in itertools.product(p.ARMS_A,p.SEEDS[:5])}|{('b',a,s) for a,s in itertools.product(p.ARMS_B,p.SEEDS)}|{('c',a,None) for a in ['ep5','ep20']}
    if {(r['panel'],r['arm'],r['seed']) for r in rows}!=expected_keys or len(rows)!=57:raise ValueError('Incomplete/duplicate experiment set')
    return rows,curves,records

def figures(out,rows,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    titles=['RMSE ↓','Pearson r ↑','Spearman ρ ↑','R² ↑']
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,4,figsize=(13,10),squeeze=False)
    for i,panel in enumerate(['a','b','c']):
        arms=p.ARMS_A if panel=='a' else p.ARMS_B if panel=='b' else ['baseline','ep5','ep20']
        for j,key in enumerate(p.METRICS):
            ax=axes[i,j]
            for pos,arm in enumerate(arms):
                src='a' if panel=='c' and arm=='baseline' else panel
                values=np.array([r[key] for r in rows if r['panel']==src and r['arm']==arm])
                color=COLORS[arm] if panel!='b' or arm!='ep20' else '#C3313D'
                if panel=='c' and arm!='baseline':
                    ax.scatter(values,[pos],marker='D',facecolor='white',edgecolor=color,s=60,zorder=3)
                else:
                    ax.scatter(values,pos+np.linspace(-.10,.10,len(values)),color=color,alpha=.4,s=17)
                    ax.errorbar(values.mean(),pos,xerr=values.std(ddof=0),fmt='o',color=color,capsize=4)
                ax.text(.98,1-(pos+.4)/len(arms),f'{values.mean():.4f}',transform=ax.transAxes,ha='right',color=color,fontsize=9)
            names=[LABELS[a] if panel!='c' or a=='baseline' else a.upper().replace('EP','EP-')+' + ridge' for a in arms]
            ax.set_yticks(range(len(arms)),names if j==0 else []);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
            if i==0:ax.set_title(titles[j])
            if j==0:ax.set_ylabel(panel+'.',rotation=0,labelpad=18,fontweight='bold')
    fig.supxlabel('Dots: training seeds; circles: mean ± population SD; diamonds: deterministic ridge fits',fontsize=10)
    fig.tight_layout();fig.savefig(out/'panels_abc.pdf');fig.savefig(out/'panels_abc.png',dpi=220);plt.close(fig)
    if not curves:return
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,panel in zip(axes,['a','b']):
        for arm in dict.fromkeys(r['arm'] for r in curves if r['panel']==panel):
            for split,ls in [('train_loss','--'),('val_loss','-')]:
                x=sorted({r['epoch'] for r in curves if r['panel']==panel and r['arm']==arm})
                y=[np.mean([r[split] for r in curves if r['panel']==panel and r['arm']==arm and r['epoch']==e]) for e in x]
                ax.plot(x,y,ls,color=COLORS[arm],label=LABELS[arm]+' '+split.replace('_loss',''))
        ax.set(title='Panel '+panel,xlabel='Epoch (1-based)',ylabel='Mean batched MSE');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'learning_curves.pdf');plt.close(fig)

def render(out,rows,curves):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows=sorted(rows,key=lambda r:(r['panel'],(p.ARMS_B if r['panel']=='b' else p.ARMS_A).index(r['arm']),p.SEEDS.index(r['seed']) if r['seed'] is not None else -1))
    agg=aggregate(rows);pairs=paired(rows)
    csv_write(out/'results.csv',rows);csv_write(out/'aggregate.csv',agg);csv_write(out/'paired_differences.csv',pairs)
    grouped=[]
    for panel,comparison in dict.fromkeys((r['panel'],r['comparison']) for r in pairs):
        values=[r for r in pairs if r['panel']==panel and r['comparison']==comparison]
        entry=dict(panel=panel,comparison=comparison,n=len(values))
        for k in p.METRICS:
            x=np.array([r[k] for r in values]);entry[k]=float(x.mean());entry[k+'_sd']=float(x.std(ddof=0))
        grouped.append(entry)
    csv_write(out/'paired_summary.csv',grouped)
    if curves:csv_write(out/'learning_curves.csv',curves)
    lines=['# Paper LBA reproduction','', 'Population SD across training seeds (ddof=0); ridge has no seed SD. No significance claim.','',
        '| Panel | Arm | n | RMSE | Pearson | Spearman | R² |','|---|---|---:|---:|---:|---:|---:|']
    tex=['% Generated from validated predictions. Population SD; deterministic ridge has no SD.',r'\begin{tabular}{llrrrrr}',r'Panel & Model & $n$ & RMSE & Pearson & Spearman & $R^2$ \\',r'\hline']
    for r in agg:
        vals=[f'{r[k]:.4f}'+(f" ± {r[k+'_sd']:.4f}" if r[k+'_sd'] is not None else '') for k in p.METRICS]
        label=LABELS[r['arm']] if r['panel']!='c' else r['arm'].upper().replace('EP','EP-')+' + ridge'
        lines.append('| '+' | '.join([r['panel'],label,str(r['n'])]+vals)+' |')
        tex.append(' & '.join([r['panel'],label,str(r['n'])]+[v.replace('±',r'$\pm$') for v in vals])+r' \\')
    tex.append(r'\end{tabular}')
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n');(out/'tables.tex').write_text('\n'.join(tex)+'\n')
    figures(out,rows,curves)

def report(c):
    root=Path(c['output_root']);info=p.run_info(c);rows,curves,records=collect(root,info)
    out=root/'report'
    if out.exists():raise FileExistsError('Report already exists; retain it and use a new run')
    temporary=p.attempt(root,'report');render(temporary,rows,curves)
    files={str(q.relative_to(root)):p.sha(q) for q in records}
    files.update({str(Path('report')/q.name):p.sha(q) for q in temporary.iterdir()})
    p.write(temporary/'AUDIT.json',dict(run_sha256=p.sha(root/'run.json'),files=files,
        passed=True,evaluations=57,test_count=len(info['targets']['test'])))
    temporary.rename(out)  # publish complete report and audit together



def reference_report(out):
    """Independent archived-evidence path; not an input to fresh production."""
    reference=Path(__file__).parent/'reference';manifest=p.read(reference/'manifest.json')
    rows=[];targets=None
    for r in manifest['records']:
        path=reference/r['prediction']
        if p.sha(path)!=r['sha256']:raise ValueError('Reference predictions changed')
        entries=list(csv.DictReader(path.open()))
        if targets is None:targets={x['id']:float(x['target']) for x in entries}
        metrics=p.validate_predictions(path,targets)
        if any(abs(metrics[k]-r['expected'][k])>1e-10 for k in p.METRICS):raise ValueError('Reference metric mismatch')
        rows.append(dict(panel=r['panel'],arm=r['arm'],seed=r['seed'],**metrics,result=r['prediction']))
    if len(rows)!=57:raise ValueError('Missing paper records')
    render(out,rows,[])
    p.write(Path(out)/'AUDIT.json',dict(passed=True,evaluations=57,test_count=len(targets),source_manifest_sha256=p.sha(reference/'manifest.json'),note='Archived historical predictions; no models were retrained.'))
