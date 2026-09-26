"""Fresh-start orchestration and report contracts; never calls a real scheduler."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import pipeline as p
import reporting

@pytest.fixture
def cfg(tmp_path):
    path=tmp_path/'config.json'
    value={k:str(tmp_path/k) for k in p.PATHS};value['scheduler']={'pqr_shards':2}
    path.write_text(json.dumps(value));return p.config(path)

def test_every_paper_task_mapping(cfg):
    assert [p.task('panel_a',i) for i in range(5)]==p.SEEDS[:5]
    assert [p.task('panel_b',i) for i in range(40)]==[(a,s) for a in p.ARMS_B for s in p.SEEDS]
    assert [p.task('ridge',i) for i in range(2)]==['ep5','ep20']
    assert {p.task('controls',i)[0] for i in range(6)}=={'raw_qr','random_ep'}
    for i in [-1,40,None]:
        with pytest.raises(ValueError):p.task('panel_b',i)
    assert p.plan(cfg)['production_trainings']==55

def test_dry_run_needs_no_checkpoints_caches_or_history(cfg):
    result=subprocess.run([sys.executable,str(p.HERE/'pipeline.py'),'--config',cfg['configuration_file'],'--stage','all','--dry-run'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['ridge_models']==2
    assert not Path(cfg['output_root']).exists()

def test_submission_is_explicit(cfg):
    result=subprocess.run([sys.executable,str(p.HERE/'pipeline.py'),'--config',cfg['configuration_file'],'--stage','all'],capture_output=True,text=True)
    assert result.returncode!=0 and '--submit' in result.stderr
    assert not Path(cfg['output_root']).exists()

def test_scheduler_dependencies_resources_and_partial_journal(cfg,monkeypatch):
    root=Path(cfg['output_root']);root.mkdir();p.write(root/'run.json',{})
    monkeypatch.setattr(p,'run_info',lambda c:{})
    calls=[]
    def submit(command,**kwargs):
        assert command[0]=='sbatch';calls.append(command)
        return str(100+len(calls))+'\n'
    monkeypatch.setattr(p.subprocess,'check_output',submit)
    jobs=p.submit(cfg)
    assert len(jobs)==len(p.stages(cfg))
    assert '--array=0-39%2' in calls[11]
    assert '--constraint=L40' in calls[10] and '--constraint=L40' not in calls[11]
    for i,command in enumerate(calls[1:],1):assert '--dependency=afterok:'+str(100+i) in command
    with pytest.raises(ValueError,match='journal'):p.submit(cfg)

def test_partial_submission_records_successful_jobs(cfg,monkeypatch):
    root=Path(cfg['output_root']);root.mkdir();p.write(root/'run.json',{})
    monkeypatch.setattr(p,'run_info',lambda c:{})
    calls=[]
    def submit(command,**kwargs):
        calls.append(command)
        if len(calls)==3:raise RuntimeError('scheduler unavailable')
        return str(100+len(calls))
    monkeypatch.setattr(p.subprocess,'check_output',submit)
    with pytest.raises(RuntimeError):p.submit(cfg)
    assert len((root/'submissions.jsonl').read_text().splitlines())==2
    assert not (root/'jobs.json').exists()

@pytest.mark.parametrize('damage',['duplicate','ids','target','nan'])
def test_prediction_failures(tmp_path,damage):
    rows=[['a',1,1.1],['b',2,2.1],['c',3,2.8]]
    if damage=='duplicate':rows[-1][0]='a'
    if damage=='ids':rows[-1][0]='d'
    if damage=='target':rows[-1][1]=99
    if damage=='nan':rows[-1][2]=float('nan')
    f=tmp_path/'p.csv'
    with f.open('w',newline='') as h:w=csv.writer(h);w.writerow(['id','target','prediction']);w.writerows(rows)
    with pytest.raises(ValueError):p.validate_predictions(f,{'a':1,'b':2,'c':3})

def test_receipt_and_duplicate_protection(tmp_path):
    p.write(tmp_path/'run.json',{});(tmp_path/'a').write_text('data')
    p.receipt(tmp_path,tmp_path/'done.json',[tmp_path/'a']);p.verify(tmp_path,tmp_path/'done.json')
    with pytest.raises(FileExistsError):p.receipt(tmp_path,tmp_path/'done.json',[])
    (tmp_path/'a').write_text('changed')
    with pytest.raises(ValueError,match='changed'):p.verify(tmp_path,tmp_path/'done.json')

@pytest.mark.parametrize('failure',['train','oom','test','checkpoint'])
def test_native_failure_never_publishes_completion(cfg,monkeypatch,failure):
    root=Path(cfg['output_root']);root.mkdir();p.write(root/'run.json',{'targets':{'test':{'a':1,'b':2,'c':3}}})
    monkeypatch.setattr(p,'check_audits',lambda c:None);monkeypatch.setattr(p,'verify',lambda *a:{})
    def child(c,kind,out,**kwargs):
        out.mkdir(parents=True,exist_ok=True)
        if kind=='native_train' and failure in ['train','oom']:raise RuntimeError(failure)
        if failure!='checkpoint':(out/'LBA_seed42_best.pt').write_bytes(b'checkpoint')
        if kind=='native_test':raise RuntimeError('test failed')
        return out/'log'
    monkeypatch.setattr(p,'run_child',child)
    with pytest.raises((ValueError,RuntimeError)):p.train_b(cfg,0)
    assert not (root/'completed/b_baseline_42.json').exists()

def test_run_child_rejects_skipped_oom(cfg,monkeypatch,tmp_path):
    def run(command,stdout,**kw):stdout.write('Skipped batch due to OOM\n');stdout.flush()
    monkeypatch.setattr(p.subprocess,'run',run)
    with pytest.raises(ValueError,match='partial'):p.run_child(cfg,'native_train',tmp_path/'attempt',arm='baseline',seed=42)

@pytest.mark.reporting
def test_report_from_fresh_stubbed_completions(cfg,monkeypatch):
    root=Path(cfg['output_root']);root.mkdir();targets={'a':1.,'b':2.,'c':4.}
    info={'targets':{'test':targets}};p.write(root/'run.json',info)
    p.receipt(root,root/'diagnostics.json',[],passed=True)
    def one(panel,arm,seed):
        out=root/'attempts'/panel/arm/str(seed);out.mkdir(parents=True)
        with (out/'predictions.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['id','target','prediction']);w.writerows((k,v,v+.1) for k,v in targets.items())
        p.write(out/'result.json',dict(metrics=p.validate_predictions(out/'predictions.csv',targets),curves=[dict(epoch_1based=1,train_loss=.1,val_loss=.2)] if panel!='c' else []))
        return out
    for seed in p.SEEDS[:5]:
        dirs={a:one('a',a,seed) for a in p.ARMS_A}
        p.receipt(root,root/'completed'/f'a_{seed}.json',[f for d in dirs.values() for f in d.iterdir()],panel='a',seed=seed,results={a:str((d/'result.json').relative_to(root)) for a,d in dirs.items()})
    for arm in p.ARMS_B:
        for seed in p.SEEDS:
            d=one('b',arm,seed);p.receipt(root,root/'completed'/f'b_{arm}_{seed}.json',list(d.iterdir()),panel='b',seed=seed,results={arm:str((d/'result.json').relative_to(root))})
    for arm in ['ep5','ep20']:
        d=one('c',arm,None);p.receipt(root,root/'completed'/f'ridge_{arm}.json',list(d.iterdir()),panel='c',seed=None,results={arm:str((d/'result.json').relative_to(root))})
    monkeypatch.setattr(p,'run_info',lambda c:info)
    reporting.report(cfg)
    rows=list(csv.DictReader((root/'report/results.csv').open()))
    assert len(rows)==57 and {r['panel'] for r in rows}=={'a','b','c'}
    assert (root/'report/panels_abc.pdf').is_file()
    import numpy as np
    for fitted in (root/'attempts').glob('ridge_*/*/features.npz'):
        with np.load(fitted) as matrix:
            assert matrix['train_x'].dtype==np.float64 and matrix['test_x'].dtype==np.float64
    assert p.read(root/'report/AUDIT.json')['passed']


def test_native_computational_functions_unchanged():
    import ast,hashlib
    expected=p.read(p.HERE/'NATIVE_FUNCTION_HASHES.json')
    tree=ast.parse((p.HERE/'native_functions.py').read_text())
    actual={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in expected}
    assert actual==expected


def test_controlled_diagnostic_failure_blocks_production(cfg,monkeypatch):
    root=Path(cfg['output_root']);root.mkdir();p.write(root/'run.json',{})
    monkeypatch.setattr(p,'check_audits',lambda c:None)
    def child(c,kind,out,**kw):
        out.mkdir(parents=True);p.write(out/'result.json',{'initial_sha256':out.name})
        log=out.with_suffix('.log');log.write_text('stub');return log
    monkeypatch.setattr(p,'run_child',child)
    with pytest.raises(ValueError,match='repetition'):p.diagnostics(cfg)
    assert not (root/'diagnostics.json').exists()
    with pytest.raises(FileNotFoundError):p.train_a(cfg,0)


def test_random_model_is_frozen_and_rng_isolated(monkeypatch):
    import torch,types
    import v9_lba,model_loader
    monkeypatch.setattr(v9_lba,'read_checkpoint',lambda _:(None,{'encoder_final_norm':True}))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Linear(3,3);self.decoder=torch.nn.Linear(3,3)
    monkeypatch.setattr(model_loader,'load_model_module',lambda *a:types.SimpleNamespace(create_model_with_distance_features=Tiny))
    before=torch.random.get_rng_state().clone();a=p.initialize_random('unused');b=p.initialize_random('unused')
    assert torch.equal(before,torch.random.get_rng_state())
    assert not a.training and not any(x.requires_grad for x in a.parameters())
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())


@pytest.mark.reporting
def test_empty_output_to_panels_with_tiny_inputs(cfg,monkeypatch):
    """Real converters/writers/auditors/ridge/report; only external tools/models/training are stubbed."""
    import types,hashlib
    import pandas as pd
    import torch
    import v9_lba as v9
    import atom3d.datasets
    import model_loader
    cfg['estats_root']=str(Path(__file__).resolve().parents[3]/'Estats')
    cfg['gvp_root']=str(Path(__file__).resolve().parents[2])
    class Env:
        def close(self):pass
    class DS:
        _env=Env()
        def __init__(self,path,*a,**kw):self.split=Path(path).name
        def ids(self):return ['toy0','toy1','toy2']
        def __len__(self):return 3
        def __getitem__(self,i):
            df=pd.DataFrame(dict(element=['N','C'],resname=['ALA','ALA'],name=['N','CA'],chain=['A','A'],residue=[1,1],x=[float(i),float(i+1)],y=[0.,0.],z=[0.,0.]))
            return dict(id=self.ids()[i],atoms_pocket=df,atoms_protein=df.copy(),scores={'neglog_aff':float(i+1)})
    class Encoder(torch.nn.Module):
        def forward(self,b):return torch.nested.nested_tensor([b['coords'].unbind()[0][:,:1].repeat(1,256)],layout=torch.jagged)
    class Decoder(torch.nn.Module):
        def __init__(self):super().__init__();self.head=torch.nn.Linear(256,1)
        def forward(self,e,b,q):return self.head(q[:,:,:1].repeat(1,1,256)+e.unbind()[0].mean(0))
    class Model(torch.nn.Module):
        def __init__(self,**kw):super().__init__();self.encoder=Encoder();self.decoder=Decoder()
    for a in ['ep5','ep20']:Path(cfg[a+'_checkpoint']).write_text(a)
    hashes={a:p.sha(cfg[a+'_checkpoint']) for a in ['ep5','ep20']}
    monkeypatch.setattr(p,'CHECKPOINTS',hashes);monkeypatch.setattr(p,'COUNTS',dict.fromkeys(p.SPLITS,3))
    data={}
    for split in p.SPLITS:
        d=Path(cfg['lmdb_root'])/split;d.mkdir(parents=True);(d/'data.mdb').write_text(split);data[split]=p.sha(d/'data.mdb')
    monkeypatch.setattr(p,'reference_cohort',lambda:dict(split_ids={s:DS(s).ids() for s in p.SPLITS},dataset_sha256=data))
    monkeypatch.setattr(atom3d.datasets,'LMDBDataset',DS)
    monkeypatch.setattr(p,'preflight',lambda c:dict(passed=True))
    monkeypatch.setattr(v9,'read_checkpoint',lambda cp:({'epoch':4 if Path(cp).read_text()=='ep5' else 19},{'encoder_final_norm':True}))
    monkeypatch.setattr(v9,'load_v9_model',lambda *a:Model())
    monkeypatch.setattr(model_loader,'load_model_module',lambda *a:types.SimpleNamespace(create_model_with_distance_features=Model))
    original=v9.df_to_v8_batch
    monkeypatch.setattr(v9,'df_to_v8_batch',lambda a,b,device,*args:original(a,b,'cpu',*args))
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:'[]')
    root=Path(cfg['output_root']);assert not root.exists()
    p.initialize(cfg)
    def external(cmd,**kw):
        assert 'build_pqr_cache.py' in cmd[1]
        if '--aggregate-only' in cmd:return
        split=cmd[cmd.index('--split')+1];shard=int(cmd[cmd.index('--shard')+1]);n=int(cmd[cmd.index('--n-shards')+1])
        d=root/'pqr'/split;d.mkdir(parents=True,exist_ok=True)
        for i in range(shard,3,n):
            (d/f'toy{i}.pqr').write_text(f'ATOM 1 N ALA A 1 {i} 0 0 -0.3 1.5\nATOM 2 CA ALA A 1 {i+1} 0 0 0.2 1.7\n')
    monkeypatch.setattr(p.subprocess,'run',external)
    def child(c,kind,out,**options):
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        log=out.with_suffix('.'+kind+'.log');log.write_text('stub execution\n')
        seed=options['seed']
        if kind=='native_train':
            (out/f'LBA_seed{seed}_best.pt').write_bytes(b'stub checkpoint')
            log.write_text(''.join(f'EPOCH {e} TRAIN loss: 0.1\nEPOCH {e} VAL   loss: 0.2\n' for e in range(50)))
            return log
        with (out/'predictions.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['id','target','prediction']);w.writerows((f'toy{i}',i+1,i+1.1) for i in range(3))
        if kind=='controlled':
            p.write(out/'result.json',dict(initial_sha256='same',shared_initial_sha256='same',
                metrics=p.validate_predictions(out/'predictions.csv',{f'toy{i}':i+1. for i in range(3)}),
                curves=[dict(epoch_1based=e+1,train_loss=.1,val_loss=.2,train_order_sha256='same',val_order_sha256='same') for e in range(50)]))
        return log
    monkeypatch.setattr(p,'run_child',child)
    for stage,resource,count in p.stages(cfg):
        for index in (range(count) if count else [None]):p.execute_stage(cfg,stage,index)
    assert p.read(root/'report/AUDIT.json')['evaluations']==57
    assert len(list(csv.DictReader((root/'report/results.csv').open())))==57
    assert (root/'report/panels_abc.pdf').is_file()
    import numpy as np
    for fitted in (root/'attempts').glob('ridge_*/*/features.npz'):
        with np.load(fitted) as matrix:
            assert matrix['train_x'].dtype==np.float64 and matrix['test_x'].dtype==np.float64
