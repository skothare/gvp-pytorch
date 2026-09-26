"""CPU verification for the isolated controlled experiment, no Slurm/GPU use."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
import unified_lba_runner as runner
import unified_lba_workflow as workflow


def graph(index=0,fusion=False):
    gen=torch.Generator().manual_seed(80+index)
    edge=torch.tensor([[0,1,1,2,2,3,3,0],[1,0,2,1,3,2,0,3]])
    d=Data(x=torch.randn(4,3,generator=gen),atoms=torch.tensor([1,2,1,3]),
           edge_index=edge,edge_s=torch.randn(8,16,generator=gen),
           edge_v=torch.randn(8,1,3,generator=gen),label=torch.tensor([float(index+1)]),
           lig_flag=torch.tensor([False,False,False,True]))
    d.complex_id='id'+str(index)
    if fusion:
        d.v3_emb=torch.randn(4,256,generator=gen);d.v3_emb[-1]=0
    return d


@pytest.mark.parametrize('arm',['baseline','ep5','ep20'])
def test_pooling_preserves_predictions_gradients_and_checkpoint(arm,tmp_path):
    runner.seed_all(42);native=runner.make_model(arm,False)
    controlled=runner.make_model(arm,True);controlled.load_state_dict(native.state_dict(),strict=True)
    batch=Batch.from_data_list([graph(0,arm!='baseline'),graph(1,arm!='baseline')])
    native.eval();controlled.eval()
    a,b=native(batch),controlled(batch)
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    a.square().sum().backward();b.square().sum().backward()
    for x,y in zip(native.parameters(),controlled.parameters()):
        if x.grad is not None:
            torch.testing.assert_close(x.grad,y.grad,rtol=1e-4,atol=1e-6)
    torch.save(controlled.state_dict(),tmp_path/'model.pt')
    native.load_state_dict(torch.load(tmp_path/'model.pt',weights_only=True),strict=True)


def test_shared_initialization_and_fusion_width():
    runner.seed_all(123);a=runner.make_model('baseline')
    runner.seed_all(123);b=runner.make_model('ep5')
    runner.seed_all(123);c=runner.make_model('ep20')
    assert runner.state_digest(a.state_dict(),True)==runner.state_digest(b.state_dict(),True)
    assert runner.state_digest(b.state_dict())==runner.state_digest(c.state_dict())
    assert a.W_v[0].scalar_norm.normalized_shape==(9,)
    assert b.W_v[0].scalar_norm.normalized_shape==(265,)


def fake_loaders(info,arm,seed,workers):
    return {split:DataLoader([graph(i,arm!='baseline') for i in range(4)],batch_size=2,
                            shuffle=split!='test',generator=torch.Generator().manual_seed(seed+off))
            for off,split in enumerate(['train','val','test'])}


def test_repeatable_cpu_training_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(runner,'loaders',fake_loaders)
    info=dict(lr=1e-4,epochs=2,split_ids={'test':['id'+str(i) for i in range(4)]})
    a=runner.run(info,'baseline',42,tmp_path/'a',workers=0,device='cpu')
    b=runner.run(info,'baseline',42,tmp_path/'b',workers=0,device='cpu')
    assert all(workflow.comparison(a,b).values())
    assert (tmp_path/'a/predictions.csv').read_bytes()==(tmp_path/'b/predictions.csv').read_bytes()
    assert len(a['curves'])==2 and a['best_epoch_1based'] in [1,2]
    with pytest.raises(FileExistsError):runner.run(info,'baseline',42,tmp_path/'a',device='cpu')


def test_arms_use_same_batch_schedule(tmp_path,monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(runner,'loaders',fake_loaders)
    info=dict(lr=1e-4,epochs=1,split_ids={'test':['id'+str(i) for i in range(4)]})
    runs=[runner.run(info,arm,7,tmp_path/arm,device='cpu') for arm in workflow.ARMS]
    for r in runs[1:]:
        assert r['shared_initial_sha256']==runs[0]['shared_initial_sha256']
        for k in ['train_order_sha256','val_order_sha256']:
            assert r['curves'][0][k]==runs[0]['curves'][0][k]


def test_receipt_detects_corruption(tmp_path):
    workflow.write_new(tmp_path/'run.json',{'test':1})
    (tmp_path/'model.pt').write_bytes(b'original')
    workflow.write_new(tmp_path/'done.json',workflow.receipt(tmp_path,[tmp_path/'model.pt']))
    workflow.verify_receipt(tmp_path,tmp_path/'done.json')
    (tmp_path/'model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        workflow.verify_receipt(tmp_path,tmp_path/'done.json')


def test_diagnostic_failure_blocks_publication(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'config',lambda _:dict(diagnostic_seeds=[42]))
    monkeypatch.setenv('SLURM_JOB_ID','stub')
    def fake(root,arm,seed,out,native=False,steps=None):
        out.mkdir();out.with_suffix('.log').write_text('stub')
        return dict(initial_sha256=out.name if not native else 'same')
    monkeypatch.setattr(workflow,'execute',fake)
    with pytest.raises(ValueError,match='production is blocked'):
        workflow.diagnostics(tmp_path)
    assert not (tmp_path/'DIAGNOSTICS_PASSED.json').exists()
    assert (tmp_path/'diagnostics/stub/FAILED.json').exists()


def test_submission_dependencies_and_five_seeds(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'config',lambda _:dict(seeds=workflow.SEEDS))
    commands=[]
    def fake(command,**kwargs):
        assert command[0]=='sbatch';commands.append(command)
        return str(900+len(commands))+'\n'
    monkeypatch.setattr(workflow.subprocess,'check_output',fake)
    workflow.submit(tmp_path)
    assert '--array=0-4%2' in commands[1]
    assert '--dependency=afterok:901' in commands[1]
    assert '--dependency=afterok:902' in commands[2]
    assert json.loads((tmp_path/'jobs.json').read_text())['production']=='902'
    with pytest.raises(ValueError,match='Already submitted'):workflow.submit(tmp_path)


def test_training_requires_diagnostic_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'config',lambda _:dict(seeds=[42]))
    with pytest.raises(FileNotFoundError):workflow.train(tmp_path,0)
    assert not (tmp_path/'attempts').exists()
