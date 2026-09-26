import numpy as np
import pytest
import torch
from torch_geometric.data import Batch,Data
from torch_geometric.loader import DataLoader
import controlled_runner as runner
import pipeline as workflow
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
    assert all(workflow.same_run(a,b).values())
    assert (tmp_path/'a/predictions.csv').read_bytes()==(tmp_path/'b/predictions.csv').read_bytes()
    assert len(a['curves'])==2 and a['best_epoch_1based'] in [1,2]
    with pytest.raises(FileExistsError):runner.run(info,'baseline',42,tmp_path/'a',device='cpu')

def test_arms_use_same_batch_schedule(tmp_path,monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(runner,'loaders',fake_loaders)
    info=dict(lr=1e-4,epochs=1,split_ids={'test':['id'+str(i) for i in range(4)]})
    runs=[runner.run(info,arm,7,tmp_path/arm,device='cpu') for arm in workflow.ARMS_A]
    for r in runs[1:]:
        assert r['shared_initial_sha256']==runs[0]['shared_initial_sha256']
        for k in ['train_order_sha256','val_order_sha256']:
            assert r['curves'][0][k]==runs[0]['curves'][0][k]
