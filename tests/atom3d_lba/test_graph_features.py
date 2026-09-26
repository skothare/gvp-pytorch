import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch
from gvp.atom3d import LBAModel,V3LBAModel,V3LBATransform

@pytest.mark.parametrize('width',[2,256])
def test_graph_row_binding_ligand_zeros_batch_gradient_reload(tmp_path,width):
    pocket=pd.DataFrame(dict(element=['N','C','O'],x=[0.,1.,2.],y=[0.,0.,0.],z=[0.,0.,0.]))
    elem=dict(id='toy',atoms_pocket=pocket,atoms_ligand=pocket.iloc[:1].copy(),scores={'neglog_aff':5.})
    features=torch.arange(3*width).reshape(3,width).float()/100
    torch.save(features,tmp_path/'toy_pocket.pt')
    graphs=[V3LBATransform(str(tmp_path))(elem) for _ in range(2)]
    batch=Batch.from_data_list(graphs)
    assert torch.equal(batch.v3_emb[~batch.lig_flag],features.repeat(2,1))
    assert torch.count_nonzero(batch.v3_emb[batch.lig_flag])==0
    assert LBAModel().W_v[1].ws.in_features==9
    model=V3LBAModel(v3_dim=width)
    assert model.W_v[1].ws.in_features==9+width
    out=model(batch);assert out.shape==(2,) and torch.isfinite(out).all()
    out.square().mean().backward();assert torch.isfinite(model.W_v[1].ws.weight.grad).all()
    model.eval()
    with torch.no_grad():expected=model(batch)
    torch.save(model.state_dict(),tmp_path/'model.pt')
    restored=V3LBAModel(v3_dim=width).eval()
    restored.load_state_dict(torch.load(tmp_path/'model.pt',weights_only=True),strict=True)
    with torch.no_grad():torch.testing.assert_close(restored(batch),expected,rtol=0,atol=0)
