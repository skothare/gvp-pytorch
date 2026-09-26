"""Subprocess boundary preserves the historical training/test RNG lifecycle."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
import pipeline

def native(c,info,arm,seed,out,testing):
    import numpy as np
    import torch
    import torch_geometric
    from atom3d.datasets import LMDBDataset
    from functools import partial
    from gvp.atom3d import LBAModel,V3LBAModel,LBATransform,V3LBATransform
    import native_functions as n
    from cache_utils import input_fingerprint,tensor_hash
    import v9_lba as v9
    out=Path(out)
    if not testing:out.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('Production native training/testing requires a GPU allocation')
    torch.zeros(1).cuda()
    torch.manual_seed(seed);np.random.seed(seed)
    class CheckedTransform:
        def __init__(self,split):
            self.directory=None;self.entries=None
            if arm=='ep20':self.directory=v9.directory(Path(c['output_root'])/'caches/ep20','decoder',split)
            elif arm!='baseline':self.directory=Path(c['output_root'])/'controls'/arm/split
            if self.directory:
                self.entries=pipeline.read(self.directory/'_manifest.json')['entries']
                self.transform=V3LBATransform(str(self.directory))
            else:self.transform=LBATransform()
        def __call__(self,elem):
            graph=self.transform(elem)
            if self.entries is not None:
                rec=self.entries[elem['id']];x=graph.v3_emb[~graph.lig_flag]
                if (input_fingerprint(elem)!=rec['input_sha256'] or tensor_hash(x)!=rec['tensor_sha256'] or
                    x.shape!=(rec['n_atoms'],2 if arm=='raw_qr' else 256) or not torch.isfinite(x).all()):
                    raise ValueError('Feature input/alignment/checksum mismatch')
            if testing:graph.complex_id=elem['id']
            return graph
    datasets=[]
    for split in pipeline.SPLITS:
        ds=LMDBDataset(str(Path(c['lmdb_root'])/split),transform=CheckedTransform(split))
        if list(ds.ids())!=info['split_ids'][split]:raise ValueError('Dataset order changed')
        datasets.append(ds)
    # Same constructor ordering, shuffle behavior and RNG use as the native runner.
    loader=partial(torch_geometric.data.DataLoader,num_workers=4,batch_size=8,shuffle=True)
    trainset,valset,testset=map(loader,datasets)
    model=(LBAModel() if arm=='baseline' else V3LBAModel(v3_dim=2 if arm=='raw_qr' else 256)).to('cuda')
    n.args=SimpleNamespace(task='LBA',seed=seed,lr=1e-4,epochs=50,train_time=0,val_time=0,smp_idx=None,
        predictions_file=str(out/'predictions.csv') if testing else None,test=str(out/f'LBA_seed{seed}_best.pt'))
    n.models_dir=str(out);n.device='cuda'
    start=time.time()
    if testing:n.test(model,testset)
    else:n.train(model,trainset,valset)
    pipeline.write(out/('test_runtime.json' if testing else 'train_runtime.json'),dict(
        seed=seed,arm=arm,seconds=time.time()-start,node=platform.node(),gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,cuda=torch.version.cuda,deterministic=torch.are_deterministic_algorithms_enabled(),
        argv=sys.argv,gvp_source=__import__('gvp').__file__))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--kind',choices=['controlled','native_train','native_test'],required=True)
    p.add_argument('--arm',choices=['baseline','ep5','ep20','raw_qr','random_ep'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True)
    p.add_argument('--native',type=int,default=0);p.add_argument('--steps',type=int)
    a=p.parse_args();c=pipeline.config(a.config);info=pipeline.run_info(c)
    pipeline.check_audits(c)
    if a.kind=='controlled':
        import controlled_runner as runner
        runner.run(info,a.arm,a.seed,a.out,strict=not a.native,steps=a.steps,workers=4,device='cuda')
    else:
        pipeline.verify(Path(c['output_root']),Path(c['output_root'])/'controls/audit.json')
        native(c,info,a.arm,a.seed,a.out,a.kind=='native_test')
    pipeline.run_info(c)

if __name__=='__main__':main()
