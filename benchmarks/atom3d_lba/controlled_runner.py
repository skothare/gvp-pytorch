"""Isolated LBA runner: same GVP mathematics, explicit RNGs and CSR pooling.

Executed from a run-private snapshot, never imported by historical runners.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from atom3d.datasets import LMDBDataset
from torch_geometric.loader import DataLoader
from torch_scatter import segment_csr
from gvp.atom3d import LBAModel, V3LBAModel, LBATransform, V3LBATransform


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(state, exclude_input=False):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if exclude_input and name.startswith('W_v.'):
            continue
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def configure(strict):
    if strict and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts')
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = strict
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class ControlledBaseline(LBAModel):
    def forward(self, batch):
        nodes = super().forward(batch, scatter_mean=False, dense=False)
        return self.dense(segment_csr(nodes, batch.ptr, reduce='mean')).squeeze(-1)


class ControlledFusion(V3LBAModel):
    def __init__(self):
        super().__init__(v3_dim=256)

    def forward(self, batch):
        nodes = super().forward(batch, scatter_mean=False, dense=False)
        return self.dense(segment_csr(nodes, batch.ptr, reduce='mean')).squeeze(-1)


def make_model(arm, strict=True):
    if strict:
        return ControlledBaseline() if arm == 'baseline' else ControlledFusion()
    return LBAModel() if arm == 'baseline' else V3LBAModel(v3_dim=256)


class CheckedDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb, ids, cache=None, entries=None):
        self.dataset = LMDBDataset(str(lmdb))
        self.ids = list(ids)
        if list(self.dataset.ids()) != self.ids:
            raise ValueError('Dataset membership/order changed')
        self.cache = Path(cache) if cache else None
        self.entries = entries
        self.transform = V3LBATransform(str(cache)) if cache else LBATransform()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item = self.dataset[index]
        pid = self.ids[index]
        if item['id'] != pid:
            raise ValueError('LMDB record ID mismatch')
        graph = self.transform(item)
        if self.cache:
            rec = self.entries[pid]
            features = graph.v3_emb[~graph.lig_flag]
            if (tuple(features.shape) != (rec['n_atoms'],256)
                    or features.dtype != torch.float32 or not torch.isfinite(features).all()
                    or digest_tensor(features) != rec['tensor_sha256']):
                raise ValueError('Cache tensor corrupt/changed: '+pid)
        graph.complex_id = pid
        return graph


def loaders(info, arm, seed, workers):
    result = {}
    for offset, split in enumerate(['train','val','test']):
        cache = None
        entries = None
        if arm != 'baseline':
            cache = Path(info['caches'][arm]['root'])/'decoder/charge_real'/split
            entries = json.loads((cache/'_manifest.json').read_text())['entries']
        dataset = CheckedDataset(Path(info['lmdb_root'])/split, info['split_ids'][split],cache,entries)
        generator = torch.Generator().manual_seed(seed+10000*(offset+1))
        result[split] = DataLoader(dataset,batch_size=info['batch'],shuffle=split!='test',
                                  num_workers=workers,generator=generator,worker_init_fn=seed_worker)
    return result


def regression(y, p):
    from scipy.stats import pearsonr,spearmanr
    y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite predictions/targets')
    return dict(rmse=float(np.sqrt(np.mean((y-p)**2))),pearson_r=float(pearsonr(y,p).statistic),
                spearman_r=float(spearmanr(y,p).statistic),
                r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)))


def run(info, arm, seed, out, strict=True, steps=None, workers=4, device='cuda'):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    configure(strict)
    if device == 'cuda':
        if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA L40':
            raise RuntimeError('This experiment requires an NVIDIA L40 GPU')
    seed_all(seed)
    model=make_model(arm,strict)
    initial=state_digest(model.state_dict())
    shared=state_digest(model.state_dict(),True)
    model=model.to(device)
    data=loaders(info,arm,seed,workers)
    # Architecture construction cannot shift dropout RNG or loader order.
    seed_all(seed+100000)
    optimizer=torch.optim.Adam(model.parameters(),lr=info['lr'])
    start=time.time()
    trace=[]; curves=[];best=float('inf');best_epoch=None
    with (out/'events.jsonl').open('x',buffering=1) as events:
        for epoch in range(1 if steps else info['epochs']):
            row={'epoch_1based':epoch+1}
            for split in ['train','val']:
                if steps and split=='val':
                    continue
                training=split=='train';model.train(training)
                total=0.;count=0;records=0;order=hashlib.sha256()
                with torch.set_grad_enabled(training):
                    for j,batch in enumerate(data[split]):
                        if steps and j>=steps:
                            break
                        order.update(json.dumps(batch.complex_id).encode())
                        graph_hash=None
                        if training and epoch==0 and j<8:
                            graph_hash=state_digest({k:getattr(batch,k) for k in ['x','atoms','edge_index','edge_s','edge_v','batch','ptr']})
                        batch=batch.to(device)
                        if training:
                            optimizer.zero_grad()
                        pred=model(batch)
                        loss=torch.nn.functional.mse_loss(pred,batch.label)
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite loss')
                        if training:
                            loss.backward()
                            if epoch==0 and j<8:
                                trace.append(dict(batch=j,ids=list(batch.complex_id),graph_sha256=graph_hash,
                                                  loss=float(loss),gradient_sha256=state_digest({k:p.grad for k,p in model.named_parameters() if p.grad is not None})))
                            optimizer.step()
                            if epoch==0 and j<8:
                                trace[-1]['updated_state_sha256']=state_digest(model.state_dict())
                        total+=float(loss.detach());count+=1;records+=len(batch.complex_id)
                if not steps and records!=len(data[split].dataset):
                    raise ValueError('Incomplete epoch')
                row[split+'_loss']=total/count
                row[split+'_order_sha256']=order.hexdigest()
                row[split+'_count']=records
            if not steps:
                torch.save(model.state_dict(),out/'last.pt')
                if row['val_loss']<best:
                    best=row['val_loss'];best_epoch=epoch+1
                    torch.save(model.state_dict(),out/'best.pt')
            curves.append(row);events.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
    final=state_digest(model.state_dict())
    summary=dict(arm=arm,seed=seed,strict=strict,initial_sha256=initial,shared_initial_sha256=shared,
                 final_sha256=final,trace=trace,curves=curves,best_epoch_1based=best_epoch,
                 node=platform.node(),gpu=torch.cuda.get_device_name(0) if device=='cuda' else 'cpu',
                 started_at=start,seconds=time.time()-start,
                 torch_version=torch.__version__,cuda_version=torch.version.cuda,
                 cudnn_version=torch.backends.cudnn.version(),
                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                 cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),pythonhashseed=os.environ.get('PYTHONHASHSEED'))
    summary['gvp_source']=__import__('gvp').__file__
    summary['cuda_visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
    if device=='cuda':
        summary['nvidia_smi']=subprocess.check_output(
            ['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    if not steps:
        model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True),strict=True)
        summary['best_state_sha256']=state_digest(model.state_dict())
        model.eval();y=[];p=[];ids=[]
        with torch.no_grad():
            for batch in data['test']:
                ids.extend(batch.complex_id);batch=batch.to(device)
                y.extend(batch.label.cpu().tolist());p.extend(model(batch).cpu().tolist())
        if ids!=info['split_ids']['test']:
            raise ValueError('Test coverage/order mismatch')
        with (out/'predictions.csv').open('x',newline='') as f:
            writer=csv.writer(f);writer.writerow(['id','target','prediction']);writer.writerows(zip(ids,y,p))
        summary['metrics']=regression(y,p)
        summary['predictions_sha256']=hashlib.sha256((out/'predictions.csv').read_bytes()).hexdigest()
    summary['seconds']=time.time()-start
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--arm',choices=['baseline','ep5','ep20'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--native',action='store_true');p.add_argument('--steps',type=int)
    a=p.parse_args();info=json.loads((a.run_root/'run.json').read_text())
    run(info,a.arm,a.seed,a.out,not a.native,a.steps,info['workers'])


if __name__=='__main__':
    main()
