"""Original LBA train/test/loop functions, with unchanged execution semantics."""
import csv, time, os
import numpy as np
import torch, tqdm
import torch.nn as nn
from functools import partial
from metrics import regression_metrics
print=partial(print,flush=True)
def get_metrics(task): return {}
def test(model, testset):
    model.load_state_dict(torch.load(args.test))
    model.eval()
    t = tqdm.tqdm(testset)
    metrics = get_metrics(args.task)
    targets, predicts, ids = [], [], []
    with torch.no_grad():
        for batch in t:
            pred = forward(model, batch, device)
            label = get_label(batch, args.task, args.smp_idx)
            if args.task == 'RES':
                pred = pred.argmax(dim=-1)
            if args.predictions_file:
                ids.extend(batch.complex_id)
            if args.task in ['PSR', 'RSR']:
                ids.extend(batch.id)
            targets.extend(list(label.cpu().numpy()))
            predicts.extend(list(pred.cpu().numpy()))


    if args.task == 'LBA':
        if args.predictions_file:
            import csv
            with open(args.predictions_file, 'x', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['id', 'target', 'prediction'])
                writer.writerows(zip(ids, targets, predicts))
        result = regression_metrics(targets, predicts)
        print("\n=== LBA Test Metrics ===")
        for name, value in result.items():
            print(f"  {name}: {value:.4f}")
    else:
        for name, func in metrics.items():
            if args.task in ['PSR', 'RSR']:
                func = partial(func, ids=ids)
            value = func(targets, predicts)
            print(f"{name}: {value}")

def train(model, trainset, valset):
    """
    Updated to save the best checkpoint to a named path at the end of train() - SK 25Jun2026.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    seed = getattr(args, 'seed', 0)
    last_path = f"{models_dir}/{args.task}_seed{seed}_last.pt"
    best_path = f"{models_dir}/{args.task}_seed{seed}_best.pt"
    best_val = np.inf

    for epoch in range(args.epochs):
        model.train()
        train_loss = loop(trainset, model, optimizer=optimizer,
                          max_time=args.train_time)
        print(f'\nEPOCH {epoch} TRAIN loss: {train_loss:.8f}')

        model.eval()
        with torch.no_grad():
            val_loss = loop(valset, model, max_time=args.val_time)
        print(f'EPOCH {epoch} VAL   loss: {val_loss:.8f}')

        # Always overwrite last — lets you resume a crashed job
        torch.save(model.state_dict(), last_path)

        # Overwrite best only when val loss improves
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f'BEST  updated → {best_path}  (val loss: {best_val:.8f})')
        else:
            print(f'BEST  unchanged (val loss: {best_val:.8f})')

    print(f'\nTraining complete. Best checkpoint: {best_path}')
    print(f'Best val loss: {best_val:.8f}')

def loop(dataset, model, optimizer=None, max_time=None):
    start = time.time()

    loss_fn = get_loss(args.task)
    t = tqdm.tqdm(dataset)
    total_loss, total_count = 0, 0

    for batch in t:
        if max_time and (time.time() - start) > 60*max_time: break
        if optimizer: optimizer.zero_grad()
        try:
            out = forward(model, batch, device)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue

        label = get_label(batch, args.task, args.smp_idx)
        loss_value = loss_fn(out, label)
        total_loss += float(loss_value)
        total_count += 1

        if optimizer:
            try:
                loss_value.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise(e)
                torch.cuda.empty_cache()
                print('Skipped batch due to OOM', flush=True)
                continue

        t.set_description(f"{total_loss/total_count:.8f}")

    return total_loss / total_count

def get_label(batch, task, smp_idx=None):
    if type(batch) in [list, tuple]: batch = batch[0]
    if task == 'SMP':
        assert smp_idx is not None
        return batch.label[smp_idx::20]
    return batch.label

def get_loss(task):
    if task in ['PSR', 'RSR', 'SMP', 'LBA']: return nn.MSELoss() # regression
    elif task in ['PPI', 'MSP', 'LEP']: return nn.BCELoss() # binary classification
    elif task in ['RES']: return nn.CrossEntropyLoss() # multiclass classification

def forward(model, batch, device):
    if type(batch) in [list, tuple]:
        batch = batch[0].to(device), batch[1].to(device)
    else:
        batch = batch.to(device)
    return model(batch)
