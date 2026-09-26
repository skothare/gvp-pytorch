"""Isolated raw q/r, frozen random decoder and within-pocket shuffled EP controls."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
import torch
import lba_ep20 as ep
import lba_control_features as f

ARMS = list(f.ARMS)
SPLITS = ["val", "test", "train"]
SCHEMA = "lba-ep20-controls-v1"


def own_sources():
    names = ["lba_controls.py", "lba_control_features.py", "run_atom3d_controls.py",
             "submit_lba_controls_gpu.sh", "submit_lba_controls_cpu.sh", "lba_controls_job.sh"]
    return {ep.relative(ep.GVP/n): ep.sha(ep.GVP/n) for n in names}


def config(root):
    root = Path(root)
    info = ep.read(root/"run.json")
    if (info["schema"] != SCHEMA or info["arms"] != ARMS or info["seeds"] != ep.SEEDS
            or info["epochs"] != 50 or info["counts"] != ep.COUNTS
            or info["feature_widths"] != f.WIDTHS):
        raise ValueError("Unexpected control protocol")
    parent = ep.resolve(info["parent_run"])
    if ep.sha(parent/"run.json") != info["parent_run_sha256"]:
        raise ValueError("Parent run changed")
    source = ep.ready(parent)
    if ep.sha(ep.resolve(source["cache_root"])/"audit.json") != info["parent_audit_sha256"]:
        raise ValueError("Parent cache audit changed")
    for name, digest in info["source_sha256"].items():
        if ep.sha(ep.resolve(name)) != digest:
            raise ValueError("Source changed: " + name)
    if ep.sha(ep.resolve(info["random_state"])) != info["random_state_sha256"]:
        raise ValueError("Random EP initialization changed")
    for key in ("seeds", "epochs", "counts", "split_ids", "dataset_sha256", "architecture",
                "batch", "lr", "workers", "lmdb_root", "pqr_root"):
        if info[key] != source[key]:
            raise ValueError("Control/parent mismatch: " + key)
    return info


def initialize(root, parent, random_seed=1729, shuffle_seed=271828):
    import v9_lba as v9
    root, parent = Path(root).resolve(), Path(parent).resolve()
    base = ep.ready(parent)
    cache = ep.EVAL/("v9_lba_precomputed_cache_"+root.name)
    if root.exists() or cache.exists():
        raise ValueError("Choose fresh output/cache directories")
    for name, digest in base["dataset_sha256"].items():
        if ep.sha(ep.resolve(name)) != digest:
            raise ValueError("Parent LMDB changed")
    sources = dict(base["source_sha256"], **own_sources())
    model = f.initialize_random_model(base["architecture"], random_seed)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    pretrained, _ = v9.read_checkpoint(ep.resolve(base["checkpoint"]))
    reference = pretrained["model_state_dict"]
    if state.keys() != reference.keys() or any(state[k].shape != reference[k].shape for k in state):
        raise ValueError("Random/pretrained architecture mismatch")
    changed = [k for k in state if not torch.equal(state[k], reference[k])]
    if not any(k.startswith("encoder.") for k in changed) or not any(
            k.startswith("decoder.") and ".head." not in k for k in changed):
        raise ValueError("Random control must change encoder and decoder, not just the head")
    cache.mkdir(parents=True)
    random_path = cache/"random_ep_initialization.pt"
    v9.atomic_write(random_path, state, tensor=True)
    copied = {k: base[k] for k in ("seeds", "epochs", "counts", "split_ids", "dataset_sha256",
              "architecture", "batch", "lr", "workers", "lmdb_root", "pqr_root")}
    info = dict(copied, schema=SCHEMA, arms=ARMS, feature_widths=f.WIDTHS,
                parent_run=ep.relative(parent), parent_run_sha256=ep.sha(parent/"run.json"),
                parent_cache=base["cache_root"],
                parent_audit_sha256=ep.sha(ep.resolve(base["cache_root"])/"audit.json"),
                parent_training_job=ep.read(parent/"jobs.json")["train"],
                cache_root=ep.relative(cache), random_state=ep.relative(random_path),
                random_state_sha256=ep.sha(random_path), random_tensor_sha256=f.state_hash(state),
                random_init_seed=random_seed, shuffle_seed=shuffle_seed,
                random_parameter_count=sum(p.numel() for p in model.parameters()),
                random_changed_state_keys=len(changed), source_sha256=sources,
                initialization_runtime=dict(torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version()),
                protocol=dict(raw="train-atom standardization; original PQR inputs/fallbacks",
                              random="whole V9 model random initialized; frozen eval; pre-head decoder",
                              shuffle="fixed within-pocket complete-row permutation; all splits",
                              ligand="zero added features after any pocket standardization",
                              gpu_scheduling="afterany parent training array; no competing new GPUs",
                              initialization_replicates=1), created_at=time.time())
    ep.write_new(root/"run.json", info)
    config(root)
    print("CONTROLS_INITIALIZED", ep.relative(root), flush=True)


def directory(info, arm, split, smoke=False):
    if arm not in ARMS or split not in SPLITS:
        raise ValueError("Unknown control/split")
    cache = ep.resolve(info["cache_root"])
    return cache/("smoke" if smoke else "features")/arm/split


def manifest(root, arm, split, smoke=False, complete=True):
    info = config(root)
    path = directory(info, arm, split, smoke)
    m = ep.read(path/"manifest.json")
    ids = info["split_ids"][split][:2] if smoke else info["split_ids"][split]
    if (m["schema"] != SCHEMA or m["run_sha256"] != ep.sha(Path(root)/"run.json")
            or m["arm"] != arm or m["split"] != split or m["smoke"] != smoke
            or m["width"] != f.WIDTHS[arm] or m["expected_ids"] != ids
            or (complete and (not m["complete"] or m["pending"] is not None
                              or set(m["entries"]) != set(ids)))):
        raise ValueError("Invalid/incomplete control manifest")
    return m


def dataset(info, split):
    from atom3d.datasets import LMDBDataset
    ds = LMDBDataset(str(ep.resolve(info["lmdb_root"])/split))
    if list(ds.ids()) != info["split_ids"][split]:
        raise ValueError("Dataset ID order changed")
    return ds


def original_inputs(info, elem, split, source_manifest):
    import v9_lba as v9
    rec, lut, stems = v9.entry_inputs(elem, "real", ep.resolve(info["pqr_root"])/split)
    if any(source_manifest["entries"][str(elem["id"])].get(k) != v for k, v in rec.items()):
        raise ValueError("Control must use identical EP-20 inputs/PQR/fallbacks")
    return rec, lut, stems


def source_manifest(info, split):
    return ep.read(ep.resolve(info["parent_cache"])/"decoder/charge_real"/split/"_manifest.json")


def original_features(info, elem, split, m):
    import v9_lba as v9
    return v9.validate_entry(elem, ep.resolve(info["parent_cache"])/"decoder/charge_real"/split,
                            m["entries"][str(elem["id"])], "real", ep.resolve(info["pqr_root"])/split)


def random_model(info, device):
    state = torch.load(ep.resolve(info["random_state"]), map_location="cpu", weights_only=True)
    if f.state_hash(state) != info["random_tensor_sha256"]:
        raise ValueError("Random model state mismatch")
    model = f.initialize_random_model(info["architecture"], info["random_init_seed"])
    if f.state_hash(model.state_dict()) != info["random_tensor_sha256"]:
        raise ValueError("Initialization is not reproducible in this runtime")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def prepare_split(root, arm, split, smoke=False, device="cpu"):
    import v9_lba as v9
    info = config(root)
    path = directory(info, arm, split, smoke)
    ds = dataset(info, split)
    ids = info["split_ids"][split][:2] if smoke else info["split_ids"][split]
    src = source_manifest(info, split)
    model = random_model(info, device) if arm == "random_ep" else None
    with ep.lock(root, ("smoke_" if smoke else "cache_")+arm+"_"+split):
        path.mkdir(parents=True, exist_ok=True)
        mpath = path/"manifest.json"
        if mpath.exists():
            m = manifest(root, arm, split, smoke, complete=False)
        else:
            if list(path.glob("*.pt")):
                raise ValueError("Unrecorded control cache files")
            m = dict(schema=SCHEMA, run_sha256=ep.sha(Path(root)/"run.json"),
                     arm=arm, width=f.WIDTHS[arm], split=split, smoke=smoke,
                     expected_ids=ids, entries={}, pending=None, complete=False, extraction_attempts=[])
            v9.atomic_write(mpath, m)
        if not m["complete"]:
            m["extraction_attempts"].append(dict(job=os.environ.get("SLURM_JOB_ID"),
                node=platform.node(), device=device, torch=str(torch.__version__),
                numpy=np.__version__, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                started_at=time.time()))
            v9.atomic_write(mpath, m)
        if m["pending"]:
            pending = m["pending"]
            elem = ds[ids.index(pending["id"])]
            expected, _, _ = original_inputs(info, elem, split, src)
            if any(pending["record"].get(k) != v for k, v in expected.items()):
                raise ValueError("Pending extraction inputs changed")
            if (path/(pending["id"]+"_pocket.pt")).exists():
                f.load_features(path, pending["id"], pending["record"], f.WIDTHS[arm])
                m["entries"][pending["id"]] = pending["record"]
            m.update(pending=None, complete=False)
            v9.atomic_write(mpath, m)
        for i, identifier in enumerate(ids):
            elem = ds[i]
            if str(elem["id"]) != identifier:
                raise ValueError("Dataset index/ID changed")
            rec, lut, stems = original_inputs(info, elem, split, src)
            if identifier in m["entries"]:
                if any(m["entries"][identifier].get(k) != v for k, v in rec.items()):
                    raise ValueError("Cached control input changed")
                f.load_features(path, identifier, m["entries"][identifier], f.WIDTHS[arm])
                continue
            if (path/(identifier+"_pocket.pt")).exists():
                raise ValueError("Orphan control tensor")
            payload = {}
            if arm == "raw_qr":
                value = f.raw_features(elem, lut, stems)
            elif arm == "shuffled_ep":
                original = original_features(info, elem, split, src)
                perm = f.permutation(len(original), split, identifier, info["shuffle_seed"])
                value = original[perm].clone()
                payload["permutation"] = perm
                rec.update(permutation_sha256=f.tensor_hash(perm),
                           source_tensor_sha256=f.tensor_hash(original),
                           fixed_points=int((perm == torch.arange(len(perm))).sum()))
            else:
                batch, _ = v9.df_to_v8_batch(elem["atoms_pocket"], elem["atoms_protein"],
                                           device, "real", lut, stems)
                value = v9.extract_features(model, batch)["decoder"]
            f.validate_features(value, len(elem["atoms_pocket"]), f.WIDTHS[arm])
            payload["features"] = value
            rec["tensor_sha256"] = f.tensor_hash(value)
            m.update(complete=False, pending=dict(id=identifier, record=rec))
            v9.atomic_write(mpath, m)
            v9.atomic_write(path/(identifier+"_pocket.pt"), payload, tensor=True)
            m["entries"][identifier] = rec
            m["pending"] = None
            v9.atomic_write(mpath, m)
            if (i+1) % 50 == 0 or i+1 == len(ids):
                print("CONTROL_CACHE", arm, split, i+1, "/", len(ids), flush=True)
        m["complete"] = True
        v9.atomic_write(mpath, m)
    return m


def smoke(root):
    torch.zeros(1).cuda()
    info = config(root)
    prepare_split(root, "raw_qr", "train", smoke=True)
    for arm in ARMS:
        prepare_split(root, arm, "val", smoke=True, device="cuda")
    m = manifest(root, "raw_qr", "train", smoke=True)
    scaler = f.fit_scaler(f.load_features(directory(info, "raw_qr", "train", True),
                         identifier, m["entries"][identifier], 2)["features"] for identifier in m["expected_ids"])
    ep.write_verified(ep.resolve(info["cache_root"])/"smoke/raw_scaler.json", scaler)
    ds = dataset(info, "val")
    src = source_manifest(info, "val")
    m = manifest(root, "random_ep", "val", smoke=True)
    for i, identifier in enumerate(m["expected_ids"]):
        value = f.load_features(directory(info, "random_ep", "val", True), identifier,
                                m["entries"][identifier], 256)["features"]
        if torch.allclose(value, original_features(info, ds[i], "val", src)):
            raise ValueError("Random features unexpectedly equal pretrained features")
    ep.write_verified(Path(root)/"smoke_extraction.json", dict(passed=True,
                      run_sha256=ep.sha(Path(root)/"run.json")))
    print("CONTROL_EXTRACTION_SMOKE_PASSED", flush=True)


def smoke_gvp(root):
    from torch_geometric.data import Batch
    from gvp.atom3d import V3LBAModel
    info = config(root)
    if ep.read(Path(root)/"smoke_extraction.json") != dict(
            passed=True, run_sha256=ep.sha(Path(root)/"run.json")):
        raise ValueError("Missing extraction smoke")
    torch.zeros(1).cuda()
    ds = dataset(info, "val")
    reports = {}
    for arm in ARMS:
        m = manifest(root, arm, "val", smoke=True)
        scaler = ep.read(ep.resolve(info["cache_root"])/"smoke/raw_scaler.json") if arm == "raw_qr" else None
        transform = f.ControlTransform(directory(info, arm, "val", True), m, scaler)
        graphs = [transform(ds[i]) for i in range(2)]
        batch = Batch.from_data_list(graphs).to("cuda")
        torch.manual_seed(ep.SEEDS[0])
        model = V3LBAModel(v3_dim=f.WIDTHS[arm]).cuda()
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=info["lr"])
        pred = model(batch)
        loss = torch.nn.functional.mse_loss(pred, batch.label)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError("Nonfinite/absent GVP smoke gradients")
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(batch)
        if pred.shape != batch.label.shape or not torch.isfinite(pred).all():
            raise ValueError("Invalid GVP smoke predictions")
        reports[arm] = dict(loss=float(loss.detach()), parameter_count=sum(p.numel() for p in model.parameters()),
                            input_scalars=9+f.WIDTHS[arm])
    ep.write_new(Path(root)/"smoke_gvp.json", dict(passed=True,
                 run_sha256=ep.sha(Path(root)/"run.json"), arms=reports))
    print("CONTROL_GVP_SMOKE_PASSED", json.dumps(reports), flush=True)


def audit(root):
    info = config(root)
    for name, digest in info["dataset_sha256"].items():
        if ep.sha(ep.resolve(name)) != digest:
            raise ValueError("LMDB changed")
    manifests = {}
    with ep.lock(root, "audit"):
        train_values = []
        for split in SPLITS:
            ds, src = dataset(info, split), source_manifest(info, split)
            ms = {arm: manifest(root, arm, split) for arm in ARMS}
            for i, identifier in enumerate(info["split_ids"][split]):
                elem = ds[i]
                rec, lut, stems = original_inputs(info, elem, split, src)
                for arm in ARMS:
                    stored = ms[arm]["entries"][identifier]
                    if any(stored.get(k) != v for k, v in rec.items()):
                        raise ValueError("Audit input/PQR mismatch")
                    payload = f.load_features(directory(info, arm, split), identifier, stored, f.WIDTHS[arm])
                    value = payload["features"]
                    if arm == "raw_qr":
                        if not torch.equal(value, f.raw_features(elem, lut, stems)):
                            raise ValueError("Raw q/r differs from EP inputs")
                        if split == "train":
                            train_values.append(value)
                    elif arm == "shuffled_ep":
                        original = original_features(info, elem, split, src)
                        perm = f.permutation(len(original), split, identifier, info["shuffle_seed"])
                        if (not torch.equal(perm, payload["permutation"])
                                or not torch.equal(value, original[perm])
                                or stored["source_tensor_sha256"] != f.tensor_hash(original)):
                            raise ValueError("Shuffle does not preserve source rows")
                if (i+1) % 500 == 0:
                    print("CONTROL_AUDIT", split, i+1, flush=True)
            for arm in ARMS:
                manifests[arm+"/"+split] = ep.sha(directory(info, arm, split)/"manifest.json")
        scaler = f.fit_scaler(train_values)
        cache = ep.resolve(info["cache_root"])
        ep.write_verified(cache/"raw_scaler.json", scaler)
        config(root)
        smoke_record = ep.read(Path(root)/"smoke_gvp.json")
        if not smoke_record["passed"] or smoke_record["run_sha256"] != ep.sha(Path(root)/"run.json"):
            raise ValueError("GVP smoke not complete")
        ep.write_verified(Path(root)/"cache_ready.json", dict(passed=True,
                          run_sha256=ep.sha(Path(root)/"run.json"), manifests=manifests,
                          scaler_sha256=ep.sha(cache/"raw_scaler.json"),
                          smoke_sha256=ep.sha(Path(root)/"smoke_gvp.json")))
    print("CONTROL_CACHE_AUDIT_PASSED", flush=True)


def ready(root):
    info = config(root)
    r = ep.read(Path(root)/"cache_ready.json")
    if not r["passed"] or r["run_sha256"] != ep.sha(Path(root)/"run.json"):
        raise ValueError("Control cache not audited")
    for arm in ARMS:
        for split in SPLITS:
            path = directory(info, arm, split)/"manifest.json"
            if ep.sha(path) != r["manifests"][arm+"/"+split]:
                raise ValueError("Control manifest changed after audit")
    if (ep.sha(ep.resolve(info["cache_root"])/"raw_scaler.json") != r["scaler_sha256"]
            or ep.sha(Path(root)/"smoke_gvp.json") != r["smoke_sha256"]):
        raise ValueError("Scaler/smoke changed after audit")
    return info


def get_datasets(root, arm, data_path):
    from atom3d.datasets import LMDBDataset
    info = ready(root)
    if Path(data_path).resolve() != ep.resolve(info["lmdb_root"]):
        raise ValueError("Runner dataset differs from control cohort")
    scaler = ep.read(ep.resolve(info["cache_root"])/"raw_scaler.json") if arm == "raw_qr" else None
    datasets = []
    for split in ("train", "val", "test"):
        m = manifest(root, arm, split)
        ds = LMDBDataset(str(Path(data_path)/split), transform=f.ControlTransform(directory(info, arm, split), m, scaler))
        if list(ds.ids()) != m["expected_ids"]:
            raise ValueError("Control dataset order mismatch")
        datasets.append(ds)
    print("CONTROL_FUSION", arm, f.WIDTHS[arm], "features", flush=True)
    return tuple(datasets)


def train(root, index):
    from summarize_lba_v8 import parse_log
    root = Path(root).resolve()
    info = ready(root)
    arm, seed = ep.task(info, index)
    with ep.lock(root, arm+"_"+str(seed)):
        completion = root/"completed"/(arm+"_"+str(seed)+".json")
        if completion.exists():
            raise ValueError("Control task already complete")
        attempt = root/"attempts"/(arm+"_"+str(seed))/(
            os.environ["SLURM_JOB_ID"]+"_r"+os.environ.get("SLURM_RESTART_COUNT", "0"))
        attempt.mkdir(parents=True, exist_ok=False)
        (attempt/"models").mkdir()
        torch.zeros(1).cuda()
        meta = dict(arm=arm, seed=seed, random_init_seed=info["random_init_seed"],
                    shuffle_seed=info["shuffle_seed"], job=os.environ["SLURM_JOB_ID"],
                    array=os.environ.get("SLURM_ARRAY_JOB_ID"), index=index, node=platform.node(),
                    gpu=torch.cuda.get_device_name(0), started_at=time.time(),
                    packages=json.loads(subprocess.check_output(
                        [sys.executable, "-m", "pip", "list", "--format=json"], text=True)))
        common = [sys.executable, "-u", "run_atom3d_controls.py", "LBA", "--lba-split", "30",
                  "--controls-run", ep.relative(root, ep.GVP), "--control-arm", arm,
                  "--batch", str(info["batch"]), "--num-workers", str(info["workers"]),
                  "--lr", str(info["lr"]), "--seed", str(seed),
                  "--models-dir", ep.relative(attempt/"models", ep.GVP)]
        os.environ["EP20_LMDB_ROOT"] = str(ep.resolve(info["lmdb_root"]))
        log = attempt/"run.log"
        start = time.monotonic()
        ep.execute(common+["--epochs", str(info["epochs"]), "--train-time", "0", "--val-time", "0"], log)
        meta["train_seconds"] = time.monotonic()-start
        best = attempt/"models"/("LBA_seed"+str(seed)+"_best.pt")
        if not best.is_file():
            raise ValueError("Missing selected control checkpoint")
        with log.open("a") as h:
            h.write("\nTRAIN_OK\n")
        start = time.monotonic()
        ep.execute(common+["--test", ep.relative(best, ep.GVP),
                           "--predictions-file", ep.relative(attempt/"predictions.csv", ep.GVP)], log)
        meta["test_seconds"] = time.monotonic()-start
        with log.open("a") as h:
            h.write("\nTEST_OK\n")
        logged = parse_log(log, info["epochs"])
        for stage in ("TRAIN", "VAL"):
            losses = [float(v) for v in re.findall(r"EPOCH \d+ "+stage+r"\s+loss:\s*(\S+)", log.read_text())]
            if len(losses) != info["epochs"] or not np.isfinite(losses).all():
                raise ValueError("Incomplete/nonfinite control learning curve")
            if stage == "VAL":
                meta["best_epoch_1based"] = int(np.argmin(losses))+1
        targets = ep.read(ep.resolve(info["parent_cache"])/"test_targets.json")
        metrics = ep.validate_predictions(attempt/"predictions.csv", targets)
        if any(abs(metrics[k]-logged[k]) > .00011 for k in ep.METRICS):
            raise ValueError("Control prediction/log mismatch")
        counts = [int(v) for v in re.findall(r"CONTROL_PARAMETER_COUNT (\d+)", log.read_text())]
        if len(counts) != 2 or counts[0] != counts[1]:
            raise ValueError("Missing/inconsistent parameter counts")
        meta.update(parameter_count=counts[0], finished_at=time.time())
        ep.write_new(attempt/"attempt.json", meta)
        ready(root)
        files = [log, attempt/"predictions.csv", attempt/"attempt.json", best,
                 attempt/"models"/("LBA_seed"+str(seed)+"_last.pt")]
        ep.write_new(completion, ep.artifact_record(root, files, arm=arm, seed=seed, metrics=metrics,
                                                   cache_ready_sha256=ep.sha(root/"cache_ready.json")))
    print("CONTROL_TRAIN_COMPLETE", arm, seed, flush=True)


def verified_result(root, arm, seed, targets):
    root = Path(root)
    suffix = "" if seed is None else "_"+str(seed)
    r = ep.read(root/"completed"/(arm+suffix+".json"))
    if r["run_sha256"] != ep.sha(root/"run.json") or r["arm"] != arm or r["seed"] != seed:
        raise ValueError("Wrong completion identity")
    for path, digest in r["files"].items():
        if ep.sha(ep.resolve(path, root)) != digest:
            raise ValueError("Result changed: "+path)
    pred = next(p for p in r["files"] if p.endswith("/predictions.csv"))
    metrics = ep.validate_predictions(ep.resolve(pred, root), targets)
    if metrics != r["metrics"]:
        raise ValueError("Recomputed result differs")
    if arm in ARMS and r["cache_ready_sha256"] != ep.sha(root/"cache_ready.json"):
        raise ValueError("Control result/cache mismatch")
    return dict(arm=arm, seed=seed, **metrics, prediction_file=ep.relative(ep.resolve(pred, root)),
                completion_file=ep.relative(root/"completed"/(arm+suffix+".json")))


def report(root):
    info = ready(root)
    parent = ep.resolve(info["parent_run"])
    targets = ep.read(ep.resolve(info["parent_cache"])/"test_targets.json")
    rows = [verified_result(root, arm, seed, targets) for arm in ARMS for seed in ep.SEEDS]
    rows += [verified_result(parent, arm, seed, targets) for arm in ep.ARMS for seed in ep.SEEDS]
    rows += [verified_result(parent, "ep20_only", None, targets)]
    labels = dict(ep.LABELS, **f.LABELS)
    order = ["baseline", "raw_qr", "random_ep", "shuffled_ep", "ep20"]
    arrays = {arm: np.array([[r[k] for k in ep.METRICS] for r in rows if r["arm"] == arm]) for arm in order}
    out = Path(root)/"report"
    out.mkdir(exist_ok=True)
    with (out/"results.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    differences = []
    for arm in ARMS:
        for reference in ("baseline", "ep20"):
            for seed, delta in zip(ep.SEEDS, arrays[arm]-arrays[reference]):
                differences.append(dict(arm=arm, reference=reference, seed=seed, **dict(zip(ep.METRICS, delta))))
    with (out/"paired_differences.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(differences[0])); w.writeheader(); w.writerows(differences)
    lines = ["# ATOM3D LBA-30: EP-20 feature controls", "",
             "Ten matched GVP seeds, 50 downstream epochs, fixed split. Mean +/- population SD (ddof=0).",
             "Random EP uses ONE frozen initialization; its initialization variability is not estimated.",
             "Shuffle is fixed within each pocket. Ridge is deterministic. No significance claim.", "",
             "| Model | n | RMSE | Pearson r | Spearman rho | R2 |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in order:
        x = arrays[arm]
        lines.append("| "+labels[arm]+" | 10 | "+" | ".join(
            f"{m:.4f} +/- {s:.4f}" for m, s in zip(x.mean(0), x.std(0)))+" |")
    lines.append("| "+labels["ep20_only"]+" | 1 | "+" | ".join(f"{rows[-1][k]:.4f}" for k in ep.METRICS)+" |")
    (out/"COMPARISON.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    short = ["GVP", "+ raw q/r", "+ random EP", "+ shuffled EP", "+ EP-20"]
    for j, ax in enumerate(axes.flat):
        values = np.stack([arrays[a][:, j] for a in order], axis=1)
        for row in values:
            ax.plot(range(len(order)), row, color="0.85", linewidth=.8)
        for i in range(len(order)):
            ax.scatter(np.full(10, i), values[:, i], s=14)
            ax.errorbar(i+.12, values[:, i].mean(), yerr=values[:, i].std(), fmt="ks", capsize=3)
        ax.set_xticks(range(len(order)), short, rotation=15)
        ax.set_title(ep.METRICS[j]+(" (lower better)" if j == 0 else " (higher better)"))
    fig.suptitle("ATOM3D LBA-30: 10 GVP seeds, mean +/- population SD\nRandom EP: one initialization; shuffle: within-pocket rows")
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out/("control_metrics."+ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    (out/"RESULTS_AUDIT.json").write_text(json.dumps(dict(passed=True, evaluations=len(rows),
        test_count=len(targets), run_sha256=ep.sha(Path(root)/"run.json"),
        source_completions={r["completion_file"]: ep.sha(ep.resolve(r["completion_file"])) for r in rows}), indent=2)+"\n")
    print("CONTROL_REPORT_COMPLETE", flush=True)


def submit(root, dry_run=False):
    root = Path(root).resolve()
    info = config(root)
    if (root/"submissions.jsonl").exists():
        raise ValueError("Submission journal exists; do not submit the chain twice")
    jobs = {}
    stages = [("cache_cpu", "cpu", "0-5%2", "03:00:00", []),
              ("smoke", "gpu", "0", "01:00:00", []),
              ("smoke_gvp", "gpu", "0", "01:00:00", ["smoke"]),
              ("cache_random", "gpu", "0-2%2", "06:00:00", ["smoke_gvp"]),
              ("audit", "cpu", None, "03:00:00", ["cache_cpu", "cache_random"]),
              ("train", "gpu", "0-29%2", "12:00:00", ["audit"]),
              ("report", "cpu", None, "01:00:00", ["train"])]
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    for stage, resource, array, walltime, predecessors in stages:
        cmd = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", "--job-name=epctrl_"+stage,
               "--time="+walltime, "--output="+ep.relative(root/(stage+"_%A_%a.out"), ep.GVP),
               "--error="+ep.relative(root/(stage+"_%A_%a.err"), ep.GVP)]
        if array:
            cmd.append("--array="+array)
        if stage == "smoke":
            cmd.append("--dependency=afterany:"+info["parent_training_job"])
        elif predecessors:
            cmd.append("--dependency=afterok:"+":".join(jobs[p] for p in predecessors))
        cmd += ["submit_lba_controls_"+resource+".sh", stage, ep.relative(root, ep.GVP)]
        print(json.dumps(cmd), flush=True)
        if dry_run:
            jobs[stage] = "<"+stage+">"
            continue
        result = subprocess.check_output(cmd, cwd=ep.GVP, env=env, text=True).strip()
        job = result.split(";")[0]
        if not job.isdigit():
            raise ValueError("Unexpected sbatch response: "+result)
        jobs[stage] = job
        with (root/"submissions.jsonl").open("a") as h:
            h.write(json.dumps(dict(stage=stage, job=job, command=cmd, submitted_at=time.time()))+"\n")
            h.flush(); os.fsync(h.fileno())
    if not dry_run:
        ep.write_new(root/"jobs.json", jobs)
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["initialize", "submit", "smoke", "smoke_gvp", "cache_cpu", "cache_random", "audit", "train", "report"])
    p.add_argument("--run-root", required=True)
    p.add_argument("--parent-run")
    p.add_argument("--index", type=int, default=int(os.environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in os.environ else None)
    p.add_argument("--random-init-seed", type=int, default=1729)
    p.add_argument("--shuffle-seed", type=int, default=271828)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    root = Path(a.run_root).resolve()
    if a.stage == "initialize":
        if not a.parent_run:
            p.error("initialize requires --parent-run")
        initialize(root, a.parent_run, a.random_init_seed, a.shuffle_seed)
    elif a.stage == "submit":
        submit(root, a.dry_run)
    elif a.stage == "cache_cpu":
        if a.index is None or not 0 <= a.index < 6:
            p.error("cache_cpu index must be 0..5")
        prepare_split(root, ["raw_qr", "shuffled_ep"][a.index//3], SPLITS[a.index%3])
    elif a.stage == "cache_random":
        if a.index is None or not 0 <= a.index < 3:
            p.error("cache_random index must be 0..2")
        torch.zeros(1).cuda()
        prepare_split(root, "random_ep", SPLITS[a.index], device="cuda")
    elif a.stage == "train":
        train(root, a.index)
    else:
        {"smoke": smoke, "smoke_gvp": smoke_gvp, "audit": audit, "report": report}[a.stage](root)


if __name__ == "__main__":
    main()
