"""Portable, isolated EP-20 LBA submission, validation, and reporting workflow."""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

GVP = Path(__file__).resolve().parent
WORKSPACE = GVP.parent
EVAL = WORKSPACE / "Estats/dl_model/evaluate_proteinshake_V2"
sys.path.insert(0, str(EVAL))
COUNTS = {"train": 3507, "val": 466, "test": 490}
SEEDS = [42, 123, 7, 2026, 17, 29, 53, 101, 211, 307]
ARMS = ["ep20", "baseline"]
CHECKPOINT_SHA = "950e4183f32ec959ac2b16b262806a9427015e8666925100c963f0aff4d17748"
METRICS = ["rmse", "pearson_r", "spearman_r", "r2"]
LABELS = {"ep20": "GVP + EP-20 (V9 decoder)", "baseline": "GVP",
          "ep20_only": "EP-20 (V9 decoder), ridge"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive publication prevents silently overwriting runs and completions.
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_verified(path, value):
    """Recover an interrupted audit only when an existing artifact is identical."""
    if Path(path).exists():
        if read(path) != value:
            raise ValueError("Existing audit artifact differs: " + str(path))
    else:
        write_new(path, value)


def relative(path, root=WORKSPACE):
    return os.path.relpath(Path(path).resolve(), Path(root).resolve())


def resolve(path, root=WORKSPACE):
    if Path(path).is_absolute():
        raise ValueError("Manifest paths must be relative to their declared root")
    return (Path(root) / path).resolve()


def task(info, index):
    n = len(info["seeds"])
    if index is None or not 0 <= index < len(info["arms"]) * n:
        raise ValueError("Training task index out of range")
    return info["arms"][index // n], info["seeds"][index % n]


@contextlib.contextmanager
def lock(root, name):
    path = Path(root) / "locks" / (name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def config(root):
    info = read(Path(root) / "run.json")
    if (info["schema"] != "ep20-lba-v1" or info["checkpoint_sha256"] != CHECKPOINT_SHA
            or info["ep_completed_epochs"] != 20 or info["seeds"] != SEEDS
            or info["arms"] != ARMS or info["epochs"] != 50):
        raise ValueError("Unexpected EP-20 protocol")
    for name, digest in info["source_sha256"].items():
        if sha(resolve(name)) != digest:
            raise ValueError("Source changed since initialization: " + name)
    if sha(resolve(info["checkpoint"])) != info["checkpoint_sha256"]:
        raise ValueError("EP checkpoint changed")
    return info


def initialize(root):
    import v9_lba as v9
    from atom3d.datasets import LMDBDataset
    root = Path(root).resolve()
    checkpoint = WORKSPACE / "Estats/dl_model/V9/V9_20_epoch_dpss.pt"
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Unexpected checkpoint content")
    obj, architecture = v9.read_checkpoint(checkpoint)
    if obj["epoch"] != 19:
        raise ValueError("Expected zero-based epoch 19")
    model = v9.load_v9_model(checkpoint, "cpu")
    assert not any(p.requires_grad for p in model.parameters())
    del model, obj
    base = v9.cache_contract(checkpoint, "train", "decoder")
    sources = {"Estats/" + p for p in base["source_sha256"]}
    sources.update("gvp-pytorch/" + p for p in [
        "lba_ep20.py", "run_atom3d_ep20.py", "submit_lba_ep20_gpu.sh",
        "submit_lba_ep20_cpu.sh", "ep20_job.sh", "gvp/atom3d.py", "gvp/__init__.py",
        "summarize_lba_v8.py", "lba_v9_workflow.py"])
    sources.update("Estats/dl_model/evaluate_proteinshake_V2/" + p for p in
                   ["check_v9_lba_cache.py", "v9_lba_probes.py", "metrics.py"])
    split_ids = {}
    dataset_sha256 = {}
    for split, count in COUNTS.items():
        ds = LMDBDataset(str(v9.DEFAULT_LMDB / split))
        previous = GVP / "atom3d-data/LBA/splits/split-by-sequence-identity-30/data" / split
        # py-lmdb forbids opening the same environment twice through symlink aliases.
        old = ds if os.path.samefile(v9.DEFAULT_LMDB / split, previous) else LMDBDataset(str(previous))
        if len(ds) != count or ds.ids() != old.ids() or len(set(ds.ids())) != count:
            raise ValueError("LBA cohort/order differs from the original GVP dataset")
        if v9.dataset_ids(ds) != list(ds.ids()):
            raise ValueError("LMDB IDs are not in index order; prediction annotation needs adaptation")
        split_ids[split] = list(ds.ids())
        datafile = v9.DEFAULT_LMDB / split / "data.mdb"
        oldfile = GVP / "atom3d-data/LBA/splits/split-by-sequence-identity-30/data" / split / "data.mdb"
        digest = sha(datafile)
        if not os.path.samefile(datafile, oldfile) and sha(oldfile) != digest:
            raise ValueError("LBA LMDB differs from the historical GVP dataset")
        dataset_sha256[relative(datafile)] = digest
    cache = EVAL / ("v9_lba_precomputed_cache_" + root.name)
    smoke = EVAL / ("v9_lba_precomputed_cache_" + root.name + "_smoke")
    if root.exists() or cache.exists() or smoke.exists():
        raise ValueError("Choose a fresh run root; outputs already exist")
    info = dict(schema="ep20-lba-v1", model_family="ElectroProt", model_version="V9",
                ep_completed_epochs=20, checkpoint_epoch=19, checkpoint=relative(checkpoint),
                checkpoint_sha256=CHECKPOINT_SHA, architecture=architecture,
                representation="decoder", charge_mode="real", width=256,
                cache_root=relative(cache), smoke_cache_root=relative(smoke),
                lmdb_root=relative(v9.DEFAULT_LMDB), pqr_root=relative(v9.DEFAULT_PQR),
                arms=ARMS, seeds=SEEDS, epochs=50, batch=8, lr=1e-4, workers=4,
                train_time=0, val_time=0, counts=COUNTS, split_ids=split_ids,
                dataset_sha256=dataset_sha256,
                source_sha256={p: sha(resolve(p)) for p in sorted(sources)},
                revisions={r: subprocess.check_output(
                    ["git", "-C", str(WORKSPACE/r), "rev-parse", "HEAD"], text=True).strip()
                           for r in ["Estats", "gvp-pytorch"]},
                git_status={r: subprocess.check_output(
                    ["git", "-C", str(WORKSPACE/r), "status", "--short"], text=True)
                            for r in ["Estats", "gvp-pytorch"]},
                created_at=time.time())
    write_new(root / "run.json", info)
    print("INITIALIZED", relative(root))


def validate_cache(cache):
    import v9_lba as v9
    cache = Path(cache).resolve()
    audit = read(cache / "audit.json")
    if (audit.get("passed") is not True or audit.get("cache_root") != "."
            or audit.get("checkpoint_sha256") != CHECKPOINT_SHA
            or audit.get("model_family") != "v9" or audit.get("charge_mode") != "real"
            or {s["split"]: s["count"] for s in audit["splits"]} != COUNTS):
        raise ValueError("A complete portable EP-20 cache audit is required")
    for rep in ("encoder", "decoder"):
        for split in COUNTS:
            path = v9.directory(cache, rep, split)
            if sha(path / "_manifest.json") != audit["manifest_sha256"][rep + "/" + split]:
                raise ValueError("Cache changed after audit")
            m = v9.read_manifest(path)
            c = m["contract"]
            if (len(m["entries"]) != COUNTS[split] or c["checkpoint_sha256"] != CHECKPOINT_SHA
                    or c["checkpoint_epoch"] != 19 or c["representation"] != rep
                    or c["split"] != split):
                raise ValueError("Wrong EP checkpoint/representation/cohort")
            for name, digest in c["source_sha256"].items():
                if sha(WORKSPACE / "Estats" / name) != digest:
                    raise ValueError("Extraction source changed: " + name)
    if sha(cache / "test_targets.json") != audit["test_targets_sha256"]:
        raise ValueError("Canonical targets changed")
    return audit


def audit_cache(info, cache, smoke=False):
    import torch
    import v9_lba as v9
    from check_v9_lba_cache import audit_split
    from atom3d.datasets import LMDBDataset
    reports, hashes, targets = [], {}, {}
    for split in (["val"] if smoke else COUNTS):
        ds = LMDBDataset(str(resolve(info["lmdb_root"]) / split))
        if list(ds.ids()) != info["split_ids"][split]:
            raise ValueError("Dataset membership/order changed")
        if smoke:
            ds = torch.utils.data.Subset(ds, range(5))
        reports.append(audit_split(ds, cache, resolve(info["pqr_root"]),
                                   resolve(info["checkpoint"]), split))
        for rep in ("encoder", "decoder"):
            hashes[rep + "/" + split] = sha(v9.directory(cache, rep, split) / "_manifest.json")
        if split == "test":
            targets = {str(ds[i]["id"]): float(ds[i]["scores"]["neglog_aff"]) for i in range(len(ds))}
    report = dict(passed=True, model_family="v9", charge_mode="real", cache_root=".",
                  checkpoint_sha256=info["checkpoint_sha256"], splits=reports,
                  manifest_sha256=hashes, smoke=smoke)
    if not smoke:
        write_verified(cache / "test_targets.json", targets)
        report["test_targets_sha256"] = sha(cache / "test_targets.json")
    write_verified(cache / "audit.json", report)


def extract(root, index, smoke=False):
    info = config(root)
    import torch
    torch.zeros(1).cuda()
    if not smoke and (index is None or not 0 <= index <= 2):
        raise ValueError("Cache array index must be 0, 1 or 2")
    split = "val" if smoke else ["val", "test", "train"][index]
    cache = resolve(info["smoke_cache_root"] if smoke else info["cache_root"])
    command = [sys.executable, "-u", str(EVAL / "precompute_v9_lba.py"),
               "--lmdb-root", str(resolve(info["lmdb_root"])), "--pqr-root", str(resolve(info["pqr_root"])),
               "--checkpoint", str(resolve(info["checkpoint"])), "--out-root", str(cache),
               "--split", split, "--device", "cuda"]
    if smoke:
        command += ["--limit", "5"]
    subprocess.run(command, check=True)
    if smoke:
        audit_cache(info, cache, smoke=True)
    print("EP20_EXTRACTION_COMPLETE", split, "smoke=" + str(smoke), flush=True)


def audit(root):
    info = config(root)
    cache = resolve(info["cache_root"])
    with lock(root, "audit"):
        for path, digest in info["dataset_sha256"].items():
            if sha(resolve(path)) != digest:
                raise ValueError("LMDB changed after initialization: " + path)
        audit_cache(info, cache)
        validate_cache(cache)
        write_verified(Path(root) / "cache_ready.json",
                  dict(run_sha256=sha(Path(root)/"run.json"), audit_sha256=sha(cache/"audit.json")))
    print("EP20_FULL_CACHE_AUDIT_PASSED", flush=True)


def ready(root):
    root = Path(root)
    info = config(root)
    cache = resolve(info["cache_root"])
    receipt = read(root / "cache_ready.json")
    if (receipt["run_sha256"] != sha(root/"run.json") or
            receipt["audit_sha256"] != sha(cache/"audit.json")):
        raise ValueError("Cache readiness receipt changed")
    validate_cache(cache)
    return info


def validate_predictions(path, targets):
    import numpy as np
    from metrics import regression_metrics
    rows = list(csv.DictReader(Path(path).open()))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(targets):
        raise ValueError("Prediction ID coverage mismatch")
    y = np.array([float(row["target"]) for row in rows])
    p = np.array([float(row["prediction"]) for row in rows])
    canonical = np.array([targets[i] for i in ids])
    if not np.allclose(y, canonical, rtol=0, atol=1e-6):
        raise ValueError("Prediction targets disagree with LMDB")
    result = regression_metrics(y, p)
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite predictions/metrics")
    return result


def execute(command, log):
    with log.open("a") as handle:
        handle.write("COMMAND_JSON: " + json.dumps(command) + "\n")
        handle.flush()
        child = subprocess.Popen(command, cwd=GVP, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
        for line in child.stdout:
            handle.write(line)
            print(line, end="", flush=True)
        if child.wait():
            raise RuntimeError("Command failed; see " + str(log))
    text = log.read_text()
    if any(x in text for x in ["Skipped batch due to OOM", "CUDA out of memory",
                              "Traceback (most recent call last)"]):
        raise ValueError("Failed/skipped training or test batch")


def artifact_record(root, paths, **metadata):
    return dict(metadata, run_sha256=sha(root/"run.json"),
                files={relative(p, root): sha(p) for p in paths}, completed_at=time.time())


def train(root, index):
    import numpy as np
    import torch
    from summarize_lba_v8 import parse_log
    root = Path(root).resolve()
    info = ready(root)
    arm, seed = task(info, index)
    with lock(root, arm + "_" + str(seed)):
        completion = root / "completed" / (arm + "_" + str(seed) + ".json")
        if completion.exists():
            raise ValueError("Task already complete")
        attempt = root / "attempts" / (arm + "_" + str(seed)) / (
            os.environ["SLURM_JOB_ID"] + "_r" + os.environ.get("SLURM_RESTART_COUNT", "0"))
        attempt.mkdir(parents=True, exist_ok=False)
        (attempt/"models").mkdir()
        torch.zeros(1).cuda()
        meta = dict(arm=arm, seed=seed, job=os.environ["SLURM_JOB_ID"],
                    array=os.environ.get("SLURM_ARRAY_JOB_ID"), index=index, node=platform.node(),
                    gpu=torch.cuda.get_device_name(0), started_at=time.time(),
                    packages=json.loads(subprocess.check_output(
                        [sys.executable, "-m", "pip", "list", "--format=json"], text=True)))
        common = [sys.executable, "-u", "run_atom3d_ep20.py", "LBA", "--lba-split", "30",
                  "--batch", str(info["batch"]), "--num-workers", str(info["workers"]),
                  "--lr", str(info["lr"]), "--seed", str(seed),
                  "--models-dir", relative(attempt/"models", GVP)]
        if arm == "ep20":
            common += ["--v9-decoder-cache", relative(resolve(info["cache_root"]), GVP)]
        os.environ["EP20_LMDB_ROOT"] = str(resolve(info["lmdb_root"]))
        log = attempt / "run.log"
        start = time.monotonic()
        execute(common + ["--epochs", str(info["epochs"]), "--train-time", "0", "--val-time", "0"], log)
        meta["train_seconds"] = time.monotonic() - start
        best = attempt/"models"/("LBA_seed" + str(seed) + "_best.pt")
        if not best.is_file():
            raise ValueError("Missing selected checkpoint")
        with log.open("a") as handle:
            handle.write("\nTRAIN_OK\n")
        start = time.monotonic()
        execute(common + ["--test", relative(best, GVP),
                          "--predictions-file", relative(attempt/"predictions.csv", GVP)], log)
        meta["test_seconds"] = time.monotonic() - start
        with log.open("a") as handle:
            handle.write("\nTEST_OK\n")
        logged = parse_log(log, info["epochs"])
        for stage in ["TRAIN", "VAL"]:
            values = [float(v) for v in re.findall(r"EPOCH \d+ " + stage + r"\s+loss:\s*(\S+)", log.read_text())]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite learning curve")
            if stage == "VAL":
                meta["best_epoch_1based"] = int(np.argmin(values)) + 1
        metrics = validate_predictions(attempt/"predictions.csv",
                                        read(resolve(info["cache_root"])/"test_targets.json"))
        if any(abs(metrics[k] - logged[k]) > 0.00011 for k in METRICS):
            raise ValueError("Predictions/log metrics disagree")
        meta["finished_at"] = time.time()
        write_new(attempt/"attempt.json", meta)
        ready(root)
        paths = [log, attempt/"predictions.csv", attempt/"attempt.json", best,
                 attempt/"models"/("LBA_seed" + str(seed) + "_last.pt")]
        write_new(completion, artifact_record(root, paths, arm=arm, seed=seed, metrics=metrics))
    print("EP20_TRAIN_COMPLETE", arm, seed, flush=True)


def ridge(root):
    import numpy as np
    import torch
    from atom3d.datasets import LMDBDataset
    from v9_lba_probes import pool_residues, fit_probe
    from v8_lba import input_fingerprint, tensor_hash
    root = Path(root).resolve()
    info = ready(root)
    cache = resolve(info["cache_root"])
    with lock(root, "ep20_only"):
        if (root/"completed/ep20_only.json").exists():
            raise ValueError("EP-only ridge is already complete")
        out = root/"attempts/ep20_only"/(
            os.environ["SLURM_JOB_ID"] + "_r" + os.environ.get("SLURM_RESTART_COUNT", "0"))
        out.mkdir(parents=True, exist_ok=False)
        arrays = {}
        for split in COUNTS:
            ds = LMDBDataset(str(resolve(info["lmdb_root"])/split))
            manifest = read(cache/"decoder/charge_real"/split/"_manifest.json")
            if list(ds.ids()) != info["split_ids"][split]:
                raise ValueError("Probe dataset changed")
            features, labels, ids = [], [], []
            for i in range(len(ds)):
                elem = ds[i]
                pid = elem["id"]
                rec = manifest["entries"][pid]
                tensor = torch.load(cache/"decoder/charge_real"/split/(pid+"_pocket.pt"),
                                    map_location="cpu", weights_only=True)
                if (input_fingerprint(elem) != rec["input_sha256"] or
                        tensor_hash(tensor) != rec["tensor_sha256"]):
                    raise ValueError("Probe input/cache changed")
                features.append(pool_residues(tensor.numpy(), elem["atoms_pocket"]))
                labels.append(float(elem["scores"]["neglog_aff"]))
                ids.append(pid)
            arrays[split+"_x"] = np.array(features, dtype=float)
            arrays[split+"_y"] = np.array(labels)
            arrays[split+"_ids"] = np.array(ids)
        np.savez_compressed(out/"features.npz", **arrays)
        scaler, model, search = fit_probe(arrays["train_x"], arrays["train_y"],
                                          arrays["val_x"], arrays["val_y"])
        p = model.predict(scaler.transform(arrays["test_x"]))
        np.savez(out/"model.npz", coef=model.coef_, intercept=model.intercept_,
                 mean=scaler.mean_, scale=scaler.scale_)
        with (out/"predictions.csv").open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "target", "prediction"])
            writer.writerows(zip(arrays["test_ids"], arrays["test_y"], p))
        metrics = validate_predictions(out/"predictions.csv", read(cache/"test_targets.json"))
        write_new(out/"result.json", dict(arm="ep20_only", label=LABELS["ep20_only"],
                  checkpoint_sha256=CHECKPOINT_SHA, readout="linear ridge",
                  pooling="mean atoms within residue, then mean residues",
                  alpha=float(model.alpha), validation_search=search, metrics=metrics,
                  fitting="train only; validation alpha selection; no train+val refit",
                  packages=json.loads(subprocess.check_output(
                      [sys.executable, "-m", "pip", "list", "--format=json"], text=True))))
        ready(root)
        write_new(root/"completed/ep20_only.json", artifact_record(root,
                  [out/p for p in ["features.npz", "model.npz", "predictions.csv", "result.json"]],
                  arm="ep20_only", seed=None, metrics=metrics))
    print("EP20_RIDGE_COMPLETE", flush=True)


def report(root):
    import numpy as np
    root = Path(root).resolve()
    info = ready(root)
    targets = read(resolve(info["cache_root"])/"test_targets.json")
    rows = []
    for arm, seed in [task(info, i) for i in range(20)] + [("ep20_only", None)]:
        name = arm + ("_" + str(seed) if seed is not None else "")
        receipt = read(root/"completed"/(name+".json"))
        if (receipt["run_sha256"] != sha(root/"run.json") or
                receipt["arm"] != arm or receipt["seed"] != seed):
            raise ValueError("Wrong completion identity")
        for path, digest in receipt["files"].items():
            if sha(resolve(path, root)) != digest:
                raise ValueError("Result artifact changed: " + path)
        pred = next(p for p in receipt["files"] if p.endswith("/predictions.csv"))
        metrics = validate_predictions(resolve(pred, root), targets)
        if metrics != receipt["metrics"]:
            raise ValueError("Changed metrics")
        rows.append(dict(arm=arm, label=LABELS[arm], seed=seed, **metrics,
                         prediction_file=pred, checkpoint_sha256=CHECKPOINT_SHA if arm != "baseline" else ""))
    out = root/"report"
    out.mkdir(exist_ok=True)
    with (out/"results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arrays = {a: np.array([[r[k] for k in METRICS] for r in rows if r["arm"] == a]) for a in ARMS}
    diff = arrays["ep20"] - arrays["baseline"]
    with (out/"paired_differences.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed"] + METRICS)
        writer.writerows([[s, *v] for s, v in zip(SEEDS, diff)])
    lines = ["# EP-20 / ATOM3D LBA-30", "",
             "Ten matched GVP seeds; mean ± population SD (ddof=0). Ridge is deterministic.",
             "Seed variability on one fixed split; no confidence interval or significance claim.", "",
             "| Model | n | RMSE | Pearson r | Spearman rho | R² |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, x in arrays.items():
        lines.append("| "+LABELS[arm]+" | 10 | "+" | ".join(
            f"{m:.4f} ± {s:.4f}" for m, s in zip(x.mean(0), x.std(0)))+" |")
    lines.append("| "+LABELS["ep20_only"]+" | 1 | "+
                 " | ".join(f"{rows[-1][k]:.4f}" for k in METRICS)+" |")
    lines.append("| Paired EP-20 − GVP | 10 | "+" | ".join(
        f"{m:+.4f} ± {s:.4f}" for m, s in zip(diff.mean(0), diff.std(0)))+" |")
    (out/"COMPARISON.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for j, ax in enumerate(axes.flat):
        a, b = arrays["baseline"][:, j], arrays["ep20"][:, j]
        for x, y in zip(a, b):
            ax.plot([0, 1], [x, y], color="0.8", zorder=0)
        for i, vals in enumerate([a, b]):
            ax.scatter(np.full(10, i), vals, s=24)
            ax.errorbar(i+0.12, vals.mean(), yerr=vals.std(ddof=0), fmt="ks", capsize=4)
        ax.set_xticks([0, 1], ["GVP", "GVP + EP-20"])
        ax.set_title(METRICS[j] + (" ↓" if j == 0 else " ↑"))
    fig.suptitle("ATOM3D LBA-30: 10 matched seeds\nMean ± population SD; EP pretraining 20 epochs, GVP training 50 epochs")
    fig.tight_layout()
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(out/("paired_metrics."+suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    (out/"RESULTS_AUDIT.json").write_text(json.dumps(dict(
        passed=True, run_sha256=sha(root/"run.json"), test_count=len(targets),
        evaluation_count=len(rows), checkpoint_sha256=CHECKPOINT_SHA,
        scope="Source/cache readiness and completion hashes; canonical targets and full-precision predictions.",
        wins=dict(zip(METRICS, [int((diff[:, 0] < 0).sum())] +
                      [int((diff[:, j] > 0).sum()) for j in range(1, 4)]))), indent=2)+"\n")
    print("\n".join(lines), flush=True)


def submit(root, dry_run=False):
    root = Path(root).resolve()
    config(root)
    if (root/"submissions.jsonl").exists():
        raise ValueError("Submission journal exists; inspect/retry individual stages explicitly")
    jobs = {}
    stages = [
        ("smoke", "gpu", "0", "01:00:00", []),
        ("cache", "gpu", "0-2%2", "06:00:00", ["smoke"]),
        ("audit", "cpu", None, "02:00:00", ["cache"]),
        ("train", "gpu", "0-19%2", "12:00:00", ["audit"]),
        ("ridge", "cpu", None, "02:00:00", ["audit"]),
        ("report", "cpu", None, "01:00:00", ["train", "ridge"])]
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    for stage, resource, array, walltime, predecessors in stages:
        command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes",
                   "--job-name=ep20_"+stage, "--time="+walltime,
                   "--output="+relative(root/(stage+"_%A_%a.out"), GVP),
                   "--error="+relative(root/(stage+"_%A_%a.err"), GVP)]
        if array:
            command.append("--array="+array)
        if predecessors:
            command.append("--dependency=afterok:" + ":".join(jobs[p] for p in predecessors))
        command += ["submit_lba_ep20_"+resource+".sh", stage, relative(root, GVP)]
        print(json.dumps(command), flush=True)
        if dry_run:
            jobs[stage] = "<"+stage+">"
            continue
        result = subprocess.check_output(command, cwd=GVP, env=env, text=True).strip()
        job = result.split(";")[0]
        if not job.isdigit():
            raise ValueError("Unexpected sbatch response: " + result)
        jobs[stage] = job
        with (root/"submissions.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(stage=stage, job=job, command=command,
                                         submitted_at=time.time()))+"\n")
            handle.flush()
            os.fsync(handle.fileno())
    if not dry_run:
        write_new(root/"jobs.json", jobs)
    print(json.dumps(jobs, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["initialize", "submit", "smoke", "cache", "audit", "train", "ridge", "report"])
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--index", type=int, default=int(os.environ["SLURM_ARRAY_TASK_ID"])
                        if "SLURM_ARRAY_TASK_ID" in os.environ else None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    if args.stage == "initialize":
        initialize(root)
    elif args.stage == "submit":
        submit(root, args.dry_run)
    elif args.stage in ("smoke", "cache"):
        extract(root, args.index, smoke=args.stage == "smoke")
    elif args.stage == "train":
        train(root, args.index)
    else:
        {"audit": audit, "ridge": ridge, "report": report}[args.stage](root)


if __name__ == "__main__":
    main()
