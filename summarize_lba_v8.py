"""Validate completed LBA runs and summarize paired seeds (population std)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import os
import time

import numpy as np

ARMS = ("v8_real", "v8_zero", "baseline")
METRICS = ("rmse", "pearson_r", "spearman_r", "r2")
ROOT = Path(__file__).resolve().parent
ESTATS = ROOT.parent / "Estats"
EVAL = ESTATS / "dl_model/evaluate_proteinshake_V2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_info(root):
    return {"revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "status": subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True)}


def initialize(log_root, model_root, cache_root, phase, array=False, representation="encoder"):
    if representation not in ("encoder", "decoder"):
        raise ValueError("Unknown V8 representation")
    if representation == "decoder" and not array:
        raise ValueError("Decoder runs require --array")
    path = Path(log_root) / "run.json"
    if path.exists():
        raise ValueError(f"Refusing to replace existing run provenance: {path}")
    cache_root = Path(cache_root).resolve()
    audit = None
    if array:
        sys.path.insert(0, str(EVAL))
        if representation == "decoder":
            from v8_decoder_lba import read_manifest
        else:
            from v8_lba import read_manifest
        counts = dict(train=3507, val=466, test=490)
        audit_path = cache_root / "audit_recheck.json"
        if not audit_path.exists():
            audit_path = cache_root / "audit.json"
        audit = json.loads(audit_path.read_text())
        if audit.get("passed") is not True or len(audit.get("splits", [])) != 3 or {
            s["split"]: s["count"] for s in audit["splits"]
        } != counts:
            raise ValueError("Array initialization requires a successful complete cache audit")
        for mode in ("real", "zero"):
            for split, count in counts.items():
                manifest = read_manifest(cache_root / f"charge_{mode}" / split)
                if (len(manifest["entries"]) != count or
                    manifest["contract"]["charge_mode"] != mode or
                    manifest["contract"]["split"] != split):
                    raise ValueError("Wrong cache coverage/mode/split")
    manifest_paths = {f"{mode}/{split}": cache_root / f"charge_{mode}" / split / "_manifest.json"
                      for mode in ("real", "zero") for split in ("train", "val", "test")}
    manifests = {key: json.loads(path.read_text())["contract"] for key, path in manifest_paths.items()}
    if representation == "decoder":
        if (audit.get("representation") != "decoder" or
            audit.get("cache_root") != str(cache_root) or
            audit.get("manifest_sha256") != {key: sha(path) for key, path in manifest_paths.items()}):
            raise ValueError("Decoder audit is stale or belongs to another cache")
        if len({c["checkpoint_sha256"] for c in manifests.values()}) != 1:
            raise ValueError("Decoder cache arms/splits disagree on checkpoint")
    import torch
    info = dict(representation=representation, phase=phase, epochs=1 if phase == "rehearsal" else 50,
                seeds=[42] if phase == "rehearsal" else [42, 123, 7], arms=list(ARMS),
                batch=8, lr=1e-4, num_workers=4, train_time=0, val_time=0,
                model_root=str(Path(model_root).resolve()), cache_root=str(cache_root),
                gvp=git_info(ROOT), estats=git_info(ESTATS), cache_contracts=manifests,
                cache_manifest_sha256={key: sha(path) for key, path in manifest_paths.items()},
                source_sha256={name: sha(ROOT / name) for name in
                               ("run_atom3d.py", "gvp/atom3d.py", "gvp/__init__.py",
                                "submit_lba_v8_comparison.sh", "summarize_lba_v8.py")},
                python=sys.executable, torch=torch.__version__,
                environment=subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"], text=True))
    if array:
        info["execution"] = "slurm_array_v1"
        info["audit"] = dict(path=str(audit_path), sha256=sha(audit_path), result=audit)
        info["source_sha256"]["submit_lba_v8_array.sh"] = sha(ROOT / "submit_lba_v8_array.sh")
    if representation == "decoder":
        info["estats_source_sha256"] = {name: sha(EVAL / name) for name in
                                        ("v8_lba.py", "v8_decoder_lba.py")}
    info["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    info["slurm_node"] = os.environ.get("SLURMD_NODENAME")
    # Exclusive creation also protects simultaneous initialization attempts.
    with path.open("x") as handle:
        handle.write(json.dumps(info, indent=2) + "\n")


def array_config(log_root):
    """Cheap immutable-input checks; no repeated LMDB/PQR audit or GPU work."""
    info = json.loads((Path(log_root) / "run.json").read_text())
    if info.get("execution") != "slurm_array_v1":
        raise ValueError("Initialize this run with --array first")
    info.setdefault("representation", "encoder")
    if info["representation"] not in ("encoder", "decoder"):
        raise ValueError("Unknown V8 representation")
    phase = info["phase"]
    if (phase not in ("rehearsal", "production") or info["arms"] != list(ARMS) or
        info["seeds"] != ([42] if phase == "rehearsal" else [42, 123, 7]) or
        info["epochs"] != (1 if phase == "rehearsal" else 50) or
        any(info[k] != v for k, v in dict(batch=8, lr=1e-4, num_workers=4,
                                         train_time=0, val_time=0).items())):
        raise ValueError("Array protocol differs from the agreed experiment")
    for name, digest in info["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Source changed after initialization: {name}")
    for name, digest in info.get("estats_source_sha256", {}).items():
        if sha(EVAL / name) != digest:
            raise ValueError(f"Cache validation source changed after initialization: {name}")
    for name, digest in info["cache_manifest_sha256"].items():
        mode, split = name.split("/")
        if sha(Path(info["cache_root"]) / f"charge_{mode}" / split / "_manifest.json") != digest:
            raise ValueError(f"Cache manifest changed: {name}")
    if info["representation"] == "decoder":
        sys.path.insert(0, str(EVAL))
        from v8_decoder_lba import PROTOCOL, QUERY_PROTOCOL, QUERY_CHUNK_SIZE
        for contract in info["cache_contracts"].values():
            if (contract.get("representation"), contract.get("input_protocol"),
                contract.get("query_protocol"), contract.get("query_chunk_size")) != (
                    "decoder", PROTOCOL, QUERY_PROTOCOL, QUERY_CHUNK_SIZE):
                raise ValueError("Run metadata is not a decoder cache contract")
    for key in ("cache_root", "model_root"):
        if "\n" in info[key]:
            raise ValueError("Paths containing newlines are unsupported")
    return info


def publish_array_task(log_root, arm, seed, attempt_log, attempt_model, record):
    """Called under the shell's task lock; publish immutable links, receipt last."""
    root = Path(log_root)
    info = array_config(root)
    if arm not in info["arms"] or seed not in info["seeds"]:
        raise ValueError("Unexpected array arm/seed")
    attempt_log, attempt_model = Path(attempt_log), Path(attempt_model)
    parse_log(attempt_log / "run.log", info["epochs"])
    files = {
        Path(info["model_root"]) / arm / f"LBA_seed{seed}_{kind}.pt":
        attempt_model / f"LBA_seed{seed}_{kind}.pt" for kind in ("best", "last")
    }
    files[root / arm / f"seed{seed}.log"] = attempt_log / "run.log"
    for destination, source in files.items():
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"Missing/empty task output: {source}")
        if os.path.lexists(destination):
            raise ValueError(f"Refusing to replace published output: {destination}")
    receipt = root / arm / f"seed{seed}.complete.json"
    if receipt.exists():
        raise ValueError("Task already completed")
    record.update(arm=arm, seed=seed, epochs=info["epochs"],
                  run_sha256=sha(root / "run.json"), completed_at=time.time(),
                  checkpoint_sha256=sha(attempt_model / f"LBA_seed{seed}_best.pt"),
                  log_sha256=sha(attempt_log / "run.log"))
    for destination, source in files.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)  # exclusive, atomic; attempts and outputs share a filesystem
    temporary = attempt_log / "complete.json"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.link(temporary, receipt)


def parse_log(path, epochs):
    text = Path(path).read_text()
    for failure in ("Skipped batch due to OOM", "Traceback (most recent call last)", "CUDA out of memory"):
        if failure in text:
            raise ValueError(f"{path}: failed/skipped batch: {failure}")
    if text.count("\nTRAIN_OK\n") != 1 or text.count("\nTEST_OK\n") != 1:
        raise ValueError(f"{path}: missing or duplicate successful train/test markers")
    for stage in ("TRAIN", "VAL"):
        actual = [int(x) for x in re.findall(rf"EPOCH (\d+) {stage}\s+loss:", text)]
        if actual != list(range(epochs)):
            raise ValueError(f"{path}: incomplete {stage} epochs: {actual}")
    if text.count("=== LBA Test Metrics ===") != 1:
        raise ValueError(f"{path}: expected exactly one test evaluation")
    block = text.split("=== LBA Test Metrics ===")[1]
    result = {}
    for metric in METRICS:
        matches = re.findall(rf"^\s*{metric}:\s*([-+\d.eE]+)\s*$", block, re.MULTILINE)
        if len(matches) != 1 or not np.isfinite(float(matches[0])):
            raise ValueError(f"{path}: missing/invalid {metric}")
        result[metric] = float(matches[0])
    return result


def summarize(log_root):
    root = Path(log_root)
    info = json.loads((root / "run.json").read_text())
    phase = info["phase"]
    expected_seeds = [42] if phase == "rehearsal" else [42, 123, 7]
    expected_epochs = 1 if phase == "rehearsal" else 50
    if phase not in ("rehearsal", "production") or info["seeds"] != expected_seeds or info["epochs"] != expected_epochs or info["arms"] != list(ARMS):
        raise ValueError("Run protocol differs from the agreed experiment")
    rows = []
    for arm in ARMS:
        for seed in expected_seeds:
            model = Path(info["model_root"]) / arm / f"LBA_seed{seed}_best.pt"
            if not model.is_file():
                raise ValueError(f"Missing best checkpoint: {model}")
            result = parse_log(root / arm / f"seed{seed}.log", expected_epochs)
            if info.get("execution") == "slurm_array_v1":
                receipt = json.loads((root / arm / f"seed{seed}.complete.json").read_text())
                if (receipt.get("arm") != arm or receipt.get("seed") != seed or
                    receipt.get("epochs") != expected_epochs or
                    receipt.get("run_sha256") != sha(root / "run.json") or
                    receipt.get("checkpoint_sha256") != sha(model) or
                    receipt.get("log_sha256") != sha(root / arm / f"seed{seed}.log")):
                    raise ValueError(f"Invalid array completion record: {arm}/{seed}")
            rows.append(dict(representation=info.get("representation", "encoder"), arm=arm, seed=seed, **result, checkpoint_sha256=sha(model)))
    summary, paired = {}, {}
    for arm in ARMS:
        summary[arm] = {metric: dict(mean=float(np.mean([r[metric] for r in rows if r["arm"] == arm])),
                                     std=float(np.std([r[metric] for r in rows if r["arm"] == arm], ddof=0)))
                        for metric in METRICS}
    for first, second in (("v8_real", "v8_zero"), ("v8_real", "baseline"), ("v8_zero", "baseline")):
        paired[f"{first} - {second}"] = {}
        for metric in METRICS:
            differences = [next(r[metric] for r in rows if r["arm"] == first and r["seed"] == seed) -
                           next(r[metric] for r in rows if r["arm"] == second and r["seed"] == seed)
                           for seed in expected_seeds]
            paired[f"{first} - {second}"][metric] = dict(per_seed=differences, mean=float(np.mean(differences)),
                                                        std=float(np.std(differences, ddof=0)))
    with (root / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "metrics.json").write_text(json.dumps(dict(representation=info.get("representation", "encoder"), phase=phase, rows=rows, summary=summary, paired=paired), indent=2) + "\n")
    lines = [f"# GVP + V8 {info.get('representation', 'encoder')} LBA: {phase}", "", "Real versus zero changes both charges and radii. "
             "Metrics retain the runner log's four-decimal precision. Standard deviations use ddof=0.", "",
             "| Arm | Seed | RMSE | Pearson | Spearman | R² |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['arm']} | {r['seed']} | " + " | ".join(f"{r[m]:.4f}" for m in METRICS) + " |" for r in rows]
    lines += ["", "| Arm | RMSE | Pearson | Spearman | R² |", "|---|---|---|---|---|"]
    lines += [f"| {arm} | " + " | ".join(f"{summary[arm][m]['mean']:.4f} ± {summary[arm][m]['std']:.4f}" for m in METRICS) + " |" for arm in ARMS]
    lines += ["", "| Paired difference | RMSE | Pearson | Spearman | R² |", "|---|---|---|---|---|"]
    lines += [f"| {contrast} | " + " | ".join(f"{values[m]['mean']:+.4f} ± {values[m]['std']:.4f}" for m in METRICS) + " |" for contrast, values in paired.items()]
    lines += ["", "Three seeds provide limited uncertainty estimates; no significance claim is inferred. "
              "Historical V3 results are not mixed into these fresh runs."]
    (root / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    print(f"{phase.upper()} COMPLETE: {len(rows)} successful evaluations; {root / 'COMPARISON.md'}")
    return rows, summary, paired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-root", required=True)
    p.add_argument("--initialize", action="store_true")
    p.add_argument("--array", action="store_true", help="Initialize provenance for the shell array launcher")
    p.add_argument("--model-root")
    p.add_argument("--cache-root", default=str(EVAL / "v8_lba_precomputed_cache"))
    p.add_argument("--representation", choices=("encoder", "decoder"), default="encoder")
    p.add_argument("--phase", choices=("rehearsal", "production"), default="production")
    args = p.parse_args()
    if args.initialize:
        if not args.model_root:
            p.error("--initialize requires --model-root")
        initialize(args.log_root, args.model_root, args.cache_root, args.phase, array=args.array, representation=args.representation)
    else:
        summarize(args.log_root)


if __name__ == "__main__":
    main()
