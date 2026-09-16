"""Validate a finished LBA comparison and render CPU-only PNG/PDF figures.

Uses the logged four-decimal test metrics; does not rerun models or change
training provenance. Requires numpy and matplotlib in the plotting interpreter.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
with warnings.catch_warnings(record=True) as plotting_warnings:
    warnings.simplefilter("always")
    import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from summarize_lba_v8 import METRICS, sha, summarize

ORDER = ("baseline", "v8_zero", "v8_real")
LABELS = {"baseline": "GVP only", "v8_zero": "GVP + V8\nzero / unit", "v8_real": "GVP + V8\nreal charges / radii"}
COLORS = ("#536878", "#D98624", "#00897B")
TITLES = ("RMSE (lower is better)", "Pearson r (higher is better)",
          "Spearman ρ (higher is better)", "R² (higher is better)")


def audit_logs(root, rows):
    info = json.loads((root / "run.json").read_text())
    warnings = Counter()
    records, curves, errors = [], {}, []
    # Summary already checks all completion markers, epoch sequences, finite
    # test metrics, and completion-bound checkpoint/log/provenance hashes.
    for row in rows:
        arm, seed = row["arm"], row["seed"]
        path = root / arm / f"seed{seed}.log"
        text = path.read_text()
        for line in text.splitlines():
            if re.search(r"\b\w*Warning:", line):
                warnings[line] += 1
        values = {}
        for stage in ("TRAIN", "VAL"):
            matches = re.findall(rf"EPOCH (\d+) {stage}\s+loss:\s*(\S+)", text)
            if [int(e) for e, _ in matches] != list(range(info["epochs"])):
                raise ValueError(f"Incomplete {stage} history: {path}")
            values[stage] = np.array([float(v) for _, v in matches])
            if not np.isfinite(values[stage]).all():
                raise ValueError(f"Nonfinite loss in {path}")
        best = int(np.argmin(values["VAL"]))
        record = dict(arm=arm, seed=seed, best_epoch_zero_based=best,
                      best_logged_val_mse=float(values["VAL"][best]))
        if info.get("execution") == "slurm_array_v1":
            receipt = json.loads((root / arm / f"seed{seed}.complete.json").read_text())
            record.update({k: receipt[k] for k in ("array_job_id", "task_id", "job_id", "gpu", "node", "train_seconds", "test_seconds")})
            prefix = root.parent / f"v8_train_{receipt['array_job_id']}_{receipt['task_id']}"
            for suffix in (".out", ".err"):
                outer = Path(str(prefix) + suffix)
                if not outer.exists():
                    errors.append(f"Missing scheduler log: {outer}")
                    continue
                content = outer.read_text()
                if suffix == ".err" and content.strip():
                    errors.append(f"Nonempty scheduler stderr requires review: {outer}")
                if suffix == ".out":
                    marker = f"TASK_COMPLETE arm={arm} seed={seed}"
                    if marker not in content:
                        errors.append(f"Missing task completion in {outer}")
                    if re.search(r"Traceback \(most recent call last\)|Skipped batch due to OOM|CUDA out of memory|AF_UNIX path too long|DUE TO TIME LIMIT|CANCELLED|Segmentation fault", content):
                        errors.append(f"Error marker in {outer}")
        curves[arm, seed] = values
        records.append(record)
    source_checks = {name: sha(Path(__file__).parent / name) == digest
                     for name, digest in info.get("source_sha256", {}).items()}
    manifest_checks = {}
    for name, digest in info.get("cache_manifest_sha256", {}).items():
        mode, split = name.split("/")
        manifest_checks[name] = sha(Path(info["cache_root"]) / f"charge_{mode}" / split / "_manifest.json") == digest
    report = dict(passed=not errors, errors=errors, runs=records,
                  warning_counts=dict(warnings), source_hash_matches=source_checks,
                  cache_manifest_hash_matches=manifest_checks,
                  precision="Test metrics are limited to four decimals printed by the runner.",
                  scope="Artifact/log validation, not a live Slurm accounting query or a new chemical audit.")
    (root / "RESULTS_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Production result checks", "", f"Accepted evaluations: {len(rows)}",
             f"Scheduler log checks passed: {not errors}", "",
             "The existing summary validator checked epoch sequences, train/test markers, finite test metrics,",
             "best checkpoint availability, and completion-bound log/checkpoint/run hashes.",
             "Additional checks verified finite learning curves and scheduler output/completion markers.", "",
             "## Warnings in canonical per-seed logs", ""]
    lines += [f"- {count} occurrence(s): `{warning}`" for warning, count in warnings.items()]
    lines += ["", "## Provenance", "",
              f"Current source hashes match: {all(source_checks.values())}",
              f"Current cache manifest hashes match: {all(manifest_checks.values())}", "",
              "Test metrics have four-decimal precision. No inference or training was rerun.",
              "Three seeds do not establish statistical significance. Zero mode changes both charges and radii."]
    if errors:
        lines += ["", "## Requires inspection", ""] + [f"- {e}" for e in errors]
    (root / "RESULTS_AUDIT.md").write_text("\n".join(lines) + "\n")
    if errors:
        raise ValueError("; ".join(errors))
    return curves


def save(fig, directory, stem):
    fig.savefig(directory / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot(root):
    info = json.loads((root / "run.json").read_text())
    rows, summary, paired = summarize(root)
    curves = audit_logs(root, rows)
    seeds = list(dict.fromkeys(r["seed"] for r in rows))
    markers = ("o", "s", "^")
    values = {(r["arm"], r["seed"]): r for r in rows}
    directory = root / "plots"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, metric, title in zip(axes.flat, METRICS, TITLES):
        for j, seed in enumerate(seeds):
            y = [values[a, seed][metric] for a in ORDER]
            ax.plot(range(3), y, color="#BBBBBB", linewidth=0.8, zorder=1)
            for i, v in enumerate(y):
                ax.scatter(i, v, marker=markers[j], color=COLORS[i], s=55, zorder=3)
        for i, arm in enumerate(ORDER):
            ax.errorbar(i + .13, summary[arm][metric]["mean"],
                        yerr=summary[arm][metric]["std"], fmt="D", color="black", capsize=4, markersize=5)
        ax.set(xticks=range(3), xticklabels=[LABELS[a] for a in ORDER], title=title, xlim=(-.4, 2.5))
        ax.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], color="gray", marker=m, linestyle="none", label=f"Seed {s}") for s,m in zip(seeds,markers)]
    handles += [Line2D([], [], color="black", marker="D", linestyle="none", label="Mean ± population SD")]
    axes[0, 0].legend(handles=handles, fontsize=8)
    fig.suptitle(f"ATOM3D LBA-30 • held-out test results\n{info['epochs']} epochs; best-validation checkpoint; n = {len(seeds)} seeds", fontsize=14)
    save(fig, directory, "test_metrics")

    contrasts = [("v8_real", "v8_zero"), ("v8_real", "baseline"), ("v8_zero", "baseline")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, metric, title in zip(axes.flat, METRICS, TITLES):
        for i, (a,b) in enumerate(contrasts):
            data = paired[f"{a} - {b}"][metric]
            for j, delta in enumerate(data["per_seed"]):
                ax.scatter(i + (j-1)*.055, delta, color=COLORS[i], marker=markers[j], s=55)
            ax.errorbar(i+.18, data["mean"], yerr=data["std"], fmt="D", color="black", capsize=4, markersize=5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set(xticks=range(3), xticklabels=["Real − zero", "Real − baseline", "Zero − baseline"],
               title=title.replace("RMSE", "Δ RMSE").replace("Pearson", "Δ Pearson").replace("Spearman", "Δ Spearman").replace("R²", "Δ R²"), xlim=(-.4,2.5))
        ax.grid(axis="y", alpha=.2)
    axes[0,0].legend(handles=handles, fontsize=8)
    fig.suptitle("Paired test-metric differences by training seed\nNegative favors first arm for RMSE; positive favors first arm for correlations and R²", fontsize=12)
    save(fig, directory, "paired_differences")

    fig, axes = plt.subplots(3, len(seeds), figsize=(13, 10), squeeze=False, sharex=True, sharey=True, layout="constrained")
    for i, arm in enumerate(ORDER):
        for j, seed in enumerate(seeds):
            ax = axes[i,j]
            c = curves[arm,seed]
            epochs = np.arange(1, len(c["VAL"])+1)
            best = int(np.argmin(c["VAL"]))
            ax.plot(epochs,c["TRAIN"], color="#536878",label="Train")
            ax.plot(epochs,c["VAL"], color="#D98624",label="Validation")
            ax.scatter(best+1,c["VAL"][best], color="black",marker="*",s=70,label="Lowest logged val loss")
            ax.set(title=f"{LABELS[arm].replace(chr(10), ' ')} • seed {seed}")
            ax.grid(alpha=.2)
            if i == 2: ax.set_xlabel("Epoch (1-based)")
            if j == 0: ax.set_ylabel("Logged mean batch MSE")
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f"Training and validation histories • all {len(rows)} runs\nShared axes; checkpoint selection uses validation loss, not test metrics",fontsize=13)
    save(fig, directory, "learning_curves")
    provenance = dict(python=sys.executable, matplotlib=matplotlib.__version__, numpy=np.__version__,
                      script_sha256=sha(__file__), metrics_sha256=sha(root / 'metrics.json'),
                      warnings=[str(w.message) for w in plotting_warnings],
                      note="All figures are 2D; Axes3D is not used.",
                      outputs_sha256={p.name: sha(p) for p in directory.iterdir() if p.suffix in ('.pdf','.png')})
    (directory / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f"Plots and result audit saved under {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    plot(parser.parse_args().log_root.resolve())
