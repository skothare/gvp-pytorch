# Isolated five-seed LBA comparison

This workflow does not edit previous source files, environments, jobs, caches or results. All GPU execution uses Slurm. Existing control jobs remain independent.

## Experiment

Seeds: 42, 123, 7, 2026, 17, selected as the first five historical seeds, not by outcomes. Arms: GVP baseline, GVP + frozen five-epoch V9 decoder, GVP + frozen twenty-epoch V9 decoder. There are 15 production trainings.

Use the full original LBA-30 cohort (3507/466/490), existing audited real-mode caches, 50 downstream epochs, Adam 1e-4, batch 8, four workers, no time truncation and best-validation checkpoint selection. Retain mean-minibatch validation MSE and shuffled training/validation; test has fixed order. No fallback exclusions, cache recomputation or ridge reruns.

One L40 GPU allocation runs all three arms for a seed sequentially. Rotate arm order by seed index. At most two seed tasks run concurrently; exclude g021. New jobs consume resources and may compete for scheduling capacity, but never modify existing jobs.

## New execution controls

Explicit Python/NumPy/PyTorch seeds; independent loader generators and worker seeding; reset training RNG after model initialization; strict deterministic PyTorch; CUBLAS_WORKSPACE_CONFIG=:4096:8; TF32 disabled. Shared backbone initialization and epoch batch schedules must match across arms.

Graph mean pooling uses deterministic CSR segment mean instead of atomic scatter mean. Nodes are already grouped by graph by PyG. There are no new parameters, gates, normalizations or architecture changes. CPU tests compare original and controlled predictions, gradients and checkpoint compatibility. PyG message aggregation remains under strict PyTorch settings; unsupported operations fail. This is a new execution protocol, not a bitwise reproduction of historical runs.

## Isolation and provenance

New files: unified_lba_runner.py, unified_lba_workflow.py, submit_lba_unified_gpu.sh, submit_lba_unified_cpu.sh, test_unified_lba.py. Initialization copies execution sources and gvp/{__init__,atom3d,data}.py into the new run's code/ directory. Jobs execute that private snapshot. Later live-source edits cannot change queued work.

Initialization checks LMDB hashes against the earlier EP20 record, both audit receipts, manifest hashes, identical feature protocols and ordered inputs. Per-task checks verify private source hashes, cache provenance and environment versions. Cache tensor checksums are validated when loaded. No packages are installed: GPU jobs use gvp; CPU plotting uses estats.

## Diagnostic gate

For baseline seeds 42 and 307, one GPU allocation performs two independent 16-minibatch native-reduction checks, followed by two independent full 50-epoch controlled runs. Native checks use controlled RNGs and instrumentation; they do not replay the complete historical environment. Then each fusion arm runs two short repeatability checks on seed 42.

Controlled repetitions must agree exactly on initial states, first eight batch graph/gradient/update hashes, epoch losses and orders, final/selected weights, predictions and metrics. Native differences are recorded but not a failure. Diagnostic repetitions are excluded from production statistics.

Production depends on successful diagnostics and also verifies DIAGNOSTICS_PASSED.json. Failures block production. Reporting depends on all five production tasks; it checks artifact hashes, recomputes predictions against canonical test labels, and writes report/results.csv, COMPARISON.md, RESULTS_AUDIT.json and metric plots.

## Initialize and submit a NEW experiment

Do not repeat these commands for an already queued run.

```bash
WORK_ROOT=/net/galaxy/home/koes/skothare
source "$WORK_ROOT/miniconda3/bin/activate" gvp
export PYTHONNOUSERSITE=1
cd "$WORK_ROOT/gvp-pytorch"
export RUN_ROOT="$PWD/logs/lba_unified_$(date -u +%Y%m%dT%H%M%SZ)"
CUDA_VISIBLE_DEVICES='' python unified_lba_workflow.py initialize --run-root "$RUN_ROOT"
python "$RUN_ROOT/code/unified_lba_workflow.py" submit --run-root "$RUN_ROOT"
cat "$RUN_ROOT/jobs.json"
squeue -u skothare
```

jobs.json records all three stage IDs. Slurm logs are directly in the run root. Detailed logs/models/predictions are under diagnostics/<job>/ and attempts/seed<seed>/<job>_r<restart>/<arm>/. Completion receipts are completed/seed<seed>.json. Never overwrite completed attempts.

If diagnostics fail, inspect the detailed .log and FAILED.json when present. Resolve the cause in new live source, test it, then initialize a fresh root. Do not change a queued snapshot or its metadata. Retry an incomplete production seed with its original snapshot and a new job ID; completed seeds refuse reruns. Inspect submissions.jsonl after any partial submission failure before submitting again.

To extend later, initialize a new root with --seeds 29 53 101 211 307 under the unchanged protocol. Verify compatibility before combining reports. Do not modify a running experiment's seed list.

## Interpretation

The checkpoint pretraining-membership/resume differences remain. This controls downstream execution and is not a pure pretraining-duration ablation. Report both checkpoints, distinguish controlled from historical results, and treat five-seed SDs as descriptive variability rather than confidence intervals.
