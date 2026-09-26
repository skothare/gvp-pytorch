# Fresh-start reproduction of GVP–ElectroProt panels a–c

This package reconstructs the reported experiments from **two V9 checkpoints and the original ATOM3D LBA-30 data**. It creates PQR, both encoder/decoder caches, controls, training runs, ridge models and reports in a new external output root. Historical run directories, job IDs, PQR and embedding caches are not inputs. Archived identified predictions are included only for an independent reporting check.

Use the matching `benchmarks/atom3d_lba` package in Estats. Both branches are `repro/lba-paper`; `RELEASE_PAIR.md` records how to identify the compatible commits. The source addition is isolated from main's evaluation system. GVP's `run_atom3d.py` is unchanged. The only core GVP addition is the existing configurable cache transform and fusion model in `gvp/atom3d.py`; the LBA baseline remains unchanged.

## What is run

| Panel | Models | Seeds | Production fits |
|---|---|---|---:|
| a | GVP, GVP + EP-5, GVP + EP-20 | 42, 123, 7, 2026, 17 | 15 |
| b | GVP, GVP + EP-20, GVP + charges/radii, GVP + random EP | preceding five plus 29, 53, 101, 211, 307 | 40 |
| c | frozen EP-5 + ridge; frozen EP-20 + ridge | deterministic | 2 selected ridge models |

All 55 GVP fits use 50 epochs, batch size 8, Adam at 1e-4, four data workers, no time truncation, and best validation-MSE checkpoint selection. Ridge performs seven training-only candidate fits for each checkpoint and selects alpha on validation MSE. Panel c reuses panel a's GVP reference, not a third baseline cohort. No shuffle or fingerprint experiment is included.

**Panel a is a controlled execution protocol:** exact NVIDIA L40 hardware check, deterministic algorithms, deterministic CSR graph pooling, shared backbone initialization, identical fusion initialization and explicit matched batch schedules across arms. The three arms for one seed run sequentially in one allocation. Diagnostics first check repeated executions, initialization and predictions; failure stops production.

**Panel b retains the native historical execution protocol:** separate training/test processes, original RNG lifecycle, loaders and scatter pooling. It seeds Torch and NumPy as before, but does not promise deterministic GPU reductions or equal results on different GPUs. It is a separate cohort even when seed integers coincide with panel a. The random-EP arm uses one frozen EP-20-architecture initialization, seed 1729; downstream GVP still has ten seeds. Charges/radii are standardized with training-pocket atom statistics only. The baseline has 9 learned element scalars, direct-input fusion has 11, and EP fusion has 265; ligand feature additions are zeros. GVP graph edges still encode protein–ligand geometry.

Checkpoint metadata differs beyond epoch count. EP-5 versus EP-20 is a comparison of these complete supplied checkpoints, not an isolated causal study of training duration. New training results are new experimental realizations; they must not be relabeled as the archived paper numbers.

## Inputs and environments

1. Obtain the ATOM3D **LBA sequence-identity 30% split** from the ATOM3D dataset distribution (https://www.atom3d.ai/; upstream project https://github.com/drorlab/atom3d). Use its original `train/data.mdb`, `val/data.mdb`, `test/data.mdb` files under one root. Expected counts are 3,507 / 466 / 490. `COHORT.json` contains the expected ordered IDs and file hashes. Repacked or different datasets are rejected; do not substitute the 60% split or rename another dataset into this layout.
2. Obtain `V9_5_epoch_dpss.pt` and `V9_20_epoch_dpss.pt` from the ElectroProt authors' checkpoint distribution. These large weights are not in Git and no public download URL is asserted by this release. SHA-256:
   - EP-5: `eee0825fa6646bc2b64bd3352d3f4382f6a85cbe529364ecea62ed1e1e4da255`
   - EP-20: `950e4183f32ec959ac2b16b262806a9427015e8666925100c963f0aff4d17748`
3. Check out the compatible Estats/GVP pair. Use absolute paths in your private configuration. Relative paths resolve against the configuration file, not the shell working directory.
4. Create separate Linux environments from the recorded conda builds, then install the pip-only dependencies. This restores the measured software stack without modifying an existing environment:

```bash
conda create -n lba-estats --file /path/to/Estats/benchmarks/atom3d_lba/environment/conda-linux-64.txt
conda activate lba-estats
python -m pip install --no-deps -r /path/to/Estats/benchmarks/atom3d_lba/environment/pip-only.txt --extra-index-url https://download.pytorch.org/whl/cu121
conda create -n lba-gvp --file /path/to/gvp-pytorch/benchmarks/atom3d_lba/environment/conda-linux-64.txt
conda activate lba-gvp
python -m pip install --no-deps -r /path/to/gvp-pytorch/benchmarks/atom3d_lba/environment/pip-only.txt --extra-index-url https://download.pytorch.org/whl/cu121 --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

Python 3.10 / Torch 2.5.1+cu121 are recorded; extraction uses PDB2PQR 3.6.1, PDBFixer 1.12.0, OpenMM 8.2.0. GPU workers require a compatible NVIDIA driver. The master prepends the configured extraction environment's `bin` directory for PDB2PQR. Imports resolve GVP from the configured checkout, so no installed editable GVP package is required. `preflight` fails on missing required imports. Run the tests in the intended environments; an `estats` interpreter without PyG is not the GVP test environment.

## Commands from an empty output root

Copy `config.example.json` **outside either repository**, edit every path and the cluster-specific partition settings, then use:

```bash
GVP_ROOT=/path/to/gvp-pytorch
CONFIG=/path/to/private/lba-paper.json
ESTATS_PY=/path/to/envs/lba-estats/bin/python
GVP_PY=/path/to/envs/lba-gvp/bin/python
export PYTHONNOUSERSITE=1
export LBA_DRIVER_PYTHON="$ESTATS_PY"
bash "$GVP_ROOT/benchmarks/atom3d_lba/reproduce.sh" --config "$CONFIG" --stage preflight
bash "$GVP_ROOT/benchmarks/atom3d_lba/reproduce.sh" --config "$CONFIG" --stage all --dry-run
# CPU-only initialization hashes data, strictly loads both checkpoints and records targets/provenance.
bash "$GVP_ROOT/benchmarks/atom3d_lba/reproduce.sh" --config "$CONFIG" --stage initialize --execute
# Explicit user action: this is the only command below that submits jobs.
bash "$GVP_ROOT/benchmarks/atom3d_lba/reproduce.sh" --config "$CONFIG" --stage all --submit
```

Do not pre-create `output_root`; initialization requires a new directory. The `all --submit` command initializes automatically if needed. `all --dry-run` is stdlib-only, writes nothing and calls no scheduler. `preflight` only reads inputs/imports. Individual stages require `--execute`, and GPU stages must be run inside a GPU allocation. No production command automatically falls back to CPU.

## Stages, resources and outputs

The dependency chain is: PQR → PQR audit → EP-5 cache → audit → EP-20 cache → audit → control initialization → control caches → audit → diagnostics → panel a → panel b → ridge → report. Each scheduler stage depends on successful completion of the preceding array. The launcher requests one GPU per GPU task, four CPUs and 32 GB, excludes g021, and limits arrays to two concurrent tasks. Only diagnostics and panel a request the L40 constraint. The 12-hour default is an allocation request, not a measured guarantee; adapt it after rehearsal on the target cluster. PQR has 3 × 32 shards by default; preparation may be the slowest stage and parallelism remains capped at two in this release. Full diagnostics include four additional 50-epoch baseline runs, short native repeats and short fusion repeats, outside the 55 production fits.

```text
output_root/
  run.json, config.json, PLAN.json, test_targets.json, jobs.json, submissions.jsonl
  pqr/{train,val,test}/                         # PQR, failures, repair records, release manifests
  caches/{ep5,ep20}/{encoder,decoder}/charge_real/{train,val,test}/
  controls/{raw_qr,random_ep}/{train,val,test}/  # plus scaler and random initialization
  attempts/                                   # separate attempt for every task/retry
  completed/                                  # hash-bound successful fit receipts
  stages/                                     # successful stage receipts
  slurm/                                      # stage + array job/task IDs
  report/                                     # raw metrics, summaries, figures and audit
```

No tensor is silently trusted because its filename exists. Cache resume verifies checkpoint, representation, input order, PQR provenance and checksums; corrupt/unrecorded files fail. Writes are transactional and mode/split locked. Completed tasks refuse replacement. Historical cache schema is unchanged; the portable schema is version 2. Interrupted **release** caches can resume, but historical source-bound manifests must not be rewritten or silently reused.

## Monitor, retry and report

`jobs.json` and the append-only `submissions.jsonl` preserve IDs. Use `squeue -r -j JOBID` and inspect both `slurm/*.out` and `*.err`; an empty queue alone is not success. A failed scheduler submission leaves a journal and refuses a second blanket submission. Preserve it; resume only missing stages, using the prior job IDs and `afterok` dependencies. A failed array task can be retried from its recorded command in `submissions.jsonl` with `--array=FAILED_INDEX` and a new job ID; do not resubmit completed indices. If dependents were cancelled due to failure, resubmit their stages in dependency order after the retry succeeds. Never delete completed receipts to force an overwrite.

For an individual allocated task:

```bash
# Example: panel b index 13 = EP-20, seed 2026 (0–9 baseline, 10–19 EP-20,
# 20–29 charges/radii, 30–39 random EP).
"$GVP_PY" "$GVP_ROOT/benchmarks/atom3d_lba/pipeline.py" --config "$CONFIG"   --stage panel_b --index 13 --execute
# Audit/report manually only when all required completion receipts exist:
"$ESTATS_PY" "$GVP_ROOT/benchmarks/atom3d_lba/pipeline.py" --config "$CONFIG" --stage report --execute
```

Each retry trains from scratch in a new attempt directory; failed artifacts remain. Reporting verifies 55 GVP and two ridge completions, exact test IDs/targets, finite predictions, metric recomputation and artifact hashes. It produces per-seed `results.csv`, population-SD `aggregate.csv`, per-seed `paired_differences.csv`, learning curves, LaTeX `tables.tex`, and `panels_abc.pdf/png`. Error bars are seed SD, not confidence intervals. No significance claim is generated. The archived-reference command is separate:

```bash
"$ESTATS_PY" "$GVP_ROOT/benchmarks/atom3d_lba/pipeline.py" --config "$CONFIG" --stage reference-report
```

It rebuilds paper metrics and figures from the 57 small, checksummed prediction CSVs under `reference/`, without any dataset/cache/training dependency. It writes `output_root/historical_reference_report` and refuses to overwrite it. Fresh runs never read these prediction files.

## Required release validation

```bash
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LBA_ESTATS_ROOT=/path/to/Estats
"$ESTATS_PY" -m pytest -q "$LBA_ESTATS_ROOT/tests/atom3d_lba" -m 'not slow'
"$GVP_PY" -m pytest -q "$GVP_ROOT/tests/atom3d_lba" -m 'not reporting'
# Reporting and fresh-start integration use matplotlib in the extraction environment.
"$ESTATS_PY" -m pytest -q "$GVP_ROOT/tests/atom3d_lba/test_pipeline.py"
```

Run the real-input slow suite using the four `LBA_*` input variables documented in Estats. The fresh-start integration test creates tiny inputs, stubs external preparation/model/training/scheduler execution, and exercises the actual stage sequence through 57 verified evaluations and report generation. Separate tests exercise real checkpoints/PDB2PQR, graph gradients, failures/OOM, cache corruption, locks, and strict model selection. See `VALIDATION.md` for performed checks and the unresolved CPU-versus-archived-GPU decoder difference. Before a full production reproduction, users must validate extraction and diagnostic runs on their target GPU stack. This release does not claim GPU validation occurred during its CPU-only implementation.

## Attribution and limits

GVP-GNN is Dror Lab's implementation; its MIT `LICENSE`, README and references remain intact. Cite *Learning from Protein Structure with Geometric Vector Perceptrons* and *Equivariant Graph Neural Networks for 3D Macromolecular Structure*, plus ATOM3D and ElectroProt as appropriate. We add frozen features, controls and experiment orchestration; we do not claim authorship of GVP. Small evidence files reproduce reported scores, not independent validation or SOTA. Full-cohort fallback chemistry limitations are explicitly documented in the paired Estats README. The publication's checkpoint artifacts must be distributed separately before readers can execute a fresh run.
