# Release validation (2026-09-26)

Status: implementation and CPU acceptance passed. No GPU or Slurm job was submitted by this release implementation. GPU acceptance and fresh production training remain user-run steps; this is not a claim that 55 new trainings have completed.

| Check | Outcome |
|---|---|
| Real extraction/PQR/unit suite in estats | 87 passed; includes separate strict EP-5/EP-20 loads and missing-weight rejection |
| GVP suite in gvp | 28 passed; two reporting cases selected separately in estats |
| Pipeline/report suite in estats | 20 passed (18 overlap the GVP suite) |
| Relocated source-only export | 82 extraction unit tests, 28 GVP tests, 20 pipeline tests passed; five real-input cases deliberately selected separately |
| Empty-output integration | Complete stage sequence, cache/audit/control/ridge/report functions; 55 stubbed production trainings + two fitted readouts; no historical logs |
| Scheduler/worker stubs | Stage dependencies, indices, explicit submission guard, partial submission journal, duplicates, failed train/test, missing best checkpoint, OOM rejection |
| Read-only real preflight | Both exact checkpoint hashes and required environment imports passed |
| Dry run / shell syntax | Complete 55-fit/two-ridge plan; no scheduler calls |
| Archived predictions | 57 evaluations × 490 identified test predictions validated; published panel means/SDs regenerated |
| Ridge refit on archived feature matrices | EP-5 alpha=100, RMSE=1.5311361454; EP-20 alpha=0.1, RMSE=1.6745109509; all metrics agree to 1e-10 |
| Real random-EP initialization | Seed 1729; 6,721,637 parameters; state digest matches archived control exactly |

Expected warnings: PyTorch's pre-norm/nested-tensor notice and PyG's torch-cluster deprecation. No required imports are skipped. The two reporting tests use the extraction environment because the measured GVP environment lacks matplotlib. The fresh-start test uses tiny fixtures/stubs and cannot establish production GPU correctness or runtime.

## Numerical compatibility and remaining GPU check

On validation complex `5d3l`, original research code and release code produce **bit-identical CPU encoder and decoder tensors for both checkpoints**, in separate processes. Namespaced model sources are retained; historical helper math and native training function AST hashes are preserved. Ridge retains float64 design matrices (the original scripts converted pooled float32 features before fitting); omission of that conversion was found and corrected during release validation.

Original CPU outputs differ from the archived GPU caches. Maximum encoder differences are approximately 4.9e-6 (EP-5) / 1.2e-5 (EP-20). Maximum decoder differences are 0.24018 / 0.19875, with mean absolute differences 0.005218 / 0.002162. These differences occur in the original code too; their cause has not been established. The CPU-to-archived-GPU decoder comparison **does not pass** the reference tolerance. We have not silently widened it or described it as a passed GPU equivalence test.

Estats tests include separate CPU and archived CUDA fixtures plus their hashes and input provenance. To perform the pending backend-matched GPU check, inside an allocated GPU environment with the four documented input variables:

```bash
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# Keep the scheduler-assigned CUDA_VISIBLE_DEVICES; do not set it to an empty string here.
LBA_TEST_DEVICE=cuda /path/to/envs/estats/bin/python -m pytest -q   /path/to/Estats/tests/atom3d_lba/test_real_checkpoints.py -m slow
```

The check explicitly requires CUDA and compares both representations to archived GPU tensors. It has not been executed during this session. If it fails, inspect numerical/backend and PQR input differences before claiming exact feature reproduction. The full controlled diagnostic gate must also pass before panel a begins. New publication tables should identify new runs and their environments; archived numbers and freshly trained results remain separate.

## Preservation and source scope

Research snapshots: Estats 94130bb5eb1c177c79de2d10c0f8e21f1c8bde91 and GVP 631deaab54f79d6dcef20bd5a61fcd27fd4eb9d1. Separate worktrees based on the intended main branches contain this release. Original branches, datasets, PQR/feature caches, checkpoints and results were not rewritten. Snapshot tags and verified Git bundles preserve tracked research code; the large-artifact inventory is in-place metadata, not an independent data backup.

The release contains only reported experiments and required helpers. No shuffle/fingerprint arm or old run metadata is needed. Small prediction evidence and numeric fixtures are intentionally tracked; large scientific artifacts are not. Exact compatible revisions are associated with the paired `repro/lba-paper-v1` tags and release receipt.
