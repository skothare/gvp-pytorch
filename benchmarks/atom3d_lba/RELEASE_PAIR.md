# Compatible reproduction pair

Interface: `lba-paper-v1`; extraction cache schema: 2.

Use both repositories' `repro/lba-paper` branches at the paired release tag `repro/lba-paper-v1`. Resolve the tag in EACH repository with `git rev-parse repro/lba-paper-v1^{commit}` and record both returned hashes. Each fresh run records source-file hashes from both checkouts; moving or editing a running checkout is rejected.

The exact paired implementation revisions are recorded in the accompanying release receipt and PR descriptions. The immutable snapshot tag `snapshot/20260926-lba-paper` is the earlier research backup, not this portable release. Large data, PQR/embedding caches, downstream checkpoints and results are outside Git; small CPU fixtures and identified historical predictions are tracked explicitly.
