# Final submission guide

This repository is the public code and reproducibility component of Amirfaham
Fallahpour's pancreas multi-task take-home submission.

## Submit these files

The local `delivery/v7_final/` directory contains the assembled submission:

- `Amirfaham_Fallahpour_results.pdf` — final report;
- `Amirfaham_Fallahpour_results.zip` — 72 test masks and
  `subtype_results.csv` for the private assessment upload;
- `pancreas-multitask-v7-source.zip` — source snapshot;
- `pancreas-multitask-v7-evidence.zip` — aggregate JSON evidence and the fitted
  stage-1 classifier; and
- `SHA256SUMS.txt` — integrity hashes for every deliverable.

The results ZIP is intentionally not committed or attached to a public GitHub
release because it contains derived medical-image masks. Submit it only through
the assessment's private upload channel.

## Verified result

| Requirement | V7 result | Status |
|---|---:|---:|
| Whole-pancreas Dice >= 0.91 | 0.9201569021 | Pass |
| Lesion Dice >= 0.31 | 0.6196343545 | Pass |
| Three-class macro-F1 >= 0.70 | 0.7445103206 | Pass |
| At least 10% faster than stock | Not established by an eligible complete audit | Not met |

The report explains why the recovered H100 `+11.17%` timing is not used: its
candidate arm omitted required classifier work. The first three metrics were
independently recomputed from saved artifacts.

## Public-repository contents

The GitHub branch/release contains source, tests, configuration, report source,
aggregate evidence, the small fitted classifier, AI disclosure, and
reproduction instructions. It excludes source data, checkpoints, NIfTI files,
raw predictions, credentials, W&B local internals, temporary outputs, and
interview-preparation notes.

## Final checks completed

- Full repository suite: 463 passed.
- Final PDF build: 42 pages, final-source guards passed.
- V7 classifier SHA-256:
  `bbdb0fc79b35cfc81400550ad558636be6c15663f623b230813ddcb46264d0df`.
- Recovered result archive: 72 masks, 72 unique CSV rows, valid flat layout.
- W&B evidence: three real offline run IDs with explicit replay provenance and
  exact sync commands in `docs/evidence/v7/wandb_runs.json`.
