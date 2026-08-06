# H100 80 GB Linux handoff

This is a train-only, local-logging run. Do not add validation or test files, change a
configuration, tune from the OOF result, or enable W&B.

## Payload

- Data: 3D CT volumes and voxelwise pancreas/lesion masks in compressed NIfTI
  (`.nii.gz`). The subtype label is the `subtype0`, `subtype1`, or `subtype2` training
  directory.
- Cases: 252 (`62/106/84` by subtype).
- Files: 252 CTs, 252 masks, `dataset.json`, `plans.json`, and one train-derived
  ResEnc-M checkpoint.
- Exact uncompressed payload: 507 files and 1,112,850,313 bytes (about 1.04 GiB).
- Validation/test/external data: none.

The masks are included only so a later segmentation-guard fine-tune can run without a
second transfer. The current V6 case-feature extractor does not open them.

## Run

Requirements: Linux x86-64, H100 80 GB, NVIDIA driver compatible with CUDA 12.8,
Python 3.12, about 15 GiB free disk, and an uninterrupted job window.

From the extracted code directory:

```bash
bash scripts/run_h100_v6.sh /absolute/path/input_bundle /absolute/path/h100_output
```

The launcher creates an isolated `.venv-h100`, verifies every private input hash, runs
the focused tests, extracts all 252 cases, runs the locked 15-fold nested train-only
selection, fits the three final seeds, and writes local JSON/NPZ/PTH evidence. It
refuses an existing output directory to avoid mixing attempts.

Planning estimate, not a promised H100 benchmark:

- H100 80 GB: approximately 0.2–0.8 GPU-hours total; setup/download time is separate.
- RTX 4060 Laptop 8 GB: approximately 0.8–1.5 GPU-hours total.

The 4060 estimate is anchored to a real locked-case smoke: 3.26 seconds for one
whole-volume extraction at 2.25 GB peak allocation, plus 2.25 seconds for five
case-head epochs at 0.44 GB peak allocation. The H100 estimate is an extrapolation;
the small case head may not saturate that GPU.

Use the measured values in `RESULTS_SUMMARY.json`; they supersede these estimates.

## What to inspect

```text
h100_output/
  run.log
  RESULTS_SUMMARY.json
  features/extraction_audit.json
  features/train_v6_features.npz
  training/training_audit.json
  training/v6_train_only_predictions.npz
  training/v6_final_case_head_bundle.pth
```

Primary training-only evidence is in `training_audit.json`:

- mean macro-F1 across the three complete repeated OOF predictions;
- minimum repeat macro-F1;
- minimum per-class recall;
- every fold/candidate/epoch decision;
- resubstitution-to-OOF gap;
- elapsed seconds, GPU-hours, peak CUDA memory, and output hashes.

The prospective gate is mean OOF macro-F1 >= 0.60, every repeat >= 0.57, and every
repeat/class recall >= 0.50. This gate is not the final validation score.

## Return

Return the complete `h100_output` directory without renaming or deleting files. If the
job fails, return the partial directory and `run.log`; do not modify the code or invent
a retry. An evaluating agent should first read `RESULTS_SUMMARY.json`, then verify the
two audit JSON files and the hashes of the final PTH/NPZ artifacts.
