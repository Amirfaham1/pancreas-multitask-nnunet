# Experiment worklog

This log separates measured evidence from plans. `PENDING` means the artifact does not yet exist; it is never a guessed value.

## Fixed constraints and decisions

| ID | Decision | Evidence/rationale |
|---|---|---|
| D-001 | Preserve the supplied 252-case training and 36-case validation split. | The assessment prohibits validation cases in optimization; `splits_final.json` contains one explicit fold with no overlap. |
| D-002 | Use nnU-Net v2.8.1 3D ResEnc M. | Mandatory assessment architecture; generated `nnUNetResEncUNetMPlans.json` identifies `nnUNetPlannerResEncM`. |
| D-003 | Use no external data or pretrained weights. | Mandatory fairness constraint; training starts from the nnU-Net plan's random initialization. |
| D-004 | Repair only verified near-integer mask values in a separate copy. | 214/288 masks contained value `1.0000152587890625`; maximum repair distance was `1.52587890625e-05`, below the declared `1e-3` tolerance. Source files remain untouched. |
| D-005 | Use hybrid global-average and learned-query cross-attention pooling. | Global average supplies a stable whole-patch summary; attention can focus on discriminative bottleneck tokens without replacing that summary. |
| D-006 | Weight subtype cross-entropy by inverse training frequency. | Counts 62/106/84 give weights 1.35484/0.79245/1.0. Label smoothing is 0.05; no-lesion crops retain a 0.25 subtype-loss weight. |
| D-007 | Train for 200×125 updates, validate for 30 iterations/epoch. | The 25,000-update schedule is deadline-feasible after the real CUDA benchmark and materially stronger than the initial 10,000-update plan. |
| D-008 | Record W&B offline, then sync. | Avoids an authentication/network failure stopping the overnight run. |
| D-009 | Compare final, native segmentation-best, and multitask-best checkpoints on all 36 cases. | Online subtype validation samples random patches and incomplete case coverage; it is unsuitable as sole final-selection evidence. |
| D-010 | Do not attempt a TPU port. | nnU-Net's tested path is PyTorch/CUDA; a deadline-night PyTorch/XLA port would add compatibility risk without strengthening the required core submission. |
| D-011 | Prepare one train-metric-triggered frozen-head rescue from `checkpoint_final.pth`. | After observing collapsed online patch classification, but before any full-volume validation evaluation, freeze a single 30 x 125 AdamW schedule. Activation uses only the predeclared epoch-40/50 training CE/accuracy rule; validation cannot activate or alter it. |

## 2026-08-05 — Source audit and conversion

Status: **DONE**

- Verified 252 training, 36 validation, and 72 test cases.
- Verified training subtype counts `62 / 106 / 84` and validation counts `9 / 15 / 12`.
- Verified 288 image/mask pairs, 72 test geometries, and no split overlap.
- Raw mask values: `0`, `1`, `1.0000152587890625`, `2`.
- Corrected output labels: exact `uint8 {0,1,2}`.
- Corrected voxel counts: background `498,469,423`; pancreas `26,396,648`; lesion `7,917,460`.
- Train-only fingerprint extraction produced CT statistics without validation leakage.
- Full preprocessing then covered all 288 labeled cases while `splits_final.json` retained the supplied 252/36 partition.

Primary reproducible commands are in `README.md`; the verified sequence was:

1. prepare with `--validation-layout separate --dry-run`, then repeat without
   `--dry-run`;
2. run `nnUNetv2_plan_and_preprocess -d 501
   --verify_dataset_integrity --no_pp -pl nnUNetPlannerResEncM -c 3d_fullres
   -npfp 2`;
3. prepare again with `--validation-layout imagesTr`;
4. run `nnUNetv2_preprocess -d 501 -plans_name
   nnUNetResEncUNetMPlans -c 3d_fullres -np 2`; and
5. copy only `splits_final.json` into the matching preprocessed dataset
   directory; classification metadata remains under the raw dataset root.

The machine-readable evidence is `data_audit.json`, `split_manifest.json`,
`classification_manifest.json`, and `classification_labels.json` in the
untracked nnU-Net dataset.

## 2026-08-05 — Environment and implementation

Status: **DONE**

- Python 3.12.13.
- PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128.
- nnU-Net v2.8.1; dynamic-network-architectures 0.4.4.
- CUDA tensor allocation verified on NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB).
- Custom external trainer discovery verified through `nnUNet_extTrainer=<repository>/src`.
- Plan: batch 2; patch `64×128×192`; spacing `2.0×0.73046875×0.73046875` mm; six-stage ResidualEncoderUNet with features `32/64/128/256/320/320`.
- Implemented shared-encoder segmentation/classification model, joint trainer, W&B metrics, checkpoint logic, deterministic evaluator, strict package validator, and explicit joint inference across mirrors/tiles/folds.

## 2026-08-05 — Verification

Status: **DONE**

- Historical pre-launch gate: **46 passed** in the provisioned environment.
- Current expanded repository suite: **92 passed**.
- The delivery packager is parser-tested and uses a validate-first atomic ZIP
  replacement; an existing valid archive remains untouched if staged ZIP
  validation fails.
- CLI smoke: one real GPU training/validation update completed without OOM.
- Stock segmentation inference completed over all 36 held-out volumes, confirming the wrapper preserves nnU-Net's default tensor return contract.
- Guarded timing probe: five training plus two validation updates; epoch compute 6.13 s; total process 26.18 s including initialization and cleanup.
- Peak CUDA memory: 6,159 MiB allocated / 6,716 MiB reserved.
- Optional network-architecture rendering reported missing `hiddenlayer`; training and inference were unaffected.
- Smoke outputs were moved intact to `D:\MLQuizWork\smoke_results\nnUNetTrainerPancreasMultiTask_smoke_20260805_1918` before production training.

## 2026-08-05 — Production run

Status: **RUNNING**

| Field | Value |
|---|---|
| Start | 2026-08-05 19:25 America/Toronto |
| Dataset/configuration/fold | `501 / 3d_fullres / 0` |
| Trainer/plans | `nnUNetTrainerPancreasMultiTask / nnUNetResEncUNetMPlans` |
| Schedule | 200 epochs; 125 train + 30 validation updates/epoch |
| Optimizer | SGD, Nesterov 0.99, weight decay `3e-5` |
| Learning rate | 0.01, polynomial decay |
| Segmentation loss | native nnU-Net Dice + cross-entropy with deep supervision |
| Classification loss | inverse-frequency weighted CE; smoothing 0.05 |
| Task weights | segmentation 1.0; classification 0.5 |
| Sampling | 50% foreground oversampling; no-lesion subtype patch weight 0.25 |
| Precision | CUDA automatic mixed precision |
| Gradient control | global norm clip at 12; no accumulation |
| W&B | offline run; public URL PENDING sync/login |

Launch command, executed after a same-process execution-policy bypass and
dot-sourcing `Set-QuizEnvironment.ps1`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
. .\scripts\Set-QuizEnvironment.ps1 `
  -WorkRoot D:\MLQuizWork -WandbMode offline -DataAugmentationProcesses 1

& 'D:\MLQuizWork\.venv\Scripts\nnUNetv2_train.exe' `
  501 3d_fullres 0 `
  -tr nnUNetTrainerPancreasMultiTask `
  -p nnUNetResEncUNetMPlans -device cuda
```

The initial foreground process was cut off at the orchestration tool's
20-minute execution boundary after epoch 7. Both best checkpoints were already
complete. An attempted detached resume was followed by a duplicate launch,
which produced CUDA out-of-memory and Windows shared-mapping error 1455 before
either copy completed epoch 8. The exact duplicate process tree was stopped;
no completed epoch or saved best checkpoint was lost. For recovery,
`checkpoint_best_multitask.pth` was copied to an independent, hash-verified
`checkpoint_latest.pth`, and the run resumed at epoch 8 at 19:53. A new
`Start-ProductionTraining.ps1` launcher now refuses to start whenever a matching
trainer process exists; a deliberate repeat-launch check left the process count
unchanged. W&B retained the original run ID across separate offline segments,
which will be synchronized after training.

Guarded resume command:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\Start-ProductionTraining.ps1 -Resume -WandbMode offline
```

Early stability evidence:

| Epoch | Train total loss | Validation total loss | Online foreground pseudo-Dice `(label1,label2)` |
|---:|---:|---:|---:|
| 0 | 0.9485 | 0.6793 | 0.0000 / 0.0000 |
| 1 | 0.6915 | 0.6518 | 0.0000 / 0.0000 |
| 2 | 0.6322 | 0.5980 | 0.0000 / 0.0000 |
| 3 | 0.5727 | 0.5524 | 0.0013 / 0.0000 |
| 4 | 0.5327 | 0.4906 | 0.3907 / 0.0000 |
| 5 | 0.5009 | 0.4569 | 0.4015 / 0.0000 |

These are random-patch monitoring values, not final case-level validation metrics.

The classification path was confirmed to be registered in the optimizer and
to receive nonzero updates; this was not a frozen-parameter bug. Through epoch
30, its latest ten training epochs had mean classification CE `1.1303`, mean
patch accuracy `0.3140`, and CE slope `-0.000682` per epoch, while online patch
predictions remained single-class collapsed. Online validation diagnostics had
already been observed, so the contingency is described precisely as frozen
before full-volume validation evaluation, not before all validation
observation. `configs/experiment.yaml` and
`docs/classification_head_rescue.md` predeclare a train-only epoch-40/50 gate,
fixed `checkpoint_final.pth` source, one 30 x 125 AdamW attempt, frozen
encoder/decoder, and no validation batches, stopping, or schedule changes.

## 2026-08-06 — Fixed validation

Status: **PENDING training completion**

Required evidence:

- evaluate `checkpoint_final.pth`, `checkpoint_best.pth`, and `checkpoint_best_multitask.pth` with identical full-volume settings, plus the predeclared classification rescue only if its train-only gate activates;
- save restored masks and subtype probabilities for every validation case;
- calculate mean/std/bootstrap confidence intervals for whole-pancreas and lesion Dice;
- calculate fixed-class macro-F1, per-class precision/recall/F1, accuracy, and confusion matrix;
- select the checkpoint by declared validation evidence only;
- record checkpoint SHA-256, case CSV, aggregate JSON, figures, qualitative cases, and inference runtime.

## 2026-08-06 — Test inference and packaging

Status: **PENDING checkpoint selection**

Completion criteria:

- 72 readable integer NIfTI masks with labels limited to `{0,1,2}`;
- mask names and geometry match the supplied test images;
- `subtype_results.csv` has exactly `Names,Subtype`, 72 unique `.nii.gz` names, and values in `{0,1,2}`;
- flat ZIP root with no extra files;
- directory and cleanly extracted ZIP both pass `scripts/validate_submission.py`;
- final ZIP and report hashes recorded.

## AI collaboration record

The assessment explicitly asks for more than 50% AI-generated code. OpenAI Codex produced a substantial majority of the initial code and documentation and supported debugging, auditing, monitoring, and packaging. Amirfaham Fallahpour set the objective and quality bar, supplied access and compute, chose the undergrad-first/deadline-safe scope, reviewed decisions, and owns the final submission. Attribution is functional rather than invented line ownership. Generated work is accepted only after tests or artifact-based verification; no result is entered from expectation or visual guesswork.
