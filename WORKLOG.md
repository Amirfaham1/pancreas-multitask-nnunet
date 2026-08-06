# Experiment worklog

This log records measured evidence and the decisions that produced it. Values
below come from saved checkpoints, evaluator outputs, validator artifacts, or
the public experiment record; no expected value is presented as measured.

## Fixed constraints and decisions

| ID | Decision | Evidence/rationale |
|---|---|---|
| D-001 | Preserve the supplied 252-case training and 36-case validation split. | The assessment prohibits validation cases in optimization; `splits_final.json` contains one explicit fold with no overlap. |
| D-002 | Use nnU-Net v2.8.1 3D ResEnc M. | Mandatory assessment architecture; generated `nnUNetResEncUNetMPlans.json` identifies `nnUNetPlannerResEncM`. |
| D-003 | Use no external data or pretrained weights. | Mandatory fairness constraint; training starts from the nnU-Net plan's random initialization. |
| D-004 | Repair only verified near-integer mask values in a separate copy. | 214/288 masks contained value `1.0000152587890625`; maximum repair distance was `1.52587890625e-05`, below the declared `1e-3` tolerance. Source files remain untouched. |
| D-005 | Use hybrid global-average and learned-query cross-attention pooling. | Global average supplies a stable whole-patch summary; attention can focus on discriminative bottleneck tokens without replacing that summary. |
| D-006 | Weight subtype cross-entropy by inverse training frequency. | Counts 62/106/84 give weights 1.35484/0.79245/1.0. Label smoothing is 0.05; no-lesion crops retain a 0.25 subtype-loss weight. |
| D-007 | Train for 200 x 125 updates, validate for 30 iterations/epoch. | The 25,000-update schedule was deadline-feasible after the real CUDA benchmark and materially stronger than the initial 10,000-update plan. |
| D-008 | Record W&B offline, then sync. | Avoids an authentication/network failure stopping the overnight run. |
| D-009 | Compare final, native segmentation-best, multitask-best, and the activated rescue checkpoint on all 36 cases. | Online subtype validation samples random patches and incomplete case coverage; it is unsuitable as sole final-selection evidence. The train-only gate activated, so the fixed pass admitted four candidates. |
| D-010 | Do not attempt a TPU port. | nnU-Net's tested path is PyTorch/CUDA; a deadline-night PyTorch/XLA port would add compatibility risk without strengthening the required core submission. |
| D-011 | Prepare one train-metric-triggered frozen-head rescue from `checkpoint_final.pth`. | After observing collapsed online patch classification, but before any full-volume validation evaluation, freeze a single 30 x 125 AdamW schedule. Activation used only the predeclared epoch-40/50 training CE/accuracy rule; validation could not activate or alter it. |
| D-012 | Permit one disclosed zero-update numerical execution recovery, but only before custom joint fixed validation. | The first rescue process failed on batch 1 after the finite-loss guard and before `AdamW.step`; it made zero updates and wrote no checkpoint. Preserve/hash both logs, retain one update-bearing trajectory, keep every model/schedule choice fixed, move only the trainable classification path to FP32, count two process launches, and prohibit any further recovery. |

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
5. copy `splits_final.json` and its immutable `split_manifest.json` into the
   matching preprocessed dataset directory; classification metadata remains
   under the raw dataset root.

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
- Current expanded repository suite: **193 passed**; the four warnings are
  third-party `batchgenerators` deprecations.
- Ruff, all PowerShell parser checks, `pip check`, and the Git-diff check passed.
- Tested source and generated-result snapshot: commit
  `41ab3abc6227e6eb070958a22e61eb76dd5d2254`.
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

Status: **DONE**

| Field | Value |
|---|---|
| Start/end | 2026-08-05 19:25:31 EDT / 2026-08-06 02:13:34 EDT |
| Joint wall time | 6:48:03.248 |
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
| Final joint checkpoint | `checkpoint_final.pth`; SHA-256 `c3c33d067fd7a2832a7865edf73da1810c76d4ce47c07331fe65440708f7624f` |
| W&B | finished public run `hrs05iyx`; exactly 200 canonical history rows, steps 0--199 |

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
unchanged. W&B retained the original run ID across separate offline segments;
the segments were synchronized once after training into the canonical 200-step
public history.

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
fixed `checkpoint_final.pth` source, one 30 x 125 update-bearing AdamW trajectory, frozen
encoder/decoder, and no validation batches, stopping, or schedule changes.

Epoch 40 completed at 2026-08-05 20:59:46 EDT, after the rescue protocol and
fail-closed evaluation integration had been pushed publicly in commits
`859a70e` and `907235a`. Before any restored full-volume validation inference,
the predeclared training-only window for epochs 31--40 had mean classification
CE `1.11514384`, mean training-patch accuracy `0.3236`, and CE ordinary-least-
squares slope `-0.000249559` per epoch. These respectively pass the frozen
`>=1.05`, `<=0.42`, and `>=-0.001` conditions, so the epoch-40 gate is
affirmative. This snapshot was recomputed from the 41-entry logger in
`checkpoint_best.pth` at `current_epoch=41`; the rescue remained prohibited
until the joint run completed 200 epochs and the same decision was recomputed
from `checkpoint_final.pth` into a SHA-256-bound activation audit. No
full-volume validation output existed at this decision point.

The primary W&B run is the sole joint training run: it logs joint
training/validation losses and patch-level performance for both tasks. Rescue
history stays in its immutable audit JSON rather than being appended as
epochs 200--229, because the optimizer, trainable parameter scope, and metric
semantics differ and the rescue deliberately consumes zero validation batches.

The completed joint trainer then ran stock nnU-Net segmentation-only validation
during teardown (`Validation complete` at 2026-08-06 02:15:02 EDT). Its logged
mean foreground Dice `0.753518646` was observed in monitoring before recovery
authorization, but it did not drive the numerical repair, schedule, seed, or
recovery decision. The hash-bound activation audit was
created at 02:15:07 EDT from
training metrics only and approved the predeclared epoch-40 gate.

The first rescue process launch failed on its first training batch when
fail-fast clipping rejected a non-finite gradient norm after AMP unscaling.
The finite-loss check had passed, while `scaler.step`/`AdamW.step` occurs later,
so the failed process made zero optimizer updates, completed zero epochs, and
wrote no rescue checkpoint or rescue audit. The watcher stdout/stderr were
preserved under `classification_rescue_recovery_evidence/`; their SHA-256
digests and the source/activation/Git bindings are recorded in
`classification_rescue_zero_update_recovery.json`.

Before the custom joint fixed-validation candidate pass began, the execution
policy was amended once: the frozen encoder remains under CUDA autocast, but
the detached bottleneck and the trainable classification forward/loss/backward,
clipping, and AdamW update run in FP32 without GradScaler. Source checkpoint,
reset seed, training keys, augmentation, LR, weight decay, clip threshold, and
the 3,750-successful-update schedule remain fixed. Provenance counts two process
launches, one zero-update recovery, and one update-bearing trajectory; another
recovery is prohibited.

The repaired rescue then completed all 30 frozen-head epochs and exactly 3,750
successful AdamW updates with zero validation batches. Encoder and decoder
hashes were unchanged across the run. Its checkpoint SHA-256 is
`d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116`.
This is a distinct 30-epoch head-only trajectory following the 200-epoch joint
run, not a claim of 230 joint-training epochs.

## 2026-08-06 — Fixed validation

Status: **DONE**

All four admitted candidates were inferred and evaluated exactly once with the
same full-volume settings over all 36 held-out cases. The frozen equal-weight
selection rule produced:

| Rank | Candidate | Whole Dice | Lesion Dice | Macro-F1 | Selection score |
|---:|---|---:|---:|---:|---:|
| 1 | `checkpoint_classification_rescue` | 0.9201588643239327 | 0.6196727519510179 | 0.46399340516987575 | 0.6679416738149421 |
| 2 | `checkpoint_best_multitask` | 0.9161983302241753 | 0.6248261075531881 | 0.19607843137254902 | 0.5790342897166375 |
| 3 | `checkpoint_best` | 0.9201459341269391 | 0.620453032119171 | 0.16666666666666666 | 0.5690885443042589 |
| 4 | `checkpoint_final` | 0.9201588643239327 | 0.6196727519510179 | 0.13333333333333333 | 0.557721649869428 |

Selected checkpoint: `checkpoint_classification_rescue.pth`, SHA-256
`d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116`.
The selected fixed-validation evidence was:

- whole-pancreas Dice mean `0.9201588643239327`, sample SD
  `0.03578438909583714`, 95% bootstrap CI
  `[0.9078962203213721, 0.9308140253653576]`;
- lesion Dice mean `0.6196727519510179`, sample SD
  `0.3206719417556869`, 95% bootstrap CI
  `[0.5148320705748086, 0.7165657179330268]`;
- macro-F1 `0.46399340516987575`, 95% bootstrap CI
  `[0.2795513293036513, 0.6314441497200117]`;
- accuracy `0.5`, 95% bootstrap CI
  `[0.3333333333333333, 0.6666666666666666]`;
- macro precision `0.5148148148148147`, macro recall
  `0.47592592592592586`; and
- confusion matrix, rows reference and columns prediction:
  `[[4, 5, 0], [2, 11, 2], [3, 6, 3]]`.

Using unrounded point estimates, the undergraduate whole-pancreas Dice target
`>=0.90` and lesion Dice target `>=0.27` were met. The macro-F1 target `>=0.60`
was not met. This miss is retained explicitly. Restored masks, subtype
probabilities, case CSV, aggregate JSON, runtime evidence, deterministic plots,
and selected-checkpoint provenance were all generated.

The public W&B run is `finished` with exactly 200 canonical history rows at
steps 0--199, no missing or duplicate steps. A sanitized `full_volume/*`
summary was published after local dry-run validation. It records 36 cases, four
candidates, the selected checkpoint/hash, score, and aggregate metrics; no
case-level rows or local paths were uploaded.

## 2026-08-06 — Test inference and packaging

Status: **DONE**

- Fresh inference used the selected rescue checkpoint for all 72 test cases.
- Runtime: `248.11512969993055` seconds total,
  `3.446043468054591` seconds/case; peak CUDA allocation/reservation was
  2,173.27/2,492 MiB.
- The flat archive contains exactly 73 root files: 72 readable integer NIfTI
  masks and `subtype_results.csv` with 72 unique rows and exact header
  `Names,Subtype`.
- Every mask name and geometry matches the supplied test image, labels are
  limited to `{0,1,2}`, and every subtype is in `{0,1,2}`.
- Prediction-directory validation, staged/committed archive validation, and an
  independent source-test-to-archive audit all passed with zero issues.
- Final test ZIP size: 783,389 bytes. SHA-256:
  `5de55f4ccc1eea78ef8974d0f362039523404a1d6315d06d0ec41ec8f0d08391`.

## AI collaboration record

The assessment explicitly asks for more than 50% AI-generated code. OpenAI
Codex generated an estimated 85--95% of the initial implementation and
documentation and supported debugging, auditing, monitoring, evaluation, and
packaging. Amirfaham Fallahpour set the objective and quality bar, supplied
access and compute, chose the undergrad-first/deadline-safe scope, and reviewed
intermediate decisions and evidence. He owns the submission and retains the
responsibility to perform the final human review and upload; this log does not
claim that final review has occurred. Attribution is functional rather than
invented line ownership. Generated work was accepted only after tests or
artifact-based verification; no result was entered from expectation or visual
guesswork.
