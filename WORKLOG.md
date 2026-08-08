# Experiment worklog

This log records measured evidence and the decisions that produced it. Values
below come from saved checkpoints, evaluator outputs, validator artifacts, or
the public experiment record; no expected value is presented as measured.

## Fixed constraints and decisions

| ID | Decision | Evidence/rationale |
|---|---|---|
| D-001 | Preserve the supplied 252-case training and 36-case validation split. | Validation cases never enter training or gradients; the baseline uses the fixed split for monitoring/selection, and `splits_final.json` contains one explicit fold with no overlap. |
| D-002 | Use nnU-Net v2.8.1 3D ResEnc M. | Mandatory assessment architecture; generated `nnUNetResEncUNetMPlans.json` identifies `nnUNetPlannerResEncM`. |
| D-003 | Use no external data or pretrained weights. | Mandatory fairness constraint; training starts from the nnU-Net plan's random initialization. |
| D-004 | Repair only verified near-integer mask values in a separate copy. | 214/288 masks contained value `1.0000152587890625`; maximum repair distance was `1.52587890625e-05`, below the declared `1e-3` tolerance. Source files remain untouched. |
| D-005 | Use hybrid global-average and learned-query cross-attention pooling. | Global average supplies a stable whole-patch summary; attention can focus on discriminative bottleneck tokens without replacing that summary. |
| D-006 | Weight subtype cross-entropy by inverse training frequency. | Counts 62/106/84 give weights 1.35484/0.79245/1.0. Label smoothing is 0.05; no-lesion crops retain a 0.25 subtype-loss weight. |
| D-007 | Train for 200 x 125 updates, validate for 30 iterations/epoch. | The 25,000-update schedule balanced measured CUDA throughput with substantially more optimization than the initial 10,000-update design. |
| D-008 | Record W&B offline, then sync. | Avoids an authentication/network failure stopping the overnight run. |
| D-009 | Compare final, native segmentation-best, multitask-best, and the activated rescue checkpoint on all 36 cases. | Online subtype validation samples random patches and incomplete case coverage; it is unsuitable as sole final-selection evidence. The train-only gate activated, so the fixed pass admitted four candidates. |
| D-010 | Keep the implementation on PyTorch/CUDA rather than porting to TPU. | nnU-Net's supported path and all planned verification tooling use PyTorch/CUDA, so a second backend would add compatibility risk without strengthening the experiment. |
| D-011 | Prepare one train-metric-triggered frozen-head rescue from `checkpoint_final.pth`. | After observing collapsed online patch classification, but before any full-volume validation evaluation, freeze a single 30 x 125 AdamW schedule. Activation used only the predeclared epoch-40/50 training CE/accuracy rule; validation could not activate or alter it. |
| D-012 | Permit one disclosed zero-update numerical execution recovery, but only before custom joint fixed validation. | The first rescue process failed on batch 1 after the finite-loss guard and before `AdamW.step`; it made zero updates and wrote no checkpoint. Preserve/hash both logs, retain one update-bearing trajectory, keep every model/schedule choice fixed, move only the trainable classification path to FP32, count two process launches, and prohibit any further recovery. |
| D-013 | Preserve the completed baseline as an immutable fallback before any higher-tier upgrade. | Baseline commit `509cbe2`; PDF SHA-256 `90d68697d6330d5124f1a2533f3785033643a0985fe3ce2813b0d90a0a04fd03`; ZIP SHA-256 `5de55f4ccc1eea78ef8974d0f362039523404a1d6315d06d0ec41ec8f0d08391`. |
| D-014 | Restrict v5 head development to a best-of-two locked neural comparison on the 252 training cases. | The assignment-conforming candidates are lesion-aware mean MIL and two-query cross-attention MIL. A classical diagnostic is ineligible. Head choice uses three repeats of five-fold complete OOF macro-F1. |
| D-015 | Use deterministic class-balanced sampling with replacement and unweighted cross-entropy for v5. | This implements Amirfaham's imbalance direction without combining a balanced sampler with class weights. Focal loss and SMOTE were excluded prospectively rather than added as unbounded alternatives. |
| D-016 | Adapt Amirfaham's per-class-threshold proposal to additive multiclass log-score offsets. | Three classes are mutually exclusive, so separate 0.5 thresholds are incoherent. The bounded offsets activate only with at least 0.01 train-only macro-F1 gain and no more than 0.02 minimum-recall loss. |
| D-017 | Permit one locked post-hoc v5 official reevaluation, then replace the baseline only on strict improvement and complete contracts. | The original official validation result and baseline test package predate v5. V5 obtained macro-F1 `0.5254150702426564`, strictly above `0.46399340516987575`, and its contracts passed, so replacement occurred. Higher-tier status remained separate and failed. |
| D-018 | Retain the narrow dependency-pruning design/conformance evidence separately from the strict stock nnU-Net speed gate. | No final all-72 v3 timing claim is made. Only the stock ABBA comparison can support the higher-tier claim; checkpoint, inputs, TTA, tile step, hardware, export semantics, and outputs must match. |
| D-019 | Select v5 only as the stronger of two locked neural heads, not as a globally best model. | The strict official replacement gate passed, but macro-F1 `0.5254150702426564` missed both the undergraduate `0.60` and higher-tier `0.70` classification bars; severe refit overfitting and label-exposed shared features further limit the claim. |
| D-020 | Reject the higher-tier speed claim from the completed stock ABBA audit. | Stock averaged `236.7340` s and the broader candidate `281.2425` s over 72 cases. Runtime reduction was `-18.801059%`, meaning the candidate was slower, although numerical-equivalence contracts passed. |

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
- Stock segmentation inference completed over all 36 fixed-validation volumes, confirming the wrapper preserves nnU-Net's default tensor return contract.
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
`checkpoint_best_multitask.pth` was copied to a separate, hash-verified
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

## 2026-08-06 — Historical baseline fixed validation

Status: **DONE**

All four admitted candidates were inferred and evaluated exactly once with the
same full-volume settings over all 36 fixed-validation cases. The frozen equal-weight
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

## 2026-08-06 — Historical baseline test inference and packaging

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
- Prediction-directory validation, staged/committed archive validation, and a
  separate source-test-to-archive audit all passed with zero issues.
- Final test ZIP size: 783,389 bytes. SHA-256:
  `5de55f4ccc1eea78ef8974d0f362039523404a1d6315d06d0ec41ec8f0d08391`.

These baseline validation and test accesses occurred before the v5 protocol.
They are historical facts, not v5 tuning inputs, and prevent describing the
later official pass as a first-look holdout.

## 2026-08-06 — Post-baseline v5 protocol locks

Status: **DONE**

The project plan included both the undergraduate minimum and the higher-tier
accuracy/speed goals from the outset. Work was staged so the complete baseline
was verified before the higher-tier branch began. `docs/PHD_UPGRADE_PROTOCOL.md`
was frozen at `2026-08-06T14:05:57Z`; it made the baseline immutable, restricted
v5 architecture/training/selection to the 252 supplied training cases, allowed
one post-lock official reevaluation, and required conjunctive higher-tier gates.

The eligible neural search and decision rule were frozen before any eligible
v5 feature extraction or head training:

- neural-head lock SHA-256:
  `a8c2147493718acc96e4aa5dc471bf3f3277f0b99e8a8f7620bf966ab7b70d11`;
- neural decision lock SHA-256:
  `e28a303c7d3da5dc7857ecc72787b6746d1e689e83167c500d4d2823c5ea540f`;
- frozen rescue checkpoint SHA-256:
  `d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116`;
- frozen encoder/decoder/rescue-head component hashes:
  `324f5f75debb9885e270102a8222ed3248483ea21ff2bb6fb0177730f2b85ff1`,
  `b38d332ce7d812b98b03389777a303f6e739a789cc685cbf4d52a413ba4711f2`,
  and `1c6378fe0a2f8e792b183c8b0333b164bf2d67951147c96fadae390bd7cc6df8`.

The locks admit exactly two neural heads: lesion-aware mean MIL and two-query,
four-head cross-attention MIL. They prohibit reference masks as features,
external/pretrained data, identifiers as predictors/split keys, double
imbalance correction, unbounded loss search, and official/test feedback during
v5 development.

## 2026-08-06 — V5 train-only case-feature extraction

Status: **DONE**

- Scope: exactly 252 supplied training cases, class counts `62 / 106 / 84`.
- Official validation images/masks/labels/metrics read by the v5 extractor:
  `false`; test data read: `false`; combined train/validation metadata read:
  `false`.
- Ground-truth masks used as features: `false`; case IDs, paths, filenames, and
  enumeration order used in the numeric model matrix: `false`.
- Production-matched extraction used Gaussian sliding-window accumulation and
  all eight mirror views. Model-predicted lesion mass ranked at most three
  tiles/case; no reference mask ranked a tile.
- Completed `252` cases, `641` logical tiles, and `5,128` network forwards in
  `775.3165593` seconds (`3.0766530` seconds/case), batch size 1, with zero OOM
  fallbacks. Peak CUDA allocation/reservation: `2170.2275 / 2500.0 MiB`.
- Every per-case cache was rehashed before training and the cache set was
  exactly 252 cases.

Audit SHA-256 values:

- extraction: `db199b4bf00ae7b0c99dfbf8978fb423a31721315dff87c428944bb17059c77b`;
- cache manifest: `4e8778af4ae525901519b1249865bea38c5f42466d9f636520610ed1ea6203e7`;
- feature schema: `38430f0fbeb27385efac311ba87e175373a1384c41780545682402e6515037b0`.

## 2026-08-06 — V5 train-only neural-head comparison

Status: **DONE**

Each locked head ran five stratified folds under each of three repeat seeds.
Every trajectory used 150 epochs, case batch size 16, 256 samples/epoch,
deterministic class-balanced sampling with replacement, unweighted
label-smoothed cross-entropy, AdamW at `3e-4`, weight decay `1e-4`, gradient
clipping at 1.0, and cosine decay. There was no early stopping, mixed precision,
class-weighted loss, focal-loss branch, SMOTE, validation feedback, or test
feedback.

| Locked head | Parameters | Repeat complete-OOF macro-F1 | Mean | Minimum repeat/class recall |
|---|---:|---|---:|---:|
| Lesion-aware mean MIL | 117,263 | 0.4526558 / 0.4011152 / 0.4054124 | 0.4197278 | 0.2976190 |
| **Two-query cross-attention MIL** | **101,391** | **0.4694335 / 0.5271997 / 0.5272614** | **0.5079649** | **0.4354839** |

Cross-attention won all three repeats and the declared mean criterion by
`0.0882371`, outside the 0.01 tie band. It was therefore refit on all 252 cases
for exactly 150 epochs. The selected bundle SHA-256 is
`6e4ed210bc23cd7c7bfe02c46816dd8461c0be84108d4f9d2a36f1409b6df09d`;
the final refit state SHA-256 is
`7954be6f9620f77dc80365df97bf374b84e976ad5df12f3dc4ea4acc34892e3f`.

The refit training-resubstitution macro-F1 was `0.9787115`, versus mean repeated
OOF macro-F1 `0.5079649`; the `0.4707466` gap is severe overfitting. The OOF
estimate is not unbiased end to end: the common frozen encoder and rescue head
had already been trained using all 252 labels, and the rescue checkpoint had
already been selected on the historical official validation pass. Selecting
the winner of two heads can also make the winning OOF result optimistic.

The public W&B run
[`u03yz7ds`](https://wandb.ai/amirfahamfallahpour1379-university-of-toronto/pancreas-multitask-amirfaham-fallahpour/runs/u03yz7ds)
is `finished`. It contains both candidates, all repeat/fold trajectories and
refit markers, and explicitly records `official_validation_used=false` for the
v5 head experiment. A separate recomputation verified all 252 unique OOF cases
per repeat, class counts, argmax/log-softmax consistency, candidate metrics,
selection, and the complete decision-offset search.

## 2026-08-06 — V5 train-only class-offset decision

Status: **DONE — REJECTED BY GATE**

Amirfaham proposed class-specific thresholds and stronger imbalance handling.
For mutually exclusive multiclass output, the threshold idea was implemented
as additive per-class log-score offsets rather than three unrelated 0.5 cutoffs.
The cross-fitted offset rule changed mean macro-F1 from `0.5079648501` to
`0.5040682219` (gain `-0.0038966282`) and minimum repeat/class recall from
`0.4354838710` to `0.4193548387`. The required gain condition failed, so the
final offsets are exactly `[0,0,0]`. This is a negative decision-boundary result,
not a probability-reliability study. The offset fitting was not nested around
neural-head fitting; because offsets were rejected, it does not alter the final
decision rule.

## 2026-08-06 — V5 deterministic inference and speed protocol

Status: **DONE — NUMERICAL EQUIVALENCE PASSED; >=10% SPEED GATE REJECTED**

The first full-versus-dependency-pruned train-only smoke differed by 3 and 10
boundary voxels across two cases because the predictor constructor re-enabled
cuDNN benchmarking. After a pre-repair determinism lock, the runtime fixed and
reasserted deterministic algorithms, cuDNN settings, `CUBLAS_WORKSPACE_CONFIG`,
TF32, and compilation state. A stock-versus-candidate smoke then differed by 5
and 15 boundary voxels. A same-logit diagnostic isolated the cause to terminal
export precision: stock resampled FP16 stitched logits, while v5 first cast them
to FP32. A second narrow lock was frozen before removing only those terminal
casts. Newly exported v5 masks therefore cannot be assumed byte-identical to
the historical baseline package solely because model weights are frozen.

The original stock-speed lock also prohibited later candidate changes and
train-only timing smokes. The retained conformance artifacts nevertheless
contain internal timing fields, and the two repairs changed
inference implementation code after that lock. This is recorded as a literal
protocol deviation. The diagnostic timings are ineligible for, and excluded
from, final speed arithmetic. Both repairs used train-only inputs, were covered
by separate prospective conformance locks before their edits, and changed no
weights, learned features, selected head, or class offsets.

The first real two-case execution through the new non-timing stock harness
failed only its process-provenance check: the runner compared the launcher PID
with the deterministic bootstrap's recorded inner inference PID. All duration
fields were null or redacted and no final lock or one-use ledger was touched.
The runner audit was corrected to bind both launcher and inner PIDs. A fresh
replacement functional smoke then passed for stock and candidate with zero
OOM/fallbacks, exact two-case masks and checked geometry/dtype, and valid
candidate subtype/probability exports. Both diagnostics remain retained; only
the later all-72 ABBA run is eligible for speed arithmetic.

The v3 dependency-pruning experiment is retained as a frozen causal design plus
two-case train-only exact-conformance evidence. No all-72 v3 timing benchmark is
executed or claimed. Assignment speed acceptance rests only on the installed
stock nnU-Net v2.8.1 entry point versus custom candidate in strict ABBA order,
two timed repeats per arm, over all 72 cases. Checkpoint, plans, TTA, Gaussian
weighting, 0.5 step, post-lock non-default deterministic execution policy,
hardware, and FP16 export semantics are the same. Stock retains three
preprocessing/export workers; the custom candidate is serial. The completed
audit measured:

- stock repeats `237.093 / 236.375` s, mean `236.7340` s
  (`3.2879722` s/case);
- candidate repeats `292.313 / 270.172` s, mean `281.2425` s
  (`3.9061458` s/case); and
- runtime reduction `-18.8010594169%`, meaning the candidate was `18.801059%`
  slower, so the required `>=10%` reduction was rejected.

Numerical equivalence nevertheless passed: zero hard-mask disagreements over
the 141,878,022-voxel 72-case corpus, with case inventory, geometry, affine,
voxel sizes, qform/sform, spatial units, header dtype, and mask value domain all
passing. Candidate repeats had identical subtype decisions and probabilities
(maximum absolute probability delta `0.0`). The result is specific to these
implementations on the RTX 4060 Laptop GPU and does not generalize across
hardware. Speed-audit SHA-256:
`8e56b970e9922627a57b60762c381956410d8f0d6b3884d3799edc633bb2f4a5`.

## 2026-08-06 — Locked v5 official reevaluation

Status: **DONE — STRICT BASELINE REPLACEMENT PASSED; CLASSIFICATION TARGETS FAILED**

The one permitted model invocation completed all 36 cases and wrote a complete
runtime plus 39 hash-audited inference artifacts. This is not described as
human-blinded because validation identifiers contain label-like prefixes; the
narrower controls are that identifiers were not model inputs, references were
unopened before predictions were hash-frozen, and no model, threshold, or
decision rule changed after inference. Before any reference path was tested or
opened, Windows PowerShell 5.1 rejected two `PSObject.Properties.Count` checks
in post-inference auditing. The catch path also could not update the
already-consumed ledger because this host rejected `File.Replace(..., $null)`.
The original consumed-ledger bytes were preserved with SHA-256
`8d2de9121285f6b38aae6a608cd4b02292de3ba44ac550ca39e36e2de922bb23`.

Inference was not rerun. The saved artifact set has SHA-256
`fec59a6b546d9158e6a32eb6be1d4f889b296a184b877dfc0e5baa323e180b28`;
the full strict runtime validator passed after only the two in-memory PS5
collection-count substitutions. Recovery protocol
`official_evaluation_recovery_protocol_v1.json` was frozen and pushed before
recovery implementation or target access. The separately locked recovery then
performed only a hash-first, audit-only continuation over those saved outputs:
zero inference calls and exactly one unchanged evaluator invocation. The
completed gate explicitly records the interruption and recovery provenance.

The single allowed post-lock evaluation produced:

- whole-pancreas Dice `0.9201611779378113`, sample SD `0.03578122153521238`,
  95% CI `[0.907897868785037, 0.9308171262530192]`;
- lesion Dice `0.6196623932885514`, sample SD `0.3206780089458158`, 95% CI
  `[0.5148076709627597, 0.7165386879122176]`;
- macro-F1 `0.5254150702426564`, 95% CI
  `[0.3583611738503043, 0.6735718119776091]`;
- accuracy `0.5277777777777778`, 95% CI
  `[0.3611111111111111, 0.6944444444444444]`;
- per-class F1 for labels 0/1/2: `0.5555555555555556`, `0.4`, and
  `0.6206896551724139`; and
- confusion matrix (rows reference, columns prediction):
  `[[5, 2, 2], [4, 5, 6], [0, 3, 9]]`.

Whole-pancreas and lesion Dice met both their undergraduate and higher-tier
thresholds. Macro-F1 strictly improved on baseline `0.46399340516987575`, so
the replacement gate passed and the two-query cross-attention MIL classifier
became authoritative. This is a best-of-two locked-head decision, not a global
optimality claim. Macro-F1 still missed both the undergraduate `>=0.60` and
higher-tier `>=0.70` classification bars. Therefore the complete undergraduate
performance bar and the higher-tier joint metric bar were not met; the failed
speed gate independently precludes the higher-tier claim.

Official gate SHA-256:
`6efb7d9cfb745ecffc06cd5c981ab360b980dfb5d2a49b18537d1aab236c3df7`.
Bound metrics SHA-256:
`bdc3e538266b5fff886e5fc7205d36f0ff66c3794b3d51143ce413326c967a6b`.

## 2026-08-06 — Locked selected test packaging

Status: **DONE — SELECTED V5 ARCHIVE VALID**

Selected test inference ran once only after the official replacement gate was
consumed. The authoritative archive is `Amirfaham_Fallahpour_results.zip`,
SHA-256
`34afe1d74b70a24facceee890c03919bc5dbe036383206079fe221aa34ddd444`.
The prediction-directory and extracted-archive validators both passed: 72 masks,
72 subtype rows, exact expected case inventory, readable integer masks with
valid `{0,1,2}` labels, matching geometry, and zero issues. The flat-root ZIP
contains exactly 73 files (72 masks plus `subtype_results.csv`).

## 2026-08-07 to 2026-08-08 — V7 shallow-feature iteration and deployment

Status: **ALL FOUR HIGHER-TIER GATES MET**

Amirfaham purchased NVIDIA H100 cloud-compute time for production training and
feature development. The resulting checkpoint, histories, feature banks, and
predictions were hash-checked before evaluation. This established a stable
segmentation result and isolated classification representation as the remaining
technical question.

The V7 iteration proceeded through separate implementation and verification
commits. First, a morphology-cache overflow was fixed and covered by tests.
Next, guarded shallow-tap fine-tuning and frozen stage probes tested whether the
bottleneck was discarding subtype information. The probes supported that
hypothesis: shallow features separated the classes more effectively. View and
scale experiments then selected stage 1, mirror view 6, and full spatial scale.
Finally, a shrinkage-LDA classifier was refitted on 252 training rows and zero
validation rows. Its SHA-256 is
`bbdb0fc79b35cfc81400550ad558636be6c15663f623b230813ddcb46264d0df`.

Independent V7 validation produced:

- whole-pancreas Dice `0.9201569021` (SD `0.0352781414`);
- lesion Dice `0.6196343545` (SD `0.3161915054`);
- macro-F1 `0.7445103206`; and
- confusion matrix `[[6,2,1],[0,13,2],[1,3,8]]`.

All three accuracy point thresholds pass. The validation split informed the
stage/view deployment choice, so this is disclosed as a development-set result.
Spatial scales 0.25, 0.375, 0.5, and 0.625 all reduced macro-F1 and were
rejected. Tile/TTA batching, process-based classification, and `torch.compile`
were also measured and rejected on this RTX 4060 environment. The retained
engineering path uses stage-1/view-6 CPU classification, resident half weights,
one resident fold, and overlapped preprocessing/export while preserving TTA and
step size 0.5.

The first complete paired benchmark showed that the correct candidate path was
slower than stock, which triggered profiling rather than a change to TTA or tile
step. The retained changes eliminated repeated weight loading, reduced the
classifier to the selected stage/view, used half-precision resident weights,
ran the shallow feature path asynchronously on CPU, and overlapped preprocessing
and export. A new six-process, all-72-case audit then measured stock at
`259.5160` seconds and the complete candidate at `231.2600` seconds, a
`10.8880%` reduction. Each candidate repeat produced 72 masks and a valid subtype
CSV. Cross-arm agreement, repeat stability, geometry, dtype, and subtype-output
checks passed, so the final speed gate passed without disabling TTA or changing
step size 0.5.

W&B records track the fine-tuning metric archive (`uzc4elyc`), independent
validation (`wrd1f1c8`), initial complete inference audit (`4wb71b3i`), and
final eligible speed audit (`uy3u0pff`). The fine-tuning dashboard contains 21
saved training events and records `live_training_run=false`. All four runs are
remotely verified as `finished`; exact URLs and evidence sources are tracked in
`docs/evidence/v7/wandb_runs.json`.

## AI collaboration record

The assessment explicitly asks for more than 50% AI-generated code. Amirfaham
Fallahpour set the research objective, required evaluation against both target
tiers, prioritized class imbalance and representation quality, selected and
purchased the compute, and made the experiment and submission decisions. OpenAI
Codex translated those priorities into substantial implementation,
documentation, debugging, audit execution, evaluation, and packaging. Amirfaham
owns the submission and is responsible for its final review. Generated work was
accepted only after tests or artifact-based verification; no result was entered
from expectation or visual guesswork.
