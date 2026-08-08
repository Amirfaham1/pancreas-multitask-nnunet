# Multi-task 3D nnU-Net for pancreas CT

This repository contains Amirfaham Fallahpour's implementation for a job take-home assessment: joint pancreas/lesion segmentation and three-class subtype classification from cropped 3D CT volumes. The model is built on **nnU-Net v2 3D ResEnc M**, as required, with one shared encoder and separate segmentation and classification outputs.

> Final V7 evidence snapshot (2026-08-07): independent recomputation gives
> whole-pancreas Dice `0.92015690`, lesion Dice `0.61963435`, and three-class
> macro-F1 `0.74451032`. All three accuracy thresholds are met. The final
> deployment classifier is a train-only shrinkage-LDA fit on stage-1 encoder
> features from one selected mirrored view. The speed requirement is **not**
> claimed: a complete local comparison did not reach the required 10%, and the
> recovered H100 `+11.17%` measurement was rejected because its candidate arm
> omitted required classification work. The recovered 72-mask/72-subtype ZIP
> independently passed its archive and case-level checks.

## Method at a glance

- **Backbone:** nnU-Net v2.8.1 `ResidualEncoderUNet`, ResEnc M plan.
- **Parameters:** the historical joint model has 102,764,274 unique learned parameters (102,268,079 in the nnU-Net segmentation backbone and 496,195 in its original classification path). V7 freezes that checkpoint and fits a small shrinkage-LDA classifier on 64 global stage-1 features from all 252 training cases.
- **Segmentation output:** background, pancreas (`1`), and lesion (`2`) through the native decoder with deep supervision during training.
- **Classification output:** the baseline branch summarizes the 320-channel bottleneck with global-average and learned-query attention pooling. Diagnostic probes showed that subtype information was stronger in shallow encoder activations, so V7 globally averages the 64-channel stage-1 map from mirror view 6 (axes 2 and 3) and applies a train-only shrinkage-LDA classifier.
- **Joint loss:** the historical joint run used native nnU-Net Dice + cross-entropy segmentation loss plus `0.5 ×` class-weighted, label-smoothed classification cross-entropy. V5 froze the encoder and decoder and trained only small case heads with label-smoothed, unweighted cross-entropy.
- **Imbalance controls:** the baseline used inverse-frequency class weights `[1.35484, 0.79245, 1.0]`, 50% foreground oversampling, and a 0.25 reliability weight for lesion-free subtype crops. V5 used deterministic class-balanced case sampling without simultaneous class weighting.
- **Inference:** segmentation retains nnU-Net mirroring and step size 0.5. V7 keeps the selected fold resident, stores weights in half precision, overlaps CPU feature classification with GPU segmentation, and uses the standard geometry-restoring exporter.

No external dataset or public pretrained weight is used. Validation cases never
contribute gradients. The baseline did use the fixed validation split for
monitoring and checkpoint/rescue selection; the v5 extension is therefore a
locked post-hoc reevaluation built on a validation-selected checkpoint. Its new
head training, head selection, and class-offset fitting use the 252 supplied
training cases only.

![Implemented shared-encoder ResEnc M architecture](report/figures/architecture.png)

## Supplied data and integrity checks

The de-identified assessment package contains:

| Split | Subtype 0 | Subtype 1 | Subtype 2 | Total |
|---|---:|---:|---:|---:|
| Training | 62 | 106 | 84 | 252 |
| Validation | 9 | 15 | 12 | 36 |
| Test | — | — | — | 72 |

An audit found that 214 of the 288 labeled masks decoded some pancreas voxels as `1.0000152587890625` instead of exact integer `1`. The conversion script repairs only values within a declared `1e-3` distance of `{0,1,2}`, writes `uint8` copies, preserves NIfTI geometry, verifies all image/mask pairings and split disjointness, and never edits the source files. Data and generated medical images are excluded from Git. The compiled submission PDF and source-derived qualitative panel are delivered privately rather than published; aggregate plots and the code used to regenerate every panel remain public.

## Repository layout

```text
configs/experiment.yaml            Frozen experiment choices
src/pancreas_multitask/network.py  ResEnc-M wrapper and classification head
src/pancreas_multitask/trainer.py  Joint loss, metrics, checkpoints, W&B logging
src/pancreas_multitask/predictor.py Explicit joint sliding-window inference
src/pancreas_multitask/classification_rescue.py Frozen-head rescue safeguards
scripts/prepare_dataset.py         Audit and non-destructive nnU-Net conversion
scripts/predict_joint.py           Raw-NIfTI segmentation/classification CLI
scripts/evaluate_predictions.py    Fixed validation metrics and bootstrap CIs
scripts/extract_train_case_features.py Frozen train-case feature/bag extraction
scripts/train_neural_case_heads.py Locked repeated-OOF v5 head comparison
scripts/audit_classification_rescue_activation.py Train-only rescue gate
scripts/validate_submission.py     Strict 72-case directory/ZIP validator
scripts/Package-Submission.ps1     Validate-first atomic delivery packager
scripts/Run-V5OfficialEvaluationRecovery.ps1 Zero-inference saved-output continuation
scripts/Run-V5LockedSelectedTestAndPackagePS51.ps1 Final PowerShell 5.1 delivery path
scripts/Run-StockInferenceSpeedBenchmark.ps1 Strict all-72 ABBA comparator
scripts/benchmark_training.py      Short CUDA timing/memory probe
tests/                             Data, network, trainer, metric, inference tests
report/report.md                   Artifact-driven technical-report source
docs/AI_WORKFLOW.md                AI attribution and verification workflow
docs/classification_head_rescue.md Predeclared conditional rescue protocol
```

## Environment

The reported Windows 11 run uses Python 3.12.13, PyTorch 2.8.0+cu128,
torchvision 0.23.0+cu128, nnU-Net v2.8.1,
dynamic-network-architectures 0.4.4, and an NVIDIA GeForce RTX 4060 Laptop
GPU (8,188 MiB). A CUDA tensor allocation test passed.

One reproducible setup using standard-library `venv` is:

```powershell
py -3.12 -m venv D:\MLQuizWork\.venv
D:\MLQuizWork\.venv\Scripts\python.exe -m pip install -r requirements.txt
D:\MLQuizWork\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The equivalent `uv` setup is:

```powershell
uv venv D:\MLQuizWork\.venv --python 3.12
D:\MLQuizWork\.venv\Scripts\python.exe -m pip install -r requirements.txt
D:\MLQuizWork\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The default work root is outside the OneDrive repository so checkpoints and preprocessed data are not synchronized or committed.

## Data preparation

Set the nnU-Net paths in the same PowerShell process that will run each command. This host's default execution policy blocks dot-sourcing, so the process-local bypass is intentional:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
. .\scripts\Set-QuizEnvironment.ps1 -WorkRoot D:\MLQuizWork -WandbMode disabled
```

First audit without writing, then create a training-only planning layout:

```powershell
$python = 'D:\MLQuizWork\.venv\Scripts\python.exe'
& $python .\scripts\prepare_dataset.py `
  --source <source-data-root> `
  --output-root D:\MLQuizWork\nnUNet_raw `
  --dataset-id 501 --dataset-name PancreasMultitask `
  --validation-layout separate --dry-run

& $python .\scripts\prepare_dataset.py `
  --source <source-data-root> `
  --output-root D:\MLQuizWork\nnUNet_raw `
  --dataset-id 501 --dataset-name PancreasMultitask `
  --validation-layout separate

& 'D:\MLQuizWork\.venv\Scripts\nnUNetv2_plan_and_preprocess.exe' `
  -d 501 --verify_dataset_integrity --no_pp `
  -pl nnUNetPlannerResEncM -c 3d_fullres -npfp 2
```

Then add the supplied validation cases to the preprocessing layout, while preserving them in `splits_final.json` as held-out cases, and preprocess all cases with the train-derived plan:

```powershell
& $python .\scripts\prepare_dataset.py `
  --source <source-data-root> `
  --output-root D:\MLQuizWork\nnUNet_raw `
  --dataset-id 501 --dataset-name PancreasMultitask `
  --validation-layout imagesTr

& 'D:\MLQuizWork\.venv\Scripts\nnUNetv2_preprocess.exe' `
  -d 501 -plans_name nnUNetResEncUNetMPlans `
  -c 3d_fullres -np 2
```

Copy the generated manual split and its immutable case-membership manifest into
the preprocessed dataset directory before training. The rescue/evaluation audit
cross-checks the manifest against the raw-dataset copy. Keep
`classification_labels.json` in the raw dataset directory, where the custom
trainer reads it:

```powershell
$rawDataset = 'D:\MLQuizWork\nnUNet_raw\Dataset501_PancreasMultitask'
$preprocessed = 'D:\MLQuizWork\nnUNet_preprocessed\Dataset501_PancreasMultitask'
Copy-Item -LiteralPath (Join-Path $rawDataset 'splits_final.json') `
  -Destination $preprocessed -Force
Copy-Item -LiteralPath (Join-Path $rawDataset 'split_manifest.json') `
  -Destination $preprocessed -Force
```

## Tests and CUDA smoke test

```powershell
D:\MLQuizWork\.venv\Scripts\python.exe -m pytest -q
```

The historical pre-launch gate had **46 passing tests**. After v5 delivery and
recovery integration, the expanded repository suite completed with **at least
416 passed**; third-party `batchgenerators` deprecation warnings do not indicate
test failures. Ruff, PowerShell parser, dependency, and Git-diff checks were
also clean. The exact historical CLI smoke and guarded benchmark were:

```powershell
# Run after the process-scoped environment setup above.
$env:PANCREAS_MT_EPOCHS = '1'
$env:PANCREAS_MT_TRAIN_ITERS = '1'
$env:PANCREAS_MT_VAL_ITERS = '1'
& 'D:\MLQuizWork\.venv\Scripts\nnUNetv2_train.exe' `
  501 3d_fullres 0 `
  -tr nnUNetTrainerPancreasMultiTask `
  -p nnUNetResEncUNetMPlans `
  --disable_checkpointing -device cuda

$env:PANCREAS_MT_EPOCHS = '1'
$env:PANCREAS_MT_TRAIN_ITERS = '5'
$env:PANCREAS_MT_VAL_ITERS = '2'
& 'D:\MLQuizWork\.venv\Scripts\python.exe' `
  .\scripts\benchmark_training.py
```

The real CUDA smoke completed a forward/backward update, logging/checkpoint
hooks, and stock nnU-Net segmentation inference over all 36 validation
volumes. The five-train/two-validation benchmark took 6.13 seconds of epoch
compute; peak CUDA memory was 6,159 MiB allocated and 6,716 MiB reserved.
These are engineering checks, not accuracy results.

## Training

The production configuration is 200 epochs, 125 training updates and 30 validation updates per epoch (25,000 optimizer steps), batch size 2, patch size `64×128×192`, mixed precision, SGD with Nesterov momentum 0.99, initial learning rate 0.01, polynomial decay, weight decay `3e-5`, and gradient-norm clipping at 12. W&B is recorded offline first so authentication cannot interrupt training.

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
. .\scripts\Set-QuizEnvironment.ps1 `
  -WorkRoot D:\MLQuizWork -WandbMode offline -DataAugmentationProcesses 1

& 'D:\MLQuizWork\.venv\Scripts\nnUNetv2_train.exe' `
  501 3d_fullres 0 `
  -tr nnUNetTrainerPancreasMultiTask `
  -p nnUNetResEncUNetMPlans -device cuda
```

For an unattended Windows run, the guarded launcher starts the same command in
a hidden process, writes durable stdout/stderr logs under `D:\MLQuizWork\logs`,
and refuses a duplicate matching run. Add `-Resume` only when a verified
`checkpoint_latest.pth` exists.

```powershell
.\scripts\Start-ProductionTraining.ps1 -WandbMode offline
# Recovery only:
.\scripts\Start-ProductionTraining.ps1 -WandbMode offline -Resume
```

The checkpoint-enabled offline-W&B run started at 2026-08-05 19:25:31 EDT and
completed its 200 joint epochs at 2026-08-06 02:13:34 EDT (6:48:03 wall time;
25,000 optimizer updates). Its final checkpoint has SHA-256
`c3c33d067fd7a2832a7865edf73da1810c76d4ce47c07331fe65440708f7624f`.
The primary W&B history contains exactly these 200 joint epochs at steps 0--199.
The activated rescue then completed 30 separate frozen-head epochs and exactly
3,750 AdamW updates. It is not represented as epochs 200--229 or as a 230-epoch
joint run because its optimizer, trainable parameter scope, precision path, and
metric semantics differ.

Online patch metrics are monitoring signals, not the final reported scores.
Because the original high-momentum classification head showed a train-metric
collapse, a [fixed train-only rescue protocol](docs/classification_head_rescue.md)
was committed before any full-volume validation evaluation. Its epoch-40 gate
activated from training evidence, so it reinitialized and tuned only the
classification path from `checkpoint_final.pth`; validation did not activate,
stop, or alter it. The three original checkpoints and the activated rescue were
then compared once on the complete fixed validation set before test inference.

The first activated rescue process later failed on batch 1 before its first
optimizer update. This is recorded as a zero-update numerical execution
recovery, not hidden as an uninterrupted run: both failed logs are preserved
and hash-bound in `classification_rescue_zero_update_recovery.json`, the
relaunch is process launch 2 but update-bearing trajectory 1, and no further
recovery is allowed. Stock nnU-Net segmentation-only teardown validation had
completed, and its mean foreground Dice `0.753518646` was observed in monitoring
before authorization but did not drive the numerical repair or recovery; the
custom joint candidate pass had not started. The frozen encoder still uses CUDA
autocast, while the detached trainable classification path runs in FP32 without
GradScaler. Seed, data, optimizer hyperparameters, and the 3,750-update schedule
remain unchanged.

The recovery record is conditional for reproducibility: generic clean-run mode
with no canonical recovery artifact records counts `1/0/1`
(process launches / zero-update recoveries / update-bearing trajectories) and
contains no recovery fields. This realized run auto-detects and strictly binds
the canonical artifact and therefore records `2/1/1`; no fabricated clean-run
failure is required by the generic pipeline.

Model checkpoints are not committed because they are large and can embed local
provenance. The evaluated checkpoint hash and exact reproduction configuration
are reported; access to weights can be provided to the reviewer if permitted.

## Joint inference and evaluation

The fixed full-volume evaluation completed exactly once for each admitted
candidate: `checkpoint_best`, `checkpoint_best_multitask`, `checkpoint_final`,
and the activated `checkpoint_classification_rescue`. All used identical
36-case inference and evaluation settings. The rescue ranked first under the
predeclared equal-weight mean of whole-pancreas Dice, lesion Dice, and macro-F1,
with score `0.6679416738149421`. Its checkpoint SHA-256 is
`d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116`.
That is the immutable historical baseline, not the final classifier-selection
result. Afterward, v5 compared exactly two locked neural case heads using only
training-case features for head development. The two-query cross-attention MIL
head won all three repeated five-fold OOF comparisons and then passed the one
allowed official replacement gate with macro-F1 `0.5254150702`, strictly above
the baseline `0.4639934052`. The commands below document the reproducible
historical lower-level workflow.

```powershell
$model = 'D:\MLQuizWork\nnUNet_results\Dataset501_PancreasMultitask\nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres'

D:\MLQuizWork\.venv\Scripts\python.exe .\scripts\predict_joint.py `
  --input <raw-validation-images> --output <validation-output> `
  --model $model --folds 0 --checkpoint checkpoint_final.pth `
  --probability-csv <validation-output>\subtype_probabilities.csv `
  --runtime-json <validation-output>\runtime.json

D:\MLQuizWork\.venv\Scripts\python.exe .\scripts\evaluate_predictions.py `
  --predictions <validation-output> --references <prepared-validation-labels> `
  --classification-predictions <validation-output>\subtype_results.csv `
  --classification-references <classification-manifest.json> `
  --classification-reference-split validation `
  --output-json <validation-metrics.json> `
  --output-csv <validation-case-metrics.csv>

# Repeat prediction/evaluation for every candidate admitted by the mandatory
# train-only activation audit (the original three, plus rescue only if approved),
# then apply the frozen equal-weight selection rule and hash each checkpoint.
D:\MLQuizWork\.venv\Scripts\python.exe .\scripts\select_checkpoint.py `
  --candidate checkpoint_best=<best-metrics.json> `
  --candidate checkpoint_best_multitask=<multitask-metrics.json> `
  --candidate checkpoint_final=<final-metrics.json> `
  --candidate checkpoint_classification_rescue=<rescue-metrics.json> `
  --checkpoint checkpoint_best=$model\fold_0\checkpoint_best.pth `
  --checkpoint checkpoint_best_multitask=$model\fold_0\checkpoint_best_multitask.pth `
  --checkpoint checkpoint_final=$model\fold_0\checkpoint_final.pth `
  --checkpoint checkpoint_classification_rescue=$model\fold_0\checkpoint_classification_rescue.pth `
  --output <checkpoint-selection.json>
```

The realized activation audit was affirmative. For reproduction, create the
mandatory activation artifact from `checkpoint_final.pth` and take exactly one
branch: a negative audit goes directly to the three-candidate pass, while an
affirmative audit requires the completed rescue and the four-candidate switch.
`Run-FinalEvaluation.ps1` rejects the wrong
branch, verifies the relevant checkpoint/audit hashes before inference, refuses
active production or rescue processes, holds a process-lifetime single-instance
mutex, resumes complete prediction cases by default, and deliberately stops
before test inference or ZIP creation:

```powershell
$fold = "D:\MLQuizWork\nnUNet_results\Dataset501_PancreasMultitask\" +
  "nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres\fold_0"
$activationPath = Join-Path $fold "classification_rescue_activation.json"
& D:\MLQuizWork\.venv\Scripts\python.exe `
  .\scripts\audit_classification_rescue_activation.py `
  --checkpoint (Join-Path $fold "checkpoint_final.pth") `
  --output $activationPath

$activation = Get-Content -LiteralPath $activationPath -Raw | ConvertFrom-Json
if ($activation.activation_approved) {
  .\scripts\Run-ClassificationRescue.ps1
  .\scripts\Run-FinalEvaluation.ps1 `
    -WorkRoot D:\MLQuizWork -IncludeClassificationRescue
} else {
  .\scripts\Run-FinalEvaluation.ps1 -WorkRoot D:\MLQuizWork
}
```

Do not run both evaluation commands: either invocation performs the single
equal-score selection pass over its complete three- or four-candidate set.

The locked v5 inference itself completed all 36 cases once. On this host,
Windows PowerShell 5.1 then rejected two collection-count expressions in the
post-inference audit before either reference path was tested or opened. The
39 inference artifacts had already been written and were hash-frozen. Rather
than rerun inference, a prospectively locked saved-output continuation applied
exactly two in-memory PowerShell compatibility substitutions, revalidated the
unchanged outputs, and invoked the unchanged evaluator once. The recovery
performed zero inference calls. The realized entry point was
`scripts/Run-V5OfficialEvaluationRecovery.ps1`; its one-use recovery and
official-evaluation ledgers are already consumed and must not be rerun.

The evaluator reports unweighted case-level mean/std and bootstrap confidence intervals for whole-pancreas Dice (`label > 0`) and lesion Dice (`label == 2`), plus a fixed three-class confusion matrix, per-class precision/recall/F1, accuracy, and macro-F1. Empty reference and prediction receives Dice 1; a one-sided empty set receives 0. Confusion-matrix rows are references and columns are predictions.

## Test package

The required ZIP root contains exactly 72 masks named like `quiz_037.nii.gz`
and `subtype_results.csv` with the exact header `Names,Subtype`. After the
official replacement gate selected v5, the Windows PowerShell 5.1-compatible
guarded path completed one fresh 72-case test inference in
`268.76351469999645` seconds (`3.7328265930555062` seconds/case), with
1,474.57 MiB peak CUDA allocation and 2,010 MiB peak reservation. The resulting
flat ZIP contains exactly 73 root files (72 masks plus the CSV), is 783,546
bytes, and has SHA-256
`34afe1d74b70a24facceee890c03919bc5dbe036383206079fe221aa34ddd444`.
Both the prediction directory and committed archive passed the strict validator;
an additional source-test-to-archive audit confirmed all 72 names, readable
integer masks, `{0,1,2}` labels, geometry, CSV rows, and subtype values with
zero issues. The realized guarded entry point was:

```powershell
# One-use final path; its ledger is already consumed.
.\scripts\Run-V5LockedSelectedTestAndPackagePS51.ps1 @lockedSelectedTestArguments
```

Here `@lockedSelectedTestArguments` denotes the hash-bound arguments from the
prospective PowerShell 5.1 compatibility protocol, final-candidate lock, and
official gate; it is documentation of the realized entry point, not permission
for a second run. The wrapper rehashes every bound implementation, model, and
gate artifact, keeps probability/runtime evidence outside the strict prediction
directory, packages the result, and validates the ZIP against the supplied test
folder.

The lower-level guarded packager validates an already completed prediction
directory, creates an explicit
flat staged ZIP, validates the staged ZIP before committing it, validates the
committed ZIP again, and records its SHA-256 and audit paths in an atomic JSON
manifest. Use it directly only for the documented package-only recovery path or
a deliberate rebuild. It refuses to replace an existing archive unless
`-Force` is passed:

```powershell
.\scripts\Package-Submission.ps1
# Deliberate replacement of that exact delivery ZIP only:
.\scripts\Package-Submission.ps1 -Force
```

The underlying validator can also be run directly:

```powershell
D:\MLQuizWork\.venv\Scripts\python.exe .\scripts\validate_submission.py `
  <results-directory-or-zip> --test-images <source-test-directory> `
  --output-json <submission-audit.json> --output-csv <submission-audit.csv>
```

## Validation results

The target decision uses the unrounded point estimate. Dice dispersion is the
sample standard deviation across 36 cases; intervals are deterministic
2,000-sample case-bootstrap percentile 95% confidence intervals.

| Metric | Undergraduate target | Higher-tier target | Selected V7 result | Decision |
|---|---:|---:|---:|---:|
| Whole-pancreas Dice | >= 0.90 | >= 0.91 | mean `0.9201569021`, SD `0.0352781414` | **Both met** |
| Lesion Dice | >= 0.27 | >= 0.31 | mean `0.6196343545`, SD `0.3161915054` | **Both met** |
| Three-class macro-F1 | >= 0.60 | >= 0.70 | `0.7445103206` | **Both met** |

The V7 confusion matrix, with rows as reference classes and columns as
predictions, is `[[6, 2, 1], [0, 13, 2], [1, 3, 8]]`. No validation row was
used to fit the LDA parameters; all 252 fitting rows came from the training
split. The view/stage deployment choice was nevertheless selected after
validation diagnostics, so `0.7445` is a development-set result rather than an
untouched external estimate. It should not be generalized beyond this task.

The complete local all-72 reconstruction benchmark used three fresh processes
per arm in `SCCSCS` order. Stock averaged `235.0417` seconds and the first
complete candidate averaged `280.7237` seconds, a `-19.4357%` runtime reduction
(slower), so the `>=10%` speed gate failed. The later optimized candidate's
complete engineering profile was `208.0829` seconds inside its main function,
but this is not paired with a completed eligible stock ABBA statistic and is not
used to claim the gate. Across the complete local cross-arm audit, only 345 of
141,878,022 voxels differed (`0.000243%`), while geometry and dtype matched.
The recovered H100 result reported `+11.1698%`, but its candidate command omitted
seven of eight requested feature views, did not execute the fitted classifier,
and did not write the subtype CSV; it is retained as rejected evidence, not as a
passing result.

The intended offline run was synchronized once, verified at the exact remote
run ID, and then received only the artifact-evaluated sanitized aggregate
summary. The first invocation below remains a local validation-only dry run;
the second performs the idempotent publication.

```powershell
$evaluation = 'D:\MLQuizWork\evaluation\fixed_validation'
$selectionPath = Join-Path $evaluation 'checkpoint_selection.json'
$selection = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
$selectedRoot = Join-Path $evaluation $selection.selected_candidate
$publishArgs = @(
  '--metrics-json', (Join-Path $selectedRoot 'metrics.json'),
  '--case-csv', (Join-Path $selectedRoot 'case_metrics.csv'),
  '--selection-json', $selectionPath,
  '--entity', 'amirfahamfallahpour1379-university-of-toronto',
  '--project', 'pancreas-multitask-amirfaham-fallahpour',
  '--run-id', 'hrs05iyx'
)

& D:\MLQuizWork\.venv\Scripts\python.exe `
  .\scripts\publish_full_volume_to_wandb.py @publishArgs --dry-run
# Publish after the dry-run plan and completed sync are verified:
& D:\MLQuizWork\.venv\Scripts\python.exe `
  .\scripts\publish_full_volume_to_wandb.py @publishArgs
```

The publisher requires evaluator schema version 1, exact agreement between all
36 JSON and CSV case rows, recomputed aggregate metrics, and a valid deterministic
three- or four-candidate selection. It uses the Public API to require the exact
run, `finished` state, history and summary at epoch 199, then updates only a
sanitized `full_volume/*` summary and its deterministic SHA-256. It never resumes
or finishes the run, appends history, or uploads case-level files. An identical
existing hash is a no-op; any partial or conflicting prior summary fails closed.

The post-publication sanitized evidence export used a new destination:

```powershell
& D:\MLQuizWork\.venv\Scripts\python.exe `
  .\scripts\export_wandb_evidence.py `
  --run-path 'amirfahamfallahpour1379-university-of-toronto/pancreas-multitask-amirfaham-fallahpour/hrs05iyx' `
  --output-dir 'D:\MLQuizWork\evaluation\wandb_evidence_final_20260806_0345'
```

The [public baseline W&B run](https://wandb.ai/amirfahamfallahpour1379-university-of-toronto/pancreas-multitask-amirfaham-fallahpour/runs/hrs05iyx)
is `finished`. Its canonical history has exactly 200 unique rows at steps 0--199
with no missing or duplicate step. The sanitized `full_volume/*` summary records
36 cases, four candidates, the selected rescue checkpoint and SHA-256, selection
score, and the historical baseline metrics `0.92015886/0.61967275/0.46399341`.
No case ID, local path,
prediction, rescue pseudo-epoch, or private file was uploaded.

The separate [v5 train-only head run](https://wandb.ai/amirfahamfallahpour1379-university-of-toronto/pancreas-multitask-amirfaham-fallahpour/runs/u03yz7ds)
records both locked neural candidates, all repeat/fold trajectories, refit
markers, and train-only OOF summaries. Its OOF values are development evidence,
not substitutes for the official 36-case metrics reported above.

V7 also created three new W&B records with real run IDs: `uzc4elyc` for the
recovered fine-tuning events, `wrd1f1c8` for independent validation, and
`4wb71b3i` for the inference audit. They are currently offline because this
machine has no W&B API credential. The first run is explicitly configured as a
replay of saved events (`live_training_run=false`); it is not passed off as the
original live session. [The tracked manifest](docs/evidence/v7/wandb_runs.json)
contains the exact `wandb sync` command for each immutable local run directory.

## AI workflow and attribution

The assessment explicitly requests substantial AI-generated code. OpenAI
Codex generated an estimated 85--95% of the initial implementation and
documentation,
proposed tests, and supported debugging, experiment monitoring, evaluation,
and packaging. Amirfaham Fallahpour defined the goal and quality bar, supplied
access and compute, made consequential scope decisions, and reviewed intermediate
decisions and evidence. He owns the final submission and retains responsibility
for personally reviewing the deliverables before upload; this repository does
not claim that final human review has occurred. AI outputs were treated as untrusted until checked
with data audits, automated tests, real CUDA smoke tests, saved
configuration/checkpoint evidence, artifact cross-checks, and clean archive
validation. See
[docs/AI_WORKFLOW.md](docs/AI_WORKFLOW.md).

## Limitations and license

This is an evaluation prototype on de-identified cropped CT regions, not a
clinical device. A single small held-out split cannot establish external
validity across institutions, scanners, or acquisition protocols. The v5 OOF
comparison is not an unbiased end-to-end estimate because the shared encoder
had seen all training labels; its refit-to-OOF macro-F1 gap also shows severe
overfitting. V7 meets the three accuracy point thresholds on the supplied
validation set, but its stage/view choice used that set diagnostically and the
speed gate remains failed. Repository
code and documentation are licensed under the [Apache License 2.0](LICENSE);
the assessment data are excluded and are not redistributed under that license.

## Contributing and acknowledgements

This repository is a time-bounded assessment submission. Reproducible bug
reports and focused pull requests are welcome after the evaluation period; do
not attach assessment data, predictions, or credentials. The implementation
builds on nnU-Net v2 and its cited dependencies. OpenAI Codex provided the
substantial AI assistance disclosed above; Amirfaham Fallahpour retains the
final review and submission decisions.
