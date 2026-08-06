# Conditional classification-head rescue

## Status and activation rule

This is a **prepared, opt-in contingency**, not part of the active production
run. Activation uses production training diagnostics only. At completed epoch
40, the last ten completed epochs trigger rescue when mean classification CE is
at least `1.05`, mean training-patch accuracy is at most `0.42`, and the CE
ordinary-least-squares slope is at least `-0.001` CE/epoch (a smaller decline
is treated as practically flat). The hard epoch-50 audit triggers
when the latest ten-epoch mean CE is above `1.03` or mean accuracy is below
`0.45`. If neither rule activates it, fixed-validation results cannot activate
it later.

Even if precommitted at epoch 40 or 50, do not launch until the original
200-epoch run exits cleanly. The schedule is then fixed; validation cannot
change its length, learning rate, initialization, or checkpoint. There is no
hyperparameter search and no second update-bearing rescue trajectory. Process
recovery is counted separately from model attempts and must be disclosed.

This contingency was prospectively frozen after online patch diagnostics had
suggested classification collapse, but before any full-volume fixed-validation
evaluation. Those observed online diagnostics must be disclosed; they are not
claimed to be unseen.

## Disclosed zero-update execution recovery

The first rescue process launch on 2026-08-06 failed on its first training
batch: the finite-loss guard passed, but mixed-precision gradient scaling
produced a non-finite norm at fail-fast clipping. The exception occurred before
`scaler.step`/`AdamW.step`, so the failed launch made zero optimizer updates,
completed zero epochs, and wrote no rescue checkpoint or audit. Its stdout and
stderr were preserved byte-for-byte and bound by SHA-256 in
`classification_rescue_zero_update_recovery.json`.

Stock nnU-Net segmentation-only validation had completed during joint-trainer
teardown before this recovery decision. Its mean foreground Dice
`0.753518646` was observed in monitoring before recovery authorization, but it
did not drive the numerical repair, schedule, seed, or recovery decision. The
custom joint four-candidate
fixed-validation pass had not started, and the rescue process itself opened no
validation images and consumed zero validation batches.

Before any custom joint fixed-validation result existed, the numerical scope
was amended once: the frozen encoder forward remains under CUDA autocast, while
the detached bottleneck, trainable pooling/head, classification loss,
backpropagation, gradient clipping, and AdamW update use FP32 without a gradient
scaler. The source checkpoint, reset seed, data, augmentation, optimizer
hyperparameters, clip threshold, and `30 x 125` successful-update schedule are
unchanged. The relaunch is therefore reported as process launch 2, zero-update
recovery 1, and update-bearing trajectory 1. No further recovery is allowed.

Recovery evidence is conditional, not fabricated as a universal requirement.
On a clean future execution where the canonical recovery artifact is absent,
the wrapper launches normally and the rescue audit must record process launches
`1`, zero-update recoveries `0`, and update-bearing trajectories `1`, with no
`execution_recovery*` fields. For this realized run, the canonical artifact is
present, so the wrapper validates and passes it automatically and the only
accepted counts are `2 / 1 / 1`. Evaluation and packaging reject every other
combination and reject a clean audit that conflicts with a canonical artifact.

## Frozen method

The rescue always starts from `checkpoint_final.pth`; validation does not
select its initialization. It then:

1. loads the complete joint checkpoint strictly;
2. records exact SHA-256 fingerprints of encoder, decoder, and classification
   states;
3. reinitializes only learned-query hybrid pooling and the classification MLP
   with seed `20260806`;
4. freezes every encoder and decoder parameter and holds those modules in
   evaluation mode;
5. bypasses the decoder entirely, computing the frozen encoder bottleneck
   under `torch.no_grad()`;
6. optimizes every and only registered pooling/head parameter with constant
   AdamW learning rate `3e-4`, weight decay `1e-4`, and gradient-norm clipping
   at `1.0`;
7. runs exactly `30 × 125 = 3,750` training-patch updates, batch size 2, using
   the original training augmentation, foreground oversampling, inverse class
   weights, label smoothing `0.05`, and no-lesion patch weight `0.25`; and
8. verifies that encoder and decoder state hashes remain bit-identical before
   committing the final checkpoint.

The implementation constructs a single-threaded loader directly from the 252
training keys. It does **not** call nnU-Net's normal `get_dataloaders()` method,
because that method creates and primes a validation loader. Validation case IDs
are read from `splits_final.json` only to verify cardinality and zero overlap.
The shared classification metadata file is loaded, but only training keys are
indexed for targets; no validation image or segmentation volume is opened and
no validation batch is constructed or consumed. The audit records zero
validation batches, no validation gradients, and no validation stopping.
Cardinality alone is not accepted: the live training and validation case-ID
hashes must exactly match the corresponding lists in the frozen pretraining
`split_manifest.json`, whose file hash is also recorded in the rescue audit.

## Exact invocation

After production exits, first create the machine-readable activation artifact
from `checkpoint_final.pth`. This script reads only its saved
`train_cls_losses` and `train_cls_accuracy`, computes both frozen audits, and
binds the decision to the checkpoint SHA-256:

Do **not** run `Run-FinalEvaluation.ps1` before this gate; the orchestrator now
refuses a missing activation artifact. If rescue activates, finish it first and
then evaluate all four candidates in one extended fixed-validation pass. If it
does not activate, proceed with the original three-candidate workflow.

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
$fold = "D:\MLQuizWork\nnUNet_results\Dataset501_PancreasMultitask\" +
  "nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres\fold_0"
$python = "D:\MLQuizWork\.venv\Scripts\python.exe"
& $python .\scripts\audit_classification_rescue_activation.py `
  --checkpoint (Join-Path $fold "checkpoint_final.pth") `
  --output (Join-Path $fold "classification_rescue_activation.json")
```

Read `activation_approved` from that artifact and take exactly one branch. A
negative audit prohibits rescue and admits exactly the three original
candidates. An affirmative audit requires the one frozen rescue to finish
before all four candidates are evaluated together:

```powershell
$activationPath = Join-Path $fold "classification_rescue_activation.json"
$activation = Get-Content -LiteralPath $activationPath -Raw | ConvertFrom-Json
if ($activation.activation_approved) {
  .\scripts\Run-ClassificationRescue.ps1
  .\scripts\Run-FinalEvaluation.ps1 `
    -WorkRoot D:\MLQuizWork -IncludeClassificationRescue
} else {
  .\scripts\Run-FinalEvaluation.ps1 -WorkRoot D:\MLQuizWork
}
```

Run only the one evaluation command reached by this branch. The orchestrator
requires the activation artifact in both cases. It rejects
`-IncludeClassificationRescue` after a negative decision; after an affirmative
decision it refuses to evaluate anything without that switch and a completed,
hash-bound rescue checkpoint and audit.

The wrapper refuses a missing, negative, or hash-mismatched activation audit,
auto-validates and passes the canonical recovery artifact when it exists,
refuses an explicitly requested but missing recovery artifact, refuses active
production/rescue processes, forces W&B off, and runs in the
foreground. It shares a process-lifetime named mutex with fixed validation, so
duplicate or cross-stage launchers cannot contend for the model/GPU. The fixed
provenance contract permits exactly one uninterrupted update-bearing
trajectory, so `-Resume` is rejected. If that trajectory is interrupted,
preserve its artifacts, abandon the rescue candidate, and evaluate only the
original checkpoints.

Python owns an exclusive `checkpoint_classification_rescue.pth.lock` file for
the process lifetime. The wrapper never deletes it automatically, because a
second near-simultaneous launcher must not be able to remove a live lock. A
hard process/power loss can leave a stale file; verify that no
`train_classification_rescue.py` process exists before removing only that exact
direct-child lock. Removing a stale lock does not authorize another rescue
launch.

Outputs are:

- `fold_0/checkpoint_classification_rescue.pth`; and
- `fold_0/checkpoint_classification_rescue.pth.audit.json`.

The `.pth` file contains the standard keys nnU-Net joint inference consumes:
`network_weights`, `trainer_name`, `init_args`, and mirroring axes. It also has
a namespaced `classification_rescue` block containing the exact schedule,
training-only history, optimizer and RNG state, the disabled-scaler/FP32
precision policy, split hashes, and
component hashes. It intentionally omits nnU-Net's top-level joint-optimizer
state, because pretending the head-only AdamW state can resume the original
all-parameter SGD trainer would be unsafe. Use a complete checkpoint normally
with `predict_joint.py` for inference; do not resume rescue optimization.

## One allowed evaluation and final choice

After all 3,750 updates finish, the affirmative branch evaluates the original
three checkpoints plus `checkpoint_classification_rescue.pth` exactly once
under the same full-volume settings and frozen equal-mean selection score. The
negative branch evaluates the original three exactly once. Validation cannot
activate, stop, extend, or restart rescue optimization. The encoder and decoder
are bit-identical to `checkpoint_final.pth`, so segmentation logits should also
be identical; verify this rather than assume it. Report every admitted result.
Training-patch diagnostics in the audit are not generalization metrics.

## Unresolved risks

- A frozen encoder may not contain subtype-separable features. This method can
  repair head optimization or collapse, but cannot create missing
  representation capacity.
- Patch-level labels remain a weak proxy for case subtype, while inference
  averages probabilities over mirror views and spatial tiles.
- Reinitialization discards any useful classification parameters learned by the
  joint run; the fixed comparison can therefore be worse and must not be hidden.
- Single-threaded augmentation and per-epoch snapshots make an interruption
  auditable, but the fixed provenance contract prohibits resuming it. Its actual
  wall-clock time has not been GPU-benchmarked; make no runtime claim before
  execution.
- The checkpoint is standard for joint inference but intentionally not
  compatible with nnU-Net's `--c` all-parameter SGD resume path.
