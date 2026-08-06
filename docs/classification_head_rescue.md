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
hyperparameter search and no second rescue attempt.

This contingency was prospectively frozen after online patch diagnostics had
suggested classification collapse, but before any full-volume fixed-validation
evaluation. Those observed online diagnostics must be disclosed; they are not
claimed to be unseen.

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

## Exact invocation

After production exits, first create the machine-readable activation artifact
from `checkpoint_final.pth`. This script reads only its saved
`train_cls_losses` and `train_cls_accuracy`, computes both frozen audits, and
binds the decision to the checkpoint SHA-256:

Do **not** run the current three-candidate `Run-FinalEvaluation.ps1` before this
gate. If rescue activates, finish it first and then evaluate all four candidates
in one extended fixed-validation pass. If it does not activate, proceed with
the original three-candidate workflow.

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
$fold = "D:\MLQuizWork\nnUNet_results\Dataset501_PancreasMultitask\" +
  "nnUNetTrainerPancreasMultiTask__nnUNetResEncUNetMPlans__3d_fullres\fold_0"
$python = "D:\MLQuizWork\.venv\Scripts\python.exe"
& $python .\scripts\audit_classification_rescue_activation.py `
  --checkpoint (Join-Path $fold "checkpoint_final.pth") `
  --output (Join-Path $fold "classification_rescue_activation.json")

.\scripts\Run-ClassificationRescue.ps1
```

The wrapper refuses a missing, negative, or hash-mismatched activation audit,
refuses active production/rescue processes, forces W&B off, and runs in the
foreground. If interrupted after an epoch checkpoint is committed, continue
from its embedded AdamW, AMP, and RNG state:

```powershell
.\scripts\Run-ClassificationRescue.ps1 -Resume
```

Outputs are:

- `fold_0/checkpoint_classification_rescue.pth`; and
- `fold_0/checkpoint_classification_rescue.pth.audit.json`.

The `.pth` file contains the standard keys nnU-Net joint inference consumes:
`network_weights`, `trainer_name`, `init_args`, and mirroring axes. It also has
a namespaced `classification_rescue` block containing the exact schedule,
training-only history, optimizer/scaler state, RNG state, split hashes, and
component hashes. It intentionally omits nnU-Net's top-level joint-optimizer
state, because pretending the head-only AdamW state can resume the original
all-parameter SGD trainer would be unsafe. Resume this checkpoint only through
the rescue command; use it normally with `predict_joint.py` for inference.

## One allowed evaluation and final choice

After all 3,750 updates finish, evaluate the original three checkpoints plus
`checkpoint_classification_rescue.pth` exactly once under the same full-volume
settings and frozen equal-mean selection score. Validation cannot activate,
stop, extend, or restart rescue optimization. The encoder and decoder are
bit-identical to `checkpoint_final.pth`, so segmentation logits should also be
identical; verify this rather than assume it. Report all four results.
Training-patch diagnostics in the audit are not generalization metrics.

## Unresolved risks

- A frozen encoder may not contain subtype-separable features. This method can
  repair head optimization or collapse, but cannot create missing
  representation capacity.
- Patch-level labels remain a weak proxy for case subtype, while inference
  averages probabilities over mirror views and spatial tiles.
- Reinitialization discards any useful classification parameters learned by the
  joint run; the fixed comparison can therefore be worse and must not be hidden.
- Single-threaded augmentation makes interruption/resume semantics auditable,
  but transform-local caches and backend flags are not checkpointed, so a
  resumed run is valid but not promised bit-exact. Its actual wall-clock time
  has not been GPU-benchmarked; make no runtime claim before execution.
- The checkpoint is standard for joint inference but intentionally not
  compatible with nnU-Net's `--c` all-parameter SGD resume path.
