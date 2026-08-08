# V6 validation-guided optimization protocol

Status: prospective for every eligible V6 training run and validation attempt.

## Why V6 is legitimate

The assignment explicitly permits the supplied validation split for debugging and
monitoring, while prohibiting its use as training data. V5 adopted a stricter
one-evaluation rule. That rule remains true for V5, whose artifacts and consumed
ledgers are immutable, but it does not prohibit a separately named V6 development
lineage. V6 therefore amends only the future-development policy. It does not reopen,
rewrite, or relabel any V5 run.

V6 is openly validation-guided. Its 36-case validation result is a development and
model-selection result, not an untouched test estimate. The 72 test cases remain
blind because no test targets are supplied and test predictions cannot influence
training.

## Immutable starting point and known information

V6 starts from Git tag `v5-evidence-final` at commit
`e8b54793f14b4a3f68841ac0948e68ceeb2784cf`. The V5 official result is known:
whole-pancreas Dice 0.92016118, lesion Dice 0.61966239, and subtype macro-F1
0.52541507. Its confusion matrix shows subtype 1 as the main error source. Before
this protocol was written, one explicitly exploratory diagnostic searched additive
class offsets on the saved V5 validation probabilities. Even direct validation
optimization reached only 0.63966184 macro-F1; it cannot explain or support a 0.70
claim and is ineligible as a final rule. A previously locked train-only classical
feature search was also launched as a diagnostic. These facts are part of the V6
design history and will not be hidden.

All V5 model files, predictions, reports, packages, hashes, and ledgers remain
unchanged. V6 writes only below `D:\MLQuizWork\phd_upgrade_v6` and uses new config,
attempt, evaluation, speed, and package records.

## Data boundary

- Exactly 252 supplied training cases may contribute gradients, fitted
  normalization, sampling weights, calibration, decision offsets, or learned
  parameters.
- The 36 supplied validation cases may be evaluated in `eval`/`no_grad` mode and
  may guide generic architecture, optimization, early-stopping, and candidate
  selection decisions.
- Validation images, masks, and subtype targets may never appear in an optimizer
  batch. Validation ground-truth masks may be read only by the unchanged evaluator.
- The 72 test images may be inferred only after a final candidate is selected.
  Test outputs may not trigger retraining.
- Case identifiers, paths, directory names, file enumeration order, and the
  subtype-bearing filename prefix may not enter a model matrix or decision rule.
- No external dataset or externally pretrained weight is permitted. Weights learned
  solely from the supplied training cases, including the existing segmentation
  checkpoint, are allowed and must be disclosed.
- Every loader must enforce the 252-case training allowlist because the physical
  preprocessed directory also contains validation cases.

## Scientific objective

The primary gap is subtype classification. Segmentation already exceeds the
higher-tier thresholds by a large margin, so every classifier change must retain
whole-pancreas Dice at least 0.91 and lesion Dice at least 0.31. The classification
target is three-class macro-F1 at least 0.70. The engineering target is complete
multitask inference at least 10% faster than an otherwise matched stock nnU-Net
run, without using TTA removal or a larger tile step as the speed method.

The final system must retain the mandated nnU-Net v2 3D ResEnc M segmentation
network, a shared 3D encoder, and a separate classification head. A candidate that
achieves a high score through identifiers, case-specific corrections, validation
training, ground-truth-mask inference features, or hidden exclusions is invalid
regardless of its metric.

## Bounded experiment ladder

Before each update-bearing run, an attempt JSON must freeze the exact architecture,
trainable modules, loss, sampler, augmentation, optimizer, schedule, seed, split,
stopping rule, and expected outputs.

The eligible ladder contains at most four supplied-validation attempts:

1. A strongly regularized, coordinate-aware, multiscale lesion-aware case head that
   corrects V5's missing positional information and high resubstitution gap.
2. If frozen representations remain limiting, a case-level lesion-aware fine-tune
   of the classification head and the smallest prospectively specified terminal
   encoder block set. Segmentation supervision or a frozen-weight penalty must guard
   the shared representation.
3. If minority recall remains the limiting error, one predeclared alternative
   imbalance treatment: either natural sampling plus weighted/focal loss or balanced
   sampling plus unweighted cross-entropy, never both forms of class correction.
4. A final train-only ensemble or decision-rule candidate using cross-fitted training
   predictions. No validation-fitted per-case rule is allowed.

Cheap heads must use repeated stratified train-only OOF evaluation. An encoder
fine-tune must use fold-isolated subtype supervision when time permits; if the
deadline permits only the assignment's supplied train/validation workflow, that
limitation must be stated and the validation score must not be called independent.
All candidates report macro-F1, per-class precision/recall/F1, confusion matrices,
resubstitution gaps, and seed/fold variability. Candidate count and failed attempts
remain visible.

An attempt may consume validation only after it has a complete checkpoint, leakage
audit, train-only metrics, deterministic inference configuration, and a written
hypothesis. Predictions are hashed before references are evaluated. The exposure
registry records every attempt, including failures. A validation result may motivate
the next generic attempt, but not a hard-coded case correction.

## Selection and stopping

V6 replaces V5 only if all output/leakage contracts pass, both segmentation gates
pass, and validation macro-F1 is strictly greater than 0.52541507. Among eligible
attempts, maximize validation macro-F1; values within 0.01 are tied and are resolved
by higher train-only OOF macro-F1, then higher minimum class recall, then lower
runtime and parameter count.

Predictive iteration stops when any condition is reached:

- macro-F1 is at least 0.70 and both segmentation thresholds pass;
- four validation attempts have been consumed;
- two consecutive attempts improve macro-F1 by less than 0.01;
- train-only evidence does not justify another candidate; or
- 09:00 America/Toronto on 2026-08-07 is reached, reserving three hours for final
  verification, report generation, packaging, and upload.

The point estimate determines threshold status. Bootstrap confidence intervals are
descriptive because validation is adaptively reused.

## Speed and delivery

Predictive weights and decision rules are frozen before the final speed gate. Up to
three implementation variants may be developed on a train-only timing subset.
Permitted strategies include exact graph/pipeline optimization, activation reuse,
batched or overlapped preprocessing/export, compilation, or an explicitly validated
reduced-compute method. Disabling TTA or increasing step size cannot be credited.

The final benchmark uses matched inputs, hardware, TTA, tile step, precision,
preprocessing, export semantics, and fresh-process ABBA ordering. It reports the
complete workload, not a selectively timed kernel. Segmentation geometry/value
checks and subtype decisions must remain valid; any numerical tolerance is declared
before the benchmark.

Only the selected model receives one final 72-case test inference and package. The
delivery ZIP must contain exactly 72 masks and one 72-row subtype CSV and must pass
the independent validator. V5 and earlier packages are retained under their
original hashes rather than overwritten.

## Claim boundary

"Best model" means the strongest valid candidate within this disclosed, bounded V6
development process. It does not mean globally optimal, externally validated, or
clinically ready. The master/PhD performance bar is claimed only if all three metric
thresholds are met; the complete higher-tier bar additionally requires the speed
gate. If a threshold is missed, the report states that directly.
