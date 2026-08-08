# Higher-tier branch protocol

The project targeted both the undergraduate minimum and the master/PhD criteria
from the outset through a staged plan. This branch protocol was frozen at
2026-08-06T14:05:57Z after the baseline stage was complete and its
fixed-validation results were known. The overall higher-tier objective was
preplanned; the exact v5 candidate set and decision rules below were fixed at
this later branch boundary.

## Immutable fallback

The validated baseline remains available outside the repository at
`D:\MLQuizWork\baseline_20260806_509cbe2`:

- PDF SHA-256:
  `90d68697d6330d5124f1a2533f3785033643a0985fe3ce2813b0d90a0a04fd03`;
- ZIP SHA-256:
  `5de55f4ccc1eea78ef8974d0f362039523404a1d6315d06d0ec41ec8f0d08391`;
- public repository baseline commit: `509cbe2`.

No upgrade step may overwrite this directory. If the upgrade fails, times out,
or cannot be fully audited, these artifacts remain the submission.

## Objectives

The assignment's master/PhD expectations are treated as conjunctive:

- whole-pancreas Dice at least `0.91`;
- lesion Dice at least `0.31`;
- three-class macro-F1 at least `0.70`;
- at least 10% faster inference through a method other than disabling TTA or
  increasing sliding-window step size.

The baseline already exceeds both segmentation thresholds. The upgrade will
therefore keep the selected segmentation encoder and decoder frozen and focus
on case-level subtype classification plus an inference-engine optimization.

## Development-data boundary

Architecture choice, feature choice, hyperparameters, epoch count, stopping,
and classifier selection may use only the 252 supplied training cases. A
deterministic stratified inner split or stratified cross-validation must be
saved before fitting. Official validation images, masks, labels, case metrics,
or aggregate metrics must not be read by development scripts or used for
candidate selection.

The baseline official validation result is acknowledged as the motivation for
this post-hoc extension, but it is not a tuning signal. Any process that reads
the 36 official validation targets before the candidate is locked invalidates
the upgrade attempt.

## Classification candidate

The candidate must retain the trained nnU-Net v2 3D ResEnc M shared encoder and
must not modify the selected segmentation encoder or decoder. The intended
family is an inference-matched, case-level head using frozen shared-encoder
features, optionally with lesion-aware pooling driven by the model's own
segmentation output. Ground-truth masks may support training-data analysis but
must not create a train/inference feature mismatch in the locked method.

All candidate families and their bounded search space must be written to a
machine-readable lock artifact before the search begins. The final method is
selected exclusively by a predeclared train-only macro-F1 criterion, then
refitted according to that lock. Seeds, case IDs, software versions, feature
hashes, fitted parameters, and out-of-fold predictions must be retained.

## Official validation gate

After code, configuration, checkpoint, and hashes are frozen, exactly one new
full-volume evaluation is permitted on all 36 official validation cases.
Results are handled as follows:

1. A PhD-level claim requires whole Dice `>= 0.91`, lesion Dice `>= 0.31`, and
   macro-F1 `>= 0.70`.
2. A new classifier replaces the baseline classifier only if its macro-F1 is
   strictly greater than `0.46399340516987575` and all submission contracts
   pass. Otherwise the immutable baseline remains authoritative.
3. Results below `0.70` must be reported as below the PhD classification
   expectation even if they improve the baseline.
4. No second official-validation classifier iteration is permitted.

## Inference-speed gate

The speed method must keep TTA, tile step, checkpoint, inputs, output semantics,
and device fixed between baseline and optimized measurements. Development may
use a fixed train-only timing subset. The final paired benchmark must record
warm-up policy, case IDs, repeats, wall time, CUDA memory, software/hardware,
and output agreement.

The speed claim is accepted only if the optimized end-to-end mean runtime is at
least 10% lower than the batch-one implementation and predicted masks and
subtype decisions agree under a predeclared numerical-equivalence policy. An
OOM fallback may improve robustness but cannot be counted as the speed result.

## Finalization rule

New deliverables are eligible only after complete tests, leakage/provenance
audits, W&B publication, report regeneration, 72-case archive validation,
public-link checks, and an independent final review. Any incomplete candidate
is excluded; only a fully validated artifact set may replace the baseline.
