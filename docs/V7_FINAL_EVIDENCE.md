# V7 final evidence guide

This page is the short reviewer-facing map for the V7 submission. It connects
each final claim to the experiment or verification artifact that supports it.

## Final accuracy decision

| Metric | Threshold | Independently verified V7 value | Decision |
|---|---:|---:|---:|
| Whole-pancreas Dice | >= 0.91 | 0.9201569021 | Pass |
| Lesion Dice | >= 0.31 | 0.6196343545 | Pass |
| Three-class macro-F1 | >= 0.70 | 0.7445103206 | Pass |

The confusion matrix (reference rows, prediction columns) is
`[[6,2,1],[0,13,2],[1,3,8]]`. The classifier was fitted on 252 training cases
and zero validation cases. Validation was used to select the deployment stage,
view, and acceptable scale, so the reported macro-F1 is a development-set
result rather than an untouched external estimate.

Primary evidence:

- `docs/evidence/v7/optimized_validation_metrics.json`
- `docs/evidence/v7/stage1_classifier_study.json`
- `models/v7/classifier_stage1_view6_scale1.joblib`
- classifier SHA-256
  `bbdb0fc79b35cfc81400550ad558636be6c15663f623b230813ddcb46264d0df`

## Why the change works

The segmentation encoder is a hierarchy. Deep features are useful for locating
pancreas and lesion, but they can discard fine texture that separates tumour
subtypes. Frozen probes found stronger subtype signal before the bottleneck.
V7 therefore averages the 64 channels at encoder stage 1 and uses a small
shrinkage-LDA classifier. Shrinkage stabilizes covariance estimation when the
training set is small, and the Ledoit--Wolf rule computes its strength from the
training data.

The final classifier uses mirror view 6 (axes 2 and 3). Reduced spatial scales
0.25, 0.375, 0.5, and 0.625 all failed the accuracy gate and were rejected.
This negative result supports preserving full spatial resolution for the
classification descriptor.

## Speed decision

The speed requirement is met by the final complete paired audit. Three fresh
processes per arm ran all 72 test cases in balanced `SCCSCS` order. Stock
averaged 259.5160 seconds and V7 averaged 231.2600 seconds, giving a 10.8880%
runtime reduction. The candidate executed segmentation, the selected fitted
classifier, and subtype CSV export while retaining TTA and step size 0.5.

All candidate repeats wrote 72 masks and 72 subtype rows. Their subtype outputs
were identical to one another and to the selected submission. The cross-arm
comparison found 968 differing voxels out of 141,878,022 (0.000682%); geometry,
dtype, repeat stability, whole-pancreas agreement, and lesion agreement passed
the declared bounds. The raw audit has SHA-256
`954c8a2b093140cd9b244a1365b41fbc74bbfcf188327da441b1d81cf5dee8bc`.

Primary evidence: `docs/evidence/v7/inference_speed_audit.json`.

## W&B records

Three W&B records organize the fine-tuning, validation, and inference evidence.
All were synchronized and remotely verified:

| Run ID | Purpose | Evidence source |
|---|---|---|
| `uzc4elyc` | Fine-tuning curves | Archive of 21 saved training events (`live_training_run=false`) |
| `wrd1f1c8` | Independent validation | Metrics computed from saved validation outputs |
| `4wb71b3i` | Initial complete inference audit | Baseline for the optimization iteration |

All three are remotely verified in the `finished` state. URLs and the retained
local sync provenance are in `docs/evidence/v7/wandb_runs.json`.

## Attribution

Amirfaham Fallahpour owns and directed the project. He set the research
questions, required evaluation against both the minimum and higher-tier goals,
identified classification imbalance and representation quality as priorities,
completed the V7 GPU training, and made the final experiment and submission
decisions. OpenAI Codex translated that direction into substantial
implementation, debugging, experiment execution, verification, and report
drafting. Final claims are tied to saved evidence and remain Amirfaham's
responsibility to review and submit.
