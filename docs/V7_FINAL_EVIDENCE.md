# V7 final evidence guide

This page is the short reviewer-facing map for the V7 submission. It separates
recovered artifacts from work independently executed during finalization.

## Final accuracy decision

| Metric | Threshold | Recomputed V7 value | Decision |
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

The speed requirement is not met or claimed. A complete three-run-per-arm local
benchmark on the RTX 4060 measured 235.0417 seconds for stock and 280.7237
seconds for the first complete candidate. The final optimized candidate later
profiled at 208.0829 seconds for 72 cases, but that engineering profile is not
a completed paired ABBA gate statistic.

The recovered H100 file reports `+11.1698%`. It is ineligible because the
candidate arm omitted required classifier execution and output. Keeping this
as a rejected result prevents a fast but incomplete pipeline from being called
equivalent to stock plus classification.

Primary evidence: `docs/evidence/v7/inference_speed_audit.json`.

## W&B records

The finalization created three genuine W&B runs offline first and then
synchronized and remotely verified them:

| Run ID | Purpose | Provenance |
|---|---|---|
| `uzc4elyc` | Fine-tuning curves | Explicit replay of recovered saved events; not live training |
| `wrd1f1c8` | Independent validation | Newly recomputed metrics |
| `4wb71b3i` | Speed/equivalence audit | Complete local negative result plus rejected H100 result |

All three are remotely verified in the `finished` state. URLs and the retained
local sync provenance are in `docs/evidence/v7/wandb_runs.json`.

## Attribution

Amirfaham Fallahpour owns and directed the project, completed the V7 GPU
training, supplied the saved deliverable, set the quality and submission goals,
and is responsible for reviewing and submitting the final result. OpenAI Codex
provided substantial implementation, reconstruction, debugging, experiment,
verification, and writing assistance. The repository intentionally describes
this as candidate-directed AI-assisted work and does not relabel reconstructed
logs or AI-written code as something else.
