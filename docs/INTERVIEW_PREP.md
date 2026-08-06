# Technical Interview Preparation

This document is a frozen, evidence-backed interview aid. It is designed to help Amirfaham explain the project accurately after submission—not to memorize claims that the artifacts do not support.

## 60-second project explanation

> I extended nnU-Net v2’s 3D ResEnc M segmentation pipeline into a multi-task model for cropped pancreas CT volumes. A shared 3D encoder learns representations used by a segmentation decoder for background, pancreas, and lesion, and by a classification branch for three lesion subtypes. I preserved the provided train/validation split, used no external data or pretrained weights, tracked both tasks in W&B, and evaluated whole-pancreas Dice, lesion Dice, and macro-F1. The selected frozen-backbone classification-rescue checkpoint scored 0.9202 whole-pancreas Dice and 0.6197 lesion Dice, meeting both segmentation targets, but macro-F1 was 0.4640 and missed the 0.60 target. The principal failure was subtype discrimination: subtype 2 recall was 0.25, including six subtype-2 cases predicted as subtype 1.

These values come from the frozen 36-case full-volume evaluation and `report/report.md`; use the exact defense-sheet values below when precision matters.

## System map

```text
NIfTI CT + fixed split manifest
        │
        ▼
nnU-Net v2 planning / preprocessing / augmentation
        │
        ▼
3D ResEnc M shared encoder
        ├────────► nnU-Net segmentation decoder ─► voxel labels 0/1/2
        │
        └────────► classification branch ─────────► subtype 0/1/2
        │
        ▼
combined loss, shared gradient update, W&B tracking
        │
        ▼
fixed validation evaluation → saved checkpoint → test inference/package
```

## Concepts to be able to explain

### Why 3D rather than 2D?

The anatomy and lesion extend across slices. A 3D convolution can use through-plane context directly, while a purely 2D model treats slices independently unless context is added another way. The trade-off is much higher memory and compute use and greater sensitivity to voxel spacing.

### Why nnU-Net v2?

nnU-Net provides a strong, self-configuring medical-segmentation pipeline: dataset fingerprinting, spacing-aware preprocessing, patch planning, augmentation, sliding-window inference, and a proven encoder-decoder training recipe. The assessment also mandates it, so the goal is to extend the framework without discarding its reliable segmentation machinery.

### What “ResEnc M” means

It is nnU-Net v2’s medium residual-encoder preset. Residual blocks let each stage learn a correction to its input, improving gradient flow in a deeper encoder. Be ready to point to the exact instantiated configuration and demonstrate that the implementation did not silently substitute a smaller model.

### Why multi-task learning?

Both tasks analyze the same anatomy. Sharing an encoder can improve data efficiency and encourage features useful at voxel and case levels. It can also hurt when one task dominates the gradients or when subtype cues and boundary cues conflict; task weighting and per-task monitoring are therefore important.

### Why a separate classification branch?

Segmentation produces spatial predictions, while subtype classification needs one case-level prediction. The implemented branch concatenates global-average pooling with an eight-head learned-query cross-attention summary of the deepest 320-channel encoder map, then applies LayerNorm, a 128-unit GELU layer, dropout 0.30, and three logits.

### What the losses do

The combined objective has the general form

\[
L = \lambda_{seg}L_{seg} + \lambda_{cls}L_{cls}.
\]

Here, $\lambda_{seg}=1$ and $\lambda_{cls}=0.5$. `L_seg` is nnU-Net's deeply supervised memory-efficient soft-Dice plus voxel cross-entropy objective. `L_cls` is inverse-frequency-weighted, label-smoothed ($\epsilon=0.05$) three-class cross-entropy; crops without a lesion voxel receive a 0.25 classification reliability weight.

### Why accuracy is not enough

Lesions occupy a small fraction of voxels, so a model can achieve high voxel accuracy by predicting background. Dice focuses on overlap of the foreground region. Classification accuracy can also hide poor performance on a minority subtype, whereas macro-F1 assigns equal weight to each subtype’s F1.

## Metric definitions

### Dice similarity coefficient

For prediction set \(P\) and reference set \(G\):

\[
DSC = \frac{2|P \cap G|}{|P| + |G|}.
\]

- Whole pancreas: binarize with `label > 0`.
- Lesion: binarize with `label == 2`.
- Empty-reference behavior: Dice is 1 when both masks are empty and 0 when only one is empty. The headline metric is the unweighted mean of the 36 case-level Dice values.

### F1 and macro-F1

For each subtype, \(F1 = 2PR/(P+R)\). Macro-F1 is the unweighted mean of the three subtype F1 values. It prevents the largest class from dominating the headline classification result.

### Confusion matrix

Rows and columns reveal which subtypes are confused. In the implemented evaluator, rows are reference (ground-truth) subtypes and columns are predictions.

## High-probability interview questions

### How did you prevent data leakage?

Strong answer outline:

- retained the supplied training and validation folders as an immutable split;
- asserted that case IDs do not overlap;
- fit planning/statistical preprocessing using the intended training pipeline only, as verified in the implementation;
- never used validation cases for gradient updates;
- used test inputs only for final inference and never for selection.

Update this answer if the actual implementation differs.

### How did you address class imbalance?

Answer from the final configuration, not the list of possibilities. Discuss both:

- **voxel imbalance:** background versus pancreas versus small lesion; nnU-Net patch sampling and segmentation loss;
- **case imbalance:** subtype counts `62/106/84` in training produce inverse-frequency loss weights `1.35484/0.79245/1.0`.

Then cite per-class F1 and whether the strategy helped. Do not claim causality without a comparison.

### How did you control overfitting?

Possible mechanisms include on-the-fly augmentation, weight decay, dropout in the classification branch, a constrained head, early stopping/model selection, and monitoring the training-validation gap. State only mechanisms actually enabled. Also acknowledge that repeated decisions on one small validation set can overfit the validation split.

### How did you balance the tasks?

Explain the selected loss weights, their scale, whether both gradients reached the encoder, and what the W&B curves showed. A good answer acknowledges that equal numeric weights do not imply equal gradient influence.

### Why take classification features from that encoder level?

Deeper features have larger receptive fields and stronger semantic abstraction, useful for a global subtype decision; shallower features retain detail but cost more to aggregate. This model uses the deepest 320-channel bottleneck and combines global-average and learned-query attention summaries. The rationale is complementary global/focal evidence, not a claim of superiority without an ablation.

### How did you handle the abnormal mask value?

The audit found that 214 of 288 labeled masks decoded a near-integer pancreas value (`1.0000153`) instead of exact integer `1`. The source files were not edited. `scripts/prepare_dataset.py` maps only values within `1e-3` of allowed labels to `uint8`, preserves and verifies image geometry, and writes `data_audit.json`, `split_manifest.json`, and classification manifests before nnU-Net integrity checking.

### Why is lesion Dice much harder than whole-pancreas Dice?

Lesions are smaller and more heterogeneous. A few voxels of boundary error cause a larger proportional Dice penalty, and small lesions may be missed entirely. Whole-pancreas Dice combines labels 1 and 2, so a lesion mislabeled as pancreas can still count as correct whole-organ foreground.

### What would you do with more time?

Prioritize evidence-based improvements:

1. repeat training across seeds or cross-validation to quantify variance;
2. tune task balancing without repeatedly overfitting the held-out split;
3. compare classification pooling methods in a controlled ablation;
4. inspect lesion-size-stratified failure cases;
5. evaluate calibration and confidence;
6. validate on an external, institutionally distinct dataset if permitted.

### Can this be used clinically?

No. It is a take-home prototype on de-identified cropped ROIs, with one small fixed validation split and no external validation, calibration study, reader study, or deployment risk analysis.

### What did the AI do, and what did you do?

Use the final facts from [AI_WORKFLOW.md](AI_WORKFLOW.md). A truthful concise answer:

> OpenAI Codex generated a substantial majority of the initial implementation and documentation and helped test and monitor the workflow. I owned the project direction, quality priorities, access, consequential decisions, review, and final submission. I did not treat generated code as automatically correct: we used data audits, contract tests, smoke tests, saved configurations, W&B evidence, and final artifact validation. The honest estimate is 85–95% of the initial repository content, based on file-level provenance rather than a post-formatting line count.

## Results defense sheet

Frozen from the selected-candidate artifacts and final report.

| Question | Evidence-backed answer |
|---|---|
| Which checkpoint was selected, and why? | `checkpoint_classification_rescue.pth` (joint epoch 200 plus 30 frozen-backbone head-only epochs). Its equal-weight mean of whole Dice, lesion Dice, and macro-F1 was `0.6679416738`, the highest of four candidates. Relative to `checkpoint_final.pth`, it raised macro-F1 from `0.1333333333` to `0.4639934052` while segmentation stayed numerically identical because the encoder and decoder were frozen. |
| Whole-pancreas Dice? | Mean `0.9201588643`, sample SD `0.0357843891`, 95% bootstrap CI `[0.9078962203, 0.9308140254]`; the `>=0.90` target was met. |
| Lesion Dice? | Mean `0.6196727520`, sample SD `0.3206719418`, 95% bootstrap CI `[0.5148320706, 0.7165657179]`; the `>=0.27` target was met. |
| Macro-F1? | `0.4639934052`, 95% bootstrap CI `[0.2795513293, 0.6314441497]`; the point estimate missed the `>=0.60` target. Accuracy was `0.5000`. |
| Per-subtype precision/recall/F1? | Subtype 0 (support 9): `0.4444/0.4444/0.4444`; subtype 1 (support 15): `0.5000/0.7333/0.5946`; subtype 2 (support 12): `0.6000/0.2500/0.3529`. |
| Strongest classification confusion? | With rows as references and columns as predictions, the matrix was `[[4,5,0],[2,11,2],[3,6,3]]`. The largest directional error was subtype 2 -> 1 (six cases), followed by subtype 0 -> 1 (five); subtype 1 was overpredicted 22 times for 15 references. |
| Best qualitative case? | The two rule-selected strong cases were `quiz_1_164` (whole `0.9043`, lesion `0.9245`, subtype `1 -> 1`) and `quiz_0_184` (whole `0.9405`, lesion `0.9232`, subtype `0 -> 1`). The latter shows that excellent masks do not guarantee correct subtype classification. |
| Most informative failure? | Both rule-selected weak cases had zero lesion overlap: `quiz_2_191` (whole `0.7935`, subtype `2 -> 0`) had a spatially remote 20,223-voxel prediction versus a 4,248-voxel reference; `quiz_1_227` (whole `0.9347`, subtype `1 -> 0`) predicted 287 lesion voxels versus 1,724 reference voxels. |
| Peak training VRAM and runtime? | No instrumented production-training peak was recorded. The planned-patch CUDA preflight measured `6,159 MiB` allocated and `6,716 MiB` reserved; a production `nvidia-smi` sample was about `6,849 MiB` steady usage, not a peak. The 200 joint epochs took `6:48:03.248` wall time (`6:32:30.457` summed epoch compute); the 30-epoch rescue added 3,750 updates and `1,754.212 s` summed epoch compute. |
| Test inference runtime? | The matched 36-case validation run took `112.306 s` (`3.1196 s/case`) with peak CUDA allocation/reservation `2,173.889/2,492 MiB`. Fresh inference over all 72 test cases took `248.115 s` total, or `3.4460 s/case`, with `2,173.272/2,492 MiB` peak allocation/reservation. |
| Largest limitation? | The evidence comes from one fixed validation split of only 36 cases (nine subtype-0 cases), with no repeated seeds or cross-validation, so optimization variance and split sensitivity are not quantified. There is also no external validation. |

Canonical sources are `fixed_validation/checkpoint_selection.json`, the selected
checkpoint's `metrics.json`, `case_metrics.csv`, and `runtime.json`,
`selected_test/runtime.json`, `final_evidence_summary.json`, and
`report/report.md`. The qualitative cases were selected by the frozen lesion-Dice
rule, not by visual appeal.

## Code walk-through checklist

Before an interview, be able to locate and explain:

- [ ] dataset conversion and split manifest;
- [ ] classification target lookup;
- [ ] shared encoder and both output branches;
- [ ] combined loss and task weights;
- [ ] W&B logging calls;
- [ ] validation aggregation and empty-lesion rule;
- [ ] checkpoint selection logic;
- [ ] test prediction naming and CSV generation;
- [ ] archive validator;
- [ ] one test that caught or could catch a serious bug.

## Five-minute whiteboard rehearsal

1. Draw the two-branch architecture.
2. Write the combined loss.
3. Define whole-pancreas Dice, lesion Dice, and macro-F1.
4. Explain the fixed split and no-external-data constraint.
5. Give one successful decision, one failure, and one next experiment.

If any of these requires reading verbatim from the report, revisit the corresponding section and explain it in your own words.
