# Technical Interview Preparation

This document is intentionally a scaffold until the implementation and validation results are frozen. It is designed to help Amirfaham explain the project accurately after submission—not to memorize claims that the artifacts do not support.

## 60-second project explanation

> I extended nnU-Net v2’s 3D ResEnc M segmentation pipeline into a multi-task model for cropped pancreas CT volumes. A shared 3D encoder learns representations used by a segmentation decoder for background, pancreas, and lesion, and by a classification branch for three lesion subtypes. I preserved the provided train/validation split, used no external data or pretrained weights, tracked both tasks in W&B, and evaluated whole-pancreas Dice, lesion Dice, and macro-F1. The main engineering challenges were correcting near-integer mask labels safely, fitting a 3D network into limited GPU memory, balancing two task losses, and avoiding validation leakage. The measured results were `PENDING`; the most important failure mode was `PENDING`.

Do not fill the final two fields from memory. Copy them from the frozen result artifacts and `report/report.md`.

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

Segmentation produces spatial predictions, while subtype classification needs one case-level prediction. The branch aggregates encoder features across space and maps them to three logits. The final pooling design is `PENDING`; explain only the version actually used.

### What the losses do

The combined objective has the general form

\[
L = \lambda_{seg}L_{seg} + \lambda_{cls}L_{cls}.
\]

`L_seg` trains voxel predictions, usually combining overlap- and distribution-based terms in nnU-Net. `L_cls` trains the subtype logits, usually using cross-entropy or an imbalance-aware variant. The reported weights and losses are `PENDING` and must match the configuration.

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
- Empty-reference behavior and aggregation: `PENDING`; state the exact implemented rule.

### F1 and macro-F1

For each subtype, \(F1 = 2PR/(P+R)\). Macro-F1 is the unweighted mean of the three subtype F1 values. It prevents the largest class from dominating the headline classification result.

### Confusion matrix

Rows and columns reveal which subtypes are confused. Always state which axis is ground truth; ours is `PENDING until evaluation code is frozen`.

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
- **case imbalance:** subtype counts `62/106/84` in training; `PENDING selected strategy` such as weighted loss or balanced sampling.

Then cite per-class F1 and whether the strategy helped. Do not claim causality without a comparison.

### How did you control overfitting?

Possible mechanisms include on-the-fly augmentation, weight decay, dropout in the classification branch, a constrained head, early stopping/model selection, and monitoring the training-validation gap. State only mechanisms actually enabled. Also acknowledge that repeated decisions on one small validation set can overfit the validation split.

### How did you balance the tasks?

Explain the selected loss weights, their scale, whether both gradients reached the encoder, and what the W&B curves showed. A good answer acknowledges that equal numeric weights do not imply equal gradient influence.

### Why take classification features from that encoder level?

Deeper features have larger receptive fields and stronger semantic abstraction, useful for a global subtype decision; shallower features retain detail but cost more to aggregate. The selected feature level/pooling is `PENDING` and should be defended with memory constraints and observed behavior.

### How did you handle the abnormal mask value?

The audit found that 214 of 288 labeled masks included a near-integer pancreas value (`1.0000153`) instead of exact integer `1`. The source files were not edited. A deterministic conversion step maps only verified label values to `uint8`, preserves image geometry, logs affected cases, and runs nnU-Net integrity checks. Final artifact names/commands are `PENDING`.

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

> OpenAI Codex generated a substantial majority of the initial implementation and documentation and helped test and monitor the workflow. I owned the project direction, quality priorities, access, consequential decisions, review, and final submission. I did not treat generated code as automatically correct: we used data audits, contract tests, smoke tests, saved configurations, W&B evidence, and final artifact validation. The exact final contribution estimate was `PENDING`.

## Results defense sheet

Fill only after final evaluation.

| Question | Evidence-backed answer |
|---|---|
| Which checkpoint was selected, and why? | `PENDING` |
| Whole-pancreas Dice? | `PENDING` |
| Lesion Dice? | `PENDING` |
| Macro-F1? | `PENDING` |
| Per-subtype precision/recall/F1? | `PENDING` |
| Strongest classification confusion? | `PENDING` |
| Best qualitative case? | `PENDING` |
| Most informative failure? | `PENDING` |
| Peak training VRAM and runtime? | `PENDING` |
| Test inference runtime? | `PENDING` |
| Largest limitation? | `PENDING` |

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
