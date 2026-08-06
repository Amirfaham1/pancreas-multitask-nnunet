---
title: "Multi-Task 3D Pancreas CT Segmentation and Subtype Classification"
title-running: "Multi-Task Pancreas CT Analysis"
author:
  - "Amirfaham Fallahpour"
author-running: "A. Fallahpour"
institute: "Undergraduate Neuroscience Specialist, University of Toronto Scarborough (UTSC), Toronto, Canada"
date: "August 2026"
geometry: margin=1in
fontsize: 10pt
colorlinks: true
linkcolor: blue
urlcolor: blue
abstract: |
  This work develops a joint system for pancreas/lesion segmentation and three-class subtype classification from cropped three-dimensional computed-tomography (CT) regions of interest. The mandatory nnU-Net v2 3D Residual Encoder Medium (ResEnc M) network is retained as the segmentation backbone. Its encoder is shared with a classification branch that combines global average pooling with learned-query cross-attention over the deepest feature map. A deterministic preparation pipeline preserves the supplied 252/36 training/validation split and repairs a source-label representation defect without modifying the original files: 18,620,040 nominal pancreas voxels in 214 of 288 labelled masks were decoded as `1.0000152587890625` and were safely mapped to integer label `1`. Planning statistics were computed from the 252 training cases only.

  Joint optimization uses nnU-Net's deeply supervised Dice-plus-cross-entropy segmentation objective and a class-weighted, label-smoothed cross-entropy classification objective. The classification term is downweighted for patches without visible lesion, and foreground oversampling is increased to 50%. A separate evaluator reports unweighted case-level whole-pancreas Dice, lesion Dice, macro-F1, per-class metrics, distributions, and case-bootstrap uncertainty in accordance with the task-oriented principles of Metrics Reloaded. The final fixed-split results are mean whole-pancreas Dice `PENDING_WHOLE_DICE`, mean lesion Dice `PENDING_LESION_DICE`, and macro-F1 `PENDING_MACRO_F1`. The project uses substantial OpenAI Codex assistance as requested in the brief, with explicit attribution, candidate review, and artifact-based verification.
keywords:
  - "pancreas CT"
  - "medical image segmentation"
  - "multi-task learning"
  - "nnU-Net"
  - "cross-attention"
---

<!--
FINALIZATION CONTRACT
1. Keep the visible DRAFT warning until every PENDING token is resolved.
2. Populate results from the frozen JSON/CSV evaluator artifacts, never from dashboard impressions.
3. Cross-check all methods against the reported Git commit, checkpoint, plans.json, and debug.json.
4. Replace figure blocks with exported figures and verify that every figure is cited in the text.
5. The final LLNCS document must be at least eight pages excluding references.
6. Export as Amirfaham_Fallahpour_results.pdf unless the employer specifies another filename.
-->

> **DRAFT — NOT READY FOR SUBMISSION.** Every `PENDING_*` field denotes a result, link, figure, hash, or final review item that does not yet exist. No placeholder is an estimate.

**Public repository:** [github.com/Amirfaham1/pancreas-multitask-nnunet](https://github.com/Amirfaham1/pancreas-multitask-nnunet)

**Weights & Biases run:** [public run `hrs05iyx`](https://wandb.ai/amirfahamfallahpour1379-university-of-toronto/pancreas-multitask-amirfaham-fallahpour/runs/hrs05iyx)

**Reported Git commit:** `PENDING_GIT_COMMIT`

# Introduction

Pancreatic CT analysis in this take-home combines two prediction problems at different spatial scales. Semantic segmentation assigns background, normal-pancreas, or lesion status to every voxel. Subtype classification assigns one of three labels to the complete cropped CT case. A useful model must therefore learn both fine spatial boundaries and a global representation that separates disease subtypes. The supplied images also vary in voxel spacing and appearance, making data-adaptive preprocessing and augmentation important.

The brief mandates nnU-Net v2's 3D ResEnc M configuration, a shared encoder, separate segmentation and classification outputs, W&B tracking for both tasks, explicit classification-imbalance and overfitting strategies, Metrics Reloaded-aligned evaluation, public code, an AI-workflow description, and predictions for 72 test cases. External datasets, pretrained weights, and optimization on the supplied validation images are prohibited. The required undergraduate validation targets are whole-pancreas Dice at least 0.90, lesion Dice at least 0.27, and three-class macro-F1 at least 0.60. The more demanding inference-speed experiment is required only for master/PhD candidates.

This submission is evaluated against the undergraduate targets; architecture, leakage-control, reproducibility, and reporting requirements are otherwise unchanged. Under the deadline, the priority order was: (i) correct data and split handling, (ii) a genuine ResEnc M multi-task implementation, (iii) defensible validation and complete test artifacts, and only then (iv) optional speed work.

The main technical contributions are:

1. a non-destructive NIfTI audit/conversion pipeline with strict pairing, geometry, split-overlap, and categorical-label checks;
2. a ResEnc M extension with a shared residual encoder, standard nnU-Net decoder, and hybrid global-average/learned-query cross-attention classification branch;
3. an imbalance-aware joint objective and patch-evidence weighting strategy compatible with nnU-Net's patch training;
4. explicit joint sliding-window inference that aggregates segmentation logits and subtype probabilities across mirror views, tiles, and folds without hidden mutable state;
5. an evaluator and submission validator independent of the training metrics; and
6. an auditable AI-assisted workflow that distinguishes candidate ownership from AI-generated implementation.

# Data, integrity, and split control

## Dataset composition

The provided data are de-identified, cropped 3D pancreas CT regions. Cropping makes training possible on modest GPUs, but also removes the full-scan detection/localization problem. Each labelled image is named `quiz_<subtype>_<case>_0000.nii.gz`, and its mask has the same stem without `_0000`. Test filenames omit the hidden subtype. The masks define background `0`, normal pancreas `1`, and pancreatic lesion `2`.

The assessment package does not provide scanner models, acquisition protocols, contrast status, patient demographics, annotation procedure, annotator expertise, or inter-rater measurements. These missing provenance variables limit subgroup and external-validity analysis. No registration is applied because each case contains one CT channel with a paired mask already on the same verified grid. Published pancreatic-CT systems demonstrate the broader promise of deep learning in this domain [2], but their populations and acquisition settings do not establish transfer to this dataset.

| Supplied split | Subtype 0 | Subtype 1 | Subtype 2 | Total |
|---|---:|---:|---:|---:|
| Training | 62 | 106 | 84 | 252 |
| Validation | 9 | 15 | 12 | 36 |
| Test | — | — | — | 72 |

: Supplied case counts and subtype composition. {#tbl:split}

Subtype 1 forms 42.1% of the training set, subtype 2 forms 33.3%, and subtype 0 forms 24.6%. This imbalance is moderate rather than extreme, but an accuracy-only objective could still favor subtype 1. Classification is therefore trained with inverse-frequency class weights and evaluated with macro-F1 and per-class recall/precision.

All 36 validation references contain both whole-pancreas and lesion foreground. Lesion sizes in the original validation grid range from 673 to 137,595 voxels (median 7,552.5), illustrating why a mean lesion Dice can conceal substantially different small- and large-target behavior.

## Source audit and deterministic label repair

The preparation script first discovers every case and rejects unexpected names, missing image/mask pairs, duplicate identifiers, split overlap, non-3D images, non-finite masks, image/mask shape differences, affine or spacing disagreements, and labels outside the declared set. It reads all 288 labelled image/mask pairs and all 72 test geometries before writing any generated dataset.

The audit found four source mask values: `0.0`, `1.0`, `1.0000152587890625`, and `2.0`. The third value is only $1.52587890625\times10^{-5}$ above one, but nnU-Net's integrity logic treats categorical labels exactly. Across 214 masks, 18,620,040 voxels used this near-integer representation. Passing them through unchanged could fail integrity checking or create an unintended class.

The source data remain immutable. In a separate generated nnU-Net dataset, the converter:

1. requires every mask value to lie within $10^{-3}$ of an integer;
2. rounds only after that check;
3. requires every rounded value to belong to `{0,1,2}`;
4. writes the generated mask as `uint8`; and
5. verifies shape, affine, voxel spacing, qform, and sform against the source.

The maximum observed rounding error was $1.52587890625\times10^{-5}$, well inside the declared tolerance. The generated masks contain exactly 498,469,423 background voxels, 26,396,648 label-1 voxels, and 7,917,460 label-2 voxels. This is a representation repair, not spatial relabelling or resampling. The generated `data_audit.json` and split/classification manifests record counts, values, affected voxels, and case membership.

## Leakage prevention and train-only planning

The supplied folders, rather than a new random split, define fold 0. The generated `splits_final.json` contains 252 training and 36 validation IDs with an empty intersection; all 72 test IDs are also disjoint. Case-level subtype labels are derived once from the labelled source folder and stored as an explicit `case_id -> class_id` mapping. Test identifiers do not encode labels.

A two-phase procedure prevents validation information from entering preprocessing statistics:

1. the training cases are placed in `imagesTr/labelsTr`, while validation cases are initially placed in `imagesVal/labelsVal`;
2. dataset integrity, fingerprint extraction, and ResEnc M planning run on the 252 training cases only;
3. the same converter is rerun with validation cases in `imagesTr/labelsTr`, because nnU-Net preprocessing expects all cases referenced by a manual split there; and
4. all 288 labelled cases are preprocessed using the already-frozen training-only fingerprint and plan.

The produced fingerprint contains 252 cases, and the training log confirms use of the explicit 252/36 split. Validation patches are used for monitoring and checkpoint selection only; they never contribute gradients. No external data or pretrained weights are supplied to the trainer.

## Data governance

The public repository excludes all CT volumes, masks, source-derived qualitative panels, the compiled report containing those panels, model checkpoints, credentials, and local work paths. Code, configuration, tests, documentation, the architecture diagram, and suitably sanitized aggregate plots are published. The final PDF and qualitative panels are delivered privately with the take-home submission. The supplied images are de-identified, but that does not imply a license to redistribute them. This system is an evaluation prototype and not a clinical device.

# Method

## nnU-Net fingerprinting and preprocessing

nnU-Net v2 [1] automatically selects target spacing, patch size, batch size, normalization, and network topology from the dataset fingerprint and the selected ResEnc M memory target. Table 2 records the generated 3D full-resolution plan rather than a hand-written approximation.

| Planning/preprocessing item | Frozen value |
|---|---|
| nnU-Net / planner | nnU-Net v2.8.1 / `nnUNetPlannerResEncM` |
| Configuration / plans name | `3d_fullres` / `nnUNetResEncUNetMPlans` |
| Fingerprint population | 252 training cases only |
| Image reader/writer | `SimpleITKIO` |
| Median original spacing | $2.0\times0.73046875\times0.73046875$ mm |
| Target spacing | $2.0\times0.73046875\times0.73046875$ mm |
| Median resampled shape | $59\times118\times181$ voxels |
| Training patch / batch | $64\times128\times192$ / 2 |
| Image resampling | third-order in-plane/data interpolation; order 0 in separate z handling |
| Label resampling | segmentation-aware order 1; order 0 in separate z handling |
| CT normalization mask | disabled |
| CT clip bounds | training-foreground 0.5th/99.5th percentiles: -55.9961 / 179.9780 |
| CT centering/scaling | training-foreground mean 74.8919; standard deviation 44.0982 |

: Frozen train-only nnU-Net planning and preprocessing configuration. {#tbl:planning}

CT normalization clips intensities to the training-derived percentile bounds, subtracts the training-derived mean, and divides by the training-derived standard deviation. The median axial spacing is much coarser than in-plane resolution; accordingly, the first encoder kernel is $1\times3\times3$ and the first downsampling stride preserves the axial dimension.

## Shared 3D ResEnc M backbone

The segmentation foundation is the planned `ResidualEncoderUNet`, not a smaller custom substitute. It has six residual encoder stages with feature widths `[32, 64, 128, 256, 320, 320]`, encoder block counts `[1, 3, 4, 6, 6, 6]`, and one convolution per decoder stage. Instance normalization with learnable affine parameters and leaky-ReLU activations follow the nnU-Net plan.

For the planned patch, the feature-map sizes progress approximately as

$$
(64,128,192)\rightarrow(64,64,96)\rightarrow(32,32,48)
\rightarrow(16,16,24)\rightarrow(8,8,12)\rightarrow(4,4,6).
$$

The wrapper exposes the original encoder and decoder directly so nnU-Net can still toggle deep supervision. One encoder call produces the complete skip hierarchy. Those same features feed the skip-connected segmentation decoder, while the deepest 320-channel tensor feeds the classification branch. Thus, gradients from both losses update the same encoder parameters. The instantiated model has 102,764,274 unique learned parameters: 90,259,136 in the shared encoder, 12,008,943 in the decoder, and 496,195 in the classification pooling/head path. Counts deduplicate the aliases intentionally exposed for nnU-Net compatibility.

![Implemented multi-task architecture. One six-stage 3D ResEnc M encoder (32, 64, 128, 256, 320, and 320 channels) processes each $1\times64\times128\times192$ CT patch. Its skip features drive the native five-stage nnU-Net decoder and three-class voxel output, with deep supervision used during training. The same 320-channel bottleneck also feeds global-average pooling and eight-head learned-query attention, followed by layer normalization, a 128-unit GELU layer, dropout 0.30, and three subtype logits.](figures/architecture.png){#fig:architecture width=100%}

## Segmentation branch and deep supervision

The standard nnU-Net decoder reconstructs three voxel-logit channels: background, pancreas, and lesion. During training it emits five resolutions. The highest-resolution output and four auxiliary outputs use relative deep-supervision weights `[1, 1/2, 1/4, 1/8, 0]`, normalized to sum to one; the lowest output therefore receives no direct loss. At inference, deep supervision is disabled and only the full-resolution logits are exported. This follows the established deeply supervised learning principle [4].

For each supervised scale, the segmentation loss is the unweighted sum of memory-efficient soft Dice loss (excluding background, smoothing $10^{-5}$) and robust voxel-wise cross-entropy:

$$
L_{seg}=L_{Dice}+L_{CE}.
$$

The nnU-Net implementation represents the overlap term as negative soft Dice,
so this summed optimization objective can legitimately cross below zero as
overlap improves; its absolute sign is not itself a performance metric.

This combination addresses complementary failure modes. Dice reduces sensitivity to the overwhelming number of background voxels, while cross-entropy supplies stable local class supervision. Foreground-aware patch sampling separately increases exposure to pancreatic structures; it is not a replacement for an imbalance-aware loss.

## Hybrid cross-attention classification branch

Let the deepest encoder feature be $B\in\mathbb{R}^{N\times C\times D\times H\times W}$, where $C=320$. The branch deliberately combines a conservative whole-patch summary with a learnable focal summary:

$$
g=\frac{1}{DHW}\sum_{d,h,w}B_{:,:,d,h,w}.
$$

For attention, the spatial dimensions are flattened into $DHW$ tokens, and each token is layer-normalized. A learned query $q\in\mathbb{R}^{1\times C}$ attends to these tokens through eight-head `MultiheadAttention`, following scaled dot-product multi-head attention [6]:

$$
a=\mathrm{LayerNorm}\left(q+\mathrm{MHA}(q,\mathrm{LN}(B),\mathrm{LN}(B))\right).
$$

The pooled vector is `[g ; a]`, with 640 channels. It passes through layer normalization, a 640-to-128 linear layer, GELU, dropout 0.30, and a 128-to-3 linear classifier. Only the newly added head is custom-initialized; the segmentation network retains nnU-Net's initialization. Cross-attention is used because a single query can emphasize a small discriminative region, while the parallel global average path prevents the learned attention from being the only evidence source. This is an architectural rationale, not a claim of improvement: no pooling ablation is reported unless one is actually completed.

The default network call remains segmentation-only so stock nnU-Net tensor operations are not broken. Training and the explicit joint predictor request `(segmentation_logits, classification_logits)` through a separate argument, and tests enforce both contracts.

## Joint loss, class imbalance, and patch evidence

The training subtype counts $(62,106,84)$ yield inverse-frequency weights

$$
w_c=\frac{N}{K n_c}=(1.3548387,\;0.7924528,\;1.0000000).
$$

The classifier uses three-class cross-entropy with these weights and label smoothing $\epsilon=0.05$ [5]. Label smoothing discourages extreme confidence on a small dataset. Because nnU-Net trains on patches while subtype is a case label, not every crop contains equally direct lesion evidence. For sample $i$, a reliability factor is set to $r_i=1.0$ if its highest-resolution target patch contains label 2 and $r_i=0.25$ otherwise. The batch classification loss is

$$
L_{cls}=\frac{\sum_i r_i\,CE_{w,\epsilon}(z_i,y_i)}{\sum_i r_i}.
$$

Non-lesion patches retain a non-zero contribution because surrounding pancreas and context may still be informative; they are not discarded. This heuristic also has a limitation: lesion presence is available only during training, and some subtype evidence may be diffuse rather than localized. Its value is therefore treated as a design choice, not an established causal improvement.

The joint objective is

$$
L_{total}=L_{seg}+0.5L_{cls}.
$$

The 0.5 classification coefficient is a conservative fixed weighting intended to add case supervision without overwhelming the dense segmentation task. No test data are used to tune it, and no claim of optimality is made. W&B logs both components independently so task dominance or divergence remains visible.

## Optimization and augmentation

Table 3 gives the exact launched-run configuration. The 200-epoch cap is a deadline-aware reduction from nnU-Net's nominal 1,000 epochs; each epoch is also defined explicitly in iterations, so “epoch” does not mean one complete pass over all cases.

| Item | Launched configuration |
|---|---|
| Maximum epochs | 200 |
| Training / validation iterations per epoch | 125 / 30 |
| Batch / patch | 2 / $64\times128\times192$ |
| Optimizer | SGD, momentum 0.99, Nesterov enabled |
| Initial learning rate | 0.01 |
| Schedule | polynomial decay, $lr(e)=0.01(1-e/200)^{0.9}$ |
| Weight decay | $3\times10^{-5}$ |
| Precision | CUDA automatic mixed precision with gradient scaling |
| Gradient clipping | global norm 12 |
| Gradient accumulation | none |
| Foreground oversampling | 50% of patches |
| Deep supervision | enabled during training; disabled for inference |
| PyTorch compilation | disabled |
| Training determinism | not guaranteed; fixed split, stochastic augmentation/CUDA |

: Exact production-training configuration. {#tbl:training-config}

The training transform uses nnU-Net v2.8.1's standard 3D pipeline: rotation up to $\pm 30^{\circ}$ with probability 0.2; synchronized scaling in `[0.7,1.4]` with probability 0.2; Gaussian noise (0.1); Gaussian blur (0.2); multiplicative brightness (0.15); contrast (0.15); simulated low resolution (0.25); inverted gamma (0.1); ordinary gamma (0.3); and mirroring over all three spatial axes. Elastic deformation is disabled by the inherited configuration. The fixed validation loader performs no stochastic intensity or spatial augmentation.

No global deterministic training seed is forced by the launched trainer. The supplied split itself is deterministic, but the exact optimization path is not bitwise reproducible across reruns. This is reported as a limitation rather than labelling the configuration “seed 12345”; 12345 is used only for deterministic evaluation bootstrapping.

## Overfitting controls

The model is high-capacity relative to 252 cases, and a case label is repeated over multiple patches. The enabled controls are:

- extensive spatial and intensity augmentation;
- a compact 128-unit classification hidden layer rather than a large fully connected stack;
- dropout 0.30 and label smoothing 0.05 in the classifier;
- weight decay $3\times10^{-5}$;
- separate monitoring of segmentation and classification train/validation losses;
- per-class F1 and validation-case coverage, which expose majority-class shortcuts and incomplete patch coverage;
- a fixed supplied validation split and prohibition on adding those cases to training; and
- a small, declared checkpoint candidate set rather than selecting individual test outputs.

The trainer does not use automated early stopping. Training can be manually terminated if both task losses and validation signals have clearly plateaued, as allowed by the brief; the actual stopping epoch and reason must be inserted in Section 6. Repeated design choices informed by the same 36 validation cases can still overfit the development process even though those cases never receive gradients.

# Experiment protocol and inference

## Development checks before the long run

High-risk contracts were tested before expensive training: case-ID-to-subtype mapping, inverse-frequency weights, macro-F1, patch-evidence weighting, attention-head compatibility, network tensor shapes, gradient flow into the shared encoder, segmentation-only compatibility, mirror/tile/fold classification aggregation, atomic GPU-to-CPU inference retry, categorical mask repair, fixed-split generation, evaluation conventions, report-figure generation, complete-checkpoint selection, and final archive structure. At the current report revision, all 175 repository tests pass; the four emitted warnings are deprecation warnings inside the third-party `batchgenerators` package rather than test failures. This count is rerun at the evaluated source commit.

A full planned-patch CUDA forward/backward smoke test used batch size 2 on the RTX 4060 Laptop GPU and completed within the 8 GB device budget. The production training log additionally confirms that a complete multi-task training/validation epoch finished without an out-of-memory failure.

## W&B and in-training monitoring

The run records configuration and both task streams through nnU-Net's W&B-compatible logger. Logged signals include total training/validation loss, segmentation and classification component losses, learning rate, native per-class patch Dice, whole-pancreas patch micro-Dice, classification accuracy, case-aggregated patch macro-F1 and per-class F1, validation case coverage, fraction of patches containing lesion, epoch duration, and a multi-task checkpoint score.

These in-training validation signals are intentionally called *patch diagnostics*. Thirty validation iterations at batch size two sample 60 patches, and repeated patch logits are averaged per observed case. The case-coverage field reveals whether all 36 cases appeared that epoch. Patch Dice and patch-aggregated subtype F1 are useful for learning curves but are not the final reported metrics. Section 5's frozen evaluator instead scores one restored full-volume prediction per validation case.

The final W&B record must include:

- project/run URL: [public W&B run](https://wandb.ai/amirfahamfallahpour1379-university-of-toronto/pancreas-multitask-amirfaham-fallahpour/runs/hrs05iyx);
- run ID: `hrs05iyx`;
- synchronization status: `PENDING_WANDB_SYNC_STATUS`; and
- screenshots/exports used in Figures 2 and 3: `PENDING_WANDB_EXPORT_PATHS`.

> **Figure 2 — PENDING_LOSS_CURVES.** Raw epoch-level training and validation-patch loss curves for the segmentation and classification objectives. These optimization traces are not full-volume results.

> **Figure 3 — PENDING_PERFORMANCE_CURVES.** Native label-1/lesion patch Dice, whole-pancreas and mean-foreground patch diagnostics, plus patch-aggregated macro-F1 and accuracy. These curves are monitoring diagnostics rather than final full-volume metrics.

## Predeclared conditional classification-head rescue

Online patch diagnostics had already indicated classification collapse before this contingency was specified; the rescue is therefore not presented as a blinded design choice. It was nevertheless prospectively frozen and committed before any restored full-volume prediction or evaluation on the fixed validation set. **Conditional status:** `PENDING_CLASSIFICATION_RESCUE_ACTIVATION`. Until the hash-bound train-only audit is complete, the rescue is only an opt-in contingency and no rescue execution or outcome is claimed.

Activation is determined only from classification CE and training-patch accuracy saved in the completed joint run. For the ten-epoch window ending at epoch 40, activation requires all three predeclared conditions: mean training classification CE at least 1.05, mean training-patch accuracy at most 0.42, and an ordinary-least-squares CE slope at least $-0.001$ per epoch. If that rule is negative, the hard audit of the ten-epoch window ending at epoch 50 activates when either mean training CE is above 1.03 or mean training-patch accuracy is below 0.45. The audit is bound by SHA-256 to `checkpoint_final.pth`. If neither gate activates, later fixed-validation results cannot activate the rescue.

Even if the gate activates at epoch 40 or 50, the original 200-epoch joint run must finish cleanly and `checkpoint_final.pth` remains the fixed initialization. One and only one head-only attempt is allowed: learned-query hybrid pooling and the classification MLP are reinitialized with seed 20260806, then optimized for exactly $30\times125=3{,}750$ training-patch updates at batch size 2. The fixed optimizer is AdamW with constant learning rate $3\times10^{-4}$, weight decay $10^{-4}$, and gradient-norm clipping at 1.0. It retains the original training augmentation, foreground oversampling, inverse-frequency class weights, label smoothing 0.05, and non-lesion patch weight 0.25. There is no hyperparameter search or second rescue attempt.

This is not a second round of multi-task fine-tuning. Every encoder and decoder parameter is frozen, both modules remain in evaluation mode, the frozen encoder bottleneck is computed under `no_grad`, and the decoder forward path is bypassed. Exact component hashes before and after optimization verify that the encoder and decoder did not change. The rescue loader indexes only the 252 training keys. The shared 288-case classification metadata is parsed and range-validated, but only training-key labels are indexed into rescue targets or losses; no validation image, segmentation volume, batch, gradient, or stopping signal is consumed. Exact training and validation case-ID hashes are additionally bound to the frozen pretraining split manifest. Validation cannot activate, initialize, stop, extend, restart, or otherwise tune the rescue schedule.

## Checkpoint production and final selection

The joint run retains three original checkpoint types:

1. nnU-Net's stock `checkpoint_best.pth`, selected by exponential moving average of mean foreground patch Dice;
2. `checkpoint_best_multitask.pth`, saved whenever
   $$
   S_{epoch}=\tfrac12(\text{mean foreground patch Dice}+\text{patch-aggregated macro-F1})
   $$
   improves; and
3. `checkpoint_final.pth`, representing the last completed epoch.

Checkpoint evaluation is deliberately a single three-or-four-candidate full-volume pass. If the train-only gate is negative, the three original checkpoints are each evaluated once. If the gate is affirmative and the single fixed rescue completes, `checkpoint_classification_rescue.pth` is added and all four candidates are each evaluated once under identical inference and evaluator settings. No preliminary full-volume fixed-validation pass occurs before the activation audit, and the candidate comparison cannot feed back into rescue optimization.

The test set never influences checkpoint selection. Each eligible candidate is evaluated with restored full-volume predictions on the fixed validation set, and the final score is predeclared as

$$
S_{final}=\tfrac13(\overline{DSC}_{whole}+\overline{DSC}_{lesion}+Macro\text{-}F1).
$$

The candidate with the largest equal-weight score is selected; this prevents the much easier whole-organ task from being the only selection signal. The selected artifact remains `PENDING_SELECTED_CHECKPOINT_NAME` until training and full-volume evaluation complete. Its SHA-256 is `PENDING_CHECKPOINT_SHA256`.

## Explicit joint sliding-window inference

The joint predictor is separate from nnU-Net's stock segmentation-only path. For each tile and enabled mirror orientation, it averages segmentation logits after spatially reversing mirrored outputs and averages three-class softmax probabilities. Across spatial tiles, segmentation logits use the standard Gaussian overlap weighting, while subtype probabilities receive an equal tile mean. Across folds, segmentation logits and subtype probabilities are again averaged separately. The final subtype is the probability argmax.

All accumulators are local to one prediction attempt. If storing segmentation accumulation on the GPU fails, the complete attempt is discarded and retried with result arrays on CPU; classification votes from a partial failed attempt cannot leak into the retry. This behavior is covered by synthetic tests.

The final inference switches are:

| Setting | Final value |
|---|---|
| Sliding-window step size | 0.5 of the patch extent |
| Gaussian weighting | enabled for segmentation overlap |
| Mirroring/TTA axes | enabled on the checkpoint-declared axes `(0,1,2)` |
| Folds/checkpoints | fold 0; `PENDING_SELECTED_CHECKPOINT_NAME` |
| Everything-on-device | enabled, with atomic CPU-results fallback on runtime failure |
| Post-processing | none predeclared; any validation-selected change must be disclosed |

: Joint full-volume inference settings. {#tbl:inference}

Probability averaging is an explicit ensemble rule, not mathematically equivalent to averaging logits. Equal tile weighting for classification is simple and reproducible, but it may give non-lesion or padded-context tiles too much influence. This is an implementation-specific limitation and a useful future ablation target.

# Evaluation aligned with Metrics Reloaded

Metrics Reloaded recommends choosing metrics from the task and target properties, explicitly stating aggregation and edge-case rules, and avoiding a single headline value without failure analysis [3]. The present tasks are semantic segmentation of two nested foreground definitions and single-label multiclass classification. Table 5 maps each question to its evidence.

| Evaluation question | Primary evidence | Supporting evidence |
|---|---|---|
| Is the entire pancreatic foreground localized? | case-level Dice for `label > 0` | median, SD, range, distribution, overlays |
| Is lesion tissue delineated? | case-level Dice for `label == 2` | empty prediction count, lesion-size plot, failures |
| Are all three subtypes separated? | macro-F1 over fixed labels 0/1/2 | class precision/recall/F1, support, confusion matrix |
| How uncertain is the fixed-split estimate? | 95% case-bootstrap interval | raw case-level CSV; explicit $n=36$ |
| Are artifacts structurally valid? | geometry/label/count validators | audit JSON and archive SHA-256 |

: Task-to-metric and supporting-evidence map. {#tbl:evaluation-map}

## Segmentation definitions

For binary prediction $P$ and reference $G$, Dice is

$$
DSC(P,G)=\frac{2|P\cap G|}{|P|+|G|}.
$$

Each restored validation label map is scored twice:

- **whole pancreas:** $P=\{\hat y>0\}$ and $G=\{y>0\}$; and
- **lesion:** $P=\{\hat y=2\}$ and $G=\{y=2\}$.

The headline number is the unweighted arithmetic mean of 36 case Dice values, so every patient-sized ROI contributes equally regardless of voxel count. The evaluator also reports sample standard deviation, median, minimum, maximum, and a 95% percentile interval for the mean from 2,000 case-resampling bootstrap draws with seed 12345. Mean $\pm$ sample SD and the confidence interval describe different quantities and are both retained.

If prediction and reference are both empty for a target, Dice is defined as 1.0; if exactly one is empty, Dice is 0.0. There are no empty lesion or whole-pancreas references among the 36 validation cases, so the empty-empty convention cannot inflate the reference-side validation score. The number of empty lesion predictions remains an important reported failure count.

Dice measures overlap but does not identify whether errors are boundary shifts, remote false positives, or complete small-lesion misses. Qualitative overlays and lesion-size-stratified review are therefore required. A surface-distance metric could add boundary information, but it is not made a headline metric because the brief specifies Dice and the short deadline prioritizes a frozen, tested evaluator.

## Classification definitions

For each fixed class $c\in\{0,1,2\}$, precision, recall, and one-vs-rest F1 are computed from the $3\times3$ confusion matrix:

$$
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c}, \qquad
Macro\text{-}F1=\frac{1}{3}\sum_{c=0}^{2}F1_c.
$$

All three classes remain in the macro average even if a model never predicts one. Any zero denominator receives 0.0. Confusion-matrix rows are reference labels and columns are predictions. Macro-F1 weights subtypes equally, while the accompanying per-class support and recall make the effects of the 9/15/12 validation imbalance transparent. Accuracy is reported only as a secondary metric.

The evaluator computes a 95% percentile case-bootstrap interval for macro-F1 and accuracy using the same 2,000 draws and seed. With only 36 cases and as few as nine cases in a subtype, this interval can be wide and discrete; it is descriptive uncertainty, not evidence of external generalization.

## Independent evaluator and geometry checks

Final numbers are produced by `scripts/evaluate_predictions.py`, which does not import PyTorch or the trainer's metric implementation. Before scoring it requires an exact case set, readable finite NIfTI arrays, exact integer labels in `{0,1,2}`, equal shapes, and affine/spacing agreement within $10^{-5}$. It writes both aggregate JSON and case-level CSV. This separation reduces the chance that a bug shared by training and evaluation silently validates itself.

The report is populated from those saved files. The JSON records all conventions, bootstrap parameters, empty-case counts, and confusion-matrix axes. The final artifact paths and hashes are:

- aggregate metrics: `PENDING_METRICS_JSON_AND_SHA256`;
- case-level metrics: `PENDING_CASE_CSV_AND_SHA256`;
- classification predictions: `PENDING_VALIDATION_SUBTYPE_FILE`; and
- validation masks: `PENDING_VALIDATION_MASK_DIRECTORY`.

## Qualitative case-selection protocol

Examples are selected by a reproducible rule after all 36 cases are scored, not by visually searching for attractive slices. The two “strong” cases are the highest lesion-Dice cases and the two “weak” cases are the lowest lesion-Dice cases; whole-pancreas Dice and case ID break ties deterministically. For each case and orientation, the slice with the greatest reference/prediction lesion-union area is shown; whole-organ union is the fallback if both lesion masks are empty. One combined panel uses a fixed CT window and contour legend. Row labels report whole-pancreas and lesion Dice, subtype reference/prediction, and reference lesion voxels.

> **Figure 4 — PENDING_DICE_DISTRIBUTION.** Paired case-level distribution (all 36 points) for whole-pancreas and lesion Dice, with mean, median, and undergraduate target lines. Optionally add lesion volume versus lesion Dice without implying causality.

> **Figure 5 — PENDING_QUALITATIVE_CASES.** One combined three-plane panel containing the two rule-selected weakest and two strongest lesion-Dice cases, with consistent reference/prediction contours for labels 1 and 2.

# Results

## Primary fixed-validation outcomes

| Task | Metric (36 cases) | Undergraduate target | Result | 95% bootstrap CI | Target met? |
|---|---|---:|---:|---:|---|
| Segmentation | Whole-pancreas Dice, mean $\pm$ SD | 0.90 | `PENDING_WHOLE_MEAN_SD` | `PENDING_WHOLE_CI` | `PENDING` |
| Segmentation | Lesion Dice, mean $\pm$ SD | 0.27 | `PENDING_LESION_MEAN_SD` | `PENDING_LESION_CI` | `PENDING` |
| Classification | Macro-F1 | 0.60 | `PENDING_MACRO_F1` | `PENDING_MACRO_F1_CI` | `PENDING` |

: Primary fixed-validation results and undergraduate targets. {#tbl:primary-results}

**Selected checkpoint:** `PENDING_CHECKPOINT_NAME_EPOCH_HASH`.  
**Completed training:** `PENDING_EPOCHS` epochs / `PENDING_WALL_CLOCK_HOURS` hours.  
**Evidence artifact:** `PENDING_METRICS_ARTIFACT`.

`PENDING_RESULT_SUMMARY_PARAGRAPH`: replace with a direct three-sentence account of which targets were met, the largest performance gap, and whether the evidence supports the joint-model hypothesis. Do not explain away a missed target.

## Segmentation distribution

| Foreground definition | Mean | Sample SD | Median | Minimum | Maximum | Empty predictions |
|---|---:|---:|---:|---:|---:|---:|
| Whole pancreas (`label > 0`) | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| Lesion (`label == 2`) | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |

: Case-level segmentation distribution over the 36 fixed validation cases. {#tbl:segmentation-results}

`PENDING_SEGMENTATION_INTERPRETATION`: report whether lesion performance varies with reference lesion size, distinguish boundary errors from complete misses, and explain why whole-pancreas binarization can remain high when lesion voxels are mislabeled as normal pancreas.

## Classification detail

| Reference subtype | Support | Predicted count | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 9 | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| 1 | 15 | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| 2 | 12 | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| **Macro average** | 36 | 36 | `PENDING` | `PENDING` | **`PENDING`** |

: Fixed-validation three-class classification results. {#tbl:classification-results}

> **Figure 6 — PENDING_CONFUSION_MATRIX.** Three-class confusion matrix with rows explicitly labelled “reference” and columns “prediction”; cells show raw counts.

`PENDING_CLASSIFICATION_INTERPRETATION`: identify the dominant confusion, minority-class recall, and any precision/recall trade-off. Do not infer pathology mechanisms from folder labels alone.

## Training behavior and checkpoint choice

The selected checkpoint occurred at epoch `PENDING_SELECTED_EPOCH`. Peak allocated/reserved VRAM was `PENDING_PEAK_VRAM`, and mean epoch duration after warm-up was `PENDING_EPOCH_TIME`. The loss and diagnostic curves show `PENDING_CONVERGENCE_AND_GAP_OBSERVATION`. The final choice follows the rule stated in Section 4.4 and the complete comparison below.

| Checkpoint candidate | Epoch | Whole Dice | Lesion Dice | Macro-F1 | Equal-weight score | Selected? |
|---|---:|---:|---:|---:|---:|---|
| `checkpoint_best.pth` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| `checkpoint_best_multitask.pth` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| `checkpoint_final.pth` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |

: Complete fixed-validation checkpoint comparison used for final selection. {#tbl:checkpoint-comparison}

`PENDING_SELECTION_EXPLANATION`

## Qualitative review

| Role | Case | Whole Dice | Lesion Dice | True/predicted subtype | Evidence-based observation |
|---|---|---:|---:|---|---|
| Strong 1 | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| Strong 2 | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| Weak 1 | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| Weak 2 | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |

: Deterministically selected qualitative validation cases. {#tbl:qualitative-cases}

`PENDING_QUALITATIVE_SYNTHESIS`: connect observed false positives, false negatives, boundary errors, and subtype mistakes to the quantitative distributions without presenting visual correlation as causation.

## Inference efficiency

Acceleration is optional at the undergraduate level, so no optimized-runtime claim is made. The required baseline joint inference is still measured end to end with CUDA synchronization implicit at process completion; it includes NIfTI I/O, preprocessing, mirrored sliding-window prediction, geometry restoration, and export.

| Split | Cases | Total wall time | Mean time/case | Peak GPU memory |
|---|---:|---:|---:|---:|
| Validation | 36 | `PENDING` | `PENDING` | `PENDING` |
| Test | 72 | `PENDING` | `PENDING` | `PENDING` |

: Baseline end-to-end joint-inference efficiency. {#tbl:inference-runtime}

# Discussion

## Interpretation of the joint approach

`PENDING_EVIDENCE_BASED_DISCUSSION`: state whether the shared encoder supported useful subtype prediction while retaining segmentation performance. Use the final metrics and curves to discuss task compatibility; do not infer positive transfer without a segmentation-only or classification-only ablation.

Several design properties are defensible independently of the final score. First, the mandatory ResEnc M architecture and nnU-Net preprocessing were preserved rather than replaced for convenience. Second, the classifier receives the actual deepest shared representation and contributes gradients to that encoder. Third, global average and attention pooling provide complementary summaries without modifying the segmentation decoder. Fourth, independent full-volume evaluation prevents patch monitoring values from being presented as final validation outcomes.

## Expected and observed failure mechanisms

Lesion Dice is intrinsically more volatile than whole-pancreas Dice because a fixed boundary displacement occupies a larger fraction of a small target, and a complete miss produces zero. Whole-pancreas evaluation merges labels 1 and 2, so a lesion voxel predicted as ordinary pancreas remains correct for the whole-organ metric. A strong whole-pancreas result therefore cannot establish lesion delineation.

Patch-level classification creates another failure mode. A crop may contain little or no discriminative lesion tissue while retaining the case label. The non-lesion reliability factor reduces that noise during training, but at inference every tile contributes equally to the subtype probability. If `PENDING_OBSERVED_TILE_DILUTION` is observed, lesion-aware or uncertainty-weighted tile aggregation would be a justified future comparison.

Class weighting raises the optimization cost of subtype-0 mistakes, but it cannot manufacture distinguishing features. The final confusion matrix must establish whether minority recall improved at the expense of precision. Without an unweighted-loss ablation, the report cannot attribute any difference specifically to the weights.

## Limitations

The principal limitations are:

1. **Small fixed validation set.** Thirty-six cases, including only nine subtype-0 cases, produce uncertain and discrete class estimates.
2. **No repeated seeds or cross-validation.** One stochastic training trajectory cannot quantify optimization variance or split sensitivity.
3. **Validation-development reuse.** Repeated monitoring and checkpoint comparison can overfit decisions to the fixed validation set despite zero gradient leakage.
4. **Cropped ROIs.** The task omits full-scan localization, so performance cannot be extrapolated to an uncropped clinical CT workflow.
5. **No external validation.** Scanner, institution, population, and protocol generalization remain unknown.
6. **Patch/case mismatch.** Classification is supervised on patches with a case label and averaged uniformly over inference tiles.
7. **Fixed task weighting.** The 0.5 coefficient was not supported by a controlled task-weight ablation or gradient-conflict analysis.
8. **Nondeterministic training.** The split is fixed, but stochastic augmentation and CUDA execution are not made bitwise reproducible.
9. **Limited metric scope.** Dice and macro-F1 do not measure calibration, boundary distance, or clinical utility.
10. **Substantial AI assistance.** AI accelerated implementation but creates semantic-error and verification risk, especially under a deadline.

No calibration study, clinical reader study, prospective evaluation, or safety analysis was performed. The output is not suitable for diagnosis or patient care.

## Next experiments

Given more time and compute, the highest-value experiments would be repeated-seed or cross-validated training; a global-average-only versus hybrid-attention ablation; segmentation-only and classification-only controls; gradient-scale/conflict diagnostics for adaptive task weighting; lesion-volume-stratified analysis; calibrated case-level classification; and lesion-aware tile aggregation. External validation would be required before any translational claim. A speed optimization should be evaluated only under a fixed accuracy tolerance and synchronized end-to-end timing protocol.

# AI-assisted workflow and candidate contribution

The brief explicitly requests AI coding tools and more than 50% AI-generated code. This project used OpenAI Codex for a substantial majority of the initial implementation and documentation, including requirement extraction, data-audit code, the network/trainer/predictor, tests, metric and packaging utilities, debugging support, and report drafting. An estimated 85–95% of the initial repository implementation and documentation was AI-generated. This range is based on file-level provenance and the recorded workflow, not a misleading post-formatting line count; exact attribution is inherently approximate after library-generated configuration, automated formatting, candidate review, and revisions.

Amirfaham Fallahpour is the candidate, project owner, and final accountable reviewer. He set the objective and quality threshold; specified that the result should be the strongest defensible submission rather than a minimal completion; provided the data, compute, time constraints, identity, and authorization; remained available for human-only authentication and trade-offs; reviews the technical explanations and artifacts; decides what is submitted; and accepts responsibility for every final claim. These are meaningful candidate contributions without falsely claiming manual authorship of AI-generated code.

The workflow used seven controls:

1. translate each brief requirement into an implementation and evidence contract before coding;
2. preserve source data and audit all automated label repair;
3. test high-risk tensor, gradient, metric, geometry, and packaging contracts before long training;
4. tie the run to configuration, environment, split, checkpoint, Git commit, and W&B artifacts;
5. keep unmeasured report fields visibly pending;
6. have the candidate review consequential choices and final outputs; and
7. adversarially validate the public repository, PDF, and extracted ZIP before submission.

AI output was treated as an untrusted draft. Verification includes unit tests on synthetic edge cases, source and converted-data audits, a planned-patch forward/backward smoke test, explicit split checks, W&B/local training logs, independent saved-prediction evaluation, and full archive validation. One concrete failure occurred during AI-assisted environment bootstrapping: dependency resolution replaced the intended CUDA-enabled PyTorch build. A direct CUDA availability/tensor test exposed the error before training; the environment was rebuilt with the explicit PyTorch CUDA 12.8 index and `torch==2.8.0+cu128`, followed by `pip check` and a real GPU tensor operation. This illustrates why successful installation alone was not accepted as verification. The detailed responsibility matrix and prompt categories appear in `docs/AI_WORKFLOW.md`. Credentials and sensitive interaction records are not published.

# Reproducibility and deliverables

## Environment

| Component | Reported environment |
|---|---|
| Operating system | Microsoft Windows 11 Home, build 10.0.26200 |
| CPU / RAM | Intel Core i9-13980HX / 15.6 GiB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM |
| Python | 3.12.13 |
| PyTorch / CUDA | 2.8.0+cu128 / CUDA 12.8 |
| cuDNN | 9.1.0.2 |
| nnU-Net v2 | 2.8.1 |
| dynamic-network-architectures | 0.4.4 |
| Nibabel / SimpleITK | 5.4.2 / 2.5.6 |
| W&B | 0.28.1 |
| Training seed | no global deterministic seed enforced |
| Evaluation bootstrap seed | 12345 |
| Dependency specification | `requirements.txt` |
| Reported Git commit | `PENDING_GIT_COMMIT` |

: Software, hardware, and reproducibility environment. {#tbl:environment}

## Reproduction sequence

Paths below are placeholders chosen by the reproducer; no private absolute path is required.

```powershell
# Create/activate a Python 3.12 environment, then:
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

# Configure nnU-Net roots and the external trainer search path.
. .\scripts\Set-QuizEnvironment.ps1 -WorkRoot <WORK_ROOT> -WandbMode offline

# Phase 1: audit/convert with validation excluded from fingerprinting.
python .\scripts\prepare_dataset.py `
  --source <SOURCE_DATA> `
  --output-root $env:nnUNet_raw `
  --dataset-id 501 --dataset-name PancreasMultitask `
  --validation-layout separate

nnUNetv2_extract_fingerprint -d 501 --verify_dataset_integrity -np 2
nnUNetv2_plan_experiment -d 501 -pl nnUNetPlannerResEncM `
  -overwrite_plans_name nnUNetResEncUNetMPlans

# Phase 2: place validation in imagesTr but retain the training-only plan.
python .\scripts\prepare_dataset.py `
  --source <SOURCE_DATA> `
  --output-root $env:nnUNet_raw `
  --dataset-id 501 --dataset-name PancreasMultitask

nnUNetv2_preprocess -d 501 -plans_name nnUNetResEncUNetMPlans `
  -c 3d_fullres -np 2
# Copy splits_final.json and split_manifest.json into the matching
# nnUNet_preprocessed dataset.
# classification_labels.json remains in nnUNet_raw, where the trainer reads it.

# Primary run (omit -pretrained_weights; none are used).
nnUNetv2_train 501 3d_fullres 0 `
  -tr nnUNetTrainerPancreasMultiTask `
  -p nnUNetResEncUNetMPlans -device cuda

# After checkpoint_final exists, bind the train-only rescue decision to it.
python .\scripts\audit_classification_rescue_activation.py `
  --checkpoint <TRAINED_MODEL_DIRECTORY>\fold_0\checkpoint_final.pth `
  --output <TRAINED_MODEL_DIRECTORY>\fold_0\classification_rescue_activation.json

# Take exactly one branch. Affirmative audit:
.\scripts\Run-ClassificationRescue.ps1 -WorkRoot <WORK_ROOT>
.\scripts\Run-FinalEvaluation.ps1 `
  -WorkRoot <WORK_ROOT> -IncludeClassificationRescue
# Negative audit instead:
# .\scripts\Run-FinalEvaluation.ps1 -WorkRoot <WORK_ROOT>

# Full-volume joint inference; run separately for validation and test inputs.
python .\scripts\predict_joint.py `
  --input <RAW_INPUT_DIRECTORY> --output <PREDICTION_DIRECTORY> `
  --model <TRAINED_MODEL_DIRECTORY> --folds 0 `
  --checkpoint <SELECTED_CHECKPOINT_NAME> `
  --tile-step-size 0.5 --device cuda `
  --probability-csv <PROBABILITY_DETAILS_CSV>

# Frozen validation evaluation.
python .\scripts\evaluate_predictions.py `
  --predictions <VALIDATION_MASKS> `
  --references <VALIDATION_REFERENCES> `
  --classification-predictions <VALIDATION_SUBTYPES> `
  --classification-references <CLASSIFICATION_MANIFEST> `
  --classification-reference-split validation `
  --output-json <METRICS_JSON> --output-csv <CASE_METRICS_CSV> `
  --bootstrap-samples 2000 --confidence 0.95 --seed 12345

# Extract and validate the final ZIP against all 72 source test images.
python .\scripts\validate_submission.py `
  <AMIRFAHAM_FALLAHPOUR_RESULTS_ZIP> `
  --test-images <TEST_IMAGES> `
  --expected-count 72 --output-json <SUBMISSION_AUDIT_JSON>
```

The commands above are verified against each CLI's `--help` output at the evaluated source commit.

## Final test package contract

The archive root must contain exactly 72 masks named like `quiz_037.nii.gz` and one `subtype_results.csv`. The CSV header is exactly `Names,Subtype`; names match the masks one-to-one and subtype values are integers in `{0,1,2}`. No parent folder, source image, hidden file, or reference label is permitted.

The validator extracts the ZIP to a temporary directory and checks member-path safety, duplicate and missing cases, exact counts, readable finite NIfTI data, integer `{0,1,2}` labels, shape/affine/spacing agreement with every input CT, CSV schema, unique rows, and class ranges. Final evidence:

- archive name: `Amirfaham_Fallahpour_results.zip`;
- archive SHA-256: `PENDING_ARCHIVE_SHA256`; and
- validator status/audit: `PENDING_SUBMISSION_VALIDATION_ARTIFACT`.

The final report hash is written after PDF generation to the external submission
manifest; embedding a PDF's own hash inside that PDF would be self-referential.

# Conclusion

This project implements the required nnU-Net v2 3D ResEnc M multi-task system while treating data integrity, leakage prevention, independent evaluation, and AI disclosure as part of the technical result. The classifier combines global context and learned-query cross-attention over the shared bottleneck; training addresses subtype frequency imbalance and patch-level evidence without changing the supplied split or using external weights. The final system obtains `PENDING_CONCLUSION_RESULTS`. Its strongest supported property is `PENDING_PRIMARY_STRENGTH`, while `PENDING_PRIMARY_WEAKNESS` remains the main limitation. These conclusions must be replaced only after the reported checkpoint, fixed-split artifacts, W&B record, public repository, and test archive all pass final review.

# Requirement-to-evidence traceability {.unnumbered}

- **Required 3D ResEnc M:** generated `plans.json` plus the trainer architecture guard; Sections 3.1–3.3 and Fig. 1.
- **Shared encoder and two outputs:** `network.py`, tensor/gradient tests, and Sections 3.2–3.4.
- **W&B for both tasks:** custom trainer logger, exported curves, and [public run `hrs05iyx`](https://wandb.ai/amirfahamfallahpour1379-university-of-toronto/pancreas-multitask-amirfaham-fallahpour/runs/hrs05iyx); Section 4.2 and Figs. 2–3.
- **Classification imbalance and overfitting controls:** weighted classification objective, patch-reliability weighting, augmentation, and task-specific monitoring; Sections 3.5–3.7.
- **Metrics Reloaded-aligned validation:** independent evaluator, complete-checkpoint selector, saved aggregate/case artifacts, and Figs. 4–6; final evidence `PENDING_METRICS_ARTIFACT`.
- **No validation optimization:** disjoint split manifest and training log; Sections 2.3 and 4.
- **No external data or pretrained weights:** launch/provenance review and final checkpoint audit; Sections 1, 2.3, and 9.
- **AI workflow:** `docs/AI_WORKFLOW.md`, 85–95% initial-content estimate, and Section 8.
- **Public source:** [GitHub repository](https://github.com/Amirfaham1/pancreas-multitask-nnunet), verified 2026-08-05.
- **72 masks and subtype CSV:** joint inference and extracted-archive validator; final evidence `PENDING_PREDICTIONS`.

# References {.unnumbered}

1. Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods, 18*, 203–211. <https://doi.org/10.1038/s41592-020-01008-z>
2. Cao, K., Xia, Y., Yao, J., et al. (2023). Large-scale pancreatic cancer detection via non-contrast CT and deep learning. *Nature Medicine, 29*, 3033–3043. <https://doi.org/10.1038/s41591-023-02640-w>
3. Maier-Hein, L., Reinke, A., Godau, P., et al. (2024). Metrics Reloaded: recommendations for image analysis validation. *Nature Methods, 21*, 195–212. <https://doi.org/10.1038/s41592-023-02151-z>
4. Lee, C.-Y., Xie, S., Gallagher, P., Zhang, Z., & Tu, Z. (2015). Deeply-supervised nets. *Proceedings of AISTATS*, 562–570. <https://proceedings.mlr.press/v38/lee15a.html>
5. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception architecture for computer vision. *Proceedings of CVPR*, 2818–2826. <https://doi.org/10.1109/CVPR.2016.308>
6. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*. <https://papers.nips.cc/paper/7181-attention-is-all-you-need>
