# Technical Interview Preparation

This document is an artifact-backed interview aid. It is designed to help
Amirfaham explain the project accurately after submission—not to memorize
claims that the artifacts do not support. Final-result statements below come
from the consumed official gate, selected-test package ledger, and strict speed
audit.

## 60-second project explanation

> I directed an AI-assisted extension of nnU-Net v2’s 3D ResEnc M pipeline into a multi-task system for cropped pancreas CT. The immutable baseline scored 0.9202 whole-pancreas Dice, 0.6197 lesion Dice, and 0.4640 macro-F1 on the supplied validation split. After that result and its test package already existed, we ran a separately locked train-only comparison of two neural case heads using frozen production-matched encoder and predicted-lesion features. Two-query cross-attention MIL beat lesion-aware mean MIL in all three repeated five-fold OOF comparisons, averaging 0.5080 versus 0.4197 macro-F1. Its all-training refit reached 0.9787, so the 0.4707 resubstitution-to-OOF gap is severe overfitting. The one allowed official reevaluation scored 0.9202 whole Dice, 0.6197 lesion Dice, and 0.5254 macro-F1. That strict macro-F1 improvement selected v5 over the baseline, but 0.5254 still missed the 0.60 and 0.70 classification thresholds. The validated 72-case package is complete. A strict ABBA benchmark found the candidate 18.8% slower than stock nnU-Net, so the speed requirement also failed despite exact mask equivalence.

The 0.9202/0.6197/0.4640 values in the first baseline sentence are historical
full-volume results. The 0.5080 and 0.4197 values are train-only head-level OOF
comparisons, not official full-volume metrics and not unbiased end-to-end
estimates. The final `0.52541507` macro-F1 comes only from the locked 36-case
official gate.

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
        └────────► baseline patch classifier ─────► baseline subtype 0/1/2
        │
        ▼
historical fixed-validation selection → frozen rescue checkpoint
        │
        └────────► predicted-lesion-ranked train-case bags
                         │
                         ├── lesion-aware mean MIL control
                         └── two-query cross-attention MIL (OOF-selected)
                                      │
                                      ▼
                         one locked post-hoc official gate
                                      │
                                      ▼
                         selected classifier → test package
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

Segmentation produces spatial predictions, while subtype classification needs one prediction for the whole case. The baseline branch concatenates global-average pooling with an eight-head learned-query attention summary of the deepest 320-channel map. V5 keeps the shared encoder and segmentation decoder frozen, ranks at most three tiles by model-predicted lesion mass, and compares two small case-level neural heads. The selected 101,391-parameter head uses two learned queries and four-head cross-attention over projected stage-3 tokens, plus a projected 646-value case summary. It has no positional encoding, so token order itself is not represented.

### What the losses do

The combined objective has the general form

\[
L = \lambda_{seg}L_{seg} + \lambda_{cls}L_{cls}.
\]

For the baseline joint run, $\lambda_{seg}=1$ and $\lambda_{cls}=0.5$. `L_seg` is nnU-Net's deeply supervised memory-efficient soft-Dice plus voxel cross-entropy objective. Baseline `L_cls` is inverse-frequency-weighted, label-smoothed ($\epsilon=0.05$) three-class cross-entropy; crops without a lesion voxel receive a 0.25 reliability weight. V5 does not retrain segmentation or the encoder. Its new heads use unweighted label-smoothed cross-entropy with deterministic class-balanced case sampling, AdamW, and a fixed 150-epoch schedule.

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
- fit fingerprint/planning statistics on the 252 training cases;
- never added validation cases to training and never used them for gradients;
- used the fixed validation split for baseline monitoring and checkpoint selection, and disclose that the baseline result was already known before v5;
- restricted v5 feature extraction, head fitting, head selection, and offset decisions to the 252 training cases, with no official validation or test access by those development scripts; and
- allowed one post-lock v5 official reevaluation, with no second classifier iteration.

Important caveat: the v5 OOF folds exclude each case from the new head fit, but the common encoder and rescue head had already seen all 252 training labels. Therefore the OOF comparison is not an end-to-end generalization estimate. The historical baseline test package also existed before v5, although its outputs were not used as a v5 tuning signal.

### How did you address class imbalance?

Answer from the actual configurations, not the list of possibilities. Discuss both:

- **voxel imbalance:** background versus pancreas versus small lesion; nnU-Net patch sampling and segmentation loss;
- **baseline case imbalance:** subtype counts `62/106/84` produced inverse-frequency loss weights `1.35484/0.79245/1.0`;
- **v5 case imbalance:** deterministic balanced sampling with replacement plus unweighted cross-entropy, deliberately avoiding double correction.

Balanced sampling is balanced in expectation over draws, not necessarily inside every minibatch. Focal loss was excluded to avoid a second unbounded loss choice, and SMOTE was excluded because interpolated learned anatomical features are not validated patient examples. The mean-MIL comparison does not isolate the sampler, so do not claim that sampling caused the improvement.

### How did you control overfitting?

The baseline used augmentation, weight decay, dropout, label smoothing, a constrained head, and task-specific monitoring; it did not use early stopping. V5 used small heads, weight decay, dropout, label smoothing, repeated OOF evaluation, and a prospectively fixed schedule; it also did not use early stopping. The critical evidence is negative: selected-head resubstitution macro-F1 was `0.978711`, while mean repeated OOF macro-F1 was `0.507965`. That `0.470747` gap is severe overfitting. Repeated head folds do not solve representation-level overfitting because the frozen encoder was trained once on all 252 cases.

### Why did you not use a flat 0.5 or per-class thresholds?

Softmax classes are mutually exclusive and sum to one, so three separate binary 0.5 cutoffs can produce no class or conflicting classes. Amirfaham's threshold proposal was adapted to additive offsets on the three log-softmax scores, with class 1 fixed at zero for identifiability. Offsets were cross-fitted using train-only OOF logits and activated only if mean macro-F1 gained at least 0.01 without losing more than 0.02 minimum recall. They reduced mean macro-F1 from `0.507965` to `0.504068`, so the final offset vector is `[0,0,0]`. Call this decision-boundary tuning, not a probability-reliability result.

### Can you call v5 the best model?

No. It is the stronger of exactly two prospectively locked,
assignment-conforming neural heads under the declared train-only OOF rule. The
official macro-F1 of `0.52541507` strictly exceeded the baseline
`0.46399341`, so it became the selected classifier under that predeclared rule.
That is a meaningful controlled comparison, not evidence of global optimality.
The severe resubstitution-to-OOF gap, non-end-to-end OOF caveat, missed
classification targets, and failed speed gate must accompany the selection
claim.

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

1. use fully nested cross-validation that retrains the encoder inside each outer fold;
2. repeat backbone training across seeds, not just new-head fitting;
3. select stronger head regularization inside the nesting;
4. compare position-aware attention against the current permutation-invariant head;
5. inspect lesion-size-stratified and tile-ranking failures;
6. evaluate probability reliability and confidence; and
7. validate on an institutionally distinct permitted dataset.

### Can this be used clinically?

No. It is a take-home prototype on de-identified cropped ROIs, with one small fixed validation split and no external validation, probability-reliability study, reader study, or deployment risk analysis.

### What did the AI do, and what did you do?

Use the final facts from [AI_WORKFLOW.md](AI_WORKFLOW.md). A truthful concise answer:

> OpenAI Codex generated a substantial majority of the initial implementation and documentation and helped test, audit, and monitor the workflow. I owned the project direction, quality priorities, access, consequential decisions, review, and final submission. I specifically proposed class-specific thresholds and stronger imbalance handling; those became the locked multiclass log-score-offset test and balanced-sampling design. The offset result was negative and we kept it negative. I did not treat generated code as automatically correct: we used data audits, contract tests, deterministic smokes, saved configurations, cache/model hashes, recomputed OOF evidence, W&B records, and final artifact validation. The honest estimate is 85–95% of the initial repository content, based on file-level provenance rather than a post-formatting line count.

## Results defense sheet

Use the precision shown here only when it helps; the headline values are
`0.9202`, `0.6197`, and `0.5254`.

| Question | Evidence-backed answer |
|---|---|
| Which historical checkpoint produced the baseline? | `checkpoint_classification_rescue.pth` (joint epoch 200 plus 30 frozen-backbone head-only epochs). Its equal-weight whole/lesion/macro-F1 score was `0.6679416738`, the highest of the four baseline candidates. Relative to `checkpoint_final.pth`, it raised macro-F1 from `0.1333333333` to `0.4639934052`; historical baseline segmentation predictions were numerically identical because that rescue froze encoder/decoder weights and used the same export path. |
| What exactly was selected in v5? | The 101,391-parameter `neural_two_query_cross_attention_mil` head, selected only against the 117,263-parameter lesion-aware mean-MIL control. Cross-attention won all three complete OOF repeats: `0.4694335/0.5271997/0.5272614` versus `0.4526558/0.4011152/0.4054124`; mean `0.5079649` versus `0.4197278`. Say “stronger of two locked heads,” not “globally optimal.” |
| Is v5 OOF an end-to-end estimate? | No. Each case was excluded from its new-head fold fit, but the common encoder and rescue head had already seen all 252 training labels, and the rescue checkpoint had been selected on the historical validation pass. Winner selection over two heads can add optimism. |
| What is the clearest overfitting evidence? | Selected-head refit resubstitution macro-F1 `0.9787115` versus mean repeated OOF `0.5079649`, a `0.4707466` gap. Minimum class recall was `0.9622642` in resubstitution versus `0.4354839` across OOF repeats. |
| What happened to class-specific thresholds? | They became multiclass additive log-score offsets. Cross-fitted mean macro-F1 fell from `0.5079649` to `0.5040682`, so the activation gate failed and final offsets are `[0,0,0]`. |
| Final whole-pancreas Dice? | Mean `0.9201611779`, sample SD `0.0357812215`, 95% bootstrap CI `[0.9078978688, 0.9308171263]`; both `>=0.90` and `>=0.91` point thresholds were met. |
| Final lesion Dice? | Mean `0.6196623933`, sample SD `0.3206780089`, 95% bootstrap CI `[0.5148076710, 0.7165386879]`; both `>=0.27` and `>=0.31` point thresholds were met. |
| Final macro-F1 and replacement decision? | Official macro-F1 was `0.5254150702`, above the immutable baseline `0.4639934052`, so the predeclared classifier-replacement gate passed. Its 95% bootstrap CI was `[0.3583611739, 0.6735718120]`. The point estimate missed both `>=0.60` and `>=0.70`; accuracy was `0.5277777778`. The selected classifier is `neural_two_query_cross_attention_mil`. |
| Per-subtype precision/recall/F1? | Subtype 0 (support 9): `0.5556/0.5556/0.5556`; subtype 1 (support 15): `0.5000/0.3333/0.4000`; subtype 2 (support 12): `0.5294/0.7500/0.6207`. The subtype-1 recall of one third is the clearest final class-level weakness. |
| Strongest classification confusion? | With rows as references and columns as predictions, the final matrix was `[[5,2,2],[4,5,6],[0,3,9]]`. The largest directional error was subtype 1 -> 2 (six cases); only five of 15 subtype-1 cases were classified correctly. |
| Rule-selected strong qualitative examples? | The two historical baseline strong cases were `quiz_1_164` (whole `0.9043`, lesion `0.9245`, subtype `1 -> 1`) and `quiz_0_184` (whole `0.9405`, lesion `0.9232`, subtype `0 -> 1`). The latter shows that excellent masks do not guarantee correct subtype classification. |
| Most informative failure? | Both rule-selected weak cases had zero lesion overlap: `quiz_2_191` (whole `0.7935`, final subtype `2 -> 2`) had a spatially remote 20,221-voxel prediction versus a 4,248-voxel reference; `quiz_1_227` (whole `0.9347`, final subtype `1 -> 2`) predicted 287 lesion voxels versus 1,724 reference voxels. The first shows that a correct subtype does not imply a usable lesion mask. |
| Peak training VRAM and runtime? | No instrumented production-training peak was recorded. The planned-patch CUDA preflight measured `6,159 MiB` allocated and `6,716 MiB` reserved; a production `nvidia-smi` sample was about `6,849 MiB` steady usage, not a peak. The 200 joint epochs took `6:48:03.248` wall time (`6:32:30.457` summed epoch compute); the 30-epoch rescue added 3,750 updates and `1,754.212 s` summed epoch compute. |
| Selected test package? | One final v5 inference produced 72 masks and 72 subtype rows in `268.7635 s` (`3.7328 s/case`). The flat 73-file ZIP passed directory, archive, geometry, dtype, label-domain, naming, and row audits. SHA-256: `34afe1d74b70a24facceee890c03919bc5dbe036383206079fe221aa34ddd444`. |
| Strict speed outcome? | Installed stock nnU-Net averaged `236.7340 s`; the selected candidate averaged `281.2425 s`. Runtime reduction was `-18.8011%`, meaning the candidate was 18.8011% slower, so the `>=10%` speed gate failed. All stock/candidate masks were exactly equal in labels, geometry, and dtype. “Stock” means nnU-Net v2.8.1 under the same post-lock deterministic policy on the RTX 4060 Laptop GPU, with two timed repeats per arm in ABBA order. Stock uses three preprocessing/export workers; the custom path is serial. This result does not generalize across hardware. |
| What did determinism testing catch? | Initial full/pruned smokes differed by 3 and 10 boundary voxels because the predictor re-enabled cuDNN benchmarking. Later stock/candidate smokes differed by 5 and 15 because v5 cast stitched FP16 logits to FP32 before resampling. Both repairs were narrowly locked before edits. Final train-only conformance and the all-72 speed audit had zero hard-mask disagreement. |
| Was the stock-speed protocol followed perfectly? | No. The original stock lock prohibited later candidate changes and train-only timing smokes, but the retained conformance artifacts contained timing fields and the two bounded conformance repairs changed inference code. Those timings are diagnostic and excluded from the final speed calculation. Each repair was separately locked first, used only training inputs, and changed no weights, features, selected head, or offsets. |
| Why was official evaluation recovered instead of rerun? | The one locked model invocation completed 36/36 outputs, but Windows PowerShell 5.1 failed on two post-inference collection-count checks before references were opened; its ledger replacement also used an incompatible null backup. The saved 39-artifact digest and full runtime contract were independently verified. A separate prospective recovery lock permitted no inference and exactly one unchanged evaluator call. That saved-output continuation completed the gate; inference was never rerun. |
| Largest limitation? | Severe v5 refit overfitting plus non-end-to-end OOF. Official evidence still comes from one 36-case split, the source representation was trained once, the original result was known before v5, and there is no external validation. |

Historical baseline sources are `fixed_validation/checkpoint_selection.json`,
the selected checkpoint's `metrics.json`, `case_metrics.csv`, and
`runtime.json`. V5 train-only sources are `neural_case_head_selection.json`,
`neural_case_head_refit.json`, `neural_decision_calibration.json` (artifact
filename only; the rejected offsets are not a probability-reliability result),
`neural_case_head_fit_audit.json`, and W&B run `u03yz7ds`. Final claims come
from `official_evaluation_gates.json`, `official_evaluation_metrics.json`,
`selected_test_package_completion.json`, the selected-test validators,
`stock_inference_speed_audit.json`, and `report/report.md`. The realized Windows
PowerShell 5.1 delivery path used `Run-V5OfficialEvaluationRecovery.ps1` for the
zero-inference saved-output continuation and
`Run-V5LockedSelectedTestAndPackagePS51.ps1` for the single test/package run.
The qualitative cases were selected by the frozen lesion-Dice rule, not by
visual appeal.

## Code walk-through checklist

Before an interview, be able to locate and explain:

- [ ] dataset conversion and split manifest;
- [ ] classification target lookup;
- [ ] shared encoder and both output branches;
- [ ] combined loss and task weights;
- [ ] v5 case-bag extraction and why it uses predicted rather than reference lesion maps;
- [ ] the two locked v5 neural heads and their parameter counts;
- [ ] repeated OOF selection and its non-end-to-end caveat;
- [ ] balanced sampler and rejected log-score offsets;
- [ ] W&B logging calls;
- [ ] validation aggregation and empty-lesion rule;
- [ ] historical checkpoint selection versus the v5 replacement gate;
- [ ] deterministic inference bootstrap and FP16 export semantics;
- [ ] test prediction naming and CSV generation;
- [ ] archive validator;
- [ ] one test that caught or could catch a serious bug.

## Five-minute whiteboard rehearsal

1. Draw the shared ResEnc M encoder, segmentation decoder, baseline classifier, and v5 frozen-feature head.
2. Write the baseline combined loss and explain the separate v5 balanced-sampling loss.
3. Define whole-pancreas Dice, lesion Dice, macro-F1, and the multiclass offset rule.
4. Explain baseline validation history versus v5 train-only development and the one locked post-hoc gate.
5. State the 0.9787 versus 0.5080 overfit gap without minimizing it.
6. Give one successful decision, one failure, and one next experiment.

If any of these requires reading verbatim from the report, revisit the corresponding section and explain it in your own words.
