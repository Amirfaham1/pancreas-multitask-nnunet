# Dataset preparation and audit

The preparation step is deliberately non-destructive. Source NIfTI files are
never edited: CT images are copied into a generated nnU-Net raw dataset, while
segmentation masks are loaded, validated, rounded to exact integer labels, and
saved as new `uint8` NIfTI files with their affine, voxel spacing, qform, and
sform preserved.

## Source inventory

Filename-level discovery of the supplied archive found:

| Supplied split | subtype0 | subtype1 | subtype2 | Total |
|---|---:|---:|---:|---:|
| Train | 62 | 106 | 84 | 252 |
| Validation | 9 | 15 | 12 | 36 |
| Test | — | — | — | 72 |

The script rejects missing image/mask pairs, duplicate case identifiers,
unexpected filenames, overlap between train/validation/test, non-3D images,
image-mask geometry mismatches, non-finite masks, labels outside `{0, 1, 2}`,
or mask values farther than the configured tolerance from an integer.

## Why masks are rewritten

Some source masks store nominal label `1` as a nearby floating-point value
(for example, approximately `1.0000153`). nnU-Net checks segmentation labels
as exact discrete values, so passing these files through unchanged can fail
dataset verification. `prepare_dataset.py` allows only near-integer values,
rounds them, checks the result is in `{0, 1, 2}`, and stores the generated mask
as `uint8`. This is a representation repair, not a spatial resampling: no voxel
locations or image geometry are changed.

The generated `data_audit.json` records the observed values, number of masks
and voxels corrected, maximum rounding error, class distribution, and all
validated counts. It is the evidence source for the report; values should not
be copied into the report until a real run has produced this file.

## Commands

First perform a full read-only audit. `--dry-run` does not create the output
root:

```powershell
py -3 scripts/prepare_dataset.py `
  --source "ML-Quiz-3DMedImg/ML-Quiz-3DMedImg" `
  --output-root "data/nnUNet_raw" `
  --dataset-id 501 `
  --dataset-name PancreasMultitask `
  --dry-run
```

For the final training layout, place both supplied labelled splits in
`imagesTr`/`labelsTr`. Their roles remain separate in `splits_final.json` and
`split_manifest.json`:

```powershell
py -3 scripts/prepare_dataset.py `
  --source "ML-Quiz-3DMedImg/ML-Quiz-3DMedImg" `
  --output-root "data/nnUNet_raw" `
  --dataset-id 501 `
  --dataset-name PancreasMultitask
```

If strict train-only fingerprinting/planning is desired, use this two-phase
sequence:

1. Run preparation with `--validation-layout separate`. The 252 training cases
   go to `imagesTr`/`labelsTr`; the 36 validation cases are converted into
   `imagesVal`/`labelsVal`, which nnU-Net planning ignores.
2. Extract the fingerprint and generate plans.
3. Rerun preparation without `--validation-layout separate`. This safely adds
   the supplied validation cases to `imagesTr`/`labelsTr` without changing the
   source or deleting the generated plans.
4. Preprocess all labelled cases using those train-only-derived plans. Copy
   `splits_final.json` into the corresponding `nnUNet_preprocessed` dataset
   directory before training so nnU-Net uses the supplied 252/36 split.

## Generated metadata

- `dataset.json`: nnU-Net v2 modalities, segmentation labels, and case count.
- `splits_final.json`: one manual fold preserving the supplied split.
- `split_manifest.json`: explicit train, validation, test, planning, and
  `imagesTr` case lists.
- `classification_labels.json`: direct `case_id -> class_id` lookup.
- `classification_manifest.json` and `classification_labels.csv`: class name,
  integer target, supplied split, and filenames for every labelled case.
- `data_audit.json`: machine-readable conversion and validation evidence.

The `splits_final.json` file belongs in the matching directory below
`nnUNet_preprocessed` at training time; nnU-Net does not read manual splits from
the raw dataset directory.
