#!/usr/bin/env python3
"""Build or verify the minimal train-only private H100 input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pancreas_multitask.case_features import discover_train_cases, train_case_inventory_audit
from pancreas_multitask.classification_rescue import file_sha256

EXPECTED_PAYLOAD_DIGEST = "ec22997515eea6db1de655e06b788aa697ebf98413a49b70c53e5599172101b3"
EXPECTED_CASE_DIGEST = "bc9eee511612fce42d700b256d26793e6d2c8aabe06f4bf8699bb2b1abbf17bb"
EXPECTED_MODEL_HASHES = {
    "model/dataset.json": "4d35b17a700f2f92d0faa00f3db5cf056eb49f161db5a218ab1f641d22ae49ff",
    "model/plans.json": "8596a9cf4af2a0d6d2b8248e127f3514274d3bb13585491483fa630395f10a9f",
    "model/fold_0/checkpoint_classification_rescue.pth": (
        "d7248e8903fd1f062687ae33a22ad0374ca1b9927445443dcde55dcde128d116"
    ),
}


def _canonical_digest(entries: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        sorted(entries, key=lambda row: row["path"]),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inventory(bundle: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(bundle).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "bundle_manifest.json"
    ]


def verify(bundle: Path) -> dict[str, Any]:
    bundle = bundle.expanduser().resolve()
    manifest_path = bundle / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _inventory(bundle)
    if entries != manifest.get("entries"):
        raise ValueError("Live H100 payload does not exactly match bundle_manifest.json")
    digest = _canonical_digest(entries)
    if digest != EXPECTED_PAYLOAD_DIGEST or digest != manifest.get("payload_manifest_sha256"):
        raise ValueError("H100 payload canonical digest differs from the locked digest")
    paths = {row["path"] for row in entries}
    if len(entries) != 507 or not set(EXPECTED_MODEL_HASHES).issubset(paths):
        raise ValueError("H100 payload must contain exactly 504 train NIfTIs and 3 model files")
    for relative, expected_hash in EXPECTED_MODEL_HASHES.items():
        row = next(item for item in entries if item["path"] == relative)
        if row["sha256"] != expected_hash:
            raise ValueError(f"Locked model hash mismatch: {relative}")
    train_files = [path for path in paths if path.startswith("train/")]
    forbidden = [path for path in paths if "validation" in path.lower() or "test" in path.lower()]
    if len(train_files) != 504 or forbidden:
        raise ValueError("H100 payload contains a forbidden or malformed data scope")
    images = [path for path in train_files if path.endswith("_0000.nii.gz")]
    masks = [path for path in train_files if not path.endswith("_0000.nii.gz")]
    if len(images) != 252 or len(masks) != 252:
        raise ValueError("H100 payload must contain 252 CTs and 252 matching masks")
    image_stems = {path[: -len("_0000.nii.gz")] for path in images}
    mask_stems = {path[: -len(".nii.gz")] for path in masks}
    if image_stems != mask_stems:
        raise ValueError("H100 training CT and mask case sets differ")
    return {
        "status": "verified",
        "payload_manifest_sha256": digest,
        "payload_files": len(entries),
        "payload_bytes": sum(int(row["bytes"]) for row in entries),
        "training_cts": len(images),
        "training_masks": len(masks),
        "model_files": len(EXPECTED_MODEL_HASHES),
        "validation_or_test_files": 0,
    }


def build(source_train: Path, model: Path, output: Path) -> dict[str, Any]:
    source_train = source_train.expanduser().resolve()
    model = model.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite H100 input bundle: {output}")
    cases = discover_train_cases(source_train, expected_count=252)
    audit = train_case_inventory_audit(cases)
    if audit["case_ids_sha256_length_prefixed_sorted"] != EXPECTED_CASE_DIGEST:
        raise ValueError("Source training case inventory differs from the locked 252 cases")
    output.mkdir(parents=True)
    try:
        for case in cases:
            class_name = case.image_path.parent.name
            mask_path = case.image_path.with_name(f"{case.case_id}.nii.gz")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Training mask missing for {case.case_id}")
            destination = output / "train" / class_name
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(case.image_path, destination / case.image_path.name)
            shutil.copy2(mask_path, destination / mask_path.name)
        sources = {
            "model/dataset.json": model / "dataset.json",
            "model/plans.json": model / "plans.json",
            "model/fold_0/checkpoint_classification_rescue.pth": (
                model / "fold_0" / "checkpoint_classification_rescue.pth"
            ),
        }
        for relative, source in sources.items():
            if not source.is_file() or file_sha256(source) != EXPECTED_MODEL_HASHES[relative]:
                raise ValueError(f"Source model file is missing or hash-mismatched: {source}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        entries = _inventory(output)
        digest = _canonical_digest(entries)
        if digest != EXPECTED_PAYLOAD_DIGEST:
            raise ValueError(
                f"Built payload digest {digest} differs from locked {EXPECTED_PAYLOAD_DIGEST}"
            )
        manifest = {
            "schema_version": 1,
            "scope": "252_supplied_training_cases_only",
            "payload_manifest_sha256": digest,
            "case_id_digest": EXPECTED_CASE_DIGEST,
            "entries": entries,
        }
        (output / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return verify(output)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    builder = subparsers.add_parser("build")
    builder.add_argument("--source-train", type=Path, required=True)
    builder.add_argument("--model", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    verifier = subparsers.add_parser("verify")
    verifier.add_argument("--bundle", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = (
        build(args.source_train, args.model, args.output)
        if args.command == "build"
        else verify(args.bundle)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
