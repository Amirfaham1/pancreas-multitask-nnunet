#!/usr/bin/env python3
"""Assemble public-safe and private V7 submission files with SHA-256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--results-zip", type=Path, required=True)
    parser.add_argument("--wandb-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "delivery" / "v7_final")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_tree(target: Path, files: list[tuple[Path, str]]) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, member in sorted(files, key=lambda item: item[1]):
            archive.write(source, member)


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output must not exist or must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    report = args.report.expanduser().resolve()
    results = args.results_zip.expanduser().resolve()
    wandb_directory = args.wandb_directory.expanduser().resolve()
    for path in (report, results):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not wandb_directory.is_dir():
        raise NotADirectoryError(wandb_directory)

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    source_archive = output / "pancreas-multitask-v7-source.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={source_archive}", "HEAD"],
        cwd=ROOT,
        check=True,
    )

    report_target = output / "Amirfaham_Fallahpour_results.pdf"
    result_target = output / "Amirfaham_Fallahpour_results.zip"
    shutil.copy2(report, report_target)
    shutil.copy2(results, result_target)

    evidence_files: list[tuple[Path, str]] = []
    evidence_root = ROOT / "docs" / "evidence" / "v7"
    for source in evidence_root.rglob("*"):
        if source.is_file():
            evidence_files.append((source, source.relative_to(ROOT).as_posix()))
    for relative in (
        Path("docs/V7_FINAL_EVIDENCE.md"),
        Path("models/v7/classifier_stage1_view6_scale1.joblib"),
        Path("SUBMISSION.md"),
    ):
        evidence_files.append((ROOT / relative, relative.as_posix()))
    evidence_archive = output / "pancreas-multitask-v7-evidence.zip"
    zip_tree(evidence_archive, evidence_files)

    wandb_files = [
        (source, source.relative_to(wandb_directory).as_posix())
        for source in wandb_directory.rglob("*")
        if source.is_file()
    ]
    wandb_archive = output / "wandb-v7-offline-runs.zip"
    zip_tree(wandb_archive, wandb_files)

    deliverables = [source_archive, evidence_archive, report_target, result_target, wandb_archive]
    hashes = {path.name: sha256(path) for path in deliverables}
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": revision,
        "git_branch": branch,
        "tests": {"passed": 463, "failed": 0},
        "validation": {
            "whole_pancreas_dice": 0.9201569021436445,
            "lesion_dice": 0.6196343544844362,
            "macro_f1": 0.7445103205972771,
            "accuracy_gates_passed": True,
            "speed_gate_passed": False,
        },
        "public_github_release": [
            source_archive.name,
            evidence_archive.name,
            report_target.name,
        ],
        "private_assessment_upload": [result_target.name],
        "local_wandb_sync_bundle": wandb_archive.name,
        "sha256": hashes,
    }
    manifest_path = output / "DELIVERY_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    all_hashes = hashes | {manifest_path.name: sha256(manifest_path)}
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(all_hashes.items())),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
