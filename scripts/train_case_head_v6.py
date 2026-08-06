#!/usr/bin/env python3
"""Run the frozen train-only V6 nested selection and three-seed final refit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pancreas_multitask.classification_rescue import file_sha256
from pancreas_multitask.v6_case_extractor import V6_H100_ORCHESTRATION_LOCK_SHA256
from pancreas_multitask.v6_case_head import V6_CASE_CANDIDATES
from pancreas_multitask.v6_case_training import (
    LOCKED_V6_SCHEDULE,
    V6_FINAL_SEEDS,
    V6CaseTensors,
    V6EpochEvaluation,
    V6InnerCandidateDecision,
    configure_v6_deterministic_execution,
    ensemble_v6_final_logits,
    evaluate_v6_train_only_screen,
    make_v6_inner_split,
    make_v6_outer_splits,
    predict_v6_logits,
    select_v6_final_family_and_epoch,
    select_v6_inner_candidate,
    select_v6_inner_epoch,
    train_v6_trajectory,
    v6_prediction_metrics,
)

EXPECTED_CASES = 252
EXPECTED_CLASS_COUNTS = [62, 106, 84]
CANDIDATE_INDEX = {candidate: index for index, candidate in enumerate(V6_CASE_CANDIDATES)}
NUMERIC_NAMES = (
    "spatial",
    "coordinates",
    "case_lesion_mass",
    "stage5_global_mean",
    "stage5_lesion_weighted_mean",
    "rescue_mean_logits",
    "rescue_mean_probabilities",
    "morphology",
    "labels",
)


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _numeric_dataset_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_array_sha256(arrays[name])))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary output exists: {temporary}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary output exists: {temporary}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _inner_seeds(repeat_seed: int, fold_index: int, candidate: str) -> tuple[int, int]:
    model = int(repeat_seed) + 100_000 * (int(fold_index) + 1)
    model += 1_000 * (CANDIDATE_INDEX[candidate] + 1) + 11
    return model, model + 10_000_000


def _outer_seeds(repeat_seed: int, fold_index: int, candidate: str) -> tuple[int, int]:
    model = int(repeat_seed) + 200_000 * (int(fold_index) + 1)
    model += 1_000 * (CANDIDATE_INDEX[candidate] + 1) + 29
    return model, model + 20_000_000


def _load_features(path: Path, device: torch.device) -> tuple[V6CaseTensors, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in NUMERIC_NAMES}
        case_ids = np.asarray(payload["case_ids"]).astype(str)
        source_hashes = np.asarray(payload["source_image_sha256"]).astype(str)
        embedded_hash = str(payload["numeric_dataset_sha256"].item())
        order_rule = str(payload["canonical_order_rule"].item())
    if arrays["labels"].shape != (EXPECTED_CASES,):
        raise ValueError("V6 training requires exactly 252 extracted cases")
    counts = np.bincount(arrays["labels"].astype(np.int64), minlength=3).tolist()
    if counts != EXPECTED_CLASS_COUNTS:
        raise ValueError("V6 extracted class counts differ from 62/106/84")
    if case_ids.shape != (EXPECTED_CASES,) or source_hashes.shape != (EXPECTED_CASES,):
        raise ValueError("V6 feature provenance vectors are not aligned")
    if len(set(case_ids.tolist())) != EXPECTED_CASES:
        raise ValueError("V6 feature provenance contains duplicate case IDs")
    if len(set(source_hashes.tolist())) != EXPECTED_CASES:
        raise ValueError("V6 canonical content hashes are not unique")
    if source_hashes.tolist() != sorted(source_hashes.tolist()):
        raise ValueError("V6 feature rows are not in canonical content-hash order")
    if order_rule != "ascending_sha256_of_raw_training_ct_bytes":
        raise ValueError("V6 feature row-order rule differs from the prospective lock")
    actual_hash = _numeric_dataset_sha256(arrays)
    if actual_hash != embedded_hash:
        raise ValueError("V6 numeric feature-dataset hash verification failed")
    float_arrays = {
        name: arrays[name].astype(np.float32)
        for name in NUMERIC_NAMES
        if name != "labels"
    }
    rescue_sums = float_arrays["rescue_mean_probabilities"].sum(
        axis=1, keepdims=True
    )
    if not np.isfinite(rescue_sums).all() or np.any(rescue_sums <= 0):
        raise ValueError("Cached V6 rescue-probability rows have invalid sums")
    float_arrays["rescue_mean_probabilities"] /= rescue_sums
    float_tensors = {
        name: torch.from_numpy(value).to(device) for name, value in float_arrays.items()
    }
    tensors = V6CaseTensors(
        **float_tensors,
        labels=torch.from_numpy(arrays["labels"].astype(np.int64)).to(device),
    )
    return tensors, {
        "numeric_dataset_sha256": actual_hash,
        "feature_file_sha256": file_sha256(path),
        "case_ids": case_ids,
        "source_image_sha256": source_hashes,
        "class_counts": counts,
        "canonical_order_rule": order_rule,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def run(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    deterministic = configure_v6_deterministic_execution(device)
    lock_path = ROOT / "configs" / "v6_h100_orchestration_lock.json"
    if file_sha256(lock_path) != V6_H100_ORCHESTRATION_LOCK_SHA256:
        raise ValueError("V6 H100 orchestration lock hash differs from the code binding")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_bundle_path = output / "v6_final_case_head_bundle.pth"
    if final_bundle_path.exists():
        raise FileExistsError(f"Refusing to overwrite final V6 bundle: {final_bundle_path}")
    features_path = args.features.expanduser().resolve()
    tensors, feature_audit = _load_features(features_path, device)
    labels = tensors.labels.detach().cpu().numpy().astype(np.int64, copy=False)
    audited_order = np.arange(EXPECTED_CASES, dtype=np.int64)
    outer_splits = make_v6_outer_splits(labels, audited_order=audited_order)
    repeat_oof_logits = np.full((3, EXPECTED_CASES, 3), np.nan, dtype=np.float32)
    coverage = np.zeros((3, EXPECTED_CASES), dtype=np.int64)
    fold_audits: list[dict[str, Any]] = []
    selected_decisions: list[V6InnerCandidateDecision] = []
    started_at_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    for split_number, split in enumerate(outer_splits, start=1):
        inner_train, inner_validation, inner_split_seed = make_v6_inner_split(
            labels,
            split.train_indices,
            repeat_seed=split.repeat_seed,
            fold_index=split.fold_index,
        )
        candidate_decisions: list[V6InnerCandidateDecision] = []
        candidate_audits: list[dict[str, Any]] = []
        for candidate in V6_CASE_CANDIDATES:
            model_seed, sampler_seed_base = _inner_seeds(
                split.repeat_seed, split.fold_index, candidate
            )
            inner_model, trajectory = train_v6_trajectory(
                candidate,
                tensors,
                inner_train,
                epochs=LOCKED_V6_SCHEDULE.maximum_inner_epochs,
                model_seed=model_seed,
                sampler_seed_base=sampler_seed_base,
                evaluation_indices=inner_validation,
            )
            evaluations = [
                V6EpochEvaluation(
                    epoch_index=int(row["epoch_index"]),
                    macro_f1=float(row["evaluation"]["macro_f1"]),
                    nll=float(row["evaluation"]["nll"]),
                )
                for row in trajectory["epoch_history"]
            ]
            selected_epoch = select_v6_inner_epoch(evaluations)
            decision = V6InnerCandidateDecision(
                candidate_id=candidate,
                selected_epoch_count=selected_epoch.epoch_count,
                macro_f1=selected_epoch.macro_f1,
                nll=selected_epoch.nll,
            )
            candidate_decisions.append(decision)
            candidate_audits.append(
                {
                    "decision": asdict(decision),
                    "model_seed": model_seed,
                    "sampler_seed_base": sampler_seed_base,
                    "trajectory": trajectory,
                }
            )
            del inner_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        selected, selection_basis = select_v6_inner_candidate(candidate_decisions)
        selected_decisions.append(selected)
        outer_model_seed, outer_sampler_seed_base = _outer_seeds(
            split.repeat_seed, split.fold_index, selected.candidate_id
        )
        outer_model, outer_trajectory = train_v6_trajectory(
            selected.candidate_id,
            tensors,
            split.train_indices,
            epochs=selected.selected_epoch_count,
            model_seed=outer_model_seed,
            sampler_seed_base=outer_sampler_seed_base,
        )
        held_logits = predict_v6_logits(
            outer_model,
            tensors,
            split.held_indices,
            batch_size=LOCKED_V6_SCHEDULE.batch_size,
        )
        repeat_oof_logits[split.repeat_index, split.held_indices] = held_logits
        coverage[split.repeat_index, split.held_indices] += 1
        held_metrics = v6_prediction_metrics(labels[split.held_indices], held_logits)
        fold_audits.append(
            {
                "split_number": split_number,
                "repeat_index": split.repeat_index,
                "repeat_seed": split.repeat_seed,
                "fold_index": split.fold_index,
                "outer_train_count": int(split.train_indices.size),
                "outer_held_count": int(split.held_indices.size),
                "inner_train_count": int(inner_train.size),
                "inner_validation_count": int(inner_validation.size),
                "inner_split_seed": inner_split_seed,
                "candidate_audits": candidate_audits,
                "selected_decision": asdict(selected),
                "selection_basis": selection_basis,
                "outer_model_seed": outer_model_seed,
                "outer_sampler_seed_base": outer_sampler_seed_base,
                "outer_trajectory": outer_trajectory,
                "held_metrics": held_metrics,
            }
        )
        print(
            f"Nested V6 fold {split_number}/{len(outer_splits)}: "
            f"{selected.candidate_id}, epoch {selected.selected_epoch_count}",
            flush=True,
        )
        del outer_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not np.all(coverage == 1) or not np.isfinite(repeat_oof_logits).all():
        raise RuntimeError("V6 OOF coverage is incomplete")
    final_family, final_epoch_count, final_selection = select_v6_final_family_and_epoch(
        selected_decisions
    )
    final_states: list[dict[str, torch.Tensor]] = []
    final_state_hashes: list[str] = []
    final_logits_rows: list[np.ndarray] = []
    final_trajectories: list[dict[str, Any]] = []
    all_rows = np.arange(EXPECTED_CASES, dtype=np.int64)
    for final_seed in V6_FINAL_SEEDS:
        sampler_seed_base = int(final_seed) + 30_000_000
        model, trajectory = train_v6_trajectory(
            final_family,
            tensors,
            all_rows,
            epochs=final_epoch_count,
            model_seed=int(final_seed),
            sampler_seed_base=sampler_seed_base,
        )
        logits = predict_v6_logits(
            model,
            tensors,
            all_rows,
            batch_size=LOCKED_V6_SCHEDULE.batch_size,
        )
        final_logits_rows.append(logits)
        final_state_hashes.append(str(trajectory["final_state_sha256"]))
        final_trajectories.append(trajectory)
        final_states.append(
            {
                name: value.detach().cpu().contiguous().clone()
                for name, value in model.state_dict().items()
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    stacked_final_logits = torch.from_numpy(np.stack(final_logits_rows))
    resub_logits_tensor, resub_probabilities_tensor = ensemble_v6_final_logits(
        stacked_final_logits
    )
    resub_logits = resub_logits_tensor.numpy()
    resub_probabilities = resub_probabilities_tensor.numpy()
    resub_metrics = v6_prediction_metrics(labels, resub_logits)
    train_only_screen = evaluate_v6_train_only_screen(
        labels,
        repeat_oof_logits,
        coverage,
        resubstitution_macro_f1=float(resub_metrics["macro_f1"]),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    bundle = {
        "schema_version": 1,
        "candidate_id": final_family,
        "epoch_count": final_epoch_count,
        "final_model_seeds": list(V6_FINAL_SEEDS),
        "final_sampler_seed_bases": [int(seed) + 30_000_000 for seed in V6_FINAL_SEEDS],
        "state_dicts": final_states,
        "final_state_sha256": final_state_hashes,
        "numeric_train_dataset_sha256": feature_audit["numeric_dataset_sha256"],
        "orchestration_lock_sha256": V6_H100_ORCHESTRATION_LOCK_SHA256,
        "class_offsets": [0.0, 0.0, 0.0],
        "inference_rule": "mean_three_raw_logits_then_one_softmax",
    }
    _atomic_torch_save(final_bundle_path, bundle)
    oof_path = output / "v6_train_only_predictions.npz"
    _atomic_savez(
        oof_path,
        labels=labels,
        case_ids=feature_audit["case_ids"],
        source_image_sha256=feature_audit["source_image_sha256"],
        repeat_oof_logits=repeat_oof_logits,
        coverage_counts=coverage,
        final_resubstitution_logits=resub_logits,
        final_resubstitution_probabilities=resub_probabilities,
    )
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        }
    audit = {
        "schema_version": 1,
        "status": "complete",
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0.0,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "environment": environment,
        "deterministic_execution": deterministic,
        "orchestration_lock_sha256": V6_H100_ORCHESTRATION_LOCK_SHA256,
        "feature_input": {key: value for key, value in feature_audit.items() if key != "case_ids"},
        "schedule": asdict(LOCKED_V6_SCHEDULE),
        "seed_formulas": {
            "inner_model": "repeat_seed+100000*(fold+1)+1000*(candidate_index+1)+11",
            "inner_sampler": "inner_model+10000000",
            "outer_model": "repeat_seed+200000*(fold+1)+1000*(candidate_index+1)+29",
            "outer_sampler": "outer_model+20000000",
            "final_sampler": "final_model_seed+30000000",
        },
        "fold_audits": fold_audits,
        "final_selection": {
            "candidate_id": final_family,
            "epoch_count": final_epoch_count,
            **final_selection,
        },
        "final_trajectories": final_trajectories,
        "final_state_sha256": final_state_hashes,
        "resubstitution_metrics": resub_metrics,
        "train_only_screen": train_only_screen,
        "validation_or_test_used": False,
        "outputs": {
            "bundle": {
                "path": final_bundle_path.name,
                "sha256": file_sha256(final_bundle_path),
                "bytes": final_bundle_path.stat().st_size,
            },
            "predictions": {
                "path": oof_path.name,
                "sha256": file_sha256(oof_path),
                "bytes": oof_path.stat().st_size,
            },
        },
    }
    audit_path = output / "training_audit.json"
    _atomic_json(audit_path, audit)
    print(json.dumps(train_only_screen, indent=2, sort_keys=True), flush=True)
    return final_bundle_path


def main() -> int:
    result = run(build_parser().parse_args())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
