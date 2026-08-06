#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash scripts/run_h100_v6.sh /absolute/path/input_bundle /absolute/path/output" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_BUNDLE="$(realpath "$1")"
OUTPUT_ROOT="$(realpath -m "$2")"
VENV_ROOT="${REPO_ROOT}/.venv-h100"

if [[ ! -d "${INPUT_BUNDLE}" ]]; then
  echo "input bundle does not exist: ${INPUT_BUNDLE}" >&2
  exit 2
fi
if [[ "${OUTPUT_ROOT}" == "${INPUT_BUNDLE}" || "${OUTPUT_ROOT}" == "${REPO_ROOT}" ]]; then
  echo "output must be a separate directory" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "output already exists; use a new empty path: ${OUTPUT_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee "${OUTPUT_ROOT}/run.log") 2>&1

echo "== H100 preflight =="
date -u +%FT%TZ
nvidia-smi
python3.12 --version

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  python3.12 -m venv "${VENV_ROOT}"
fi
"${VENV_ROOT}/bin/python" -m pip install --upgrade pip
"${VENV_ROOT}/bin/python" -m pip install -r "${REPO_ROOT}/requirements-h100.txt"
"${VENV_ROOT}/bin/python" -m pip check

export PYTHONPATH="${REPO_ROOT}/src"
export PYTHONHASHSEED=0
export nnUNet_extTrainer="${REPO_ROOT}/src"

"${VENV_ROOT}/bin/python" "${REPO_ROOT}/scripts/h100_bundle.py" verify \
  --bundle "${INPUT_BUNDLE}"

"${VENV_ROOT}/bin/python" -m pytest -q \
  "${REPO_ROOT}/tests/test_v6_case_head.py" \
  "${REPO_ROOT}/tests/test_v6_case_training.py" \
  "${REPO_ROOT}/tests/test_v6_case_extractor.py"

mkdir -p "${OUTPUT_ROOT}/features"
"${VENV_ROOT}/bin/python" "${REPO_ROOT}/scripts/extract_train_case_features_v6.py" \
  --train-root "${INPUT_BUNDLE}/train" \
  --model "${INPUT_BUNDLE}/model" \
  --output "${OUTPUT_ROOT}/features" \
  --device cuda \
  --resume

export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "${OUTPUT_ROOT}/training"
"${VENV_ROOT}/bin/python" "${REPO_ROOT}/scripts/train_case_head_v6.py" \
  --features "${OUTPUT_ROOT}/features/train_v6_features.npz" \
  --output "${OUTPUT_ROOT}/training" \
  --device cuda

"${VENV_ROOT}/bin/python" - <<'PY' "${OUTPUT_ROOT}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
extraction = json.loads((root / "features" / "extraction_audit.json").read_text())
training = json.loads((root / "training" / "training_audit.json").read_text())
summary = {
    "status": "complete",
    "total_gpu_hours": extraction["gpu_hours"] + training["gpu_hours"],
    "extraction_gpu_hours": extraction["gpu_hours"],
    "training_gpu_hours": training["gpu_hours"],
    "peak_cuda_memory_bytes": max(
        extraction["peak_cuda_memory_bytes"], training["peak_cuda_memory_bytes"]
    ),
    "train_only_screen": training["train_only_screen"],
    "final_selection": training["final_selection"],
    "final_bundle": training["outputs"]["bundle"],
}
(root / "RESULTS_SUMMARY.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

nvidia-smi
date -u +%FT%TZ
echo "Return the complete output directory: ${OUTPUT_ROOT}"
