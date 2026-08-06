from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_report.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("powershell.exe")


def _run_final_guard(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("Windows PowerShell is unavailable")

    report = tmp_path / "report.md"
    report.write_text(source, encoding="utf-8")
    dummy_executable = tmp_path / "unused.exe"
    dummy_executable.write_bytes(b"")

    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BUILD_SCRIPT),
            "-InputPath",
            str(report),
            "-OutputPath",
            str(tmp_path / "report.pdf"),
            "-PandocPath",
            str(dummy_executable),
            "-TectonicPath",
            str(dummy_executable),
            "-Final",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Official result: `V5_OFFICIAL_MACRO_F1`\n", "unresolved v5 result token"),
        ("Status: TODO_FINAL_TABLE\n", "TODO/TBD/PLACEHOLDER"),
        ("DRAFT - NOT READY FOR SUBMISSION\n", "DRAFT warning"),
    ],
)
def test_final_build_rejects_unresolved_report_markers(
    tmp_path: Path,
    source: str,
    expected: str,
) -> None:
    result = _run_final_guard(tmp_path, source)

    assert result.returncode != 0
    assert expected in (result.stdout + result.stderr)


def test_current_report_cannot_build_final_while_v5_tokens_remain(tmp_path: Path) -> None:
    result = _run_final_guard(
        tmp_path,
        (REPO_ROOT / "report" / "report.md").read_text(encoding="utf-8"),
    )

    assert result.returncode != 0
    assert "unresolved v5 result token" in (result.stdout + result.stderr)

