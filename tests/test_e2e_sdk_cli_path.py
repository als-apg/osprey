"""Every agent run the e2e suite starts executes the pinned Claude Code.

The runtime hands the Agent SDK an explicit ``cli_path`` (see
``osprey.agent_runner.launcher.agent_cli_path``); without one the SDK runs the
binary bundled inside its own package, a different build from the one the
project pins. A model that build does not know is priced as an unknown one, so
a per-query budget ends the run after its first turn and the test reads an
empty answer as the agent's fault.

The e2e suite and the benchmark scripts drive the SDK directly, outside the
harness adapter, so each ``ClaudeAgentOptions(...)`` they build has to name the
executable itself. This scan holds that line for every call site, including
ones added later.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_TREES = (REPO_ROOT / "tests" / "e2e", REPO_ROOT / "scripts" / "benchmark")


def _options_calls_without_cli_path(source: str) -> list[int]:
    """Line numbers of ``ClaudeAgentOptions(...)`` calls that pass no ``cli_path``."""
    if "ClaudeAgentOptions" not in source:
        return []
    missing: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "ClaudeAgentOptions":
            continue
        if not any(kw.arg == "cli_path" for kw in node.keywords):
            missing.append(node.lineno)
    return missing


def _scanned_files() -> list[Path]:
    return sorted(p for tree in SCANNED_TREES for p in tree.rglob("*.py"))


def test_scan_finds_the_call_sites() -> None:
    """The scan reaches the trees it guards, so a pass is not an empty one."""
    assert any("ClaudeAgentOptions" in p.read_text(encoding="utf-8") for p in _scanned_files())


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ClaudeAgentOptions(model='m')", [1]),
        ("sdk.ClaudeAgentOptions(model='m')", [1]),
        ("ClaudeAgentOptions(model='m', cli_path=p)", []),
        ("other(model='m')", []),
    ],
)
def test_scan_detects_a_missing_cli_path(source: str, expected: list[int]) -> None:
    assert _options_calls_without_cli_path(source) == expected


def test_every_sdk_run_names_the_pinned_cli() -> None:
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in _scanned_files()
        for line in _options_calls_without_cli_path(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "These ClaudeAgentOptions(...) calls pass no cli_path, so the SDK would run the "
        "Claude Code bundled in its own package instead of the pinned one: " + ", ".join(offenders)
    )
