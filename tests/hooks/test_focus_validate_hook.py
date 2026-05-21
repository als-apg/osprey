"""Tests for the osprey_focus_validate hook.

This UserPromptSubmit hook reads ``_agent_data/focus_state.txt``, drops
lines that reference deleted artifact IDs (by cross-checking
``_agent_data/artifacts/artifacts.json``), and prints the cleaned content
to stdout. It is defensive — never blocks a prompt — so all error paths
fail open with exit code 0.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK_SCRIPT = (
    Path(__file__).parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "claude_code"
    / "claude"
    / "hooks"
    / "osprey_focus_validate.py"
)


def _run(project_dir: Path) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    result = subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input="",
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


def _seed(project_dir: Path, focus: str, entries: list[dict] | None) -> None:
    agent_data = project_dir / "_agent_data"
    (agent_data / "artifacts").mkdir(parents=True, exist_ok=True)
    (agent_data / "focus_state.txt").write_text(focus)
    if entries is not None:
        (agent_data / "artifacts" / "artifacts.json").write_text(json.dumps({"entries": entries}))


@pytest.mark.unit
def test_focus_validator_drops_stale_ids(tmp_path):
    """Lines referencing missing artifact IDs are dropped; valid lines kept."""
    focus = (
        "[Gallery Focus]\n"
        '  artifact: "Live Plot" (id=valid123)\n'
        '  pinned:   "Deleted Plot" (id=stale456)\n'
        '  pinned:   "Other Live" (id=valid789)\n'
    )
    _seed(
        tmp_path,
        focus,
        entries=[{"id": "valid123"}, {"id": "valid789"}],
    )

    rc, stdout, _ = _run(tmp_path)
    assert rc == 0
    assert "id=valid123" in stdout
    assert "id=valid789" in stdout
    assert "id=stale456" not in stdout
    assert stdout.startswith("[Gallery Focus]")


@pytest.mark.unit
def test_focus_validator_drops_all_returns_empty(tmp_path):
    """When every line is stale, output is empty (header is also dropped)."""
    focus = '[Gallery Focus]\n  artifact: "Gone" (id=stale1)\n  pinned:   "Also Gone" (id=stale2)\n'
    _seed(tmp_path, focus, entries=[])

    rc, stdout, _ = _run(tmp_path)
    assert rc == 0
    assert stdout == ""


@pytest.mark.unit
def test_focus_validator_fails_open_on_missing_index(tmp_path):
    """If artifacts.json is missing, hook prints original contents and exits 0."""
    focus = '[Gallery Focus]\n  artifact: "Anything" (id=whatever)\n'
    _seed(tmp_path, focus, entries=None)  # No artifacts.json

    rc, stdout, _ = _run(tmp_path)
    assert rc == 0
    assert stdout == focus


@pytest.mark.unit
def test_focus_validator_fails_open_on_malformed_index(tmp_path):
    """Malformed artifacts.json is treated as 'unknown' → fail open."""
    agent_data = tmp_path / "_agent_data"
    (agent_data / "artifacts").mkdir(parents=True, exist_ok=True)
    focus = '[Gallery Focus]\n  artifact: "Anything" (id=whatever)\n'
    (agent_data / "focus_state.txt").write_text(focus)
    (agent_data / "artifacts" / "artifacts.json").write_text("not json {{{")

    rc, stdout, _ = _run(tmp_path)
    assert rc == 0
    assert stdout == focus


@pytest.mark.unit
def test_focus_validator_empty_focus_file(tmp_path):
    """Empty focus_state.txt yields empty output (no spurious header)."""
    _seed(tmp_path, "", entries=[{"id": "anything"}])

    rc, stdout, _ = _run(tmp_path)
    assert rc == 0
    assert stdout == ""


@pytest.mark.unit
def test_focus_validator_missing_focus_file(tmp_path):
    """No focus_state.txt at all yields empty output, exit 0."""
    (tmp_path / "_agent_data" / "artifacts").mkdir(parents=True)
    (tmp_path / "_agent_data" / "artifacts" / "artifacts.json").write_text(
        json.dumps({"entries": []})
    )

    rc, stdout, _ = _run(tmp_path)
    assert rc == 0
    assert stdout == ""
