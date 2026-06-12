"""Tests for the osprey_config_drift.py SessionStart hook.

The hook warns (never blocks) when config.yml has drifted from the rendered
.claude/settings.json. It is stdlib-only and reads CLAUDE_PROJECT_DIR, so these
tests drive it as a subprocess with an explicitly controlled environment and
pinned mtimes — relying on the shared hook_runner (which copies os.environ) would
leak the parent's CLAUDE_PROJECT_DIR into the subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = (
    Path(__file__).parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "claude_code"
    / "claude"
    / "hooks"
    / "osprey_config_drift.py"
)


def _run(project_dir: Path):
    """Run the drift hook against project_dir; return (returncode, stdout, stderr)."""
    env = {k: v for k, v in os.environ.items() if k not in {"OSPREY_CONFIG", "CONFIG_FILE"}}
    env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input="{}",
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


def _make_project(tmp_path, *, writes_enabled: bool, channel_write_denied: bool):
    """Create config.yml + .claude/settings.json with chosen (possibly mismatched) state."""
    project_dir = tmp_path
    (project_dir / ".claude").mkdir(parents=True, exist_ok=True)
    config_path = project_dir / "config.yml"
    settings_path = project_dir / ".claude" / "settings.json"

    config_path.write_text(
        "control_system:\n"
        "  type: mock\n"
        f"  writes_enabled: {'true' if writes_enabled else 'false'}\n",
        encoding="utf-8",
    )
    deny = ["Bash", "Edit"]
    if channel_write_denied:
        deny.append("mcp__controls__channel_write")
    settings_path.write_text(
        json.dumps({"permissions": {"deny": deny}}, indent=2), encoding="utf-8"
    )
    return config_path, settings_path


def _set_mtimes(config_path: Path, settings_path: Path, *, config_newer: bool):
    """Pin mtimes so the mtime drift signal is deterministic."""
    base = 1_700_000_000
    if config_newer:
        os.utime(settings_path, (base, base))
        os.utime(config_path, (base + 100, base + 100))
    else:
        os.utime(config_path, (base, base))
        os.utime(settings_path, (base + 100, base + 100))


def test_silent_when_in_sync(tmp_path):
    config_path, settings_path = _make_project(
        tmp_path, writes_enabled=False, channel_write_denied=True
    )
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = _run(tmp_path)
    assert code == 0
    assert err.strip() == ""
    assert out.strip() == ""


def test_warns_on_writes_mismatch(tmp_path):
    # config says writes enabled, but artifacts still deny the write → #244 case.
    config_path, settings_path = _make_project(
        tmp_path, writes_enabled=True, channel_write_denied=True
    )
    # Even with settings newer (mtime in sync), the targeted check still fires.
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = _run(tmp_path)
    assert code == 0
    assert "writes_enabled: true" in err
    assert "regen" in err.lower()
    # Also surfaced as SessionStart additionalContext on stdout.
    payload = json.loads(out)
    assert "writes_enabled: true" in payload["hookSpecificOutput"]["additionalContext"]


def test_warns_when_config_disables_writes_but_agent_permits(tmp_path):
    # The safety-critical direction: config.yml turned writes OFF, but the stale
    # artifacts still permit them — the agent could write hardware the operator
    # believes is locked out.
    config_path, settings_path = _make_project(
        tmp_path, writes_enabled=False, channel_write_denied=False
    )
    # Settings newer than config: the targeted check must fire on its own.
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = _run(tmp_path)
    assert code == 0
    assert "writes_enabled: false" in err
    assert "permits writes" in err
    assert "regen" in err.lower()
    payload = json.loads(out)
    assert "permits writes" in payload["hookSpecificOutput"]["additionalContext"]


def test_warns_on_mtime_drift(tmp_path):
    # writes_enabled is consistent, but config.yml was edited after the last regen.
    config_path, settings_path = _make_project(
        tmp_path, writes_enabled=False, channel_write_denied=True
    )
    _set_mtimes(config_path, settings_path, config_newer=True)

    code, out, err = _run(tmp_path)
    assert code == 0
    assert "changed since" in err
    assert "regen" in err.lower()


def test_fails_open_when_artifacts_missing(tmp_path):
    # No config.yml / settings.json at all → silent, never blocks the session.
    code, out, err = _run(tmp_path)
    assert code == 0
    assert err.strip() == ""
    assert out.strip() == ""


def test_fails_open_on_malformed_permissions(tmp_path):
    # Hand-edited settings.json with `permissions: null` must not crash the targeted
    # check; it falls back to the mtime signal (here pinned in-sync → silent).
    (tmp_path / ".claude").mkdir(parents=True)
    config_path = tmp_path / "config.yml"
    settings_path = tmp_path / ".claude" / "settings.json"
    config_path.write_text("control_system:\n  writes_enabled: true\n", encoding="utf-8")
    settings_path.write_text(json.dumps({"permissions": None}), encoding="utf-8")
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, out, err = _run(tmp_path)
    assert code == 0
    assert err.strip() == ""
    assert out.strip() == ""


@pytest.mark.parametrize("writes_enabled", [True, False])
def test_consistent_states_are_silent(tmp_path, writes_enabled):
    config_path, settings_path = _make_project(
        tmp_path, writes_enabled=writes_enabled, channel_write_denied=not writes_enabled
    )
    _set_mtimes(config_path, settings_path, config_newer=False)

    code, _out, err = _run(tmp_path)
    assert code == 0
    assert err.strip() == ""
