"""What a build renders from the ``watchdogs:`` block: only what is enabled.

Enabled watchdogs reach ``config.yml``, the build's ``watchdogs/`` directory, the
deny list, ``hook_config.json`` and the trigger hook's row and file, whatever the
profile's ``hooks:`` list says. With none enabled, the render carries no trace.

The trigger's template ships with its own task; while it is absent from the
library the fixture lays a stand-in beside a copy of the templates, so this file
pins the render's selection rather than the hook's behaviour.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.templates import artifact_library, claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.utils.workspace import session_hold_dir
from tests._preset_data import copy_bundle_data
from tests.cli._lifecycle_build import invoke_build, rendered_config

TRIGGER = "osprey_watchdog_trigger.py"
HOOKS = ["turn-state", "memory-guard", "writes-check"]

POSTURE = {
    "control_system.type": "virtual_accelerator",
    "control_system.connector.virtual_accelerator.serving": "in_process",
    "archiver.type": "mock_archiver",
    "approval.enabled": False,
    "approval.default_policy": "always",
    "hooks.debug": False,
}


def _decision(*, enabled: bool = True, threshold: float = 0.5) -> dict[str, Any]:
    return {
        "engine": "decision",
        "enabled": enabled,
        "threshold": threshold,
        "options": [
            {"id": "risky", "text": "A risky prompt", "flag": True},
            {"id": "routine", "text": "A routine prompt", "flag": False},
        ],
    }


def _agent(*, enabled: bool = True) -> dict[str, Any]:
    return {"engine": "agent", "enabled": enabled, "tools": ["Read"], "timeout_s": 30}


@pytest.fixture(autouse=True)
def trigger_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A template tree holding the trigger hook, for the manager and the library."""
    real = TemplateManager()._get_template_root()
    root = real
    if not (real / "claude_code/claude/hooks" / TRIGGER).exists():
        root = tmp_path / "template-tree"
        shutil.copytree(real, root, ignore=shutil.ignore_patterns("__pycache__"))
        (root / "claude_code/claude/hooks" / TRIGGER).write_text(
            '"""Stand-in.\n\n---\nname: watchdog-trigger\nevents: [UserPromptSubmit]\n---\n"""\n',
            encoding="utf-8",
        )
    monkeypatch.setattr(TemplateManager, "_get_template_root", lambda self: root)
    monkeypatch.setattr(artifact_library, "_claude_code_root", lambda: root / "claude_code")
    claude_code._framework_hook_filenames.cache_clear()


def _repo(tmp_path: Path, block: dict[str, Any], *, hooks: list[str] | None = None) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    copy_bundle_data(repo / "data", "hello_world")
    for name in block:
        directory = repo / "watchdogs" / name
        (directory / "references").mkdir(parents=True)
        (directory / "instructions.md").write_text(f"Judge as {name}.\n", encoding="utf-8")
        (directory / "references" / "note.md").write_text("A reference.\n", encoding="utf-8")
    profile = {
        "name": "w",
        "project_name": "w",
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "channel_finder_mode": "hierarchical",
        "data": "data",
        "hooks": HOOKS if hooks is None else hooks,
        "config": POSTURE,
        "watchdogs": block,
    }
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return repo


def _build(repo: Path) -> Path:
    result = invoke_build(repo)
    assert result.exit_code == 0, result.output
    return repo / "build"


def _settings(build: Path) -> dict[str, Any]:
    return json.loads((build / ".claude/settings.json").read_text(encoding="utf-8"))


def _hook_config(build: Path) -> dict[str, Any]:
    return json.loads((build / ".claude/hooks/hook_config.json").read_text(encoding="utf-8"))


def _trigger_rows(build: Path) -> list[dict[str, Any]]:
    rules = _settings(build)["hooks"].get("UserPromptSubmit", [])
    return [hook for rule in rules for hook in rule["hooks"] if TRIGGER in hook["command"]]


def _holds_rules(build: Path) -> list[str]:
    return [d for d in _settings(build)["permissions"]["deny"] if "session_holds" in d]


def _watchdog_rules(build: Path) -> list[str]:
    return [d for d in _settings(build)["permissions"]["deny"] if "watchdogs/" in d]


def test_config_map_holds_enabled_watchdogs_only_saying_enabled(tmp_path: Path) -> None:
    live = _decision(threshold=0.7)
    repo = _repo(tmp_path, {"live": live, "idle": _decision(enabled=False)})
    _build(repo)
    assert rendered_config(repo)["watchdogs"] == {"live": live}
    assert rendered_config(repo)["watchdogs"]["live"]["enabled"] is True


def test_four_watchdogs_render_under_their_names(tmp_path: Path) -> None:
    block = {
        "decide-a": _decision(),
        "decide-b": _decision(threshold=0.3),
        "agent-a": _agent(),
        "agent-b": _agent(),
    }
    repo = _repo(tmp_path, block)
    build = _build(repo)
    assert rendered_config(repo)["watchdogs"] == block
    assert sorted(p.name for p in (build / "watchdogs").iterdir()) == sorted(block)
    for name in block:
        assert (build / "watchdogs" / name / "instructions.md").read_text(
            encoding="utf-8"
        ) == f"Judge as {name}.\n"
        assert (build / "watchdogs" / name / "references" / "note.md").is_file()


def test_a_selection_omitting_the_trigger_still_renders_row_and_file(tmp_path: Path) -> None:
    build = _build(_repo(tmp_path, {"a": _decision()}))
    [row] = _trigger_rows(build)
    assert row["timeout"] == 2
    assert (build / ".claude/hooks" / TRIGGER).is_file()
    assert _hook_config(build)["watchdogs_enabled"] is True


def test_denies_cover_the_render_and_the_hold_store(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"a": _agent()})
    build = _build(repo)
    assert "Edit(/watchdogs/**)" in _watchdog_rules(build)
    assert "Edit(watchdogs/**)" in _watchdog_rules(build)
    assert _holds_rules(build) == [f"Edit(/{session_hold_dir(repo).resolve()}/**)"]


def test_a_disabled_watchdog_is_not_copied_beside_an_enabled_one(tmp_path: Path) -> None:
    build = _build(_repo(tmp_path, {"live": _agent(), "idle": _agent(enabled=False)}))
    assert [p.name for p in (build / "watchdogs").iterdir()] == ["live"]
    assert all("idle" not in rule for rule in _watchdog_rules(build))


@pytest.mark.parametrize("hooks", [HOOKS, [*HOOKS, "watchdog-trigger"]])
def test_all_disabled_leaves_no_trace(tmp_path: Path, hooks: list[str]) -> None:
    repo = _repo(tmp_path, {"idle": _decision(enabled=False)}, hooks=hooks)
    build = _build(repo)
    assert "watchdogs" not in rendered_config(repo)
    assert not (build / "watchdogs").exists()
    assert _trigger_rows(build) == []
    assert not (build / ".claude/hooks" / TRIGGER).exists()
    assert "watchdogs_enabled" not in _hook_config(build)
    assert _holds_rules(build) == []
    assert _watchdog_rules(build) == []


def test_no_block_matches_all_disabled(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {})
    build = _build(repo)
    assert "watchdogs" not in rendered_config(repo)
    assert _trigger_rows(build) == []
    assert not (build / ".claude/hooks" / TRIGGER).exists()
    assert _holds_rules(build) == []


def test_session_hold_dir_is_under_the_state_zone(tmp_path: Path) -> None:
    assert session_hold_dir(tmp_path) == tmp_path / "var" / "state" / "session_holds"
    assert session_hold_dir(str(tmp_path)) == tmp_path / "var" / "state" / "session_holds"


def test_selection_filter_adds_and_removes_the_trigger() -> None:
    select = claude_code.watchdog_hook_selection
    assert select(["a"], True) == ["a", "watchdog-trigger"]
    assert select(["watchdog-trigger", "a"], True) == ["a", "watchdog-trigger"]
    assert select(["a", "watchdog-trigger"], False) == ["a"]
    assert select([], False) == []
