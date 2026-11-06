"""Tests for the core ``safety_chain`` health category.

Every fixture is built under ``tmp_path``: a repo root holding ``build/`` (the
render: ``config.yml``, ``.claude/settings.json``, ``.claude/hooks/`` and the
limits file) and ``var/`` (the agent-data root and the session-hold store). The
hook files are copied from the framework's own templates so the "sha = render"
comparison is exercised against the real sources.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from osprey.health.core import safety_chain as sc
from osprey.health.models import Status

CHAIN = ("osprey_writes_check", "osprey_limits", "osprey_approval")
WRITE_TOOL = "mcp__controls__channel_write"
EXEC_TOOL = "mcp__python__execute"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "OSPREY_AGENT_DATA_ROOT",
        "OSPREY_CONTROL_CONTEXT_DIR",
        "OSPREY_CONTROL_CONTEXT_TREE",
    ):
        monkeypatch.delenv(var, raising=False)


def _command(hook: str) -> dict[str, Any]:
    return {"type": "command", "command": f'python "$CLAUDE_PROJECT_DIR/.claude/hooks/{hook}.py"'}


def _settings(entries: list[tuple[str, list[str]]]) -> dict[str, Any]:
    return {
        "hooks": {
            "PreToolUse": [
                {"matcher": matcher, "hooks": [_command(h) for h in hooks]}
                for matcher, hooks in entries
            ]
        }
    }


DEFAULT_ENTRIES = [
    ("mcp__.*", ["osprey_plan_mode"]),
    (WRITE_TOOL, list(CHAIN)),
    (EXEC_TOOL, ["osprey_writes_check", "osprey_approval"]),
]


def _render(
    tmp_path: Path,
    *,
    entries: list[tuple[str, list[str]]] | None = None,
    write_tools: list[str] | None = None,
    config_extra: dict[str, Any] | None = None,
    limits: dict[str, Any] | None | str = None,
    writes: str = "asked",
) -> tuple[dict[str, Any], Path]:
    """Build a repo under *tmp_path*; return ``(config, render_dir)``."""
    repo = tmp_path / "repo"
    build = repo / "build"
    hooks = build / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    for hook in CHAIN:
        shutil.copy(sc.template_hooks_dir() / f"{hook}.py", hooks / f"{hook}.py")
    (hooks / "hook_config.json").write_text(
        json.dumps({"write_tools": [WRITE_TOOL, EXEC_TOOL] if write_tools is None else write_tools})
    )
    (build / ".claude" / "settings.json").write_text(
        json.dumps(_settings(DEFAULT_ENTRIES if entries is None else entries))
    )
    control_system: dict[str, Any] = {
        "type": "mock",
        "writes": writes,
        "limits_checking": {"enabled": True, "database_path": "data/channel_limits.json"},
    }
    if limits is not None:
        (build / "data").mkdir()
        body = limits if isinstance(limits, str) else json.dumps(limits)
        (build / "data" / "channel_limits.json").write_text(body)
    config: dict[str, Any] = {"project_root": str(repo), "control_system": control_system}
    config.update(config_extra or {})
    return config, build


GOOD_LIMITS = {
    "SR:A:SP": {"min_value": 0.0, "max_value": 1.0, "writable": True},
    "SR:B:SP": {"min_value": -1.0, "max_value": 1.0, "writable": False},
}


def _run(config: dict[str, Any], build: Path) -> dict[str, Any]:
    rows = sc.safety_chain(config, cwd=build)()
    by_name = {r.name: r for r in rows}
    assert len(by_name) == len(rows), "row names are unique"
    for row in rows:
        assert row.category == "safety_chain"
        assert row.value, f"{row.name} sets a value"
    return by_name


# -- hooks -------------------------------------------------------------------


def test_registered_chain_matching_the_render_is_ok(tmp_path: Path) -> None:
    rows = _run(*_render(tmp_path, limits=GOOD_LIMITS))
    for hook in CHAIN:
        row = rows[f"hooks.{hook}"]
        assert row.status is Status.OK, row.message
        assert row.value == "registered · sha = render"
    assert rows["hooks.order"].status is Status.OK
    assert rows["hooks.order"].value == "writes_check → limits → approval"


def test_hook_not_registered_is_an_error(tmp_path: Path) -> None:
    entries = [(WRITE_TOOL, ["osprey_writes_check", "osprey_approval"]), (EXEC_TOOL, list(CHAIN))]
    entries[1] = (EXEC_TOOL, ["osprey_writes_check", "osprey_approval"])
    rows = _run(*_render(tmp_path, entries=entries, limits=GOOD_LIMITS))
    assert rows["hooks.osprey_limits"].status is Status.ERROR
    assert rows["hooks.osprey_limits"].value == "not registered"


def test_hook_missing_on_one_write_tool_is_an_error(tmp_path: Path) -> None:
    entries = [(WRITE_TOOL, list(CHAIN)), (EXEC_TOOL, ["osprey_approval"])]
    rows = _run(*_render(tmp_path, entries=entries, limits=GOOD_LIMITS))
    row = rows["hooks.osprey_writes_check"]
    assert row.status is Status.ERROR
    assert EXEC_TOOL in row.value


def test_out_of_order_chain_is_an_error(tmp_path: Path) -> None:
    entries = [
        (WRITE_TOOL, ["osprey_approval", "osprey_limits", "osprey_writes_check"]),
        (EXEC_TOOL, ["osprey_writes_check", "osprey_approval"]),
    ]
    rows = _run(*_render(tmp_path, entries=entries, limits=GOOD_LIMITS))
    assert rows["hooks.order"].status is Status.ERROR
    assert WRITE_TOOL in rows["hooks.order"].value


def test_matcher_is_a_regex(tmp_path: Path) -> None:
    entries = [("mcp__controls__channel_write|mcp__python__execute", list(CHAIN))]
    rows = _run(*_render(tmp_path, entries=entries, limits=GOOD_LIMITS))
    assert all(rows[f"hooks.{h}"].status is Status.OK for h in CHAIN)


def test_hook_file_differing_from_template_is_a_warning(tmp_path: Path) -> None:
    config, build = _render(tmp_path, limits=GOOD_LIMITS)
    hook = build / ".claude" / "hooks" / "osprey_approval.py"
    hook.write_text(hook.read_text() + "\n# edited\n")
    rows = _run(config, build)
    row = rows["hooks.osprey_approval"]
    assert row.status is Status.WARNING
    assert row.value == "registered · sha differs from render"
    assert row.fix == "osprey build"


def test_hook_file_missing_is_an_error(tmp_path: Path) -> None:
    config, build = _render(tmp_path, limits=GOOD_LIMITS)
    (build / ".claude" / "hooks" / "osprey_limits.py").unlink()
    rows = _run(config, build)
    assert rows["hooks.osprey_limits"].status is Status.ERROR
    assert rows["hooks.osprey_limits"].value == "registered · file missing"


def test_render_without_write_tools_skips_the_hooks(tmp_path: Path) -> None:
    rows = _run(*_render(tmp_path, entries=[], write_tools=[], limits=GOOD_LIMITS))
    for hook in CHAIN:
        assert rows[f"hooks.{hook}"].status is Status.SKIP
        assert rows[f"hooks.{hook}"].value == "no write tools in this render"
    assert "hooks.order" not in rows


def test_unreadable_settings_is_an_error(tmp_path: Path) -> None:
    config, build = _render(tmp_path, limits=GOOD_LIMITS)
    (build / ".claude" / "settings.json").write_text("{not json")
    rows = _run(config, build)
    assert rows["hooks.osprey_writes_check"].status is Status.ERROR
    assert rows["hooks.osprey_writes_check"].value == "settings.json unreadable"


@pytest.mark.parametrize("body", [None, "{not json", '["a list"]'])
def test_a_missing_or_unreadable_hook_config_fails_closed(tmp_path: Path, body) -> None:
    config, build = _render(tmp_path, limits=GOOD_LIMITS)
    hook_config = build / ".claude" / "hooks" / "hook_config.json"
    if body is None:
        hook_config.unlink()
    else:
        hook_config.write_text(body)
    rows = _run(config, build)
    for hook in CHAIN:
        row = rows[f"hooks.{hook}"]
        assert row.status is Status.ERROR, row.message
        assert row.fix == "osprey build"


def test_no_rendered_claude_dir_skips_the_hooks(tmp_path: Path) -> None:
    config, build = _render(tmp_path, limits=GOOD_LIMITS)
    shutil.rmtree(build / ".claude")
    rows = _run(config, build)
    assert rows["hooks.osprey_approval"].status is Status.SKIP
    assert rows["hooks.osprey_approval"].value == "not rendered"


# -- limits ------------------------------------------------------------------


def test_limits_loaded(tmp_path: Path) -> None:
    rows = _run(*_render(tmp_path, limits=GOOD_LIMITS))
    assert rows["limits.parse"].status is Status.OK
    assert rows["limits.parse"].value == "2 channels · 0 errors"
    assert rows["limits.age"].status is Status.OK
    assert "mtime" in rows["limits.age"].value


def test_limits_parse_error_is_an_error(tmp_path: Path) -> None:
    rows = _run(*_render(tmp_path, limits="{broken"))
    assert rows["limits.parse"].status is Status.ERROR
    assert rows["limits.parse"].value == "parse error"
    assert "limits.age" in rows


def test_limits_missing_with_writes_on_is_an_error(tmp_path: Path) -> None:
    rows = _run(*_render(tmp_path, limits=None, writes="asked"))
    assert rows["limits.parse"].status is Status.ERROR
    assert rows["limits.parse"].value == "file missing"
    assert "limits.age" not in rows


def test_limits_missing_with_writes_never_is_a_skip(tmp_path: Path) -> None:
    config, build = _render(tmp_path, limits=None, writes="never")
    config["control_system"]["limits_checking"] = {"enabled": True}
    rows = _run(config, build)
    assert rows["limits.parse"].status is Status.SKIP
    assert rows["limits.parse"].value == "not configured"


def test_limits_checking_off_with_writes_on_is_a_warning(tmp_path: Path) -> None:
    config, build = _render(tmp_path, limits=GOOD_LIMITS)
    config["control_system"]["limits_checking"]["enabled"] = False
    rows = _run(config, build)
    assert rows["limits.parse"].status is Status.WARNING
    assert rows["limits.parse"].value == "limits checking off"


def test_limits_age_formats_hours(tmp_path: Path) -> None:
    config, build = _render(tmp_path, limits=GOOD_LIMITS)
    path = build / "data" / "channel_limits.json"
    past = time.time() - 3 * 3600
    os.utime(path, (past, past))
    rows = _run(config, build)
    assert rows["limits.age"].value.startswith("3.0 h")


# -- write switch / authorizations ---------------------------------------------


def _users_config(tmp_path: Path, **kw: Any) -> tuple[dict[str, Any], Path]:
    config, build = _render(tmp_path, limits=GOOD_LIMITS, **kw)
    repo = build.parent
    for tier, writes in (("readwrite", "asked"), ("readonly", "never")):
        render = repo / "build" / f"proj-{tier}"
        render.mkdir(parents=True)
        (render / "config.yml").write_text(f"control_system:\n  writes: {writes}\n")
    config["modules"] = {
        "web_terminals": {
            "enabled": True,
            "default_tier": "readonly",
            "users": [{"name": "alice", "tier": "readwrite"}, {"name": "bob"}],
            "renders": {
                "readwrite": {"kind": "tier", "project_path": "build/proj-readwrite"},
                "readonly": {"kind": "tier", "project_path": "build/proj-readonly"},
            },
        }
    }
    return config, build


def _record(repo: Path, login: str, switch: dict[str, str], target: str = "va") -> Path:
    directory = repo / "var" / "agent_data" / "control_target" / login
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "owner": None,
        "target": target,
        "generation": 0,
        "posture": {},
        "last_switch": None,
        "switch": switch,
        "switch_epoch": 1,
    }
    (directory / "control_context.json").write_text(json.dumps(payload))
    return directory


def test_write_switch_per_login(tmp_path: Path) -> None:
    config, build = _users_config(tmp_path)
    repo = build.parent
    _record(repo, "alice", {"va": "asked"})
    _record(repo, "bob", {"va": "allowed"})
    rows = _run(config, build)
    assert rows["write_switch.alice"].value == "asked on va"
    assert rows["write_switch.alice"].status is Status.OK
    assert rows["write_switch.bob"].value == "never (tier readonly)"


def test_write_switch_without_record_reads_off(tmp_path: Path) -> None:
    config, build = _users_config(tmp_path)
    rows = _run(config, build)
    assert rows["write_switch.alice"].value == "off"
    assert rows["write_switch.alice"].status is Status.OK


def test_write_switch_above_ceiling_is_capped(tmp_path: Path) -> None:
    config, build = _users_config(tmp_path)
    _record(build.parent, "alice", {"va": "allowed"})
    rows = _run(config, build)
    assert rows["write_switch.alice"].value == "allowed on va · capped at asked"


def test_unreadable_switch_record_is_a_warning(tmp_path: Path) -> None:
    config, build = _users_config(tmp_path)
    directory = build.parent / "var" / "agent_data" / "control_target" / "alice"
    directory.mkdir(parents=True)
    (directory / "control_context.json").write_text("garbage")
    rows = _run(config, build)
    assert rows["write_switch.alice"].status is Status.WARNING
    assert rows["write_switch.alice"].value == "record unreadable"


def test_single_user_deployment_reports_the_acting_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sc, "acting_identity", lambda: "op")
    config, build = _render(tmp_path, limits=GOOD_LIMITS, writes="never")
    rows = _run(config, build)
    assert rows["write_switch.op"].value == "never (writes: never)"


def test_write_authorizations_counts_live_grants(tmp_path: Path) -> None:
    config, build = _users_config(tmp_path)
    directory = _record(build.parent, "alice", {"va": "asked"}) / "authorizations"
    directory.mkdir()
    now = time.time_ns()
    for i, expires in enumerate((now + 10**12, now - 10**9)):
        grant_id = f"{i:032x}"
        grant = {
            "schema": 1,
            "id": grant_id,
            "owner": "alice",
            "holder": "optimizer",
            "target": "va",
            "switch_epoch": 1,
            "channels": {"SR:A:SP": {"min": 0.0, "max": 1.0}},
            "steps": 3,
            "issued_at_ns": now - 10**10,
            "expires_at_ns": expires,
            "lease_s": 30.0,
            "approver": {"identity": "alice", "surface": "chip", "at_ns": now - 10**10},
            "label": "",
            "revoked": None,
        }
        (directory / f"{grant_id}.json").write_text(json.dumps(grant))
    rows = _run(config, build)
    assert rows["write_authorizations"].value == "1"
    assert rows["write_authorizations"].status is Status.OK


def test_no_authorizations(tmp_path: Path) -> None:
    config, build = _users_config(tmp_path)
    rows = _run(config, build)
    assert rows["write_authorizations"].value == "0"


# -- watchdogs ------------------------------------------------------------------


def test_no_watchdogs_configured(tmp_path: Path) -> None:
    rows = _run(*_render(tmp_path, limits=GOOD_LIMITS))
    row = rows["watchdog.flags_open"]
    assert row.status is Status.SKIP
    assert row.value == "no watchdogs configured"


def test_watchdogs_and_open_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from osprey.services.watchdog.config import WatchdogDefinition
    from osprey_connectors.session_hold import ensure_session, open_flag

    config, build = _render(tmp_path, limits=GOOD_LIMITS)
    monkeypatch.setattr(
        sc,
        "_load_watchdogs",
        lambda _cfg, _dir: (WatchdogDefinition(name="radiation", engine="decision", enabled=True),),
    )
    root = sc.session_hold_dir(build.parent)
    ensure_session("s1", root=root)
    open_flag(
        "s1",
        flag_id="f1",
        source="watchdog:radiation",
        reason="dose",
        prompt_ref="p1",
        prompt_excerpt="",
        root=root,
    )
    ensure_session("s2", root=root)
    rows = _run(config, build)
    assert rows["watchdog.radiation"].status is Status.OK
    assert rows["watchdog.radiation"].value == "enabled · decision"
    flags = rows["watchdog.flags_open"]
    assert flags.status is Status.WARNING
    assert flags.value == "1 open · 1 session held"


def test_watchdogs_without_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from osprey.services.watchdog.config import WatchdogDefinition

    config, build = _render(tmp_path, limits=GOOD_LIMITS)
    monkeypatch.setattr(
        sc,
        "_load_watchdogs",
        lambda _cfg, _dir: (WatchdogDefinition(name="beam-loss", engine="agent", enabled=True),),
    )
    rows = _run(config, build)
    assert rows["watchdog.flags_open"].status is Status.OK
    assert rows["watchdog.flags_open"].value == "0 · no hold"


def test_invalid_watchdogs_block_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_cfg: Any, _dir: Any) -> Any:
        raise ValueError("bad block")

    monkeypatch.setattr(sc, "_load_watchdogs", boom)
    rows = _run(*_render(tmp_path, limits=GOOD_LIMITS))
    assert rows["watchdog.flags_open"].status is Status.ERROR
    assert rows["watchdog.flags_open"].value == "watchdogs block invalid"


# -- exec sandbox ----------------------------------------------------------------


def test_exec_sandbox_guards_and_timeout(tmp_path: Path) -> None:
    config, build = _render(tmp_path, limits=GOOD_LIMITS)
    config["python_executor"] = {"execution_timeout_seconds": 300}
    rows = _run(config, build)
    assert rows["exec_sandbox.guards"].status is Status.OK
    assert rows["exec_sandbox.guards"].value == "2 / 2 modes compile"
    assert rows["exec_sandbox.timeout"].value == "300 s"


def test_exec_sandbox_invalid_timeout_is_an_error(tmp_path: Path) -> None:
    config, build = _render(tmp_path, limits=GOOD_LIMITS)
    config["python_executor"] = {"execution_timeout_seconds": -1}
    rows = _run(config, build)
    assert rows["exec_sandbox.timeout"].status is Status.ERROR
    assert rows["exec_sandbox.timeout"].value == "invalid: -1"


def test_exec_sandbox_wrapper_failure_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_mode: str) -> str:
        raise RuntimeError("guard import failed")

    monkeypatch.setattr(sc, "_wrapper_source", boom)
    rows = _run(*_render(tmp_path, limits=GOOD_LIMITS))
    assert rows["exec_sandbox.guards"].status is Status.ERROR
    assert rows["exec_sandbox.guards"].value == "0 / 2 modes compile"


# -- factory contract --------------------------------------------------------------


def test_no_config_yields_no_rows(tmp_path: Path) -> None:
    assert sc.safety_chain(None, cwd=tmp_path)() == []


def test_rows_never_raise_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("surprise")

    monkeypatch.setattr(sc, "_hook_rows", boom)
    rows = _run(*_render(tmp_path, limits=GOOD_LIMITS))
    assert rows["hooks"].status is Status.ERROR
    assert "RuntimeError" in rows["hooks"].value


def test_registered_as_core_category() -> None:
    from osprey.health.core import CORE_CATEGORIES

    assert CORE_CATEGORIES["safety_chain"] is sc.safety_chain
