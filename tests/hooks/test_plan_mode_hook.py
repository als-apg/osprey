"""Tests for the osprey_plan_mode hook.

In plan mode every OSPREY tool that acts is refused, from the one list the
render derives into ``hook_config.json``. Every case the hook cannot read with
certainty — another mode, a missing or malformed list, an unreadable config, a
tool name that is not a string — leaves the call to the rest of the chain,
exactly as if the hook were absent.
"""

import json
from pathlib import Path

import pytest

from tests.hooks.conftest import hook_audit_records, import_hook

HOOK = "osprey_plan_mode.py"

#: Marks a payload that carries no ``permission_mode`` field at all.
_MISSING = object()

#: A rendered list in the shape the build writes: exact names plus the
#: whole-server matcher a custom ``writes_check`` server contributes.
PLAN_MODE_LIST = [
    "mcp__controls__channel_write",
    "mcp__bluesky__queue_add",
    "mcp__sitectl__.*",
]


def _hook_config(denied=PLAN_MODE_LIST):
    config = {
        "server_prefixes": ["mcp__controls__", "mcp__bluesky__", "mcp__sitectl__"],
        "approval_prefixes": ["mcp__controls__", "mcp__bluesky__", "mcp__sitectl__"],
    }
    if denied is not None:
        config["plan_mode_denied_tools"] = denied
    return config


def _run(hook_runner, tool_name, mode="plan", hook_config=None, cwd=None):
    extra = {"hook_event_name": "PreToolUse", "tool_use_id": "toolu_plan1"}
    if mode is not _MISSING:
        extra["permission_mode"] = mode
    return hook_runner(
        HOOK,
        tool_name,
        {},
        cwd=cwd,
        hook_input_extra=extra,
        hook_config=_hook_config() if hook_config is None else hook_config,
    )


def _deny_reason(result):
    output = result["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    return output["permissionDecisionReason"]


# ── Denies ───────────────────────────────────────────────────────────


def test_plan_mode_denies_a_listed_tool_with_the_one_reason(tmp_path, hook_runner, monkeypatch):
    from osprey_connectors_tests._harness import pin_identity

    identity = pin_identity(monkeypatch)
    module = import_hook("osprey_plan_mode")

    result = _run(hook_runner, "mcp__controls__channel_write", cwd=tmp_path)

    assert _deny_reason(result) == module.PLAN_MODE_DENY_REASON.format(tool="channel_write")
    records = hook_audit_records(tmp_path, "hook_plan_mode")
    assert len(records) == 1
    assert records[0]["actor"] == identity
    assert records[0]["decision"] == "refused"
    assert records[0]["subject"] == "mcp__controls__channel_write"
    assert records[0]["reason"] == "plan_mode"
    assert records[0]["tool_use_id"] == "toolu_plan1"


def test_plan_mode_denies_a_tool_matched_by_a_whole_server_entry(hook_runner):
    module = import_hook("osprey_plan_mode")

    result = _run(hook_runner, "mcp__sitectl__set_point")

    assert _deny_reason(result) == module.PLAN_MODE_DENY_REASON.format(tool="set_point")


# ── Silent: the rest of the chain decides ────────────────────────────


@pytest.mark.parametrize("tool", ["mcp__bluesky__stop_run", "mcp__controls__channel_read"])
def test_plan_mode_is_silent_for_an_unlisted_tool(hook_runner, tool):
    assert _run(hook_runner, tool) is None


@pytest.mark.parametrize(
    "mode",
    ["default", "acceptEdits", "bypassPermissions", "Plan", "planning", None, _MISSING],
    ids=["default", "acceptEdits", "bypassPermissions", "Plan", "planning", "none", "missing"],
)
def test_any_other_mode_is_silent(hook_runner, mode):
    assert _run(hook_runner, "mcp__controls__channel_write", mode=mode) is None


@pytest.mark.parametrize(
    "denied",
    [None, "mcp__controls__channel_write", {"mcp__controls__channel_write": True}],
    ids=["key-missing", "a-string", "a-mapping"],
)
def test_plan_mode_without_a_readable_list_is_silent(hook_runner, denied):
    assert (
        _run(hook_runner, "mcp__controls__channel_write", hook_config=_hook_config(denied)) is None
    )


def test_plan_mode_with_an_unreadable_hook_config_is_silent(tmp_path, hook_runner, monkeypatch):
    broken = tmp_path / "hook_config.json"
    broken.write_text("{not json")
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(broken))

    result = hook_runner(
        HOOK,
        "mcp__controls__channel_write",
        {},
        hook_input_extra={"hook_event_name": "PreToolUse", "permission_mode": "plan"},
    )

    assert result is None


@pytest.mark.parametrize("tool_name", [42, None, ["mcp__controls__channel_write"]])
def test_plan_mode_with_a_non_string_tool_name_is_silent(hook_runner, tool_name):
    assert _run(hook_runner, tool_name) is None


# ── The predicate, in-process ────────────────────────────────────────


def _payload(mode=_MISSING, tool="mcp__controls__channel_write"):
    payload = {"tool_name": tool, "tool_input": {}}
    if mode is not _MISSING:
        payload["permission_mode"] = mode
    return payload


@pytest.mark.parametrize(
    ("hook_input", "denied", "expected"),
    [
        pytest.param(_payload("plan"), PLAN_MODE_LIST, True, id="plan-listed"),
        pytest.param(
            _payload("plan", "mcp__sitectl__anything"), PLAN_MODE_LIST, True, id="plan-wildcard"
        ),
        pytest.param(
            _payload("plan", "mcp__bluesky__stop_run"), PLAN_MODE_LIST, False, id="plan-unlisted"
        ),
        pytest.param(_payload("default"), PLAN_MODE_LIST, False, id="default"),
        pytest.param(_payload("acceptEdits"), PLAN_MODE_LIST, False, id="acceptEdits"),
        pytest.param(_payload("bypassPermissions"), PLAN_MODE_LIST, False, id="bypass"),
        pytest.param(_payload("planning"), PLAN_MODE_LIST, False, id="unknown-mode"),
        pytest.param(_payload(None), PLAN_MODE_LIST, False, id="none-mode"),
        pytest.param(_payload(), PLAN_MODE_LIST, False, id="missing-mode"),
        pytest.param(_payload("plan"), None, False, id="key-missing"),
        pytest.param(_payload("plan"), "mcp__controls__channel_write", False, id="non-list"),
        pytest.param(_payload("plan", 42), PLAN_MODE_LIST, False, id="non-string-tool"),
        pytest.param("plan", PLAN_MODE_LIST, False, id="non-dict-input"),
        pytest.param(None, PLAN_MODE_LIST, False, id="no-input"),
    ],
)
def test_plan_mode_denies_truth_table(tmp_path, monkeypatch, hook_input, denied, expected):
    config_path = tmp_path / "hook_config.json"
    config_path.write_text(json.dumps(_hook_config(denied)))
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(config_path))
    module = import_hook("osprey_plan_mode")

    assert module.plan_mode_denies(hook_input) is expected


def test_plan_mode_denies_is_false_on_an_unreadable_hook_config(tmp_path, monkeypatch):
    config_path = tmp_path / "hook_config.json"
    config_path.write_text("{not json")
    monkeypatch.setenv("OSPREY_HOOK_CONFIG", str(config_path))
    module = import_hook("osprey_plan_mode")

    assert module.plan_mode_denies(_payload("plan")) is False


def test_the_list_key_is_the_one_the_render_writes():
    """The hook reads the key the template emits, spelled once in each."""
    module = import_hook("osprey_plan_mode")
    template = (Path(module.__file__).parent / "hook_config.json.j2").read_text(encoding="utf-8")

    assert f'"{module.PLAN_MODE_DENIED_KEY}":' in template
