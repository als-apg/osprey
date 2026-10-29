"""The hooks' stdlib write helpers against the connector functions they restate.

The PreToolUse hooks cannot import ``osprey_connectors`` (they run outside the
osprey venv), so ``osprey_hook_log`` restates ``write_target``'s three tool-name
helpers and ``osprey_target_state`` restates ``posture_store``'s launch-pin term.
The connector originals are private: nothing outside their own module calls
them, and the hook copies are the only reason they are spelled twice. Two copies
of one rule drift silently, so each pair is driven here over one table of inputs
and must give the same answer for every row.

The signatures differ and stay different: the hook's ``is_write_call`` takes
the full tool name and resolves a short name itself when the caller passes none;
the connector's takes the short name its caller already resolved. Only the
answers are compared.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_target_state as target_state
from osprey_connectors import posture_store, write_target

from .conftest import import_hook

PREFIXES = ["mcp__controls__", "mcp__controls_va__", "mcp__python__"]

WRITE_TOOLS = [
    "mcp__controls__channel_write",
    "mcp__python__execute",
    "mcp__bluesky__.*",
    17,
    None,
]

TOOL_NAMES = [
    "mcp__controls__channel_write",
    "mcp__controls__channel_read",
    "mcp__controls_va__channel_write",
    "mcp__python__execute",
    "mcp__pyva__execute",
    "mcp__bluesky__queue_start",
    "mcp__bluesky__",
    "mcp__bluesky",
    "mcp__",
    "Bash",
    "",
]

TOOL_INPUTS = [
    {},
    {"execution_mode": "readonly"},
    {"execution_mode": "write"},
    {"execution_mode": None},
    {"code": "print(1)"},
    MappingProxyType({"execution_mode": "write"}),
    MappingProxyType({}),
    None,
    "execution_mode=write",
    ["execution_mode", "write"],
]

LAUNCH_STAMPS = [
    None,
    "",
    "   ",
    "live=sandbox",
    " va = sandbox ",
    "standin=sandbox",
    "*=sandbox",
    "live=writes",
    "*=writes",
    "live",
    "live=",
    "=sandbox",
    "nowhere=sandbox",
    "live=readonly",
    "live=sandbox=extra",
]

TARGETS = ["live", "va", "standin", None, ""]


@pytest.fixture(scope="module")
def hook_log():
    return import_hook("osprey_hook_log")


@pytest.mark.parametrize("prefixes", [PREFIXES, [], None, ["mcp__controls__", 3]])
@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_short_tool_name_agrees(hook_log, tool_name, prefixes):
    assert hook_log.short_tool_name(tool_name, prefixes) == write_target._short_tool_name(
        tool_name, prefixes
    )


@pytest.mark.parametrize("write_tools", [WRITE_TOOLS, [], None, ["mcp__python__.*"]])
@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_is_write_tool_agrees(hook_log, tool_name, write_tools):
    assert hook_log.is_write_tool(tool_name, write_tools) == write_target._is_write_tool(
        tool_name, write_tools
    )


@pytest.mark.parametrize("tool_input", TOOL_INPUTS, ids=repr)
@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_is_write_call_agrees(hook_log, tool_name, tool_input):
    short_name = write_target._short_tool_name(tool_name, PREFIXES)
    expected = write_target._is_write_call(tool_input, short_name)
    assert hook_log.is_write_call(tool_name, tool_input, short_name) is expected


@pytest.mark.parametrize("tool_input", TOOL_INPUTS, ids=repr)
@pytest.mark.parametrize("tool_name", TOOL_NAMES)
def test_is_write_call_without_a_short_name_agrees_on_the_mcp_shape(
    hook_log, tool_name, tool_input
):
    expected = write_target._is_write_call(tool_input, write_target._short_tool_name(tool_name, ()))
    assert hook_log.is_write_call(tool_name, tool_input) is expected


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("stamp", LAUNCH_STAMPS)
def test_launch_pin_term_agrees(monkeypatch, stamp, target):
    assert target_state.LAUNCH_POSTURE_ENV_VAR == posture_store.LAUNCH_POSTURE_ENV_VAR
    if stamp is None:
        monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, stamp)
    assert target_state.launch_pin_term(target) == posture_store._launch_pin_term(target)
