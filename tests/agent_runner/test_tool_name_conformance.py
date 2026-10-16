"""Every built-in tool name OSPREY names is a tool the pinned CLI build has.

OSPREY runs one Claude Code build: the pinned one
(``_DEFAULT_CLAUDE_CLI_VERSION``), which the containers install and which the
terminal and every SDK path execute. Its built-in tool inventory is recorded
under ``tests/fixtures/cli_tool_inventory/<version>.json`` by
``scripts/cli_tool_inventory.py``, from two probes: offline (``tools``) and with
remote configuration reachable (``remote_config_tools``), which adds flag-gated
tools such as ``Monitor``. Both probes run under the environment OSPREY gives
every Claude Code process it starts, so the inventory lists the task-list tools
that environment turns on. CI re-proves every recorded name against the real
binary. This module checks the four hand-written deny lists in
:mod:`osprey.agent_runner.tool_names` against that inventory. ``mcp__`` entries
are outside the check: the inventory probe loads no MCP server, so it cannot say
which server tools exist.

The dispatch pass-through set is an allow list for the dispatch worker, so its
names are checked against the offline core (``tools``): a pass-through name has
to exist without remote configuration.

The same inventory checks the lists that gate or name tools without denying
them: the build's write-capable list, every standalone hook matcher, and the
tools a persona may name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import osprey
from osprey.agent_runner.tool_names import (
    DENY_DEFAULTS,
    DISPATCH_DENIED_TOOLS,
    OPEN_MODE_EGRESS_TOOLS,
    READ_ONLY_DENIED_BUILTINS,
    WRITE_CAPABLE_BUILTINS,
)
from osprey.agent_runner.tool_policy import PASSTHROUGH_TOOLS
from osprey.cli.templates.artifact_library import parse_hook_frontmatter
from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

INVENTORY_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cli_tool_inventory"
REMEDY = "uv run python scripts/cli_tool_inventory.py --write"
PROBE_KEYS = ("tools", "remote_config_tools")

PINNED_BUILD = _DEFAULT_CLAUDE_CLI_VERSION
DENY_LISTS = {
    "DENY_DEFAULTS": DENY_DEFAULTS,
    "OPEN_MODE_EGRESS_TOOLS": OPEN_MODE_EGRESS_TOOLS,
    "READ_ONLY_DENIED_BUILTINS": READ_ONLY_DENIED_BUILTINS,
    "DISPATCH_DENIED_TOOLS": sorted(DISPATCH_DENIED_TOOLS),
}


def _inventory(version: str) -> frozenset[str]:
    data = json.loads((INVENTORY_DIR / f"{version}.json").read_text())
    return frozenset(name for key in PROBE_KEYS for name in data[key])


def _unknown(entries, tools: frozenset[str]) -> list[str]:
    return sorted(e for e in entries if not e.startswith("mcp__") and e not in tools)


def test_the_pinned_build_has_an_inventory():
    path = INVENTORY_DIR / f"{PINNED_BUILD}.json"
    assert path.is_file(), f"the pinned build {PINNED_BUILD} has no inventory {path}; run {REMEDY}"
    data = json.loads(path.read_text())
    assert data["cli_version"] == PINNED_BUILD
    for key in PROBE_KEYS:
        tools = data.get(key)
        assert tools, f"{path.name} records no {key}; run {REMEDY}"
        assert tools == sorted(tools), f"{path.name} {key} is not sorted"
        assert not [t for t in tools if t.startswith("mcp__")], f"{path.name} {key} has MCP names"


def test_only_the_pinned_build_has_an_inventory():
    stems = {p.stem for p in INVENTORY_DIR.glob("*.json")}
    assert stems == {PINNED_BUILD}, f"stale or missing inventories; run {REMEDY}"


@pytest.mark.parametrize("list_name", list(DENY_LISTS))
def test_denied_names_exist_in_the_build(list_name):
    unknown = _unknown(DENY_LISTS[list_name], _inventory(PINNED_BUILD))
    assert unknown == [], (
        f"{list_name} names tools the pinned build {PINNED_BUILD} does not have: {unknown}. "
        "A CLI that renamed a tool may still honour the old spelling as a deny alias, "
        "so spell the current name; never delete the entry."
    )


def test_legacy_deny_aliases_are_not_inventory_names():
    legacy = ["BashOutput", "KillShell", "KillBash", "MultiEdit", "Agent"]
    assert _unknown(legacy, _inventory(PINNED_BUILD)) == sorted(legacy)


def test_mcp_entries_are_outside_the_check():
    assert _unknown(["mcp__plugin_playwright_playwright__*"], frozenset()) == []


# -- dispatch pass-through set ----------------------------------------------


def _offline_core(version: str) -> frozenset[str]:
    return frozenset(json.loads((INVENTORY_DIR / f"{version}.json").read_text())["tools"])


def test_passthrough_names_exist_in_the_dispatch_build():
    unknown = _unknown(PASSTHROUGH_TOOLS, _offline_core(PINNED_BUILD))
    assert unknown == [], (
        f"PASSTHROUGH_TOOLS names tools the pinned build {PINNED_BUILD} does not have: "
        f"{unknown}. A pass-through name must be a tool the build lists under the "
        "environment OSPREY starts it with, so spell the current name or remove the entry: "
        "an allow-list entry the build lacks grants nothing."
    )


def test_retired_passthrough_names_are_not_dispatch_inventory_names():
    retired = ["TodoWrite", "WaitForMcpServers"]
    assert _unknown(retired, _offline_core(PINNED_BUILD)) == retired


# ---------------------------------------------------------------------------
# Lists that gate or name tools without denying them
# ---------------------------------------------------------------------------

#: Search tools the build lists only when ``Bash`` is denied, whether by
#: ``--disallowedTools`` or by ``permissions.deny``. Every OSPREY floor denies
#: ``Bash``, so an OSPREY agent always has them, while the deny-free inventory
#: never records them.
SHELL_DENIED_SEARCH_TOOLS: frozenset[str] = frozenset({"Glob", "Grep"})

HOOKS_DIR = (
    Path(osprey.__file__).resolve().parent / "templates" / "claude_code" / "claude" / "hooks"
)

_BARE_TOOL = re.compile(r"^[A-Za-z]+$")


def agent_tools(version: str) -> frozenset[str]:
    """The built-in tools that build offers an OSPREY agent."""
    return _inventory(version) | SHELL_DENIED_SEARCH_TOOLS


def tools_every_build_has() -> frozenset[str]:
    """The tools a persona may name: those the pinned build offers an agent."""
    return agent_tools(PINNED_BUILD)


def _standalone_matchers() -> dict[str, str]:
    """Each standalone hook's file name mapped to its ``tools:`` matcher."""
    matchers: dict[str, str] = {}
    for path in sorted(HOOKS_DIR.glob("*.py")):
        meta = parse_hook_frontmatter(path)
        if isinstance(meta, dict):
            matchers[path.name] = meta["tools"]
    return matchers


def test_shell_denied_search_tools_are_not_inventory_names():
    # A build that records them makes the exception unneeded: remove it, never widen it.
    assert SHELL_DENIED_SEARCH_TOOLS.isdisjoint(_inventory(PINNED_BUILD))


def test_write_capable_builtins_exist_in_the_build():
    version = PINNED_BUILD
    unknown = _unknown(WRITE_CAPABLE_BUILTINS, _inventory(version))
    assert unknown == [], (
        f"WRITE_CAPABLE_BUILTINS names tools the pinned build {version} does not have: "
        f"{unknown}. Spell the current name of every tool that can write or shell out."
    )


def test_standalone_hook_matchers_name_tools_the_build_has():
    version = PINNED_BUILD
    tools = agent_tools(version)
    matchers = _standalone_matchers()
    assert "osprey_memory_guard.py" in matchers, f"no standalone hook matchers under {HOOKS_DIR}"

    problems: list[str] = []
    for hook, matcher in matchers.items():
        alternatives = [alt.strip() for alt in matcher.split("|")]
        for alt in alternatives:
            if alt.startswith("mcp__"):
                continue
            if not _BARE_TOOL.match(alt):
                problems.append(f"{hook}: {alt!r} is not a bare tool name")
            elif alt not in tools:
                problems.append(f"{hook}: {alt!r} is not a tool the pinned build {version} has")
    assert problems == [], "\n".join(problems)
