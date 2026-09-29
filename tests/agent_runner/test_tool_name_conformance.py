"""Every built-in tool name OSPREY denies is a tool both pinned CLI builds have.

OSPREY runs two Claude Code builds: the npm-pinned build
(``_DEFAULT_CLAUDE_CLI_VERSION``), which the containers install, and the build
bundled inside the Agent SDK (``claude_agent_sdk._cli_version.__cli_version__``),
which every SDK path runs. Each build's built-in tool inventory is recorded under
``tests/fixtures/cli_tool_inventory/<version>.json`` by
``scripts/cli_tool_inventory.py``, from two probes: offline (``tools``) and with
remote configuration reachable (``remote_config_tools``), which adds flag-gated
tools such as ``Monitor``. CI re-proves every recorded name against the real
binaries. This module checks the four hand-written deny lists in
:mod:`osprey.agent_runner.tool_names` against both inventories. ``mcp__`` entries
are outside the check: the inventory probe loads no MCP server, so it cannot say
which server tools exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from claude_agent_sdk._cli_version import __cli_version__

from osprey.agent_runner.tool_names import (
    DENY_DEFAULTS,
    DISPATCH_DENIED_TOOLS,
    OPEN_MODE_EGRESS_TOOLS,
    READ_ONLY_DENIED_BUILTINS,
)
from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

INVENTORY_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cli_tool_inventory"
REMEDY = "uv run python scripts/cli_tool_inventory.py --write"
PROBE_KEYS = ("tools", "remote_config_tools")

PINNED_BUILDS = {"npm-pinned": _DEFAULT_CLAUDE_CLI_VERSION, "sdk-bundled": __cli_version__}
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


@pytest.mark.parametrize("build", sorted(PINNED_BUILDS))
def test_every_pinned_build_has_an_inventory(build):
    version = PINNED_BUILDS[build]
    path = INVENTORY_DIR / f"{version}.json"
    assert path.is_file(), f"{build} build {version} has no inventory {path}; run {REMEDY}"
    data = json.loads(path.read_text())
    assert data["cli_version"] == version
    for key in PROBE_KEYS:
        tools = data.get(key)
        assert tools, f"{path.name} records no {key}; run {REMEDY}"
        assert tools == sorted(tools), f"{path.name} {key} is not sorted"
        assert not [t for t in tools if t.startswith("mcp__")], f"{path.name} {key} has MCP names"


def test_only_pinned_builds_have_an_inventory():
    stems = {p.stem for p in INVENTORY_DIR.glob("*.json")}
    assert stems == set(PINNED_BUILDS.values()), f"stale or missing inventories; run {REMEDY}"


@pytest.mark.parametrize("build", sorted(PINNED_BUILDS))
@pytest.mark.parametrize("list_name", list(DENY_LISTS))
def test_denied_names_exist_in_the_build(list_name, build):
    version = PINNED_BUILDS[build]
    unknown = _unknown(DENY_LISTS[list_name], _inventory(version))
    assert unknown == [], (
        f"{list_name} names tools the {build} build {version} does not have: {unknown}. "
        "A CLI that renamed a tool may still honour the old spelling as a deny alias, "
        "so spell the current name; never delete the entry."
    )


@pytest.mark.parametrize("build", sorted(PINNED_BUILDS))
def test_legacy_deny_aliases_are_not_inventory_names(build):
    legacy = ["BashOutput", "KillShell", "KillBash", "MultiEdit"]
    assert _unknown(legacy, _inventory(PINNED_BUILDS[build])) == sorted(legacy)


def test_mcp_entries_are_outside_the_check():
    assert _unknown(["mcp__plugin_playwright_playwright__*"], frozenset()) == []
