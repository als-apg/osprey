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

The dispatch pass-through set is an allow list for the dispatch worker alone,
which runs the SDK-bundled build, so its names are checked against that build's
inventory only, and against its offline core (``tools``): a pass-through name has
to exist without remote configuration. The npm-pinned build lists no task-list
tool in a headless run.

The same inventories check the lists that gate or name tools without denying
them: the build's write-capable list, every standalone hook matcher, and the
tools a persona may name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from claude_agent_sdk._cli_version import __cli_version__

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

PINNED_BUILDS = {"npm-pinned": _DEFAULT_CLAUDE_CLI_VERSION, "sdk-bundled": __cli_version__}
DISPATCH_BUILD = "sdk-bundled"
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
    legacy = ["BashOutput", "KillShell", "KillBash", "MultiEdit", "Agent"]
    assert _unknown(legacy, _inventory(PINNED_BUILDS[build])) == sorted(legacy)


def test_mcp_entries_are_outside_the_check():
    assert _unknown(["mcp__plugin_playwright_playwright__*"], frozenset()) == []


# -- dispatch pass-through set ----------------------------------------------


def _offline_core(version: str) -> frozenset[str]:
    return frozenset(json.loads((INVENTORY_DIR / f"{version}.json").read_text())["tools"])


def test_passthrough_names_exist_in_the_dispatch_build():
    version = PINNED_BUILDS[DISPATCH_BUILD]
    unknown = _unknown(PASSTHROUGH_TOOLS, _offline_core(version))
    assert unknown == [], (
        f"PASSTHROUGH_TOOLS names tools the {DISPATCH_BUILD} build {version} does not have: "
        f"{unknown}. A pass-through name must be a tool the dispatch build lists, so spell "
        "the current name or remove the entry: an allow-list entry the build lacks grants "
        "nothing."
    )


def test_retired_passthrough_names_are_not_dispatch_inventory_names():
    retired = ["TodoWrite", "WaitForMcpServers"]
    assert _unknown(retired, _offline_core(PINNED_BUILDS[DISPATCH_BUILD])) == retired


# ---------------------------------------------------------------------------
# Lists that gate or name tools without denying them
# ---------------------------------------------------------------------------

#: Search tools both builds list only when ``Bash`` is denied, whether by
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
    """A persona runs under either build, so it may name only these."""
    return frozenset.intersection(*(agent_tools(v) for v in PINNED_BUILDS.values()))


def _standalone_matchers() -> dict[str, str]:
    """Each standalone hook's file name mapped to its ``tools:`` matcher."""
    matchers: dict[str, str] = {}
    for path in sorted(HOOKS_DIR.glob("*.py")):
        meta = parse_hook_frontmatter(path)
        if isinstance(meta, dict):
            matchers[path.name] = meta["tools"]
    return matchers


@pytest.mark.parametrize("build", sorted(PINNED_BUILDS))
def test_shell_denied_search_tools_are_not_inventory_names(build):
    # A build that records them makes the exception unneeded: remove it, never widen it.
    assert SHELL_DENIED_SEARCH_TOOLS.isdisjoint(_inventory(PINNED_BUILDS[build]))


@pytest.mark.parametrize("build", sorted(PINNED_BUILDS))
def test_write_capable_builtins_exist_in_the_build(build):
    version = PINNED_BUILDS[build]
    unknown = _unknown(WRITE_CAPABLE_BUILTINS, _inventory(version))
    assert unknown == [], (
        f"WRITE_CAPABLE_BUILTINS names tools the {build} build {version} does not have: "
        f"{unknown}. Spell the current name of every tool that can write or shell out."
    )


@pytest.mark.parametrize("build", sorted(PINNED_BUILDS))
def test_standalone_hook_matchers_name_tools_the_build_has(build):
    version = PINNED_BUILDS[build]
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
                problems.append(f"{hook}: {alt!r} is not a tool the {build} build {version} has")
    assert problems == [], "\n".join(problems)
