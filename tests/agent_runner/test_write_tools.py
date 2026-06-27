"""Tests for osprey.agent_runner.write_tools.load_write_tools.

Covers:
  - Happy path: hook_config.json present with a write_tools list.
  - Missing file: falls back to _FALLBACK_WRITE_TOOLS.
  - Malformed JSON: falls back without raising.
  - Safety invariant: channel_write is always present even on the loaded path.
  - Drift guard: canonical _FALLBACK_WRITE_TOOLS in osprey_writes_check.py
    must equal the module-level replica in write_tools.py.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import osprey
from osprey.agent_runner.write_tools import (
    _BUILTIN_UNSAFE_TOOLS,
    _FALLBACK_WRITE_TOOLS,
    load_write_tools,
    read_only_disallowed_tools,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hook_config_dir(tmp_path: Path) -> Path:
    """Return the .claude/hooks dir inside a fake project root, created."""
    hooks = tmp_path / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    return hooks


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_write_tools_from_config(tmp_path: Path) -> None:
    """Returns the list verbatim when the config already contains the full floor."""
    hooks = _hook_config_dir(tmp_path)
    # Includes both canonical floor tools, so nothing is injected and the
    # configured list (custom tool included) is returned unchanged.
    expected = [
        "mcp__controls__channel_write",
        "mcp__python__execute",
        "mcp__custom__tool",
    ]
    (hooks / "hook_config.json").write_text(json.dumps({"write_tools": expected}))

    result = load_write_tools(tmp_path)

    assert result == expected


# ---------------------------------------------------------------------------
# Missing file → fallback
# ---------------------------------------------------------------------------


def test_load_write_tools_missing_file_returns_fallback(tmp_path: Path) -> None:
    """Missing hook_config.json → returns fallback; includes channel_write."""
    result = load_write_tools(tmp_path)

    assert result == _FALLBACK_WRITE_TOOLS
    assert "mcp__controls__channel_write" in result


# ---------------------------------------------------------------------------
# Malformed JSON → fallback, no exception
# ---------------------------------------------------------------------------


def test_load_write_tools_malformed_json_returns_fallback(tmp_path: Path) -> None:
    """Malformed JSON in hook_config.json → returns fallback without raising."""
    hooks = _hook_config_dir(tmp_path)
    (hooks / "hook_config.json").write_text("{not valid json")

    result = load_write_tools(tmp_path)

    assert result == _FALLBACK_WRITE_TOOLS
    assert "mcp__controls__channel_write" in result


# ---------------------------------------------------------------------------
# Missing / empty write_tools key → fallback
# ---------------------------------------------------------------------------


def test_load_write_tools_missing_key_returns_fallback(tmp_path: Path) -> None:
    """hook_config.json present but lacks write_tools key → returns fallback."""
    hooks = _hook_config_dir(tmp_path)
    (hooks / "hook_config.json").write_text(json.dumps({"other_key": []}))

    result = load_write_tools(tmp_path)

    assert result == _FALLBACK_WRITE_TOOLS


def test_load_write_tools_empty_list_returns_fallback(tmp_path: Path) -> None:
    """hook_config.json with empty write_tools list → returns fallback."""
    hooks = _hook_config_dir(tmp_path)
    (hooks / "hook_config.json").write_text(json.dumps({"write_tools": []}))

    result = load_write_tools(tmp_path)

    assert result == _FALLBACK_WRITE_TOOLS


def test_load_write_tools_scalar_write_tools_returns_fallback(tmp_path: Path) -> None:
    """A scalar (non-list) write_tools value must fall closed to the fallback.

    Regression guard: ``list("mcp__...")`` would character-split the string,
    silently dropping every gated tool (including mcp__python__execute) and
    leaving only an injected channel_write. The type check prevents that.
    """
    hooks = _hook_config_dir(tmp_path)
    (hooks / "hook_config.json").write_text(
        json.dumps({"write_tools": "mcp__controls__channel_write"})
    )

    result = load_write_tools(tmp_path)

    assert result == _FALLBACK_WRITE_TOOLS
    assert "mcp__python__execute" in result  # the other gated tool survives


# ---------------------------------------------------------------------------
# Safety invariant: channel_write always present even on the loaded-config path
# ---------------------------------------------------------------------------


def test_load_write_tools_injects_channel_write_if_absent(tmp_path: Path) -> None:
    """Config that omits channel_write → it is injected; other tools preserved."""
    hooks = _hook_config_dir(tmp_path)
    configured = ["mcp__python__execute", "mcp__custom__tool"]
    (hooks / "hook_config.json").write_text(json.dumps({"write_tools": configured}))

    result = load_write_tools(tmp_path)

    assert "mcp__controls__channel_write" in result
    # other configured tools must still be present
    for tool in configured:
        assert tool in result


def test_load_write_tools_floors_entire_canonical_set(tmp_path: Path) -> None:
    """A loaded config missing BOTH canonical write tools gets both injected.

    Write-safety floor: under bypassPermissions disallowed_tools is the sole
    write guard, so neither channel_write nor python execute may silently drop
    out of a project's write_tools through config drift.
    """
    hooks = _hook_config_dir(tmp_path)
    (hooks / "hook_config.json").write_text(json.dumps({"write_tools": ["mcp__custom__tool"]}))

    result = load_write_tools(tmp_path)

    assert "mcp__controls__channel_write" in result
    assert "mcp__python__execute" in result
    assert "mcp__custom__tool" in result  # configured tool preserved


# ---------------------------------------------------------------------------
# Drift guard: canonical file must match module replica
# ---------------------------------------------------------------------------


def test_fallback_write_tools_matches_canonical_hook() -> None:
    """_FALLBACK_WRITE_TOOLS must equal the list in osprey_writes_check.py.

    Locates the canonical file relative to the installed osprey package so
    the check works in editable installs and worktrees alike.
    """
    package_root = Path(osprey.__file__).parent
    canonical_path = (
        package_root / "templates" / "claude_code" / "claude" / "hooks" / "osprey_writes_check.py"
    )
    assert canonical_path.exists(), f"canonical hook file not found: {canonical_path}"

    source = canonical_path.read_text()
    tree = ast.parse(source)

    # Walk top-level assignments looking for _FALLBACK_WRITE_TOOLS = [...]
    canonical_list: list[str] | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_FALLBACK_WRITE_TOOLS"
            and isinstance(node.value, ast.List)
        ):
            canonical_list = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
            break

    assert canonical_list is not None, "_FALLBACK_WRITE_TOOLS not found in osprey_writes_check.py"
    assert _FALLBACK_WRITE_TOOLS == canonical_list, (
        f"write_tools.py fallback {_FALLBACK_WRITE_TOOLS!r} "
        f"differs from canonical hook {canonical_list!r} — update one to match"
    )


# ---------------------------------------------------------------------------
# Headless read-only disallowed set: MCP writes + built-in unsafe tools
# ---------------------------------------------------------------------------


def test_read_only_disallowed_tools_unions_mcp_and_builtins(tmp_path: Path) -> None:
    """read_only_disallowed_tools blocks the MCP write set AND the built-in
    write/exec/network tools (the escape hatch under bypassPermissions)."""
    result = read_only_disallowed_tools(tmp_path)  # no hook_config → fallback

    # MCP write tools
    assert "mcp__controls__channel_write" in result
    assert "mcp__python__execute" in result
    # Built-in unsafe tools
    for builtin in _BUILTIN_UNSAFE_TOOLS:
        assert builtin in result, f"{builtin} must be disallowed in the read-only path"


def test_read_only_disallowed_tools_no_duplicates(tmp_path: Path) -> None:
    """A built-in already present in write_tools is not duplicated."""
    hooks = _hook_config_dir(tmp_path)
    (hooks / "hook_config.json").write_text(
        json.dumps({"write_tools": ["mcp__controls__channel_write", "Bash"]})
    )

    result = read_only_disallowed_tools(tmp_path)

    assert result.count("Bash") == 1


def test_read_only_disallowed_tools_blocks_approval_required_actions(tmp_path: Path) -> None:
    """Approval-required side-effect tools that are NOT in the hardware-write
    kill-switch list must still be blocked (the round-5 escape hatch):
    execute_file, entry_create, draft_concept, setup_patch."""
    result = read_only_disallowed_tools(tmp_path)

    for tool in (
        "mcp__python__execute_file",
        "mcp__ariel__entry_create",
        "mcp__osprey_facility_knowledge__draft_concept",
        "mcp__osprey_workspace__setup_patch",
    ):
        assert tool in result, f"{tool} (side-effecting) must be disallowed"


def test_read_only_disallowed_tools_blocks_destructive_workspace_tools(tmp_path: Path) -> None:
    """Destructive deletes live in permissions_ALLOW (auto-approved interactively)
    so the registry walk misses them — they must still be blocked (round-6 gap).
    A read-only query must never permanently delete persistent gallery/data."""
    result = read_only_disallowed_tools(tmp_path)

    for tool in (
        "mcp__osprey_workspace__data_delete",
        "mcp__osprey_workspace__artifact_delete",
        "mcp__osprey_workspace__artifact_delete_all",
    ):
        assert tool in result, f"{tool} (destructive) must be disallowed"


def test_read_only_blocks_custom_server_ask_tools(tmp_path: Path) -> None:
    """A facility-custom MCP server's write tools (declared via permissions.ask
    in config.yml) must be blocked, while its read tools stay callable. These
    never reach hook_config write_tools and aren't in the framework registry,
    so they would otherwise leak under bypassPermissions (round-7 gap)."""
    (tmp_path / "config.yml").write_text(
        "claude_code:\n"
        "  servers:\n"
        "    my-server:\n"
        "      hooks: {pre_tool_use: [approval]}\n"
        "      permissions:\n"
        "        allow: [read_data]\n"
        "        ask: [write_data, push_setpoint]\n"
    )

    result = read_only_disallowed_tools(tmp_path)

    assert "mcp__my-server__write_data" in result
    assert "mcp__my-server__push_setpoint" in result
    assert "mcp__my-server__read_data" not in result  # reads preserved


def test_read_only_blocks_custom_server_destructive_allow_tool(tmp_path: Path) -> None:
    """A destructive-named tool a custom server puts in permissions.allow is
    still blocked (mirrors the framework workspace-delete handling), while its
    non-destructive reads stay callable."""
    (tmp_path / "config.yml").write_text(
        "claude_code:\n"
        "  servers:\n"
        "    my-archive:\n"
        "      permissions:\n"
        "        allow: [list_runs, delete_run, purge_all]\n"
    )

    result = read_only_disallowed_tools(tmp_path)

    assert "mcp__my-archive__delete_run" in result
    assert "mcp__my-archive__purge_all" in result
    assert "mcp__my-archive__list_runs" not in result  # non-destructive read preserved


def test_read_only_blocks_writes_check_custom_server_at_server_level(tmp_path: Path) -> None:
    """A custom server that writes-check-gates ALL its tools is blocked
    server-level (the per-tool ``mcp__name__.*`` matcher is a no-op in the SDK
    disallow engine, so server-level is the only form that actually blocks)."""
    (tmp_path / "config.yml").write_text(
        "claude_code:\n"
        "  servers:\n"
        "    danger-server:\n"
        "      hooks: {pre_tool_use: [writes_check]}\n"
        "      permissions: {allow: [do_thing]}\n"
    )

    result = read_only_disallowed_tools(tmp_path)

    assert "mcp__danger-server" in result


def test_read_only_disabled_custom_server_not_enumerated(tmp_path: Path) -> None:
    """A custom server explicitly disabled contributes no tools."""
    (tmp_path / "config.yml").write_text(
        "claude_code:\n"
        "  servers:\n"
        "    off-server:\n"
        "      enabled: false\n"
        "      permissions: {ask: [write_data]}\n"
    )

    result = read_only_disallowed_tools(tmp_path)

    assert "mcp__off-server__write_data" not in result


def test_no_destructive_permissions_allow_tool_is_reachable(tmp_path: Path) -> None:
    """Drift guard: any tool a framework server auto-allows whose name implies
    destruction (delete/remove/clear) must be in the read-only disallow set.
    Catches a future destructive permissions_allow tool that the permissions_ask
    + writes-check registry walk would silently miss."""
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    result = set(read_only_disallowed_tools(tmp_path))
    # Independent marker list (do not import the production constant, so this
    # test fails loudly if that constant is narrowed).
    destructive_markers = ("delete", "remove", "clear", "wipe", "purge", "destroy")
    for server in FRAMEWORK_SERVERS.values():
        for tool in (*server.permissions_allow, *server.fixed_allow):
            if any(m in tool.lower() for m in destructive_markers):
                name = f"mcp__{server.name}__{tool}"
                assert name in result, (
                    f"{name} is auto-allowed but its name implies destruction and "
                    f"it is NOT blocked in the read-only path"
                )


def test_read_only_disallowed_tools_allows_audited_reads(tmp_path: Path) -> None:
    """Approval-gated *reads* must stay callable — the read-only run still needs
    to read PVs/archiver. Only writes/side-effects are blocked, not all
    approval-gated tools."""
    result = read_only_disallowed_tools(tmp_path)

    assert "mcp__controls__channel_read" not in result
    assert "mcp__controls__archiver_read" not in result


def test_read_only_disallowed_covers_registry_side_effect_tools(tmp_path: Path) -> None:
    """Drift guard: every framework-server permissions_ask tool and every
    writes-check-gated matcher in the registry must appear in the read-only
    disallow set. Adding a new approval-required tool to a server thus
    auto-extends the read-only floor (or fails this test)."""
    from osprey.registry.mcp import _WRITES_CHECK, FRAMEWORK_SERVERS

    result = set(read_only_disallowed_tools(tmp_path))

    for server in FRAMEWORK_SERVERS.values():
        # Both ask lists render into the interactive approval list → side-effecting.
        for tool in (*server.permissions_ask, *server.fixed_ask):
            name = f"mcp__{server.name}__{tool}"
            assert name in result, (
                f"{name} is in {server.name}.permissions_ask/fixed_ask (side-effecting) "
                f"but missing from read_only_disallowed_tools — read-only floor drifted"
            )
        for rule in server.hooks_pre:
            if _WRITES_CHECK in rule.hooks:
                assert rule.matcher in result, (
                    f"{rule.matcher} is writes-check-gated but missing from "
                    f"read_only_disallowed_tools"
                )


def test_python_server_registers_only_execute_tools_we_block(tmp_path: Path) -> None:
    """Guard against a future unlisted python tool (the execute_file class of
    bug): every tool the python executor server registers must be blocked,
    since the whole server is exec-capable and has no read-only tools.

    Introspects the FastMCP singleton directly (importing the tool modules
    registers them) rather than running create_server(), which would do heavy
    config/workspace startup.
    """
    import asyncio

    from osprey.mcp_server.python_executor import server as py_server
    from osprey.mcp_server.python_executor.tools import (  # noqa: F401 — registers tools
        python_execute,
        python_execute_file,
    )

    tools = asyncio.run(py_server.mcp._list_tools())
    registered = {getattr(t, "name", t) for t in tools}
    assert registered, "expected the python server to register at least one tool"

    result = set(read_only_disallowed_tools(tmp_path))
    for short in registered:
        assert f"mcp__python__{short}" in result, (
            f"python server registers {short!r} but mcp__python__{short} is not "
            f"in the read-only disallow set — arbitrary-exec escape hatch"
        )


def test_builtin_unsafe_tools_cover_interactive_deny_defaults() -> None:
    """Drift guard: every built-in (non-mcp) tool OSPREY denies interactively
    (settings.json.j2 deny_defaults) must also be in the headless read-only
    floor — the headless path must never be more permissive than interactive,
    since its permission/deny layer is inert under bypassPermissions.
    """
    package_root = Path(osprey.__file__).parent
    settings_j2 = package_root / "templates" / "claude_code" / "claude" / "settings.json.j2"
    assert settings_j2.exists(), f"settings template not found: {settings_j2}"

    source = settings_j2.read_text()
    match = re.search(r"set\s+deny_defaults\s*=\s*(\[[^\]]*\])", source)
    assert match, "deny_defaults list not found in settings.json.j2"
    deny_defaults = ast.literal_eval(match.group(1))

    # Built-in tools are the non-mcp__ entries (mcp__ entries are plugin/facility
    # servers, absent from a built project's query path).
    builtin_denies = [d for d in deny_defaults if not d.startswith("mcp__")]
    assert builtin_denies, "expected at least one built-in tool in deny_defaults"
    for tool in builtin_denies:
        assert tool in _BUILTIN_UNSAFE_TOOLS, (
            f"{tool!r} is denied interactively (settings.json.j2 deny_defaults) but "
            f"missing from _BUILTIN_UNSAFE_TOOLS — the headless read-only floor would "
            f"be more permissive than the interactive policy"
        )
