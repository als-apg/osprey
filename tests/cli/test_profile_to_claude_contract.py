"""Framework-level drift tests for the profile → .claude/ contract.

These tests lock down the translation from profile inputs into rendered
``.claude/`` artifacts. They guard against silent regressions in:

1. MCP server permission flattening (``settings.json`` ``permissions.allow`` /
   ``permissions.ask``).
2. Framework-agent frontmatter (``tools:`` / ``disallowedTools:``).
3. Overlay agent frontmatter survival through ``regenerate_claude_code``.
4. The crown-jewel invariant: every ``mcp__`` tool an agent declares must have
   a matching entry in the project's ``permissions.allow``.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.templates.manager import TemplateManager
from osprey.cli.validate_claude_artifacts import (
    validate_agent_tools_against_permissions,
)
from osprey.registry.mcp import (
    CHANNEL_FINDER_TOOLS_BY_PIPELINE,
    FRAMEWORK_AGENTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def _parse_frontmatter(md_path: Path) -> dict:
    text = md_path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    assert m, f"no YAML frontmatter in {md_path}"
    data = yaml.safe_load(m.group(1))
    assert isinstance(data, dict), f"frontmatter is not a mapping in {md_path}"
    return data


def _split_csv(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    return [t.strip() for t in str(value).split(",") if t.strip()]


def _build_control_assistant(tmp_path_factory) -> Path:
    """Build a control-assistant project via TemplateManager (no venv, no lifecycle)."""
    out_dir = tmp_path_factory.mktemp("ca_build")
    manager = TemplateManager()
    return manager.create_project(
        project_name="ca-contract",
        output_dir=out_dir,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )


def _build_hello_world(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("hw_build")
    manager = TemplateManager()
    return manager.create_project(
        project_name="hw-contract",
        output_dir=out_dir,
        data_bundle="hello_world",
    )


# ---------------------------------------------------------------------------
# Module-scope fixtures (each preset is built once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_control_assistant_project(tmp_path_factory) -> Path:
    return _build_control_assistant(tmp_path_factory)


@pytest.fixture(scope="module")
def built_hello_world_project(tmp_path_factory) -> Path:
    return _build_hello_world(tmp_path_factory)


# ---------------------------------------------------------------------------
# MCP permission round-trip
# ---------------------------------------------------------------------------


def _flatten_expected_allow(servers_config: dict, framework_allow: dict) -> set[str]:
    """Compute the structural expectation: every server.permissions.allow tool
    should round-trip to ``mcp__<server>__<tool>`` in settings.json.
    """
    expected: set[str] = set()
    for name, tools in framework_allow.items():
        for tool in tools:
            expected.add(f"mcp__{name}__{tool}")
    # Profile-defined custom servers from config.yml claude_code.servers
    for name, spec in servers_config.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("enabled") is False:
            continue
        for tool in (spec.get("permissions", {}) or {}).get("allow", []) or []:
            expected.add(f"mcp__{name}__{tool}")
    return expected


def _framework_allow_snapshot(settings_allow: set[str]) -> dict[str, set[str]]:
    """Reverse-engineer which framework-server tools are present in allow.

    Returns ``{server_name: {tool, …}}`` for entries of form ``mcp__<server>__<tool>``.
    """
    by_server: dict[str, set[str]] = {}
    for entry in settings_allow:
        if not entry.startswith("mcp__"):
            continue
        parts = entry.split("__", 2)
        if len(parts) < 3:
            continue
        _, server, tool = parts
        by_server.setdefault(server, set()).add(tool)
    return by_server


def test_mcp_permissions_round_trip_control_assistant(built_control_assistant_project):
    """For control-assistant: every framework-server tool that ships enabled
    appears as ``mcp__<server>__<tool>`` in settings.json permissions.allow,
    and the channel-finder pipeline tools round-trip from the registry constant.
    """
    project = built_control_assistant_project
    settings = json.loads((project / ".claude" / "settings.json").read_text())
    allow = set(settings["permissions"]["allow"])

    by_server = _framework_allow_snapshot(allow)

    # Workspace tools (a representative subset) round-trip.
    assert "submit_response" in by_server.get("osprey_workspace", set())
    assert "create_static_plot" in by_server.get("osprey_workspace", set())

    # Ariel tools round-trip.
    assert "keyword_search" in by_server.get("ariel", set())
    assert "sql_query" in by_server.get("ariel", set())

    # Channel-finder pipeline tools come from the single source of truth.
    for tool in CHANNEL_FINDER_TOOLS_BY_PIPELINE["hierarchical"]:
        assert f"mcp__channel-finder__{tool}" in allow, (
            f"channel-finder tool {tool!r} missing from permissions.allow"
        )

    # No bare-prefix wildcard — explicit enumeration only.
    assert "mcp__channel-finder" not in allow


def test_mcp_permissions_round_trip_hello_world(built_hello_world_project):
    """Hello-world inherits framework defaults (no explicit mcp_servers block).

    Structural invariant: every mcp__ entry in allow has the form
    ``mcp__<server>__<tool>`` (no bare prefixes; no wildcards).
    """
    project = built_hello_world_project
    settings = json.loads((project / ".claude" / "settings.json").read_text())
    allow = set(settings["permissions"]["allow"])

    mcp_entries = [e for e in allow if e.startswith("mcp__")]
    assert mcp_entries, "hello-world should have at least some framework MCP tools"
    for entry in mcp_entries:
        assert "*" not in entry, f"unexpected wildcard in permissions.allow: {entry!r}"
        # Either form mcp__<server>__<tool>; bare prefix would have count==1
        assert entry.count("__") >= 2, (
            f"bare-prefix entry {entry!r} in permissions.allow — should be explicit"
        )


def test_mcp_permissions_ask_round_trip(built_control_assistant_project):
    """``permissions.ask`` flattens the same way as ``permissions.allow``."""
    project = built_control_assistant_project
    settings = json.loads((project / ".claude" / "settings.json").read_text())
    ask = set(settings["permissions"]["ask"])

    # Framework-known ask entries: controls.channel_write, workspace.setup_patch,
    # ariel.entry_create.
    assert "mcp__controls__channel_write" in ask
    assert "mcp__osprey_workspace__setup_patch" in ask
    assert "mcp__ariel__entry_create" in ask


# ---------------------------------------------------------------------------
# Framework agent frontmatter preservation
# ---------------------------------------------------------------------------


_FRAMEWORK_AGENT_EXPECTED: dict[str, dict[str, list[str]]] = {
    "channel-finder": {
        # tools: rendered from CHANNEL_FINDER_TOOLS_BY_PIPELINE['hierarchical']
        #        + mcp__osprey_workspace__submit_response
        "tools": [
            *(
                f"mcp__channel-finder__{t}"
                for t in CHANNEL_FINDER_TOOLS_BY_PIPELINE["hierarchical"]
            ),
            "mcp__osprey_workspace__submit_response",
        ],
        "disallowedTools": [
            "Bash", "Read", "Write", "Edit", "Glob", "Grep",
            "WebFetch", "WebSearch", "NotebookEdit", "Task", "Skill", "Agent",
        ],
    },
    "logbook-search": {
        "tools": [
            "mcp__ariel__keyword_search", "mcp__ariel__semantic_search",
            "mcp__ariel__browse", "mcp__ariel__filter_options",
            "mcp__ariel__entry_get", "mcp__ariel__capabilities",
            "mcp__osprey_workspace__submit_response",
        ],
        "disallowedTools": [
            "Bash", "Read", "Write", "Edit", "Glob", "Grep",
            "WebFetch", "WebSearch", "NotebookEdit", "Task", "Skill", "Agent",
        ],
    },
    "logbook-deep-research": {
        "tools": [
            "mcp__ariel__keyword_search", "mcp__ariel__semantic_search",
            "mcp__ariel__browse", "mcp__ariel__filter_options",
            "mcp__ariel__entry_get", "mcp__ariel__capabilities",
            "mcp__ariel__sql_query", "mcp__ariel__entries_by_ids",
            "mcp__osprey_workspace__submit_response",
        ],
        "disallowedTools": [
            "Bash", "Read", "Write", "Edit", "Glob", "Grep",
            "WebFetch", "WebSearch", "NotebookEdit", "NotebookRead",
            "Task", "Skill", "Agent",
        ],
    },
    "data-visualizer": {
        "tools": [
            "mcp__osprey_workspace__create_static_plot",
            "mcp__osprey_workspace__create_interactive_plot",
            "mcp__osprey_workspace__create_dashboard",
            "mcp__osprey_workspace__create_document",
            "mcp__osprey_workspace__artifact_save",
            "mcp__osprey_workspace__artifact_get",
            "mcp__osprey_workspace__data_list",
            "mcp__osprey_workspace__data_read",
            "mcp__osprey_workspace__facility_description",
            "Read",
        ],
        "disallowedTools": [
            "Bash", "Write", "Edit", "Glob", "Grep",
            "WebFetch", "WebSearch", "NotebookEdit", "Task", "Skill", "Agent",
        ],
    },
}


@pytest.mark.parametrize("agent_name", list(_FRAMEWORK_AGENT_EXPECTED))
def test_framework_agent_frontmatter_preserved(
    built_control_assistant_project, agent_name
):
    """Each framework agent's rendered frontmatter matches the locked-down spec."""
    md = built_control_assistant_project / ".claude" / "agents" / f"{agent_name}.md"
    assert md.exists(), f"missing rendered agent file: {md}"
    fm = _parse_frontmatter(md)
    expected = _FRAMEWORK_AGENT_EXPECTED[agent_name]

    assert fm.get("name") == agent_name
    assert _split_csv(fm.get("tools")) == expected["tools"]
    assert _split_csv(fm.get("disallowedTools")) == expected["disallowedTools"]


def test_disallowed_tools_includes_skill_and_agent(built_control_assistant_project):
    """Every framework agent must deny Skill and Agent (commit 3757e95c lockdown)."""
    for agent_name in FRAMEWORK_AGENTS:
        md = (
            built_control_assistant_project / ".claude" / "agents" / f"{agent_name}.md"
        )
        if not md.exists():
            pytest.skip(f"framework agent {agent_name} not rendered in this build")
        fm = _parse_frontmatter(md)
        disallowed = _split_csv(fm.get("disallowedTools"))
        assert "Skill" in disallowed, f"{agent_name}: Skill not denied"
        assert "Agent" in disallowed, f"{agent_name}: Agent not denied"


# ---------------------------------------------------------------------------
# Overlay agent frontmatter preservation
# ---------------------------------------------------------------------------


def test_overlay_agent_frontmatter_preserved(tmp_path):
    """A custom .md dropped into .claude/agents/ survives regenerate_claude_code
    with its frontmatter intact, and the auto-discovery path picks it up.
    """
    manager = TemplateManager()
    project = manager.create_project(
        project_name="overlay-test",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )

    # Drop a custom agent file with a known frontmatter shape.
    custom = project / ".claude" / "agents" / "my-overlay.md"
    custom.write_text(
        textwrap.dedent(
            """\
            ---
            name: my-overlay
            description: A custom facility agent for contract testing.
            tools: mcp__osprey_workspace__submit_response, Read
            disallowedTools: Bash, Edit, Skill, Agent
            ---

            # My Overlay Agent
            Custom facility content.
            """
        ),
        encoding="utf-8",
    )

    # Regen the Claude Code artifacts; the overlay must survive intact.
    manager.regenerate_claude_code(project)

    assert custom.exists(), "overlay agent file deleted by regen"
    fm = _parse_frontmatter(custom)
    assert fm["name"] == "my-overlay"
    assert _split_csv(fm["tools"]) == [
        "mcp__osprey_workspace__submit_response", "Read"
    ]
    assert _split_csv(fm["disallowedTools"]) == [
        "Bash", "Edit", "Skill", "Agent"
    ]


# ---------------------------------------------------------------------------
# Crown-jewel invariant
# ---------------------------------------------------------------------------


def test_crown_jewel_invariant_passes_on_preset(built_control_assistant_project):
    """The shipped control-assistant preset must satisfy the invariant."""
    errors = validate_agent_tools_against_permissions(built_control_assistant_project)
    assert errors == [], "preset violates agent-tools-vs-permissions invariant:\n" + "\n".join(errors)


def test_crown_jewel_invariant_catches_missing_tool(tmp_path):
    """Inject a tool entry not in permissions.allow → validator must complain."""
    manager = TemplateManager()
    project = manager.create_project(
        project_name="missing-tool-test",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )

    bogus = project / ".claude" / "agents" / "bogus.md"
    bogus.write_text(
        textwrap.dedent(
            """\
            ---
            name: bogus
            description: Test agent declaring an unbacked tool.
            tools: mcp__nonexistent__phantom_tool, Read
            ---

            # Bogus
            """
        ),
        encoding="utf-8",
    )

    errors = validate_agent_tools_against_permissions(project)
    assert any(
        "bogus" in e and "mcp__nonexistent__phantom_tool" in e for e in errors
    ), f"expected error naming the bogus agent + tool; got: {errors}"


def test_crown_jewel_invariant_rejects_wildcards(tmp_path):
    """Wildcards in agent tools: must be rejected — list MCP tools explicitly."""
    manager = TemplateManager()
    project = manager.create_project(
        project_name="wildcard-test",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )

    wild = project / ".claude" / "agents" / "wild.md"
    wild.write_text(
        textwrap.dedent(
            """\
            ---
            name: wild
            description: Test agent using a wildcard tool entry.
            tools: mcp__osprey_workspace__*, Read
            ---

            # Wild
            """
        ),
        encoding="utf-8",
    )

    errors = validate_agent_tools_against_permissions(project)
    assert any(
        "wild" in e and "wildcard" in e.lower() for e in errors
    ), f"expected wildcard-rejection error; got: {errors}"


def test_crown_jewel_invariant_accepts_prefix_match_on_existing_allow(
    built_control_assistant_project,
):
    """Sanity: an exact-literal tool that IS in permissions.allow passes."""
    # data-visualizer declares mcp__osprey_workspace__create_static_plot,
    # which IS in workspace.permissions_allow → must pass.
    errors = validate_agent_tools_against_permissions(built_control_assistant_project)
    # Filter to errors mentioning data-visualizer specifically — there should be none.
    data_viz_errors = [e for e in errors if "data-visualizer" in e]
    assert data_viz_errors == [], (
        f"data-visualizer has explicit tools all backed by permissions.allow; "
        f"unexpected errors: {data_viz_errors}"
    )


# ---------------------------------------------------------------------------
# End-to-end: build() must fail on violation
# ---------------------------------------------------------------------------


def test_build_command_fails_on_violation(tmp_path, monkeypatch):
    """``osprey build`` must abort when an overlay agent declares an unbacked tool.

    Uses a synthetic profile with an overlay agent .md that references a
    tool which is not in any framework MCP server's permissions.allow.
    """
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    overlay_dir = profile_dir / "agents"
    overlay_dir.mkdir()
    (overlay_dir / "bogus.md").write_text(
        textwrap.dedent(
            """\
            ---
            name: bogus
            description: An overlay agent with an unbacked tool — should fail build.
            tools: mcp__nonexistent__phantom_tool
            ---

            # Bogus
            """
        ),
        encoding="utf-8",
    )

    profile_yaml = profile_dir / "broken.yml"
    profile_yaml.write_text(
        yaml.dump(
            {
                "name": "Broken Profile",
                "data_bundle": "hello_world",
                "provider": "anthropic",
                "model": "claude-haiku-4-5",
                "overlay": {
                    "agents/bogus.md": ".claude/agents/bogus.md",
                },
                "config": {"control_system.type": "mock"},
            },
            default_flow_style=False,
        )
    )

    runner = CliRunner()
    result = runner.invoke(
        build,
        [
            "broken-build",
            str(profile_yaml),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-deps",
            "--skip-lifecycle",
            "--force",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code != 0, (
        f"build should have failed; stdout:\n{result.output}"
    )
    # Validator's diagnostic should mention the bogus agent and tool.
    assert "bogus" in result.output or "phantom_tool" in result.output, (
        f"build output should name the violation; got:\n{result.output}"
    )
