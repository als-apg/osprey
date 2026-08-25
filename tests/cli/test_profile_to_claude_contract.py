"""Framework-level drift tests for the profile → .claude/ contract.

These tests lock down the translation from profile inputs into rendered
``.claude/`` artifacts. They guard against silent regressions in:

1. MCP server permission flattening (``settings.json`` ``permissions.allow`` /
   ``permissions.ask``).
2. Framework-agent frontmatter (``tools:`` / ``disallowedTools:``).
3. Overlay agent frontmatter survival through ``regenerate_claude_code``.
4. The crown-jewel invariant: every ``mcp__`` tool an agent declares must have
   a matching entry in the project's ``permissions.allow``.
5. The tier floor: the privileges the ``control-assistant`` base takes away
   from every tier built on it, and the ``remove_deny`` that gives them back to
   the admin tier alone — asserted on the rendered artifacts, since a profile
   pin is only worth what the build writes out.
"""

from __future__ import annotations

import json
import logging
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
    FRAMEWORK_SERVERS,
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
        #        + mcp__osprey_workspace__submit_response, and nothing else.
        # The subagent gets exactly its paradigm's vocabulary: the `graph`
        # server this fixture's render also enables (control_assistant ships a
        # `services.graphdb` block) is a MAIN-agent server and stays off this
        # list. The tool list is read from the registry rather than spelled out,
        # so it cannot drift from what the render uses.
        "tools": [
            *(
                f"mcp__channel-finder__{t}"
                for t in CHANNEL_FINDER_TOOLS_BY_PIPELINE["hierarchical"]
            ),
            "mcp__osprey_workspace__submit_response",
        ],
        "disallowedTools": [
            "Bash",
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "WebFetch",
            "WebSearch",
            "NotebookEdit",
            "Task",
            "Skill",
            "Agent",
        ],
    },
    "logbook-search": {
        "tools": [
            "mcp__ariel__keyword_search",
            "mcp__ariel__semantic_search",
            "mcp__ariel__browse",
            "mcp__ariel__filter_options",
            "mcp__ariel__entry_get",
            "mcp__ariel__capabilities",
            "mcp__osprey_workspace__submit_response",
        ],
        "disallowedTools": [
            "Bash",
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "WebFetch",
            "WebSearch",
            "NotebookEdit",
            "Task",
            "Skill",
            "Agent",
        ],
    },
    "logbook-deep-research": {
        "tools": [
            "mcp__ariel__keyword_search",
            "mcp__ariel__semantic_search",
            "mcp__ariel__browse",
            "mcp__ariel__filter_options",
            "mcp__ariel__entry_get",
            "mcp__ariel__capabilities",
            "mcp__ariel__sql_query",
            "mcp__ariel__entries_by_ids",
            "mcp__osprey_workspace__submit_response",
        ],
        "disallowedTools": [
            "Bash",
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "WebFetch",
            "WebSearch",
            "NotebookEdit",
            "NotebookRead",
            "Task",
            "Skill",
            "Agent",
        ],
    },
    "data-visualizer": {
        "tools": [
            "mcp__osprey_workspace__create_static_plot",
            "mcp__osprey_workspace__create_interactive_plot",
            "mcp__osprey_workspace__create_dashboard",
            "mcp__osprey_workspace__create_document",
            "mcp__osprey_workspace__artifact_get",
            "mcp__osprey_workspace__artifact_list",
            "mcp__osprey_workspace__artifact_read",
            "mcp__osprey_workspace__facility_description",
            "Read",
        ],
        "disallowedTools": [
            "Bash",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "WebFetch",
            "WebSearch",
            "NotebookEdit",
            "Task",
            "Skill",
            "Agent",
        ],
    },
    "pyat-specialist": {
        "tools": [
            "mcp__python__execute",
            "mcp__osprey_workspace__submit_response",
            "mcp__osprey_workspace__artifact_read",
            "Read",
        ],
        "disallowedTools": [
            "Bash",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "WebFetch",
            "WebSearch",
            "NotebookEdit",
            "Task",
            "Skill",
            "Agent",
        ],
    },
    "facility-knowledge": {
        "tools": [
            "mcp__osprey_facility_knowledge__list_concepts",
            "mcp__osprey_facility_knowledge__read_concept",
            "mcp__osprey_facility_knowledge__search",
            "mcp__osprey_workspace__submit_response",
        ],
        "disallowedTools": [
            "Bash",
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "WebFetch",
            "WebSearch",
            "NotebookEdit",
            "Task",
            "Skill",
            "Agent",
        ],
    },
}


@pytest.mark.parametrize("agent_name", list(_FRAMEWORK_AGENT_EXPECTED))
def test_framework_agent_frontmatter_preserved(built_control_assistant_project, agent_name):
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
        md = built_control_assistant_project / ".claude" / "agents" / f"{agent_name}.md"
        if not md.exists():
            pytest.skip(f"framework agent {agent_name} not rendered in this build")
        fm = _parse_frontmatter(md)
        disallowed = _split_csv(fm.get("disallowedTools"))
        assert "Skill" in disallowed, f"{agent_name}: Skill not denied"
        assert "Agent" in disallowed, f"{agent_name}: Agent not denied"


# ---------------------------------------------------------------------------
# Narrowed artifact selections
# ---------------------------------------------------------------------------


def test_narrowed_skill_selection_renders_only_the_selected_skills(tmp_path):
    """A caller-supplied selection that NARROWS ``skills`` renders only those skills.

    Skills are the one artifact family whose rendering is decided solely by the
    resolved output manifest: hooks and rules render from ``.j2`` templates that
    gate on the selection themselves, and agents are filtered from the registry
    before the copy, so for all three the manifest can only ever remove. Skills
    are plain ``.md`` files copied whenever the manifest names them, so a
    manifest built from anything other than the caller's own selection decides
    the skill set on its own — which is how a persona that drops a skill by name
    still ended up shipping it.

    Pins a selection strictly smaller than the ``control_assistant`` bundle's,
    because that is the case a same-or-wider selection cannot distinguish.
    """
    manager = TemplateManager()
    project = manager.create_project(
        project_name="narrowed-skills",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
        artifacts={
            "hooks": ["hook-log", "hook-config", "memory-guard"],
            "rules": ["safety", "timezone"],
            "skills": ["session-report"],
            "output_styles": ["control-operator"],
            "web_panels": ["ariel"],
        },
    )

    skills_dir = project / ".claude" / "skills"
    rendered = sorted(p.name for p in skills_dir.iterdir()) if skills_dir.exists() else []
    assert rendered == ["session-report"]


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
    assert _split_csv(fm["tools"]) == ["mcp__osprey_workspace__submit_response", "Read"]
    assert _split_csv(fm["disallowedTools"]) == ["Bash", "Edit", "Skill", "Agent"]


# ---------------------------------------------------------------------------
# Extends (second framework-server instance) rendered artifacts
# ---------------------------------------------------------------------------


def test_extends_phoebus2_rendered_artifacts(tmp_path, monkeypatch):
    """A config-declared extends clone (claude_code.servers.phoebus2.extends:
    phoebus, applied via the dotted-override path a facility declares in
    config.yml) renders exactly like the deleted framework phoebus2 entry:

    * .mcp.json: python -m osprey.mcp_server.phoebus with the ${...} bridge URL
      preserved literally — even with PHOEBUS2_BRIDGE_URL set in the build env.
    * settings.json: the four reads in allow, drive + open_panel in ask (and
      NOT in allow), PreToolUse approval hook under exactly
      mcp__phoebus2__phoebus_drive (drive-only — no wildcard rule).
    * hook_config.json: both phoebus prefixes in server/approval prefixes.
    """
    manager = TemplateManager()
    project = manager.create_project(
        project_name="extends-contract",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )

    from osprey.utils.config_writer import config_update_fields

    config_update_fields(
        project / "config.yml",
        {
            "claude_code.servers.phoebus.enabled": True,
            "claude_code.servers.phoebus2.extends": "phoebus",
            "claude_code.servers.phoebus2.env.PHOEBUS_BRIDGE_URL": (
                "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"
            ),
        },
    )

    # Set during the regen: claude_code.servers.*.env must stay literal anyway
    # (expanded by Claude Code at MCP launch, not at build time).
    monkeypatch.setenv("PHOEBUS2_BRIDGE_URL", "http://10.0.0.5:7980")
    manager.regenerate_claude_code(project)

    mcp = json.loads((project / ".mcp.json").read_text())
    p2 = mcp["mcpServers"]["phoebus2"]
    assert p2["args"] == ["-m", "osprey.mcp_server.phoebus"]
    assert p2["command"], "clone must render the framework interpreter command"
    assert p2["env"]["PHOEBUS_BRIDGE_URL"] == "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"
    assert p2["env"]["OSPREY_CONFIG"].endswith("/config.yml")

    settings = json.loads((project / ".claude" / "settings.json").read_text())
    allow = set(settings["permissions"]["allow"])
    ask = set(settings["permissions"]["ask"])
    for tool in (
        "phoebus_list_displays",
        "phoebus_perceive",
        "phoebus_perceive_region",
        "phoebus_snapshot",
        "phoebus_open_panel",
    ):
        assert f"mcp__phoebus2__{tool}" in allow, f"mcp__phoebus2__{tool} missing from allow"
    for tool in ("phoebus_drive",):
        assert f"mcp__phoebus2__{tool}" in ask, f"mcp__phoebus2__{tool} missing from ask"
        assert f"mcp__phoebus2__{tool}" not in allow

    pre = settings["hooks"]["PreToolUse"]
    drive_rules = [r for r in pre if r["matcher"] == "mcp__phoebus2__phoebus_drive"]
    assert len(drive_rules) == 1, "expected exactly one drive-only approval rule for the clone"
    assert any("osprey_approval.py" in h["command"] for h in drive_rules[0]["hooks"])
    assert not any(r["matcher"] == "mcp__phoebus2__.*" for r in pre), (
        "clone must be drive-only gated, not wildcard-gated"
    )

    hook_cfg = json.loads((project / ".claude" / "hooks" / "hook_config.json").read_text())
    for prefix in ("mcp__phoebus__", "mcp__phoebus2__"):
        assert prefix in hook_cfg["server_prefixes"]
        assert prefix in hook_cfg["approval_prefixes"]


# ---------------------------------------------------------------------------
# hook_config.json: control_system.write_tools merge + empty-server rendering
# ---------------------------------------------------------------------------


def _read_hook_config(project: Path) -> dict:
    return json.loads((project / ".claude" / "hooks" / "hook_config.json").read_text())


def _server_names_from_prefixes(hook_cfg: dict) -> list[str]:
    """Recover enabled server names from ``mcp__<name>__`` hook_config prefixes."""
    return [p[len("mcp__") : -len("__")] for p in hook_cfg["server_prefixes"]]


def test_hook_config_write_tools_dedupe(tmp_path):
    """``control_system.write_tools`` merges into hook_config without duplicating.

    The writes-check hook rules of every enabled server already contribute their
    matcher to ``write_tools``. A facility that re-states one of those matchers
    in ``control_system.write_tools`` (a natural thing to do when spelling out
    the kill-switch set explicitly) must not get it twice — only genuinely new
    entries are appended, in config order, after the server-derived ones.
    """
    manager = TemplateManager()
    project = manager.create_project(
        project_name="write-tools-dedupe",
        output_dir=tmp_path,
        data_bundle="hello_world",
    )

    baseline = _read_hook_config(project)["write_tools"]
    assert "mcp__controls__channel_write" in baseline, (
        f"expected the controls writes-check rule to seed write_tools; got {baseline}"
    )

    from osprey.utils.config_writer import config_update_fields

    config_update_fields(
        project / "config.yml",
        {
            "control_system.write_tools": [
                "mcp__controls__channel_write",  # already server-derived
                "mcp__facility__custom_write",  # genuinely new
            ]
        },
    )
    manager.regenerate_claude_code(project)

    write_tools = _read_hook_config(project)["write_tools"]

    assert write_tools.count("mcp__controls__channel_write") == 1, (
        f"already-present entry duplicated by the config merge: {write_tools}"
    )
    assert write_tools == [*baseline, "mcp__facility__custom_write"], (
        f"expected only the new entry appended to {baseline}; got {write_tools}"
    )


def test_hook_config_with_no_enabled_servers(tmp_path):
    """Disabling every server still yields valid JSON with empty lists.

    The hook runtime reads this file unconditionally, so the all-disabled corner
    must render as a well-formed document with empty collections rather than a
    truncated or absent one.
    """
    manager = TemplateManager()
    project = manager.create_project(
        project_name="no-servers",
        output_dir=tmp_path,
        data_bundle="hello_world",
    )

    enabled = _server_names_from_prefixes(_read_hook_config(project))
    assert enabled, "fixture precondition: the preset must ship some enabled servers"

    from osprey.utils.config_writer import config_update_fields

    config_update_fields(
        project / "config.yml",
        {f"claude_code.servers.{name}.enabled": False for name in enabled},
    )
    manager.regenerate_claude_code(project)

    # json.loads is the validity assertion — a truncated render raises here.
    hook_cfg = _read_hook_config(project)
    assert hook_cfg == {
        "server_prefixes": [],
        "approval_prefixes": [],
        "write_tools": [],
    }, f"all-disabled build should render empty lists; got {hook_cfg}"


# ---------------------------------------------------------------------------
# The graph MCP server on the MAIN agent's surface
#
# The channel-finder subagent's own frontmatter is pinned by
# ``_FRAMEWORK_AGENT_EXPECTED`` above and by
# ``tests/cli/test_channel_finder_graph_tools.py``. What those cover is the
# subagent; what they do not is the main agent, whose whole tool surface is the
# three files asserted here — ``settings.json`` (may it call the tool),
# ``.mcp.json`` (can the tool be launched at all) and ``hook_config.json``
# (which hook layer sees the call). A server present in one and absent from
# another is a build that renders without complaint and misbehaves at runtime.
#
# The two app templates that ship no ``services.graphdb`` block are the negative
# half: ``hello_world`` below, and ``channel_finder_standalone`` in
# ``tests/cli/test_graph_agent_surface.py``, which owns the renders this
# module's single control_assistant fixture does not provide.
# ---------------------------------------------------------------------------


def _graph_permission_entries() -> list[str]:
    """The four rendered permission strings, derived from the registry.

    ``ServerDefinition.permissions_allow`` holds BARE tool names — the settings
    template splices the ``mcp__<server>__`` prefix onto the whole list — so the
    qualification happens here rather than being spelled out a second time where
    it could drift from what the render actually emits.
    """
    return sorted(f"mcp__graph__{tool}" for tool in FRAMEWORK_SERVERS["graph"].permissions_allow)


def _graph_entries(values) -> list[str]:
    return sorted(entry for entry in values if str(entry).startswith("mcp__graph__"))


def test_graph_server_reaches_the_main_agent_surface(built_control_assistant_project):
    """control_assistant ships ``services.graphdb``, so the main agent gets the
    graph server: exactly the registry's four tools in ``permissions.allow``,
    none of them behind ``ask`` or ``deny``, a launchable ``.mcp.json`` entry
    carrying both config-path variables, and the PostToolUse prefix.

    ``approval_prefixes`` must NOT carry ``mcp__graph__``: every tool on this
    server reads, so an approval prefix would put a human in the loop on each
    one — the read/approve split is what makes the tools usable unprompted.
    """
    project = built_control_assistant_project

    permissions = json.loads((project / ".claude" / "settings.json").read_text())["permissions"]
    assert _graph_entries(permissions["allow"]) == _graph_permission_entries()
    for gate in ("ask", "deny"):
        assert _graph_entries(permissions.get(gate) or []) == [], (
            f"graph tools are read-only; nothing belongs in permissions.{gate}"
        )

    graph_server = json.loads((project / ".mcp.json").read_text())["mcpServers"]["graph"]
    assert graph_server["args"] == ["-m", "osprey.mcp_server.graph"]
    assert graph_server["command"], "the server must render a launchable interpreter command"
    for var in ("OSPREY_CONFIG", "CONFIG_FILE"):
        assert graph_server["env"][var].endswith("/config.yml"), (
            f"{var} must name the rendered config the server resolves the store from"
        )

    hook_cfg = _read_hook_config(project)
    assert "mcp__graph__" in hook_cfg["server_prefixes"]
    assert "mcp__graph__" not in hook_cfg["approval_prefixes"]


def test_hello_world_renders_no_graph_surface_at_all(built_hello_world_project):
    """A template with no ``services.graphdb`` block renders no trace of the
    server — asserted over every file under ``.claude/``, not just
    ``settings.json``, because an agent file, a hook config or a rule that named
    a tool the project cannot launch is exactly as broken and just as invisible
    from a settings-only check.
    """
    project = built_hello_world_project

    hits = sorted(
        str(path.relative_to(project))
        for path in (project / ".claude").rglob("*")
        if path.is_file() and "mcp__graph__" in path.read_text(encoding="utf-8", errors="ignore")
    )
    assert hits == [], f"hello_world configures no graph store but rendered graph tools in {hits}"
    assert "graph" not in json.loads((project / ".mcp.json").read_text())["mcpServers"]


# ---------------------------------------------------------------------------
# Panel-awareness hooks: rendered into the project and wired to their event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hook_file,event",
    [
        ("osprey_panels_context.py", "SessionStart"),
        ("osprey_workspace_delta.py", "UserPromptSubmit"),
    ],
)
def test_panel_hooks_rendered_and_wired(built_control_assistant_project, hook_file, event):
    """Both halves of panel awareness must survive a fresh render.

    The template hook directory is catalog-bound: a hook file that is not listed
    in the build-artifact catalog and the template manifest never reaches a
    project's ``.claude/hooks/``, and one that is shipped but wired to no event
    never runs. Neither failure is visible from the template tree alone, which
    is what this pair of assertions covers.
    """
    project = built_control_assistant_project

    assert (project / ".claude" / "hooks" / hook_file).is_file(), (
        f"{hook_file} was not rendered into the project — check the build-artifact "
        f"catalog and the template manifest"
    )

    settings = json.loads((project / ".claude" / "settings.json").read_text())
    commands = [hook["command"] for rule in settings["hooks"][event] for hook in rule["hooks"]]
    assert any(hook_file in command for command in commands), (
        f"{hook_file} is shipped but wired to no {event} entry in settings.json"
    )


# ---------------------------------------------------------------------------
# Crown-jewel invariant
# ---------------------------------------------------------------------------


def test_crown_jewel_invariant_passes_on_preset(built_control_assistant_project):
    """The shipped control-assistant preset must satisfy the invariant."""
    errors = validate_agent_tools_against_permissions(built_control_assistant_project)
    assert errors == [], "preset violates agent-tools-vs-permissions invariant:\n" + "\n".join(
        errors
    )


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
    assert any("bogus" in e and "mcp__nonexistent__phantom_tool" in e for e in errors), (
        f"expected error naming the bogus agent + tool; got: {errors}"
    )


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
    assert any("wild" in e and "wildcard" in e.lower() for e in errors), (
        f"expected wildcard-rejection error; got: {errors}"
    )


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


def test_build_command_fails_on_violation(tmp_path, monkeypatch, caplog):
    """``osprey build`` must abort when a profile agent declares an unbacked tool.

    Uses a synthetic profile whose ``agents/`` convention directory holds an
    agent .md referencing a tool that is not in any framework MCP server's
    permissions.allow.
    """
    profile_dir = tmp_path / "broken-deployment"
    profile_dir.mkdir()
    agents_dir = profile_dir / "agents"
    agents_dir.mkdir()
    (agents_dir / "bogus.md").write_text(
        textwrap.dedent(
            """\
            ---
            name: bogus
            description: A profile agent with an unbacked tool — should fail build.
            tools: mcp__nonexistent__phantom_tool
            ---

            # Bogus
            """
        ),
        encoding="utf-8",
    )

    profile_yaml = profile_dir / "profile.yml"
    profile_yaml.write_text(
        yaml.dump(
            {
                "name": "Broken Profile",
                "data_bundle": "hello_world",
                "provider": "anthropic",
                "model": "claude-haiku-4-5",
                "config": {"control_system.type": "mock"},
            },
            default_flow_style=False,
        )
    )

    runner = CliRunner()
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            build,
            ["--repo", str(profile_dir), "--skip-deps", "--skip-lifecycle"],
            catch_exceptions=False,
        )

    assert result.exit_code != 0, f"build should have failed; stdout:\n{result.output}"
    # The validator's diagnostic is logged, so it reaches the operator on
    # stderr rather than stdout. click's Result.output mixes both streams and
    # cannot distinguish them, so assert on the record itself.
    assert "bogus" in caplog.text or "phantom_tool" in caplog.text, (
        f"build should name the violation; got records:\n"
        f"{[record.getMessage()[:120] for record in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Deny-source split: remove_deny subtracts profile-authored denies only
# ---------------------------------------------------------------------------


def _rendered_deny(project: Path) -> list[str]:
    return json.loads((project / ".claude" / "settings.json").read_text())["permissions"]["deny"]


def _killswitch_project(tmp_path, name: str, fields: dict) -> Path:
    """A built project regenerated with ``fields`` applied to config.yml.

    The regen path is required here: ``create_project`` never runs the
    writes-off kill-switch block, so only a re-render reflects a
    ``writes_enabled`` setting.
    """
    from osprey.utils.config_writer import config_update_fields

    manager = TemplateManager()
    project = manager.create_project(
        project_name=name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config_update_fields(project / "config.yml", fields)
    manager.regenerate_claude_code(project)
    return project


def test_killswitch_deny_survives_profile_remove_deny(tmp_path):
    """A writes-off kill-switch deny is NOT liftable through ``remove_deny``.

    ``remove_deny`` is authored in the profile, so anything it can reach is
    something a profile can switch off. The writes-off denies must not be in
    that set: a facility that names ``mcp__controls__channel_write`` under
    ``claude_code.permissions.remove_deny`` while running writes-off would
    otherwise hand its agent back the control-system write tool — the exact
    hole the deny-source split closes.
    """
    project = _killswitch_project(
        tmp_path,
        "killswitch-remove-deny",
        {
            "control_system.writes_enabled": False,
            "claude_code.permissions.remove_deny": ["mcp__controls__channel_write"],
        },
    )
    deny = _rendered_deny(project)
    assert "mcp__controls__channel_write" in deny, (
        "remove_deny must not be able to lift a writes-off kill-switch deny; "
        f"rendered deny was {deny}"
    )
    # Rendered once, not twice: the kill switch skips a matcher the earlier
    # (filtered) deny parts already render.
    assert deny.count("mcp__controls__channel_write") == 1, f"duplicate deny entry in {deny}"


def test_remove_deny_still_subtracts_profile_authored_deny(tmp_path):
    """The other half of the split: what a profile ADDED, a profile may remove.

    Both profile-authored deny sources — the ``deny_defaults`` floor and the
    facility's own ``permissions.deny`` — stay subtractable through
    ``remove_deny``. Without this the split would have made ``remove_deny``
    inert rather than merely kill-switch-proof.
    """
    project = _killswitch_project(
        tmp_path,
        "remove-deny-subtracts",
        {
            "control_system.writes_enabled": False,
            "claude_code.permissions.deny": ["mcp__nonframework__facility_authored"],
            "claude_code.permissions.remove_deny": [
                "WebSearch",
                "mcp__nonframework__facility_authored",
            ],
        },
    )
    deny = _rendered_deny(project)
    assert "WebSearch" not in deny, f"remove_deny must subtract from deny_defaults; got {deny}"
    assert "mcp__nonframework__facility_authored" not in deny, (
        f"remove_deny must subtract from the facility's own deny list; got {deny}"
    )
    # The rest of the floor is untouched, and the kill switch still fires.
    assert "Bash" in deny and "Edit" in deny
    assert "mcp__controls__channel_write" in deny


def test_killswitch_deny_absent_when_writes_enabled(tmp_path):
    """With writes ON there is no kill-switch deny to protect, and none renders.

    Guards the split against the opposite failure: a ``killswitch_deny`` key
    that leaked entries into a writes-enabled render would deny the control
    system's write tool on a deployment that is supposed to have it.
    """
    project = _killswitch_project(
        tmp_path,
        "killswitch-writes-on",
        {
            "control_system.writes_enabled": True,
            "claude_code.permissions.remove_deny": ["mcp__controls__channel_write"],
        },
    )
    assert "mcp__controls__channel_write" not in _rendered_deny(project)


def test_killswitch_dedupe_when_profile_also_denies(tmp_path):
    """A profile that itself denies a kill-switch matcher renders it exactly once.

    Pins the ``already_denied`` dedupe path: the facility-authored deny renders
    the matcher (filtered part 2), so the kill switch must skip its own append.
    If either side's filter logic drifted, this would render twice (dedupe
    lost) or zero times (deny silently absent) — both must fail here.
    """
    project = _killswitch_project(
        tmp_path,
        "killswitch-profile-denies",
        {
            "control_system.writes_enabled": False,
            "claude_code.permissions.deny": ["mcp__controls__channel_write"],
        },
    )
    deny = _rendered_deny(project)
    assert deny.count("mcp__controls__channel_write") == 1, (
        f"expected exactly one deny entry for the kill-switch matcher; got {deny}"
    )


# ---------------------------------------------------------------------------
# Tier floor: what a preset's config: layer does to the rendered project
# ---------------------------------------------------------------------------

#: The agent's deployment-editing tool. The ``control-assistant`` base denies it
#: for every tier built on it; ``control-assistant-admin`` is the one profile
#: that subtracts the deny back off with ``permissions.remove_deny``.
SETUP_PATCH_TOOL = "mcp__osprey_workspace__setup_patch"


def _render_preset_project(tmp_path_factory, preset: str) -> Path:
    """A project rendered the way a real build of ``preset`` renders one.

    The module's other fixtures build straight from a ``data_bundle``, which is
    the template layer alone — a preset's ``config:`` block never reaches them.
    That layer is exactly what the tier floor lives in, so pinning it needs the
    second half of the pipeline as well: apply the resolved ``config:``
    overrides to the project's config.yml (through ``config_update_fields``,
    the same writer ``_apply_config_overrides`` calls) and re-render
    ``.claude/`` from the result.
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.utils.config_writer import config_update_fields

    profile, _profile_dir = resolve_build_profile(None, preset=preset)
    manager = TemplateManager()
    project = manager.create_project(
        project_name=f"{preset}-floor",
        output_dir=tmp_path_factory.mktemp("floor_build"),
        data_bundle=profile.data_bundle,
        context={"channel_finder_mode": "hierarchical"},
    )
    config_update_fields(project / "config.yml", profile.config)
    manager.regenerate_claude_code(project)
    return project


@pytest.fixture(scope="module")
def tier_floor_projects(tmp_path_factory) -> dict[str, Path]:
    """The base and admin renders, built once for the whole module.

    Both halves of the floor contract need a real render — the base to show the
    privilege is gone, the admin tier to show it comes back — and neither is
    readable off the profile alone.
    """
    return {
        name: _render_preset_project(tmp_path_factory, name)
        for name in ("control-assistant", "control-assistant-admin")
    }


def _rendered_config(project: Path) -> dict:
    return yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))


def test_tier_floor_denies_setup_patch_in_the_base_render(tier_floor_projects):
    """The base preset's floor reaches settings.json as a real deny entry.

    The tool stays in ``permissions.ask`` — the workspace server declares it
    there, and the osprey_approval hook matches on it — so the ask entry alone
    proves nothing about whether the tier may call it. The deny is what closes
    the path, and deny outranks ask, so both are asserted together: a render
    that lost the deny would still look supervised and would in fact be open.
    """
    settings = json.loads(
        (tier_floor_projects["control-assistant"] / ".claude" / "settings.json").read_text()
    )
    permissions = settings["permissions"]
    assert SETUP_PATCH_TOOL in permissions["deny"]
    assert SETUP_PATCH_TOOL in permissions["ask"]
    assert SETUP_PATCH_TOOL not in permissions.get("allow", [])


def test_admin_render_lifts_the_setup_patch_deny(tier_floor_projects):
    """``remove_deny`` in the admin profile subtracts the floor in the render.

    The other half of the same contract: the deny is gone from the admin
    project's settings.json while the ``ask`` entry survives untouched. That
    surviving ask is the point — lifting the floor gives the tier the path, not
    an unsupervised one, so every ``setup_patch`` call still stops at the
    approval prompt.

    Deliberately overlaps test_preset_render.py's admin render assertions:
    that family pins a real ``osprey init`` + ``osprey build``; this one pins
    the ``regenerate_claude_code`` half of the pipeline this module owns.
    """
    settings = json.loads(
        (tier_floor_projects["control-assistant-admin"] / ".claude" / "settings.json").read_text()
    )
    permissions = settings["permissions"]
    assert SETUP_PATCH_TOOL not in permissions["deny"]
    assert SETUP_PATCH_TOOL in permissions["ask"]


def test_tier_floor_web_keys_render_into_config(tier_floor_projects):
    """The browser-side half of the floor lands in the rendered config.yml.

    The web tier reads these two keys at runtime, so a profile that pins them
    is only as good as the config.yml the build writes. Both renders are
    asserted in one place because the pair is the contract: the base off, the
    admin tier on, from the same base preset.

    Deliberately overlaps test_preset_render.py's tier-floor assertions: that
    family pins a real ``osprey init`` + ``osprey build``; this one pins the
    ``config_update_fields`` half of the pipeline this module owns.
    """
    base_web = _rendered_config(tier_floor_projects["control-assistant"])["web"]
    assert base_web["config_panel"]["enabled"] is False
    assert base_web["scaffold_gallery"]["write_enabled"] is False

    admin_web = _rendered_config(tier_floor_projects["control-assistant-admin"])["web"]
    assert admin_web["config_panel"]["enabled"] is True
    assert admin_web["scaffold_gallery"]["write_enabled"] is True


# Hook helper libraries: selected everywhere, wired nowhere
# ---------------------------------------------------------------------------


def test_hook_helper_libraries_are_copied_but_never_wired(tmp_path):
    """Selection COPIES a hook file; docstring frontmatter WIRES it to an event.

    ``osprey_hook_log.py`` and ``osprey_target_state.py`` are shared libraries
    the real hooks import — a JSONL logger and the control-target state reader.
    They ship in every deployment because the hooks that import them do, which
    is why profiles name them in ``hooks:`` like any other artifact. But neither
    declares an ``event`` in its docstring, so neither may appear in
    ``settings.json``: a registration for a module with no handler is a hook
    Claude Code would run on every matching tool call to no effect.

    Pinning both halves together is the point. Either half alone passes while
    the feature is broken — copied-but-wired fires a no-op hook, and
    wired-but-not-copied is an ImportError inside the hooks that import it.
    """
    manager = TemplateManager()
    project = manager.create_project(
        project_name="hook-helpers",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
        artifacts={
            # memory-guard is along not for this test's sake but for the
            # build's write-gate lint: a profile whose PreToolUse chain never
            # matches `Write` refuses to build at all, and a test that pins
            # copied-vs-wired needs a profile that builds.
            "hooks": ["hook-log", "target-state", "hook-config", "approval", "memory-guard"],
            "rules": ["safety"],
        },
    )

    helpers = ("osprey_hook_log.py", "osprey_target_state.py")

    # Half one: the files are on disk beside the hook that imports them.
    hooks_dir = project / ".claude" / "hooks"
    for helper in helpers:
        assert (hooks_dir / helper).is_file(), f"{helper} was not copied into .claude/hooks/"
    assert (hooks_dir / "osprey_approval.py").is_file(), "the importing hook must ship too"

    # Half two: nothing in settings.json names them.
    settings = (project / ".claude" / "settings.json").read_text(encoding="utf-8")
    for helper in helpers:
        assert helper not in settings, f"{helper} has no event handler and must not be wired"
    assert "osprey_approval.py" in settings, "a real hook must still be wired"
