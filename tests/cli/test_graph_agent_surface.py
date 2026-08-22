"""The rendered agent surface of the ``graph`` MCP server, per app template.

The ``graph`` server is a **main-agent** server: a declared store puts its four
read-only tools on the orchestrator's surface, and on no subagent's. Three
claims, each of which a settings-only check would miss:

1. **The store enables it, and it stops at the main agent.**
   ``ariel_standalone`` ships a ``services.graphdb`` block and no channel-finder
   agent at all, so the graph tools must reach the MAIN agent there — settings,
   ``.mcp.json`` and the hook prefixes — while no agent frontmatter mentions
   them. A wiring that put the server on a subagent's list would render an
   ARIEL project whose orchestrator cannot query its own store.

2. **A template with no store renders no trace of the server**, asserted by
   walking every file under ``.claude/`` rather than by reading one of them.
   ``hello_world`` is asserted the same way in
   ``tests/cli/test_profile_to_claude_contract.py``, which already builds it;
   ``channel_finder_standalone`` is asserted here.

3. **A real ``osprey build`` agrees with ``create_project``.** The contract
   tests and ``tests/cli/test_channel_finder_graph_tools.py`` drive the render
   functions; the build verb runs the render, then feeds the result to
   ``validate_agent_tools_against_permissions`` and raises ``BuildProfileError``
   on a frontmatter tool no permission backs. That arm has to be exercised
   through the CLI, because it is the one that decides whether the benchmark
   fallback lever (``claude_code.servers.graph.enabled: false``) leaves a
   buildable project or an aborting one.

A channel-finder subagent that queries a knowledge graph does so through the
``graph`` *paradigm* instead — its tools arrive under the ``channel-finder``
server name, and ``tests/cli/test_channel_finder_graph_tools.py`` owns them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.templates.manager import TemplateManager
from osprey.registry.mcp import FRAMEWORK_SERVERS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graph_permission_entries() -> list[str]:
    """The four rendered permission strings, derived from the registry.

    ``ServerDefinition.permissions_allow`` holds BARE tool names — the settings
    template splices the ``mcp__<server>__`` prefix onto the whole list — so the
    qualification happens here rather than being restated where it could drift
    from what the render emits.
    """
    return sorted(f"mcp__graph__{tool}" for tool in FRAMEWORK_SERVERS["graph"].permissions_allow)


def _graph_tool_hits(root: Path) -> list[str]:
    """Every file under *root* whose text names a ``mcp__graph__`` tool."""
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and "mcp__graph__" in path.read_text(encoding="utf-8", errors="ignore")
    )


def _assert_main_agent_surface(project: Path) -> None:
    """The three files that together make a server callable by the main agent.

    Split across ``settings.json`` (may it call the tool), ``.mcp.json`` (can the
    tool be launched) and ``hook_config.json`` (which hook layer sees the call).
    Any one of them alone renders happily while the project misbehaves at
    runtime, which is why all three are asserted together.
    """
    permissions = json.loads((project / ".claude" / "settings.json").read_text())["permissions"]
    allow = sorted(e for e in permissions["allow"] if str(e).startswith("mcp__graph__"))
    assert allow == _graph_permission_entries()
    for gate in ("ask", "deny"):
        assert not [e for e in permissions.get(gate) or [] if str(e).startswith("mcp__graph__")], (
            f"graph tools are read-only; nothing belongs in permissions.{gate}"
        )

    graph_server = json.loads((project / ".mcp.json").read_text())["mcpServers"]["graph"]
    assert graph_server["args"] == ["-m", "osprey.mcp_server.graph"]
    for var in ("OSPREY_CONFIG", "CONFIG_FILE"):
        assert graph_server["env"][var].endswith("/config.yml")

    hook_cfg = json.loads((project / ".claude" / "hooks" / "hook_config.json").read_text())
    assert "mcp__graph__" in hook_cfg["server_prefixes"]
    assert "mcp__graph__" not in hook_cfg["approval_prefixes"]


# ---------------------------------------------------------------------------
# ariel_standalone: a graph store with no channel-finder subagent
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_ariel_standalone_project(tmp_path_factory) -> Path:
    return TemplateManager().create_project(
        project_name="ariel-graph-surface",
        output_dir=tmp_path_factory.mktemp("ariel_standalone_build"),
        data_bundle="ariel_standalone",
    )


def test_ariel_standalone_ships_the_store(built_ariel_standalone_project):
    """Fixture precondition: without the block every assertion below is vacuous."""
    config = yaml.safe_load((built_ariel_standalone_project / "config.yml").read_text())
    assert isinstance((config.get("services") or {}).get("graphdb"), dict)


def test_ariel_standalone_gives_the_main_agent_the_graph_tools(built_ariel_standalone_project):
    """The store is configured, so the main agent can query it."""
    _assert_main_agent_surface(built_ariel_standalone_project)


def test_ariel_standalone_puts_the_graph_tools_on_no_agent(built_ariel_standalone_project):
    """No subagent frontmatter names a graph tool.

    Asserted as an exact file set rather than as a loop over ``.claude/agents/``,
    because this template currently renders no framework agents at all — a loop
    over an empty directory would pass forever, including on the day an agent
    starts shipping here.
    """
    project = built_ariel_standalone_project

    assert _graph_tool_hits(project / ".claude") == [
        "hooks/hook_config.json",
        "settings.json",
    ]
    assert not (project / ".claude" / "agents" / "channel-finder.md").exists(), (
        "this template's whole point here is that it has no channel-finder agent"
    )


# ---------------------------------------------------------------------------
# channel_finder_standalone: a subagent with no graph store
# ---------------------------------------------------------------------------


def test_channel_finder_standalone_renders_no_graph_surface_at_all(tmp_path):
    """The mirror image of ariel_standalone: the agent is there, the store is
    not, and nothing under ``.claude/`` names a graph tool.

    ``tests/cli/test_channel_finder_graph_tools.py`` already pins this bundle's
    frontmatter and its settings permissions; what it does not do is walk the
    rest of the tree, where a rule or a hook config naming an unlaunchable tool
    would be just as broken and just as quiet.
    """
    project = TemplateManager().create_project(
        project_name="cf-standalone-graph-surface",
        output_dir=tmp_path,
        data_bundle="channel_finder_standalone",
        context={"channel_finder_mode": "hierarchical", "default_provider": "anthropic"},
    )

    config = yaml.safe_load((project / "config.yml").read_text())
    assert (config.get("services") or {}).get("graphdb") is None, "the bundle ships no store"

    assert (project / ".claude" / "agents" / "channel-finder.md").exists(), (
        "fixture precondition: the agent that WOULD carry the tools must be rendered"
    )
    assert _graph_tool_hits(project / ".claude") == []
    assert "graph" not in json.loads((project / ".mcp.json").read_text())["mcpServers"]


# ---------------------------------------------------------------------------
# A real `osprey build`, both ways round
# ---------------------------------------------------------------------------


def _write_profile(repo: Path, config: dict | None = None) -> Path:
    """A minimal deployment repo that builds the control_assistant template.

    Deliberately not the ``control-assistant`` preset: that one hosts the
    multi-user web tier and so renders three persona projects beside the host,
    which is a different (and separately tested) claim. This profile isolates
    the one thing under test — the build verb's own render-then-validate pass.
    """
    repo.mkdir(parents=True)
    (repo / "profile.yml").write_text(
        yaml.dump(
            {
                "name": "Graph Surface",
                "app_template": "control_assistant",
                "provider": "anthropic",
                "model": "haiku",
                "channel_finder_mode": "hierarchical",
                "config": {"control_system.type": "mock", **(config or {})},
            },
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return repo


def _run_build(repo: Path):
    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def test_osprey_build_ships_the_graph_tools_to_the_main_agent_only(tmp_path):
    """The build verb completes, the main agent gets the server, and the
    channel-finder subagent rendered beside it gets none of it — the direction
    ``create_project`` cannot prove, since the build runs its own render and
    then validates the result.

    The two halves belong in one test: a build where the tools reached neither
    surface would satisfy the frontmatter assertion on its own.
    """
    repo = _write_profile(tmp_path / "graph-on")

    result = _run_build(repo)
    assert result.exit_code == 0, result.output

    project = repo / "build"
    _assert_main_agent_surface(project)

    agent = project / ".claude" / "agents" / "channel-finder.md"
    frontmatter = yaml.safe_load(agent.read_text(encoding="utf-8").split("---")[1])
    tools = [entry.strip() for entry in str(frontmatter["tools"]).split(",") if entry.strip()]
    assert [t for t in tools if t.startswith("mcp__graph__")] == []
    assert _graph_tool_hits(project / ".claude") == [
        "hooks/hook_config.json",
        "settings.json",
    ], "the server's whole rendered trace is the main agent's three files"


def test_osprey_build_completes_with_the_graph_server_switched_off(tmp_path):
    """The fallback lever: the store stays declared, every ``mcp__graph__`` entry
    disappears, and the build still completes.

    This is the arm that has to go through the CLI. Disabling the server drops
    the permissions; if the agent frontmatter were gated on the STORE rather than
    on the server it would still list the tools, and the build's own
    ``validate_agent_tools_against_permissions`` pass would abort with a
    ``BuildProfileError`` — turning a cheap benchmark toggle into a broken build.
    """
    repo = _write_profile(tmp_path / "graph-off", {"claude_code.servers.graph.enabled": False})

    result = _run_build(repo)
    assert result.exit_code == 0, result.output

    project = repo / "build"
    config = yaml.safe_load((project / "config.yml").read_text())
    assert isinstance((config.get("services") or {}).get("graphdb"), dict), (
        "the lever must switch off the SERVER while leaving the store configured"
    )

    assert _graph_tool_hits(project) == []
    assert "graph" not in json.loads((project / ".mcp.json").read_text())["mcpServers"]
