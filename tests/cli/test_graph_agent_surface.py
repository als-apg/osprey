"""The rendered agent surface of the ``graph`` MCP server, per app template.

The ``graph`` server belongs to the **facility-knowledge-graph subagent**: a
declared store renders the subagent, its frontmatter carries the four read-only
tools, and the main agent's route to the graph is delegation — its CLAUDE.md
roster, not its own tool calls. Three claims, each of which a settings-only
check would miss:

1. **The store enables it, and the tools land on exactly one agent.**
   ``ariel_standalone`` ships a ``services.graphdb`` block and no channel-finder
   agent, so the session surface must be rendered there — settings,
   ``.mcp.json`` and the hook prefixes — and the one frontmatter naming the
   tools must be ``facility-knowledge-graph.md``. A wiring that dropped the
   subagent would render an ARIEL project whose orchestrator has no route to
   its own store.

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

from ._vocabulary_guard import hardcoded_vocabulary_hits

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


def _assert_graph_session_surface(project: Path) -> None:
    """The three files that together make the server callable in a session.

    Split across ``settings.json`` (may it be called), ``.mcp.json`` (can the
    tool be launched) and ``hook_config.json`` (which hook layer sees the call).
    Subagents share all three with the session, so the facility-knowledge-graph
    agent needs every one of them. Any one alone renders happily while the
    project misbehaves at runtime, which is why they are asserted together.
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


def test_ariel_standalone_renders_the_graph_session_surface(built_ariel_standalone_project):
    """The store is configured, so the session can launch and call the server."""
    _assert_graph_session_surface(built_ariel_standalone_project)


def test_ariel_standalone_puts_the_graph_tools_on_the_subagent_only(built_ariel_standalone_project):
    """Exactly one agent frontmatter names the graph tools: facility-knowledge-graph.

    Asserted as an exact file set rather than as a loop over ``.claude/agents/``,
    so the day another file starts naming a graph tool this fails instead of
    passing by omission.
    """
    project = built_ariel_standalone_project

    assert _graph_tool_hits(project / ".claude") == [
        "agents/facility-knowledge-graph.md",
        "hooks/hook_config.json",
        "settings.json",
    ]
    frontmatter = yaml.safe_load(
        (project / ".claude" / "agents" / "facility-knowledge-graph.md")
        .read_text(encoding="utf-8")
        .split("---")[1]
    )
    tools = [entry.strip() for entry in str(frontmatter["tools"]).split(",") if entry.strip()]
    assert sorted(t for t in tools if t.startswith("mcp__graph__")) == _graph_permission_entries()
    assert not (project / ".claude" / "agents" / "channel-finder.md").exists(), (
        "this template's whole point here is that it has no channel-finder agent"
    )


def test_the_rendered_agent_ships_the_snapshot_placeholder(built_ariel_standalone_project):
    """The build ships the marker pair the seed-time bake rewrites.

    ``prompt_snapshot.apply_snapshot`` refuses a file without both markers, so
    a build that dropped them would leave every deployment's agent on the
    placeholder-forever path — tools still work, but the seed-time capture
    silently never lands.
    """
    from osprey.services.facility_knowledge.seeder import prompt_snapshot

    text = (
        built_ariel_standalone_project / ".claude" / "agents" / "facility-knowledge-graph.md"
    ).read_text(encoding="utf-8")

    assert prompt_snapshot.SNAPSHOT_BEGIN in text
    assert prompt_snapshot.SNAPSHOT_END in text
    assert text.find(prompt_snapshot.SNAPSHOT_BEGIN) < text.find(prompt_snapshot.SNAPSHOT_END)
    assert "No snapshot has been captured yet" in text, (
        "the placeholder must say why the section is empty and what fills it"
    )


def test_the_rendered_agent_does_not_claim_where_direction_came_from(
    built_ariel_standalone_project,
):
    """The prompt may not name the artifact a binding's direction came from.

    The sentence that tells the agent to trust the ``READSSIGNAL`` /
    ``WRITESSIGNAL`` edge over its own reading of the address text is
    load-bearing, and it is what an operator gets back when they ask "how do
    you know this is writable?". Direction reaches the corpus by more than one
    route — the generator's PV-grammar fallback when no channel limits file
    resolves, or a deployment-supplied mapping that never opens one — so naming
    a single artifact as the source is true on one path and reassuringly wrong
    on the others. The instruction survives without the provenance claim; the
    per-deployment answer belongs in the snapshot, which is stamped from what
    the store actually holds.
    """
    text = (
        built_ariel_standalone_project / ".claude" / "agents" / "facility-knowledge-graph.md"
    ).read_text(encoding="utf-8")

    assert "limits file" not in text.lower(), (
        "the rendered agent names the limits file as the direction source; "
        "direction may come from the PV-grammar fallback or from a "
        "deployment-supplied mapping instead"
    )


def test_the_rendered_agent_carries_no_hardcoded_vocabulary(built_ariel_standalone_project):
    """The prompt may not name a facility's device kinds out of the framework.

    Ruling of 2026-08-27, on issue #739: facility terminology has one source of
    truth, and for the graph paradigm it is the store's ``(c:Class).altLabel``,
    captured into the *Graph at Hand* block at seed time. A hard-coded row
    naming ``Quadrupole`` or ``dcct`` ships every deployment class labels and
    synonyms its own ontology may not carry — and a label that does not exist
    returns zero rows rather than an error, which is the failure mode this very
    prompt spends a section warning about.
    """
    text = (
        built_ariel_standalone_project / ".claude" / "agents" / "facility-knowledge-graph.md"
    ).read_text(encoding="utf-8")

    assert hardcoded_vocabulary_hits(text) == [], (
        "the rendered facility-knowledge-graph agent hard-codes facility "
        "vocabulary; the class synonyms belong in the Vocabulary section of "
        "the Graph at Hand block, captured from the store at seed time"
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
    # The bundle's source zone `osprey init` lays down beside the profile; the
    # Reach Contract refuses a render whose bind source is not there.
    (repo / "data" / "facility_knowledge").mkdir(parents=True)
    return repo


def _run_build(repo: Path):
    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def test_osprey_build_ships_the_graph_tools_to_the_graph_subagent_only(tmp_path):
    """The build verb completes, the facility-knowledge-graph subagent gets the
    tools, and the channel-finder subagent rendered beside it gets none of them
    — the direction ``create_project`` cannot prove, since the build runs its
    own render and then validates the result.

    The halves belong in one test: a build where the tools reached neither
    surface would satisfy the channel-finder assertion on its own.
    """
    repo = _write_profile(tmp_path / "graph-on")

    result = _run_build(repo)
    assert result.exit_code == 0, result.output

    project = repo / "build"
    _assert_graph_session_surface(project)

    agent = project / ".claude" / "agents" / "channel-finder.md"
    frontmatter = yaml.safe_load(agent.read_text(encoding="utf-8").split("---")[1])
    tools = [entry.strip() for entry in str(frontmatter["tools"]).split(",") if entry.strip()]
    assert [t for t in tools if t.startswith("mcp__graph__")] == []
    assert _graph_tool_hits(project / ".claude") == [
        "agents/facility-knowledge-graph.md",
        "hooks/hook_config.json",
        "settings.json",
    ], "the server's whole rendered trace is the subagent frontmatter plus the session's files"


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
    assert not (project / ".claude" / "agents" / "facility-knowledge-graph.md").exists(), (
        "the subagent rides the server (server_dependency): switching the server "
        "off must take the agent with it, or the render ships frontmatter naming "
        "tools no permission backs"
    )
