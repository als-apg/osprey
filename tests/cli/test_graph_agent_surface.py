"""The rendered agent surface of the ``graph`` MCP server, per app template.

The ``graph`` server belongs to the **facility-knowledge-graph subagent**: a
declared store renders the subagent, its frontmatter carries the four read-only
tools, and the main agent's route to the graph is delegation — its CLAUDE.md
roster, not its own tool calls. Three claims, each of which a settings-only
check would miss:

1. **The store enables it, and the tools land on exactly one agent.**
   ``control_assistant`` ships a ``services.graphdb`` block, so the session
   surface must be rendered there — settings, ``.mcp.json`` and the hook
   prefixes — and, under a non-graph channel-finder paradigm, the one
   frontmatter naming the tools must be ``facility-knowledge-graph.md``. A
   wiring that dropped the subagent would render a project whose orchestrator
   has no route to its own store.

2. **A template with no store renders no trace of the server**, asserted by
   walking every file under ``.claude/`` rather than by reading one of them.
   ``hello_world`` is asserted the same way in
   ``tests/cli/test_profile_to_claude_contract.py``, which already builds it;
   a ``control_assistant`` render with its store removed is asserted here.

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
import shutil
from pathlib import Path

import pytest
import yaml

from osprey.cli.templates.manager import TemplateManager
from tests._preset_data import create_project_as_built
from tests._vocabulary import hardcoded_vocabulary_hits
from tests.cli._lifecycle_build import invoke_build
from tests.cli._template_projects import graph_permission_entries

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    assert allow == graph_permission_entries()
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
# a graph store under a non-graph channel-finder paradigm
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_store_project(tmp_path_factory) -> Path:
    return create_project_as_built(
        TemplateManager(),
        project_name="store-graph-surface",
        output_dir=tmp_path_factory.mktemp("store_build"),
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical", "default_provider": "anthropic"},
    )


def test_the_project_ships_the_store(built_store_project):
    """Fixture precondition: without the block every assertion below is vacuous."""
    config = yaml.safe_load((built_store_project / "config.yml").read_text())
    assert isinstance((config.get("services") or {}).get("graphdb"), dict)


def test_the_store_renders_the_graph_session_surface(built_store_project):
    """The store is configured, so the session can launch and call the server."""
    _assert_graph_session_surface(built_store_project)


def test_the_graph_tools_land_on_the_subagent_only(built_store_project):
    """Exactly one agent frontmatter names the graph tools: facility-knowledge-graph.

    Asserted as an exact file set rather than as a loop over ``.claude/agents/``,
    so the day another file starts naming a graph tool this fails instead of
    passing by omission.
    """
    project = built_store_project

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
    assert sorted(t for t in tools if t.startswith("mcp__graph__")) == graph_permission_entries()
    assert (project / ".claude" / "agents" / "channel-finder.md").exists(), (
        "fixture precondition: the channel-finder agent is rendered and carries no graph tool"
    )


def test_the_rendered_agent_ships_the_snapshot_placeholder(built_store_project):
    """The build ships the marker pair the seed-time bake rewrites.

    ``prompt_snapshot.apply_snapshot`` refuses a file without both markers, so
    a build that dropped them would leave every deployment's agent on the
    placeholder-forever path — tools still work, but the seed-time capture
    silently never lands.
    """
    from osprey.services.facility_knowledge.seeder import prompt_snapshot

    text = (built_store_project / ".claude" / "agents" / "facility-knowledge-graph.md").read_text(
        encoding="utf-8"
    )

    assert prompt_snapshot.SNAPSHOT_BEGIN in text
    assert prompt_snapshot.SNAPSHOT_END in text
    assert text.find(prompt_snapshot.SNAPSHOT_BEGIN) < text.find(prompt_snapshot.SNAPSHOT_END)
    assert "No snapshot has been captured yet" in text, (
        "the placeholder must say why the section is empty and what fills it"
    )


def test_the_rendered_agent_does_not_claim_where_direction_came_from(
    built_store_project,
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
    text = (built_store_project / ".claude" / "agents" / "facility-knowledge-graph.md").read_text(
        encoding="utf-8"
    )

    assert "limits file" not in text.lower(), (
        "the rendered agent names the limits file as the direction source; "
        "direction may come from the PV-grammar fallback or from a "
        "deployment-supplied mapping instead"
    )


def test_the_rendered_agent_carries_no_hardcoded_vocabulary(built_store_project):
    """The prompt may not name a facility's device kinds out of the framework.

    Ruling of 2026-08-27, on issue #739: facility terminology has one source of
    truth, and for the graph paradigm it is the store's ``(c:Class).altLabel``,
    captured into the *Graph at Hand* block at seed time. A hard-coded row
    naming ``Quadrupole`` or ``dcct`` ships every deployment class labels and
    synonyms its own ontology may not carry — and a label that does not exist
    returns zero rows rather than an error, which is the failure mode this very
    prompt spends a section warning about.
    """
    text = (built_store_project / ".claude" / "agents" / "facility-knowledge-graph.md").read_text(
        encoding="utf-8"
    )

    assert hardcoded_vocabulary_hits(text) == [], (
        "the rendered facility-knowledge-graph agent hard-codes facility "
        "vocabulary; the class synonyms belong in the Vocabulary section of "
        "the Graph at Hand block, captured from the store at seed time"
    )


# ---------------------------------------------------------------------------
# a channel-finder subagent with no graph store
# ---------------------------------------------------------------------------


def test_a_project_without_a_store_renders_no_graph_surface_at_all(tmp_path):
    """The mirror image of the store project: the agent is there, the store is
    not, and nothing under ``.claude/`` names a graph tool.

    ``tests/cli/test_channel_finder_graph_tools.py`` already pins this render's
    frontmatter and its settings permissions; what it does not do is walk the
    rest of the tree, where a rule or a hook config naming an unlaunchable tool
    would be just as broken and just as quiet.
    """
    from osprey.utils.config_writer import config_delete_field

    manager = TemplateManager()
    project = create_project_as_built(
        manager,
        project_name="nostore-graph-surface",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={
            "channel_finder_mode": "hierarchical",
            "default_provider": "anthropic",
            "deploy_services": False,
        },
    )
    config_delete_field(project / "config.yml", "services.graphdb")
    manager.regenerate_claude_code(project)

    config = yaml.safe_load((project / "config.yml").read_text())
    assert (config.get("services") or {}).get("graphdb") is None, "the render shape under test"

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

    Extends the ``control-assistant`` preset rather than materializing it, so
    the profile spells only what is under test — the build verb's own
    render-then-validate pass. The preset's roster and ``dispatch.tier`` name
    the bundled tiers and apps, which a repo carries as files, so they are
    copied in verbatim the way ``osprey init`` lays them down.
    """
    repo.mkdir(parents=True)
    (repo / "profile.yml").write_text(
        yaml.dump(
            {
                "name": "Graph Surface",
                "project_name": repo.name,
                # The preset carries the posture floor and names the bundle.
                "extends": "control-assistant",
                "data": "data",
                "provider": "anthropic",
                "model": "claude-haiku-4-5",
                "channel_finder_mode": "hierarchical",
                # The preset deploys a virtual accelerator, which serves the
                # project's own channels and refuses a build whose data tree
                # stages none. This repo ships the source zone only, and the
                # accelerator is not what these tests are about.
                "virtual_accelerator": None,
                "config": {
                    "control_system.type": "virtual_accelerator",
                    "control_system.connector.virtual_accelerator.serving": "in_process",
                    **(config or {}),
                },
            },
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    # The facility data tree `osprey init` lays down beside the profile. Copied
    # whole rather than stubbed: the Reach Contract refuses a render whose
    # source zone is not there, and the limits validator reads
    # `channel_limits.json` out of the same tree.
    from tests._preset_data import copy_bundle_data

    copy_bundle_data(repo / "data")
    (repo / "data" / "facility" / "knowledge").mkdir(parents=True, exist_ok=True)
    from osprey.cli.build_profile_presets import _presets_dir

    shutil.copytree(_presets_dir() / "tiers", repo / "tiers")
    return repo


def test_osprey_build_ships_the_graph_tools_to_the_graph_subagent_only(tmp_path):
    """The build verb completes, the facility-knowledge-graph subagent gets the
    tools, and the channel-finder subagent rendered beside it gets none of them
    — the direction ``create_project`` cannot prove, since the build runs its
    own render and then validates the result.

    The halves belong in one test: a build where the tools reached neither
    surface would satisfy the channel-finder assertion on its own.
    """
    repo = _write_profile(tmp_path / "graph-on")

    result = invoke_build(repo)
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

    result = invoke_build(repo)
    assert result.exit_code == 0, result.output

    project = repo / "build"
    config = yaml.safe_load((project / "config.yml").read_text())
    assert isinstance((config.get("services") or {}).get("graphdb"), dict), (
        "the lever must switch off the SERVER while leaving the store configured"
    )

    # A `lists: own` login states its own server set, so the host's switch does
    # not reach those renders; the host and every tier render over it lose the
    # server.
    app_renders = tuple(
        prefix
        for app in ("channel-finder", "knowledge", "logbook")
        for prefix in (f"{repo.name}-{app}/", f".image/{repo.name}-{app}/")
    )
    assert [hit for hit in _graph_tool_hits(project) if not hit.startswith(app_renders)] == []
    assert "graph" not in json.loads((project / ".mcp.json").read_text())["mcpServers"]
    assert not (project / ".claude" / "agents" / "facility-knowledge-graph.md").exists(), (
        "the subagent rides the server (server_dependency): switching the server "
        "off must take the agent with it, or the render ships frontmatter naming "
        "tools no permission backs"
    )
