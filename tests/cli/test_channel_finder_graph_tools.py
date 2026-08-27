"""The channel-finder subagent's rendered ``tools:`` list, per paradigm.

Two claims, and the second is what makes the first worth asserting:

1. **The subagent's vocabulary is exactly its paradigm's.**
   ``channel-finder.md.j2`` renders ``CHANNEL_FINDER_TOOLS_BY_PIPELINE[mode]``
   qualified with ``mcp__channel-finder__``, plus ``submit_response``. Nothing
   else reaches the list — not even on a project that enables other servers.

2. **The standalone ``graph`` server is not on it, in any paradigm.**
   That server belongs to the main agent (``tests/cli/test_graph_agent_surface.py``
   owns its surface). The channel-finder subagent reaches a knowledge graph
   only through the ``graph`` paradigm, whose tools arrive under the
   ``channel-finder`` server name like every other paradigm's — which is what
   lets the benchmark harness, the feedback hook and the permission validator
   treat the graph paradigm as a peer rather than a special case.

The tests run on ``control_assistant``, which ships a ``services.graphdb``
block: the ``graph`` server IS enabled on those renders, so "no ``mcp__graph__``
entry in the frontmatter" is a claim about the agent template rather than a
by-product of a project with no store. Both render paths are exercised, because
they are separate code: ``create_project`` (manager.py) and
``regenerate_claude_code`` — the render ``osprey build`` runs — each assemble
their own context.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.cli.templates.manager import TemplateManager
from osprey.cli.validate_claude_artifacts import (
    validate_agent_tools_against_permissions,
)
from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE, FRAMEWORK_SERVERS

from ._vocabulary_guard import hardcoded_vocabulary_hits

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)

_SUBMIT_RESPONSE = "mcp__osprey_workspace__submit_response"

#: The one line of the terminology partial that no other paradigm's partial
#: carries — proof that ``_terminology/graph.md.j2`` was included and not one of
#: its siblings.
_GRAPH_TERMINOLOGY_MARKER = "This is a **graph** channel database"


def _graph_paradigm_instructions(body: str) -> str:
    """The graph arm of a rendered channel finder, without the shared wrapper.

    Every paradigm renders the same YAML frontmatter and the same closing
    sections (report format, ``submit_response``), and both carry operator
    phrasings and demo addresses that illustrate *routing and formatting*
    rather than telling the agent what the facility's device kinds are called.
    The ruling is about the latter, so the guard is held to the slice it
    governs: the graph arm plus the terminology partial and the snapshot region
    it includes.
    """
    _before, separator, after_frontmatter = body.partition("\n---\n")
    assert separator, "no YAML frontmatter in the rendered channel finder"
    arm, marker, _tail = after_frontmatter.partition("You are exploring the")
    assert marker, "the shared closing sections moved; re-derive the slice"
    assert _GRAPH_TERMINOLOGY_MARKER in arm, "the slice lost the graph arm"
    return arm


def _expected_frontmatter_tools(mode: str) -> list[str]:
    """The whole ``tools:`` list for *mode*, derived the way the render derives it.

    Read from the registry rather than spelled out, so a paradigm's vocabulary
    has exactly one owner and this file cannot drift from it.
    """
    return [
        *(f"mcp__channel-finder__{tool}" for tool in CHANNEL_FINDER_TOOLS_BY_PIPELINE[mode]),
        _SUBMIT_RESPONSE,
    ]


def _graph_server_permission_entries() -> list[str]:
    """The standalone ``graph`` server's four rendered permission strings.

    ``ServerDefinition.permissions_allow`` holds BARE tool names — the settings
    template splices the ``mcp__<server>__`` prefix onto the whole list — so the
    qualification happens here rather than being restated where it could drift.
    """
    return sorted(f"mcp__graph__{tool}" for tool in FRAMEWORK_SERVERS["graph"].permissions_allow)


def _agent_path(project_dir: Path) -> Path:
    return project_dir / ".claude" / "agents" / "channel-finder.md"


def _frontmatter_tools(project_dir: Path) -> list[str]:
    """The ``tools:`` CSV of the rendered channel-finder agent, split."""
    md = _agent_path(project_dir)
    assert md.exists(), f"missing rendered agent file: {md}"
    match = _FRONTMATTER_RE.match(md.read_text(encoding="utf-8"))
    assert match, f"no YAML frontmatter in {md}"
    frontmatter = yaml.safe_load(match.group(1))
    assert isinstance(frontmatter, dict), f"frontmatter is not a mapping in {md}"
    return [entry.strip() for entry in str(frontmatter["tools"]).split(",") if entry.strip()]


def _graph_server_entries(values) -> list[str]:
    return sorted(entry for entry in values if str(entry).startswith("mcp__graph__"))


def _settings_permission_entries(project_dir: Path) -> list[str]:
    settings = json.loads((project_dir / ".claude" / "settings.json").read_text(encoding="utf-8"))
    permissions = settings.get("permissions", {})
    return [
        str(entry)
        for key in ("allow", "ask", "deny")
        for entry in permissions.get(key, []) or []
        if isinstance(entry, str)
    ]


def _control_assistant(tmp_path: Path, name: str, mode: str, *, deploy_services: bool = True):
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={
            "channel_finder_mode": mode,
            "deploy_services": deploy_services,
        },
    )
    return manager, project_dir


# ---------------------------------------------------------------------------
# The subagent's list is its paradigm's vocabulary, and nothing else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", VALID_CHANNEL_FINDER_MODES)
def test_frontmatter_is_exactly_the_paradigm_vocabulary(tmp_path, mode):
    """Every registered paradigm renders its own tools, plus ``submit_response``.

    Asserted as the whole list rather than as a subset: a tool that arrives from
    somewhere other than the paradigm's registry entry is the failure this
    catches, and a subset assertion would let it through.
    """
    _manager, project_dir = _control_assistant(tmp_path, f"cf-vocab-{mode}", mode)

    tools = _frontmatter_tools(project_dir)
    assert tools == _expected_frontmatter_tools(mode)
    assert tools[-1] == _SUBMIT_RESPONSE, "submit_response stays last in the list"


@pytest.mark.parametrize("mode", VALID_CHANNEL_FINDER_MODES)
def test_no_paradigm_puts_the_graph_server_on_the_subagent(tmp_path, mode):
    """The standalone ``graph`` server reaches the main agent and stops there.

    The fixture's precondition carries the weight: this template ships a
    ``services.graphdb`` block, so the server is enabled and its four
    permissions ARE in ``settings.json``. What must be empty is the subagent's
    frontmatter — the main agent can query the store; the channel-finder
    subagent is confined to its paradigm's tools.
    """
    _manager, project_dir = _control_assistant(tmp_path, f"cf-nograph-{mode}", mode)

    assert _graph_server_entries(_settings_permission_entries(project_dir)) == (
        _graph_server_permission_entries()
    ), "fixture precondition: the graph server must be enabled on this render"

    assert _graph_server_entries(_frontmatter_tools(project_dir)) == []
    assert "mcp__graph__" not in _agent_path(project_dir).read_text(encoding="utf-8"), (
        "the agent's prose must not name a tool its frontmatter cannot call"
    )


# ---------------------------------------------------------------------------
# The graph paradigm, specifically
# ---------------------------------------------------------------------------


def test_graph_paradigm_renders_the_four_channel_finder_tools(tmp_path):
    """Spelled out once, as the names an operator would read in the file.

    Everywhere else in this module the expectation is derived from the registry;
    here it is literal, so a registry edit that silently changed the graph
    paradigm's vocabulary has to be made deliberately in two places.
    """
    _manager, project_dir = _control_assistant(tmp_path, "cf-graph-paradigm", "graph")

    assert _frontmatter_tools(project_dir) == [
        "mcp__channel-finder__capabilities",
        "mcp__channel-finder__example_queries",
        "mcp__channel-finder__get_schema",
        "mcp__channel-finder__read_cypher",
        _SUBMIT_RESPONSE,
    ]

    server = json.loads((project_dir / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
    assert server["channel-finder"]["args"] == ["-m", "osprey.mcp_server.channel_finder_graph"], (
        "the graph paradigm ships under the channel-finder server name"
    )


def test_graph_paradigm_renders_its_terminology_partial(tmp_path):
    """The prompt arm and ``_terminology/graph.md.j2`` both reached the file.

    Without the include the agent would still list the right tools and still
    have no idea what a ``ChannelBinding`` is — a render that looks complete and
    tells the subagent nothing about the corpus it has to query.
    """
    _manager, project_dir = _control_assistant(tmp_path, "cf-graph-terms", "graph")

    body = _agent_path(project_dir).read_text(encoding="utf-8")
    assert _GRAPH_TERMINOLOGY_MARKER in body, "the graph terminology partial was not included"
    for tool in ("example_queries", "get_schema", "read_cypher"):
        assert tool in body, f"the prompt arm never names {tool}"
    assert "fullPv" in body, "the prompt must name the property the addresses come from"


def test_graph_paradigm_renders_on_the_build_path(tmp_path):
    """``regenerate_claude_code`` — the render ``osprey build`` runs — agrees.

    The rendered agent is deleted first, so what is asserted is what the build
    direction wrote and not a leftover from ``create_project``.
    """
    manager, project_dir = _control_assistant(tmp_path, "cf-graph-build", "graph")
    _agent_path(project_dir).unlink()

    manager.regenerate_claude_code(project_dir)

    assert _frontmatter_tools(project_dir) == _expected_frontmatter_tools("graph")
    assert _GRAPH_TERMINOLOGY_MARKER in _agent_path(project_dir).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Every frontmatter tool is backed by a permissions entry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", VALID_CHANNEL_FINDER_MODES)
def test_frontmatter_tools_are_backed_by_permissions(tmp_path, mode):
    """The crown-jewel invariant: ``osprey build`` raises ``BuildProfileError``
    on any error this returns, so an empty list is the difference between a
    paradigm shipping and the build aborting.
    """
    _manager, project_dir = _control_assistant(tmp_path, f"cf-backed-{mode}", mode)

    assert validate_agent_tools_against_permissions(project_dir) == []


def test_disabling_the_graph_server_leaves_the_graph_paradigm_intact(tmp_path):
    """``claude_code.servers.graph.enabled: false`` is a main-agent lever only.

    It switches off the standalone ``graph`` server while the store stays
    declared. Because the paradigm's tools come from the ``channel-finder``
    server, the subagent is untouched: it keeps all four tools, they stay backed
    by permissions, and the build does not abort.
    """
    manager, project_dir = _control_assistant(tmp_path, "cf-graph-server-off", "graph")

    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert (config.get("services") or {}).get("graphdb") is not None, "the store stays declared"
    config.setdefault("claude_code", {}).setdefault("servers", {})["graph"] = {"enabled": False}
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    manager.regenerate_claude_code(project_dir)

    assert _graph_server_entries(_settings_permission_entries(project_dir)) == []
    assert _frontmatter_tools(project_dir) == _expected_frontmatter_tools("graph")
    assert validate_agent_tools_against_permissions(project_dir) == []


def test_a_project_with_no_graph_store_still_renders_the_graph_paradigm(tmp_path):
    """``deploy_services: false`` renders ``services: {}`` — no store at all.

    The paradigm is a rendering decision and the store is a deployment one, so
    the agent renders the same either way; what changes is whether the tools it
    calls find anything at the other end. Pinned because the opposite wiring —
    a frontmatter gated on ``graphdb_configured`` — would silently produce an
    agent with no channel-finding tools whenever the store is external or
    attached rather than deployed.
    """
    _manager, project_dir = _control_assistant(
        tmp_path, "cf-graph-nostore", "graph", deploy_services=False
    )

    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
    assert (config.get("services") or {}).get("graphdb") is None, "the render shape under test"

    assert _frontmatter_tools(project_dir) == _expected_frontmatter_tools("graph")
    assert _graph_server_entries(_settings_permission_entries(project_dir)) == []
    assert validate_agent_tools_against_permissions(project_dir) == []


def test_channel_finder_standalone_renders_its_paradigm_and_no_graph_server(tmp_path):
    """The bundle that ships neither a store nor a graph paradigm."""
    project_dir = TemplateManager().create_project(
        project_name="cf-standalone-tools",
        output_dir=tmp_path,
        data_bundle="channel_finder_standalone",
        context={"channel_finder_mode": "hierarchical", "default_provider": "anthropic"},
    )

    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
    assert (config.get("services") or {}).get("graphdb") is None, "the bundle ships no store"

    assert _frontmatter_tools(project_dir) == _expected_frontmatter_tools("hierarchical")
    assert _graph_server_entries(_settings_permission_entries(project_dir)) == []
    assert validate_agent_tools_against_permissions(project_dir) == []


def test_graph_paradigm_prompt_requires_a_definitive_channel_list(tmp_path):
    """The graph arm ends its reports with an explicit ``Channels:`` list.

    The graph paradigm retrieves supersets by design, so the narrowing happens
    at the report boundary — and the first benchmark round showed that free
    prose leaks "related channels" into the answer. The rendered prompt must
    therefore carry the report contract: a final list holding exactly the
    addresses that answer the question, with context kept in prose above it.
    """
    _manager, project_dir = _control_assistant(tmp_path, "cf-graph-report", "graph")

    body = _agent_path(project_dir).read_text(encoding="utf-8")
    assert "## Report Format" in body, "the graph arm must carry the report-format section"
    assert "**Channels found**" in body, "the contract must reuse the existing list label"
    assert "**Channels found**: none" in body, "the empty answer must have a spelled-out form"


# ---------------------------------------------------------------------------
# The graph arm carries the seed-time snapshot region; the others carry none
# ---------------------------------------------------------------------------


def test_graph_paradigm_prompt_carries_the_snapshot_placeholder(tmp_path):
    """The graph arm ships the marker pair the seed-time bake rewrites.

    The channel finder's graph paradigm reads the same store the
    facility-knowledge-graph subagent does, and until it carried the region it
    paid a three-call schema prelude on every delegation that the other agent
    was designed to skip.
    """
    from osprey.services.facility_knowledge.seeder import prompt_snapshot

    _manager, project_dir = _control_assistant(tmp_path, "cf-graph-snapshot", "graph")
    body = _agent_path(project_dir).read_text(encoding="utf-8")

    assert body.count(prompt_snapshot.SNAPSHOT_BEGIN) == 1
    assert body.count(prompt_snapshot.SNAPSHOT_END) == 1
    assert body.find(prompt_snapshot.SNAPSHOT_BEGIN) < body.find(prompt_snapshot.SNAPSHOT_END)
    assert "No snapshot has been captured yet" in body


def test_the_rendered_agent_carries_no_hardcoded_vocabulary(tmp_path):
    """The prompt may not name a facility's device kinds out of the framework.

    Ruling of 2026-08-27, on issue #739: facility terminology has one source of
    truth, and for the graph paradigm it is the store's ``(c:Class).altLabel``,
    captured into the *Graph at Hand* block at seed time. A hard-coded row
    naming ``Quadrupole`` or ``dcct`` ships every deployment class labels and
    synonyms its own ontology may not carry — and a label that does not exist
    returns zero rows rather than an error, which is the failure mode this very
    prompt spends a section warning about.

    Held to the graph arm only: the YAML frontmatter's ``description`` and the
    shared "Submitting Results" section name a BPM in a delegation example and
    in a title/entry_ids format example, and both are illustrative prose about
    routing and formatting rather than a claim about what this facility's
    device kinds are called.
    """
    _manager, project_dir = _control_assistant(tmp_path, "cf-graph-vocab", "graph")
    body = _agent_path(project_dir).read_text(encoding="utf-8")

    assert hardcoded_vocabulary_hits(_graph_paradigm_instructions(body)) == [], (
        "the rendered graph-paradigm channel finder hard-codes facility "
        "vocabulary; the class synonyms belong in the Vocabulary section of "
        "the Graph at Hand block, captured from the store at seed time"
    )


@pytest.mark.parametrize("mode", [m for m in VALID_CHANNEL_FINDER_MODES if m != "graph"])
def test_other_paradigms_carry_no_snapshot_region(tmp_path, mode):
    """A file-backed paradigm has no store to snapshot, so it ships no markers."""
    from osprey.services.facility_knowledge.seeder import prompt_snapshot

    _manager, project_dir = _control_assistant(tmp_path, f"cf-nosnap-{mode}", mode)
    body = _agent_path(project_dir).read_text(encoding="utf-8")

    assert prompt_snapshot.SNAPSHOT_BEGIN not in body
    assert "Graph at Hand" not in body
