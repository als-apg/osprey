"""The graph tools in the channel-finder subagent's rendered frontmatter.

``channel-finder.md.j2`` splices ``graph_tools`` into its ``tools:`` list behind
``{% if "graph" in enabled_servers %}``. The predicate is deliberately server
enablement rather than ``graphdb_configured``: the two diverge under
``claude_code.servers.graph.enabled: false``, which leaves ``services.graphdb``
in the config while the server (and therefore every ``mcp__graph__`` permission
entry) is gone. Gating on the store instead would render frontmatter tools that
``validate_agent_tools_against_permissions`` reports as unbacked, and
``osprey build`` turns that report into a ``BuildProfileError`` — the benchmark
fallback lever would abort the build it is meant to make cheap.

The tests below pin both render paths, because they are separate code:
``create_project`` (manager.py) and ``regenerate_claude_code`` — the render
``osprey build`` runs — each assemble their own context.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from osprey.cli.templates.manager import TemplateManager
from osprey.cli.validate_claude_artifacts import (
    validate_agent_tools_against_permissions,
)
from osprey.registry.mcp import FRAMEWORK_SERVERS

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)

_SUBMIT_RESPONSE = "mcp__osprey_workspace__submit_response"


def _expected_graph_tools() -> list[str]:
    """The four tool names, derived the way the render derives them.

    ``config_derived_context`` qualifies ``FRAMEWORK_SERVERS['graph']``'s bare
    ``permissions_allow`` names, so reading the registry here keeps this test
    from restating a list that has one owner.
    """
    return [f"mcp__graph__{tool}" for tool in FRAMEWORK_SERVERS["graph"].permissions_allow]


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


def _graph_entries(values) -> list[str]:
    return [entry for entry in values if entry.startswith("mcp__graph__")]


def _settings_permission_entries(project_dir: Path) -> list[str]:
    settings = json.loads((project_dir / ".claude" / "settings.json").read_text(encoding="utf-8"))
    permissions = settings.get("permissions", {})
    return [
        str(entry)
        for key in ("allow", "ask", "deny")
        for entry in permissions.get(key, []) or []
        if isinstance(entry, str)
    ]


def _control_assistant(tmp_path: Path, name: str, *, deploy_services: bool = True):
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=name,
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={
            "channel_finder_mode": "hierarchical",
            "deploy_services": deploy_services,
        },
    )
    return manager, project_dir


# ---------------------------------------------------------------------------
# The graph tools land in the frontmatter — on both render paths
# ---------------------------------------------------------------------------


def test_channel_finder_graph_tools_render_on_the_create_path(tmp_path):
    """``create_project`` on control_assistant ships the store and the tools."""
    _manager, project_dir = _control_assistant(tmp_path, "cf-graph-create")

    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
    assert (config.get("services") or {}).get("graphdb") is not None, (
        "the app template is expected to ship a services.graphdb block; "
        "without it this test would pass vacuously"
    )

    tools = _frontmatter_tools(project_dir)
    assert _graph_entries(tools) == _expected_graph_tools()
    assert tools[-1] == _SUBMIT_RESPONSE, "submit_response stays last in the list"


def test_channel_finder_graph_tools_render_on_the_build_path(tmp_path):
    """``regenerate_claude_code`` — the render ``osprey build`` runs — agrees.

    The rendered agent is deleted first, so what is asserted is what the build
    direction wrote and not a leftover from ``create_project``.
    """
    manager, project_dir = _control_assistant(tmp_path, "cf-graph-build")
    _agent_path(project_dir).unlink()

    manager.regenerate_claude_code(project_dir)

    tools = _frontmatter_tools(project_dir)
    assert _graph_entries(tools) == _expected_graph_tools()
    assert tools[-1] == _SUBMIT_RESPONSE


def test_channel_finder_graph_tools_appear_when_a_store_is_added(tmp_path):
    """Adding the block to an attached project's config.yml is enough.

    ``deploy_services: false`` renders ``services: {}``, so this starts from a
    control_assistant with no store, proves the tools are absent, then adds the
    block and re-renders. It is the same assertion as the build-path test from
    the other direction, and it cannot be satisfied by a stale file.
    """
    manager, project_dir = _control_assistant(tmp_path, "cf-graph-added", deploy_services=False)
    assert _graph_entries(_frontmatter_tools(project_dir)) == []

    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("services", {})["graphdb"] = {"port_host": 7687}
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    manager.regenerate_claude_code(project_dir)

    assert _graph_entries(_frontmatter_tools(project_dir)) == _expected_graph_tools()


# ---------------------------------------------------------------------------
# Every frontmatter tool is backed by a permissions entry
# ---------------------------------------------------------------------------


def test_channel_finder_graph_tools_are_backed_by_permissions(tmp_path):
    """The crown-jewel invariant, on a graphdb-configured render.

    ``osprey build`` raises ``BuildProfileError`` on any error this returns, so
    an empty list is the difference between the graph tools shipping and the
    build aborting.
    """
    _manager, project_dir = _control_assistant(tmp_path, "cf-graph-backed")

    assert _graph_entries(_frontmatter_tools(project_dir)) == _expected_graph_tools(), (
        "the render under test must actually carry the graph tools"
    )
    assert validate_agent_tools_against_permissions(project_dir) == []


# ---------------------------------------------------------------------------
# The two ways a project gets none of them
# ---------------------------------------------------------------------------


def test_channel_finder_graph_tools_absent_when_the_server_is_disabled(tmp_path):
    """``claude_code.servers.graph.enabled: false`` with the store still declared.

    The benchmark fallback lever: it must render, not raise, and must leave the
    graph out of BOTH surfaces — the frontmatter and settings.json — because a
    tool in one without the other is exactly what aborts the build.
    """
    manager, project_dir = _control_assistant(tmp_path, "cf-graph-off")

    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert (config.get("services") or {}).get("graphdb") is not None, "the store stays declared"
    config.setdefault("claude_code", {}).setdefault("servers", {})["graph"] = {"enabled": False}
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    manager.regenerate_claude_code(project_dir)

    assert _graph_entries(_frontmatter_tools(project_dir)) == []
    assert _graph_entries(_settings_permission_entries(project_dir)) == []
    assert validate_agent_tools_against_permissions(project_dir) == []


def test_channel_finder_graph_tools_absent_without_a_graph_store(tmp_path):
    """channel_finder_standalone ships no ``services.graphdb`` block at all."""
    project_dir = TemplateManager().create_project(
        project_name="cf-graph-standalone",
        output_dir=tmp_path,
        data_bundle="channel_finder_standalone",
        context={"channel_finder_mode": "hierarchical", "default_provider": "anthropic"},
    )

    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
    assert (config.get("services") or {}).get("graphdb") is None, "the bundle ships no store"

    tools = _frontmatter_tools(project_dir)
    assert _graph_entries(tools) == []
    assert tools[-1] == _SUBMIT_RESPONSE
    assert _graph_entries(_settings_permission_entries(project_dir)) == []
    assert validate_agent_tools_against_permissions(project_dir) == []


# ---------------------------------------------------------------------------
# The prose that tells the agent when to reach for them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deploy_services", [True, False])
def test_channel_finder_graph_guidance_follows_the_same_predicate(tmp_path, deploy_services):
    """The graph section appears exactly where the tools do, and never alone."""
    _manager, project_dir = _control_assistant(
        tmp_path, f"cf-graph-prose-{str(deploy_services).lower()}", deploy_services=deploy_services
    )

    body = _agent_path(project_dir).read_text(encoding="utf-8")
    has_tools = bool(_graph_entries(_frontmatter_tools(project_dir)))
    assert has_tools is deploy_services, "the render shape under test"
    assert ("## The Facility Knowledge Graph" in body) is deploy_services
