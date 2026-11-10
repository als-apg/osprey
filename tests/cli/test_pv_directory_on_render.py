"""A configured ``pv_directory`` section reaches every rendered ``.claude/`` artifact.

The demo project is built once, the way ``test_profile_to_claude_contract`` builds
it: render, stamp the manifest, overlay the preset's ``config:``, regenerate. Each
configuration below is a copy of that build with one ``pv_directory`` section
written into ``config.yml`` and ``.claude/`` regenerated from it.

A configured section renders the ``pv-directory`` server, its allow rows, the
channel-finder subagent's tools and the main agent's directory arm. A gated tool
appears exactly when its sub-block does: ``alarm_history`` with ``alarm_log``,
``alarm_state`` with ``alarm_state``, ``put_history`` with ``put_log``, and
``alarm_config`` with either alarm source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from osprey.cli.templates.manager import TemplateManager
from osprey.health.core.rendered import rendered
from osprey.health.models import Status
from osprey.registry.mcp import FRAMEWORK_SERVERS
from tests._builds import copy_build
from tests._preset_data import bundle_data_root
from tests.cli._template_projects import create_project_with_manifest

SERVER = "pv-directory"
PREFIX = f"mcp__{SERVER}__"
DIRECTORY_TOOLS = ("directory_properties", "directory_search", "ioc_record", "pv_record")
GATED_TOOLS = ("alarm_config", "alarm_history", "alarm_state", "put_history")

_DIRECTORY = {"type": "channelfinder", "settings": {"url": "http://directory.invalid"}}
_ALARM_LOG = {"type": "phoebus_alarm_logger", "settings": {"url": "http://alarms.invalid"}}
_ALARM_STATE = {
    "type": "phoebus_alarm_server",
    "settings": {"bootstrap_servers": "kafka.invalid:9092", "topic": "Accelerator"},
}
_PUT_LOG = {"type": "olog_put_log", "settings": {"url": "http://puts.invalid"}}

#: name -> (section, the gated tools it serves)
SECTIONS = {
    "full": (
        {**_DIRECTORY, "alarm_log": _ALARM_LOG, "alarm_state": _ALARM_STATE, "put_log": _PUT_LOG},
        {"alarm_config", "alarm_history", "alarm_state", "put_history"},
    ),
    "directory-only": (dict(_DIRECTORY), set()),
    "alarm-log-only": (
        {**_DIRECTORY, "alarm_log": _ALARM_LOG},
        {"alarm_config", "alarm_history"},
    ),
    "alarm-state-only": (
        {**_DIRECTORY, "alarm_state": _ALARM_STATE},
        {"alarm_config", "alarm_state"},
    ),
    "put-log-only": ({**_DIRECTORY, "put_log": _PUT_LOG}, {"put_history"}),
    "va-demo-replay": (
        {"alarm_state": {"replay_file": "demo:example-alarms"}},
        {"alarm_config", "alarm_state"},
    ),
}

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
_ARM_HEADING = "## Control-System Directory"
_IOC_PAGE_FALLBACK = "open the IOC page from `ioc_record`'s `links.panel`"


# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    """The demo project with the preset's config and no ``pv_directory`` section."""
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.utils.config_writer import config_update_fields
    from tests._facility_file import write_facility_views

    manager = TemplateManager()
    project = create_project_with_manifest(
        manager,
        project_name="pv-directory-on",
        output_dir=tmp_path_factory.mktemp("pv_directory_on_base"),
        data_bundle="demo",
        context={"channel_finder_mode": "hierarchical"},
        data_root=bundle_data_root("demo"),
    )
    profile, _profile_dir = resolve_build_profile(None, preset="demo")
    # The demo preset ships a pv_directory section; the base is the preset without it.
    config_update_fields(
        project / "config.yml",
        {key: value for key, value in profile.config.items() if key != "pv_directory"},
    )
    write_facility_views(project, "demo")
    manager.regenerate_claude_code(project)
    return project


@pytest.fixture(scope="module")
def renders(base, tmp_path_factory) -> dict[str, Path]:
    """One copy of *base* per entry of :data:`SECTIONS`, regenerated with its section."""
    from osprey.utils.config_writer import config_update_fields

    out: dict[str, Path] = {}
    for name, (section, _tools) in SECTIONS.items():
        project = copy_build(base, tmp_path_factory.mktemp(f"pv_directory_on_{name}"))
        config_update_fields(project / "config.yml", {"pv_directory": section})
        TemplateManager().regenerate_claude_code(project)
        out[name] = project
    return out


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _servers(project: Path) -> dict:
    return json.loads((project / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]


def _permissions(project: Path) -> dict:
    return json.loads((project / ".claude" / "settings.json").read_text(encoding="utf-8"))[
        "permissions"
    ]


def _own(entries) -> list[str]:
    return [e.removeprefix(PREFIX) for e in entries if e.startswith(PREFIX)]


def _subagent_tools(project: Path) -> list[str]:
    text = (project / ".claude" / "agents" / "channel-finder.md").read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    assert match, "channel-finder.md has no frontmatter"
    tools = yaml.safe_load(match.group(1))["tools"]
    return [t.strip() for t in str(tools).split(",") if t.strip()]


def _arm(project: Path) -> str:
    text = (project / "CLAUDE.md").read_text(encoding="utf-8")
    assert _ARM_HEADING in text, "CLAUDE.md renders no directory arm"
    return text.split(_ARM_HEADING, 1)[1].split("\n## ", 1)[0]


def _arm_tools(project: Path) -> list[str]:
    line = next(ln for ln in _arm(project).splitlines() if "Its tools:" in ln)
    return re.findall(r"`([a-z_]+)`", line.split("Its tools:", 1)[1])


def _expected(name: str) -> list[str]:
    """The section's tools, in registry order."""
    served = set(DIRECTORY_TOOLS) | SECTIONS[name][1]
    return [t for t in FRAMEWORK_SERVERS[SERVER].permissions_allow if t in served]


# ---------------------------------------------------------------------------
# Unconfigured
# ---------------------------------------------------------------------------


def test_without_a_section_nothing_names_the_directory(base: Path) -> None:
    assert SERVER not in _servers(base)
    permissions = _permissions(base)
    assert _own(permissions["allow"]) == []
    assert _own(permissions.get("ask", [])) == []
    assert not any(t.startswith(PREFIX) for t in _subagent_tools(base))
    assert _ARM_HEADING not in (base / "CLAUDE.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Configured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SECTIONS))
def test_the_server_renders_with_its_section(renders: dict[str, Path], name: str) -> None:
    servers = _servers(renders[name])
    assert SERVER in servers
    assert servers[SERVER]["env"]["OSPREY_CONFIG"] == servers[SERVER]["env"]["CONFIG_FILE"]


@pytest.mark.parametrize("name", list(SECTIONS))
def test_allow_rows_are_exactly_the_served_tools(renders: dict[str, Path], name: str) -> None:
    permissions = _permissions(renders[name])
    assert sorted(_own(permissions["allow"])) == sorted(_expected(name))
    assert _own(permissions.get("ask", [])) == []


@pytest.mark.parametrize("name", list(SECTIONS))
def test_the_subagent_lists_the_served_tools(renders: dict[str, Path], name: str) -> None:
    assert _own(_subagent_tools(renders[name])) == _expected(name)


@pytest.mark.parametrize("name", list(SECTIONS))
def test_the_main_arm_names_the_subagents_tools(renders: dict[str, Path], name: str) -> None:
    project = renders[name]
    assert _arm_tools(project) == _expected(name)
    assert set(_arm_tools(project)) == set(_own(_subagent_tools(project)))


@pytest.mark.parametrize("name", list(SECTIONS))
@pytest.mark.parametrize("tool", GATED_TOOLS)
def test_a_gated_tool_renders_exactly_with_its_sub_block(
    renders: dict[str, Path], name: str, tool: str
) -> None:
    project = renders[name]
    served = tool in SECTIONS[name][1]
    assert (tool in _own(_permissions(project)["allow"])) is served
    assert (tool in _own(_subagent_tools(project))) is served
    assert (tool in _arm_tools(project)) is served


def test_the_va_demo_replay_serves_alarm_state_and_config_only(
    renders: dict[str, Path],
) -> None:
    tools = _subagent_tools(renders["va-demo-replay"])
    assert _own(tools) == [
        "alarm_config",
        "alarm_state",
        "directory_properties",
        "directory_search",
        "ioc_record",
        "pv_record",
    ]
    assert not any(t.endswith(("__alarm_history", "__put_history")) for t in tools)


def test_the_main_arm_reads_ioc_health_with_channel_read(renders: dict[str, Path]) -> None:
    project = renders["va-demo-replay"]
    assert "controls" in _servers(project)
    arm = _arm(project)
    assert "`channel_read`" in arm
    assert _IOC_PAGE_FALLBACK not in arm


def test_the_rendered_health_row_counts_the_server(base: Path, renders: dict[str, Path]) -> None:
    project = renders["va-demo-replay"]
    rows = {row.name: row for row in rendered({}, cwd=project)()}
    row = rows["rendered.mcp_json"]

    assert row.status is Status.OK, row
    assert set(_servers(project)) == set(_servers(base)) | {SERVER}
    assert row.value.startswith(f"{len(_servers(base)) + 1} servers · ")
