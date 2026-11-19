"""The prompts name the directory's tools only when the directory server renders.

``config_derived_context`` writes ``channel_directory_tools``: the ``channel-directory``
registry entry's allowed tools minus those the section's predicates withhold,
in registry order. The main agent's CLAUDE.md, the channel-finder login's
CLAUDE.md and the channel-finder subagent read that one key, and every arm is
gated on the server being among the render's enabled servers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from tests._vocabulary import PROTOCOL_WORDS

from osprey.cli.templates.claude_code import config_derived_context
from osprey.cli.templates.manager import TemplateManager

_DIRECTORY = {"type": "channelfinder", "settings": {"url": "http://directory.invalid"}}
_ALARM_LOG = {"type": "phoebus_alarm_logger", "settings": {"url": "http://alarms.invalid"}}
_ALARM_STATE = {
    "type": "kafka",
    "settings": {"bootstrap_servers": "kafka.invalid:9092", "topic": "Accelerator"},
}

_FLEET = {**_DIRECTORY, "fleet": {}}

_MAIN = "claude_code/CLAUDE.md.j2"
_APP = "claude_code/CLAUDE.channel-finder.md.j2"
_SUBAGENT = "claude_code/claude/agents/channel-finder.md.j2"

#: The heading every arm opens with, in the three templates.
_ARM_HEADINGS = {
    _MAIN: "## Control-System Directory",
    _APP: "## Control-System Directory",
    _SUBAGENT: "## Control-System Directory",
}


def _context(tmp_path: Path, section: dict[str, Any] | None, *, served: bool) -> dict[str, Any]:
    config: dict[str, Any] = {"project_name": "demo"}
    if section is not None:
        config["channel_directory"] = section
    ctx = config_derived_context(config, tmp_path)
    servers = {"controls", "osprey_workspace", "channel-finder"}
    if served:
        servers.add("channel-directory")
    ctx.update(
        {
            "facility_name": "Demo",
            "enabled_servers": servers,
            "enabled_agents": {"channel-finder"},
            "agents": [
                {"name": "channel-finder", "enabled": True, "description": "Finds channels"}
            ],
            "channel_finder_mode": "in_context",
            "channel_finder_tools": ["query_channels", "get_channels"],
        }
    )
    return ctx


def _render(template: str, ctx: dict[str, Any]) -> str:
    return TemplateManager().jinja_env.get_template(template).render(**ctx)


def _arm(template: str, text: str) -> str:
    heading = _ARM_HEADINGS[template]
    assert heading in text, f"{template} renders no directory arm"
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def _frontmatter_tools(text: str) -> list[str]:
    line = next(line for line in text.splitlines() if line.startswith("tools: "))
    return [t.strip() for t in line.removeprefix("tools: ").split(",")]


# -- the context key ---------------------------------------------------------


def test_a_full_section_lists_every_tool_in_registry_order(tmp_path: Path) -> None:
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    section = {
        **_DIRECTORY,
        "alarm_log": _ALARM_LOG,
        "alarm_state": _ALARM_STATE,
        "put_log": {"type": "olog_put_log", "settings": {"url": "http://puts.invalid"}},
        "fleet": {},
    }
    ctx = config_derived_context({"channel_directory": section}, tmp_path)
    assert (
        ctx["channel_directory_tools"] == FRAMEWORK_SERVERS["channel-directory"].permissions_allow
    )


def test_a_section_with_only_alarm_log_lists_history_and_config(tmp_path: Path) -> None:
    tools = config_derived_context(
        {"channel_directory": {**_DIRECTORY, "alarm_log": _ALARM_LOG}}, tmp_path
    )["channel_directory_tools"]
    assert "alarm_history" in tools
    assert "alarm_config" in tools
    assert "alarm_state" not in tools
    assert "active_alarms" not in tools
    assert "put_history" not in tools


def test_a_section_with_only_alarm_state_names_no_put_history(tmp_path: Path) -> None:
    ctx = _context(tmp_path, {**_DIRECTORY, "alarm_state": _ALARM_STATE}, served=True)
    assert "put_history" not in ctx["channel_directory_tools"]
    assert "alarm_state" in ctx["channel_directory_tools"]
    assert "active_alarms" in ctx["channel_directory_tools"]
    for template in (_MAIN, _APP, _SUBAGENT):
        assert "put_history" not in _render(template, ctx), template


def test_the_only_alarm_log_lists_reach_every_rendered_arm(tmp_path: Path) -> None:
    ctx = _context(tmp_path, {**_DIRECTORY, "alarm_log": _ALARM_LOG}, served=True)
    for template in (_MAIN, _APP):
        arm = _arm(template, _render(template, ctx))
        assert "`alarm_history`" in arm and "`alarm_config`" in arm, template
        assert "alarm_state" not in arm and "put_history" not in arm, template
    tools = _frontmatter_tools(_render(_SUBAGENT, ctx))
    assert "mcp__channel-directory__alarm_history" in tools
    assert "mcp__channel-directory__alarm_state" not in tools


# -- rendered when served, absent otherwise ----------------------------------


@pytest.mark.parametrize("template", [_MAIN, _APP, _SUBAGENT])
def test_the_arm_renders_when_the_server_does(tmp_path: Path, template: str) -> None:
    arm = _arm(template, _render(template, _context(tmp_path, _DIRECTORY, served=True)))
    assert "curated channels first" in arm
    assert "not curated" in arm
    assert "no_match" in arm and "not evidence" in arm
    assert "did not answer" in arm
    assert "the user's words" in arm
    assert "links.panel" in arm
    assert PROTOCOL_WORDS.search(arm) is None, arm


@pytest.mark.parametrize("template", [_MAIN, _APP, _SUBAGENT])
@pytest.mark.parametrize("section", [_DIRECTORY, _FLEET], ids=["directory", "fleet"])
def test_on_a_tango_machine_the_arm_names_no_epics_word(
    tmp_path: Path, template: str, section: dict[str, Any]
) -> None:
    ctx = _context(tmp_path, section, served=True)
    ctx["control_system"] = {"type": "tango"}

    arm = _arm(template, _render(template, ctx))

    assert re.search(r"\bPVs?\b|\bIOCs?\b", arm) is None, arm
    assert "server" in arm


@pytest.mark.parametrize("template", [_MAIN, _APP, _SUBAGENT])
@pytest.mark.parametrize(
    ("section", "served"),
    [(None, False), (_DIRECTORY, False)],
    ids=["no-section", "server-disabled"],
)
def test_the_arm_is_absent_when_the_server_is(
    tmp_path: Path, template: str, section: dict | None, served: bool
) -> None:
    text = _render(template, _context(tmp_path, section, served=served))
    assert _ARM_HEADINGS[template] not in text
    assert "channel-directory" not in text
    assert "directory_search" not in text


def test_the_main_arms_route_a_channel_or_server_to_open_panel(tmp_path: Path) -> None:
    ctx = _context(tmp_path, _DIRECTORY, served=True)
    for template in (_MAIN, _APP):
        arm = _arm(template, _render(template, ctx))
        assert "`open_panel`" in arm, template
        assert "split" in arm, template
        assert "replace" not in arm, template


def test_the_main_agent_reads_server_health_with_channel_read(tmp_path: Path) -> None:
    arm = _arm(_MAIN, _render(_MAIN, _context(tmp_path, _DIRECTORY, served=True)))
    assert "`server_record`" in arm
    assert "`channel_read`" in arm


def test_the_app_answers_server_health_by_opening_the_server_page(tmp_path: Path) -> None:
    arm = _arm(_APP, _render(_APP, _context(tmp_path, _DIRECTORY, served=True)))
    assert "channel_read" not in arm
    assert "server page" in arm
    assert "`server_record`" in arm


def test_the_subagent_carries_the_link_and_never_opens_a_panel(tmp_path: Path) -> None:
    text = _render(_SUBAGENT, _context(tmp_path, _DIRECTORY, served=True))
    assert "open_panel" not in text
    tools = _frontmatter_tools(text)
    assert tools[-1] == "mcp__osprey_workspace__submit_response"
    assert "mcp__channel-directory__directory_search" in tools
    assert "mcp__channel-finder__query_channels" in tools


def test_the_subagent_frontmatter_is_unchanged_without_the_server(tmp_path: Path) -> None:
    text = _render(_SUBAGENT, _context(tmp_path, _DIRECTORY, served=False))
    assert _frontmatter_tools(text) == [
        "mcp__channel-finder__query_channels",
        "mcp__channel-finder__get_channels",
        "mcp__osprey_workspace__submit_response",
    ]


# -- server liveness with the fleet on and off --------------------------------

#: The liveness line each arm renders with the fleet off, as before the fleet.
_LIVENESS_WITHOUT_FLEET = {
    _MAIN: (
        "- Whether a server is alive, and its engineer: `server_record` names its health "
        "channels; read them with `channel_read`. An advancing HEARTBEAT means alive.\n"
    ),
    _APP: (
        "- Whether a server is alive, and its engineer: open its server page, the\n"
        "  `links.panel` of `server_record`; the page reads its health channels.\n"
    ),
}


def _without_controls(ctx: dict[str, Any]) -> dict[str, Any]:
    return {**ctx, "enabled_servers": ctx["enabled_servers"] - {"controls"}}


def test_the_fleet_tool_is_listed_only_with_the_fleet(tmp_path: Path) -> None:
    assert (
        "server_fleet"
        in config_derived_context({"channel_directory": _FLEET}, tmp_path)[
            "channel_directory_tools"
        ]
    )
    assert (
        "server_fleet"
        not in config_derived_context({"channel_directory": _DIRECTORY}, tmp_path)[
            "channel_directory_tools"
        ]
    )


@pytest.mark.parametrize("template", [_MAIN, _APP])
@pytest.mark.parametrize("controls", [True, False], ids=["controls", "no-controls"])
def test_with_the_fleet_the_arm_names_server_fleet_and_the_record_state(
    tmp_path: Path, template: str, controls: bool
) -> None:
    ctx = _context(tmp_path, _FLEET, served=True)
    if not controls:
        ctx = _without_controls(ctx)
    arm = _arm(template, _render(template, ctx))
    assert "`server_fleet`" in arm
    assert "which servers are down" in arm.lower()
    assert "`server_record` carries its state" in arm
    # Quoted, not backticked: a backticked word in a render is a tool name
    # (tests/templates/test_shipped_prompt_registry.py).
    assert '"unreached"' in arm and '"unknown"' in arm and "not that" in arm
    assert "HEARTBEAT" not in arm
    assert "channel_read" not in arm
    assert "reads its health channels" not in arm
    assert "`open_panel`" in arm
    assert PROTOCOL_WORDS.search(arm) is None, arm


@pytest.mark.parametrize("template", [_MAIN, _APP])
@pytest.mark.parametrize("controls", [True, False], ids=["controls", "no-controls"])
def test_without_the_fleet_the_arm_is_unchanged(
    tmp_path: Path, template: str, controls: bool
) -> None:
    ctx = _context(tmp_path, _DIRECTORY, served=True)
    if not controls:
        ctx = _without_controls(ctx)
    arm = _arm(template, _render(template, ctx))
    assert "server_fleet" not in arm
    assert "carries its state" not in arm
    if template == _MAIN and not controls:
        assert "open the server page from `server_record`'s `links.panel`" in arm
    else:
        assert _LIVENESS_WITHOUT_FLEET[template] in arm


@pytest.mark.parametrize("template", [_MAIN, _APP])
def test_the_fleet_changes_only_the_liveness_lines(tmp_path: Path, template: str) -> None:
    off = _render(template, _context(tmp_path, _DIRECTORY, served=True))
    on = _render(template, _context(tmp_path, _FLEET, served=True))
    changed = [line for line in on.splitlines() if line not in off.splitlines()]
    gone = [line for line in off.splitlines() if line not in on.splitlines()]
    assert changed and gone
    # The prose still names the server in the control system's word, the tools in ours.
    assert all(
        "ioc" in line.lower() or "server" in line.lower() or line.startswith("  ")
        for line in changed + gone
    )
    before, after = off.split(_ARM_HEADINGS[template], 1), on.split(_ARM_HEADINGS[template], 1)
    assert before[0] == after[0]
    assert before[1].split("\n## ", 1)[1] == after[1].split("\n## ", 1)[1]


def test_the_subagent_changes_only_by_the_fleet_tool(tmp_path: Path) -> None:
    def without_fleet_tool(text: str) -> str:
        return text.replace("`server_fleet`, ", "").replace(
            " mcp__channel-directory__server_fleet,", ""
        )

    off = _render(_SUBAGENT, _context(tmp_path, _DIRECTORY, served=True))
    on = _render(_SUBAGENT, _context(tmp_path, _FLEET, served=True))
    assert "server_fleet" in on
    assert without_fleet_tool(on) == off
