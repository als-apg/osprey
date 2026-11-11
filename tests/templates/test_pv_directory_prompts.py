"""The prompts name the directory's tools only when the directory server renders.

``config_derived_context`` writes ``pv_directory_tools``: the ``pv-directory``
registry entry's allowed tools minus those the section's predicates withhold,
in registry order. The main agent's CLAUDE.md, the channel-finder login's
CLAUDE.md and the channel-finder subagent read that one key, and every arm is
gated on the server being among the render's enabled servers.
"""

from __future__ import annotations

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
        config["pv_directory"] = section
    ctx = config_derived_context(config, tmp_path)
    servers = {"controls", "osprey_workspace", "channel-finder"}
    if served:
        servers.add("pv-directory")
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
    }
    ctx = config_derived_context({"pv_directory": section}, tmp_path)
    assert ctx["pv_directory_tools"] == FRAMEWORK_SERVERS["pv-directory"].permissions_allow


def test_a_section_with_only_alarm_log_lists_history_and_config(tmp_path: Path) -> None:
    tools = config_derived_context(
        {"pv_directory": {**_DIRECTORY, "alarm_log": _ALARM_LOG}}, tmp_path
    )["pv_directory_tools"]
    assert "alarm_history" in tools
    assert "alarm_config" in tools
    assert "alarm_state" not in tools
    assert "put_history" not in tools


def test_a_section_with_only_alarm_state_names_no_put_history(tmp_path: Path) -> None:
    ctx = _context(tmp_path, {**_DIRECTORY, "alarm_state": _ALARM_STATE}, served=True)
    assert "put_history" not in ctx["pv_directory_tools"]
    assert "alarm_state" in ctx["pv_directory_tools"]
    for template in (_MAIN, _APP, _SUBAGENT):
        assert "put_history" not in _render(template, ctx), template


def test_the_only_alarm_log_lists_reach_every_rendered_arm(tmp_path: Path) -> None:
    ctx = _context(tmp_path, {**_DIRECTORY, "alarm_log": _ALARM_LOG}, served=True)
    for template in (_MAIN, _APP):
        arm = _arm(template, _render(template, ctx))
        assert "`alarm_history`" in arm and "`alarm_config`" in arm, template
        assert "alarm_state" not in arm and "put_history" not in arm, template
    tools = _frontmatter_tools(_render(_SUBAGENT, ctx))
    assert "mcp__pv-directory__alarm_history" in tools
    assert "mcp__pv-directory__alarm_state" not in tools


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
    assert "pv-directory" not in text
    assert "directory_search" not in text


def test_the_main_arms_route_a_channel_or_ioc_to_open_panel(tmp_path: Path) -> None:
    ctx = _context(tmp_path, _DIRECTORY, served=True)
    for template in (_MAIN, _APP):
        arm = _arm(template, _render(template, ctx))
        assert "`open_panel`" in arm, template
        assert "split" in arm, template
        assert "replace" not in arm, template


def test_the_main_agent_reads_ioc_health_with_channel_read(tmp_path: Path) -> None:
    arm = _arm(_MAIN, _render(_MAIN, _context(tmp_path, _DIRECTORY, served=True)))
    assert "`ioc_record`" in arm
    assert "`channel_read`" in arm


def test_the_app_answers_ioc_health_by_opening_the_ioc_page(tmp_path: Path) -> None:
    arm = _arm(_APP, _render(_APP, _context(tmp_path, _DIRECTORY, served=True)))
    assert "channel_read" not in arm
    assert "IOC page" in arm
    assert "`ioc_record`" in arm


def test_the_subagent_carries_the_link_and_never_opens_a_panel(tmp_path: Path) -> None:
    text = _render(_SUBAGENT, _context(tmp_path, _DIRECTORY, served=True))
    assert "open_panel" not in text
    tools = _frontmatter_tools(text)
    assert tools[-1] == "mcp__osprey_workspace__submit_response"
    assert "mcp__pv-directory__directory_search" in tools
    assert "mcp__channel-finder__query_channels" in tools


def test_the_subagent_frontmatter_is_unchanged_without_the_server(tmp_path: Path) -> None:
    text = _render(_SUBAGENT, _context(tmp_path, _DIRECTORY, served=False))
    assert _frontmatter_tools(text) == [
        "mcp__channel-finder__query_channels",
        "mcp__channel-finder__get_channels",
        "mcp__osprey_workspace__submit_response",
    ]
