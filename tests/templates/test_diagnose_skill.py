"""The shipped ``diagnose`` skill reads the simulator within the rendered permissions.

A failed physics model serves its variables as NaN/INVALID and its status
channel, ``<code>:SIM:<model>:STATUS``, holds ``ok`` or the engine's own error
text. The skill teaches the rendered agent to read that channel and the model's
log with the tools the preset grants — ``channel_read`` and ``Read`` — and to
state a failure reason only from the status value. ``osprey sim status`` is the
operator's terminal equivalent; the agent cannot run it, because the rendered
``settings.json`` denies ``Bash``.

The assertions run against the real control-assistant build, so they read the
skill and the deny list exactly as a deployed agent receives them.
"""

from __future__ import annotations

import json
import re
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
from tests._builds import BuiltProject

SKILL = Path(".claude") / "skills" / "diagnose" / "SKILL.md"
SETTINGS = Path(".claude") / "settings.json"

#: The status read's tool as the rendered settings and hooks name it.
CHANNEL_READ = "mcp__controls__channel_read"

#: A tool the skill tells the agent to call: ``name(`` inside inline code.
_CALL = re.compile(r"`([a-z_]+)\(")


@pytest.fixture(scope="module")
def skill_text(built_control_assistant: BuiltProject) -> str:
    return (built_control_assistant.build_dir / SKILL).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def deny(built_control_assistant: BuiltProject) -> list[str]:
    settings = json.loads((built_control_assistant.build_dir / SETTINGS).read_text("utf-8"))
    entries: list[str] = settings["permissions"]["deny"]
    return entries


def _denied(tool: str, deny: list[str]) -> list[str]:
    """The deny entries that match ``tool``, by full name or by an MCP entry's tool part."""
    matches = []
    for entry in deny:
        if fnmatchcase(tool, entry):
            matches.append(entry)
        elif entry.startswith("mcp__") and entry.count("__") == 2:
            if fnmatchcase(tool, entry.rsplit("__", 1)[1]):
                matches.append(entry)
    return matches


def test_the_skill_reads_each_model_status_channel_through_channel_read(skill_text: str) -> None:
    assert "data/simulator/addresses.json" in skill_text
    assert "`status`" in skill_text
    assert "<code>:SIM:<model>:STATUS" in skill_text
    assert "`channel_read`" in skill_text
    assert CHANNEL_READ in skill_text


def test_the_skill_reads_the_model_log_of_each_target(skill_text: str) -> None:
    assert "var/simulator/standin/<model>.log" in skill_text
    assert "`standin`" in skill_text
    assert "var/simulator/<model>.log" in skill_text
    assert "`virtual_accelerator`" in skill_text
    assert "`inprocess`" in skill_text


def test_the_skill_names_the_terminal_equivalent_and_the_log_paths_it_prints(
    skill_text: str,
) -> None:
    assert "osprey sim status" in skill_text
    assert "log:" in skill_text


def test_the_reason_comes_only_from_the_status_value(skill_text: str) -> None:
    assert "only from its status value" in skill_text
    assert "(log, instance <i>, pid <p>)" in skill_text
    assert "never the reason" in skill_text


def test_read_is_not_denied(deny: list[str]) -> None:
    assert _denied("Read", deny) == []


def test_no_step_depends_on_a_denied_tool(skill_text: str, deny: list[str]) -> None:
    called = sorted(set(_CALL.findall(skill_text)))
    assert "channel_read" in skill_text
    tools = ["Read", CHANNEL_READ, *called]

    denied = {tool: matches for tool in tools if (matches := _denied(tool, deny))}

    assert denied == {}


def test_the_terminal_equivalent_is_the_operator_s_never_the_agent_s(
    skill_text: str, deny: list[str]
) -> None:
    assert _denied("Bash", deny) == ["Bash"]
    assert "```bash" not in skill_text
    assert "operator" in skill_text
