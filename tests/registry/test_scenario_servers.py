"""The framework's check of scenario sessions against its MCP server names.

The facility description checks a ``sessions`` block's shape, channels and
logbook ids; every tool step naming a known OSPREY server is the framework's
check, which reads ``FRAMEWORK_SERVERS`` and joins the build through
``extra_checks``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from osprey.registry.mcp import FRAMEWORK_SERVERS
from osprey.registry.scenario_servers import check_scenario_servers, scenario_server_errors
from osprey_facility.scenarios import check_scenario_sessions
from osprey_facility.validate import Validated

_PATH = "simulation/scenarios/warm.yaml `sessions[0].parts[0].turns[0]"

_TURN: dict[str, Any] = {
    "prompt": "How did the beam current look overnight?",
    "at_offset_s": 0,
    "channels": ["SR:DCCT"],
    "steps": [
        {
            "tool": {"server": "controls", "name": "channel_read"},
            "input": {"channels": ["SR:DCCT"]},
            "result": {"SR:DCCT": 500.0},
        },
        {"answer": "Steady at 500 mA."},
    ],
}

_SESSION: dict[str, Any] = {
    "slug": "overnight",
    "title": "Overnight",
    "when": {"days_ago": 1, "time": "08:00"},
    "parts": [{"title": "Current", "turns": [_TURN]}],
}


def _document(*servers: str, channels: list[str] | None = None) -> dict[str, Any]:
    session = copy.deepcopy(_SESSION)
    turn = session["parts"][0]["turns"][0]
    if channels is not None:
        turn["channels"] = channels
    turn["steps"] = [
        {"tool": {"server": server, "name": "channel_read"}, "input": {}, "result": "ok"}
        for server in servers
    ] + [{"answer": "Done."}]
    warm = {"name": "warm", "sessions": [session]}
    return {"channels": [{"id": "SR:DCCT"}], "scenarios": [{"name": "nominal"}, warm]}


def _validated(document: dict[str, Any]) -> Validated:
    return Validated(Path("facility"), "p", document=document)


def test_known_servers_pass() -> None:
    assert check_scenario_servers(_validated(_document("controls", "python"))) == []


def test_an_unknown_server_is_one_reference_missing() -> None:
    (error,) = check_scenario_servers(_validated(_document("nowhere")))
    assert error.format_message() == (
        f"facility: reference-missing: scenario warm — {_PATH}.steps[0].tool.server` "
        f"names server nowhere, not an OSPREY server; fix: name one of "
        f"{', '.join(sorted(FRAMEWORK_SERVERS))} in "
        "`sessions[0].parts[0].turns[0].steps[0].tool.server`"
    )


def test_every_unknown_step_stops_in_step_order() -> None:
    errors = check_scenario_servers(_validated(_document("a", "controls", "b")))
    assert [e.detail.split("`")[1] for e in errors] == [
        "sessions[0].parts[0].turns[0].steps[0].tool.server",
        "sessions[0].parts[0].turns[0].steps[2].tool.server",
    ]


def test_a_malformed_block_is_left_to_the_description() -> None:
    document = _document("nowhere")
    document["scenarios"][1]["sessions"] = {"slug": "not-a-list"}
    assert check_scenario_servers(_validated(document)) == []
    assert len(check_scenario_sessions(document)) == 1


def test_a_scenario_without_sessions_passes() -> None:
    document = {"scenarios": [{"name": "nominal"}, {"name": "warm"}]}
    assert check_scenario_servers(_validated(document)) == []


def test_the_server_names_are_supplied() -> None:
    assert scenario_server_errors(_document("mine"), {"mine"}) == []
    (error,) = scenario_server_errors(_document("controls"), {"mine"})
    assert "fix: name one of mine in" in error.format_message()


def test_the_description_no_longer_checks_servers() -> None:
    assert check_scenario_sessions(_document("nowhere")) == []
