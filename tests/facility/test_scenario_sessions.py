"""A scenario's agent sessions, checked at build time.

A ``sessions`` block parses through the session model (shape, and a tool
input's channels against the turn's ``channels:``), every ``channels:`` entry
names a channel of the facility, every ``logbook_ids:`` entry names an entry of
``nominal`` or of the same scenario, and every tool step names a known OSPREY
server. Each stop is one line naming the scenario file and the path.
"""

from __future__ import annotations

import copy
from typing import Any

from osprey.facility.scenarios import check_scenario_events, check_scenario_sessions

CHANNELS = [{"id": "SR:DCCT"}, {"id": "SR:RF:VOLT"}]

NOMINAL = {"name": "nominal", "logbook": [{"entry_id": "nom-1"}]}

_TURN: dict[str, Any] = {
    "prompt": "How did the beam current look overnight?",
    "at_offset_s": 0,
    "channels": ["SR:DCCT"],
    "logbook_ids": ["nom-1", "own-1"],
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
    "title": "Overnight current",
    "when": {"days_ago": 1, "time": "08:00"},
    "parts": [{"title": "Current", "turns": [_TURN]}],
}


def _document(**turn: Any) -> dict[str, Any]:
    session = copy.deepcopy(_SESSION)
    session["parts"][0]["turns"][0].update(turn)
    warm = {"name": "warm", "logbook": [{"entry_id": "own-1"}], "sessions": [session]}
    other = {"name": "other", "logbook": [{"entry_id": "other-1"}]}
    return {"channels": CHANNELS, "scenarios": [NOMINAL, other, warm]}


def _line(document: dict[str, Any]) -> str:
    (error,) = check_scenario_sessions(document)
    return error.format_message()


_PATH = "scenarios/warm.yaml `sessions[0].parts[0].turns[0]"


def test_scenario_session_valid_block_passes() -> None:
    assert check_scenario_sessions(_document()) == []


def test_scenario_session_absent_block_passes() -> None:
    document = {"channels": CHANNELS, "scenarios": [NOMINAL, {"name": "warm"}]}
    assert check_scenario_sessions(document) == []


def test_scenario_session_unknown_channel_refused() -> None:
    line = _line(
        _document(
            channels=["SR:DCCT", "SR:NOPE"],
        )
    )
    assert line == (
        f"facility: reference-missing: scenario warm — {_PATH}.channels` names channel "
        "SR:NOPE; fix: add channel SR:NOPE or correct "
        "`sessions[0].parts[0].turns[0].channels`"
    )


def test_scenario_session_tool_input_channel_outside_channels_refused() -> None:
    line = _line(_document(channels=["SR:RF:VOLT"]))
    assert line.startswith(
        "facility: value-invalid: scenario warm — scenarios/warm.yaml `sessions`"
    )
    assert "steps[0].input.channels[0]" in line
    assert "'SR:DCCT' is not in the turn's channels: list" in line


def test_scenario_session_malformed_block_refused() -> None:
    document = _document()
    document["scenarios"][2]["sessions"] = {"slug": "not-a-list"}
    line = _line(document)
    assert line.startswith(
        "facility: value-invalid: scenario warm — scenarios/warm.yaml `sessions`"
    )
    assert "must be a list of sessions" in line


def test_scenario_session_unknown_logbook_id_refused() -> None:
    line = _line(_document(logbook_ids=["nom-1", "ghost"]))
    assert line == (
        f"facility: reference-missing: scenario warm — {_PATH}.logbook_ids` names logbook "
        "entry ghost, which neither `nominal` nor this scenario states; fix: add entry "
        "ghost to this scenario's `logbook` or correct "
        "`sessions[0].parts[0].turns[0].logbook_ids`"
    )


def test_scenario_session_another_scenarios_logbook_id_refused() -> None:
    line = _line(_document(logbook_ids=["other-1"]))
    assert "names logbook entry other-1, which neither `nominal` nor this scenario" in line


def test_scenario_session_unknown_server_refused() -> None:
    steps = [
        {
            "tool": {"server": "nowhere", "name": "channel_read"},
            "input": {},
            "result": "ok",
        }
    ]
    line = _line(_document(steps=steps))
    assert line.startswith(
        f"facility: reference-missing: scenario warm — {_PATH}.steps[0].tool.server` "
        "names server nowhere, not an OSPREY server"
    )


def test_scenario_session_errors_join_the_build_check() -> None:
    """The build reaches the sessions check through the scenario checks it runs."""
    messages = [
        e.format_message() for e in check_scenario_events(_document(channels=["SR:DCCT", "X:Y"]))
    ]
    assert any("names channel X:Y" in message for message in messages)
