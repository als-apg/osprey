"""A scenario's ``sessions`` block: parsed, refused naming the path, placed in time, round-tripped."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from datetime import time as dtime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from osprey_connectors.relative_time import RelativeTimestamp
from osprey_connectors.simulation.sessions import (
    AnswerStep,
    ArtifactSpec,
    ResolvedSession,
    ToolStep,
    from_dict,
    parse_sessions,
    resolve,
    to_dict,
)

GAUGE = "SR:VAC:GAUGE:SR07:PRESSURE:RB"
DCCT = "SR:DIAG:DCCT:01:CURRENT:RB"
HCM = "SR:MAG:HCM13:SETPOINT"


def _block() -> list[dict[str, Any]]:
    """A two-part session (the second part is the half after a /clear) and a one-part one."""
    return [
        {
            "slug": "beam-loss-review",
            "title": "Why did we lose beam at 14:32?",
            "when": {"days_ago": 1, "time": "15:10:00"},
            "parts": [
                {
                    "title": "Beam loss at 14:32",
                    "turns": [
                        {
                            "prompt": "What happened to the beam at 14:32 yesterday?",
                            "channels": [DCCT, GAUGE],
                            "logbook_ids": ["DEMO-036"],
                            "steps": [
                                {
                                    "tool": {"server": "controls", "name": "archiver_read"},
                                    "input": {"channels": [DCCT, GAUGE], "start": "-2h"},
                                    "result": "DCCT fell from 500 mA to 0 at 14:32:08.",
                                },
                                {"answer": "A vacuum burst in SR07 tripped the beam."},
                            ],
                            "artifacts": [
                                {
                                    "slug": "loss-plot",
                                    "form": "plot",
                                    "title": "DCCT and SR07 pressure",
                                    "source": "artifacts/loss-plot.json",
                                }
                            ],
                        },
                        {
                            "prompt": "Is the pressure back to normal?",
                            "at_offset_s": 95,
                            "channels": [GAUGE],
                            "steps": [
                                {
                                    "tool": {"server": "controls", "name": "channel_read"},
                                    "input": {"channels": [GAUGE]},
                                    "result": {"value": 2.1e-10},
                                },
                                {"answer": "Yes, back at its base pressure."},
                            ],
                        },
                    ],
                },
                {
                    "title": "Beam loss at 14:32 (continued)",
                    "turns": [
                        {
                            "prompt": "Nudge HCM13 by 0.1 A.",
                            "at_offset_s": 600.5,
                            "channels": [HCM],
                            "steps": [
                                {
                                    "tool": {"server": "controls", "name": "channel_write"},
                                    "input": {"operations": [{"channel": HCM, "value": 1.3}]},
                                    "result": "Rehearsed: HCM13 1.2 -> 1.3",
                                    "approved": True,
                                },
                                {"answer": "Done."},
                            ],
                        }
                    ],
                },
            ],
        },
        {
            "slug": "morning-check",
            "title": "Morning check",
            "when": {"days_ago": 3, "time": "08:00:00"},
            "parts": [
                {
                    "title": "Morning check",
                    "turns": [
                        {
                            "prompt": "Anything in the logbook overnight?",
                            "logbook_ids": ["DEMO-011", "DEMO-004"],
                            "steps": [{"answer": "Two entries, both routine."}],
                        }
                    ],
                }
            ],
        },
    ]


ANCHOR = datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)


class TestParse:
    def test_a_two_part_session_parses_into_the_model(self) -> None:
        first, second = parse_sessions(_block())

        assert first.slug == "beam-loss-review"
        assert first.when == RelativeTimestamp(days_ago=1, time=dtime(15, 10))
        assert [p.title for p in first.parts] == [
            "Beam loss at 14:32",
            "Beam loss at 14:32 (continued)",
        ]
        turn = first.parts[0].turns[0]
        assert turn.at_offset_s == 0
        assert turn.channels == (DCCT, GAUGE)
        assert turn.logbook_ids == ("DEMO-036",)
        assert turn.steps[0] == ToolStep(
            server="controls",
            name="archiver_read",
            input={"channels": [DCCT, GAUGE], "start": "-2h"},
            result="DCCT fell from 500 mA to 0 at 14:32:08.",
        )
        assert turn.steps[1] == AnswerStep(answer="A vacuum burst in SR07 tripped the beam.")
        assert turn.artifacts == (
            ArtifactSpec(
                slug="loss-plot",
                form="plot",
                title="DCCT and SR07 pressure",
                source="artifacts/loss-plot.json",
            ),
        )
        write = first.parts[1].turns[0].steps[0]
        assert isinstance(write, ToolStep) and write.approved is True
        assert second.parts[0].turns[0].channels == ()
        assert second.parts[0].turns[0].logbook_ids == ("DEMO-011", "DEMO-004")

    def test_no_block_is_no_sessions(self) -> None:
        assert parse_sessions(None) == ()
        assert parse_sessions([]) == ()

    def test_the_prefix_names_the_scenario_in_every_error(self) -> None:
        block = _block()
        block[0]["when"] = {"days_ago": 0, "time": "15:10:00"}
        with pytest.raises(ValueError, match=r"^scenario 'demo-history': sessions\[0\]\.when"):
            parse_sessions(block, prefix="scenario 'demo-history': sessions")


def _refused(block: list[dict[str, Any]], path: str) -> str:
    with pytest.raises(ValueError) as excinfo:
        parse_sessions(block)
    message = str(excinfo.value)
    assert message.startswith(path), message
    return message


class TestWhen:
    @pytest.mark.parametrize(
        ("when", "path"),
        [
            pytest.param("yesterday", "sessions[0].when", id="not-a-mapping"),
            pytest.param({"days_ago": 1}, "sessions[0].when", id="no-time"),
            pytest.param(
                {"days_ago": 1, "time": "15:10", "tz": "UTC"}, "sessions[0].when", id="extra-key"
            ),
            pytest.param(
                {"days_ago": "1", "time": "15:10:00"},
                "sessions[0].when.days_ago",
                id="days-as-text",
            ),
            pytest.param(
                {"days_ago": True, "time": "15:10:00"},
                "sessions[0].when.days_ago",
                id="days-as-bool",
            ),
            pytest.param(
                {"days_ago": 1, "time": "25:00:00"}, "sessions[0].when.time", id="hour-25"
            ),
            pytest.param({"days_ago": 1, "time": 900}, "sessions[0].when.time", id="time-number"),
            pytest.param(
                {"days_ago": 1, "time": "15:10:00+02:00"}, "sessions[0].when.time", id="tz-offset"
            ),
        ],
    )
    def test_a_bad_when_is_refused_naming_its_path(self, when: Any, path: str) -> None:
        block = _block()
        block[0]["when"] = when
        _refused(block, path)

    def test_days_ago_zero_is_refused_because_a_session_is_in_the_past(self) -> None:
        block = _block()
        block[1]["when"] = {"days_ago": 0, "time": "08:00:00"}
        message = _refused(block, "sessions[1].when.days_ago")
        assert "at least 1" in message

    def test_a_session_running_past_midnight_is_refused(self) -> None:
        block = _block()
        block[0]["when"] = {"days_ago": 1, "time": "23:55:00"}
        # The last turn sits 600.5 s after the start: 00:05:00.5 the next day.
        message = _refused(block, "sessions[0].parts[1].turns[0].at_offset_s")
        assert "midnight" in message

    def test_a_session_ending_just_before_midnight_parses(self) -> None:
        block = _block()
        block[0]["when"] = {"days_ago": 1, "time": "23:49:59"}
        assert parse_sessions(block)[0].parts[1].turns[0].at_offset_s == 600.5

    def test_turns_running_backwards_in_time_are_refused(self) -> None:
        block = _block()
        block[0]["parts"][1]["turns"][0]["at_offset_s"] = 30
        _refused(block, "sessions[0].parts[1].turns[0].at_offset_s")


class TestChannelCrossCheck:
    def test_a_channel_read_naming_a_channel_missing_from_channels_is_refused(self) -> None:
        block = _block()
        block[0]["parts"][0]["turns"][1]["steps"][0]["input"]["channels"] = [GAUGE, DCCT]
        message = _refused(block, "sessions[0].parts[0].turns[1].steps[0].input.channels[1]")
        assert repr(DCCT) in message

    def test_a_channel_write_naming_a_channel_missing_from_channels_is_refused(self) -> None:
        block = _block()
        block[0]["parts"][1]["turns"][0]["channels"] = [GAUGE]
        message = _refused(
            block, "sessions[0].parts[1].turns[0].steps[0].input.operations[0].channel"
        )
        assert repr(HCM) in message

    @pytest.mark.parametrize("name", ["archiver_read", "channel_limits"])
    def test_the_other_channel_list_tools_are_checked_too(self, name: str) -> None:
        block = _block()
        step = block[0]["parts"][0]["turns"][1]["steps"][0]
        step["tool"]["name"] = name
        step["input"]["channels"] = ["SR:NOT:LISTED"]
        _refused(block, "sessions[0].parts[0].turns[1].steps[0].input.channels[0]")

    def test_a_tool_of_another_server_is_not_cross_checked(self) -> None:
        block = _block()
        step = block[0]["parts"][0]["turns"][1]["steps"][0]
        step["tool"] = {"server": "osprey_workspace", "name": "channel_read"}
        step["input"]["channels"] = ["SR:NOT:LISTED"]
        assert parse_sessions(block)

    def test_a_malformed_operations_list_is_refused(self) -> None:
        block = _block()
        block[0]["parts"][1]["turns"][0]["steps"][0]["input"]["operations"] = [{"value": 1}]
        _refused(block, "sessions[0].parts[1].turns[0].steps[0].input.operations[0]")


class TestShape:
    @pytest.mark.parametrize(
        ("mutate", "path"),
        [
            pytest.param(lambda b: b[0].update(color="red"), "sessions[0]", id="unknown-key"),
            pytest.param(
                lambda b: b[0].update(slug="Beam Loss"), "sessions[0].slug", id="bad-slug"
            ),
            pytest.param(
                lambda b: b[1].update(slug="beam-loss-review"),
                "sessions[1].slug",
                id="duplicate-slug",
            ),
            pytest.param(lambda b: b[0].update(parts=[]), "sessions[0].parts", id="no-parts"),
            pytest.param(
                lambda b: b[0]["parts"][0].update(turns=[]),
                "sessions[0].parts[0].turns",
                id="no-turns",
            ),
            pytest.param(
                lambda b: b[0]["parts"][0]["turns"][0].update(prompt=""),
                "sessions[0].parts[0].turns[0].prompt",
                id="empty-prompt",
            ),
            pytest.param(
                lambda b: b[0]["parts"][0]["turns"][0].update(at_offset_s=-1),
                "sessions[0].parts[0].turns[0].at_offset_s",
                id="negative-offset",
            ),
            pytest.param(
                lambda b: b[0]["parts"][0]["turns"][0].update(channels="X"),
                "sessions[0].parts[0].turns[0].channels",
                id="channels-not-list",
            ),
            pytest.param(
                lambda b: b[0]["parts"][0]["turns"][0]["steps"][0].update(
                    tool="mcp__controls__archiver_read"
                ),
                "sessions[0].parts[0].turns[0].steps[0].tool",
                id="tool-as-string",
            ),
            pytest.param(
                lambda b: b[0]["parts"][0]["turns"][0]["steps"][0].update(
                    tool={"server": "controls", "name": "mcp__controls__archiver_read"}
                ),
                "sessions[0].parts[0].turns[0].steps[0].tool.name",
                id="cli-tool-name",
            ),
            pytest.param(
                lambda b: b[0]["parts"][0]["turns"][0]["steps"][0].update(approved="yes"),
                "sessions[0].parts[0].turns[0].steps[0].approved",
                id="approved-not-bool",
            ),
            pytest.param(
                lambda b: b[0]["parts"][0]["turns"][0]["steps"][0].pop("result"),
                "sessions[0].parts[0].turns[0].steps[0]",
                id="no-result",
            ),
            pytest.param(
                lambda b: b[0]["parts"][0]["turns"][0]["steps"].append({"say": "hi"}),
                "sessions[0].parts[0].turns[0].steps[2]",
                id="neither-tool-nor-answer",
            ),
            pytest.param(
                lambda b: b[0]["parts"][0]["turns"][0]["artifacts"][0].update(form="movie"),
                "sessions[0].parts[0].turns[0].artifacts[0].form",
                id="bad-form",
            ),
            pytest.param(
                lambda b: b[0]["parts"][0]["turns"][1].update(
                    artifacts=copy.deepcopy(b[0]["parts"][0]["turns"][0]["artifacts"])
                ),
                "sessions[0].parts[0].turns[1].artifacts[0].slug",
                id="duplicate-artifact",
            ),
        ],
    )
    def test_a_malformed_block_is_refused_naming_its_path(self, mutate: Any, path: str) -> None:
        block = _block()
        mutate(block)
        _refused(block, path)

    def test_a_block_that_is_not_a_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"^sessions: must be a list"):
            parse_sessions({"slug": "x"})


class TestResolve:
    def test_resolve_against_a_fixed_anchor_gives_the_expected_instants(self) -> None:
        first, second = resolve(parse_sessions(_block()), ANCHOR)

        start = datetime(2026, 11, 4, 15, 10, tzinfo=UTC)
        assert first.at == start
        part0, part1 = first.parts
        assert [rt.at for rt in part0.turns] == [start, start + timedelta(seconds=95)]
        assert part1.turns[0].at == datetime(2026, 11, 4, 15, 20, 0, 500000, tzinfo=UTC)
        (artifact,) = part0.turns[0].artifacts
        assert artifact.spec.slug == "loss-plot"
        assert artifact.at == start
        assert second.at == datetime(2026, 11, 2, 8, 0, tzinfo=UTC)

    def test_wall_clock_times_hold_in_the_anchors_timezone(self) -> None:
        anchor = datetime(2026, 11, 5, 9, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        (first, _) = resolve(parse_sessions(_block()), anchor)
        assert first.at.tzinfo == anchor.tzinfo
        assert (first.at.date().isoformat(), first.at.time()) == ("2026-11-04", dtime(15, 10))

    def test_every_instant_lies_before_the_anchor(self) -> None:
        for session in resolve(parse_sessions(_block()), ANCHOR):
            for part in session.parts:
                for turn in part.turns:
                    assert turn.at < ANCHOR

    def test_a_naive_anchor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            resolve(parse_sessions(_block()), datetime(2026, 11, 5, 12, 0))


class TestRoundTrip:
    def test_round_trip_through_json_is_lossless(self) -> None:
        resolved = resolve(parse_sessions(_block()), ANCHOR)
        for session in resolved:
            wire = json.loads(json.dumps(to_dict(session)))
            back = from_dict(wire)
            assert isinstance(back, ResolvedSession)
            assert back == session
            assert to_dict(back) == to_dict(session)

    def test_the_dict_keeps_the_blocks_own_shape_plus_instants(self) -> None:
        (first, _) = resolve(parse_sessions(_block()), ANCHOR)
        data = to_dict(first)
        assert data["at"] == "2026-11-04T15:10:00+00:00"
        assert data["parts"][0]["turns"][0]["artifacts"][0]["at"] == "2026-11-04T15:10:00+00:00"
        assert data["parts"][1]["turns"][0]["steps"][0]["tool"] == {
            "server": "controls",
            "name": "channel_write",
        }
        # Without the instants it is a valid block again.
        stripped = copy.deepcopy(data)
        stripped.pop("at")
        for part in stripped["parts"]:
            for turn in part["turns"]:
                turn.pop("at")
                for artifact in turn["artifacts"]:
                    artifact.pop("at")
        assert parse_sessions([stripped]) == (first.session,)

    def test_from_dict_refuses_a_missing_instant_naming_its_path(self) -> None:
        (first, _) = resolve(parse_sessions(_block()), ANCHOR)
        data = to_dict(first)
        del data["parts"][1]["turns"][0]["at"]
        with pytest.raises(ValueError, match=r"^session\.parts\[1\]\.turns\[0\]"):
            from_dict(data)

    def test_from_dict_refuses_a_naive_instant(self) -> None:
        (first, _) = resolve(parse_sessions(_block()), ANCHOR)
        data = to_dict(first)
        data["at"] = "2026-11-04T15:10:00"
        with pytest.raises(ValueError, match=r"^session\.at"):
            from_dict(data)
