"""Tests for ``osprey.agent_runner.transcript_author``.

An authored part is only useful if the surfaces that read a real transcript read it
the same way, so these read it back through the session-chat parser of the Web
Terminal (``routes/session.py``) rather than checking line fields one by one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from osprey.agent_runner.transcript_author import APPROVED_CALL_NOTE, author_transcript
from osprey.interfaces.web_terminal.routes import session as session_routes
from osprey_connectors.simulation.sessions import parse_sessions, resolve

SESSION_ID = "0b6f4f5e-6a1d-5d2c-9a43-3c1f2b7d9e10"
CWD = "/home/alice/project"
CLI_VERSION = "2.1.200"
ANCHOR = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)

SESSIONS = [
    {
        "slug": "beam-loss",
        "title": "Beam loss at 14:32",
        "when": {"days_ago": 1, "time": "14:40:00"},
        "parts": [
            {
                "title": "Investigating",
                "turns": [
                    {
                        "prompt": "What happened to the beam at 14:32?",
                        "channels": ["SR:DIAG:DCCT:01:CURRENT:RB"],
                        "steps": [
                            {
                                "tool": {"server": "controls", "name": "archiver_read"},
                                "input": {"channels": ["SR:DIAG:DCCT:01:CURRENT:RB"]},
                                "result": {"points": 120, "min": 0.0, "max": 500.1},
                            },
                            {"answer": "The current dropped to zero at 14:32:08."},
                        ],
                    },
                    {
                        "prompt": "Set the corrector back to 1.5 A.",
                        "at_offset_s": 300,
                        "channels": ["SR:MAG:CORR:01:CURRENT:SP"],
                        "steps": [
                            {
                                "tool": {"server": "controls", "name": "channel_write"},
                                "input": {
                                    "operations": [
                                        {"channel": "SR:MAG:CORR:01:CURRENT:SP", "value": 1.5}
                                    ]
                                },
                                "result": "Wrote 1.5 to SR:MAG:CORR:01:CURRENT:SP.",
                                "approved": True,
                            },
                            {"answer": "Done: the corrector reads 1.5 A."},
                        ],
                    },
                ],
            }
        ],
    }
]


def _part():
    (session,) = resolve(parse_sessions(SESSIONS), ANCHOR)
    return session.parts[0]


def _lines() -> list[dict]:
    return author_transcript(_part(), session_id=SESSION_ID, cwd=CWD, cli_version=CLI_VERSION)


def _jsonl(lines: list[dict]) -> str:
    return "".join(json.dumps(line) + "\n" for line in lines)


def _replayed(tmp_path: Path) -> list[dict]:
    """The chat turns the session-chat parser makes of the authored part."""
    path = tmp_path / f"{SESSION_ID}.jsonl"
    path.write_text(_jsonl(_lines()))
    return session_routes._chat_turns(session_routes._transcript_nodes(path))


class TestAuthorTranscriptLines:
    def test_lines_carry_the_cli_fields(self):
        """Every conversation line names its session, cwd, CLI build and a timestamp."""
        for line in _lines():
            assert line["sessionId"] == SESSION_ID
            assert line["cwd"] == CWD
            assert line["version"] == CLI_VERSION
            assert datetime.fromisoformat(line["timestamp"]).tzinfo is not None
            assert line["type"] in ("user", "assistant", "attachment")

    def test_uuids_are_unique_and_parent_uuids_chain(self):
        lines = _lines()
        uuids = [line["uuid"] for line in lines]
        assert len(set(uuids)) == len(uuids)
        assert lines[0]["parentUuid"] is None
        for previous, line in zip(lines, lines[1:], strict=False):
            assert line["parentUuid"] == previous["uuid"]

    def test_uuids_derive_from_the_transcript_id(self):
        """Another transcript id gives other line uuids; the same one gives the same."""
        other = author_transcript(
            _part(),
            session_id="1c7a5b6f-0000-4000-8000-000000000000",
            cwd=CWD,
            cli_version=CLI_VERSION,
        )
        assert {line["uuid"] for line in other}.isdisjoint(line["uuid"] for line in _lines())

    def test_authoring_twice_is_byte_identical(self):
        assert _jsonl(_lines()) == _jsonl(_lines())

    def test_tool_calls_use_the_cli_tool_names(self):
        calls = [
            block
            for line in _lines()
            if line["type"] == "assistant"
            for block in line["message"]["content"]
            if block["type"] == "tool_use"
        ]
        assert [c["name"] for c in calls] == [
            "mcp__controls__archiver_read",
            "mcp__controls__channel_write",
        ]
        assert calls[0]["input"] == {"channels": ["SR:DIAG:DCCT:01:CURRENT:RB"]}

    def test_only_the_approved_call_gets_the_note(self):
        lines = _lines()
        notes = [line for line in lines if line["type"] == "attachment"]
        assert len(notes) == 1
        attachment = notes[0]["attachment"]
        assert attachment["hookEvent"] == "PostToolUse"
        assert APPROVED_CALL_NOTE in attachment["content"]
        write_call = next(
            block["id"]
            for line in lines
            if line["type"] == "assistant"
            for block in line["message"]["content"]
            if block["type"] == "tool_use" and block["name"].endswith("channel_write")
        )
        assert attachment["toolUseID"] == write_call

    def test_timestamps_never_run_backwards_or_into_the_next_turn(self):
        part = _part()
        stamps = [datetime.fromisoformat(line["timestamp"]) for line in _lines()]
        assert stamps == sorted(stamps)
        second_turn = part.turns[1].at
        first_turn_stamps = [s for s in stamps if s < second_turn]
        # The first turn's prompt, call, result and answer all sit before the second prompt.
        assert len(first_turn_stamps) == 4


class TestReadBackBySessionChat:
    def test_prompts_and_answers_replay(self, tmp_path):
        turns = _replayed(tmp_path)
        assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
        assert turns[0]["content"] == "What happened to the beam at 14:32?"
        assert turns[1]["content"] == "The current dropped to zero at 14:32:08."
        assert turns[2]["content"] == "Set the corrector back to 1.5 A."
        assert turns[3]["content"] == "Done: the corrector reads 1.5 A."

    def test_tool_rows_replay_with_their_results(self, tmp_path):
        turns = _replayed(tmp_path)
        first = [e for e in turns[1]["events"] if e["type"] in ("tool_use", "tool_result")]
        assert [e["type"] for e in first] == ["tool_use", "tool_result"]
        assert first[0]["tool_name_raw"] == "mcp__controls__archiver_read"
        assert first[1]["tool_use_id"] == first[0]["tool_use_id"]
        assert not first[1]["is_error"]
        assert "500.1" in first[1]["preview"]

    def test_the_approved_call_shows_allowed(self, tmp_path):
        turns = _replayed(tmp_path)
        (read_result,) = [e for e in turns[1]["events"] if e["type"] == "tool_result"]
        (write_result,) = [e for e in turns[3]["events"] if e["type"] == "tool_result"]
        assert read_result["decision"] is None
        assert write_result["decision"] == "allowed"
        assert "Wrote 1.5" in write_result["preview"]

    def test_every_agent_turn_closes(self, tmp_path):
        """Each answer ends its turn, so the replay does not leave the last turn running."""
        turns = _replayed(tmp_path)
        for turn in (turns[1], turns[3]):
            (result,) = [e for e in turn["events"] if e["type"] == "result"]
            assert result["is_error"] is False
            assert result["duration_ms"] > 0


def test_the_route_reads_the_note_from_here():
    """The replay matches the note this module writes: one constant, two readers."""
    assert session_routes._APPROVED_CALL_NOTE is APPROVED_CALL_NOTE
