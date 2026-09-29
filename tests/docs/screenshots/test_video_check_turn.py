"""Unit tests for the demo-video turn-end check.

CI-safe: every case writes a small JSONL transcript into ``tmp_path`` and
drives the clock through the ``now`` argument, so nothing waits in real time.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from docs.screenshots.video_check_turn import transcript_path, turn_state

SUBMITTED = datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC).timestamp()
MTIME = SUBMITTED + 30.0


def _stamp(offset_s: float) -> str:
    return datetime.fromtimestamp(SUBMITTED + offset_s, tz=UTC).isoformat()


def _user(text: str, offset_s: float, **extra) -> dict:
    return {
        "type": "user",
        "timestamp": _stamp(offset_s),
        "message": {"role": "user", "content": text},
        **extra,
    }


def _assistant(blocks: list[dict], stop_reason, offset_s: float, model="claude-opus-5-5"):
    return {
        "type": "assistant",
        "timestamp": _stamp(offset_s),
        "message": {
            "role": "assistant",
            "model": model,
            "content": blocks,
            "stop_reason": stop_reason,
        },
    }


def _system(offset_s: float) -> dict:
    return {"type": "system", "subtype": "turn_duration", "timestamp": _stamp(offset_s)}


def _write(tmp_path: Path, entries: list, mtime: float = MTIME) -> Path:
    path = tmp_path / "session.jsonl"
    lines = [e if isinstance(e, str) else json.dumps(e) for e in entries]
    path.write_text("\n".join(lines) + "\n")
    os.utime(path, (mtime, mtime))
    return path


def _clock(t: float):
    return lambda: t


TEXT = [{"type": "text", "text": "Done."}]
THINKING = [{"type": "thinking", "thinking": "Let me plot it.", "signature": "x"}]
TOOL_USE = [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]


def _prompt() -> dict:
    return _user("plot the orbit", 0.5)


# --- ended -----------------------------------------------------------------


def test_thinking_then_text_end_turn_is_ended(tmp_path):
    path = _write(
        tmp_path,
        [_prompt(), _assistant(THINKING, None, 5.0), _assistant(TEXT, "end_turn", 6.0)],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "ended"


def test_trailing_system_lines_are_skipped(tmp_path):
    path = _write(
        tmp_path,
        [
            _prompt(),
            _assistant(TEXT, "end_turn", 6.0),
            _system(6.5),
            {"type": "file-history-snapshot", "snapshot": {}},
            "",
            "{not json",
        ],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "ended"


def test_trailing_meta_user_entry_is_skipped(tmp_path):
    path = _write(
        tmp_path,
        [_prompt(), _assistant(TEXT, "end_turn", 6.0), _user("caveat", 7.0, isMeta=True)],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "ended"


def test_ended_waits_for_stable_mtime(tmp_path):
    path = _write(tmp_path, [_prompt(), _assistant(TEXT, "end_turn", 6.0)])
    assert turn_state(path, SUBMITTED, stable_s=2.0, now=_clock(MTIME + 1.0)) == "running"
    assert turn_state(path, SUBMITTED, stable_s=2.0, now=_clock(MTIME + 2.0)) == "ended"


def test_stable_s_is_honoured(tmp_path):
    path = _write(tmp_path, [_prompt(), _assistant(TEXT, "end_turn", 6.0)])
    assert turn_state(path, SUBMITTED, stable_s=10.0, now=_clock(MTIME + 5.0)) == "running"


# --- background subagents ----------------------------------------------------
#
# The agent delegates to async subagents and ends its own turn while they run
# ("I'm waiting on channel-finder…"); their hand-back resumes it. Only a turn
# that owes no launched agent a completion notice is over.


def _launch(tool_use_id: str, offset_s: float) -> list[dict]:
    use = {"type": "tool_use", "id": tool_use_id, "name": "Agent", "input": {"prompt": "go"}}
    result = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": "Async agent launched successfully.\nagentId: a1"}],
    }
    return [
        _assistant([use], "tool_use", offset_s),
        _user([result], offset_s + 1),
    ]


def _dequeue(offset_s: float) -> dict:
    return {"type": "queue-operation", "operation": "dequeue", "timestamp": _stamp(offset_s)}


def _notification(tool_use_id: str, offset_s: float, operation: str = "enqueue") -> dict:
    return {
        "type": "queue-operation",
        "operation": operation,
        "timestamp": _stamp(offset_s),
        "content": (
            "<task-notification>\n<task-id>a1</task-id>\n"
            f"<tool-use-id>{tool_use_id}</tool-use-id>\n<status>completed</status>"
        ),
    }


WAITING = [{"type": "text", "text": "I'm waiting on channel-finder."}]


def test_end_turn_while_a_launched_agent_runs_is_running(tmp_path):
    path = _write(
        tmp_path,
        [_prompt(), *_launch("tu-a", 2.0), _assistant(WAITING, "end_turn", 4.0)],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "running"


def test_end_turn_after_every_agent_reported_back_is_ended(tmp_path):
    path = _write(
        tmp_path,
        [
            _prompt(),
            *_launch("tu-a", 2.0),
            _assistant(WAITING, "end_turn", 4.0),
            _notification("tu-a", 10.0),
            _notification("tu-a", 11.0, operation="remove"),
            *_launch("tu-b", 12.0),
            _notification("tu-b", 20.0),
            _dequeue(20.5),
            _assistant(TEXT, "end_turn", 22.0),
        ],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "ended"


def test_a_hand_back_not_yet_answered_keeps_the_turn_running(tmp_path):
    # The completion notice resumes the agent; until its next message lands,
    # the newest message is still the old "waiting" end_turn.
    path = _write(
        tmp_path,
        [
            _prompt(),
            *_launch("tu-a", 2.0),
            _assistant(WAITING, "end_turn", 4.0),
            _notification("tu-a", 10.0),
        ],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "running"


def test_the_answer_after_the_last_hand_back_ends_the_turn(tmp_path):
    path = _write(
        tmp_path,
        [
            _prompt(),
            *_launch("tu-a", 2.0),
            _assistant(WAITING, "end_turn", 4.0),
            _notification("tu-a", 10.0),
            _dequeue(10.1),
            _assistant(TEXT, "end_turn", 12.0),
        ],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "ended"


def test_a_report_queued_while_the_agent_was_still_writing_keeps_it_running(tmp_path):
    # Both reports land in the queue while the agent writes an interim reply;
    # the reply ends its turn, then the queue delivers the report and the
    # agent resumes. Until everything queued is delivered, it is not over.
    path = _write(
        tmp_path,
        [
            _prompt(),
            *_launch("tu-a", 2.0),
            _notification("tu-a", 10.0),
            _assistant(WAITING, "end_turn", 12.0),
        ],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "running"


def test_a_report_delivered_after_the_reply_keeps_it_running(tmp_path):
    path = _write(
        tmp_path,
        [
            _prompt(),
            *_launch("tu-a", 2.0),
            _notification("tu-a", 10.0),
            _assistant(WAITING, "end_turn", 12.0),
            _dequeue(12.2),
        ],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "running"


def test_one_agent_still_owed_keeps_the_turn_running(tmp_path):
    path = _write(
        tmp_path,
        [
            _prompt(),
            *_launch("tu-a", 2.0),
            *_launch("tu-b", 3.0),
            _notification("tu-a", 10.0),
            _assistant(TEXT, "end_turn", 12.0),
        ],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "running"


def test_a_foreground_agent_owes_no_notification(tmp_path):
    use = {"type": "tool_use", "id": "tu-f", "name": "Agent", "input": {"prompt": "go"}}
    result = {"type": "tool_result", "tool_use_id": "tu-f", "content": "found 3 BPMs"}
    path = _write(
        tmp_path,
        [
            _prompt(),
            _assistant([use], "tool_use", 2.0),
            _user([result], 8.0),
            _assistant(TEXT, "end_turn", 9.0),
        ],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "ended"


def test_end_turn_before_submission_is_running(tmp_path):
    """A previous turn's end_turn must not count for the prompt just submitted."""
    path = _write(tmp_path, [_user("earlier", -20.0), _assistant(TEXT, "end_turn", -10.0)])
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "running"


def test_end_turn_without_timestamp_is_running(tmp_path):
    entry = _assistant(TEXT, "end_turn", 6.0)
    del entry["timestamp"]
    path = _write(tmp_path, [_prompt(), entry])
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "running"


# --- running ---------------------------------------------------------------


def test_tool_use_tail_is_running(tmp_path):
    path = _write(tmp_path, [_prompt(), _assistant(TOOL_USE, "tool_use", 5.0)])
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 60.0)) == "running"


def test_tool_result_tail_is_running(tmp_path):
    result = {
        "type": "user",
        "timestamp": _stamp(7.0),
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        },
    }
    path = _write(tmp_path, [_prompt(), _assistant(TOOL_USE, "tool_use", 5.0), result])
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 60.0)) == "running"


def test_thinking_only_tail_is_running(tmp_path):
    path = _write(tmp_path, [_prompt(), _assistant(THINKING, None, 5.0)])
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 60.0)) == "running"


def test_prompt_only_is_running(tmp_path):
    path = _write(tmp_path, [_prompt()])
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 60.0)) == "running"


def test_missing_file_is_running(tmp_path):
    assert turn_state(tmp_path / "absent.jsonl", SUBMITTED) == "running"


def test_none_path_is_running():
    assert turn_state(None, SUBMITTED) == "running"


def test_empty_file_is_running(tmp_path):
    path = _write(tmp_path, [])
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 60.0)) == "running"


# --- errored ---------------------------------------------------------------


def _synthetic(offset_s: float) -> dict:
    return _assistant(
        [{"type": "text", "text": "API Error: 529 overloaded"}],
        "stop_sequence",
        offset_s,
        model="<synthetic>",
    )


def test_synthetic_tail_is_errored(tmp_path):
    path = _write(tmp_path, [_prompt(), _synthetic(4.0), _system(4.5)])
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "errored"


def test_synthetic_tail_is_errored_without_stable_mtime(tmp_path):
    path = _write(tmp_path, [_prompt(), _synthetic(4.0)])
    assert turn_state(path, SUBMITTED, now=_clock(MTIME)) == "errored"


def test_synthetic_before_submission_is_running(tmp_path):
    path = _write(tmp_path, [_user("earlier", -20.0), _synthetic(-10.0)])
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "running"


def test_synthetic_then_retry_in_flight_is_running(tmp_path):
    path = _write(
        tmp_path,
        [_prompt(), _synthetic(4.0), _assistant(TOOL_USE, "tool_use", 8.0)],
    )
    assert turn_state(path, SUBMITTED, now=_clock(MTIME + 5.0)) == "running"


# --- transcript_path -------------------------------------------------------


def test_transcript_path_resolves_through_reader(tmp_path, monkeypatch):
    calls = {}

    class FakeReader:
        def __init__(self, project_dir):
            calls["project_dir"] = project_dir

        def find_transcript_by_id(self, session_id):
            calls["session_id"] = session_id
            return tmp_path / f"{session_id}.jsonl"

    from docs.screenshots import video_check_turn

    monkeypatch.setattr(video_check_turn, "TranscriptReader", FakeReader)
    result = transcript_path(tmp_path / "build", "abc-123")
    assert result == tmp_path / "abc-123.jsonl"
    assert calls == {"project_dir": tmp_path / "build", "session_id": "abc-123"}


def test_transcript_path_finds_real_layout(tmp_path, monkeypatch):
    from osprey.agent_runner.project_paths import encode_claude_project_path

    config = tmp_path / "config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    build = tmp_path / "build"
    build.mkdir()
    folder = config / "projects" / encode_claude_project_path(build)
    folder.mkdir(parents=True)
    expected = folder / "sess-1.jsonl"
    expected.write_text("")
    assert transcript_path(build, "sess-1") == expected


def test_transcript_path_reads_the_session_config_dir_not_this_process(tmp_path, monkeypatch):
    # The demo session runs under its own CLAUDE_CONFIG_DIR, set only in the
    # web terminal's environment; the recorder must look there, not in its own.
    from osprey.agent_runner.project_paths import encode_claude_project_path

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "recorder-config"))
    session_config = tmp_path / "session-config"
    build = tmp_path / "build"
    build.mkdir()
    folder = session_config / "projects" / encode_claude_project_path(build)
    folder.mkdir(parents=True)
    expected = folder / "sess-1.jsonl"
    expected.write_text("")
    assert transcript_path(build, "sess-1", config_dir=session_config) == expected
    assert transcript_path(build, "sess-2", config_dir=session_config) is None


def test_transcript_path_missing_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    assert transcript_path(tmp_path / "build", "no-such-session") is None


@pytest.mark.parametrize("state", ["running", "ended", "errored"])
def test_states_are_the_documented_literals(state):
    from typing import get_args

    from docs.screenshots.video_check_turn import TurnState

    assert state in get_args(TurnState)
