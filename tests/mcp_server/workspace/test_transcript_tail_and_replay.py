"""Tests for the transcript tail rule and full-fidelity chat replay.

The tail rule reads a transcript's newest message entry to decide whether the
agent that owns it is mid-turn; replay reads the same transcript as
conversation, without the TUI's bookkeeping and without a per-message cap.
Fixtures mirror the shapes Claude Code actually writes: a ``/model`` slash
command, an interrupted turn, a ``/clear``, and a prompt nobody answered yet.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from osprey.mcp_server.workspace.transcript_reader import (
    COMMAND_NAME_PREFIX,
    INTERRUPT_MARKER,
    LOCAL_COMMAND_STDERR_PREFIX,
    LOCAL_COMMAND_STDOUT_PREFIX,
    MAX_CHAT_MESSAGE_LENGTH,
    TranscriptReader,
    is_bookkeeping_entry,
    tail_state,
)


def _ts(minute: int) -> str:
    """Generate an ISO timestamp at a fixed date with the given minute."""
    return datetime(2026, 2, 19, 12, minute, 0, tzinfo=UTC).isoformat()


def _epoch(minute: int) -> float:
    """POSIX seconds for the same instant as :func:`_ts`."""
    return datetime(2026, 2, 19, 12, minute, 0, tzinfo=UTC).timestamp()


def _user(timestamp: str, text: str, *, is_meta: bool = False) -> dict:
    """Build a user entry carrying a single text block."""
    entry: dict = {
        "type": "user",
        "timestamp": timestamp,
        "sessionId": "tail-session",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }
    if is_meta:
        entry["isMeta"] = True
    return entry


def _assistant(timestamp: str, text: str) -> dict:
    """Build an assistant entry carrying a single text block."""
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "sessionId": "tail-session",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _tool_result(timestamp: str, tool_use_id: str, content: str) -> dict:
    """Build a user entry carrying only a tool_result block."""
    return {
        "type": "user",
        "timestamp": timestamp,
        "sessionId": "tail-session",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
        },
    }


def _slash_command(timestamp: str, name: str) -> dict:
    """Build the meta entry Claude Code writes when a slash command runs."""
    return _user(
        timestamp,
        f"<command-message>{name} is running…</command-message>\n"
        f"{COMMAND_NAME_PREFIX}/{name}</command-name>",
        is_meta=True,
    )


def _command_stdout(timestamp: str, text: str) -> dict:
    """Build the entry holding a slash command's captured output."""
    return _user(timestamp, f"{LOCAL_COMMAND_STDOUT_PREFIX}{text}</local-command-stdout>")


def _write(path: Path, entries: list[dict]) -> Path:
    """Write entries as JSONL and return the path."""
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return path


# ---------------------------------------------------------------------------
# is_bookkeeping_entry
# ---------------------------------------------------------------------------


class TestIsBookkeepingEntry:
    def test_meta_entry(self):
        """An entry flagged isMeta is bookkeeping whatever it holds."""
        assert is_bookkeeping_entry(_slash_command(_ts(0), "model")) is True

    def test_command_name_without_meta_flag(self):
        """The slash-command echo counts even when isMeta is absent."""
        entry = _user(_ts(0), f"{COMMAND_NAME_PREFIX}/clear</command-name>")

        assert is_bookkeeping_entry(entry) is True

    def test_local_command_stdout(self):
        """Captured slash-command output is bookkeeping."""
        assert is_bookkeeping_entry(_command_stdout(_ts(1), "Set model to sonnet")) is True

    def test_local_command_stderr(self):
        """Captured slash-command errors are bookkeeping."""
        entry = _user(
            _ts(1), f"{LOCAL_COMMAND_STDERR_PREFIX}no such command</local-command-stderr>"
        )

        assert is_bookkeeping_entry(entry) is True

    def test_interrupt_marker(self):
        """The cancellation marker is bookkeeping."""
        entry = _user(_ts(2), f"{INTERRUPT_MARKER}]")

        assert is_bookkeeping_entry(entry) is True

    def test_ordinary_user_message(self):
        """A real prompt is not bookkeeping."""
        assert is_bookkeeping_entry(_user(_ts(0), "What is the beam current?")) is False

    def test_assistant_message(self):
        """An assistant reply is not bookkeeping."""
        assert is_bookkeeping_entry(_assistant(_ts(1), "500 mA.")) is False

    def test_tool_result_only_entry(self):
        """A tool-result-only entry carries no text and is not bookkeeping."""
        assert is_bookkeeping_entry(_tool_result(_ts(1), "tu-1", "500")) is False

    def test_non_dict(self):
        """A non-dict entry is rejected rather than raising."""
        assert is_bookkeeping_entry("not an entry") is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# read_chat_history replay
# ---------------------------------------------------------------------------


class TestChatHistoryReplay:
    def test_skips_bookkeeping_entries(self, tmp_path):
        """A /model round trip and an interrupt leave no conversation turns."""
        transcript = _write(
            tmp_path / "session.jsonl",
            [
                _user(_ts(0), "Read the current."),
                _assistant(_ts(1), "Reading it now."),
                _slash_command(_ts(2), "model"),
                _command_stdout(_ts(3), "Set model to sonnet"),
                _user(_ts(4), f"{INTERRUPT_MARKER}]"),
                _user(_ts(5), "Try again."),
            ],
        )

        history = TranscriptReader(tmp_path).read_chat_history(transcript)

        assert [(t["role"], t["content"]) for t in history] == [
            ("user", "Read the current."),
            ("assistant", "Reading it now."),
            ("user", "Try again."),
        ]

    def test_clear_command_is_not_a_turn(self, tmp_path):
        """The /clear echo is dropped, and earlier conversation survives."""
        transcript = _write(
            tmp_path / "session.jsonl",
            [
                _user(_ts(0), "Hello."),
                _slash_command(_ts(1), "clear"),
            ],
        )

        history = TranscriptReader(tmp_path).read_chat_history(transcript)

        assert len(history) == 1
        assert history[0]["content"] == "Hello."

    def test_default_cap_preserves_existing_behaviour(self, tmp_path):
        """Without max_chars, messages are still capped as before."""
        long_text = "x" * (MAX_CHAT_MESSAGE_LENGTH + 500)
        transcript = _write(tmp_path / "session.jsonl", [_user(_ts(0), long_text)])

        history = TranscriptReader(tmp_path).read_chat_history(transcript)

        assert len(history[0]["content"]) == MAX_CHAT_MESSAGE_LENGTH + 3
        assert history[0]["content"].endswith("...")

    def test_none_cap_leaves_messages_whole(self, tmp_path):
        """max_chars=None replays the full message text."""
        long_text = "x" * (MAX_CHAT_MESSAGE_LENGTH + 500)
        transcript = _write(tmp_path / "session.jsonl", [_user(_ts(0), long_text)])

        history = TranscriptReader(tmp_path).read_chat_history(transcript, max_chars=None)

        assert history[0]["content"] == long_text

    def test_explicit_cap_is_honoured(self, tmp_path):
        """A caller-supplied cap replaces the module default."""
        transcript = _write(tmp_path / "session.jsonl", [_user(_ts(0), "abcdefghij")])

        history = TranscriptReader(tmp_path).read_chat_history(transcript, max_chars=4)

        assert history[0]["content"] == "abcd..."

    def test_by_id_passes_cap_through(self, tmp_path, monkeypatch):
        """read_chat_history_by_id forwards max_chars to the file reader."""
        long_text = "y" * (MAX_CHAT_MESSAGE_LENGTH + 10)
        transcript = _write(tmp_path / "abc-123.jsonl", [_user(_ts(0), long_text)])
        reader = TranscriptReader(tmp_path)
        monkeypatch.setattr(reader, "find_transcript_by_id", lambda _sid: transcript)

        capped = reader.read_chat_history_by_id("abc-123")
        whole = reader.read_chat_history_by_id("abc-123", max_chars=None)

        assert len(capped[0]["content"]) == MAX_CHAT_MESSAGE_LENGTH + 3
        assert whole[0]["content"] == long_text


# ---------------------------------------------------------------------------
# tail_state
# ---------------------------------------------------------------------------


class TestTailState:
    def test_unanswered_prompt_is_busy(self, tmp_path):
        """A text-bearing user entry with no reply after it means mid-turn."""
        transcript = _write(
            tmp_path / "session.jsonl",
            [_assistant(_ts(0), "Done."), _user(_ts(1), "Now set the setpoint.")],
        )

        assert tail_state(transcript, _epoch(1)) == "busy"

    def test_interrupt_marker_is_idle(self, tmp_path):
        """A cancelled turn leaves the marker as the newest entry."""
        transcript = _write(
            tmp_path / "session.jsonl",
            [_user(_ts(0), "Long job."), _user(_ts(1), f"{INTERRUPT_MARKER} for tool use]")],
        )

        assert tail_state(transcript, _epoch(0)) == "idle"

    def test_command_output_newer_than_busy_stamp_is_idle(self, tmp_path):
        """/model output written after the busy stamp means the prompt returned."""
        transcript = _write(
            tmp_path / "session.jsonl",
            [
                _user(_ts(0), "Switch model."),
                _slash_command(_ts(1), "model"),
                _command_stdout(_ts(2), "Set model to sonnet"),
            ],
        )

        assert tail_state(transcript, _epoch(1)) == "idle"

    def test_command_output_older_than_busy_stamp_is_unknown(self, tmp_path):
        """Output predating the busy stamp proves nothing about this turn."""
        transcript = _write(
            tmp_path / "session.jsonl", [_command_stdout(_ts(2), "Set model to sonnet")]
        )

        assert tail_state(transcript, _epoch(9)) == "unknown"

    def test_command_output_without_busy_stamp_is_idle(self, tmp_path):
        """With no busy stamp there is no turn to be inside of."""
        transcript = _write(
            tmp_path / "session.jsonl", [_command_stdout(_ts(2), "Set model to sonnet")]
        )

        assert tail_state(transcript, None) == "idle"

    def test_command_output_without_timestamp_is_unknown(self, tmp_path):
        """An undatable entry cannot be shown to postdate the busy stamp."""
        entry = _command_stdout(_ts(2), "Set model to sonnet")
        entry["timestamp"] = ""
        transcript = _write(tmp_path / "session.jsonl", [entry])

        assert tail_state(transcript, _epoch(0)) == "unknown"

    def test_clear_echo_is_unknown(self, tmp_path):
        """The /clear echo is bookkeeping and carries no turn evidence."""
        transcript = _write(
            tmp_path / "session.jsonl",
            [_user(_ts(0), "Hello."), _slash_command(_ts(1), "clear")],
        )

        assert tail_state(transcript, _epoch(0)) == "unknown"

    def test_assistant_tail_is_unknown(self, tmp_path):
        """An assistant entry says nothing about whether the turn finished."""
        transcript = _write(
            tmp_path / "session.jsonl",
            [_user(_ts(0), "Read it."), _assistant(_ts(1), "500 mA.")],
        )

        assert tail_state(transcript, _epoch(0)) == "unknown"

    def test_tool_result_tail_is_unknown(self, tmp_path):
        """A tool-result-only user entry falls through to the caller's store."""
        transcript = _write(
            tmp_path / "session.jsonl",
            [_user(_ts(0), "Read it."), _tool_result(_ts(1), "tu-1", "500")],
        )

        assert tail_state(transcript, _epoch(0)) == "unknown"

    def test_missing_file_is_unknown(self, tmp_path):
        """A transcript that does not exist yet is unknown, not busy."""
        assert tail_state(tmp_path / "nope.jsonl", _epoch(0)) == "unknown"

    def test_empty_file_is_unknown(self, tmp_path):
        """An empty transcript holds no message entry."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text("")

        assert tail_state(transcript, _epoch(0)) == "unknown"

    def test_malformed_trailing_lines_are_skipped(self, tmp_path):
        """A half-written final line does not hide the entry before it."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(
            json.dumps(_user(_ts(0), "Set the setpoint.")) + '\n{"type": "user", "mess'
        )

        assert tail_state(transcript, _epoch(0)) == "busy"

    def test_summary_entries_do_not_mask_the_tail(self, tmp_path):
        """Non-message entries after the last turn are stepped over."""
        transcript = _write(
            tmp_path / "session.jsonl",
            [
                _user(_ts(0), "Set the setpoint."),
                {"type": "summary", "summary": "setpoint work", "leafUuid": "abc"},
            ],
        )

        assert tail_state(transcript, _epoch(0)) == "busy"

    def test_large_transcript_reads_only_the_tail(self, tmp_path):
        """A transcript far larger than the read window still classifies."""
        padding = [_assistant(_ts(0), "z" * 4000) for _ in range(200)]
        transcript = _write(
            tmp_path / "session.jsonl", [*padding, _user(_ts(1), "Set the setpoint.")]
        )
        assert transcript.stat().st_size > 256 * 1024

        assert tail_state(transcript, _epoch(1)) == "busy"

    def test_falls_back_to_full_read_for_one_huge_entry(self, tmp_path):
        """A closing entry bigger than the window is found by the full read."""
        transcript = _write(tmp_path / "session.jsonl", [_user(_ts(1), "q" * (300 * 1024))])
        assert transcript.stat().st_size > 256 * 1024

        assert tail_state(transcript, _epoch(1)) == "busy"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[Request interrupted by user]", "idle"),
        ("[Request interrupted by user for tool use]", "idle"),
        ("Request interrupted by user", "busy"),
    ],
)
def test_interrupt_marker_variants(tmp_path, text, expected):
    """Both marker wordings read as idle; prose that merely resembles it does not."""
    transcript = _write(tmp_path / "session.jsonl", [_user(_ts(0), text)])

    assert tail_state(transcript, _epoch(0)) == expected
