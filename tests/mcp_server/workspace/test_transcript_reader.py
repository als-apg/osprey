"""Tests for TranscriptReader — reads Claude Code native transcripts.

Uses synthetic JSONL fixtures that mirror actual Claude Code transcript format.
"""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from osprey.mcp_server.workspace.transcript_reader import (
    MAX_CHAT_MESSAGE_LENGTH,
    MAX_ERROR_RESULT_LENGTH,
    MAX_RESULT_LENGTH,
    TranscriptReader,
    _is_error_response,
)


def _ts(minute: int) -> str:
    """Generate an ISO timestamp at a fixed date with the given minute."""
    return datetime(2026, 2, 19, 12, minute, 0, tzinfo=UTC).isoformat()


def _make_assistant_entry(timestamp: str, tool_uses: list[dict]) -> dict:
    """Build a Claude Code 'assistant' transcript entry with tool_use blocks."""
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "sessionId": "test-session-123",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "tool_use", **tu} for tu in tool_uses
            ],
        },
    }


def _make_user_entry(timestamp: str, tool_results: list[dict]) -> dict:
    """Build a Claude Code 'user' transcript entry with tool_result blocks."""
    return {
        "type": "user",
        "timestamp": timestamp,
        "sessionId": "test-session-123",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", **tr} for tr in tool_results
            ],
        },
    }


def _make_tool_result_entry(timestamp: str, tool_use_id: str, content: str,
                            is_error: bool = False) -> dict:
    """Build a standalone 'tool_result' transcript entry."""
    return {
        "type": "tool_result",
        "timestamp": timestamp,
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }


def _write_transcript(path: Path, entries: list[dict]) -> None:
    """Write entries as JSONL to a file."""
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


@pytest.fixture
def transcript_dir(tmp_path):
    """Create a project dir with Claude Code transcript directory structure."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    # Claude Code encodes path: /tmp/xxx/my-project → -tmp-xxx-my-project
    encoded = str(project_dir).replace("/", "-")
    claude_dir = tmp_path / ".claude" / "projects" / encoded
    claude_dir.mkdir(parents=True)
    return project_dir, claude_dir


# ---------------------------------------------------------------------------
# _is_error_response
# ---------------------------------------------------------------------------

class TestIsErrorResponse:
    def test_osprey_error_envelope(self):
        assert _is_error_response('{"error": true, "error_type": "validation_error"}')

    def test_not_error(self):
        assert not _is_error_response('{"status": "success"}')

    def test_invalid_json(self):
        assert not _is_error_response("not json")

    def test_none_input(self):
        assert not _is_error_response(None)


# ---------------------------------------------------------------------------
# TranscriptReader.find_transcript_dir
# ---------------------------------------------------------------------------

class TestFindTranscriptDir:
    def test_finds_existing_dir(self, transcript_dir):
        project_dir, claude_dir = transcript_dir
        reader = TranscriptReader(project_dir)
        # Patch home to our temp
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "home", lambda: claude_dir.parents[2])
            result = reader.find_transcript_dir()
        assert result == claude_dir

    def test_returns_none_when_missing(self, tmp_path):
        reader = TranscriptReader(tmp_path / "nonexistent")
        # Patch home to an empty dir
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "home", lambda: tmp_path)
            result = reader.find_transcript_dir()
        assert result is None

    def test_preserves_leading_dash_for_absolute_paths(self, tmp_path):
        """Encoding of absolute paths must keep the leading dash.

        Claude Code directories always start with '-' because absolute
        paths start with '/' which becomes '-'. Stripping it causes
        find_transcript_dir() to never find the directory.
        """
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()
        # The encoded name MUST start with '-' for absolute paths
        encoded = str(project_dir).replace("/", "-")
        assert encoded.startswith("-"), "Absolute path encoding must start with '-'"
        claude_dir = tmp_path / ".claude" / "projects" / encoded
        claude_dir.mkdir(parents=True)

        reader = TranscriptReader(project_dir)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "home", lambda: tmp_path)
            result = reader.find_transcript_dir()
        assert result is not None
        assert result == claude_dir


# ---------------------------------------------------------------------------
# TranscriptReader.find_current_transcript
# ---------------------------------------------------------------------------

class TestFindCurrentTranscript:
    def test_picks_most_recent(self, transcript_dir):
        project_dir, claude_dir = transcript_dir
        older = claude_dir / "old-session.jsonl"
        newer = claude_dir / "new-session.jsonl"
        older.write_text("{}\n")
        time.sleep(0.05)
        newer.write_text("{}\n")

        reader = TranscriptReader(project_dir)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "home", lambda: claude_dir.parents[2])
            result = reader.find_current_transcript()
        assert result == newer

    def test_returns_none_when_no_jsonl(self, transcript_dir):
        project_dir, claude_dir = transcript_dir
        reader = TranscriptReader(project_dir)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "home", lambda: claude_dir.parents[2])
            result = reader.find_current_transcript()
        assert result is None


# ---------------------------------------------------------------------------
# TranscriptReader.read_session — basic tool call extraction
# ---------------------------------------------------------------------------

class TestReadSession:
    def test_basic_tool_call(self, tmp_path):
        """Extracts a simple OSPREY tool call from assistant+user entries."""
        transcript = tmp_path / "session.jsonl"
        entries = [
            _make_assistant_entry(_ts(0), [
                {
                    "id": "tu-1",
                    "name": "mcp__controls__channel_read",
                    "input": {"channels": ["SR:CURRENT"]},
                },
            ]),
            _make_user_entry(_ts(1), [
                {
                    "tool_use_id": "tu-1",
                    "content": [{"type": "text", "text": '{"status": "success", "value": 500.0}'}],
                    "is_error": False,
                },
            ]),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript)

        assert len(events) == 1
        ev = events[0]
        assert ev["type"] == "tool_call"
        assert ev["tool"] == "channel_read"
        assert ev["server"] == "controls"
        assert ev["full_tool_name"] == "mcp__controls__channel_read"
        assert ev["is_error"] is False
        assert ev["session_id"] == "test-session-123"
        assert ev["agent_id"] is None
        assert ev["arguments"] == {"channels": ["SR:CURRENT"]}
        assert "500.0" in ev["result_summary"]

    def test_filters_non_osprey_tools(self, tmp_path):
        """Non-OSPREY tools (e.g., Read, Bash) are not included."""
        transcript = tmp_path / "session.jsonl"
        entries = [
            _make_assistant_entry(_ts(0), [
                {
                    "id": "tu-1",
                    "name": "Read",
                    "input": {"path": "/some/file"},
                },
                {
                    "id": "tu-2",
                    "name": "mcp__workspace__session_log",
                    "input": {"last_n": 10},
                },
            ]),
            _make_user_entry(_ts(1), [
                {
                    "tool_use_id": "tu-1",
                    "content": "file contents",
                    "is_error": False,
                },
                {
                    "tool_use_id": "tu-2",
                    "content": '{"events": []}',
                    "is_error": False,
                },
            ]),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript)

        assert len(events) == 1
        assert events[0]["tool"] == "session_log"

    def test_error_detection_native(self, tmp_path):
        """Detects is_error from native tool_result flag."""
        transcript = tmp_path / "session.jsonl"
        entries = [
            _make_assistant_entry(_ts(0), [
                {
                    "id": "tu-1",
                    "name": "mcp__controls__channel_read",
                    "input": {"channels": ["BAD:PV"]},
                },
            ]),
            _make_user_entry(_ts(1), [
                {
                    "tool_use_id": "tu-1",
                    "content": "Connection refused",
                    "is_error": True,
                },
            ]),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript)

        assert len(events) == 1
        assert events[0]["is_error"] is True

    def test_error_detection_osprey_envelope(self, tmp_path):
        """Detects is_error from OSPREY error envelope in result."""
        transcript = tmp_path / "session.jsonl"
        error_payload = json.dumps({"error": True, "error_type": "validation_error", "error_message": "bad"})
        entries = [
            _make_assistant_entry(_ts(0), [
                {
                    "id": "tu-1",
                    "name": "mcp__workspace__memory_save",
                    "input": {"key": "test"},
                },
            ]),
            _make_user_entry(_ts(1), [
                {
                    "tool_use_id": "tu-1",
                    "content": error_payload,
                    "is_error": False,
                },
            ]),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript)

        assert len(events) == 1
        assert events[0]["is_error"] is True

    def test_result_truncation_success(self, tmp_path):
        """Success results are truncated to MAX_RESULT_LENGTH."""
        transcript = tmp_path / "session.jsonl"
        long_result = "x" * (MAX_RESULT_LENGTH + 100)
        entries = [
            _make_assistant_entry(_ts(0), [
                {
                    "id": "tu-1",
                    "name": "mcp__controls__channel_read",
                    "input": {"channels": ["TEST"]},
                },
            ]),
            _make_user_entry(_ts(1), [
                {
                    "tool_use_id": "tu-1",
                    "content": long_result,
                    "is_error": False,
                },
            ]),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript)

        assert len(events[0]["result_summary"]) == MAX_RESULT_LENGTH + 3  # +3 for "..."
        assert events[0]["result_summary"].endswith("...")

    def test_result_truncation_error(self, tmp_path):
        """Error results get the longer MAX_ERROR_RESULT_LENGTH limit."""
        transcript = tmp_path / "session.jsonl"
        error_json = json.dumps({"error": True, "error_type": "x", "error_message": "y" * 3000})
        entries = [
            _make_assistant_entry(_ts(0), [
                {
                    "id": "tu-1",
                    "name": "mcp__controls__channel_write",
                    "input": {},
                },
            ]),
            _make_user_entry(_ts(1), [
                {
                    "tool_use_id": "tu-1",
                    "content": error_json,
                    "is_error": False,
                },
            ]),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript)

        assert events[0]["is_error"] is True
        assert len(events[0]["result_summary"]) == MAX_ERROR_RESULT_LENGTH + 3

    def test_multiple_tool_uses_single_message(self, tmp_path):
        """Multiple tool_use blocks in one assistant message are all extracted."""
        transcript = tmp_path / "session.jsonl"
        entries = [
            _make_assistant_entry(_ts(0), [
                {
                    "id": "tu-1",
                    "name": "mcp__controls__channel_read",
                    "input": {"channels": ["PV1"]},
                },
                {
                    "id": "tu-2",
                    "name": "mcp__controls__archiver_read",
                    "input": {"channels": ["PV2"]},
                },
            ]),
            _make_user_entry(_ts(1), [
                {
                    "tool_use_id": "tu-1",
                    "content": "result1",
                    "is_error": False,
                },
                {
                    "tool_use_id": "tu-2",
                    "content": "result2",
                    "is_error": False,
                },
            ]),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript)

        assert len(events) == 2
        tools = {e["tool"] for e in events}
        assert tools == {"channel_read", "archiver_read"}

    def test_empty_transcript(self, tmp_path):
        """Empty transcript returns empty list."""
        transcript = tmp_path / "empty.jsonl"
        transcript.write_text("")

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript)

        assert events == []

    def test_missing_file(self, tmp_path):
        """Non-existent transcript returns empty list."""
        reader = TranscriptReader(tmp_path)
        events = reader.read_session(tmp_path / "nonexistent.jsonl")

        assert events == []

    def test_skips_non_relevant_entries(self, tmp_path):
        """Non-assistant/user entries (progress, file-history-snapshot) are skipped."""
        transcript = tmp_path / "session.jsonl"
        entries = [
            {"type": "progress", "timestamp": _ts(0), "data": "loading"},
            {"type": "file-history-snapshot", "timestamp": _ts(1), "files": []},
            _make_assistant_entry(_ts(2), [
                {
                    "id": "tu-1",
                    "name": "mcp__workspace__session_log",
                    "input": {},
                },
            ]),
            _make_user_entry(_ts(3), [
                {
                    "tool_use_id": "tu-1",
                    "content": "ok",
                    "is_error": False,
                },
            ]),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript)

        assert len(events) == 1
        assert events[0]["tool"] == "session_log"

    def test_standalone_tool_result_entries(self, tmp_path):
        """Tool results as standalone 'tool_result' entries (not inside 'user')."""
        transcript = tmp_path / "session.jsonl"
        entries = [
            _make_assistant_entry(_ts(0), [
                {
                    "id": "tu-1",
                    "name": "mcp__ariel__keyword_search",
                    "input": {"query": "beam loss"},
                },
            ]),
            _make_tool_result_entry(_ts(1), "tu-1", '{"results": []}'),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript)

        assert len(events) == 1
        assert events[0]["tool"] == "keyword_search"
        assert events[0]["server"] == "ariel"


# ---------------------------------------------------------------------------
# Agent start/stop from Task tool calls
# ---------------------------------------------------------------------------

class TestAgentEvents:
    def test_task_produces_agent_start_stop(self, tmp_path):
        """Task tool_use + result produces agent_start and agent_stop events."""
        transcript = tmp_path / "session.jsonl"
        entries = [
            _make_assistant_entry(_ts(0), [
                {
                    "id": "task-1",
                    "name": "Task",
                    "input": {
                        "subagent_type": "data-visualizer",
                        "description": "Create a plot",
                        "prompt": "Make a plot",
                    },
                },
            ]),
            _make_user_entry(_ts(5), [
                {
                    "tool_use_id": "task-1",
                    "content": json.dumps({
                        "agentId": "agent-abc",
                        "transcriptPath": "/t/agent-abc.jsonl",
                    }),
                    "is_error": False,
                },
            ]),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript, include_subagents=False)

        assert len(events) == 2
        start = [e for e in events if e["type"] == "agent_start"][0]
        stop = [e for e in events if e["type"] == "agent_stop"][0]
        assert start["agent_id"] == "agent-abc"
        assert start["agent_type"] == "data-visualizer"
        assert stop["agent_id"] == "agent-abc"
        assert stop["agent_transcript_path"] == "/t/agent-abc.jsonl"


# ---------------------------------------------------------------------------
# Subagent transcript reading
# ---------------------------------------------------------------------------

class TestSubagentReading:
    def test_subagent_tool_calls_have_agent_id(self, tmp_path):
        """Tool calls from subagent transcripts have agent_id set."""
        # Main transcript
        main_transcript = tmp_path / "main-session.jsonl"
        main_entries = [
            _make_assistant_entry(_ts(0), [
                {
                    "id": "tu-main",
                    "name": "mcp__controls__channel_read",
                    "input": {"channels": ["MAIN:PV"]},
                },
            ]),
            _make_user_entry(_ts(1), [
                {
                    "tool_use_id": "tu-main",
                    "content": "main result",
                    "is_error": False,
                },
            ]),
        ]
        _write_transcript(main_transcript, main_entries)

        # Subagent transcript
        subagent_dir = tmp_path / "main-session" / "subagents"
        subagent_dir.mkdir(parents=True)
        sub_transcript = subagent_dir / "agent-abc.jsonl"
        sub_entries = [
            _make_assistant_entry(_ts(2), [
                {
                    "id": "tu-sub",
                    "name": "mcp__workspace__create_static_plot",
                    "input": {"title": "Plot"},
                },
            ]),
            _make_user_entry(_ts(3), [
                {
                    "tool_use_id": "tu-sub",
                    "content": "plot ok",
                    "is_error": False,
                },
            ]),
        ]
        _write_transcript(sub_transcript, sub_entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(main_transcript)

        assert len(events) == 2
        main_event = [e for e in events if e["tool"] == "channel_read"][0]
        sub_event = [e for e in events if e["tool"] == "create_static_plot"][0]
        assert main_event["agent_id"] is None
        assert sub_event["agent_id"] == "agent-abc"


# ---------------------------------------------------------------------------
# read_current_session
# ---------------------------------------------------------------------------

class TestReadCurrentSession:
    def test_returns_empty_when_no_transcript(self, tmp_path):
        """Returns empty list when no transcript directory exists."""
        reader = TranscriptReader(tmp_path / "nonexistent")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "home", lambda: tmp_path)
            events = reader.read_current_session()
        assert events == []


# ---------------------------------------------------------------------------
# All prefix servers covered
# ---------------------------------------------------------------------------

class TestAllPrefixes:
    @pytest.mark.parametrize("prefix,tool_suffix,expected_server", [
        ("mcp__controls__", "channel_read", "controls"),
        ("mcp__python__", "execute", "python"),
        ("mcp__workspace__", "session_log", "workspace"),
        ("mcp__ariel__", "keyword_search", "ariel"),
        ("mcp__accelpapers__", "papers_search", "accelpapers"),
        ("mcp__matlab__", "mml_search", "matlab"),
        ("mcp__channel-finder__", "channel_find", "channel-finder"),
        ("mcp__confluence__", "confluence_search", "confluence"),
    ])
    def test_prefix_recognized(self, tmp_path, prefix, tool_suffix, expected_server):
        transcript = tmp_path / "session.jsonl"
        full_name = f"{prefix}{tool_suffix}"
        entries = [
            _make_assistant_entry(_ts(0), [
                {"id": "tu-1", "name": full_name, "input": {}},
            ]),
            _make_user_entry(_ts(1), [
                {"tool_use_id": "tu-1", "content": "ok", "is_error": False},
            ]),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        events = reader.read_session(transcript)

        assert len(events) == 1
        assert events[0]["server"] == expected_server
        assert events[0]["tool"] == tool_suffix


# ---------------------------------------------------------------------------
# read_chat_history
# ---------------------------------------------------------------------------

def _make_user_text_entry(timestamp: str, text: str) -> dict:
    """Build a 'user' transcript entry with a plain text message."""
    return {
        "type": "user",
        "timestamp": timestamp,
        "sessionId": "test-session-123",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    }


def _make_assistant_text_entry(timestamp: str, text: str,
                               tool_uses: list[dict] | None = None) -> dict:
    """Build an 'assistant' entry with text and optional tool_use blocks."""
    content: list[dict] = [{"type": "text", "text": text}]
    if tool_uses:
        content.extend({"type": "tool_use", **tu} for tu in tool_uses)
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "sessionId": "test-session-123",
        "message": {
            "role": "assistant",
            "content": content,
        },
    }


class TestReadChatHistory:
    def test_extracts_user_messages(self, tmp_path):
        """User text messages are captured as conversation turns."""
        transcript = tmp_path / "session.jsonl"
        entries = [
            _make_user_text_entry(_ts(0), "What is the beam current?"),
            _make_assistant_text_entry(_ts(1), "The beam current is 500 mA."),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        history = reader.read_chat_history(transcript)

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What is the beam current?"
        assert history[0]["timestamp"] == _ts(0)
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "The beam current is 500 mA."

    def test_extracts_assistant_text_only(self, tmp_path):
        """Assistant entries with tool_use blocks only capture text, not tool calls."""
        transcript = tmp_path / "session.jsonl"
        entries = [
            _make_assistant_text_entry(
                _ts(0),
                "Let me read the current value.",
                tool_uses=[{
                    "id": "tu-1",
                    "name": "mcp__controls__channel_read",
                    "input": {"channels": ["SR:CURRENT"]},
                }],
            ),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        history = reader.read_chat_history(transcript)

        assert len(history) == 1
        assert history[0]["role"] == "assistant"
        assert history[0]["content"] == "Let me read the current value."
        assert "channel_read" not in history[0]["content"]

    def test_truncates_long_messages(self, tmp_path):
        """Individual messages exceeding MAX_CHAT_MESSAGE_LENGTH are truncated."""
        transcript = tmp_path / "session.jsonl"
        long_text = "x" * (MAX_CHAT_MESSAGE_LENGTH + 500)
        entries = [_make_user_text_entry(_ts(0), long_text)]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        history = reader.read_chat_history(transcript)

        assert len(history) == 1
        assert len(history[0]["content"]) == MAX_CHAT_MESSAGE_LENGTH + 3
        assert history[0]["content"].endswith("...")

    def test_skips_tool_result_only_entries(self, tmp_path):
        """User entries containing only tool_result blocks produce no chat turn."""
        transcript = tmp_path / "session.jsonl"
        entries = [
            _make_user_entry(_ts(0), [
                {
                    "tool_use_id": "tu-1",
                    "content": "result value",
                    "is_error": False,
                },
            ]),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        history = reader.read_chat_history(transcript)

        assert len(history) == 0

    def test_empty_transcript(self, tmp_path):
        """Empty transcript returns empty list."""
        transcript = tmp_path / "empty.jsonl"
        transcript.write_text("")

        reader = TranscriptReader(tmp_path)
        history = reader.read_chat_history(transcript)

        assert history == []

    def test_missing_file(self, tmp_path):
        """Non-existent file returns empty list."""
        reader = TranscriptReader(tmp_path)
        history = reader.read_chat_history(tmp_path / "nonexistent.jsonl")

        assert history == []

    def test_chronological_order(self, tmp_path):
        """Turns are returned in chronological (insertion) order."""
        transcript = tmp_path / "session.jsonl"
        entries = [
            _make_user_text_entry(_ts(0), "First message"),
            _make_assistant_text_entry(_ts(1), "First reply"),
            _make_user_text_entry(_ts(2), "Second message"),
            _make_assistant_text_entry(_ts(3), "Second reply"),
        ]
        _write_transcript(transcript, entries)

        reader = TranscriptReader(tmp_path)
        history = reader.read_chat_history(transcript)

        assert len(history) == 4
        assert [h["role"] for h in history] == ["user", "assistant", "user", "assistant"]
        assert history[0]["content"] == "First message"
        assert history[3]["content"] == "Second reply"
