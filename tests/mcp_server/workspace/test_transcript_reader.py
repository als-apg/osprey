"""Tests for TranscriptReader: the local lookups of Claude Code transcripts.

Uses synthetic JSONL fixtures that mirror actual Claude Code transcript format.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from osprey.agent_runner.project_paths import (
    CLAUDE_CONFIG_DIR_ENV,
    claude_project_dir,
    encode_claude_project_path,
)
from osprey.mcp_server.workspace.transcript_reader import (
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
            "content": [{"type": "tool_use", **tu} for tu in tool_uses],
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
            "content": [{"type": "tool_result", **tr} for tr in tool_results],
        },
    }


def _write_transcript(path: Path, entries: list[dict]) -> None:
    """Write entries as JSONL to a file."""
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


@pytest.fixture
def transcript_dir(tmp_path):
    """Create a project dir with Claude Code transcript directory structure.

    Uses an underscored project name so the fixture exercises the full
    encoding rule (non-alphanumerics → ``-``), not just ``/`` → ``-``.
    """
    project_dir = tmp_path / "my_project_dir"
    project_dir.mkdir()
    encoded = encode_claude_project_path(project_dir)
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

    def test_honours_claude_config_dir(self, tmp_path, monkeypatch):
        """Transcripts live under ``$CLAUDE_CONFIG_DIR/projects``, not ``~/.claude``.

        Inside a per-user web-terminal container both variables point at the
        same mounted volume, so the ``~/.claude`` spelling names a directory
        that does not exist and every session reads as "no transcript".
        """
        project_dir = tmp_path / "app" / "build"
        project_dir.mkdir(parents=True)
        config_dir = tmp_path / "data" / "claude-config"
        claude_dir = config_dir / "projects" / encode_claude_project_path(project_dir)
        claude_dir.mkdir(parents=True)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: config_dir))

        assert TranscriptReader(project_dir).find_transcript_dir() == claude_dir

    def test_preserves_leading_dash_for_absolute_paths(self, tmp_path):
        """Encoding of absolute paths must keep the leading dash.

        Claude Code directories always start with '-' because absolute
        paths start with '/' which becomes '-'. Stripping it causes
        find_transcript_dir() to never find the directory.
        """
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()
        encoded = encode_claude_project_path(project_dir)
        assert encoded.startswith("-"), "Absolute path encoding must start with '-'"
        claude_dir = tmp_path / ".claude" / "projects" / encoded
        claude_dir.mkdir(parents=True)

        reader = TranscriptReader(project_dir)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "home", lambda: tmp_path)
            result = reader.find_transcript_dir()
        assert result is not None
        assert result == claude_dir

    def test_underscores_in_path_are_normalized(self, tmp_path):
        """Underscores in the cwd must be normalized to dashes.

        Regression: pytest tmpdir paths and macOS tmp dirs commonly contain
        underscores (e.g. ``8558q21d28scmm_6ssh_wxwc0000gn``,
        ``test_channel_read_archiver_plot``). Claude Code writes its
        transcript under a dash-normalized name; if the reader only
        substitutes ``/``, the lookup silently misses the directory and
        every downstream caller sees an empty transcript.
        """
        project_dir = tmp_path / "my_test_project"
        project_dir.mkdir()
        encoded = encode_claude_project_path(project_dir)
        assert "_" not in encoded, f"Encoded name must not contain '_': {encoded!r}"
        claude_dir = tmp_path / ".claude" / "projects" / encoded
        claude_dir.mkdir(parents=True)

        reader = TranscriptReader(project_dir)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "home", lambda: tmp_path)
            result = reader.find_transcript_dir()
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
        newer.write_text("{}\n")
        # Ten seconds apart, so the order does not depend on the filesystem's
        # timestamp resolution.
        stamp = newer.stat().st_mtime_ns - 10_000_000_000
        os.utime(older, ns=(stamp, stamp))

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


class TestFindTranscriptById:
    def test_finds_existing_session(self, transcript_dir):
        project_dir, claude_dir = transcript_dir
        session_file = claude_dir / "abc-def-123.jsonl"
        session_file.write_text("{}\n")

        reader = TranscriptReader(project_dir)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "home", lambda: claude_dir.parents[2])
            result = reader.find_transcript_by_id("abc-def-123")
        assert result == session_file

    def test_returns_none_for_missing(self, transcript_dir):
        project_dir, claude_dir = transcript_dir

        reader = TranscriptReader(project_dir)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "home", lambda: claude_dir.parents[2])
            result = reader.find_transcript_by_id("nonexistent-session")
        assert result is None

    def test_returns_none_when_no_transcript_dir(self, tmp_path):
        reader = TranscriptReader(tmp_path / "nonexistent")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "home", lambda: tmp_path)
            result = reader.find_transcript_by_id("any-id")
        assert result is None


def _plant_outside(tmp_path: Path) -> Path:
    """Write a transcript outside the transcript directory that every read would find."""
    planted = tmp_path / "outside" / "planted.jsonl"
    planted.parent.mkdir(parents=True, exist_ok=True)
    _write_transcript(
        planted,
        [
            _make_user_text_entry(_ts(0), "outside the directory"),
            _make_assistant_entry(
                _ts(1),
                [{"id": "tu-out", "name": "mcp__controls__channel_read", "input": {}}],
            ),
            _make_user_entry(
                _ts(2),
                [{"tool_use_id": "tu-out", "content": "read", "is_error": False}],
            ),
        ],
    )
    return planted


class TestTranscriptIdContainment:
    """An id names a file inside the transcript directory; anything else reads as a missing transcript."""

    @pytest.mark.parametrize("shape", ["relative", "absolute"], ids=["relative", "absolute"])
    def test_session_id_outside_the_directory_reads_nothing(
        self, transcript_dir, tmp_path, monkeypatch, shape
    ):
        project_dir, claude_dir = transcript_dir
        planted = _plant_outside(tmp_path)
        if shape == "relative":
            session_id = "../" * len(claude_dir.relative_to(tmp_path).parts) + "outside/planted"
        else:
            session_id = str(planted.with_suffix(""))
        assert (claude_dir / f"{session_id}.jsonl").is_file()

        monkeypatch.setattr(Path, "home", lambda: claude_dir.parents[2])
        reader = TranscriptReader(project_dir)
        assert reader.find_transcript_by_id(session_id) is None

    @pytest.mark.parametrize(
        "session_id",
        ["", "nested/inner", "nul\x00byte"],
        ids=["empty", "separator", "nul"],
    )
    def test_an_id_that_is_not_a_plain_file_name_reads_nothing(
        self, transcript_dir, monkeypatch, session_id
    ):
        project_dir, claude_dir = transcript_dir
        _write_transcript(claude_dir / ".jsonl", [_make_user_text_entry(_ts(0), "hidden")])
        (claude_dir / "nested").mkdir()
        _write_transcript(
            claude_dir / "nested" / "inner.jsonl", [_make_user_text_entry(_ts(0), "nested")]
        )

        monkeypatch.setattr(Path, "home", lambda: claude_dir.parents[2])
        reader = TranscriptReader(project_dir)
        assert reader.find_transcript_by_id(session_id) is None

    def test_a_transcript_symlinked_out_of_the_directory_reads_nothing(
        self, transcript_dir, tmp_path, monkeypatch
    ):
        project_dir, claude_dir = transcript_dir
        planted = _plant_outside(tmp_path)
        (claude_dir / "linked.jsonl").symlink_to(planted)

        monkeypatch.setattr(Path, "home", lambda: claude_dir.parents[2])
        reader = TranscriptReader(project_dir)
        assert reader.find_transcript_by_id("linked") is None

    def test_a_symlinked_config_root_still_reads(self, transcript_dir, tmp_path, monkeypatch):
        project_dir, _ = transcript_dir
        real_config = tmp_path / "real-config"
        monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(real_config))
        claude_project_dir(project_dir).mkdir(parents=True)
        linked_config = tmp_path / "linked-config"
        linked_config.symlink_to(real_config, target_is_directory=True)
        monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(linked_config))
        _write_transcript(
            claude_project_dir(project_dir) / "abc-123.jsonl",
            [_make_user_text_entry(_ts(0), "through the link")],
        )

        reader = TranscriptReader(project_dir)
        assert reader.find_transcript_by_id("abc-123") is not None
