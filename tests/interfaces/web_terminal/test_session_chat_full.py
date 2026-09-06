"""Tests for the replay reads of ``GET /api/session-chat``.

The endpoint serves two callers from one route: the diagnostics view, which
wants a capped preview of the conversation, and a view flip, which wants the
whole thing. These cover the second caller — the ``full`` flag and the
session-key-to-transcript mapping the route reads through.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.agent_runner.project_paths import CLAUDE_CONFIG_DIR_ENV, claude_project_dir
from osprey.interfaces.web_terminal.app import create_app
from osprey.mcp_server.workspace.transcript_reader import MAX_CHAT_MESSAGE_LENGTH

SESSION_KEY = "11111111-2222-3333-4444-555555555555"
CLEARED_TRANSCRIPT = "99999999-8888-7777-6666-555555555555"


def _entry(role: str, text: str) -> dict:
    """A transcript entry holding one plain-text message."""
    return {
        "type": role,
        "timestamp": datetime(2026, 2, 19, 12, 0, 0, tzinfo=UTC).isoformat(),
        "sessionId": SESSION_KEY,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }


def _write(transcript_dir: Path, transcript_id: str, entries: list[dict]) -> None:
    """Write *entries* as the JSONL transcript named *transcript_id*."""
    path = transcript_dir / f"{transcript_id}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


@pytest.fixture
def transcripts(tmp_path, monkeypatch):
    """A project whose Claude Code transcript directory lives under ``tmp_path``."""
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(tmp_path / "claude-config"))
    project = tmp_path / "project"
    project.mkdir()
    transcript_dir = claude_project_dir(project)
    transcript_dir.mkdir(parents=True)
    return project, transcript_dir


@pytest.fixture
def client(tmp_path, transcripts):
    """A Web Terminal client reading those transcripts, mapping nothing yet."""
    project, _ = transcripts
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(workspace)},
    ):
        app = create_app(shell_command="echo", project_dir=project)
        with TestClient(app) as c:
            app.state.transcript_map = {}
            app.state.transcript_map_provisional = False
            yield c


class TestSessionChatFull:
    def test_maps_key_to_its_current_transcript(self, client, transcripts):
        """A key that has cleared reads the transcript it moved to, not its own name."""
        _, transcript_dir = transcripts
        _write(transcript_dir, SESSION_KEY, [_entry("user", "before the clear")])
        _write(transcript_dir, CLEARED_TRANSCRIPT, [_entry("user", "after the clear")])
        client.app.state.transcript_map[SESSION_KEY] = CLEARED_TRANSCRIPT

        resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}&full=1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["turns"][0]["content"] == "after the clear"

    def test_unmapped_key_reads_its_own_transcript(self, client, transcripts):
        """A key that has never cleared needs no entry — it names its own transcript."""
        _, transcript_dir = transcripts
        _write(transcript_dir, SESSION_KEY, [_entry("user", "still the first conversation")])

        resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}&full=1")

        assert resp.json()["turns"][0]["content"] == "still the first conversation"

    def test_full_leaves_long_messages_whole(self, client, transcripts):
        """``full=1`` returns the message at its real length."""
        _, transcript_dir = transcripts
        long_text = "x" * 5000
        _write(transcript_dir, SESSION_KEY, [_entry("assistant", long_text)])

        resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}&full=1")

        assert resp.json()["turns"][0]["content"] == long_text

    def test_default_keeps_the_cap(self, client, transcripts):
        """Without the flag the per-message cap stands, as the diagnostics view reads it."""
        _, transcript_dir = transcripts
        _write(transcript_dir, SESSION_KEY, [_entry("assistant", "x" * 5000)])

        resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}")

        assert resp.json()["turns"][0]["content"] == "x" * MAX_CHAT_MESSAGE_LENGTH + "..."

    def test_unknown_key_returns_an_empty_list(self, client):
        """A key with no transcript anywhere collapses to an empty conversation."""
        resp = client.get("/api/session-chat?session_id=no-such-key&full=1")

        assert resp.status_code == 200
        assert resp.json() == {"turns": [], "count": 0}

    def test_full_accepts_the_other_true_spellings(self, client, transcripts):
        """``true`` reads the same as ``1``; an unrecognized value keeps the cap."""
        _, transcript_dir = transcripts
        _write(transcript_dir, SESSION_KEY, [_entry("assistant", "y" * 5000)])

        whole = client.get(f"/api/session-chat?session_id={SESSION_KEY}&full=true")
        capped = client.get(f"/api/session-chat?session_id={SESSION_KEY}&full=maybe")

        assert len(whole.json()["turns"][0]["content"]) == 5000
        assert len(capped.json()["turns"][0]["content"]) < 5000
