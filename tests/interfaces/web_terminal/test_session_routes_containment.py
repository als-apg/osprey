"""The session diagnostics routes read a traversal-shaped id as a missing transcript."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.agent_runner.project_paths import CLAUDE_CONFIG_DIR_ENV, claude_project_dir
from osprey.interfaces.web_terminal.app import create_app


def _ts(minute: int) -> str:
    return datetime(2026, 2, 19, 12, minute, 0, tzinfo=UTC).isoformat()


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


def _plant_outside(tmp_path: Path, escaping: Path) -> str:
    """Plant a conversation outside *escaping* and return the id that walks to it."""
    planted = tmp_path / "outside" / "planted.jsonl"
    planted.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "type": "user",
            "timestamp": _ts(0),
            "sessionId": "planted",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "outside the directory"}],
            },
        },
        {
            "type": "assistant",
            "timestamp": _ts(1),
            "sessionId": "planted",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu-out",
                        "name": "mcp__controls__channel_read",
                        "input": {},
                    }
                ],
            },
        },
        {
            "type": "user",
            "timestamp": _ts(2),
            "sessionId": "planted",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-out",
                        "content": "read",
                        "is_error": False,
                    }
                ],
            },
        },
    ]
    planted.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return "../" * len(escaping.relative_to(tmp_path).parts) + "outside/planted"


@pytest.mark.parametrize("route", ["session-chat", "session-log", "session-agent-timeline"])
def test_a_traversal_id_reads_as_a_missing_transcript(client, transcripts, tmp_path, route):
    _, claude_dir = transcripts
    if route == "session-agent-timeline":
        (claude_dir / "parent.jsonl").write_text("{}\n")
        subagent_dir = claude_dir / "parent" / "subagents"
        subagent_dir.mkdir(parents=True)
        agent_id = _plant_outside(tmp_path, subagent_dir)
        resp = client.get(
            "/api/session-agent-timeline",
            params={"agent_id": agent_id, "session_id": "parent"},
        )
        expected = {"agent_id": agent_id, "timeline": [], "count": 0}
    else:
        session_id = _plant_outside(tmp_path, claude_dir)
        resp = client.get(f"/api/{route}", params={"session_id": session_id})
        expected = (
            {"turns": [], "count": 0}
            if route == "session-chat"
            else {"events": [], "total_events": 0, "showing": 0}
        )

    assert resp.status_code == 200
    assert resp.json() == expected
