"""The web terminal keeps no feedback store of its own.

A submission is one ``feedback`` entry in the agent record, written through the
record writer into the terminal's spool. These tests pin that the old file
store is gone — no module, no ``app.state`` wiring, no file written anywhere
but the spool — and that the entry the record holds has the shape ``osprey
feedback`` reads: the header, the text, the context and scrollback the user
chose, and the keys of the session's steps in place of excerpts of them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.agent_record.client import ClientAnswer
from osprey.agent_record.contract import KIND_FEEDBACK
from tests.agent_record.fakes import FakeRecordClient

SESSION_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"

#: The fields every ``feedback`` entry carries, whatever the channel.
HEADER_FIELDS = {
    "id",
    "identity",
    "created_at",
    "channel",
    "version",
    "app_name",
    "session_id",
    "include_metadata",
    "include_context",
    "excerpt",
    "context_status",
    "text_truncated",
    "context_truncated",
}
CONTENT_FIELDS = {"text", "context", "scrollback", "refs"}


def _node(key: str, kind: str, body: dict) -> dict:
    return {
        "key": key,
        "kind": kind,
        "ts": "2026-08-20T18:00:00+00:00",
        "body": body,
        "decision": None,
        "result": None,
        "subagent": None,
    }


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spool_for) -> TestClient:
    from osprey.interfaces.web_terminal.routes import feedback as module

    spool_for()
    nodes = [
        _node("sp:alice:p1", "prompt", {"text": "why did BPM 3 drift?"}),
        _node("sp:alice:a1", "answer", {"text": "BPM 3 alone drifted"}),
    ]
    record = FakeRecordClient({"timeline": ClientAnswer(data=nodes, complete=True)})
    monkeypatch.setattr(module, "record_client", lambda: record)
    store = MagicMock()
    store.list_entries.return_value = []
    monkeypatch.setattr("osprey.stores.artifact_store.ArtifactStore", lambda *a, **k: store)

    app = FastAPI()
    app.include_router(module.router)
    app.state.project_cwd = str(tmp_path / "project")
    app.state.workspace_dir = tmp_path / "workspace"
    app.state.terminal_user = "alice"
    app.state.transcript_map = {}
    app.state.transcript_map_provisional = False
    with TestClient(app) as test_client:
        yield test_client


def test_the_feedback_store_module_is_gone() -> None:
    assert importlib.util.find_spec("osprey.interfaces.web_terminal.feedback_store") is None


def test_the_route_module_reads_no_store_configuration() -> None:
    from osprey.interfaces.web_terminal.routes import feedback as module

    for name in ("prune_store", "write_record", "_feedback_dir", "_max_store_bytes"):
        assert not hasattr(module, name), name


def test_the_app_publishes_no_feedback_store() -> None:
    source = Path(importlib.util.find_spec("osprey.interfaces.web_terminal.app").origin)
    text = source.read_text(encoding="utf-8")
    for wiring in ("feedback_dir", "feedback_rel", "feedback_max_store_bytes", "max_store_bytes"):
        assert wiring not in text, wiring


def test_a_submission_is_one_feedback_entry_and_no_file(
    client: TestClient, tmp_path: Path, recorded_entries
) -> None:
    response = client.post(
        "/api/feedback",
        json={
            "text": "the orbit plot froze",
            "channel": "local",
            "include_context": True,
            "session_id": SESSION_ID,
            "scrollback": "$ osprey web\n",
        },
    )

    assert response.status_code == 200
    assert not list(tmp_path.rglob("fb-*.json"))
    assert not list(tmp_path.rglob("ctx-*.json"))
    assert not (tmp_path / "workspace" / "feedback").exists()

    (entry,) = recorded_entries(KIND_FEEDBACK)
    assert entry.identity == "alice"
    assert entry.session_id == SESSION_ID
    body = dict(entry.body)
    assert HEADER_FIELDS | CONTENT_FIELDS <= set(body)
    assert body["id"] == response.json()["id"]
    assert body["text"] == "the orbit plot froze"
    assert body["scrollback"] == "$ osprey web\n"
    assert body["refs"] == ["sp:alice:p1", "sp:alice:a1"]
    # Referenced, never copied: the steps' words are not in the entry.
    assert "BPM 3" not in body["context"]
    assert "## deployment" in body["context"]


def test_an_unticked_context_box_records_no_scrollback_and_no_refs(
    client: TestClient, recorded_entries
) -> None:
    response = client.post(
        "/api/feedback",
        json={
            "text": "just a note",
            "channel": "email",
            "include_context": False,
            "session_id": SESSION_ID,
            "scrollback": "secret scrollback",
        },
    )

    assert response.status_code == 200
    (entry,) = recorded_entries(KIND_FEEDBACK)
    body = dict(entry.body)
    assert body["scrollback"] == ""
    assert body["refs"] == []
    assert body["session_id"] is None
    assert entry.session_id in (None, "")
