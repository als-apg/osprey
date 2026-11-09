"""A chat upload becomes an artifact when the message carrying it is sent.

Nothing is saved when the composer uploads a file: the operator may still take it
back. Once ``POST /api/chat`` accepts a prompt naming the file, :func:`mark_sent`
copies it into the artifact store at the hub's shared data root, tagged with the
chat it was attached in, and returns the id saved per stored name. The uploads
folder itself (its manifest and its sweep) is unchanged, and a store that fails
logs and never fails the send.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes import chat as chat_route
from osprey.interfaces.web_terminal.routes import chat_attachments
from osprey.interfaces.web_terminal.routes.chat_attachments import (
    BATCH_HEADER,
    NAME_HEADER,
    SENT_MANIFEST,
    router,
)
from osprey.stores.artifact_forms import form_of
from osprey.stores.artifact_store import ArtifactStore

from .conftest import bare_route_app

CHAT = "aaaaaaaa-1111-2222-3333-444444444444"
BATCH = "batch-0001-aaaa"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 64
CSV = b"timestamp,value\n2026-10-22T12:00,3.1\n"


@pytest.fixture
def agent_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "agent_data"
    root.mkdir()
    monkeypatch.setattr(chat_attachments, "resolve_agent_data_root", lambda app=None: str(root))
    return root


@pytest.fixture
def shared_root(tmp_path) -> Path:
    root = tmp_path / "shared"
    root.mkdir()
    return root


@pytest.fixture
def client(agent_root, shared_root):  # noqa: ARG001 - agent_root pins the uploads root
    app = bare_route_app(router, panel_status_root=shared_root)
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, name: str, body: bytes) -> dict:
    response = client.post(
        f"/api/chat/{CHAT}/attachments",
        content=body,
        headers={NAME_HEADER: quote(name, safe=""), BATCH_HEADER: BATCH},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _prompt(*stored: dict) -> str:
    lines = "".join(f"\n- {s['path']} ({s['type']}, {s['size']} B)" for s in stored)
    return f"What are these?\n\nAttached files (read them with the Read tool):{lines}"


def _store(shared_root: Path) -> ArtifactStore:
    return ArtifactStore(workspace_root=shared_root, auto_launch=False)


def _uploads(shared_root: Path) -> list:
    return [e for e in _store(shared_root).list_entries() if e.origin == "upload"]


def _stored_name(stored: dict) -> str:
    return Path(stored["path"]).name


def test_nothing_is_saved_at_upload(client, shared_root):
    _upload(client, "beam.png", PNG)
    assert _uploads(shared_root) == []


def test_a_sent_upload_is_an_artifact_of_its_chat_at_the_shared_root(client, shared_root):
    stored = _upload(client, "beam.png", PNG)

    ids = chat_attachments.mark_sent(client.app, CHAT, _prompt(stored))

    assert list(ids) == [_stored_name(stored)]
    store = _store(shared_root)
    entry = store.get_entry(ids[_stored_name(stored)])
    assert entry is not None
    assert entry.origin == "upload"
    assert entry.tool_source == "upload"
    assert entry.session_id == CHAT
    assert entry.title == "beam.png"
    assert entry.exec_id == ""
    assert entry.mime_type == "image/png"
    assert isinstance(entry.metadata.get("sent_at"), str) and entry.metadata["sent_at"]
    path = store.get_file_path(entry.id)
    assert path is not None
    assert path.resolve().is_relative_to(shared_root.resolve())
    assert path.read_bytes() == PNG
    # The uploads folder keeps its manifest as before.
    manifest = Path(stored["path"]).parent / SENT_MANIFEST
    assert _stored_name(stored) in manifest.read_text(encoding="utf-8")


def test_an_image_is_an_image_and_a_csv_is_data(client, shared_root):
    image = _upload(client, "beam.png", PNG)
    table = _upload(client, "orbit.csv", CSV)

    ids = chat_attachments.mark_sent(client.app, CHAT, _prompt(image, table))

    store = _store(shared_root)
    assert form_of(store.get_entry(ids[_stored_name(image)])) == "images"
    csv_entry = store.get_entry(ids[_stored_name(table)])
    assert csv_entry.mime_type == "text/csv"
    assert form_of(csv_entry) == "data"


def test_a_later_message_naming_the_same_file_saves_nothing(client, shared_root):
    first = _upload(client, "beam.png", PNG)
    chat_attachments.mark_sent(client.app, CHAT, _prompt(first))
    second = _upload(client, "orbit.csv", CSV)

    assert chat_attachments.mark_sent(client.app, CHAT, _prompt(first)) == {}
    again = chat_attachments.mark_sent(client.app, CHAT, _prompt(first, second))

    assert list(again) == [_stored_name(second)]
    assert sorted(e.title for e in _uploads(shared_root)) == ["beam.png", "orbit.csv"]


def test_an_upload_taken_back_before_sending_leaves_no_artifact(client, shared_root):
    stored = _upload(client, "beam.png", PNG)
    response = client.delete(f"/api/chat/{CHAT}/attachments/{_stored_name(stored)}")
    assert response.status_code == 204

    assert chat_attachments.mark_sent(client.app, CHAT, _prompt(stored)) == {}
    assert _uploads(shared_root) == []


def test_a_prompt_naming_no_file_saves_nothing(client, shared_root):
    assert chat_attachments.mark_sent(client.app, CHAT, "just text") == {}
    assert chat_attachments.mark_sent(client.app, "not-a-key", "just text") == {}
    assert _uploads(shared_root) == []


def test_the_copy_survives_the_sweep_and_the_uploads_folder(client, agent_root, shared_root):
    stored = _upload(client, "beam.png", PNG)
    ids = chat_attachments.mark_sent(client.app, CHAT, _prompt(stored))

    # Sweep long after any retention, then lose the upload itself.
    chat_attachments.sweep_uploads(
        agent_root / "uploads",
        chat_attachments._ledger(client.app),
        now=10**12,
    )
    Path(stored["path"]).unlink()

    path = _store(shared_root).get_file_path(ids[_stored_name(stored)])
    assert path is not None and path.read_bytes() == PNG


def test_a_store_failure_logs_and_never_fails_the_send(client, monkeypatch, caplog):
    stored = _upload(client, "beam.png", PNG)

    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ArtifactStore, "save_file", broken)
    with caplog.at_level(logging.WARNING, logger=chat_attachments.__name__):
        ids = chat_attachments.mark_sent(client.app, CHAT, _prompt(stored))

    assert ids == {}
    assert any("artifact" in r.getMessage().lower() for r in caplog.records)
    # The manifest is still written: the sweep keeps the sent file.
    manifest = Path(stored["path"]).parent / SENT_MANIFEST
    assert _stored_name(stored) in manifest.read_text(encoding="utf-8")


def test_an_unopenable_store_logs_and_never_fails_the_send(client, monkeypatch, caplog):
    stored = _upload(client, "beam.png", PNG)

    def broken_store(*_args, **_kwargs):
        raise RuntimeError("index unreadable")

    monkeypatch.setattr(chat_attachments, "ArtifactStore", broken_store)
    with caplog.at_level(logging.WARNING, logger=chat_attachments.__name__):
        assert chat_attachments.mark_sent(client.app, CHAT, _prompt(stored)) == {}
    assert caplog.records


def test_the_chat_route_runs_mark_sent_off_the_event_loop(tmp_path, monkeypatch):
    from tests.interfaces.web_terminal._chat_server import live_chat_server

    seen: list[tuple[str, bool]] = []
    real = chat_route.mark_sent

    def spy(app, chat_id, prompt):
        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        seen.append((threading.current_thread().name, on_loop))
        return real(app, chat_id, prompt)

    monkeypatch.setattr(chat_route, "mark_sent", spy)
    key = str(uuid.uuid4())
    with live_chat_server(tmp_path) as (base_url, _app):
        response = httpx.post(
            f"{base_url}/api/chat",
            params={"stream": "false"},
            json={"prompt": "hello", "chat_id": key},
            timeout=30,
        )
    assert response.status_code == 200, response.text
    assert len(seen) == 1
    assert seen[0][1] is False, "mark_sent ran on the event loop"
