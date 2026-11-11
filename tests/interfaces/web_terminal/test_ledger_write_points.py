"""The hub's session-ledger write points: agent-turn reports, chat turn ends and spawns.

Each write point appends a row once per actual move and sends exactly one
``session_transcript {session_key}`` frame for it; a repeat appends nothing and
sends nothing. The chat's turn end also stamps the turn's uploads with the
transcript the turn ended on, so an upload sent with ``/clear`` belongs to the
new conversation.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.audit.posture import OSPREY_AGENT_DATA_ROOT
from osprey.interfaces.web_terminal import session_handoff, session_meta
from osprey.interfaces.web_terminal.routes import chat as chat_route
from osprey.interfaces.web_terminal.routes.agent_turn import router as agent_turn_router
from osprey.stores import session_ledger
from osprey.stores.artifact_store import ArtifactStore, owner_of
from osprey_connectors import control_context, posture_store
from tests.interfaces.web_terminal._handoff_harness import (
    KEY as HARNESS_KEY,
)
from tests.interfaces.web_terminal._handoff_harness import (
    acquire,
    chat_spawner,
    make_app,
    pty_spawner,
)

KEY = "aaaaaaaa-1111-2222-3333-444444444444"
T1 = "cccccccc-1111-2222-3333-444444444444"
T2 = "dddddddd-1111-2222-3333-444444444444"


class RecordingBroadcaster:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    def broadcast(self, data: dict, *, reliable: bool = False) -> None:  # noqa: ARG002
        self.frames.append(data)

    def transcript_frames(self) -> list[dict]:
        return [f for f in self.frames if f.get("type") == "session_transcript"]


@pytest.fixture
def shared_root(tmp_path, monkeypatch):
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv(OSPREY_AGENT_DATA_ROOT, str(root))
    control_context.invalidate_cache()
    yield root
    control_context.invalidate_cache()


def _rows() -> list[tuple[str, str]]:
    ledger = session_ledger.load(session_meta.ledger_state_dir())
    return [(r.session_key, r.transcript_id) for r in ledger.rows]


# ---- agent-turn ------------------------------------------------------------------


@pytest.mark.usefixtures("shared_root")
class TestAgentTurnWritePoint:
    def _client(self):
        app = FastAPI()
        app.include_router(agent_turn_router)
        app.state.turn_state = {}
        app.state.broadcaster = RecordingBroadcaster()
        return TestClient(app), app.state.broadcaster

    def _post(self, client, session_id, state="busy"):
        body = {"session_id": session_id, "pool_key": KEY, "state": state, "surface": "terminal"}
        response = client.post("/api/agent-turn", json=body)
        assert response.status_code == 200 and response.json()["recorded"] is True

    def test_a_report_appends_once_with_one_frame(self):
        client, broadcaster = self._client()
        self._post(client, KEY)
        self._post(client, KEY, "idle")  # the same transcript again: no row, no frame
        assert _rows() == [(KEY, KEY)]
        assert broadcaster.transcript_frames() == [
            {"type": "session_transcript", "session_key": KEY}
        ]

    def test_a_clear_appends_the_new_transcript(self):
        client, broadcaster = self._client()
        self._post(client, KEY)
        self._post(client, T1)
        self._post(client, T1, "idle")
        assert _rows() == [(KEY, KEY), (KEY, T1)]
        assert len(broadcaster.transcript_frames()) == 2

    def test_a_dropped_report_appends_nothing(self):
        client, broadcaster = self._client()
        body = {"session_id": KEY, "pool_key": KEY, "state": "busy", "surface": "chat"}
        assert client.post("/api/agent-turn", json=body).json()["recorded"] is False
        assert _rows() == []
        assert broadcaster.frames == []


# ---- chat turn end ---------------------------------------------------------------


def _chat_app(store_root):
    return SimpleNamespace(
        state=SimpleNamespace(
            transcript_map={},
            transcript_map_provisional=False,
            broadcaster=RecordingBroadcaster(),
            panel_status_root=store_root,
        )
    )


def _save_upload(store_root, chat_id) -> str:
    entry = ArtifactStore(workspace_root=store_root, auto_launch=False).save_file(
        file_content=b"payload",
        filename="beam.csv",
        artifact_type="data",
        title="beam.csv",
        mime_type="text/csv",
        tool_source="upload",
        metadata={"sent_at": "2026-11-01T10:00:00+00:00"},
        origin="upload",
        session_id=chat_id,
    )
    return entry.id


@pytest.mark.usefixtures("shared_root")
class TestChatWritePoint:
    def test_a_turn_end_appends_once_with_one_frame(self, tmp_path):
        app = _chat_app(tmp_path / "store")
        session = SimpleNamespace(transcript_id=KEY)
        chat_route._record_transcript(app, KEY, session)
        chat_route._record_transcript(app, KEY, session)
        assert _rows() == [(KEY, KEY)]
        assert app.state.broadcaster.transcript_frames() == [
            {"type": "session_transcript", "session_key": KEY}
        ]

    def test_a_non_uuid_key_records_nothing(self, tmp_path):
        app = _chat_app(tmp_path / "store")
        chat_route._record_transcript(app, "embedder-key", SimpleNamespace(transcript_id=T1))
        assert _rows() == []
        assert app.state.broadcaster.frames == []

    def test_a_clear_turns_upload_is_stamped_with_the_new_transcript(self, tmp_path):
        store_root = tmp_path / "store"
        store_root.mkdir()
        app = _chat_app(store_root)
        chat_route._record_transcript(app, KEY, SimpleNamespace(transcript_id=KEY))
        aid = _save_upload(store_root, KEY)
        # The /clear turn carried the upload and ended on a fresh transcript.
        chat_route._record_transcript(app, KEY, SimpleNamespace(transcript_id=T1), {"n": aid})
        store = ArtifactStore(workspace_root=store_root, auto_launch=False)
        (entry,) = [e for e in store.list_entries() if e.id == aid]
        assert entry.metadata["transcript_id"] == T1
        assert entry.metadata["sent_at"] == "2026-11-01T10:00:00+00:00"  # merged, not replaced
        assert owner_of(entry, session_ledger.load(session_meta.ledger_state_dir())) == T1
        assert _rows() == [(KEY, KEY), (KEY, T1)]

    def test_a_failed_stamp_never_fails_the_turn(self, tmp_path, monkeypatch, caplog):
        app = _chat_app(tmp_path / "store")

        def boom(self, updates):  # noqa: ARG001
            raise OSError("disk gone")

        monkeypatch.setattr(ArtifactStore, "merge_entry_metadata", boom)
        with caplog.at_level("WARNING"):
            chat_route._record_transcript(app, KEY, SimpleNamespace(transcript_id=T1), {"n": "x"})
        assert "stamp" in caplog.text
        assert _rows() == [(KEY, T1)]


@pytest.mark.timeout(60)
def test_a_live_chat_clear_appends_the_move_and_stamps_its_upload(tmp_path, monkeypatch):
    """Through the real route and lifespan: spawn row, turn rows, and the upload stamp."""
    from tests.interfaces.web_terminal._chat_server import PLANS, live_chat_server

    key = str(uuid.uuid4())
    moved_to = str(uuid.uuid4())
    uploads: dict[str, str] = {}
    real_mark_sent = chat_route.mark_sent

    def mark_sent(app, chat_id, prompt):
        real_mark_sent(app, chat_id, prompt)
        return dict(uploads) if prompt == "/clear" else {}

    monkeypatch.setattr(chat_route, "mark_sent", mark_sent)

    def turn(base_url, prompt):
        response = httpx.post(
            f"{base_url}/api/chat",
            params={"stream": "false"},
            json={"prompt": prompt, "chat_id": key},
            timeout=30,
        )
        assert response.status_code == 200, response.text

    with live_chat_server(tmp_path) as (base_url, app):
        frames: list[dict] = []
        real_broadcast = app.state.broadcaster.broadcast
        monkeypatch.setattr(
            app.state.broadcaster,
            "broadcast",
            lambda data, **kw: (frames.append(data), real_broadcast(data, **kw)),
        )
        PLANS["hello"] = [("text", "hi"), ("result", False, key)]
        PLANS["/clear"] = [("result", False, moved_to)]
        turn(base_url, "hello")
        uploads["beam.csv"] = _save_upload(app.state.panel_status_root, key)
        turn(base_url, "/clear")

        rows = _rows()
        assert rows == [(key, key), (key, moved_to)]
        assert [f for f in frames if f.get("type") == "session_transcript"] == [
            {"type": "session_transcript", "session_key": key}
        ] * 2
        store = ArtifactStore(workspace_root=app.state.panel_status_root, auto_launch=False)
        (entry,) = [e for e in store.list_entries() if e.id == uploads["beam.csv"]]
        assert entry.metadata["transcript_id"] == moved_to


# ---- spawn -----------------------------------------------------------------------


@pytest.fixture
def disk():
    """The transcripts on disk as the spawn sees them; tests add ids to it."""
    ids: set[str] = set()
    with patch.object(session_handoff, "_transcripts_on_disk", lambda _app: ids):
        yield ids


def _harness_app():
    app = make_app()
    app.state.broadcaster = RecordingBroadcaster()
    return app


@pytest.mark.usefixtures("shared_root")
class TestSpawnWritePoint:
    async def test_a_fresh_spawn_of_a_uuid_key_records_the_key(self, disk):  # noqa: ARG002 - pins the disk empty
        app = _harness_app()
        await acquire(app, HARNESS_KEY, "chat", object(), spawn=chat_spawner(app))
        assert _rows() == [(HARNESS_KEY, HARNESS_KEY)]
        assert app.state.broadcaster.transcript_frames() == [
            {"type": "session_transcript", "session_key": HARNESS_KEY}
        ]

    async def test_a_resume_records_the_resumed_transcript(self, disk):
        app = _harness_app()
        app.state.transcript_map[HARNESS_KEY] = T2
        disk.add(T2)
        await acquire(app, HARNESS_KEY, "terminal", object(), spawn=pty_spawner(app))
        assert _rows() == [(HARNESS_KEY, T2)]
        assert len(app.state.broadcaster.transcript_frames()) == 1

    async def test_a_fresh_spawn_of_a_key_with_rows_appends_nothing(self, disk):  # noqa: ARG002 - pins the disk empty
        app = _harness_app()
        session_ledger.append(session_meta.ledger_state_dir(), HARNESS_KEY, T1)
        await acquire(app, HARNESS_KEY, "chat", object(), spawn=chat_spawner(app))
        assert _rows() == [(HARNESS_KEY, T1)]
        assert app.state.broadcaster.frames == []

    def test_a_fresh_spawn_of_a_non_uuid_key_records_nothing(self):
        app = _harness_app()
        session_handoff._record_spawn(app, "embedder-key", None)
        assert _rows() == []
        assert app.state.broadcaster.frames == []
