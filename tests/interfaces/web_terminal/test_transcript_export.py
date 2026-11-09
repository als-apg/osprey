"""``GET /api/sessions/{id}/transcript.md``: a session exported as Markdown.

Pins the export: the title first, then every operator prompt and agent reply in
order, each under a facility-local timestamp, read through the same record read
the chat replay uses; a notice line when the record answered only part of the
session; an attachment named from the title; 404 for a session the record holds
nothing of, 503 with a short message when the record is unavailable, 422 for an
id that is not a UUID.
"""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.agent_record.client import ClientAnswer
from osprey.interfaces.web_terminal import session_meta
from osprey.interfaces.web_terminal.routes import session as session_routes
from osprey_connectors import posture_store
from tests.agent_record.fakes import FakeRecordClient

TID = "aaaaaaaa-1111-4222-8333-444444444444"
UNKNOWN = "bbbbbbbb-1111-4222-8333-444444444444"
ZONE = "America/Los_Angeles"


def _entry(role: str, text: str, stamp: str, n: int) -> dict:
    return {
        "type": role,
        "uuid": f"00000000-0000-4000-8000-{n:012d}",
        "timestamp": stamp,
        "sessionId": TID,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }


CONVERSATION = [
    ("user", "What is the beam current?", "2026-11-01T17:00:00+00:00"),
    ("assistant", "The beam current is 500 mA.", "2026-11-01T17:00:05+00:00"),
    ("user", "And the lifetime?", "2026-11-01T17:01:00+00:00"),
    ("assistant", "About 8 hours.", "2026-11-01T17:01:07+00:00"),
]


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    posture_store.invalidate_cache()
    yield root
    posture_store.invalidate_cache()


@pytest.fixture
def nodes(tmp_path) -> list[dict]:
    """The timeline nodes the record's own parser makes of :data:`CONVERSATION`."""
    path: Path = tmp_path / f"{TID}.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(_entry(role, text, stamp, i))
            for i, (role, text, stamp) in enumerate(CONVERSATION)
        )
        + "\n"
    )
    found = session_routes._transcript_nodes(path)
    assert found, "the fixture transcript produced no timeline nodes"
    return found


def _serve(monkeypatch, record: FakeRecordClient) -> FakeRecordClient:
    monkeypatch.setattr(session_routes, "record_client", lambda: record)
    monkeypatch.setattr(session_routes, "facility_timezone", lambda: ZoneInfo(ZONE))
    monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
    return record


def _timeline(nodes: list[dict], *, complete: bool = True):
    def timeline(**kwargs) -> ClientAnswer:
        if kwargs["session_id"] != TID:
            return ClientAnswer(data=[], complete=True)
        return ClientAnswer(
            data=nodes, complete=complete, reason="" if complete else "a page was cut short"
        )

    return timeline


@pytest.fixture
def app(state_root) -> FastAPI:  # noqa: ARG001
    app = FastAPI()
    app.include_router(session_routes.router)
    app.state.terminal_user = "alice"
    return app


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


def _export(client: TestClient, tid: str = TID):
    return client.get(f"/api/sessions/{tid}/transcript.md")


class TestContent:
    def test_title_then_turns_in_order_with_facility_local_stamps(
        self, client, app, nodes, monkeypatch
    ):
        _serve(monkeypatch, FakeRecordClient({"timeline": _timeline(nodes)}))
        session_meta.set_title(app, TID, "Beam check")

        resp = _export(client)

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        body = resp.text
        assert body.splitlines()[0] == "# Beam check"
        texts = [text for _, text, _ in CONVERSATION]
        positions = [body.index(t) for t in texts]
        assert positions == sorted(positions), body
        # PST starts on 2026-11-01 at 09:00 UTC, so 17:00 UTC reads 09:00 PST.
        assert "2026-11-01 09:00:00 PST" in body
        assert "2026-11-01 09:01:07 PST" in body
        assert "17:00:00" not in body
        # The operator's prompts and the agent's replies are told apart.
        assert body.index("Operator") < body.index(texts[0]) < body.index("Agent")
        assert session_routes.INCOMPLETE_TRANSCRIPT_NOTICE not in body

    def test_untitled_session_takes_its_first_prompt_as_title(self, client, nodes, monkeypatch):
        _serve(monkeypatch, FakeRecordClient({"timeline": _timeline(nodes)}))

        resp = _export(client)

        assert resp.status_code == 200
        assert resp.text.splitlines()[0] == "# What is the beam current?"

    def test_served_as_attachment_named_from_title(self, client, app, nodes, monkeypatch):
        _serve(monkeypatch, FakeRecordClient({"timeline": _timeline(nodes)}))
        session_meta.set_title(app, TID, "Beam check / night shift")

        resp = _export(client)

        disposition = resp.headers["content-disposition"]
        assert disposition.startswith("attachment;")
        assert 'filename="Beam-check-night-shift.md"' in disposition

    def test_reads_through_the_chat_replay_record_read(self, client, nodes, monkeypatch):
        """The export drains and reads the session's timeline, as the chat replay does."""
        record = _serve(monkeypatch, FakeRecordClient({"timeline": _timeline(nodes)}))

        _export(client)

        calls = record.calls_to("timeline")
        assert calls and calls[0]["session_id"] == TID
        assert calls[0]["wait"] is not None


class TestIncomplete:
    def test_incomplete_record_adds_a_visible_notice(self, client, app, nodes, monkeypatch):
        _serve(monkeypatch, FakeRecordClient({"timeline": _timeline(nodes, complete=False)}))
        session_meta.set_title(app, TID, "Beam check")

        resp = _export(client)

        assert resp.status_code == 200
        lines = resp.text.splitlines()
        assert session_routes.INCOMPLETE_TRANSCRIPT_NOTICE in resp.text
        # The notice sits under the title, ahead of the conversation.
        notice_at = next(
            i for i, line in enumerate(lines) if session_routes.INCOMPLETE_TRANSCRIPT_NOTICE in line
        )
        first_turn = next(i for i, line in enumerate(lines) if "What is the beam current?" in line)
        assert 0 < notice_at < first_turn
        assert "a page was cut short" in resp.text


class TestErrors:
    def test_unknown_session_is_404(self, client, nodes, monkeypatch):
        _serve(monkeypatch, FakeRecordClient({"timeline": _timeline(nodes)}))

        resp = _export(client, UNKNOWN)

        assert resp.status_code == 404

    def test_unavailable_record_is_503_with_short_message(self, client, monkeypatch):
        _serve(monkeypatch, FakeRecordClient(unavailable="clickhouse is down"))

        resp = _export(client)

        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert isinstance(detail, str) and 0 < len(detail) <= 120

    def test_non_uuid_id_is_422(self, client, monkeypatch):
        _serve(monkeypatch, FakeRecordClient())

        resp = _export(client, "not-a-uuid")

        assert resp.status_code == 422
