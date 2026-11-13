"""``GET /api/session-activity`` and ``/api/session-activity/result``.

The Activity page reads one session through one route: the record's timeline
joined into the page's nodes, the artifacts the transcript owns with the call
that saved each, and the state of the read. The record is the fake the other
session routes are tested against.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from osprey.agent_record.client import ClientAnswer
from osprey.agent_record.contract import Drain
from osprey.interfaces.web_terminal.routes import session as session_routes
from osprey.stores.artifact_store import ArtifactStore
from tests.agent_record.fakes import FakeRecordClient
from tests.interfaces.web_terminal._fakes import FakePtySession
from tests.interfaces.web_terminal._started_app import terminal_client

KEY = "11111111-2222-4333-8444-555555555555"
TRANSCRIPT = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
OTHER = "99999999-8888-4777-8666-555555555555"

ROUTE = "/api/session-activity"
RESULT = "/api/session-activity/result"

_CURRENT = "osprey.mcp_server.workspace.transcript_reader.TranscriptReader.find_current_transcript"
_T0 = datetime(2026, 11, 12, 16, 0, tzinfo=UTC)


def _at(seconds: int) -> str:
    return (_T0 + timedelta(seconds=seconds)).isoformat()


@pytest.fixture
def client(workspace_dir):
    with terminal_client(workspace_dir) as c:
        c.app.state.transcript_map = {}
        c.app.state.transcript_map_provisional = False
        c.app.state.terminal_user = "alice"
        yield c


@pytest.fixture(autouse=True)
def record(monkeypatch):
    """A fake record the routes read; tests set ``record.answers``."""
    fake = FakeRecordClient()
    monkeypatch.setattr(session_routes, "record_client", lambda: fake)
    monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
    return fake


def _call(second: int, call_id: str, name: str, content: str, **args) -> dict:
    return {
        "kind": "tool_call",
        "key": f"k:call:{call_id}",
        "ts": _at(second),
        "tool_use_id": call_id,
        "body": {"id": call_id, "name": name, "input": args},
        "result": {
            "kind": "tool_result",
            "key": f"k:result:{call_id}",
            "ts": _at(second + 1),
            "tool_use_id": call_id,
            "body": {"content": content},
        },
    }


def _nodes() -> list[dict]:
    return [
        {"kind": "prompt", "key": "k:p0", "ts": _at(0), "body": {"text": "Read the beam current"}},
        _call(1, "t1", "Read", "file body", file_path="/x"),
        _call(
            3,
            "t2",
            "mcp__controls__channel_read",
            json.dumps({"result": {"value": 1.0}}),
            channels=["SR:DCCT"],
        ),
    ]


def _tools(body: dict) -> list[dict]:
    return [n for n in body["nodes"] if n["t"] == "tool"]


class TestActivity:
    def test_activity_returns_builtin_and_mcp_calls(self, client, record):
        record.answers["timeline"] = ClientAnswer(data=_nodes(), complete=True)

        resp = client.get(ROUTE, params={"transcript_id": TRANSCRIPT})

        assert resp.status_code == 200
        body = resp.json()
        assert [(t["name"], t["server"], t["tool"]) for t in _tools(body)] == [
            ("Read", "", "Read"),
            ("mcp__controls__channel_read", "controls", "channel_read"),
        ]
        assert body["record"] == "complete"
        assert body["artifacts"] == []
        session = body["session"]
        assert session["entries"] == 5
        assert session["kinds"] == {"prompt": 1, "tool_call": 2, "tool_result": 2}
        assert (session["start"], session["end"]) == (_at(0), _at(4))

    def test_activity_by_transcript_id(self, client, record):
        record.answers["timeline"] = ClientAnswer(data=_nodes(), complete=True)

        session = client.get(ROUTE, params={"transcript_id": TRANSCRIPT}).json()["session"]

        (call,) = record.calls_to("timeline")
        assert call["session_id"] == TRANSCRIPT
        assert call["wait"] == Drain(session_id=TRANSCRIPT)
        assert session["id"] == TRANSCRIPT
        assert session["title"] == "Read the beam current"
        assert session["identity"] == "alice"
        assert session["state"] == "idle"

    def test_activity_by_session_key_follows_transcript_map(self, client, record):
        client.app.state.transcript_map = {KEY: TRANSCRIPT}
        record.answers["timeline"] = ClientAnswer(data=_nodes(), complete=True)

        session = client.get(ROUTE, params={"session_id": KEY}).json()["session"]

        (call,) = record.calls_to("timeline")
        assert call["session_id"] == TRANSCRIPT
        assert (session["id"], session["session_key"]) == (TRANSCRIPT, KEY)

    def test_transcript_id_wins_over_the_session_key(self, client, record):
        client.app.state.transcript_map = {KEY: OTHER}

        session = client.get(ROUTE, params={"transcript_id": TRANSCRIPT, "session_id": KEY}).json()[
            "session"
        ]

        assert [c["session_id"] for c in record.calls_to("timeline")] == [TRANSCRIPT]
        assert (session["id"], session["session_key"]) == (TRANSCRIPT, None)

    def test_activity_unknown_key_has_no_fallback(self, client, record, tmp_path):
        """A key the record holds nothing of yet is that session, empty — never another one."""
        with patch(_CURRENT, return_value=tmp_path / f"{OTHER}.jsonl"):
            body = client.get(ROUTE, params={"session_id": KEY}).json()

        assert [c["session_id"] for c in record.calls_to("timeline")] == [KEY]
        assert body["session"]["id"] == KEY
        assert body["session"]["entries"] == 0
        assert (body["session"]["start"], body["session"]["end"]) == (None, None)
        assert body["nodes"] == [] and body["artifacts"] == []
        assert body["record"] == "complete"

    def test_without_a_key_the_current_transcript_names_the_session(self, client, tmp_path):
        with patch(_CURRENT, return_value=tmp_path / f"{TRANSCRIPT}.jsonl"):
            body = client.get(ROUTE).json()

        assert body["session"]["id"] == TRANSCRIPT

    def test_without_any_session_nothing_is_read(self, client, record):
        with patch(_CURRENT, return_value=None):
            body = client.get(ROUTE).json()

        assert body == {"session": None, "nodes": [], "artifacts": [], "record": "complete"}
        assert record.calls_to("timeline") == []

    def test_a_key_with_a_running_child_is_live(self, client):
        client.app.state.pty_registry._sessions[KEY] = FakePtySession()

        session = client.get(ROUTE, params={"session_id": KEY}).json()["session"]

        assert session["state"] == "live"

    def test_activity_reports_record_incomplete_with_reason(self, client, record):
        record.answers["timeline"] = ClientAnswer(
            data=_nodes(), complete=False, reason="wait mark not passed"
        )

        body = client.get(ROUTE, params={"transcript_id": TRANSCRIPT}).json()

        assert len(_tools(body)) == 2
        assert body["record"] == "incomplete"
        assert body["record_reason"] == "wait mark not passed"

    def test_activity_record_unavailable_is_200_with_state(self, client, monkeypatch):
        monkeypatch.setattr(
            session_routes, "record_client", lambda: FakeRecordClient(unavailable="down")
        )

        resp = client.get(ROUTE, params={"transcript_id": TRANSCRIPT})

        assert resp.status_code == 200
        body = resp.json()
        assert body["record"] == "unavailable"
        assert body["record_reason"] == "down"
        assert body["session"]["id"] == TRANSCRIPT
        assert body["nodes"] == [] and body["artifacts"] == []

    def test_activity_rejects_non_uuid_transcript_id(self, client, record):
        resp = client.get(ROUTE, params={"transcript_id": "../outside"})

        assert resp.status_code == 422
        assert record.calls_to("timeline") == []

    def test_a_session_id_that_is_no_record_token_reads_nothing(self, client, record):
        body = client.get(ROUTE, params={"session_id": "../../outside/planted"}).json()

        assert body["session"] is None
        assert record.calls_to("timeline") == []


class TestRetention:
    def test_the_configured_retention_is_reported(self, client, monkeypatch):
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value",
            lambda path, default=None: 30 if path == "agent_record.retention_days" else default,
        )

        session = client.get(ROUTE, params={"transcript_id": TRANSCRIPT}).json()["session"]

        assert session["retention_days"] == 30

    def test_an_unreadable_config_reports_no_retention(self, client, monkeypatch):
        def broken(*_args, **_kwargs):
            raise RuntimeError("no config")

        monkeypatch.setattr("osprey_connectors.config.get_config_value", broken)

        session = client.get(ROUTE, params={"transcript_id": TRANSCRIPT}).json()["session"]

        assert session["retention_days"] is None


class TestArtifacts:
    @staticmethod
    def _save(store: ArtifactStore, title: str, *, transcript: str, **kwargs):
        return store.save_file(
            file_content=b"<html></html>",
            filename="p.html",
            artifact_type="plot_html",
            title=title,
            mime_type="text/html",
            tool_source="create_interactive_plot",
            metadata={"transcript_id": transcript, **kwargs.pop("metadata", {})},
            **kwargs,
        )

    def test_activity_artifacts_name_their_call(self, client, record, workspace_dir):
        record.answers["timeline"] = ClientAnswer(data=_nodes(), complete=True)
        store = ArtifactStore(workspace_dir)
        mine = self._save(store, "Beam current", transcript=TRANSCRIPT, metadata={"call_id": "t2"})
        loose = self._save(store, "No call", transcript=TRANSCRIPT)
        self._save(store, "Theirs", transcript=OTHER, metadata={"call_id": "t2"})

        body = client.get(ROUTE, params={"transcript_id": TRANSCRIPT}).json()

        rows = {a["id"]: a for a in body["artifacts"]}
        assert set(rows) == {mine.id, loose.id}
        assert rows[mine.id] == {
            "id": mine.id,
            "title": "Beam current",
            "call": "t2",
            "tool": "channel_read",
            "agent": "main",
            "ts": mine.timestamp,
            "form": rows[mine.id]["form"],
            "form_label": rows[mine.id]["form_label"],
            "type": "plot_html",
            "size": len(b"<html></html>"),
        }
        assert rows[mine.id]["form"] and rows[mine.id]["form_label"]
        assert (rows[loose.id]["call"], rows[loose.id]["tool"], rows[loose.id]["agent"]) == (
            "",
            "",
            "",
        )

    def test_an_artifact_of_a_subagents_call_names_the_subagent(
        self, client, record, workspace_dir
    ):
        inner = _call(2, "t9", "mcp__plots__create_interactive_plot", "saved")
        agent = _call(1, "t8", "Agent", "done", subagent_type="data-visualizer")
        agent["subagent"] = {"agent_id": "a1", "timeline": [inner]}
        record.answers["timeline"] = ClientAnswer(data=[agent], complete=True)
        plot = self._save(
            ArtifactStore(workspace_dir), "Plot", transcript=TRANSCRIPT, metadata={"call_id": "t9"}
        )

        (row,) = client.get(ROUTE, params={"transcript_id": TRANSCRIPT}).json()["artifacts"]

        assert (row["id"], row["tool"], row["agent"]) == (
            plot.id,
            "create_interactive_plot",
            "data-visualizer",
        )


class TestResult:
    def test_result_returns_whole_text(self, client, record):
        whole = "x" * 5000
        record.answers["full_entry"] = ClientAnswer(
            data={"key": "k:result:t1", "kind": "tool_result", "body": {"content": whole}},
            complete=True,
        )

        resp = client.get(RESULT, params={"key": "k:result:t1"})

        assert resp.status_code == 200
        assert resp.json() == {"text": whole, "size": 5000, "record": "complete"}
        assert [c["key"] for c in record.calls_to("full_entry")] == ["k:result:t1"]

    def test_result_is_unwrapped_as_the_page_shows_it(self, client, record):
        record.answers["full_entry"] = ClientAnswer(
            data={"body": {"content": json.dumps({"result": {"value": 1.0}})}}, complete=True
        )

        body = client.get(RESULT, params={"key": "k"}).json()

        assert json.loads(body["text"]) == {"value": 1.0}

    def test_result_404_when_not_visible(self, client, record):
        record.answers["full_entry"] = ClientAnswer(data=None, complete=True)

        assert client.get(RESULT, params={"key": "k:other"}).status_code == 404

    def test_result_503_when_the_record_is_unavailable(self, client, monkeypatch):
        monkeypatch.setattr(
            session_routes, "record_client", lambda: FakeRecordClient(unavailable="down")
        )

        assert client.get(RESULT, params={"key": "k"}).status_code == 503

    def test_result_needs_a_key(self, client):
        assert client.get(RESULT).status_code == 422


@pytest.mark.parametrize("path", ["/api/session-agents", "/api/session-agent-timeline?agent_id=a"])
def test_removed_routes_are_gone(client, path):
    assert client.get(path).status_code == 404
