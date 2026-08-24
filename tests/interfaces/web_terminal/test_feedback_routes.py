"""Tests for the feedback routes (`routes/feedback.py`).

Two endpoints share one composer and one set of app.state reads::

    POST /api/feedback         {text, channel, include_metadata, include_context,
                                session_id?, scrollback?}
                            -> {id, payload, context_status}
    POST /api/feedback/bundle  {session_id, text_len, scrollback,
                                include_metadata}
                            -> {bundle}

The field names are pinned by the browser module ``static/js/feedback-client.js``
and must not drift. The send route writes one record and prunes the store; the
bundle route writes nothing at all.

The transcript reader and the artifact store are patched at their *source*
modules, because the route imports them lazily inside the handler (the house
convention in ``routes/session.py``).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.feedback_composer import (
    CONTEXT_NO_SESSION,
    CONTEXT_OK,
    CONTEXT_TRANSCRIPT_NOT_FOUND,
    PAYLOAD_CAP_BYTES,
    PAYLOAD_SEPARATOR,
)
from osprey.interfaces.web_terminal.routes.feedback import router

SESSION_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
OTHER_SESSION_ID = "11111111-2222-3333-4444-555555555555"


def _events(count: int = 3) -> list[dict[str, Any]]:
    """Transcript events with the real ``Z``-suffixed stamps Claude Code writes."""
    return [
        {
            "type": "tool_call",
            "timestamp": f"2026-08-20T18:0{index}:44.210Z",
            "tool": f"Tool{index}",
        }
        for index in range(count)
    ]


def _chat(count: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "timestamp": f"2026-08-20T18:0{index}:50.000Z",
            "content": f"turn {index}",
        }
        for index in range(count)
    ]


class _Env:
    """The app under test plus the doubles its handlers reach for."""

    def __init__(self, client: TestClient, reader: MagicMock, store: MagicMock, root: Path):
        self.client = client
        self.reader = reader
        self.store = store
        self.feedback_dir = root / "feedback"

    def headers(self) -> list[dict[str, Any]]:
        return [json.loads(p.read_text()) for p in sorted(self.feedback_dir.glob("fb-*.json"))]

    def contexts(self) -> list[dict[str, Any]]:
        return [json.loads(p.read_text()) for p in sorted(self.feedback_dir.glob("ctx-*.json"))]

    def files(self) -> list[str]:
        if not self.feedback_dir.exists():
            return []
        return sorted(p.name for p in self.feedback_dir.iterdir())


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Env:
    reader = MagicMock()
    reader.find_transcript_by_id.return_value = tmp_path / "transcript.jsonl"
    reader.read_session_by_id.return_value = _events()
    reader.read_chat_history_by_id.return_value = _chat()

    store = MagicMock()
    store.list_entries.return_value = []

    monkeypatch.setattr(
        "osprey.mcp_server.workspace.transcript_reader.TranscriptReader",
        lambda *args, **kwargs: reader,
    )
    monkeypatch.setattr(
        "osprey.stores.artifact_store.ArtifactStore",
        lambda *args, **kwargs: store,
    )

    app = FastAPI()
    app.include_router(router)
    app.state.project_cwd = str(tmp_path / "project")
    app.state.workspace_dir = tmp_path / "workspace"
    app.state.feedback_dir = tmp_path / "feedback"
    app.state.feedback_max_store_bytes = 256 * 1024 * 1024
    app.state.terminal_user = "operator"
    app.state.app_name = "OSPREY Test Rig"

    with TestClient(app) as client:
        yield _Env(client, reader, store, tmp_path)


def _send_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "text": "the terminal froze when I opened the ORM panel",
        "channel": "local",
        "include_metadata": True,
        "include_context": True,
        "session_id": SESSION_ID,
        "scrollback": "$ osprey web\nstarting…\n",
    }
    body.update(overrides)
    return body


# ---- POST /api/feedback: records ----


@pytest.mark.parametrize("channel", ["local", "github", "email"])
def test_send_writes_one_record_carrying_its_channel(env: _Env, channel: str) -> None:
    response = env.client.post("/api/feedback", json=_send_body(channel=channel))

    assert response.status_code == 200
    payload = response.json()
    assert payload["context_status"] == CONTEXT_OK
    assert payload["payload"]

    headers = env.headers()
    assert len(headers) == 1
    assert headers[0]["channel"] == channel
    assert headers[0]["id"] == payload["id"]
    assert headers[0]["identity"] == "operator"
    assert headers[0]["app_name"] == "OSPREY Test Rig"
    assert headers[0]["session_id"] == SESSION_ID
    assert headers[0]["version"]
    assert headers[0]["created_at"]
    assert len(env.contexts()) == 1


def test_send_keeps_the_full_text_and_context_in_the_record(env: _Env) -> None:
    long_text = "problem: " + ("детали " * 3000)
    response = env.client.post("/api/feedback", json=_send_body(text=long_text))

    assert response.status_code == 200
    context = env.contexts()[0]
    assert context["text"] == long_text
    assert "## deployment" in context["context"]
    assert context["scrollback"] == "$ osprey web\nstarting…\n"
    assert env.headers()[0]["excerpt"]


def test_send_stamps_the_payload_digest_only_for_outbound_channels(env: _Env) -> None:
    outbound = env.client.post("/api/feedback", json=_send_body(channel="github")).json()
    header = env.headers()[0]
    encoded = outbound["payload"].encode("utf-8")
    assert header["bundle_bytes"] == len(encoded)
    assert header["bundle_sha256"] == hashlib.sha256(encoded).hexdigest()

    for path in env.feedback_dir.iterdir():
        path.unlink()
    env.client.post("/api/feedback", json=_send_body(channel="local"))
    assert "bundle_sha256" not in env.headers()[0]
    assert "bundle_bytes" not in env.headers()[0]


# ---- POST /api/feedback: the byte cap ----


def test_send_caps_the_payload_at_sixty_thousand_utf8_bytes(env: _Env) -> None:
    # UTF-16 length (40_000) and UTF-8 byte count (80_000) differ, so a
    # character-counted cap would pass this text straight through.
    text = "Ж" * 40_000
    env.reader.read_session_by_id.return_value = _events(200)

    response = env.client.post(
        "/api/feedback", json=_send_body(text=text, scrollback="ы" * 200_000)
    )

    assert response.status_code == 200
    payload = response.json()["payload"]
    assert len(payload) < len(payload.encode("utf-8"))
    assert len(payload.encode("utf-8")) <= PAYLOAD_CAP_BYTES
    assert env.headers()[0]["text_truncated"] is True
    # The record keeps what the clipboard could not carry, and the truncation
    # marker names the record that actually exists on disk.
    assert env.contexts()[0]["text"] == text
    assert response.json()["id"] in payload
    assert env.headers()[0]["id"] == response.json()["id"]


def test_send_digests_the_exact_payload_it_returned_when_truncated(env: _Env) -> None:
    # The id is allocated before the header is built, so the digest covers the
    # very bytes the operator pastes — including the truncation marker naming
    # this record. A digest taken before any id substitution would not.
    text = "Ж" * 40_000
    env.reader.read_session_by_id.return_value = _events(200)

    body = env.client.post("/api/feedback", json=_send_body(text=text, channel="github")).json()

    header = env.headers()[0]
    assert header["text_truncated"] is True
    assert header["id"] in body["payload"]
    encoded = body["payload"].encode("utf-8")
    assert header["bundle_bytes"] == len(encoded)
    assert header["bundle_sha256"] == hashlib.sha256(encoded).hexdigest()


# ---- POST /api/feedback: session handling ----


def test_send_reports_transcript_not_found_for_a_dead_session(env: _Env) -> None:
    env.reader.find_transcript_by_id.return_value = None

    response = env.client.post("/api/feedback", json=_send_body())

    assert response.status_code == 200
    assert response.json()["context_status"] == CONTEXT_TRANSCRIPT_NOT_FOUND
    header = env.headers()[0]
    assert header["context_status"] == CONTEXT_TRANSCRIPT_NOT_FOUND
    assert header["session_id"] == SESSION_ID
    # Text and metadata still travel — the dialog promises exactly that.
    assert env.contexts()[0]["text"] == _send_body()["text"]
    assert "## deployment" in env.contexts()[0]["context"]
    env.reader.read_session_by_id.assert_not_called()


def test_send_rejects_a_non_uuid_session_id_before_touching_the_transcript(env: _Env) -> None:
    response = env.client.post("/api/feedback", json=_send_body(session_id="../../etc/passwd"))

    assert response.status_code == 422
    assert "session_id" in response.json()["detail"]
    env.reader.find_transcript_by_id.assert_not_called()
    assert env.files() == []


def test_send_without_context_composes_metadata_only(env: _Env) -> None:
    response = env.client.post(
        "/api/feedback", json=_send_body(include_context=False, channel="email")
    )

    assert response.status_code == 200
    assert response.json()["context_status"] == CONTEXT_NO_SESSION
    header = env.headers()[0]
    assert header["include_context"] is False
    assert header["session_id"] is None
    env.reader.find_transcript_by_id.assert_not_called()
    assert env.contexts()[0]["scrollback"] == ""


def test_send_drops_the_metadata_tier_when_the_checkbox_is_off(env: _Env) -> None:
    with_metadata = env.client.post("/api/feedback", json=_send_body()).json()["payload"]
    without = env.client.post("/api/feedback", json=_send_body(include_metadata=False)).json()[
        "payload"
    ]

    assert "OSPREY Test Rig" in with_metadata
    assert "OSPREY Test Rig" not in without
    # Identity and timestamp travel either way — the popover says so.
    assert "operator" in without


def test_send_metadata_tier_names_the_submitting_browser(env: _Env) -> None:
    # An outbound draft prefills no metadata block of its own any more (the
    # body is just a pointer line), so the browser has to reach the maintainer
    # through the composed payload — from the User-Agent header, and only under
    # the metadata checkbox, because the popover lists it there.
    agent = "Mozilla/5.0 (TestBrowser; feedback-suite)"
    with_metadata = env.client.post(
        "/api/feedback", json=_send_body(), headers={"User-Agent": agent}
    ).json()["payload"]
    assert agent in with_metadata

    for path in env.feedback_dir.iterdir():
        path.unlink()
    without = env.client.post(
        "/api/feedback",
        json=_send_body(include_metadata=False),
        headers={"User-Agent": agent},
    ).json()["payload"]
    assert agent not in without


def test_send_carries_the_session_id_in_the_deployment_tier(env: _Env) -> None:
    # Same reasoning: the pointer-line draft names the session only by an
    # 8-character title prefix, so the full id must ride in the payload the
    # operator pastes — with the context it belongs to, never without it.
    with_context = env.client.post("/api/feedback", json=_send_body(channel="github")).json()[
        "payload"
    ]
    assert SESSION_ID in with_context

    for path in env.feedback_dir.iterdir():
        path.unlink()
    without = env.client.post("/api/feedback", json=_send_body(include_context=False)).json()[
        "payload"
    ]
    assert SESSION_ID not in without


# ---- POST /api/feedback: collection discipline ----


def test_send_reads_each_transcript_source_exactly_once(env: _Env) -> None:
    env.client.post("/api/feedback", json=_send_body(channel="github"))

    assert env.reader.read_session_by_id.call_count == 1
    assert env.reader.read_chat_history_by_id.call_count == 1
    assert env.store.list_entries.call_count == 1


def test_send_handler_runs_in_the_threadpool(env: _Env) -> None:
    from osprey.interfaces.web_terminal.routes import feedback as module

    assert not inspect.iscoroutinefunction(module.submit_feedback)


def test_send_prunes_the_store_after_writing(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from osprey.interfaces.web_terminal.routes import feedback as module

    calls: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        module, "prune_store", lambda directory, max_bytes: calls.append((directory, max_bytes))
    )

    env.client.post("/api/feedback", json=_send_body())

    assert calls == [(env.feedback_dir, 256 * 1024 * 1024)]


def test_send_fails_loudly_when_the_record_cannot_be_written(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.interfaces.web_terminal.routes import feedback as module

    def _boom(*args: Any, **kwargs: Any) -> str:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(module, "write_record", _boom)

    response = env.client.post("/api/feedback", json=_send_body())

    assert response.status_code == 500
    assert "detail" in response.json()


def test_send_falls_back_to_the_process_account_without_a_terminal_user(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.client.app.state.terminal_user = ""
    monkeypatch.setattr("getpass.getuser", lambda: "hostaccount")

    env.client.post("/api/feedback", json=_send_body())

    assert env.headers()[0]["identity"] == "hostaccount"


def test_send_identity_is_unknown_when_the_process_account_is_unavailable(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.client.app.state.terminal_user = ""

    def _boom() -> str:
        raise KeyError("no pwd entry")

    monkeypatch.setattr("getpass.getuser", _boom)

    env.client.post("/api/feedback", json=_send_body())

    assert env.headers()[0]["identity"] == "unknown"


# ---- POST /api/feedback: hostile input ----


def test_send_strips_control_characters_from_the_stored_excerpt(env: _Env) -> None:
    # `osprey feedback list` prints the excerpt straight into a maintainer's
    # terminal, and rich passes ESC through untouched.
    env.client.post("/api/feedback", json=_send_body(text="\x1b[2Jwiped\x07 the screen"))

    excerpt = env.headers()[0]["excerpt"]
    assert "\x1b" not in excerpt
    assert "\x07" not in excerpt
    assert "wiped" in excerpt


def test_send_survives_a_lone_surrogate_in_the_text(env: _Env) -> None:
    # A lone surrogate reaches the handler intact and used to 500 in Starlette's
    # response renderer — after the record had been written and the store pruned.
    response = env.client.post(
        "/api/feedback",
        content=json.dumps(_send_body(text="bad \ud800 surrogate")).encode("ascii"),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert "\ud800" not in response.json()["payload"]
    assert len(env.headers()) == 1


def test_send_survives_a_lone_surrogate_in_the_scrollback(env: _Env) -> None:
    response = env.client.post(
        "/api/feedback",
        content=json.dumps(_send_body(scrollback="$ ls \udfff\n")).encode("ascii"),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert len(env.contexts()) == 1


def test_send_survives_a_lone_surrogate_read_from_the_transcript(env: _Env) -> None:
    # Nothing hostile in the request: a browser that stringified an unpaired
    # UTF-16 half wrote this into the transcript, and it comes back off disk.
    env.reader.read_chat_history_by_id.return_value = [
        {"role": "user", "timestamp": "2026-08-20T18:00:50.000Z", "content": "oops \ud800 half"}
    ]

    response = env.client.post("/api/feedback", json=_send_body(channel="github"))

    assert response.status_code == 200
    assert "\ud800" not in response.json()["payload"]
    header = env.headers()[0]
    assert "\ud800" not in env.contexts()[0]["context"]
    # The digest still describes the bytes that were returned.
    encoded = response.json()["payload"].encode("utf-8")
    assert header["bundle_bytes"] == len(encoded)
    assert header["bundle_sha256"] == hashlib.sha256(encoded).hexdigest()


def test_send_refuses_a_submission_over_the_body_limit(env: _Env) -> None:
    from osprey.interfaces.web_terminal.routes.feedback import MAX_REQUEST_BYTES

    half = MAX_REQUEST_BYTES // 2
    response = env.client.post(
        "/api/feedback", json=_send_body(text="a" * half, scrollback="b" * (half + 1))
    )

    assert response.status_code == 413
    assert "too large" in response.json()["detail"]
    assert env.files() == []
    # Refused before any transcript work — a different claim from "no record".
    env.reader.find_transcript_by_id.assert_not_called()


def test_send_accepts_a_submission_at_the_body_limit(env: _Env) -> None:
    from osprey.interfaces.web_terminal.routes.feedback import MAX_REQUEST_BYTES

    response = env.client.post(
        "/api/feedback", json=_send_body(text="a" * MAX_REQUEST_BYTES, scrollback="")
    )

    assert response.status_code == 200
    assert len(env.headers()) == 1


# ---- POST /api/feedback: app.state handling ----


def test_send_stamps_one_instant_per_request(env: _Env) -> None:
    env.client.post("/api/feedback", json=_send_body())

    header = env.headers()[0]
    assert header["created_at"] in env.contexts()[0]["context"]


def test_send_ignores_a_boolean_store_ceiling(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from osprey.interfaces.web_terminal.routes import feedback as module

    # True is an int in Python; honoured literally it is a 1-byte ceiling that
    # prunes every context in the store.
    env.client.app.state.feedback_max_store_bytes = True
    calls: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        module, "prune_store", lambda directory, max_bytes: calls.append((directory, max_bytes))
    )

    env.client.post("/api/feedback", json=_send_body())

    assert calls == [(env.feedback_dir, module.DEFAULT_MAX_STORE_BYTES)]


def test_send_falls_back_to_the_workspace_for_an_unset_store_dir(env: _Env) -> None:
    del env.client.app.state.feedback_dir
    workspace_dir = Path(env.client.app.state.workspace_dir)

    response = env.client.post("/api/feedback", json=_send_body())

    assert response.status_code == 200
    assert list((workspace_dir / "feedback").glob("fb-*.json"))


def test_send_reports_a_deployment_with_nowhere_to_store_records(env: _Env) -> None:
    del env.client.app.state.feedback_dir
    del env.client.app.state.workspace_dir

    response = env.client.post("/api/feedback", json=_send_body())

    assert response.status_code == 500
    assert "no feedback store is configured" in response.json()["detail"]


# ---- POST /api/feedback/bundle ----


def test_bundle_writes_nothing_at_all(env: _Env) -> None:
    response = env.client.post(
        "/api/feedback/bundle",
        json={"session_id": SESSION_ID, "text_len": 120, "scrollback": "$ ls\n"},
    )

    assert response.status_code == 200
    assert response.json()["bundle"]
    assert env.files() == []


@pytest.mark.parametrize("metadata_on", [True, False])
def test_bundle_is_byte_identical_to_the_posted_payload_context(
    env: _Env, monkeypatch: pytest.MonkeyPatch, metadata_on: bool
) -> None:
    from osprey.interfaces.web_terminal.routes import feedback as module

    # The deployment tier stamps the moment of submission, and the two requests
    # are not made in the same microsecond. Everything else must match byte for
    # byte — that identity is what lets an operator copy the context, keep
    # typing, and still send a report whose context is the one they pasted. It
    # has to hold for either position of the metadata checkbox, which both
    # buttons now read.
    frozen = datetime(2026, 8, 20, 18, 4, 44, tzinfo=UTC)
    monkeypatch.setattr(module, "_utc_now", lambda: frozen)

    text = "the ORM panel hangs on résumé"
    scrollback = "$ osprey web\nstarting…\n"
    posted = env.client.post(
        "/api/feedback",
        json=_send_body(text=text, scrollback=scrollback, include_metadata=metadata_on),
    ).json()["payload"]

    copied = env.client.post(
        "/api/feedback/bundle",
        json={
            "session_id": SESSION_ID,
            "text_len": len(text.encode("utf-8")),
            "scrollback": scrollback,
            "include_metadata": metadata_on,
        },
    ).json()["bundle"]

    assert PAYLOAD_SEPARATOR in posted
    assert posted.split(PAYLOAD_SEPARATOR, 1)[1] == copied


def test_bundle_drops_the_metadata_tier_when_the_checkbox_is_off(env: _Env) -> None:
    request = {"session_id": SESSION_ID, "text_len": 0, "scrollback": ""}

    with_metadata = env.client.post("/api/feedback/bundle", json=request).json()["bundle"]
    without = env.client.post(
        "/api/feedback/bundle", json={**request, "include_metadata": False}
    ).json()["bundle"]

    assert "OSPREY Test Rig" in with_metadata
    assert "OSPREY Test Rig" not in without
    # Identity and timestamp travel either way — the popover says so.
    assert "operator" in without
    assert "submitted" in without


def test_bundle_keeps_the_metadata_tier_for_a_client_that_omits_the_flag(env: _Env) -> None:
    body = env.client.post(
        "/api/feedback/bundle",
        json={"session_id": SESSION_ID, "text_len": 0, "scrollback": ""},
    ).json()["bundle"]

    assert "OSPREY Test Rig" in body


def test_bundle_rejects_a_non_uuid_session_id(env: _Env) -> None:
    response = env.client.post(
        "/api/feedback/bundle",
        json={"session_id": "not-a-uuid", "text_len": 10, "scrollback": ""},
    )

    assert response.status_code == 422
    env.reader.find_transcript_by_id.assert_not_called()
    assert env.files() == []


@pytest.mark.parametrize("text_len", [-5, None, 10**12])
def test_bundle_respects_the_cap_for_any_text_len(env: _Env, text_len: Any) -> None:
    env.reader.read_session_by_id.return_value = _events(400)

    response = env.client.post(
        "/api/feedback/bundle",
        json={"session_id": SESSION_ID, "text_len": text_len, "scrollback": "ы" * 100_000},
    )

    assert response.status_code == 200
    assert len(response.json()["bundle"].encode("utf-8")) <= PAYLOAD_CAP_BYTES


@pytest.mark.parametrize("text_len", ["abc", [1, 2], {"n": 1}])
def test_bundle_refuses_a_non_numeric_text_len(env: _Env, text_len: Any) -> None:
    # The int|None typing is what moved this handling from clamp_text_len to
    # pydantic; nothing else guards it.
    response = env.client.post(
        "/api/feedback/bundle",
        json={"session_id": SESSION_ID, "text_len": text_len, "scrollback": ""},
    )

    assert response.status_code == 422
    assert env.files() == []


def test_bundle_respects_the_cap_when_text_len_is_absent(env: _Env) -> None:
    env.reader.read_session_by_id.return_value = _events(400)

    response = env.client.post(
        "/api/feedback/bundle",
        json={"session_id": SESSION_ID, "scrollback": "ы" * 100_000},
    )

    assert response.status_code == 200
    assert len(response.json()["bundle"].encode("utf-8")) <= PAYLOAD_CAP_BYTES


def test_bundle_without_a_session_returns_metadata_only(env: _Env) -> None:
    response = env.client.post("/api/feedback/bundle", json={"text_len": 0})

    assert response.status_code == 200
    body = response.json()["bundle"]
    assert "## deployment" in body
    assert "## event log" not in body
    env.reader.find_transcript_by_id.assert_not_called()
    assert env.files() == []


def test_bundle_reads_each_transcript_source_exactly_once(env: _Env) -> None:
    env.client.post(
        "/api/feedback/bundle",
        json={"session_id": SESSION_ID, "text_len": 0, "scrollback": ""},
    )

    assert env.reader.read_session_by_id.call_count == 1
    assert env.reader.read_chat_history_by_id.call_count == 1
    assert env.store.list_entries.call_count == 1


def test_bundle_handler_runs_in_the_threadpool(env: _Env) -> None:
    from osprey.interfaces.web_terminal.routes import feedback as module

    assert not inspect.iscoroutinefunction(module.feedback_bundle)


def test_bundle_survives_a_lone_surrogate_read_from_the_transcript(env: _Env) -> None:
    env.reader.read_chat_history_by_id.return_value = [
        {"role": "user", "timestamp": "2026-08-20T18:00:50.000Z", "content": "oops \ud800 half"}
    ]

    response = env.client.post(
        "/api/feedback/bundle",
        json={"session_id": SESSION_ID, "text_len": 0, "scrollback": ""},
    )

    assert response.status_code == 200
    assert "\ud800" not in response.json()["bundle"]
    assert env.files() == []


def test_bundle_refuses_a_scrollback_over_the_body_limit(env: _Env) -> None:
    from osprey.interfaces.web_terminal.routes.feedback import MAX_REQUEST_BYTES

    response = env.client.post(
        "/api/feedback/bundle",
        json={
            "session_id": SESSION_ID,
            "text_len": 0,
            "scrollback": "b" * (MAX_REQUEST_BYTES + 1),
        },
    )

    assert response.status_code == 413
    assert "too large" in response.json()["detail"]
    assert env.files() == []
    env.reader.find_transcript_by_id.assert_not_called()


def test_bundle_survives_a_lone_surrogate_in_the_scrollback(env: _Env) -> None:
    response = env.client.post(
        "/api/feedback/bundle",
        content=json.dumps(
            {"session_id": SESSION_ID, "text_len": 0, "scrollback": "$ ls \ud800\n"}
        ).encode("ascii"),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert "\ud800" not in response.json()["bundle"]
    assert env.files() == []


def test_bundle_uses_the_session_it_was_given(env: _Env) -> None:
    env.client.post(
        "/api/feedback/bundle",
        json={"session_id": OTHER_SESSION_ID, "text_len": 0, "scrollback": ""},
    )

    env.reader.find_transcript_by_id.assert_called_once_with(OTHER_SESSION_ID)


# ---- Registration on the composite router ----


def test_registration_exposes_both_feedback_paths() -> None:
    """The composite web-terminal router exposes both feedback endpoints.

    Starlette 1.x ``include_router`` does not flatten, so registration is
    asserted through the OpenAPI schema of an app mounting the composite
    router, never through ``router.routes``.
    """
    from osprey.interfaces.web_terminal.routes import router as composite_router

    app = FastAPI()
    app.include_router(composite_router)
    paths = app.openapi()["paths"]
    assert "/api/feedback" in paths
    assert "post" in paths["/api/feedback"]
    assert "/api/feedback/bundle" in paths
    assert "post" in paths["/api/feedback/bundle"]
