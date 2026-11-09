"""The hermetic hub's e2e seams: the real record reader, the scripted chat child,
the naming seam and the seeds.

Each test boots a hub with :func:`hermetic_hub`'s ``e2e`` seeds (the
process's web credentials are per test). Pins: the
seeded sessions listed by ``GET /api/sessions`` with their keys, read through
the record's own tailer and reader; a chat turn after ``/clear`` appending the
ledger, its artifact owned by the transcript the ``/clear`` opened; a stubbed
name arriving on the hub's event stream as ``session_named``; and the
``beam_profile`` stub viewer served at ``/viewers/beam_profile.js``. Then, on
a hub without the seeds, the capture's demo session read back through the
same reader (the stub it replaces answered only ``timeline``).
"""

from __future__ import annotations

import json
import threading
import time
import uuid

import httpx
import pytest
from docs.screenshots.contact_sheet import (
    BEAM_PROFILE_KIND,
    BEAM_PROFILE_MARKER_CLASS,
    DEMO_OPENING_PROMPT,
    DEMO_SESSION_ID,
    SEEDED_SESSIONS,
    _write_fake_session,
    hermetic_hub,
)

from osprey.interfaces.web_terminal import session_meta
from osprey.stores import session_ledger
from osprey.stores.artifact_store import ArtifactStore, owner_of

_TIMEOUT = 30.0


def _http(base_url: str) -> httpx.Client:
    """A loopback client authenticated as every non-browser harness caller is:
    the operator-secret header from this process's credential holder."""
    from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
    from osprey.interfaces.web_auth import get_web_credentials

    return httpx.Client(
        base_url=base_url,
        timeout=_TIMEOUT,
        headers={OPERATOR_SECRET_HEADER: get_web_credentials().operator_secret},
    )


@pytest.fixture
def hub():
    with hermetic_hub(e2e=True) as running:
        yield running


@pytest.fixture
def client(hub):
    with _http(hub.base_url) as c:
        yield c


def _sessions(client) -> dict[str, dict]:
    response = client.get("/api/sessions")
    assert response.status_code == 200, response.text
    body = response.json()
    return {row["transcript_id"]: row for row in body["sessions"]}


def _chat(client, key: str, prompt: str) -> dict:
    response = client.post(
        "/api/chat", params={"stream": "false"}, json={"prompt": prompt, "chat_id": key}
    )
    assert response.status_code == 200, response.text
    return response.json()


# --- the record ---------------------------------------------------------------------


def test_the_seeded_sessions_are_listed_with_their_keys(client):
    rows = _sessions(client)
    for seeded in SEEDED_SESSIONS:
        row = rows[seeded.transcript_id]
        assert row["session_key"] == seeded.session_key
        assert row["title"] == seeded.prompt or row["title_source"] == "auto"
        assert row["message_count"] == 2
    rf_first, rf_after_clear, _profile = SEEDED_SESSIONS
    assert rows[rf_after_clear.transcript_id]["current_for_key"] is True
    assert rows[rf_first.transcript_id]["current_for_key"] is False


def test_the_record_rows_come_from_the_ingested_transcripts(hub):
    answer = hub.record.list_sessions()
    assert answer.available and answer.complete
    rows = {row["session_id"]: row for row in answer.data}
    # The /clear echo that opens the second RF transcript is not its first prompt.
    rf_after_clear = SEEDED_SESSIONS[1]
    assert rows[rf_after_clear.transcript_id]["prompt"] == rf_after_clear.prompt
    timeline = hub.record.timeline(rf_after_clear.transcript_id)
    assert timeline.available and timeline.data


def test_a_refused_argument_answers_as_the_service_400(hub):
    answer = hub.record.timeline("not a token")
    assert not answer.available and answer.refused


# --- the chat child -----------------------------------------------------------------


def test_a_turn_after_clear_appends_the_ledger_and_owns_its_artifact(hub, client):
    key = str(uuid.uuid4())
    _chat(client, key, "Read the storage ring current.")
    _chat(client, key, "/clear")
    _chat(client, key, "Now read the lifetime.")

    first, cleared, after = hub.chat.turns[-3:]
    assert first.transcript_id == key
    assert cleared.transcript_id != key and cleared.artifact_id is None
    assert after.transcript_id == cleared.transcript_id

    ledger = session_ledger.load(session_meta.ledger_state_dir())
    assert [row.transcript_id for row in ledger.rows if row.session_key == key] == [
        key,
        cleared.transcript_id,
    ]
    store = ArtifactStore(workspace_root=hub.project_dir, auto_launch=False)
    entry = store.get_entry(after.artifact_id)
    assert entry is not None and entry.session_id == key
    assert owner_of(entry, ledger) == cleared.transcript_id
    assert owner_of(store.get_entry(first.artifact_id), ledger) == key

    rows = _sessions(client)
    assert rows[cleared.transcript_id]["session_key"] == key
    assert rows[cleared.transcript_id]["current_for_key"] is True
    assert rows[cleared.transcript_id]["message_count"] == 2
    assert rows[key]["current_for_key"] is False


def test_the_chat_child_reads_the_prompts_uploads(hub, client):
    upload = hub.project_dir / "notes.txt"
    upload.write_text("BPM offsets\n", encoding="utf-8")
    key = str(uuid.uuid4())
    body = _chat(client, key, f"Use the attached file.\n\n- {upload} (12 B)")
    assert "Read notes.txt (12 bytes)." in json.dumps(body)
    assert hub.chat.turns[-1].uploads == (str(upload),)


# --- the naming seam ----------------------------------------------------------------


def _named(frames: list[dict], transcript_id: str) -> list[dict]:
    return [
        f
        for f in frames
        if f.get("type") == "session_named" and f.get("transcript_id") == transcript_id
    ]


def test_a_stubbed_name_arrives_as_session_named(hub, client):
    target = SEEDED_SESSIONS[2]
    hub.naming.titles[target.prompt] = "Injection beam profile"
    hub.naming.delay_s = 0.2
    frames: list[dict] = []
    opened = threading.Event()
    stop = threading.Event()

    def listen() -> None:
        with (
            _http(hub.base_url) as c,
            c.stream("GET", "/api/files/events") as stream,
        ):
            opened.set()
            for line in stream.iter_lines():
                if line.startswith("data: "):
                    frames.append(json.loads(line[len("data: ") :]))
                if stop.is_set() or _named(frames, target.transcript_id):
                    return

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()
    assert opened.wait(_TIMEOUT)
    time.sleep(0.2)  # the stream subscribes once its response starts

    started = time.monotonic()
    rows = _sessions(client)
    # The listing never waits on the model: the stub is still sleeping.
    assert time.monotonic() - started < hub.naming.delay_s + 5
    assert rows[target.transcript_id]["naming"] in ("pending", "done")

    listener.join(_TIMEOUT)
    stop.set()
    assert _named(frames, target.transcript_id) == [
        {
            "type": "session_named",
            "transcript_id": target.transcript_id,
            "title": "Injection beam profile",
            "naming": "done",
        }
    ]
    assert target.prompt in hub.naming.calls
    assert _sessions(client)[target.transcript_id]["title"] == "Injection beam profile"


# --- the extension type -------------------------------------------------------------


def test_the_beam_profile_stub_viewer_is_served(hub):
    with _http(hub.artifact_url) as c:
        viewer = c.get(f"/viewers/{BEAM_PROFILE_KIND}.js")
        registry = c.get("/api/type-registry").json()
        artifacts = c.get("/api/artifacts").json()["artifacts"]
    assert viewer.status_code == 200
    assert viewer.headers["content-type"].startswith("text/javascript")
    assert "export function render" in viewer.text
    assert BEAM_PROFILE_MARKER_CLASS in viewer.text and "ctx.theme" in viewer.text
    declared = {ext["key"]: ext for ext in registry["extension_types"]}
    assert declared[BEAM_PROFILE_KIND]["viewer_url"] == f"/viewers/{BEAM_PROFILE_KIND}.js"
    assert any(a.get("kind") == BEAM_PROFILE_KIND for a in artifacts)


# --- the capture hub ----------------------------------------------------------------


def test_the_capture_hub_reads_the_demo_session_through_the_record():
    with hermetic_hub() as plain:
        _write_fake_session(plain.session_dir)
        answer = plain.record.list_sessions()
        assert answer.available
        rows = {row["session_id"]: row for row in answer.data}
        assert rows[DEMO_SESSION_ID]["prompt"] == DEMO_OPENING_PROMPT
        timeline = plain.record.timeline(DEMO_SESSION_ID)
        assert timeline.available and timeline.data
        # The seeds are the e2e suite's alone.
        assert not any(s.transcript_id in rows for s in SEEDED_SESSIONS)
