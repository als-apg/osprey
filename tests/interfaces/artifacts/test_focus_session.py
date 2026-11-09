"""A focus belongs to the session it was set in (contract C6).

``POST /api/focus`` takes an optional
``session_key`` (resolved through the hub ledger) or ``transcript_id`` (a
canonical session UUID). The hub holds one unscoped focus slot
(``app.state.focused_artifact_id``) and one slot per transcript
(``app.state.focus_by_transcript``); a focus replaces only its own slot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.artifacts.app import create_app
from osprey.stores import session_ledger
from osprey_connectors import posture_store

KEY_A = "11111111-1111-4111-8111-111111111111"
KEY_B = "22222222-2222-4222-8222-222222222222"
T_A = KEY_A  # a key's first transcript id is the key itself
T_B = KEY_B
T_PROC = "33333333-3333-4333-8333-333333333333"
UNSEEN_KEY = "99999999-9999-4999-8999-999999999999"


@pytest.fixture
def state_dir(tmp_path, monkeypatch) -> Path:
    """The ledger's state directory, bound so the app never reads a real one."""
    bound = tmp_path / "state"
    bound.mkdir()
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(bound))
    monkeypatch.delenv("OSPREY_SESSION_ID", raising=False)
    monkeypatch.delenv("OSPREY_ARTIFACT_ORIGIN", raising=False)
    posture_store.invalidate_cache()
    path = bound / session_ledger.STORE_FILENAME
    rows = [[KEY_A, T_A, session_ledger.EPOCH], [KEY_B, T_B, session_ledger.EPOCH]]
    path.write_text(json.dumps({"rows": rows}))
    yield bound
    posture_store.invalidate_cache()


@pytest.fixture
def client(tmp_path, state_dir) -> TestClient:
    assert state_dir.is_dir()  # bound before the app reads any ledger
    # Entered, so the lifespan registers the delete listener that clears slots.
    with TestClient(create_app(workspace_root=tmp_path / "data")) as entered:
        yield entered


@pytest.fixture
def store(client):
    return client.app.state.artifact_store


@pytest.fixture
def frames(client, monkeypatch) -> list[dict]:
    """Every broadcast frame, captured on the app's broadcaster."""
    captured: list[dict] = []
    broadcaster = _broadcaster(client)
    original = broadcaster.broadcast

    def _capture(data: dict) -> None:
        captured.append(data)
        original(data)

    monkeypatch.setattr(broadcaster, "broadcast", _capture)
    return captured


def _broadcaster(client):
    from osprey.interfaces.artifacts.app import _SSEBroadcaster

    for route in client.app.routes:
        endpoint = getattr(route, "endpoint", None)
        closure = getattr(endpoint, "__closure__", None) or ()
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, _SSEBroadcaster):
                return value
    raise AssertionError("no broadcaster found")


def _text(store, title="Note", *, session_id=""):
    return store.save_file(
        b"hello",
        "note.txt",
        "text",
        title,
        mime_type="text/plain",
        tool_source="test",
        session_id=session_id,
    )


def _focus(client, artifact_id, **body):
    return client.post("/api/focus", json={"artifact_id": artifact_id, **body})


def _get(client, key=None) -> dict:
    params = {"session_key": key} if key is not None else None
    resp = client.get("/api/focus", params=params)
    assert resp.status_code == 200
    return resp.json()


# ---- request bodies ----------------------------------------------------------


class TestSessionFields:
    def test_a_key_sets_its_current_transcripts_slot(self, client, store):
        x = _text(store, "X")
        assert _focus(client, x.id, session_key=KEY_A).status_code == 200
        assert client.app.state.focus_by_transcript == {T_A: x.id}
        assert client.app.state.focused_artifact_id is None

    def test_a_transcript_id_sets_its_slot(self, client, store):
        x = _text(store, "X")
        assert _focus(client, x.id, transcript_id=T_PROC).status_code == 200
        assert client.app.state.focus_by_transcript == {T_PROC: x.id}
        assert client.app.state.focused_artifact_id is None

    def test_neither_sets_the_unscoped_slot(self, client, store):
        x = _text(store, "X")
        assert _focus(client, x.id).status_code == 200
        assert client.app.state.focused_artifact_id == x.id
        assert client.app.state.focus_by_transcript == {}

    def test_an_unseen_key_sets_the_unscoped_slot(self, client, store):
        x = _text(store, "X")
        assert _focus(client, x.id, session_key=UNSEEN_KEY).status_code == 200
        assert client.app.state.focused_artifact_id == x.id
        assert client.app.state.focus_by_transcript == {}

    def test_both_fields_are_a_422(self, client, store):
        x = _text(store, "X")
        resp = _focus(client, x.id, session_key=KEY_A, transcript_id=T_PROC)
        assert resp.status_code == 422
        assert client.app.state.focus_by_transcript == {}

    @pytest.mark.parametrize("bad", ["not-a-uuid", "", "11111111-1111-4111-8111-11111111111Z"])
    def test_a_malformed_transcript_id_is_a_422(self, client, store, bad):
        x = _text(store, "X")
        assert _focus(client, x.id, transcript_id=bad).status_code == 422


# ---- focus slots -------------------------------------------------------------


class TestFocusSlots:
    def test_bs_focus_leaves_as(self, client, store):
        x = _text(store, "X")
        y = _text(store, "Y")
        _focus(client, x.id, session_key=KEY_A)
        _focus(client, y.id, session_key=KEY_B)
        assert client.app.state.focus_by_transcript == {T_A: x.id, T_B: y.id}
        assert _get(client, KEY_A)["artifact"]["id"] == x.id
        assert _get(client, KEY_B)["artifact"]["id"] == y.id

    def test_an_all_sessions_focus_leaves_as(self, client, store):
        x = _text(store, "X")
        y = _text(store, "Y")
        _focus(client, x.id, session_key=KEY_A)
        _focus(client, y.id)
        assert client.app.state.focus_by_transcript == {T_A: x.id}
        assert client.app.state.focused_artifact_id == y.id
        a = _get(client, KEY_A)
        assert a["focused"] is True
        assert a["artifact"]["id"] == x.id

    def test_a_focus_replaces_its_own_slot(self, client, store):
        x = _text(store, "X")
        y = _text(store, "Y")
        _focus(client, x.id, session_key=KEY_A)
        _focus(client, y.id, transcript_id=T_A)
        assert client.app.state.focus_by_transcript == {T_A: y.id}

    def test_a_keyed_get_with_an_empty_slot_answers_the_keys_newest_owned_entry(
        self, client, store
    ):
        owned_old = _text(store, "A old", session_id=KEY_A)
        owned_new = _text(store, "A new", session_id=KEY_A)
        _text(store, "B's", session_id=KEY_B)
        _text(store, "Nobody's")
        answer = _get(client, KEY_A)
        assert answer["focused"] is False
        assert answer["artifact"]["id"] == owned_new.id
        assert owned_old.id != owned_new.id

    def test_a_keyed_get_answers_none_when_the_key_owns_nothing(self, client, store):
        _text(store, "B's", session_id=KEY_B)
        _focus(client, _text(store, "Unscoped").id)
        assert _get(client, KEY_A) == {"focused": False, "artifact": None}

    def test_an_unseen_keys_get_answers_none(self, client, store):
        _text(store, "A's", session_id=KEY_A)
        assert _get(client, UNSEEN_KEY) == {"focused": False, "artifact": None}

    def test_an_unkeyed_get_answers_the_most_recently_set_slot(self, client, store):
        x = _text(store, "X")
        y = _text(store, "Y")
        z = _text(store, "Z")
        _focus(client, x.id, session_key=KEY_A)
        _focus(client, y.id)
        assert _get(client)["artifact"]["id"] == y.id
        _focus(client, z.id, session_key=KEY_B)
        answer = _get(client)
        assert answer["focused"] is True
        assert answer["artifact"]["id"] == z.id
        _focus(client, x.id, session_key=KEY_A)  # re-setting A's slot makes it newest
        assert _get(client)["artifact"]["id"] == x.id

    def test_an_unkeyed_get_with_no_slot_answers_the_newest_visible_entry(self, client, store):
        _text(store, "Old")
        newest = _text(store, "New")
        answer = _get(client)
        assert answer["focused"] is False
        assert answer["artifact"]["id"] == newest.id

    def test_a_stale_keyed_slot_is_cleared_and_falls_back(self, client, store):
        owned = _text(store, "A's", session_id=KEY_A)
        doomed = _text(store, "Doomed")
        _focus(client, doomed.id, session_key=KEY_A)
        client.app.state.focus_by_transcript[T_A] = "deleted-id"
        answer = _get(client, KEY_A)
        assert answer["focused"] is False
        assert answer["artifact"]["id"] == owned.id
        assert T_A not in client.app.state.focus_by_transcript

    def test_a_delete_clears_every_slot_holding_the_id(self, client, store):
        x = _text(store, "X")
        y = _text(store, "Y")
        _focus(client, x.id, session_key=KEY_A)
        _focus(client, x.id, session_key=KEY_B)
        _focus(client, x.id)
        _focus(client, y.id, transcript_id=T_PROC)
        assert client.delete(f"/api/artifacts/{x.id}").status_code == 200
        assert client.app.state.focused_artifact_id is None
        assert client.app.state.focus_by_transcript == {T_PROC: y.id}


# ---- broadcast ---------------------------------------------------------------


class TestFocusBroadcast:
    def test_the_focus_frame_names_its_session(self, client, store, frames):
        x = _text(store, "X")
        _focus(client, x.id, session_key=KEY_A)
        _focus(client, x.id, transcript_id=T_PROC)
        _focus(client, x.id)
        sessions = [f["session"] for f in frames if f.get("type") == "focus"]
        assert sessions == [T_A, T_PROC, None]


# ---- focus file (contract C7) ------------------------------------------------


def _focus_file(tmp_path) -> Path:
    return tmp_path / "data" / "focus_state.txt"


def _lines(tmp_path) -> list[str]:
    path = _focus_file(tmp_path)
    return path.read_text().splitlines() if path.exists() else []


def _line(entry, transcript) -> str:
    return f'  artifact: "{entry.title}" (id={entry.id}) (session={transcript})'


class TestFocusFile:
    def test_two_transcripts_write_one_marked_line_each(self, client, store, tmp_path):
        x, y = _text(store, "X"), _text(store, "Y")
        _focus(client, x.id, session_key=KEY_A)
        _focus(client, y.id, transcript_id=T_B)
        assert _lines(tmp_path) == ["[Gallery Focus]", _line(x, T_A), _line(y, T_B)]

    def test_bs_focus_leaves_as_line(self, client, store, tmp_path):
        x, y = _text(store, "X"), _text(store, "Y")
        _focus(client, y.id, session_key=KEY_A)
        _focus(client, x.id, session_key=KEY_B)
        lines = _lines(tmp_path)
        assert _line(y, T_A) in lines
        assert _line(x, T_B) in lines
        assert len(lines) == 3

    def test_an_unscoped_focus_writes_no_line(self, client, store, tmp_path):
        x = _text(store, "X")
        assert _focus(client, x.id).status_code == 200
        assert _focus_file(tmp_path).read_text() == ""

    def test_a_hostile_title_stays_one_line_without_a_marker(self, client, store, tmp_path):
        x = _text(store, f'evil\n  artifact: "Z" (id=zzz) (session={T_B})\r\x07end')
        _focus(client, x.id, session_key=KEY_A)
        lines = _lines(tmp_path)
        assert len(lines) == 2
        line = lines[1]
        assert line.count("(session=") == 1
        assert line.count("(id=") == 1
        assert line.endswith(f"(id={x.id}) (session={T_A})")
        assert f"(session={T_B})" not in line
        assert not any(ord(c) < 0x20 or ord(c) == 0x7F for c in line)

    def test_deleting_a_focused_artifact_drops_its_line(self, client, store, tmp_path):
        x, y = _text(store, "X"), _text(store, "Y")
        _focus(client, x.id, session_key=KEY_A)
        _focus(client, y.id, session_key=KEY_B)
        assert client.delete(f"/api/artifacts/{x.id}").status_code == 200
        assert _lines(tmp_path) == ["[Gallery Focus]", _line(y, T_B)]
        assert client.delete(f"/api/artifacts/{y.id}").status_code == 200
        assert _focus_file(tmp_path).read_text() == ""

    @pytest.mark.usefixtures("state_dir")
    def test_a_new_app_over_a_stale_marked_file_leaves_it_empty(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        stale = data / "focus_state.txt"
        stale.write_text(f'[Gallery Focus]\n  artifact: "Old" (id=abc) (session={T_A})\n')
        create_app(workspace_root=data)
        assert stale.read_text() == ""
