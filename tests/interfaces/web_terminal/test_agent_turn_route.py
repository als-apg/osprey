"""Tests for the turn-state route (`routes/agent_turn.py`).

``POST /api/agent-turn`` is the Claude Code hook's report of a turn edge::

    request:  {"session_id": str, "pool_key": str, "state": "busy"|"idle",
               "surface": str, "ts": float, "source"?: str}
    response: {"ok": true, "recorded": bool}

Two things must come out of an accepted report: the session key's busy/idle
state on ``app.state.turn_state``, and the transcript the key currently points
at, recorded in the persisted transcript map. The key is ``pool_key`` — the
identity that never changes — while ``session_id`` is the transcript, which a
``/clear`` moves.

Anything reported for another surface is dropped rather than refused, because
a hook cannot act on a 4xx and a refusal would surface as noise in the
operator's terminal.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import WebAuthMiddleware
from osprey.interfaces.web_auth import WebCredentials, reset_web_credentials
from osprey.interfaces.web_terminal import transcript_map
from osprey.interfaces.web_terminal.routes.agent_turn import router
from osprey.interfaces.web_terminal.turn_state import get_turn_state, reset_turn_state
from osprey_connectors import session_store

#: The session key: the PTY pool key, the posture key, the audit key.
KEY = "aaaaaaaa-1111-2222-3333-444444444444"
#: The transcript a ``/clear`` moved that key to — a different id, same key.
TRANSCRIPT = "cccccccc-1111-2222-3333-444444444444"

PANEL_TOKEN = "panel-token-value"
OPERATOR_SECRET = "operator-secret-value"


@pytest.fixture(autouse=True)
def shared_root(tmp_path, monkeypatch):
    """Pin the agent-data root so the transcript map has somewhere to write.

    The same derivation the posture store uses, exercised rather than patched
    around, so "persisted" in these tests means a file on disk. Autouse
    because every accepted report touches the map: without a pinned root the
    resolution walks up to the repository and the suite's leak guard fires.
    """
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    session_store.invalidate_cache()
    yield root
    session_store.invalidate_cache()


def _make_client() -> TestClient:
    """A minimal app exposing the turn-state router, with the store installed."""
    app = FastAPI()
    app.include_router(router)
    app.state.turn_state = {}
    return TestClient(app)


def _post(client: TestClient, **overrides) -> dict:
    """POST a hook-shaped report, returning the parsed body."""
    body = {
        "session_id": KEY,
        "pool_key": KEY,
        "state": "busy",
        "surface": "expert",
        "ts": 1000.0,
    }
    body.update(overrides)
    resp = client.post("/api/agent-turn", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---- The surface filter ----


@pytest.mark.parametrize("surface", ["simple", "", "Expert", "unknown"])
def test_reports_from_other_surfaces_are_dropped_not_refused(surface):
    """Only ``expert`` is recorded; everything else is a 200 that changed nothing.

    The empty case is the one that actually happens: a hook whose surface
    environment variable is unset. It must not 4xx into the operator's
    terminal, and it must not be mistaken for an expert-view report either.
    """
    client = _make_client()
    assert _post(client, surface=surface) == {"ok": True, "recorded": False}
    assert client.app.state.turn_state == {}


def test_expert_reports_are_recorded():
    """The one accepted surface writes the entry and says so."""
    client = _make_client()
    assert _post(client, surface="expert") == {"ok": True, "recorded": True}
    assert client.app.state.turn_state[KEY]["state"] == "busy"


# ---- The two states, including the failure-shaped idle ----


@pytest.mark.parametrize(
    "source",
    ["Stop", "StopFailure", "SessionStart", None],
)
def test_idle_is_recorded_whatever_hook_event_reported_it(source):
    """A turn that ended badly is still a turn that ended.

    ``StopFailure`` fires where ``Stop`` does not, and the server must read
    both as idle — a key left busy because its turn failed would never release
    for the other view. ``source`` is context, never a condition.
    """
    client = _make_client()
    body = {"state": "idle"}
    if source is not None:
        body["source"] = source
    assert _post(client, **body)["recorded"] is True
    assert client.app.state.turn_state[KEY]["state"] == "idle"


def test_an_unknown_state_is_refused():
    """The one field the hook cannot get wrong quietly."""
    client = _make_client()
    resp = client.post(
        "/api/agent-turn",
        json={"session_id": KEY, "pool_key": KEY, "state": "thinking", "surface": "expert"},
    )
    assert resp.status_code == 422
    assert client.app.state.turn_state == {}


# ---- Keying: the pool key, not the transcript ----


def test_entry_is_keyed_on_the_pool_key_not_the_session_id():
    """A cleared session reports a new transcript under the unchanged key."""
    client = _make_client()
    _post(client, session_id=TRANSCRIPT, pool_key=KEY, state="idle", ts=1234.5)

    assert set(client.app.state.turn_state) == {KEY}
    assert client.app.state.turn_state[KEY] == {
        "state": "idle",
        "ts": 1234.5,
        "transcript_id": TRANSCRIPT,
    }


def test_a_missing_pool_key_falls_back_to_the_session_id():
    """Before anything has cleared, the two identities are the same string."""
    client = _make_client()
    _post(client, session_id=KEY, pool_key="")
    assert set(client.app.state.turn_state) == {KEY}
    assert client.app.state.turn_state[KEY]["transcript_id"] == KEY


def test_a_report_with_no_identity_at_all_is_dropped():
    """An empty id is not a session UUID, so there is nothing to record."""
    client = _make_client()
    assert _post(client, session_id="", pool_key="") == {"ok": True, "recorded": False}
    assert client.app.state.turn_state == {}


# ---- The identifier grammar ----


@pytest.mark.parametrize(
    "session_id",
    [
        "../../etc/passwd",
        f"../../{KEY}",
        f"{KEY}/../../etc/passwd",
        f"{KEY}.jsonl",
        KEY.upper(),
        "operator-abcd1234",
        "not-a-uuid",
    ],
)
def test_a_non_canonical_session_id_is_dropped(session_id):
    """The transcript id is persisted and later reaches a resume argv and a path.

    A traversal-shaped id must never get that far, and the refusal is a drop
    rather than a 4xx for the same reason every other drop here is: the hook
    that sent it cannot act on an error.
    """
    client = _make_client()
    assert _post(client, session_id=session_id, pool_key="") == {"ok": True, "recorded": False}
    assert client.app.state.turn_state == {}
    assert transcript_map.store_path().exists() is False


def test_a_non_canonical_pool_key_is_dropped():
    """The key names an entry in a store on disk, so it is closed too."""
    client = _make_client()
    assert _post(client, session_id=KEY, pool_key="../../elsewhere") == {
        "ok": True,
        "recorded": False,
    }
    assert client.app.state.turn_state == {}


def test_a_canonical_pair_is_still_recorded():
    """The grammar closes on malformed ids without costing a legitimate one."""
    client = _make_client()
    assert _post(client, session_id=TRANSCRIPT, pool_key=KEY)["recorded"] is True
    assert client.app.state.turn_state[KEY]["transcript_id"] == TRANSCRIPT


def test_a_later_report_replaces_the_entry_for_its_key():
    """The store holds the latest edge per key, not a history."""
    client = _make_client()
    _post(client, state="busy", ts=10.0)
    _post(client, state="idle", ts=20.0)
    assert client.app.state.turn_state[KEY] == {
        "state": "idle",
        "ts": 20.0,
        "transcript_id": KEY,
    }


def test_the_server_stamps_the_time_when_the_hook_sends_none():
    """A hook that omits ``ts`` still produces a fully shaped entry."""
    client = _make_client()
    client.post(
        "/api/agent-turn",
        json={"session_id": KEY, "pool_key": KEY, "state": "busy", "surface": "expert"},
    )
    assert isinstance(client.app.state.turn_state[KEY]["ts"], float)


def test_a_future_timestamp_is_clamped_to_now():
    """A stored timestamp outranks other evidence, so it cannot run ahead.

    The hook shares the server's clock. A ``ts`` in the future is a broken or
    hostile report, and left as sent it would make this entry permanently the
    freshest thing any reader has.
    """
    before = time.time()
    client = _make_client()
    _post(client, state="idle", ts=before + 10_000.0)
    after = time.time()

    stored = client.app.state.turn_state[KEY]["ts"]
    assert before <= stored <= after


def test_a_past_timestamp_is_kept_as_sent():
    """Clamping is one-sided: the hook's own reading of when the edge happened."""
    client = _make_client()
    _post(client, state="idle", ts=1234.5)
    assert client.app.state.turn_state[KEY]["ts"] == 1234.5


# ---- The transcript id: recorded and persisted ----


def test_a_diverged_transcript_is_recorded_and_persisted(shared_root):
    """``session_id != pool_key`` moves the key's mapping, on disk as well."""
    client = _make_client()
    _post(client, session_id=TRANSCRIPT, pool_key=KEY, state="idle")

    assert transcript_map.get(client.app, KEY) == TRANSCRIPT
    written = json.loads(transcript_map.store_path().read_text(encoding="utf-8"))
    assert written == {KEY: TRANSCRIPT}


def test_a_transcript_equal_to_the_key_clears_a_stale_mapping(shared_root):
    """Returning to the key's own transcript must not leave the old pointer.

    The map stores only real divergence, so an id equal to the key removes the
    entry rather than storing an identity mapping.
    """
    client = _make_client()
    _post(client, session_id=TRANSCRIPT, pool_key=KEY, state="idle")
    _post(client, session_id=KEY, pool_key=KEY, state="idle")

    assert transcript_map.get(client.app, KEY) == KEY
    assert json.loads(transcript_map.store_path().read_text(encoding="utf-8")) == {}


def test_a_dropped_report_never_touches_the_map(shared_root):
    """The surface filter gates the map write too, not only the state write."""
    client = _make_client()
    _post(client, session_id=TRANSCRIPT, pool_key=KEY, surface="simple")
    assert transcript_map.get(client.app, KEY) == KEY


# ---- The helpers spawn and teardown call ----


def test_reset_turn_state_marks_a_key_idle():
    """A key that died mid-turn must not read as busy after a respawn."""
    app = SimpleNamespace(state=SimpleNamespace(turn_state={}))
    app.state.turn_state[KEY] = {"state": "busy", "ts": 1.0, "transcript_id": TRANSCRIPT}

    reset_turn_state(app, KEY)

    entry = get_turn_state(app, KEY)
    assert entry["state"] == "idle"
    assert isinstance(entry["ts"], float)


def test_reset_turn_state_keeps_the_key_pointed_at_its_transcript(shared_root):
    """The reset clears the turn, not the conversation the key is holding."""
    app = SimpleNamespace(state=SimpleNamespace())
    transcript_map.set(app, KEY, TRANSCRIPT)

    reset_turn_state(app, KEY)

    assert get_turn_state(app, KEY)["transcript_id"] == TRANSCRIPT


def test_reset_turn_state_ignores_an_empty_key():
    """Callers hand over whatever key they hold; an absent one is not an entry."""
    app = SimpleNamespace(state=SimpleNamespace(turn_state={}))
    reset_turn_state(app, "")
    assert app.state.turn_state == {}


def test_get_turn_state_is_none_when_nothing_was_reported():
    """Unknown is not idle: a caller waiting for idleness must see the difference."""
    app = SimpleNamespace(state=SimpleNamespace(turn_state={}))
    assert get_turn_state(app, KEY) is None
    assert get_turn_state(app, "") is None


# ---- The credential the hook actually carries ----


@pytest.fixture
def panel_token_client(shared_root):
    """The router behind the real auth middleware, with a known panel token.

    Every test using it carries ``no_auth_seam``: the suite-wide seam injects
    the operator secret into any client aimed at a gated app, which would admit
    these requests on a credential the hook does not have.
    """
    reset_web_credentials()
    app = FastAPI()
    app.include_router(router)
    app.state.turn_state = {}
    app.state.web_credentials = WebCredentials(
        operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
    )
    app.add_middleware(WebAuthMiddleware, cookie_name="osprey_terminal_session_8080")
    with TestClient(app, client=("127.0.0.1", 54321)) as client:
        yield client
    reset_web_credentials()


@pytest.mark.no_auth_seam
def test_the_panel_token_alone_reaches_the_route(panel_token_client):
    """The hook runs in the agent's own sandbox and carries nothing stronger."""
    resp = panel_token_client.post(
        "/api/agent-turn",
        json={"session_id": KEY, "pool_key": KEY, "state": "idle", "surface": "expert"},
        headers={"authorization": f"Bearer {PANEL_TOKEN}"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "recorded": True}


@pytest.mark.no_auth_seam
def test_a_wrong_panel_token_is_refused(panel_token_client):
    """The route is panel-tier, not open."""
    resp = panel_token_client.post(
        "/api/agent-turn",
        json={"session_id": KEY, "pool_key": KEY, "state": "idle", "surface": "expert"},
        headers={"authorization": "Bearer not-the-token"},
    )
    assert resp.status_code in (401, 403)
    assert panel_token_client.app.state.turn_state == {}
