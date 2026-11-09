"""``GET /api/sessions`` and the per-session endpoints, against a fake record.

Pins the contract rows: the record's rows unioned with the ledger's transcripts
the record has not ingested, Decision 6's drops applied before anything is
named (dispatch by ``run_ids``, host/dispatch/watchdog terminal sessions,
promptless rows unless ``truncated``), ``session_key``/``current_for_key`` from
the ledger across two ``/clear``-s, title precedence user > auto > shortened
first prompt > "Untitled session", ``naming`` and ``state``; the record states
(capped with ``more``/``next_cursor``, incomplete, unavailable with the ledger's
transcripts still listed, and a refused ``before`` as incomplete); only
``drain=1`` waits on a drain. Then the title, pin, last-artifact, last-active and
``enter`` endpoints: rename and revert, PUTs for a just-minted key and after a
restart, ``enter`` then a spawn of the new key resuming the transcript, and a
non-UUID id answered 422.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.agent_record.client import ClientAnswer
from osprey.agent_record.contract import Drain
from osprey.build.claude_code_telemetry import (
    DISPATCH_SESSION_PREFIX,
    HOST_SESSION_PREFIX,
    WATCHDOG_SESSION_PREFIX,
)
from osprey.interfaces.web_terminal import session_handoff, session_meta, transcript_map
from osprey.interfaces.web_terminal.routes import session as session_routes
from osprey.stores import session_ledger
from osprey_connectors import posture_store
from tests.agent_record.fakes import FakeRecordClient

ME = "alice"
KEY = "aaaaaaaa-1111-4222-8333-444444444444"
T1 = KEY  # a key's first transcript is the key itself
T2 = "bbbbbbbb-1111-4222-8333-444444444444"
T3 = "cccccccc-1111-4222-8333-444444444444"
OTHER = "dddddddd-1111-4222-8333-444444444444"
FRESH = "eeeeeeee-1111-4222-8333-444444444444"


class RecordingBroadcaster:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    def broadcast(self, data: dict, *, reliable: bool = False) -> None:  # noqa: ARG002
        self.frames.append(data)


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
def record(monkeypatch):
    fake = FakeRecordClient()
    monkeypatch.setattr(session_routes, "record_client", lambda: fake)
    return fake


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(session_routes.router)
    app.state.terminal_user = ME
    app.state.broadcaster = RecordingBroadcaster()
    return app


@pytest.fixture
def client(state_root, record):  # noqa: ARG001
    return TestClient(_app())


def _row(session_id, last, prompt="hello", **fields):
    row = {
        "session_id": session_id,
        "persons": [ME],
        "identities": [ME],
        "first_ts": f"2026-11-01T09:{last:02d}:00+00:00",
        "last_ts": f"2026-11-01T10:{last:02d}:00+00:00",
        "message_count": 4,
        "prompt": prompt,
        "truncated": False,
        "run_ids": [],
        "terminal_session": "",
    }
    row.update(fields)
    return row


def _answer(rows, **kw) -> ClientAnswer:
    return ClientAnswer(data=rows, complete=kw.pop("complete", True), **kw)


def _listing(record, *pages: ClientAnswer, known: tuple[str, ...] = ()) -> None:
    """``list_sessions`` answers *pages* in order to the listing reads; a
    ``session_ids`` read (which ledger transcripts the record holds) answers
    the named ones among *known* and the pages' rows."""
    held = set(known) | {
        r["session_id"] for page in pages if isinstance(page.data, list) for r in page.data
    }
    served = []

    def answer(**kwargs):
        ids = kwargs.get("session_ids")
        if ids:
            return _answer([{"session_id": s} for s in sorted(ids) if s in held])
        served.append(kwargs)
        return pages[min(len(served), len(pages)) - 1]

    record.answers["list_sessions"] = answer


def _pages(record) -> list[dict]:
    """The keyword arguments of the listing reads (not the ``session_ids`` ones)."""
    return [c for c in record.calls_to("list_sessions") if not c.get("session_ids")]


def _ledger(*rows: tuple[str, str, str]) -> None:
    state = session_meta.ledger_state_dir()
    for key, tid, ts in rows:
        assert session_ledger.append(state, key, tid, ts)


def _by_id(body) -> dict[str, dict]:
    return {row["transcript_id"]: row for row in body["sessions"]}


# --- rows ------------------------------------------------------------------------------


class TestRows:
    def test_session_key_and_current_for_key_across_two_clears(self, client, record):
        _ledger(
            (KEY, T1, "2026-11-01T09:00:00+00:00"),
            (KEY, T2, "2026-11-01T09:20:00+00:00"),
            (KEY, T3, "2026-11-01T09:40:00+00:00"),
        )
        record.answers["list_sessions"] = _answer([_row(T3, 50), _row(T2, 30), _row(T1, 10)])
        body = client.get("/api/sessions").json()
        rows = body["sessions"]
        assert [r["transcript_id"] for r in rows] == [T3, T2, T1]
        assert all(r["session_key"] == KEY for r in rows)
        assert [r["current_for_key"] for r in rows] == [True, False, False]
        assert body["record"] == "complete" and body["more"] is False
        assert body["next_cursor"] is None
        assert rows[0]["message_count"] == 4
        assert rows[0]["first_ts"] == "2026-11-01T09:50:00+00:00"
        assert rows[0]["last_ts"] == "2026-11-01T10:50:00+00:00"
        assert rows[0]["state"] == "idle" and rows[0]["naming"] is None

    def test_a_transcript_the_ledger_never_saw_has_no_key(self, client, record):
        record.answers["list_sessions"] = _answer([_row(OTHER, 5)])
        row = client.get("/api/sessions").json()["sessions"][0]
        assert row["session_key"] is None and row["current_for_key"] is False

    def test_an_uningested_ledger_transcript_is_listed_untitled(self, client, record):
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"), (FRESH, FRESH, "2026-11-01T11:00:00+00:00"))
        record.answers["list_sessions"] = _answer([_row(T1, 10, prompt="tune the ring")])
        rows = client.get("/api/sessions").json()["sessions"]
        assert [r["transcript_id"] for r in rows] == [FRESH, T1]
        fresh = rows[0]
        assert fresh["title"] == "Untitled session" and fresh["title_source"] is None
        assert fresh["session_key"] == FRESH and fresh["current_for_key"] is True
        assert fresh["message_count"] == 0 and fresh["last_ts"] is None

    def test_dispatch_host_watchdog_and_promptless_rows_are_absent(self, client, record):
        rows = [
            _row(T1, 50),
            _row("10000000-0000-4000-8000-000000000001", 49, run_ids=["run-1"]),
            _row(
                "10000000-0000-4000-8000-000000000002",
                48,
                terminal_session=f"{DISPATCH_SESSION_PREFIX}x",
            ),
            _row(
                "10000000-0000-4000-8000-000000000003",
                47,
                terminal_session=f"{HOST_SESSION_PREFIX}x",
            ),
            _row(
                "10000000-0000-4000-8000-000000000004",
                46,
                terminal_session=f"{WATCHDOG_SESSION_PREFIX}x",
            ),
            _row("10000000-0000-4000-8000-000000000005", 45, prompt=None),
            _row("10000000-0000-4000-8000-000000000006", 44, prompt=None, truncated=True),
            _row("10000000-0000-4000-8000-000000000007", 43, identities=["bob"], persons=["bob"]),
        ]
        record.answers["list_sessions"] = _answer(rows)
        body = client.get("/api/sessions").json()
        assert [r["transcript_id"] for r in body["sessions"]] == [
            T1,
            "10000000-0000-4000-8000-000000000006",
        ]
        kept = body["sessions"][1]
        assert kept["truncated"] is True and kept["title"] == "Untitled session"

    def test_live_when_the_keys_child_runs_and_the_row_is_current(self, state_root, record):  # noqa: ARG002
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"), (KEY, T2, "2026-11-01T09:20:00+00:00"))
        record.answers["list_sessions"] = _answer([_row(T2, 30), _row(T1, 10)])
        app = _app()
        app.state.pty_registry = SimpleNamespace(
            session_keys=lambda: [KEY],
            get_session=lambda key: SimpleNamespace(is_alive=key == KEY),
        )
        rows = _by_id(TestClient(app).get("/api/sessions").json())
        assert rows[T2]["state"] == "live" and rows[T1]["state"] == "idle"

        app = _app()
        app.state.operator_registry = SimpleNamespace(chats=SimpleNamespace(keys=lambda: [KEY]))
        rows = _by_id(TestClient(app).get("/api/sessions").json())
        assert rows[T2]["state"] == "live" and rows[T1]["state"] == "idle"


class TestTitles:
    def test_precedence_user_then_auto_then_shortened_prompt(self, client, record):
        long_prompt = "please read every BPM in the storage ring and compare them " * 3
        record.answers["list_sessions"] = _answer(
            [_row(T1, 30, prompt=long_prompt), _row(T2, 20), _row(T3, 10)]
        )
        app = client.app
        session_meta.set_auto_title(app, T2, "Auto named", "done")
        session_meta.set_auto_title(app, T3, "Auto named", "done")
        session_meta.set_title(app, T3, "Mine")
        app.state.session_naming_pending.add(T1)
        rows = _by_id(client.get("/api/sessions").json())
        assert rows[T3]["title"] == "Mine" and rows[T3]["title_source"] == "user"
        assert rows[T2]["title"] == "Auto named" and rows[T2]["title_source"] == "auto"
        assert rows[T2]["naming"] == "done"
        title = rows[T1]["title"]
        assert rows[T1]["title_source"] == "prompt" and rows[T1]["naming"] == "pending"
        assert len(title) <= session_routes.PROMPT_TITLE_CHARS and title.endswith("…")
        assert long_prompt.startswith(title[:-1].rstrip())

    def test_rename_and_revert(self, client, record):
        record.answers["list_sessions"] = _answer([_row(T1, 10, prompt="first words")])
        session_meta.set_auto_title(client.app, T1, "Auto", "done")
        resp = client.put(f"/api/sessions/{T1}/title", json={"title": "Renamed"})
        assert resp.status_code == 200 and resp.json()["title"] == "Renamed"
        assert client.get("/api/sessions").json()["sessions"][0]["title"] == "Renamed"
        client.put(f"/api/sessions/{T1}/title", json={"title": ""})
        row = client.get("/api/sessions").json()["sessions"][0]
        assert (row["title"], row["title_source"]) == ("Auto", "auto")

    def test_revert_without_an_auto_title_falls_back_to_the_prompt(self, client, record):
        record.answers["list_sessions"] = _answer([_row(T1, 10, prompt="first words")])
        client.put(f"/api/sessions/{T1}/title", json={"title": "Renamed"})
        client.put(f"/api/sessions/{T1}/title", json={"title": "  "})
        row = client.get("/api/sessions").json()["sessions"][0]
        assert (row["title"], row["title_source"]) == ("first words", "prompt")


# --- record states ---------------------------------------------------------------------


class TestRecordStates:
    def test_capped_carries_more_and_a_cursor_that_pages_back(self, client, record):
        _ledger(
            (KEY, T1, "2026-11-01T10:25:00+00:00"),  # inside page 1's window, not ingested
            (OTHER, OTHER, "2026-11-01T08:00:00+00:00"),  # older than page 1's window
        )
        _listing(
            record,
            _answer([_row(T2, 30), _row(T3, 20)], more=True),
            _answer([_row(FRESH, 5)]),
        )
        first = client.get("/api/sessions").json()
        assert first["more"] is True and first["record"] == "complete"
        assert [r["transcript_id"] for r in first["sessions"]] == [T2, T1, T3]
        second = client.get("/api/sessions", params={"cursor": first["next_cursor"]}).json()
        before = _pages(record)[1]["before"]
        assert before.isoformat() == "2026-11-01T10:20:00+00:00"
        assert second["more"] is False and second["next_cursor"] is None
        assert [r["transcript_id"] for r in second["sessions"]] == [FRESH, OTHER]

    def test_incomplete_keeps_its_rows_and_says_why(self, client, record):
        record.answers["list_sessions"] = _answer(
            [_row(T1, 10)], complete=False, reason="the read reached its scan limit of 9 rows"
        )
        body = client.get("/api/sessions").json()
        assert body["record"] == "incomplete" and "scan limit" in body["record_reason"]
        assert [r["transcript_id"] for r in body["sessions"]] == [T1]

    def test_unavailable_still_lists_the_ledgers_transcripts(self, client, record):
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"), (KEY, T2, "2026-11-01T09:20:00+00:00"))
        record.unavailable = "down"
        body = client.get("/api/sessions").json()
        assert body["record"] == "unavailable" and body["record_reason"] == "down"
        rows = body["sessions"]
        assert [r["transcript_id"] for r in rows] == [T2, T1]
        assert all(r["title"] == "Untitled session" for r in rows)
        assert rows[0]["current_for_key"] is True and body["more"] is False

    def test_a_refused_before_is_incomplete_not_unavailable(self, client, record):
        _ledger((KEY, T1, "2026-11-01T08:00:00+00:00"))  # older than page 1's window
        _listing(
            record,
            _answer([_row(T2, 30)], more=True),
            ClientAnswer.rejected(
                "agent-record refused the read (400): unknown parameter(s): before"
            ),
        )
        first = client.get("/api/sessions").json()
        body = client.get("/api/sessions", params={"cursor": first["next_cursor"]}).json()
        assert body["record"] == "incomplete"
        assert "unknown parameter(s): before" in body["record_reason"]
        assert body["sessions"] == [] and body["more"] is False

    def test_a_transcript_the_record_lists_is_not_relisted_from_the_ledger(self, client, record):
        # T2's first ledger row (the backfill's epoch) falls in the last page's
        # open window; the record listed T2 on page 1, so page 2 must not.
        _ledger(
            (KEY, T1, "1970-01-01T00:00:00+00:00"),
            (KEY, T2, "1970-01-01T00:00:00+00:00"),
            (FRESH, FRESH, "1970-01-01T00:00:00+00:00"),
        )
        _listing(
            record,
            _answer([_row(T2, 30), _row(T3, 20)], more=True),
            _answer([_row(T1, 5)]),
        )
        first = client.get("/api/sessions").json()
        second = client.get("/api/sessions", params={"cursor": first["next_cursor"]}).json()
        pages = [[r["transcript_id"] for r in body["sessions"]] for body in (first, second)]
        assert pages == [[T2, T3], [T1, FRESH]]
        listed = pages[0] + pages[1]
        assert len(listed) == len(set(listed))

    def test_entering_an_old_record_only_transcript_does_not_duplicate_it(self, client, record):
        # The ledger never saw OTHER; entering it writes its first ledger row
        # now, inside page 1's open window. The record lists it on page 2.
        key = client.post(f"/api/sessions/{OTHER}/enter").json()["session_key"]
        _listing(
            record,
            _answer([_row(T2, 30)], more=True),
            _answer([_row(OTHER, 5)]),
        )
        first = client.get("/api/sessions").json()
        assert [r["transcript_id"] for r in first["sessions"]] == [T2]
        second = client.get("/api/sessions", params={"cursor": first["next_cursor"]}).json()
        rows = second["sessions"]
        assert [r["transcript_id"] for r in rows] == [OTHER]
        assert rows[0]["session_key"] == key and rows[0]["current_for_key"] is True

    def test_ledger_transcripts_the_record_cannot_vouch_for_make_the_page_incomplete(
        self, client, record
    ):
        _ledger((FRESH, FRESH, "2026-11-01T11:00:00+00:00"))

        def answer(**kwargs):
            if kwargs.get("session_ids"):
                return ClientAnswer.failed("record went away")
            return _answer([_row(T1, 10)])

        record.answers["list_sessions"] = answer
        body = client.get("/api/sessions").json()
        assert body["record"] == "incomplete" and body["record_reason"] == "record went away"
        assert [r["transcript_id"] for r in body["sessions"]] == [T1]

    def test_a_cursor_this_listing_did_not_issue_is_400(self, client, record):  # noqa: ARG002
        assert client.get("/api/sessions", params={"cursor": "nope"}).status_code == 400

    def test_only_drain_1_waits_on_a_drain(self, client, record):
        client.get("/api/sessions")
        client.get("/api/sessions", params={"drain": "0"})
        client.get("/api/sessions", params={"drain": "1"})
        waits = [call["wait"] for call in record.calls_to("list_sessions")]
        assert waits == [None, None, Drain()]


# --- the per-session endpoints ---------------------------------------------------------


class TestEndpoints:
    def test_non_uuid_ids_are_422(self, client):
        for method, path, body in [
            ("put", "/api/sessions/not-a-uuid/title", {"title": "x"}),
            ("put", "/api/sessions/not-a-uuid/last-artifact", {"artifact_id": "a"}),
            ("put", "/api/sessions/last-active", {"session_key": "not-a-uuid"}),
            ("post", f"/api/sessions/{T1.upper()}/enter", None),
        ]:
            resp = getattr(client, method)(path, json=body)
            assert resp.status_code == 422, (path, resp.text)

    def test_last_artifact(self, client, record):
        record.answers["list_sessions"] = _answer([_row(T1, 10)])
        resp = client.put(f"/api/sessions/{T1}/last-artifact", json={"artifact_id": "art-1"})
        assert resp.status_code == 200
        assert client.get("/api/sessions").json()["sessions"][0]["last_artifact_id"] == "art-1"

    def test_puts_for_a_just_minted_key_and_after_a_restart(self, client, record):
        minted = str(uuid.uuid4())
        assert client.get("/api/sessions/last-active").json()["session_key"] is None
        resp = client.put("/api/sessions/last-active", json={"session_key": minted})
        assert resp.status_code == 200 and resp.json()["session_key"] == minted
        assert client.put(f"/api/sessions/{minted}/title", json={"title": "Fresh"}).is_success

        restarted = TestClient(_app())
        assert restarted.get("/api/sessions/last-active").json()["session_key"] == minted
        record.answers["list_sessions"] = _answer([_row(minted, 10)])
        assert restarted.get("/api/sessions").json()["sessions"][0]["title"] == "Fresh"
        resp = restarted.put("/api/sessions/last-active", json={"session_key": KEY})
        assert resp.json()["session_key"] == KEY

    async def test_enter_then_a_spawn_of_the_new_key_resumes_the_transcript(
        self, client, monkeypatch
    ):
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"), (KEY, T2, "2026-11-01T09:20:00+00:00"))
        resp = client.post(f"/api/sessions/{T1}/enter")
        assert resp.status_code == 200
        new_key = resp.json()["session_key"]
        assert new_key not in (KEY, T1, T2)
        app = client.app
        assert transcript_map.get(app, new_key) == T1
        ledger = session_ledger.load(session_meta.ledger_state_dir())
        assert ledger.current(new_key) == T1 and ledger.current(KEY) == T2
        assert app.state.broadcaster.frames[-1] == {
            "type": session_meta.SESSION_TRANSCRIPT_FRAME,
            "session_key": new_key,
        }

        monkeypatch.setattr(session_handoff, "_transcripts_on_disk", lambda _app: {T1, T2})
        spawn = await session_handoff._spawn_request(app, new_key, session_handoff.SURFACE_CHAT)
        assert spawn.resume_id == T1
        rows_before = len(ledger)
        session_handoff._record_spawn(app, new_key, spawn.resume_id)
        assert len(session_ledger.load(session_meta.ledger_state_dir())) == rows_before


# --- a session's layout ----------------------------------------------------------------

LAYOUT = {
    "version": 2,
    "layout": {"root": {"kind": "tile", "id": "t1", "keys": ["terminal"]}},
    "panels": {"knowledge": {"title": "Knowledge"}},
    "pages": {"knowledge": "/panel/knowledge/page/beam"},
}


class TestPins:
    def test_pin_shows_in_the_rows_and_unpin_clears_it(self, client, record):
        _ledger((KEY, T2, "2026-11-01T09:00:00+00:00"))
        record.answers["list_sessions"] = _answer([_row(T1, 20), _row(T2, 10)])
        resp = client.put(f"/api/sessions/{T2}/pin", json={"pinned": True})
        assert resp.status_code == 200 and resp.json()["pinned"]
        body = client.get("/api/sessions").json()
        rows = _by_id(body)
        assert rows[T2]["pinned"] == resp.json()["pinned"] and rows[T1]["pinned"] is None
        assert [r["transcript_id"] for r in body["pinned_sessions"]] == [T2]
        assert (
            client.put(f"/api/sessions/{T2}/pin", json={"pinned": False}).json()["pinned"] is None
        )
        body = client.get("/api/sessions").json()
        assert body["pinned_sessions"] == [] and _by_id(body)[T2]["pinned"] is None

    def test_the_first_page_carries_pins_from_later_pages(self, client, record):
        session_meta.set_auto_title(client.app, T3, "Old one", "done")
        client.put(f"/api/sessions/{T3}/pin", json={"pinned": True})
        _listing(record, _answer([_row(T1, 30)], more=True), _answer([_row(T3, 10)]))
        first = client.get("/api/sessions").json()
        assert T3 not in _by_id(first)
        assert [(r["transcript_id"], r["title"]) for r in first["pinned_sessions"]] == [
            (T3, "Old one")
        ]
        later = client.get("/api/sessions", params={"cursor": first["next_cursor"]}).json()
        assert "pinned_sessions" not in later

    def test_a_pin_from_a_later_page_carries_its_record_row_without_a_meta_title(
        self, client, record
    ):
        _ledger((KEY, T3, "2026-11-01T08:00:00+00:00"))
        client.put(f"/api/sessions/{T3}/pin", json={"pinned": True})
        old = _row(T3, 5, prompt="Scan the injection kicker", message_count=12)
        page = _answer([_row(T1, 30)], more=True)

        def answer(**kwargs):
            ids = kwargs.get("session_ids")
            if ids:
                return _answer([old] if T3 in ids else [])
            return page

        record.answers["list_sessions"] = answer
        body = client.get("/api/sessions").json()
        assert T3 not in _by_id(body)
        [row] = body["pinned_sessions"]
        assert row["transcript_id"] == T3
        assert row["title"] != "Untitled session" and "injection kicker" in row["title"]
        assert row["message_count"] == 12
        assert row["prompt"] == "Scan the injection kicker"
        assert row["last_ts"] == old["last_ts"]

    def test_a_pin_the_record_cannot_answer_for_is_still_listed(self, client, record):
        _ledger((KEY, T3, "2026-11-01T08:00:00+00:00"))
        client.put(f"/api/sessions/{T3}/pin", json={"pinned": True})
        page = _answer([_row(T1, 30)], more=True)

        def answer(**kwargs):
            if kwargs.get("session_ids"):
                return ClientAnswer(data=None, complete=False, unavailable="down")
            return page

        record.answers["list_sessions"] = answer
        [row] = client.get("/api/sessions").json()["pinned_sessions"]
        assert row["transcript_id"] == T3 and row["message_count"] == 0

    def test_a_session_only_the_record_lists_as_mine_can_be_pinned(self, client, record):
        record.answers["list_sessions"] = _answer([_row(T2, 10)])
        resp = client.put(f"/api/sessions/{T2}/pin", json={"pinned": True})
        assert resp.status_code == 200 and resp.json()["pinned"]
        assert record.calls_to("list_sessions") == [{"session_ids": [T2], "limit": 1}]
        assert session_meta.pinned_ids(client.app) == [T2]
        resp = client.put(f"/api/sessions/{T2}/pin", json={"pinned": False})
        assert resp.status_code == 200 and resp.json()["pinned"] is None

    def test_another_operators_session_cannot_be_pinned(self, client, record):
        theirs = _row(OTHER, 10, persons=["bob"], identities=["bob"])
        record.answers["list_sessions"] = _answer([theirs])
        resp = client.put(f"/api/sessions/{OTHER}/pin", json={"pinned": True})
        assert resp.status_code == 404
        assert session_meta.pinned_ids(client.app) == []
        assert client.app.state.broadcaster.frames == []

    def test_a_row_the_listing_drops_cannot_be_pinned(self, client, record):
        dispatched = _row(OTHER, 10, run_ids=["run-1"])
        record.answers["list_sessions"] = _answer([dispatched])
        assert client.put(f"/api/sessions/{OTHER}/pin", json={"pinned": True}).status_code == 404

    def test_a_deleted_session_the_record_still_lists_cannot_be_pinned(self, client, record):
        record.answers["list_sessions"] = _answer([_row(T2, 10)])
        session_meta.mark_deleted(client.app, T2)
        assert client.put(f"/api/sessions/{T2}/pin", json={"pinned": True}).status_code == 404
        assert session_meta.pinned_ids(client.app) == []

    def test_with_the_record_down_only_a_session_the_hub_holds_can_be_pinned(self, client, record):
        record.answers["list_sessions"] = ClientAnswer(unavailable="record down")
        assert client.put(f"/api/sessions/{T2}/pin", json={"pinned": True}).status_code == 404
        _ledger((KEY, T2, "2026-11-01T09:00:00+00:00"))
        assert client.put(f"/api/sessions/{T2}/pin", json={"pinned": True}).status_code == 200

    def test_a_pin_sends_the_session_pinned_frame(self, client):
        _ledger((KEY, T2, "2026-11-01T09:00:00+00:00"))
        pinned = client.put(f"/api/sessions/{T2}/pin", json={"pinned": True}).json()["pinned"]
        client.put(f"/api/sessions/{T2}/pin", json={"pinned": False})
        assert client.app.state.broadcaster.frames == [
            {"type": "session_pinned", "transcript_id": T2, "pinned": pinned},
            {"type": "session_pinned", "transcript_id": T2, "pinned": None},
        ]

    def test_a_deleted_session_cannot_be_pinned_and_loses_its_pin(self, client, record):
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"), (KEY, T2, "2026-11-01T09:30:00+00:00"))
        client.put(f"/api/sessions/{T2}/pin", json={"pinned": True})
        session_meta.mark_deleted(client.app, T2)
        assert client.put(f"/api/sessions/{T2}/pin", json={"pinned": True}).status_code == 404
        record.answers["list_sessions"] = _answer([_row(T1, 10)])
        assert client.get("/api/sessions").json()["pinned_sessions"] == []

    def test_a_non_uuid_or_bodyless_pin_is_422(self, client):
        assert client.put("/api/sessions/not-a-uuid/pin", json={"pinned": True}).status_code == 422
        assert client.put(f"/api/sessions/{T1}/pin", json={}).status_code == 422


class TestLayout:
    def test_put_then_get_round_trips_and_the_listing_stays_lean(self, client, record):
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"))
        assert client.get(f"/api/sessions/{T1}/layout").json() == {
            "transcript_id": T1,
            "layout": None,
        }
        resp = client.put(f"/api/sessions/{T1}/layout", json={"layout": LAYOUT})
        assert resp.status_code == 200 and resp.json() == {"transcript_id": T1, "saved": True}
        assert client.get(f"/api/sessions/{T1}/layout").json()["layout"] == LAYOUT
        record.answers["list_sessions"] = _answer([_row(T1, 10)])
        assert "layout" not in client.get("/api/sessions").json()["sessions"][0]

    def test_unknown_fields_are_dropped(self, client):
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"))
        body = {"layout": {**LAYOUT, "sidebar_width": 320}}
        assert client.put(f"/api/sessions/{T1}/layout", json=body).status_code == 200
        assert "sidebar_width" not in client.get(f"/api/sessions/{T1}/layout").json()["layout"]

    def test_null_forgets_it(self, client):
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"))
        client.put(f"/api/sessions/{T1}/layout", json={"layout": LAYOUT})
        resp = client.put(f"/api/sessions/{T1}/layout", json={"layout": None})
        assert resp.json()["saved"] is False
        assert client.get(f"/api/sessions/{T1}/layout").json()["layout"] is None

    def test_a_malformed_layout_is_422(self, client):
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"))
        for bad in [{"version": 1}, {**LAYOUT, "pages": {"knowledge": "//evil.example/"}}]:
            resp = client.put(f"/api/sessions/{T1}/layout", json={"layout": bad})
            assert resp.status_code == 422, resp.text

    def test_a_layout_over_the_cap_is_413(self, client):
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"))
        big = {**LAYOUT, "panels": {"x": {"params": "a" * session_meta.LAYOUT_MAX_BYTES}}}
        assert client.put(f"/api/sessions/{T1}/layout", json={"layout": big}).status_code == 413
        assert client.get(f"/api/sessions/{T1}/layout").json()["layout"] is None

    def test_a_session_that_is_not_mine_is_404(self, client):
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"))
        assert client.get(f"/api/sessions/{OTHER}/layout").status_code == 404
        resp = client.put(f"/api/sessions/{OTHER}/layout", json={"layout": LAYOUT})
        assert resp.status_code == 404
        assert session_meta.get(client.app, OTHER)["layout"] is None

    def test_a_deleted_session_is_404_and_its_layout_is_gone(self, client):
        _ledger((KEY, T1, "2026-11-01T09:00:00+00:00"))
        client.put(f"/api/sessions/{T1}/layout", json={"layout": LAYOUT})
        session_meta.mark_deleted(client.app, T1)
        assert client.get(f"/api/sessions/{T1}/layout").status_code == 404
        assert client.put(f"/api/sessions/{T1}/layout", json={"layout": LAYOUT}).status_code == 404
        assert session_meta.get(client.app, T1)["layout"] is None

    def test_a_non_uuid_id_is_422(self, client):
        assert client.get("/api/sessions/not-a-uuid/layout").status_code == 422
        resp = client.put("/api/sessions/not-a-uuid/layout", json={"layout": LAYOUT})
        assert resp.status_code == 422


# --- deleting a session ----------------------------------------------------------------


class _Pty:
    is_alive = True


class _Ptys:
    def __init__(self, *keys: str) -> None:
        self._keys = set(keys)

    def session_keys(self) -> list[str]:
        return sorted(self._keys)

    def get_session(self, key: str):
        return _Pty() if key in self._keys else None


class _Chats:
    def __init__(self, **busy: bool) -> None:
        self.chats = {key: SimpleNamespace(is_busy=b) for key, b in busy.items()}
        self.terminated: list[str] = []

    def keys(self) -> list[str]:
        return list(self.chats)

    def get(self, key: str):
        return self.chats.get(key)

    async def terminate(self, key: str):
        self.terminated.append(key)
        return self.chats.pop(key, None)


class TestDelete:
    @pytest.fixture
    def hub(self, client, tmp_path, monkeypatch):
        """The client's app with an artifact store and a transcript directory;
        KEY went T1 → T2 → T3, so T2 is a past session."""
        from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery
        from osprey.stores.artifact_store import ArtifactStore

        app = client.app
        data_root = tmp_path / "data"
        app.state.panel_status_root = data_root
        app.state.project_cwd = str(tmp_path / "demo")
        sessions_dir = tmp_path / "projects" / "demo"
        sessions_dir.mkdir(parents=True)
        monkeypatch.setattr(SessionDiscovery, "_resolve_sessions_dir", lambda _self: sessions_dir)
        _ledger(
            (KEY, T1, "2026-11-01T09:00:00+00:00"),
            (KEY, T2, "2026-11-01T09:20:00+00:00"),
            (KEY, T3, "2026-11-01T09:40:00+00:00"),
        )
        store = ArtifactStore(workspace_root=data_root, auto_launch=False)

        def save(title: str, minute: int, **meta) -> str:
            first_ts = f"2026-11-01T09:{minute:02d}:30+00:00"
            return store.save_file(
                b"x",
                f"{title}.txt",
                "text",
                title,
                mime_type="text/plain",
                session_id=KEY,
                metadata={"first_ts": first_ts, **meta},
            ).id

        ids = {
            "t1": save("one", 5),
            "t2a": save("two-a", 25),
            "t2b": save("two-b", 30),
            "t3": save("three", 45),
            "stamped": save("stamped", 5, transcript_id=T2),
        }
        (sessions_dir / f"{T2}.jsonl").write_text("{}\n")
        (sessions_dir / T2 / "subagents").mkdir(parents=True)
        (sessions_dir / T2 / "subagents" / "agent-1.jsonl").write_text("{}\n")
        (sessions_dir / f"{T3}.jsonl").write_text("{}\n")
        return SimpleNamespace(app=app, data_root=data_root, ids=ids, sessions_dir=sessions_dir)

    def test_deletes_artifacts_ledger_meta_map_and_transcript_files(self, client, hub):
        from osprey.stores.artifact_store import ArtifactStore

        session_meta.set_title(hub.app, T2, "Named")
        other_key = str(uuid.uuid4())
        transcript_map.set(hub.app, other_key, T2)
        _ledger((other_key, T2, "2026-11-01T09:50:00+00:00"))

        resp = client.delete(f"/api/sessions/{T2}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        owned = [hub.ids["t2a"], hub.ids["t2b"], hub.ids["stamped"]]
        assert sorted(body["deleted_artifacts"]) == sorted(owned)
        assert body["transcript_files"] == 2
        store = ArtifactStore(workspace_root=hub.data_root, auto_launch=False)
        assert {e.id for e in store.list_entries()} == {hub.ids["t1"], hub.ids["t3"]}
        ledger = session_ledger.load(session_meta.ledger_state_dir())
        assert T2 not in ledger.transcripts() and ledger.current(KEY) == T3
        assert transcript_map.get(hub.app, other_key) == other_key
        meta = session_meta.get(hub.app, T2)
        assert meta["deleted"] and meta["title"] is None
        assert not (hub.sessions_dir / f"{T2}.jsonl").exists()
        assert not (hub.sessions_dir / T2).exists()
        assert (hub.sessions_dir / f"{T3}.jsonl").exists()
        assert hub.app.state.broadcaster.frames[-1] == {
            "type": session_meta.SESSION_DELETED_FRAME,
            "transcript_id": T2,
        }

    def test_the_tombstone_hides_it_although_the_record_keeps_its_rows(self, client, record, hub):
        record.answers["list_sessions"] = _answer([_row(T3, 30), _row(T2, 20), _row(T1, 10)])
        assert client.delete(f"/api/sessions/{T2}").status_code == 200
        listed = [r["transcript_id"] for r in client.get("/api/sessions").json()["sessions"]]
        assert listed == [T3, T1]
        restarted = TestClient(_app())  # the tombstone is on disk, not only in memory
        listed = [r["transcript_id"] for r in restarted.get("/api/sessions").json()["sessions"]]
        assert listed == [T3, T1]
        # Nothing writes it back: a transcript report, entering it, a second delete.
        assert not session_meta.record_transcript_move(hub.app, KEY, T2)
        assert client.post(f"/api/sessions/{T2}/enter").status_code == 404
        assert client.delete(f"/api/sessions/{T2}").status_code == 404

    def test_the_session_the_agent_is_on_is_409(self, client, hub):
        session_meta.set_last_active(hub.app, KEY)
        resp = client.delete(f"/api/sessions/{T3}")
        assert resp.status_code == 409 and "Switch to another session" in resp.json()["detail"]
        assert client.delete(f"/api/sessions/{T2}").status_code == 200

    def test_a_running_turn_or_an_open_terminal_is_409(self, client, hub):
        hub.app.state.operator_registry = SimpleNamespace(chats=_Chats(**{KEY: True}))
        resp = client.delete(f"/api/sessions/{T3}")
        assert resp.status_code == 409 and "turn is running" in resp.json()["detail"]
        hub.app.state.operator_registry = SimpleNamespace(chats=_Chats())
        hub.app.state.pty_registry = _Ptys(KEY)
        assert client.delete(f"/api/sessions/{T3}").status_code == 409
        assert session_meta.get(hub.app, T3)["deleted"] is None

    def test_an_idle_chat_still_holding_it_is_stopped(self, client, hub):
        chats = _Chats(**{KEY: False})
        hub.app.state.operator_registry = SimpleNamespace(chats=chats)
        assert client.delete(f"/api/sessions/{T3}").status_code == 200
        assert chats.terminated == [KEY]

    def test_another_operators_session_is_404(self, client, hub):  # noqa: ARG002
        assert client.delete(f"/api/sessions/{OTHER}").status_code == 404
        assert client.delete("/api/sessions/not-a-uuid").status_code == 422
