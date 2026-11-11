"""The gallery listing by form and run, and the per-session listing.

Covers the payload fields every browser-bound entry carries (form, run status,
owner, visibility, a head's parts), ``GET /api/artifacts``'s ``form``,
``after``/``before``, ``runs=grouped`` and ``search`` params, deleting a run
through its head, and ``POST /api/artifacts/session-list`` in its four body
forms against the hub's session ledger.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.artifacts.app import create_app
from osprey.stores import session_ledger
from osprey.stores.artifact_forms import RUN_RECORD_KEY
from osprey_connectors import control_context, posture_store

EXEC = "toolu_run1"
KEY = "11111111-1111-4111-8111-111111111111"
KEY_2 = "22222222-2222-4222-8222-222222222222"
T1 = KEY  # a key's first transcript id is the key itself
T2 = "33333333-3333-4333-8333-333333333333"
T3 = "44444444-4444-4444-8444-444444444444"
T_OTHER = "55555555-5555-4555-8555-555555555555"


def _at(minute: int) -> str:
    return f"2026-11-04T10:{minute:02d}:00+00:00"


# ---- fixtures ---------------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path, monkeypatch) -> Path:
    """The ledger's state directory, bound so the app never reads a real one."""
    bound = tmp_path / "state"
    bound.mkdir()
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(bound))
    monkeypatch.delenv("OSPREY_SESSION_ID", raising=False)
    monkeypatch.delenv("OSPREY_ARTIFACT_ORIGIN", raising=False)
    control_context.invalidate_cache()
    yield bound
    control_context.invalidate_cache()


@pytest.fixture
def client(tmp_path, state_dir) -> TestClient:
    assert state_dir.is_dir()  # bound before the app reads any ledger
    return TestClient(create_app(workspace_root=tmp_path / "data"))


@pytest.fixture
def store(client):
    return client.app.state.artifact_store


def _write_ledger(state_dir: Path, rows: list[tuple[str, str, str]]) -> None:
    path = state_dir / session_ledger.STORE_FILENAME
    before = path.stat().st_mtime_ns if path.exists() else None
    path.write_text(json.dumps({"rows": [list(r) for r in rows]}))
    if before is not None:
        # The app caches the ledger by mtime; make a rewrite visible even on a
        # filesystem whose clock does not tick between two writes.
        stamp = before + 1_000_000_000
        os.utime(path, ns=(stamp, stamp))


# ---- seeding ----------------------------------------------------------------


def _place(store, entry, minute: int | None = None, **attrs):
    if minute is not None:
        attrs.setdefault("timestamp", _at(minute))
    if attrs:
        store.update_entry_metadata(entry.id, **attrs)
    return store.get_entry(entry.id)


def _text(store, title="Note", *, minute=None, session_id="", first_ts=None, **attrs):
    metadata = {"first_ts": first_ts or (_at(minute) if minute is not None else None)}
    metadata = {k: v for k, v in metadata.items() if v is not None}
    metadata.update(attrs.pop("metadata", {}))
    entry = store.save_file(
        b"hello",
        "note.txt",
        "text",
        title,
        description=attrs.pop("description", ""),
        mime_type="text/plain",
        tool_source="test",
        metadata=metadata,
        session_id=session_id,
    )
    return _place(store, entry, minute, **attrs)


def _plot(store, title="Plot", *, minute=None, session_id=""):
    entry = store.save_file(
        b"<html></html>",
        "plot.html",
        "plot_html",
        title,
        mime_type="text/html",
        tool_source="create_interactive_plot",
        session_id=session_id,
    )
    return _place(store, entry, minute)


def _preview(store, exec_id=EXEC, *, minute=None, session_id="", **metadata):
    entry = store.save_file(
        b"{}",
        "review.ipynb",
        "notebook",
        "Review: fit",
        mime_type="application/x-ipynb+json",
        tool_source="osprey_approval",
        metadata={"run_part": "preview", RUN_RECORD_KEY: True, **metadata},
        exec_id=exec_id,
        session_id=session_id,
    )
    return _place(store, entry, minute)


def _notebook(store, exec_id=EXEC, *, minute=None, session_id=""):
    entry = store.save_file(
        b"{}",
        "run.ipynb",
        "notebook",
        "Notebook: fit",
        mime_type="application/x-ipynb+json",
        tool_source="execute",
        metadata={"run_part": "code", RUN_RECORD_KEY: True},
        exec_id=exec_id,
        session_id=session_id,
    )
    return _place(store, entry, minute)


def _output(store, exec_id=EXEC, *, minute=None, session_id=""):
    entry = store.save_file(
        b"png",
        "fig.png",
        "image",
        "Figure",
        mime_type="image/png",
        tool_source="execute",
        metadata={"run_part": "output", "figure": True},
        exec_id=exec_id,
        session_id=session_id,
    )
    return _place(store, entry, minute)


def _result(store, exec_id=EXEC, *, minute=None, session_id="", **metadata):
    entry = store.save_data(
        tool="execute",
        data={"stdout": ""},
        title="fit",
        metadata={"run_part": "result", RUN_RECORD_KEY: True, **metadata},
        exec_id=exec_id,
    )
    return _place(store, entry, minute, session_id=session_id)


def _ids(body) -> list[str]:
    return [e["id"] for e in body["artifacts"]]


def _list(client, **params):
    response = client.get("/api/artifacts", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _session_list(client, body):
    response = client.post("/api/artifacts/session-list", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _walk(client, path_params: dict, *, post: bool = False) -> list[str]:
    """Follow ``next_cursor`` to the end with three-entry pages."""
    ids: list[str] = []
    cursor = None
    while True:
        params = dict(path_params, limit=3)
        if cursor:
            params["cursor"] = cursor
        if post:
            body = _session_list(client, params)
        else:
            body = _list(client, **params)
        ids.extend(_ids(body))
        cursor = body["next_cursor"]
        if cursor is None:
            return ids


# ---- payload ----------------------------------------------------------------


class TestPayload:
    def test_a_plain_entry_carries_every_contract_field(self, client, store):
        plot = _plot(store, session_id="")
        body = client.get(f"/api/artifacts/{plot.id}").json()
        assert body["form"] == "plots"
        assert body["form_label"] == "Plots"
        assert body["interactive"] is True
        assert body["kind"] == ""
        assert body["exec_id"] == ""
        assert body["run_part"] is None
        assert body["run_status"] is None
        assert body["session_id"] == ""
        assert body["owner"] is None
        assert body["operator_visible"] is True
        assert body["run_record"] is False
        assert "run_parts" not in body

    def test_a_head_lists_its_parts_and_the_groups_status(self, client, store):
        p = _preview(store, minute=1)
        n = _notebook(store, minute=2)
        o = _output(store, minute=3)
        r = _result(store, minute=4, run_status="succeeded")
        head = client.get(f"/api/artifacts/{r.id}").json()
        assert head["form"] == "runs"
        assert head["form_label"] == "Runs"
        assert head["run_part"] == "result"
        assert head["run_status"] == "succeeded"
        assert head["run_record"] is True
        assert head["run_parts"] == [
            {"id": p.id, "role": "preview", "title": "Review: fit", "form": "documents"},
            {"id": n.id, "role": "code", "title": "Notebook: fit", "form": "documents"},
            {"id": o.id, "role": "output", "title": "Figure", "form": "plots"},
        ]

    def test_parts_carry_their_role_and_the_groups_status(self, client, store):
        p = _preview(store, minute=1)
        o = _output(store, minute=2)
        _result(store, minute=3, run_status="failed")
        preview = client.get(f"/api/artifacts/{p.id}").json()
        output = client.get(f"/api/artifacts/{o.id}").json()
        assert (preview["form"], preview["form_label"]) == (None, None)
        assert preview["run_part"] == "preview"
        assert preview["run_status"] == "failed"
        assert "run_parts" not in preview
        assert output["form"] == "plots"
        assert output["run_part"] == "output"
        assert output["run_status"] == "failed"

    def test_the_list_carries_the_same_payload_as_get_one(self, client, store):
        _preview(store, minute=1)
        _output(store, minute=2)
        r = _result(store, minute=3, run_status="succeeded")
        listed = {e["id"]: e for e in _list(client)["artifacts"]}
        assert listed[r.id] == client.get(f"/api/artifacts/{r.id}").json()

    def test_dispatch_and_watchdog_are_not_operator_visible(self, client, store):
        dispatch = _text(store, "dispatched", run_id="run-7")
        watchdog = _text(store, "judge", origin="watchdog")
        assert client.get(f"/api/artifacts/{dispatch.id}").json()["operator_visible"] is False
        assert client.get(f"/api/artifacts/{watchdog.id}").json()["operator_visible"] is False


# ---- run states ------------------------------------------------------------


class TestRunStates:
    def test_a_failed_run_is_one_runs_row_with_status_failed(self, client, store):
        _preview(store, minute=1)
        r = _result(store, minute=2, run_status="failed")
        body = _list(client, form="runs")
        assert _ids(body) == [r.id]
        assert body["artifacts"][0]["run_status"] == "failed"

    def test_a_refused_run_is_one_not_run_row(self, client, store):
        """Refused after an ask: the settled preview heads, ``not_run``; nothing else lists."""
        p = _preview(store, minute=1, settled="not_run")
        body = _list(client, runs="grouped")
        assert _ids(body) == [p.id]
        assert body["artifacts"][0]["form"] == "runs"
        assert body["artifacts"][0]["run_status"] == "not_run"

    def test_an_executor_refusal_is_one_not_run_row(self, client, store):
        p = _preview(store, minute=1, settled="not_run")
        r = _result(store, minute=2, run_status="not_run")
        body = _list(client, runs="grouped")
        assert _ids(body) == [r.id]
        assert body["artifacts"][0]["run_status"] == "not_run"
        assert [part["id"] for part in body["artifacts"][0]["run_parts"]] == [p.id]

    def test_an_in_flight_run_has_no_status(self, client, store):
        p = _preview(store, minute=1)
        body = _list(client, form="runs")
        assert _ids(body) == [p.id]
        assert body["artifacts"][0]["run_status"] is None

    def test_a_run_never_asked_heads_by_its_notebook(self, client, store):
        """No approval preview (an auto-approved run in flight): the notebook heads."""
        n = _notebook(store, minute=1)
        o = _output(store, minute=2)
        body = _list(client, runs="grouped")
        rows = {e["id"]: e for e in body["artifacts"]}
        assert set(rows) == {n.id, o.id}
        assert rows[n.id]["form"] == "runs"
        assert rows[n.id]["run_status"] is None
        assert rows[n.id]["run_parts"] == [
            {"id": o.id, "role": "output", "title": "Figure", "form": "plots"}
        ]


# ---- GET params -------------------------------------------------------------


class TestListParams:
    def test_form_keeps_the_listed_form(self, client, store):
        plot = _plot(store, minute=1)
        note = _text(store, minute=2)
        _preview(store, minute=3)
        o = _output(store, minute=4)
        r = _result(store, minute=5, run_status="succeeded")
        assert _ids(_list(client, form="plots")) == [o.id, plot.id]
        assert _ids(_list(client, form="documents")) == [note.id]
        assert _ids(_list(client, form="runs")) == [r.id]
        assert _list(client, form="tables")["total"] == 0

    def test_exec_id_keeps_one_runs_entries(self, client, store):
        _text(store, minute=1)
        p = _preview(store, minute=2)
        o = _output(store, minute=3)
        r = _result(store, minute=4, run_status="succeeded")
        _preview(store, exec_id="toolu_other", minute=5)
        listed = _list(client, exec_id=EXEC)
        assert _ids(listed) == [r.id, o.id, p.id]
        head = next(e for e in listed["artifacts"] if e["id"] == r.id)
        assert {part["role"] for part in head["run_parts"]} >= {"preview", "output"}
        assert _list(client, exec_id="toolu_none")["total"] == 0

    def test_after_and_before_are_half_open(self, client, store):
        a = _text(store, "a", minute=1)
        b = _text(store, "b", minute=2)
        c = _text(store, "c", minute=3)
        assert _ids(_list(client, after=_at(2))) == [c.id, b.id]
        assert _ids(_list(client, before=_at(2))) == [a.id]
        assert _ids(_list(client, after=_at(1), before=_at(3))) == [b.id, a.id]

    @pytest.mark.parametrize("bound", ["after", "before"])
    def test_a_naive_or_bad_bound_is_a_400(self, client, store, bound):
        _text(store, minute=1)
        for value in ("2026-11-04T10:00:00", "yesterday"):
            response = client.get("/api/artifacts", params={bound: value})
            assert response.status_code == 400
            assert bound in response.json()["detail"]

    def test_runs_grouped_drops_preview_and_code_parts_and_keeps_outputs(self, client, store):
        _preview(store, minute=1)
        _notebook(store, minute=2)
        o = _output(store, minute=3)
        r = _result(store, minute=4, run_status="succeeded")
        loose = _text(store, minute=5)
        assert _ids(_list(client, runs="grouped")) == [loose.id, r.id, o.id]
        # Without it every entry lists, the run's parts too.
        assert _list(client)["total"] == 5

    def test_runs_takes_only_grouped(self, client):
        assert client.get("/api/artifacts", params={"runs": "flat"}).status_code == 422

    def test_search_still_matches_filename_and_description(self, client, store):
        by_desc = _text(store, "One", minute=1, description="beam lifetime scan")
        by_file = store.save_file(
            b"x", "orbit_dump.csv", "csv", "Two", mime_type="text/csv", tool_source="test"
        )
        _text(store, "Three", minute=3)
        assert _ids(_list(client, search="LIFETIME")) == [by_desc.id]
        assert _ids(_list(client, search="orbit_dump")) == [by_file.id]

    def test_search_matches_the_form_label(self, client, store):
        plot = _plot(store, "Tune", minute=1)
        _text(store, "Note", minute=2)
        assert _ids(_list(client, search="plots")) == [plot.id]

    def test_paging_walks_a_filtered_listing(self, client, store):
        made = [_text(store, f"n{i}", minute=i).id for i in range(1, 8)]
        _plot(store, minute=9)
        assert _walk(client, {"form": "documents"}) == list(reversed(made))

    def test_a_bad_cursor_is_still_a_400(self, client):
        response = client.get("/api/artifacts", params={"cursor": "not-a-cursor"})
        assert response.status_code == 400


# ---- delete ----------------------------------------------------------------


class TestDeleteRun:
    def test_deleting_a_head_deletes_its_group(self, client, store):
        p = _preview(store, minute=1)
        o = _output(store, minute=2)
        r = _result(store, minute=3, run_status="succeeded")
        keep = _text(store, minute=4)
        response = client.delete(f"/api/artifacts/{r.id}")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "artifact_id": r.id}
        assert {p.id, o.id} & {e.id for e in store.list_entries()} == set()
        assert [e.id for e in store.list_entries()] == [keep.id]

    def test_deleting_an_in_flight_head_deletes_its_group(self, client, store):
        p = _preview(store, minute=1)
        o = _output(store, minute=2)
        assert client.delete(f"/api/artifacts/{p.id}").status_code == 200
        assert store.get_entry(o.id) is None

    def test_deleting_an_output_deletes_only_it(self, client, store):
        o = _output(store, minute=1)
        r = _result(store, minute=2, run_status="succeeded")
        assert client.delete(f"/api/artifacts/{o.id}").status_code == 200
        assert store.get_entry(o.id) is None
        assert store.get_entry(r.id) is not None

    def test_deleting_an_unknown_id_is_a_404(self, client):
        assert client.delete("/api/artifacts/nope").status_code == 404


# ---- ownership --------------------------------------------------------------


class TestOwnership:
    def test_a_stamped_transcript_owns(self, client, store, state_dir):
        upload = _text(store, minute=1, session_id=KEY, metadata={"transcript_id": T3})
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        assert client.get(f"/api/artifacts/{upload.id}").json()["owner"] == T3

    def test_key_and_first_ts_pick_the_transcript(self, client, store, state_dir):
        before_clear = _text(store, "before", minute=1, session_id=KEY)
        after_clear = _text(store, "after", minute=5, session_id=KEY)
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH), (KEY, T2, _at(3))])
        assert client.get(f"/api/artifacts/{before_clear.id}").json()["owner"] == T1
        assert client.get(f"/api/artifacts/{after_clear.id}").json()["owner"] == T2

    def test_a_touch_does_not_move_ownership(self, client, store, state_dir):
        """``first_ts`` decides, not the touched ``timestamp``."""
        old = _text(store, minute=9, session_id=KEY, first_ts=_at(1))
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH), (KEY, T2, _at(3))])
        assert client.get(f"/api/artifacts/{old.id}").json()["owner"] == T1

    def test_an_older_transcript_entered_under_a_new_key_keeps_its_new_artifacts(
        self, client, store, state_dir
    ):
        earlier = _text(store, "first visit", minute=1, session_id=KEY)
        _write_ledger(
            state_dir,
            [(KEY, T1, session_ledger.EPOCH), (KEY, T2, _at(3)), (KEY_2, T1, _at(6))],
        )
        later = _text(store, "re-entered", minute=8, session_id=KEY_2)
        assert client.get(f"/api/artifacts/{earlier.id}").json()["owner"] == T1
        assert client.get(f"/api/artifacts/{later.id}").json()["owner"] == T1
        body = _session_list(client, {"session": T1})
        assert _ids(body) == [later.id, earlier.id]

    def test_the_ledger_is_re_read_when_it_changes(self, client, store, state_dir):
        entry = _text(store, minute=5, session_id=KEY)
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        assert client.get(f"/api/artifacts/{entry.id}").json()["owner"] == T1
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH), (KEY, T2, _at(3))])
        assert client.get(f"/api/artifacts/{entry.id}").json()["owner"] == T2


# ---- session-list -----------------------------------------------------------


class TestSessionList:
    @pytest.fixture
    def seeded(self, store, state_dir):
        """Two transcripts of KEY, one of KEY_2, an untagged and an unknown-key entry."""
        _write_ledger(
            state_dir,
            [
                (KEY, T1, session_ledger.EPOCH),
                (KEY, T2, _at(10)),
                (KEY_2, T3, session_ledger.EPOCH),
            ],
        )
        return {
            "t1": [_text(store, f"t1-{i}", minute=i, session_id=KEY).id for i in (1, 2)],
            "t2": [_text(store, f"t2-{i}", minute=i, session_id=KEY).id for i in (11, 12)],
            "t3": [_text(store, "t3", minute=5, session_id=KEY_2).id],
            "untagged": _text(store, "untagged", minute=6).id,
            "unknown": _text(store, "unknown key", minute=7, session_id=T_OTHER).id,
            "dispatch": _text(store, "dispatch", minute=8, run_id="run-1").id,
            "dispatch_tagged": _text(
                store, "dispatch in session", minute=13, session_id=KEY, run_id="run-2"
            ).id,
            "watchdog": _text(store, "watchdog", minute=9, origin="watchdog").id,
        }

    def test_session_lists_the_transcripts_visible_artifacts(self, client, seeded):
        body = _session_list(client, {"session": T2})
        assert _ids(body) == list(reversed(seeded["t2"]))
        assert body["ledger"] == "ok"
        assert body["total"] == 2
        assert body["next_cursor"] is None
        assert all(e["owner"] == T2 for e in body["artifacts"])

    def test_key_lists_the_keys_current_transcript(self, client, seeded):
        assert _ids(_session_list(client, {"key": KEY})) == list(reversed(seeded["t2"]))
        assert _ids(_session_list(client, {"key": KEY_2})) == seeded["t3"]

    @pytest.mark.usefixtures("seeded")
    def test_a_key_the_ledger_never_saw_lists_nothing(self, client):
        body = _session_list(client, {"key": T_OTHER})
        assert body["artifacts"] == [] and body["total"] == 0

    def test_the_batched_form_pages_each_session(self, client, seeded):
        body = _session_list(client, {"sessions": [T1, T3, T_OTHER], "limit": 1})
        assert body["ledger"] == "ok"
        pages = body["pages"]
        assert set(pages) == {T1, T3, T_OTHER}
        assert _ids(pages[T1]) == [seeded["t1"][1]]
        assert pages[T1]["total"] == 2 and pages[T1]["next_cursor"] is not None
        assert _ids(pages[T3]) == seeded["t3"]
        assert pages[T_OTHER]["total"] == 0

    def test_the_batched_form_takes_at_most_five(self, client):
        ids = [f"{i:08d}-0000-4000-8000-000000000000" for i in range(6)]
        response = client.post("/api/artifacts/session-list", json={"sessions": ids})
        assert response.status_code == 422

    def test_unowned_artifacts_list_in_no_session_and_the_palette_finds_them(self, client, seeded):
        listed = set()
        for transcript in (T1, T2, T3):
            listed |= set(_ids(_session_list(client, {"session": transcript})))
        unowned = {seeded["unknown"], seeded["untagged"]}
        assert not unowned & listed
        assert unowned <= set(_ids(_list(client, limit=200)))

    def test_dispatch_and_watchdog_list_in_no_session(self, client, seeded):
        listed = set()
        for transcript in (T1, T2, T3):
            listed |= set(_ids(_session_list(client, {"session": transcript})))
        assert seeded["dispatch_tagged"] not in listed
        # The palette's search still finds them.
        assert _ids(_list(client, search="dispatch in session")) == [seeded["dispatch_tagged"]]

    @pytest.mark.usefixtures("seeded")
    def test_every_owned_visible_artifact_lists_exactly_once(self, client):
        listed: list[str] = []
        for transcript in (T1, T2, T3):
            listed += _ids(_session_list(client, {"session": transcript}))
        owned = [
            e["id"]
            for e in _list(client, limit=200)["artifacts"]
            if e["operator_visible"] and e["owner"] is not None
        ]
        assert sorted(listed) == sorted(owned)

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"session": T1, "key": KEY},
            {"earlier": True},
            {"sessions": []},
        ],
    )
    def test_exactly_one_body_form(self, client, body):
        response = client.post("/api/artifacts/session-list", json=body)
        assert response.status_code == 422

    def test_form_and_runs_apply(self, client, store, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        _preview(store, minute=1, session_id=KEY)
        o = _output(store, minute=2, session_id=KEY)
        r = _result(store, minute=3, session_id=KEY, run_status="succeeded")
        note = _text(store, minute=4, session_id=KEY)
        assert _ids(_session_list(client, {"session": T1, "runs": "grouped"})) == [
            note.id,
            r.id,
            o.id,
        ]
        assert _ids(_session_list(client, {"session": T1, "form": "runs"})) == [r.id]
        assert _ids(_session_list(client, {"session": T1, "form": "plots"})) == [o.id]

    def test_paging_across_a_session(self, client, store, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        mine = [_text(store, f"m{i}", minute=i, session_id=KEY).id for i in range(1, 9)]
        _text(store, "elsewhere", minute=20, session_id=KEY_2)
        walked = _walk(client, {"session": T1}, post=True)
        assert walked == list(reversed(mine))

    @pytest.mark.usefixtures("seeded")
    def test_a_bad_cursor_is_a_400(self, client):
        response = client.post(
            "/api/artifacts/session-list", json={"session": T1, "cursor": "garbage"}
        )
        assert response.status_code == 400


class TestLedgerUnavailable:
    def test_an_unreadable_ledger_says_so_and_lists_nothing_tagged(self, client, store, state_dir):
        _text(store, "tagged", minute=1, session_id=KEY)
        _text(store, "untagged", minute=2)
        (state_dir / session_ledger.STORE_FILENAME).write_text("{not json")
        by_key = _session_list(client, {"key": KEY})
        assert by_key["ledger"] == "unavailable"
        assert by_key["artifacts"] == []
        session = _session_list(client, {"session": T1})
        assert session["ledger"] == "unavailable"
        assert session["artifacts"] == []
        assert _session_list(client, {"sessions": [T1]})["ledger"] == "unavailable"

    def test_a_missing_ledger_is_ok_and_empty(self, client, store):
        _text(store, "tagged", minute=1, session_id=KEY)
        body = _session_list(client, {"key": KEY})
        assert body["ledger"] == "ok"
        assert body["artifacts"] == []

    def test_no_state_directory_is_unavailable(self, client, store, monkeypatch):
        _text(store, minute=1, session_id=KEY)
        monkeypatch.setattr(posture_store, "state_dir", lambda: None)
        assert _session_list(client, {"session": T1})["ledger"] == "unavailable"


# ---- extension kinds ----------------------------------------------------------

LATTICE_LABEL = "Ring lattice"  # differs from the title-cased key on purpose


@pytest.fixture
def lattice(monkeypatch):
    """A registered extension type ``lattice`` conforming to ``tables``."""
    from osprey.stores import type_registry

    registry = {"lattice": type_registry.ExtensionType("lattice", LATTICE_LABEL, "tables")}
    monkeypatch.setattr(type_registry, "_EXTENSION_TYPES", registry)
    return registry


def _lattice(store, *, minute=None, session_id="", exec_id="", title="Model"):
    metadata = {"run_part": "output"} if exec_id else {}
    entry = store.save_file(
        b"{}",
        "model.json",
        "json",
        title,
        mime_type="application/json",
        tool_source="test",
        metadata=metadata,
        exec_id=exec_id,
        session_id=session_id,
        kind="lattice",
    )
    return _place(store, entry, minute)


@pytest.mark.usefixtures("lattice")
class TestExtensionKind:
    def test_session_list_keeps_the_kind(self, client, store, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        lat = _lattice(store, minute=1, session_id=KEY)
        _text(store, minute=2, session_id=KEY)
        _plot(store, minute=3, session_id=KEY)
        assert _ids(_session_list(client, {"session": T1, "form": "lattice"})) == [lat.id]
        assert _ids(_session_list(client, {"session": T1, "form": "tables"})) == []

    def test_session_summary_counts_the_kind_not_its_base_form(self, client, store, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        _lattice(store, minute=1, session_id=KEY)
        _lattice(store, minute=2, session_id=KEY, title="Model 2")
        _text(store, minute=3, session_id=KEY)
        response = client.post("/api/artifacts/session-summary", json={"sessions": [T1]})
        assert response.status_code == 200
        by_form = response.json()["counts"][T1]["by_form"]
        assert by_form["lattice"] == 2
        assert "tables" not in by_form and "data" not in by_form

    def test_the_payload_carries_the_kind_and_its_registry_label(self, client, store):
        lat = _lattice(store, minute=1)
        (row,) = _list(client)["artifacts"]
        assert row["id"] == lat.id
        assert (row["form"], row["form_label"]) == ("lattice", LATTICE_LABEL)
        one = client.get(f"/api/artifacts/{lat.id}").json()
        assert (one["form"], one["form_label"]) == ("lattice", LATTICE_LABEL)

    def test_a_run_output_of_the_kind_lists_under_it_on_both_rows(self, client, store):
        _preview(store, minute=1)
        out = _lattice(store, minute=2, exec_id=EXEC)
        r = _result(store, minute=3, run_status="succeeded")
        rows = {e["id"]: e for e in _list(client)["artifacts"]}
        assert rows[out.id]["form"] == "lattice"
        parts = {p["id"]: p for p in rows[r.id]["run_parts"]}
        assert parts[out.id]["form"] == "lattice"
        assert _ids(_list(client, form="lattice")) == [out.id]

    def test_search_finds_the_registry_label(self, client, store):
        lat = _lattice(store, minute=1)
        _text(store, minute=2)
        assert _ids(_list(client, search="ring LATTICE")) == [lat.id]
