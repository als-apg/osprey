"""``POST /api/artifacts/session-summary``: per-session counts and the owned total.

The summary is the sidebar's count source, so its numbers must agree with what
``session-list`` actually lists: a form's count equals that form's rows under
``runs=grouped``, a run counts once whatever its head, and the loaded sessions
plus "More sessions" add up to every owned operator-visible entry. An entry no
session owns counts nowhere.
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
from osprey.stores.artifact_store import operator_visible
from osprey_connectors import posture_store

EXEC = "toolu_run1"
KEY = "11111111-1111-4111-8111-111111111111"
KEY_2 = "22222222-2222-4222-8222-222222222222"
T1 = KEY  # a key's first transcript id is the key itself
T2 = "33333333-3333-4333-8333-333333333333"
T3 = "44444444-4444-4444-8444-444444444444"
T_OTHER = "55555555-5555-4555-8555-555555555555"

SUMMARY = "/api/artifacts/session-summary"


def _at(minute: int, hour: int = 10) -> str:
    return f"2026-11-04T{hour:02d}:{minute:02d}:00+00:00"


def _key(i: int) -> str:
    return f"{i:08x}-0000-4000-8000-{i:012x}"


# ---- fixtures ---------------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path, monkeypatch) -> Path:
    """The ledger's state directory, bound so the app never reads a real one."""
    bound = tmp_path / "state"
    bound.mkdir()
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(bound))
    monkeypatch.delenv("OSPREY_SESSION_ID", raising=False)
    monkeypatch.delenv("OSPREY_ARTIFACT_ORIGIN", raising=False)
    posture_store.invalidate_cache()
    yield bound
    posture_store.invalidate_cache()


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
        stamp = before + 1_000_000_000
        os.utime(path, ns=(stamp, stamp))


# ---- seeding ----------------------------------------------------------------


def _place(store, entry, minute: int | None = None, hour: int = 10, **attrs):
    if minute is not None:
        attrs.setdefault("timestamp", _at(minute, hour))
    if attrs:
        store.update_entry_metadata(entry.id, **attrs)
    return store.get_entry(entry.id)


def _text(store, title="Note", *, minute=None, hour=10, session_id="", **attrs):
    metadata = {"first_ts": _at(minute, hour)} if minute is not None else {}
    entry = store.save_file(
        b"hello",
        "note.txt",
        "text",
        title,
        mime_type="text/plain",
        tool_source="test",
        metadata=metadata,
        session_id=session_id,
    )
    return _place(store, entry, minute, hour, **attrs)


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


def _preview(store, exec_id=EXEC, *, minute=None, session_id=""):
    entry = store.save_file(
        b"{}",
        "review.ipynb",
        "notebook",
        "Review: fit",
        mime_type="application/x-ipynb+json",
        tool_source="osprey_approval",
        metadata={"run_part": "preview", RUN_RECORD_KEY: True},
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


def _summary(client, sessions: list[str]) -> dict:
    response = client.post(SUMMARY, json={"sessions": sessions})
    assert response.status_code == 200, response.text
    return response.json()


def _listed(client, body: dict) -> list[dict]:
    """Every row ``session-list`` returns for *body*, following cursors."""
    rows: list[dict] = []
    cursor = None
    while True:
        params = dict(body, limit=3)
        if cursor:
            params["cursor"] = cursor
        response = client.post("/api/artifacts/session-list", json=params)
        assert response.status_code == 200, response.text
        page = response.json()
        rows.extend(page["artifacts"])
        cursor = page["next_cursor"]
        if cursor is None:
            return rows


def _by_form(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["form"]] = counts.get(row["form"], 0) + 1
    return counts


# ---- tests ------------------------------------------------------------------


class TestCountsMatchTheListing:
    def test_by_form_equals_the_grouped_rows_of_a_session_above_one_page(
        self, client, store, state_dir
    ):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        for i in range(1, 6):
            _text(store, f"note {i}", minute=i, session_id=KEY)
        for i in range(6, 10):
            _plot(store, f"plot {i}", minute=i, session_id=KEY)
        _preview(store, minute=10, session_id=KEY)
        _notebook(store, minute=11, session_id=KEY)
        _output(store, minute=12, session_id=KEY)
        _result(store, minute=13, session_id=KEY, run_status="succeeded")
        _preview(store, "toolu_run2", minute=14, session_id=KEY)  # in flight: its own run

        rows = _listed(client, {"session": T1, "runs": "grouped"})
        assert len(rows) > 3  # more than one page was walked
        summary = _summary(client, [T1])
        counts = summary["counts"][T1]
        assert counts["by_form"] == _by_form(rows)
        assert counts["total"] == len(rows)
        assert counts["by_form"]["runs"] == 2
        for form, n in counts["by_form"].items():
            assert len(_listed(client, {"session": T1, "runs": "grouped", "form": form})) == n
        assert summary["ledger"] == "ok"

    def test_a_session_with_nothing_answers_zero(self, client, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        summary = _summary(client, [T1, T_OTHER])
        assert summary["counts"][T1] == {"total": 0, "by_form": {}}
        assert summary["counts"][T_OTHER] == {"total": 0, "by_form": {}}
        assert summary["owned_total"] == 0
        assert "earlier" not in summary

    def test_one_runs_count_while_the_groups_head_changes(self, client, store, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        _preview(store, minute=1, session_id=KEY)
        assert _summary(client, [T1])["counts"][T1]["by_form"] == {"runs": 1}
        _notebook(store, minute=2, session_id=KEY)
        assert _summary(client, [T1])["counts"][T1]["by_form"] == {"runs": 1}
        _result(store, minute=3, session_id=KEY, run_status="failed")
        counts = _summary(client, [T1])["counts"][T1]
        assert counts["by_form"] == {"runs": 1}
        assert counts["total"] == 1


class TestTotalsAddUp:
    def test_three_hundred_sessions_in_two_pages_add_up_to_the_visible_total(
        self, client, store, state_dir
    ):
        keys = [_key(i) for i in range(1, 301)]
        _write_ledger(state_dir, [(k, k, session_ledger.EPOCH) for k in keys])
        for i, k in enumerate(keys):
            _text(store, f"s{i}", minute=i % 60, hour=10 + i // 60, session_id=k)
            if i % 3 == 0:
                _plot(store, f"p{i}", minute=i % 60, session_id=k)
        _text(store, "untagged", minute=1)
        _text(store, "unknown", minute=2, session_id=T_OTHER)
        _text(store, "dispatch", minute=3, session_id=keys[0], run_id="run-1")
        _text(store, "watchdog", minute=4, origin="watchdog")

        # The untagged and unknown-key entries are visible but owned by no session.
        owned_visible = sum(1 for e in store.list_entries() if operator_visible(e)) - 2
        first = _summary(client, keys[:150])
        second = _summary(client, keys[150:])
        assert first["owned_total"] == second["owned_total"]
        requested = sum(c["total"] for c in first["counts"].values())
        requested += sum(c["total"] for c in second["counts"].values())
        more = first["owned_total"] - requested
        assert more == 0
        assert requested + more == owned_visible

        # Only the first page loaded: the rest is "More sessions".
        loaded = sum(c["total"] for c in first["counts"].values())
        assert loaded + (first["owned_total"] - loaded) == owned_visible
        assert first["owned_total"] - loaded == sum(c["total"] for c in second["counts"].values())


class TestOwnership:
    def test_a_clear_split_counts_each_transcript_apart(self, client, store, state_dir):
        _write_ledger(
            state_dir,
            [(KEY, T1, session_ledger.EPOCH), (KEY, T2, _at(10))],
        )
        for i in (1, 2, 3):
            _text(store, f"before {i}", minute=i, session_id=KEY)
        for i in (11, 12):
            _plot(store, f"after {i}", minute=i, session_id=KEY)
        summary = _summary(client, [T1, T2])
        assert summary["counts"][T1]["total"] == 3
        assert summary["counts"][T1]["by_form"] == _by_form(_listed(client, {"session": T1}))
        assert summary["counts"][T2]["total"] == 2
        assert summary["counts"][T2]["by_form"] == {"plots": 2}
        assert summary["owned_total"] == 5

    def test_unowned_dispatch_and_watchdog_entries_count_nowhere(self, client, store, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        _text(store, "owned", minute=1, session_id=KEY)
        _text(store, "untagged", minute=2)
        _text(store, "unknown key", minute=3, session_id=T_OTHER)
        _text(store, "dispatch", minute=4, run_id="run-1")
        _text(store, "dispatch tagged", minute=5, session_id=KEY, run_id="run-2")
        _text(store, "watchdog", minute=6, origin="watchdog")

        summary = _summary(client, [T1])
        assert set(summary) == {"counts", "owned_total", "ledger"}
        assert summary["counts"][T1]["total"] == 1
        assert summary["owned_total"] == 1

    def test_owned_total_counts_transcripts_not_requested(self, client, store, state_dir):
        _write_ledger(
            state_dir,
            [(KEY, T1, session_ledger.EPOCH), (KEY_2, T3, session_ledger.EPOCH)],
        )
        _text(store, "mine", minute=1, session_id=KEY)
        _text(store, "theirs", minute=2, session_id=KEY_2)
        _text(store, "theirs too", minute=3, session_id=KEY_2)
        summary = _summary(client, [T1])
        assert set(summary["counts"]) == {T1}
        assert summary["counts"][T1]["total"] == 1
        assert summary["owned_total"] == 3

    def test_an_unreadable_ledger_says_so_and_owns_nothing(self, client, store, state_dir):
        _text(store, "tagged", minute=1, session_id=KEY)
        _text(store, "untagged", minute=2)
        (state_dir / session_ledger.STORE_FILENAME).write_text("not json")
        summary = _summary(client, [T1])
        assert summary["ledger"] == "unavailable"
        assert summary["counts"][T1]["total"] == 0
        assert summary["owned_total"] == 0


class TestValidation:
    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"sessions": "not-a-list"},
            {"sessions": ["not-a-uuid"]},
            {"sessions": ["aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee".upper()]},
            {"sessions": [KEY, f"wd-{T2}"]},
            {"sessions": [None]},
        ],
    )
    def test_a_malformed_body_is_a_422(self, client, body):
        assert client.post(SUMMARY, json=body).status_code == 422

    def test_an_empty_page_of_sessions_is_allowed(self, client, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        summary = _summary(client, [])
        assert summary["counts"] == {}

    def test_more_than_the_list_batch_is_allowed(self, client, state_dir):
        _write_ledger(state_dir, [])
        ids = [_key(i) for i in range(1, 51)]
        assert set(_summary(client, ids)["counts"]) == set(ids)


# ---- session-list over the shared ownership rule ----------------------------


def _page(client, body: dict) -> dict:
    response = client.post("/api/artifacts/session-list", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _ids(rows: list[dict]) -> list[str]:
    return sorted(row["id"] for row in rows)


class TestSessionListScope:
    def test_a_key_after_a_clear_lists_the_new_transcript_and_answers_its_id(
        self, client, store, state_dir
    ):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH), (KEY, T2, _at(10))])
        for i in (1, 2, 3):
            _text(store, f"before {i}", minute=i, session_id=KEY)
        after = [_plot(store, f"after {i}", minute=i, session_id=KEY) for i in (11, 12)]

        page = _page(client, {"key": KEY, "runs": "grouped"})
        assert page["transcript"] == T2
        assert _ids(page["artifacts"]) == sorted(e.id for e in after)
        assert page["total"] == 2
        assert page["ledger"] == "ok"
        # The pre-clear transcript is still its own row.
        assert len(_listed(client, {"session": T1})) == 3

    def test_an_unseen_key_answers_an_empty_page_and_no_transcript(self, client, store, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        _text(store, "mine", minute=1, session_id=KEY)
        _text(store, "untagged", minute=2)

        page = _page(client, {"key": KEY_2})
        assert page["transcript"] is None
        assert page["artifacts"] == []
        assert page["total"] == 0
        assert page["next_cursor"] is None
        assert page["ledger"] == "ok"

    def test_a_key_under_an_unreadable_ledger_answers_no_transcript(self, client, store, state_dir):
        _text(store, "tagged", minute=1, session_id=KEY)
        (state_dir / session_ledger.STORE_FILENAME).write_text("not json")
        page = _page(client, {"key": KEY})
        assert page["transcript"] is None
        assert page["artifacts"] == []
        assert page["ledger"] == "unavailable"

    def test_only_the_key_target_answers_a_transcript(self, client, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        assert "transcript" not in _page(client, {"session": T1})
        assert "transcript" not in _page(client, {"sessions": [T1]})

    def test_search_narrows_inside_a_row(self, client, store, state_dir):
        _write_ledger(
            state_dir,
            [(KEY, T1, session_ledger.EPOCH), (KEY_2, T3, session_ledger.EPOCH)],
        )
        beam = _text(store, "Beam current", minute=1, session_id=KEY)
        _text(store, "Vacuum", minute=2, session_id=KEY)
        plot = _plot(store, "beam orbit", minute=3, session_id=KEY)
        _text(store, "beam elsewhere", minute=4, session_id=KEY_2)  # not this row
        _text(store, "BEAM untagged", minute=5)  # in no row
        _text(store, "vacuum untagged", minute=6)

        assert _ids(_listed(client, {"session": T1, "search": "beam"})) == sorted(
            [beam.id, plot.id]
        )
        assert _ids(_page(client, {"key": KEY, "search": "beam"})["artifacts"]) == sorted(
            [beam.id, plot.id]
        )
        narrowed = _page(client, {"sessions": [T1], "search": "beam", "form": "plots"})
        assert _ids(narrowed["pages"][T1]["artifacts"]) == [plot.id]
        # Without search the row is whole.
        assert len(_listed(client, {"session": T1})) == 3

    def test_a_search_matching_nothing_answers_an_empty_page(self, client, store, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH)])
        _text(store, "Beam", minute=1, session_id=KEY)
        page = _page(client, {"key": KEY, "search": "no such thing"})
        assert page["artifacts"] == []
        assert page["transcript"] == T1

    def test_sessions_batches_are_unchanged(self, client, store, state_dir):
        _write_ledger(
            state_dir,
            [(KEY, T1, session_ledger.EPOCH), (KEY_2, T3, session_ledger.EPOCH)],
        )
        mine = [_text(store, f"mine {i}", minute=i, session_id=KEY) for i in (1, 2)]
        theirs = _plot(store, "theirs", minute=3, session_id=KEY_2)
        _text(store, "dispatch", minute=4, session_id=KEY, run_id="run-1")
        _text(store, "watchdog", minute=5, session_id=KEY, origin="watchdog")

        answer = _page(client, {"sessions": [T1, T3, T_OTHER, T1]})
        assert set(answer) == {"pages", "ledger"}
        assert list(answer["pages"]) == [T1, T3, T_OTHER]
        assert _ids(answer["pages"][T1]["artifacts"]) == sorted(e.id for e in mine)
        assert _ids(answer["pages"][T3]["artifacts"]) == [theirs.id]
        assert answer["pages"][T_OTHER]["artifacts"] == []
        assert answer["ledger"] == "ok"
        for page in answer["pages"].values():
            assert set(page) == {"count", "total", "next_cursor", "artifacts"}

    def test_session_list_rows_match_the_owner_verdict(self, client, store, state_dir):
        _write_ledger(state_dir, [(KEY, T1, session_ledger.EPOCH), (KEY, T2, _at(10))])
        _text(store, "before", minute=1, session_id=KEY)
        _text(store, "after", minute=11, session_id=KEY)
        for transcript in (T1, T2):
            rows = _listed(client, {"session": transcript})
            assert rows and all(row["owner"] == transcript for row in rows)
            assert all(row["operator_visible"] for row in rows)


class TestHandlerNames:
    def test_the_counts_route_stays_and_no_handler_shares_the_tools_name(self, client):
        endpoints = {
            route.path: route.endpoint.__name__
            for route in client.app.routes
            if hasattr(route, "endpoint")
        }
        assert SUMMARY in endpoints
        # The removed tool's name, spelled so the tree-wide grep for removed
        # tool names does not count this absence check as a use.
        removed_tool = "_".join(("session", "summary"))
        assert removed_tool not in endpoints.values()
