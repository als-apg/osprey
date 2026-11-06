"""``GET /api/session-summary``: the session page's Artifacts view.

The page lists what the gallery sidebar lists under the session key's current
transcript — the shared ownership rule over the hub's session ledger, one row
per run. A request with no key, or for a key whose transcript the ledger does
not record, owns nothing and says why; it never falls back to the newest
transcript or to every artifact on disk.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.artifacts.app import create_app
from osprey.interfaces.web_terminal.routes.session import router
from osprey.stores import session_ledger
from osprey.stores.artifact_forms import RUN_RECORD_KEY
from osprey.stores.artifact_store import EXAMPLE_ORIGIN, ArtifactStore
from osprey_connectors import posture_store

KEY = "11111111-1111-4111-8111-111111111111"
OTHER_KEY = "22222222-2222-4222-8222-222222222222"
T_NEW = "33333333-3333-4333-8333-333333333333"
EXEC = "toolu_run1"

ROUTE = "/api/session-summary"

_T0 = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)


def _at(minutes: int) -> str:
    return (_T0 + timedelta(minutes=minutes)).isoformat()


# ---- fixtures ---------------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path, monkeypatch) -> Path:
    """The hub ledger's directory, bound so the route never reads a real one."""
    bound = tmp_path / "state"
    bound.mkdir()
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(bound))
    monkeypatch.delenv("OSPREY_SESSION_ID", raising=False)
    monkeypatch.delenv("OSPREY_ARTIFACT_ORIGIN", raising=False)
    posture_store.invalidate_cache()
    yield bound
    posture_store.invalidate_cache()


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def store(root, state_dir) -> ArtifactStore:
    assert state_dir.is_dir()
    return ArtifactStore(workspace_root=root)


@pytest.fixture
def page(root, state_dir) -> TestClient:
    """The web terminal's session routes over *root*."""
    assert state_dir.is_dir()
    app = FastAPI()
    app.include_router(router)
    app.state.workspace_dir = root
    return TestClient(app)


def _ledger(state_dir: Path, *rows: tuple[str, str, str]) -> None:
    for key, transcript, from_ts in rows:
        assert session_ledger.append(state_dir, key, transcript, from_ts)
    posture_store.invalidate_cache()


# ---- seeding ----------------------------------------------------------------


def _save(store: ArtifactStore, title: str, *, key: str, minute: int, **kwargs):
    entry = store.save_file(
        file_content=b"<html></html>",
        filename="p.html",
        artifact_type="plot_html",
        title=title,
        mime_type="text/html",
        tool_source="create_interactive_plot",
        session_id=key,
        metadata={"first_ts": _at(minute), **kwargs.pop("metadata", {})},
        **kwargs,
    )
    store.update_entry_metadata(entry.id, timestamp=_at(minute))
    return store.get_entry(entry.id)


def _run(store: ArtifactStore, *, key: str, minute: int):
    """A run: preview and notebook (one is the head, the other folds under it)
    and its result plot (an output, listed)."""
    preview = store.save_file(
        b"{}",
        "review.ipynb",
        "notebook",
        "Review: fit",
        mime_type="application/x-ipynb+json",
        tool_source="osprey_approval",
        metadata={"run_part": "preview", RUN_RECORD_KEY: True, "first_ts": _at(minute)},
        exec_id=EXEC,
        session_id=key,
    )
    notebook = store.save_file(
        b"{}",
        "run.ipynb",
        "notebook",
        "Notebook: fit",
        mime_type="application/x-ipynb+json",
        tool_source="execute",
        metadata={"run_part": "code", RUN_RECORD_KEY: True, "first_ts": _at(minute)},
        exec_id=EXEC,
        session_id=key,
    )
    result = _save(store, "Run result", key=key, minute=minute + 1, exec_id=EXEC)
    return preview, notebook, result


def _ids(body: dict) -> set[str]:
    return {e["id"] for e in body["entries"]}


# ---- the page is the sidebar row ---------------------------------------------


class TestOwnedRows:
    def test_page_equals_the_keys_current_sidebar_row(self, store, page, root, state_dir):
        _ledger(state_dir, (KEY, KEY, _at(0)))
        mine = _save(store, "Mine", key=KEY, minute=5)
        preview, notebook, result = _run(store, key=KEY, minute=10)
        _save(store, "Theirs", key=OTHER_KEY, minute=6)
        _save(store, "Untagged", key="", minute=7)

        body = page.get(ROUTE, params={"session_id": KEY}).json()

        gallery = TestClient(create_app(workspace_root=root))
        sidebar = gallery.post(
            "/api/artifacts/session-list", json={"key": KEY, "runs": "grouped"}
        ).json()
        assert _ids(body) == {e["id"] for e in sidebar["artifacts"]}
        assert {mine.id, result.id} <= _ids(body)
        # One row per run: its head is listed, its other run part folds under it.
        assert len(_ids(body) & {preview.id, notebook.id}) == 1
        assert body["session"] == KEY
        assert body["reason"] == ""
        assert body["totals"]["entry_count"] == 3

    def test_an_earlier_transcripts_artifacts_are_absent(self, store, page, state_dir):
        _ledger(state_dir, (KEY, KEY, _at(0)), (KEY, T_NEW, _at(30)))
        before = _save(store, "Before clear", key=KEY, minute=5)
        after = _save(store, "After clear", key=KEY, minute=35)

        body = page.get(ROUTE, params={"session_id": KEY}).json()

        assert _ids(body) == {after.id}
        assert before.id not in _ids(body)
        assert body["session"] == T_NEW

    def test_rows_carry_form_and_channels(self, store, page, state_dir):
        _ledger(state_dir, (KEY, KEY, _at(0)))
        entry = store.save_data(
            tool="archiver_read",
            data={},
            title="arch",
            metadata={"data_type": "timeseries", "first_ts": _at(5)},
        )
        store.update_entry_metadata(
            entry.id, session_id=KEY, summary={"channels": ["SR:DCCT", "SR:DCCT"]}
        )

        body = page.get(ROUTE, params={"session_id": KEY}).json()

        (row,) = body["entries"]
        assert (row["form"], row["form_label"]) == ("trends", "Trends")
        assert row["channels"] == ["SR:DCCT"]
        assert body["totals"]["forms"] == [{"key": "trends", "label": "Trends", "count": 1}]

    def test_the_example_never_appears(self, store, page, state_dir):
        _ledger(state_dir, (KEY, KEY, _at(0)))
        _save(store, "Example", key="", minute=1, origin=EXAMPLE_ORIGIN)  # seeded by a past release
        mine = _save(store, "Mine", key=KEY, minute=5)

        body = page.get(ROUTE, params={"session_id": KEY}).json()

        assert _ids(body) == {mine.id}


# ---- nothing owned: empty, with the reason -----------------------------------


class TestNoSession:
    def test_no_key_answers_empty_with_a_reason(self, store, page, state_dir):
        _ledger(state_dir, (KEY, KEY, _at(0)))
        _save(store, "Mine", key=KEY, minute=5)
        _save(store, "Untagged", key="", minute=6)

        body = page.get(ROUTE).json()

        assert body["entries"] == []
        assert body["totals"]["entry_count"] == 0
        assert body["session"] is None
        assert body["reason"]

    def test_a_key_without_a_transcript_answers_empty_with_a_reason(self, store, page, state_dir):
        _ledger(state_dir, (OTHER_KEY, OTHER_KEY, _at(0)))
        _save(store, "Theirs", key=OTHER_KEY, minute=5)

        body = page.get(ROUTE, params={"session_id": KEY}).json()

        assert body["entries"] == []
        assert body["session"] is None
        assert body["reason"]

    def test_an_unreadable_ledger_answers_empty_with_a_reason(self, store, page, state_dir):
        (state_dir / session_ledger.STORE_FILENAME).write_text("not json")
        posture_store.invalidate_cache()
        _save(store, "Mine", key=KEY, minute=5)

        body = page.get(ROUTE, params={"session_id": KEY}).json()

        assert body["entries"] == []
        assert body["reason"]
