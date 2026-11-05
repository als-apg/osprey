"""Tests for the hub's per-operator session meta, its identity guard and the ledger backfill.

Pinned here:

* **Meta round-trip and reload survival**: titles, naming outcomes, the last
  artifact and the last-active key are on disk and read back by a fresh app.
* **An empty user title reverts to the auto title**, and a naming outcome never
  overwrites a user title.
* **A session's layout** round-trips, is checked (version, shape, root-relative
  pages, size cap; unknown fields dropped) and goes with the tombstone.
* **A damaged document loads defaults**; ``naming_since`` is stamped once, at the
  store's first creation, and kept across reloads.
* **The identity guard** refuses a hub whose terminal user is not the acting
  identity.
* **The backfill** (Decision 5): a cleared key, a never-cleared key, an artifact
  key with its ``.jsonl`` on disk, no ``(K, K)`` for a non-UUID key, and it runs
  only once.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.interfaces.web_terminal import session_meta, transcript_map
from osprey.stores import session_ledger
from osprey.stores.artifact_store import ArtifactStore
from osprey_connectors import posture_store

KEY = "aaaaaaaa-1111-2222-3333-444444444444"
OTHER_KEY = "bbbbbbbb-1111-2222-3333-444444444444"
T1 = "cccccccc-1111-2222-3333-444444444444"
T2 = "dddddddd-1111-2222-3333-444444444444"
ART_KEY = "eeeeeeee-1111-2222-3333-444444444444"
GONE_KEY = "ffffffff-1111-2222-3333-444444444444"
EMBEDDER_KEY = "my-embedder-session"


def _app(**state) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(**state))


@pytest.fixture
def shared_root(tmp_path, monkeypatch):
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    posture_store.invalidate_cache()
    yield root
    posture_store.invalidate_cache()


@pytest.mark.usefixtures("shared_root")
class TestMetaRoundTrip:
    def test_unknown_transcript_has_empty_record(self):
        app = _app()
        session_meta.load(app)
        assert session_meta.get(app, T1) == {
            "title": None,
            "title_source": None,
            "auto_title": None,
            "naming": None,
            "last_artifact_id": None,
            "layout": None,
            "deleted": None,
        }
        assert session_meta.get_last_active(app) is None

    def test_everything_survives_a_reload(self):
        app = _app()
        session_meta.load(app)
        session_meta.set_auto_title(app, T1, "Beam orbit check", "done")
        session_meta.set_title(app, T2, "My session")
        session_meta.set_auto_title(app, T2, None, "failed")
        session_meta.set_last_artifact(app, T1, "art-1")
        session_meta.set_last_active(app, KEY)
        since = session_meta.naming_since(app)

        fresh = _app()
        session_meta.load(fresh)
        assert session_meta.get(fresh, T1) == {
            "title": "Beam orbit check",
            "title_source": "auto",
            "auto_title": "Beam orbit check",
            "naming": "done",
            "last_artifact_id": "art-1",
            "layout": None,
            "deleted": None,
        }
        t2 = session_meta.get(fresh, T2)
        assert (t2["title"], t2["title_source"], t2["naming"]) == ("My session", "user", "failed")
        last = session_meta.get_last_active(fresh)
        assert last is not None and last["session_key"] == KEY and last["at"]
        assert session_meta.naming_since(fresh) == since

    def test_a_tombstone_replaces_the_record_and_survives_a_reload(self):
        app = _app()
        session_meta.load(app)
        session_meta.set_title(app, T1, "My session")
        session_meta.set_last_artifact(app, T1, "art-1")
        session_meta.mark_deleted(app, T1)
        fresh = _app()
        session_meta.load(fresh)
        record = session_meta.get(fresh, T1)
        assert record["deleted"] and record["title"] is None and record["last_artifact_id"] is None
        assert session_meta.deleted_ids(fresh) == {T1}

    def test_the_document_is_beside_the_transcript_map(self):
        app = _app()
        session_meta.load(app)
        path = session_meta.store_path()
        assert path is not None and path.parent == transcript_map.store_path().parent
        assert set(json.loads(path.read_text())) == {"sessions", "last_active", "naming_since"}

    def test_pending_lives_in_memory_only(self):
        app = _app()
        session_meta.load(app)
        app.state.session_naming_pending.add(T1)
        assert "pending" not in session_meta.store_path().read_text()
        session_meta.set_auto_title(app, T1, "Named", "done")
        assert T1 not in app.state.session_naming_pending

    def test_bad_naming_state_is_refused(self):
        app = _app()
        session_meta.load(app)
        with pytest.raises(ValueError):
            session_meta.set_auto_title(app, T1, "x", "pending")


@pytest.mark.usefixtures("shared_root")
class TestTitles:
    def test_empty_user_title_reverts_to_the_auto_title(self):
        app = _app()
        session_meta.load(app)
        session_meta.set_auto_title(app, T1, "Auto name", "done")
        session_meta.set_title(app, T1, "Mine")
        assert session_meta.get(app, T1)["title"] == "Mine"
        session_meta.set_title(app, T1, "")
        record = session_meta.get(app, T1)
        assert (record["title"], record["title_source"]) == ("Auto name", "auto")

    def test_empty_title_with_no_auto_title_is_untitled(self):
        app = _app()
        session_meta.load(app)
        session_meta.set_title(app, T1, "Mine")
        session_meta.set_title(app, T1, "   ")
        assert session_meta.get(app, T1)["title"] is None

    def test_a_naming_outcome_never_overwrites_a_user_title(self):
        app = _app()
        session_meta.load(app)
        session_meta.set_title(app, T1, "Mine")
        session_meta.set_auto_title(app, T1, "Auto name", "done")
        record = session_meta.get(app, T1)
        assert (record["title"], record["auto_title"]) == ("Mine", "Auto name")


LAYOUT = {
    "version": 2,
    "layout": {"root": {"kind": "tile", "id": "t1", "keys": ["terminal"]}},
    "panels": {"knowledge": {"title": "Knowledge"}},
    "pages": {"knowledge": "/panel/knowledge/page/beam"},
}


@pytest.mark.usefixtures("shared_root")
class TestLayout:
    def test_a_layout_survives_a_reload(self):
        app = _app()
        session_meta.load(app)
        session_meta.set_layout(app, T1, LAYOUT)
        fresh = _app()
        session_meta.load(fresh)
        assert session_meta.get(fresh, T1)["layout"] == LAYOUT
        assert session_meta.get(fresh, T2)["layout"] is None

    def test_unknown_fields_are_dropped_and_missing_maps_default(self):
        app = _app()
        session_meta.load(app)
        session_meta.set_layout(app, T1, {"version": 2, "layout": None, "sidebar": "wide"})
        assert session_meta.get(app, T1)["layout"] == {
            "version": 2,
            "layout": None,
            "panels": {},
            "pages": {},
        }

    @pytest.mark.parametrize(
        "bad",
        [
            [],
            {"layout": None},
            {**LAYOUT, "version": 1},
            {**LAYOUT, "layout": [1]},
            {**LAYOUT, "panels": {"knowledge": "Knowledge"}},
            {**LAYOUT, "pages": {"knowledge": "https://elsewhere.example/x"}},
            {**LAYOUT, "pages": {"knowledge": "//elsewhere.example/x"}},
            {**LAYOUT, "pages": {"knowledge": "javascript:alert(1)"}},
        ],
    )
    def test_what_is_not_a_layout_is_refused(self, bad):
        app = _app()
        session_meta.load(app)
        with pytest.raises(ValueError):
            session_meta.set_layout(app, T1, bad)
        assert session_meta.get(app, T1)["layout"] is None

    def test_a_layout_over_the_cap_is_refused(self):
        app = _app()
        session_meta.load(app)
        big = {**LAYOUT, "panels": {"x": {"params": "a" * session_meta.LAYOUT_MAX_BYTES}}}
        with pytest.raises(session_meta.LayoutTooLarge):
            session_meta.set_layout(app, T1, big)

    def test_none_forgets_it(self):
        app = _app()
        session_meta.load(app)
        session_meta.set_layout(app, T1, LAYOUT)
        session_meta.set_layout(app, T1, None)
        assert session_meta.get(app, T1)["layout"] is None

    def test_a_damaged_layout_on_disk_loads_as_none(self):
        app = _app()
        session_meta.load(app)
        session_meta.set_layout(app, T1, LAYOUT)
        path = session_meta.store_path()
        document = json.loads(path.read_text())
        document["sessions"][T1]["layout"] = {"version": 2, "layout": "nope"}
        path.write_text(json.dumps(document))
        fresh = _app()
        session_meta.load(fresh)
        assert session_meta.get(fresh, T1)["layout"] is None

    def test_the_layout_goes_with_the_tombstone(self):
        app = _app()
        session_meta.load(app)
        session_meta.set_layout(app, T1, LAYOUT)
        session_meta.mark_deleted(app, T1)
        assert session_meta.get(app, T1)["layout"] is None


@pytest.mark.usefixtures("shared_root")
class TestCorruptAndCreation:
    @pytest.mark.parametrize("text", ["{oops", "[1, 2]", '{"sessions": 5, "last_active": "x"}'])
    def test_a_damaged_document_loads_defaults(self, text):
        path = session_meta.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        app = _app()
        document = session_meta.load(app)
        assert document["sessions"] == {}
        assert document["last_active"] is None
        assert document["naming_since"]
        # And it is usable: the next write rewrites a valid document.
        session_meta.set_title(app, T1, "x")
        assert json.loads(path.read_text())["sessions"][T1]["title"] == "x"

    def test_malformed_session_entries_are_dropped(self):
        path = session_meta.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "sessions": {T1: {"title": "ok", "title_source": "user"}, T2: "junk"},
                    "naming_since": "2026-11-01T00:00:00+00:00",
                }
            )
        )
        app = _app()
        session_meta.load(app)
        assert session_meta.get(app, T1)["title"] == "ok"
        assert session_meta.get(app, T2)["title"] is None
        assert session_meta.naming_since(app) == "2026-11-01T00:00:00+00:00"

    def test_naming_since_is_stamped_on_first_creation_only(self):
        assert not session_meta.store_path().exists()
        first = _app()
        session_meta.load(first)
        since = session_meta.naming_since(first)
        assert session_meta.store_path().exists()
        again = _app()
        session_meta.load(again)
        assert session_meta.naming_since(again) == since


class TestNoLocation:
    def test_memory_only_when_the_root_does_not_resolve(self, monkeypatch):
        monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
        monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
        monkeypatch.setattr(posture_store, "state_dir", lambda: None)
        app = _app()
        session_meta.load(app)
        session_meta.set_title(app, T1, "kept in memory")
        assert session_meta.get(app, T1)["title"] == "kept in memory"


class TestIdentityGuard:
    def test_no_terminal_user_passes(self):
        session_meta.check_identity("")

    def test_matching_terminal_user_passes(self, monkeypatch):
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
        session_meta.check_identity("alice")

    def test_a_different_identity_refuses_the_start(self, monkeypatch):
        monkeypatch.setattr(session_meta, "acting_identity", lambda: "bob")
        with pytest.raises(RuntimeError, match="Refusing to start the Web Terminal"):
            session_meta.check_identity("alice")

    def test_an_unusable_terminal_user_refuses_the_start(self, monkeypatch):
        """The ladder rejects a name with a path separator, so the stores file elsewhere."""
        monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice/../bob")
        with pytest.raises(RuntimeError, match="Refusing to start"):
            session_meta.check_identity("alice/../bob")


# ---- backfill ------------------------------------------------------------------


def _jsonl(directory: Path, tid: str, first_ts: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": "summary"})]
    if first_ts:
        lines.append(json.dumps({"type": "user", "timestamp": first_ts}))
        lines.append(json.dumps({"type": "assistant", "timestamp": "2026-12-31T00:00:00.000Z"}))
    path = directory / f"{tid}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def hub(shared_root, tmp_path, monkeypatch):  # noqa: ARG001 - pins the state dir
    """An app with a transcript directory, an artifact store root and a map document."""
    sessions_dir = tmp_path / "claude-projects" / "proj"
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.session_discovery.claude_project_dir",
        lambda _project: sessions_dir,
    )
    store_root = tmp_path / "store-root"
    store_root.mkdir()
    app = _app(project_cwd=str(tmp_path / "project"), panel_status_root=store_root)
    return SimpleNamespace(app=app, sessions_dir=sessions_dir, store_root=store_root)


def _write_map(mapping: dict[str, str]) -> None:
    path = transcript_map.store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping))


def _save_artifact(store_root: Path, session_id: str) -> None:
    ArtifactStore(workspace_root=store_root, auto_launch=False).save_file(
        file_content=b"x",
        filename="x.txt",
        artifact_type="text",
        title="x",
        mime_type="text/plain",
        tool_source="test",
        session_id=session_id,
    )


def _rows():
    return [tuple(r) for r in session_ledger.load(session_meta.ledger_state_dir()).rows]


class TestBackfill:
    def test_a_cleared_key_gets_its_key_then_its_moved_transcript(self, hub):
        _write_map({KEY: T1})
        _jsonl(hub.sessions_dir, T1, "2026-10-01T08:00:00.000Z")
        assert session_meta.backfill_ledger(hub.app) == 2
        assert _rows() == [
            (KEY, KEY, session_ledger.EPOCH),
            (KEY, T1, "2026-10-01T08:00:00+00:00"),
        ]

    def test_a_deleted_transcript_is_not_seeded(self, hub):
        _write_map({KEY: T1})
        _jsonl(hub.sessions_dir, T1, "2026-10-01T08:00:00.000Z")
        session_meta.mark_deleted(hub.app, T1)
        assert session_meta.backfill_ledger(hub.app) == 1
        assert _rows() == [(KEY, KEY, session_ledger.EPOCH)]

    def test_a_cleared_key_whose_transcript_is_gone_keeps_only_its_first_row(self, hub):
        _write_map({KEY: T1})
        session_meta.backfill_ledger(hub.app)
        assert _rows() == [(KEY, KEY, session_ledger.EPOCH)]

    def test_a_never_cleared_artifact_key_with_its_jsonl_on_disk(self, hub):
        _jsonl(hub.sessions_dir, ART_KEY, "2026-10-02T00:00:00Z")
        _save_artifact(hub.store_root, ART_KEY)
        _save_artifact(hub.store_root, ART_KEY)  # distinct keys only
        _save_artifact(hub.store_root, GONE_KEY)  # no transcript on disk: Earlier
        assert session_meta.backfill_ledger(hub.app) == 1
        assert _rows() == [(ART_KEY, ART_KEY, session_ledger.EPOCH)]

    def test_no_fresh_key_row_for_a_non_uuid_key(self, hub):
        _write_map({EMBEDDER_KEY: T1})
        _jsonl(hub.sessions_dir, T1, "2026-10-01T08:00:00Z")
        _jsonl(hub.sessions_dir, EMBEDDER_KEY, "2026-10-01T08:00:00Z")
        _save_artifact(hub.store_root, EMBEDDER_KEY)
        session_meta.backfill_ledger(hub.app)
        assert all(row[0] != EMBEDDER_KEY for row in _rows())

    def test_it_runs_only_once(self, hub):
        _write_map({KEY: T1})
        assert session_meta.backfill_ledger(hub.app) == 1
        _write_map({KEY: T1, OTHER_KEY: T2})
        assert session_meta.backfill_ledger(hub.app) == 0
        assert _rows() == [(KEY, KEY, session_ledger.EPOCH)]

    def test_nothing_to_seed_still_marks_it_done(self, hub):
        assert session_meta.backfill_ledger(hub.app) == 0
        assert session_ledger.exists(session_meta.ledger_state_dir())
        _write_map({KEY: T1})
        assert session_meta.backfill_ledger(hub.app) == 0

    def test_an_existing_ledger_is_never_backfilled(self, hub):
        session_ledger.append(session_meta.ledger_state_dir(), OTHER_KEY, T2)
        _write_map({KEY: T1})
        assert session_meta.backfill_ledger(hub.app) == 0
        assert [r[0] for r in _rows()] == [OTHER_KEY]
