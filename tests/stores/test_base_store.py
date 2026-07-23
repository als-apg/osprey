"""Unit tests for the file-backed indexed store contract in ``base_store``.

Exercises :class:`BaseStore` directly through a minimal concrete subclass so
the shared infrastructure — JSON index round-trips, mtime-based staleness
detection, per-subclass listener isolation, atomic saves, and the locked
metadata mutation path — is covered independently of the ``ArtifactStore``
serialization details that ``tests/mcp_server/test_artifact_store.py`` leans on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from osprey.stores.base_store import INDEX_VERSION, BaseStore, _sanitize_for_json


@dataclass
class _Item:
    id: str
    name: str = ""
    score: float = 0.0


class _MiniStore(BaseStore[_Item]):
    """Smallest possible concrete store: one dataclass, trivial (de)serialize."""

    _store_name = "mini"
    _subdir = "mini"
    _index_filename = "mini.json"

    def _entry_from_dict(self, d: dict) -> _Item:
        return _Item(**d)

    def _entry_to_dict(self, entry: _Item) -> dict:
        return {"id": entry.id, "name": entry.name, "score": entry.score}

    def add(self, item: _Item) -> _Item:
        """Public helper mirroring how a real subclass persists a new entry."""
        with self._with_index_lock():
            self._entries.append(item)
            self._save_index()
        self._notify_listeners(item)
        return item


class _OtherStore(BaseStore[_Item]):
    """Second subclass — used to prove listener lists are per-subclass."""

    _store_name = "other"
    _subdir = "other"
    _index_filename = "other.json"

    def _entry_from_dict(self, d: dict) -> _Item:
        return _Item(**d)

    def _entry_to_dict(self, entry: _Item) -> dict:
        return {"id": entry.id, "name": entry.name, "score": entry.score}


# ---------------------------------------------------------------------------
# _sanitize_for_json — pure NaN/Inf scrubbing
# ---------------------------------------------------------------------------


class TestSanitizeForJson:
    def test_replaces_nan_and_inf_with_none(self):
        assert _sanitize_for_json(float("nan")) is None
        assert _sanitize_for_json(float("inf")) is None
        assert _sanitize_for_json(float("-inf")) is None

    def test_preserves_finite_and_non_float_values(self):
        assert _sanitize_for_json(1.5) == 1.5
        assert _sanitize_for_json(0.0) == 0.0
        assert _sanitize_for_json("text") == "text"
        assert _sanitize_for_json(7) == 7
        assert _sanitize_for_json(True) is True
        assert _sanitize_for_json(None) is None

    def test_recurses_into_nested_dicts_and_lists(self):
        payload = {
            "a": float("nan"),
            "b": [float("inf"), 1.0, {"c": float("-inf")}],
            "d": {"e": {"f": 2.0}},
        }
        out = _sanitize_for_json(payload)
        assert out == {"a": None, "b": [None, 1.0, {"c": None}], "d": {"e": {"f": 2.0}}}


# ---------------------------------------------------------------------------
# Construction / directory layout
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_explicit_workspace_and_subdir(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        assert store._workspace == tmp_path
        assert store._store_dir == tmp_path / "mini"
        assert store._index_file == tmp_path / "mini" / "mini.json"

    def test_default_workspace_is_cwd_agent_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = _MiniStore()
        assert store._workspace == tmp_path / "_agent_data"
        assert store._store_dir == tmp_path / "_agent_data" / "mini"

    def test_empty_subdir_puts_index_at_workspace_root(self, tmp_path):
        class _FlatStore(_MiniStore):
            _subdir = ""

        store = _FlatStore(workspace_root=tmp_path)
        assert store._store_dir == tmp_path
        assert store._index_file == tmp_path / "mini.json"

    def test_lock_file_is_adjacent_to_index(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        assert store._lock_file == tmp_path / "mini" / "mini.lock"

    def test_ensure_dirs_creates_store_dir(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        assert not store._store_dir.exists()
        store._ensure_dirs()
        assert store._store_dir.is_dir()

    def test_fresh_store_has_no_entries_and_no_index_file(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        assert store._entries == []
        # Construction alone must not touch disk.
        assert not store._index_file.exists()


# ---------------------------------------------------------------------------
# NotImplemented contract on the bare base
# ---------------------------------------------------------------------------


class TestAbstractMethods:
    def test_entry_from_dict_not_implemented(self, tmp_path):
        store: BaseStore = BaseStore(workspace_root=tmp_path)
        with pytest.raises(NotImplementedError):
            store._entry_from_dict({})

    def test_entry_to_dict_not_implemented(self, tmp_path):
        store: BaseStore = BaseStore(workspace_root=tmp_path)
        with pytest.raises(NotImplementedError):
            store._entry_to_dict(object())


# ---------------------------------------------------------------------------
# Index persistence, envelope, and reload
# ---------------------------------------------------------------------------


class TestIndexPersistence:
    def test_build_index_data_envelope(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        store.add(_Item(id="1", name="a"))
        store.add(_Item(id="2", name="b"))

        data = store._build_index_data()
        assert data["version"] == INDEX_VERSION
        assert data["entry_count"] == 2
        assert {e["id"] for e in data["entries"]} == {"1", "2"}
        assert "updated" in data

    def test_save_and_reload_round_trip(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        store.add(_Item(id="1", name="alpha", score=3.5))

        reloaded = _MiniStore(workspace_root=tmp_path)
        assert len(reloaded._entries) == 1
        assert reloaded._entries[0].name == "alpha"
        assert reloaded._entries[0].score == 3.5

    def test_saved_index_is_valid_json_with_no_temp_leftovers(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        store.add(_Item(id="1"))

        index = json.loads(store._index_file.read_text())
        assert index["entry_count"] == 1
        # Atomic write must not leave temp files behind.
        leftovers = [p for p in store._store_dir.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_corrupt_index_starts_fresh(self, tmp_path):
        store_dir = tmp_path / "mini"
        store_dir.mkdir(parents=True)
        (store_dir / "mini.json").write_text("{ this is not valid json")

        store = _MiniStore(workspace_root=tmp_path)
        assert store._entries == []

    def test_post_load_index_hook_fires_on_reload(self, tmp_path):
        class _HookStore(_MiniStore):
            def _post_load_index(self) -> None:
                self.hook_calls = getattr(self, "hook_calls", 0) + 1

        # Seed an index so the reload path (which invokes the hook) runs.
        seed = _HookStore(workspace_root=tmp_path)
        seed.add(_Item(id="1"))

        loaded = _HookStore(workspace_root=tmp_path)
        assert getattr(loaded, "hook_calls", 0) >= 1

    def test_custom_parse_index_data_used_on_load(self, tmp_path):
        class _LegacyStore(_MiniStore):
            def _parse_index_data(self, data):
                # Simulate migrating a legacy top-level "items" key.
                return [self._entry_from_dict(d) for d in data.get("items", [])]

        store_dir = tmp_path / "mini"
        store_dir.mkdir(parents=True)
        (store_dir / "mini.json").write_text(json.dumps({"items": [{"id": "legacy"}]}))

        store = _LegacyStore(workspace_root=tmp_path)
        assert [e.id for e in store._entries] == ["legacy"]


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------


class TestRefreshIfStale:
    def test_reloads_when_another_writer_advances_mtime(self, tmp_path):
        reader = _MiniStore(workspace_root=tmp_path)
        assert reader._entries == []

        writer = _MiniStore(workspace_root=tmp_path)
        writer.add(_Item(id="new"))
        # Force a strictly newer mtime so the coarse-grained clock can't tie.
        import os

        newer = reader._index_file.stat().st_mtime + 10
        os.utime(reader._index_file, (newer, newer))

        reader._refresh_if_stale()
        assert [e.id for e in reader._entries] == ["new"]

    def test_no_reload_when_index_unchanged(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        store.add(_Item(id="1"))

        sentinel = object()
        store._entries.append(sentinel)  # type: ignore[arg-type]
        store._refresh_if_stale()
        # Unchanged mtime → no reload → our in-memory sentinel survives.
        assert store._entries[-1] is sentinel

    def test_missing_index_file_is_a_noop(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        store._refresh_if_stale()  # no file yet — must not raise
        assert store._entries == []


# ---------------------------------------------------------------------------
# Listener registration / notification (per-subclass isolation)
# ---------------------------------------------------------------------------


class TestListeners:
    def test_listener_lists_are_isolated_per_subclass(self):
        assert _MiniStore._listeners is not _OtherStore._listeners
        assert _MiniStore._delete_listeners is not _OtherStore._delete_listeners

    def test_register_and_notify_save_listener(self, tmp_path):
        received: list = []
        _MiniStore.register_listener(received.append)
        try:
            store = _MiniStore(workspace_root=tmp_path)
            entry = store.add(_Item(id="1"))
            assert received == [entry]
        finally:
            _MiniStore.unregister_listener(received.append)

    def test_registering_on_one_subclass_does_not_leak_to_another(self, tmp_path):
        fn = lambda _e: None  # noqa: E731
        _MiniStore.register_listener(fn)
        try:
            assert fn in _MiniStore._listeners
            assert fn not in _OtherStore._listeners
        finally:
            _MiniStore.unregister_listener(fn)

    def test_notify_swallows_listener_exceptions(self, tmp_path):
        def boom(_entry):
            raise RuntimeError("listener blew up")

        _MiniStore.register_listener(boom)
        try:
            store = _MiniStore(workspace_root=tmp_path)
            # A raising listener must not propagate out of the save path.
            store.add(_Item(id="1"))
            assert len(store._entries) == 1
        finally:
            _MiniStore.unregister_listener(boom)

    def test_delete_listeners_register_and_notify(self, tmp_path):
        received: list = []
        _MiniStore.register_delete_listener(received.append)
        try:
            store = _MiniStore(workspace_root=tmp_path)
            entry = _Item(id="1")
            store._notify_delete_listeners(entry)
            assert received == [entry]
        finally:
            _MiniStore.unregister_delete_listener(received.append)

    def test_notify_delete_swallows_listener_exceptions(self, tmp_path):
        def boom(_entry):
            raise RuntimeError("delete listener blew up")

        _MiniStore.register_delete_listener(boom)
        try:
            store = _MiniStore(workspace_root=tmp_path)
            store._notify_delete_listeners(_Item(id="1"))  # must not raise
        finally:
            _MiniStore.unregister_delete_listener(boom)


# ---------------------------------------------------------------------------
# update_entry_metadata — the locked mutation path on the base
# ---------------------------------------------------------------------------


class TestUpdateEntryMetadata:
    def test_updates_and_persists(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        store.add(_Item(id="1", name="before"))

        result = store.update_entry_metadata("1", name="after", score=9.0)
        assert result is not None
        assert result.name == "after"
        assert result.score == 9.0

        reloaded = _MiniStore(workspace_root=tmp_path)
        entry = reloaded._entries[0]
        assert entry.name == "after"
        assert entry.score == 9.0

    def test_unknown_id_returns_none(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        store.add(_Item(id="1"))
        assert store.update_entry_metadata("missing", name="x") is None

    def test_unknown_attribute_raises(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        store.add(_Item(id="1"))
        with pytest.raises(AttributeError, match="no attribute 'bogus'"):
            store.update_entry_metadata("1", bogus="oops")


# ---------------------------------------------------------------------------
# _with_index_lock — reload-under-lock and lock-file management
# ---------------------------------------------------------------------------


class TestWithIndexLock:
    def test_creates_lock_file_and_dirs(self, tmp_path):
        store = _MiniStore(workspace_root=tmp_path)
        with store._with_index_lock():
            pass
        assert store._lock_file.exists()
        assert store._store_dir.is_dir()

    def test_reloads_index_before_yield(self, tmp_path):
        first = _MiniStore(workspace_root=tmp_path)
        first.add(_Item(id="1"))

        second = _MiniStore(workspace_root=tmp_path)  # sees one entry
        # A different instance appends out of band.
        first.add(_Item(id="2"))

        # Entering the lock reloads, so ``second`` observes both before mutating.
        with second._with_index_lock():
            ids = {e.id for e in second._entries}
        assert ids == {"1", "2"}
