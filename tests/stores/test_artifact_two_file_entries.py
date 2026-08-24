"""Deletion coverage for artifact entries that own two files.

An artifact normally owns exactly one file; ``data_file`` is just an
agent-facing pointer back at it. Some artifacts, though, keep a companion
alongside the primary file — an image plus the raw payload it was rendered
from — and every deletion path has to take both away, without ever following
a ``data_file`` that points outside the store directory.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from osprey.stores.artifact_store import ArtifactEntry, ArtifactStore

TIMEOUT_SECONDS = 15


@pytest.fixture
def store(tmp_path, monkeypatch) -> ArtifactStore:
    """An ArtifactStore with a real repo root, so ``data_file`` anchors resolve."""
    repo_root = tmp_path / "repo"
    (repo_root / "build").mkdir(parents=True)
    config_path = repo_root / "build" / "config.yml"
    config_path.write_text(
        f"project_root: {repo_root}\nagent_data:\n  base_dir: state/agent\n",
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.chdir(repo_root)
    return ArtifactStore(workspace_root=repo_root / "state" / "agent")


def _two_file_entry(store: ArtifactStore, title: str = "Reading") -> tuple[ArtifactEntry, Path]:
    """Save a normal entry, then give it a companion file inside the store."""
    entry = store.save_data(tool="channel_read", data={"value": 1}, title=title)
    primary = store.get_file_path(entry.id)
    assert primary is not None and primary.exists()

    companion = primary.with_suffix(".npy")
    companion.write_bytes(b"\x93NUMPY-raw-payload")
    updated = store.update_entry_metadata(
        entry.id,
        data_file=str(companion.relative_to(store.repo_root)),
    )
    assert updated is not None
    return updated, companion


def _run_isolated(fn, description: str):
    """Run *fn* on a worker thread and fail loudly instead of hanging forever."""
    box: dict[str, object] = {}

    def target() -> None:
        box["value"] = fn()

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(TIMEOUT_SECONDS)
    assert not worker.is_alive(), f"{description} did not finish — deadlock?"
    return box.get("value")


class TestTwoFileDeletion:
    def test_delete_entry_removes_both_files(self, store):
        entry, companion = _two_file_entry(store)
        primary = store.get_file_path(entry.id)

        assert store.delete_entry(entry.id) is True

        assert not primary.exists()
        assert not companion.exists()
        assert store.get_entry(entry.id) is None

    def test_delete_category_removes_both_files(self, store):
        entry, companion = _two_file_entry(store)
        primary = store.get_file_path(entry.id)
        store.update_entry_metadata(entry.id, category="channel_reading")

        removed = store.delete_category("channel_reading")

        assert [e.id for e in removed] == [entry.id]
        assert not primary.exists()
        assert not companion.exists()

    def test_delete_everything_removes_both_files(self, store):
        entry, companion = _two_file_entry(store)
        primary = store.get_file_path(entry.id)

        store.delete_everything()

        assert not primary.exists()
        assert not companion.exists()
        assert store.list_entries() == []


class TestSingleFileEntriesUnaffected:
    """Today every entry's ``data_file`` aliases its primary file — a no-op."""

    def test_save_data_pointer_is_not_treated_as_a_companion(self, store):
        entry = store.save_data(tool="archiver_read", data={"value": 1}, title="Series")

        assert entry.data_file  # the agent-facing pointer is populated ...
        assert store._entry_data_file(entry) is None  # ... but it is not a companion

    def test_delete_entry_still_removes_the_single_file(self, store):
        entry = store.save_data(tool="archiver_read", data={"value": 1}, title="Series")
        primary = store.get_file_path(entry.id)

        assert store.delete_entry(entry.id) is True
        assert not primary.exists()
        assert store.get_entry(entry.id) is None

    def test_entry_without_a_data_file_deletes_cleanly(self, store):
        entry = store.save_file(
            file_content=b"<h1>hi</h1>",
            filename="report.html",
            artifact_type="html",
            title="Report",
            description="",
            mime_type="text/html",
            tool_source="test",
        )
        primary = store.get_file_path(entry.id)

        assert store._entry_data_file(entry) is None
        assert store.delete_entry(entry.id) is True
        assert not primary.exists()

    def test_missing_companion_file_is_not_an_error(self, store):
        entry, companion = _two_file_entry(store)
        companion.unlink()

        assert store.delete_entry(entry.id) is True


class TestOutsideStoreDirectoryIsNeverUnlinked:
    """``data_file`` is index data — never a licence to unlink arbitrary paths."""

    @pytest.mark.parametrize("shape", ["relative", "absolute", "traversal"])
    def test_data_file_outside_the_store_survives(self, store, shape):
        outside = store.repo_root / "precious.npy"
        outside.write_bytes(b"do-not-touch")

        pointer = {
            "relative": "precious.npy",
            "absolute": str(outside),
            "traversal": "state/agent/artifacts/../../../precious.npy",
        }[shape]

        entry = store.save_data(tool="channel_read", data={"value": 1}, title="Escape")
        primary = store.get_file_path(entry.id)
        entry = store.update_entry_metadata(entry.id, data_file=pointer)

        assert store._entry_data_file(entry) is None
        assert store.delete_entry(entry.id) is True
        assert not primary.exists()
        assert outside.exists(), "a data_file outside the store must never be unlinked"

    def test_symlinked_companion_escaping_the_store_survives(self, store):
        outside = store.repo_root / "precious.npy"
        outside.write_bytes(b"do-not-touch")

        entry = store.save_data(tool="channel_read", data={"value": 1}, title="Symlink")
        primary = store.get_file_path(entry.id)
        link = primary.with_suffix(".npy")
        link.symlink_to(outside)
        entry = store.update_entry_metadata(
            entry.id, data_file=str(link.relative_to(store.repo_root))
        )

        assert store._entry_data_file(entry) is None
        assert store.delete_entry(entry.id) is True
        assert outside.exists()


class TestLockFreeHelper:
    def test_delete_entry_locked_works_inside_a_held_lock(self, store):
        """The whole point of the helper: usable from an open critical section.

        Re-entering the public ``delete_entry`` here would self-deadlock — the
        ``flock`` in ``base_store`` is per open-file-description and therefore
        not reentrant, unlike the in-process ``RLock`` beside it.
        """
        entry, companion = _two_file_entry(store)
        primary = store.get_file_path(entry.id)

        def delete_under_lock():
            with store._with_index_lock():
                return store._delete_entry_locked(entry.id)

        removed = _run_isolated(delete_under_lock, "_delete_entry_locked under a held lock")

        assert removed is not None and removed.id == entry.id
        assert not primary.exists()
        assert not companion.exists()
        assert store.get_entry(entry.id) is None

    def test_delete_entry_locked_persists_the_index(self, store):
        entry, _companion = _two_file_entry(store)

        def delete_under_lock():
            with store._with_index_lock():
                return store._delete_entry_locked(entry.id)

        _run_isolated(delete_under_lock, "_delete_entry_locked under a held lock")

        reopened = ArtifactStore(workspace_root=store._workspace)
        assert reopened.get_entry(entry.id) is None

    def test_persist_false_defers_the_index_write(self, store):
        """Batch callers save once; the entry is gone in memory either way."""
        entry, companion = _two_file_entry(store)

        def delete_under_lock():
            with store._with_index_lock():
                return store._delete_entry_locked(entry.id, persist=False)

        removed = _run_isolated(delete_under_lock, "_delete_entry_locked(persist=False)")

        assert removed is not None
        assert not companion.exists()
        assert store.get_entry(entry.id) is None
        reopened = ArtifactStore(workspace_root=store._workspace)
        assert reopened.get_entry(entry.id) is not None  # index not rewritten yet

    def test_delete_entry_locked_returns_none_for_an_unknown_id(self, store):
        def delete_under_lock():
            with store._with_index_lock():
                return store._delete_entry_locked("no-such-id")

        assert _run_isolated(delete_under_lock, "_delete_entry_locked(unknown)") is None
