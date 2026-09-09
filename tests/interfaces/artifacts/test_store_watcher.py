"""Tests for StoreIndexWatcher — cross-process SSE event broadcasting.

Covers:
  - Detection of new artifact entries written externally
  - Detection of deleted entries
  - Debounce behaviour
  - Ignoring non-index files
  - Handling of corrupt (invalid JSON) index files
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from osprey.interfaces.artifacts.store_watcher import StoreIndexWatcher
from osprey.stores.artifact_store import ArtifactStore
from tests.interfaces.fsevents_wait import poke_until


def _make_watcher(tmp_path):
    """Create a StoreIndexWatcher with real stores and a mock broadcaster."""
    artifact_store = ArtifactStore(workspace_root=tmp_path)
    broadcaster = MagicMock()

    watcher = StoreIndexWatcher(
        workspace_root=tmp_path,
        broadcaster=broadcaster,
        artifact_store=artifact_store,
    )
    return watcher, broadcaster, artifact_store


def _wait_for_broadcast(broadcaster, external_store, expected_calls=1, *, what):
    """Wait until the broadcaster has been called at least *expected_calls* times.

    Not a longer timeout. The observer's stream can still be arming when the
    external write lands, and a change made inside that window is delivered late
    or not at all — no ceiling recovers an event the stream never saw. So this
    re-applies the stimulus by calling ``_save_index()`` on the store that made
    it: the same tempfile-plus-``os.replace`` the real save and delete go
    through, writing the identical index the test already produced.

    The write path matters more than the file does. An atomic replace is
    delivered by Linux inotify as ``on_moved`` and nothing else — that is why
    ``StoreIndexWatcher`` overrides ``on_moved`` and routes on ``dest_path``.
    Poking with a plain in-place rewrite would substitute an ``on_modified``,
    and a regression in the ``on_moved`` route would then be answered by the
    poke's own event class and pass on CI. Going back through the store keeps
    the poke indistinguishable from the stimulus under test.

    The handler answers each replace by re-reading the index and diffing against
    the ids it snapshotted at ``start()``, so a re-save re-delivers exactly the
    addition or removal the test made — the poke carries no state of its own.

    Args:
        broadcaster: The ``MagicMock`` standing in for the SSE broadcaster.
        external_store: The store whose write produced the change under test.
        expected_calls: Broadcasts to wait for.
        what: Named in the failure message.
    """
    poke_until(
        lambda: broadcaster.broadcast.call_count >= expected_calls,
        external_store._save_index,
        what=what,
    )


@pytest.mark.unit
class TestStoreWatcher:
    """Tests for StoreIndexWatcher."""

    # No ``flaky`` marker any more. These two used to carry ``reruns=2`` because
    # the OS occasionally missed the change event entirely, and a rerun was the
    # only way to get a freshly armed observer. ``_wait_for_broadcast`` now
    # re-applies the stimulus instead, which arms in-test and converges without
    # a second attempt — so a red here is a real regression again, not weather.
    def test_detects_new_artifact_entry(self, tmp_path):
        """External write to artifacts.json triggers SSE broadcast."""
        watcher, broadcaster, artifact_store = _make_watcher(tmp_path)
        watcher.start()
        try:
            # Simulate external process saving an artifact
            external_store = ArtifactStore(workspace_root=tmp_path)
            external_store.save_file(
                file_content=b"<html>test</html>",
                filename="test.html",
                artifact_type="plot_html",
                title="External Plot",
                description="externally saved",
                mime_type="text/html",
                tool_source="test",
            )

            _wait_for_broadcast(
                broadcaster,
                external_store,
                what="the broadcast for an externally saved artifact",
            )
            call_data = broadcaster.broadcast.call_args_list[0][0][0]
            assert call_data["type"] == "artifact"
            assert call_data["title"] == "External Plot"
        finally:
            watcher.stop()

    def test_detects_deleted_entry(self, tmp_path):
        """Removing an entry from the index externally triggers delete broadcast."""
        # Pre-populate with an artifact
        pre_store = ArtifactStore(workspace_root=tmp_path)
        entry = pre_store.save_file(
            file_content=b"<html>delete me</html>",
            filename="delete.html",
            artifact_type="plot_html",
            title="To Delete",
            description="will be deleted",
            mime_type="text/html",
            tool_source="test",
        )

        watcher, broadcaster, artifact_store = _make_watcher(tmp_path)
        watcher.start()
        try:
            # Simulate external process deleting the entry
            external_store = ArtifactStore(workspace_root=tmp_path)
            external_store.delete_entry(entry.id)

            _wait_for_broadcast(
                broadcaster,
                external_store,
                what="the broadcast for an externally deleted artifact",
            )
            call_data = broadcaster.broadcast.call_args_list[0][0][0]
            assert call_data["type"] == "artifact_deleted"
            assert call_data["id"] == entry.id
        finally:
            watcher.stop()

    def test_debounce(self, tmp_path):
        """Rapid writes within debounce window produce at most one event batch."""
        watcher, broadcaster, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            # Write the index file rapidly 3 times
            artifacts_dir = tmp_path / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            index_file = artifacts_dir / "artifacts.json"
            index_data = {
                "version": 1,
                "updated": "2024-01-01T00:00:00",
                "entry_count": 0,
                "entries": [],
                "created": "2024-01-01T00:00:00",
            }
            for _ in range(3):
                index_file.write_text(json.dumps(index_data))

            time.sleep(0.5)

            # Debounce should limit events — but at minimum no crash
            assert broadcaster.broadcast.call_count <= 3
        finally:
            watcher.stop()

    def test_ignores_non_index_files(self, tmp_path):
        """Writing to a non-index file in the workspace triggers no broadcast."""
        watcher, broadcaster, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            artifacts_dir = tmp_path / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "random_file.json").write_text('{"data": "test"}')

            time.sleep(0.5)
            assert broadcaster.broadcast.call_count == 0
        finally:
            watcher.stop()

    def test_handles_corrupt_index(self, tmp_path):
        """Invalid JSON in index file doesn't crash the watcher."""
        watcher, broadcaster, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            artifacts_dir = tmp_path / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            index_file = artifacts_dir / "artifacts.json"
            index_file.write_text("{invalid json!!!")

            time.sleep(0.5)
            # Should not crash — watcher logs warning and skips
        finally:
            watcher.stop()
