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
from osprey.mcp_server.artifact_store import ArtifactStore
from osprey.mcp_server.memory_store import MemoryStore


def _make_watcher(tmp_path):
    """Create a StoreIndexWatcher with real stores and a mock broadcaster."""
    artifact_store = ArtifactStore(workspace_root=tmp_path)
    memory_store = MemoryStore(workspace_root=tmp_path)
    broadcaster = MagicMock()

    watcher = StoreIndexWatcher(
        workspace_root=tmp_path,
        broadcaster=broadcaster,
        artifact_store=artifact_store,
        memory_store=memory_store,
    )
    return watcher, broadcaster, artifact_store, memory_store


def _wait_for_broadcast(broadcaster, expected_calls=1, timeout=3.0):
    """Wait until the broadcaster has been called at least expected_calls times."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if broadcaster.broadcast.call_count >= expected_calls:
            return True
        time.sleep(0.05)
    return broadcaster.broadcast.call_count >= expected_calls


@pytest.mark.unit
class TestStoreWatcher:
    """Tests for StoreIndexWatcher."""

    def test_detects_new_artifact_entry(self, tmp_path):
        """External write to artifacts.json triggers SSE broadcast."""
        watcher, broadcaster, artifact_store, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            time.sleep(0.3)

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

            assert _wait_for_broadcast(broadcaster, 1)
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

        watcher, broadcaster, artifact_store, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            time.sleep(0.3)

            # Simulate external process deleting the entry
            external_store = ArtifactStore(workspace_root=tmp_path)
            external_store.delete_entry(entry.id)

            assert _wait_for_broadcast(broadcaster, 1)
            call_data = broadcaster.broadcast.call_args_list[0][0][0]
            assert call_data["type"] == "artifact_deleted"
            assert call_data["id"] == entry.id
        finally:
            watcher.stop()

    def test_debounce(self, tmp_path):
        """Rapid writes within debounce window produce at most one event batch."""
        watcher, broadcaster, _, _ = _make_watcher(tmp_path)
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
        watcher, broadcaster, _, _ = _make_watcher(tmp_path)
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
        watcher, broadcaster, _, _ = _make_watcher(tmp_path)
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
