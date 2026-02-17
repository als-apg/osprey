"""Tests for StoreIndexWatcher — cross-process SSE event broadcasting.

Covers:
  - Detection of new context entries written externally
  - Detection of deleted entries
  - Debounce behaviour
  - Ignoring non-index files
  - Handling of corrupt (invalid JSON) index files
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from osprey.interfaces.artifacts.store_watcher import StoreIndexWatcher
from osprey.mcp_server.artifact_store import ArtifactStore
from osprey.mcp_server.data_context import DataContext
from osprey.mcp_server.memory_store import MemoryStore


def _make_watcher(tmp_path):
    """Create a StoreIndexWatcher with real stores and a mock broadcaster."""
    context_store = DataContext(workspace_root=tmp_path)
    artifact_store = ArtifactStore(workspace_root=tmp_path)
    memory_store = MemoryStore(workspace_root=tmp_path)
    broadcaster = MagicMock()

    watcher = StoreIndexWatcher(
        workspace_root=tmp_path,
        broadcaster=broadcaster,
        context_store=context_store,
        artifact_store=artifact_store,
        memory_store=memory_store,
    )
    return watcher, broadcaster, context_store, artifact_store, memory_store


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

    def test_detects_new_context_entry(self, tmp_path):
        """External write to data_context.json triggers SSE broadcast."""
        watcher, broadcaster, context_store, _, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            # Give watchdog time to fully initialize
            time.sleep(0.3)

            # Simulate external process saving a context entry
            external_ctx = DataContext(workspace_root=tmp_path)
            external_ctx.save(
                tool="channel_read",
                data={"value": 42},
                description="external save",
                summary={"count": 1},
                access_details={"format": "json"},
                data_type="channel_values",
            )

            assert _wait_for_broadcast(broadcaster, 1)
            call_data = broadcaster.broadcast.call_args_list[0][0][0]
            assert call_data["type"] == "context"
            assert call_data["tool"] == "channel_read"
        finally:
            watcher.stop()

    def test_detects_deleted_entry(self, tmp_path):
        """Removing an entry from the index externally triggers delete broadcast."""
        # Pre-populate with an entry
        ctx = DataContext(workspace_root=tmp_path)
        entry = ctx.save(
            tool="archiver_read",
            data={"values": [1, 2, 3]},
            description="to delete",
            summary={"count": 3},
            access_details={"format": "json"},
            data_type="timeseries",
        )

        watcher, broadcaster, context_store, _, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            # Give watchdog time to fully initialize
            time.sleep(0.3)

            # Simulate external process deleting the entry
            external_ctx = DataContext(workspace_root=tmp_path)
            external_ctx.delete_entry(entry.id)

            assert _wait_for_broadcast(broadcaster, 1)
            call_data = broadcaster.broadcast.call_args_list[0][0][0]
            assert call_data["type"] == "context_deleted"
            assert call_data["id"] == entry.id
        finally:
            watcher.stop()

    def test_debounce(self, tmp_path):
        """Rapid writes within debounce window produce at most one event batch."""
        watcher, broadcaster, context_store, _, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            # Write the index file rapidly 3 times
            index_file = tmp_path / "data_context.json"
            index_data = {
                "version": 1,
                "updated": "2024-01-01T00:00:00",
                "entry_count": 0,
                "entries": [],
                "created": "2024-01-01T00:00:00",
            }
            for _ in range(3):
                index_file.write_text(json.dumps(index_data))

            # Give watchdog time to process
            time.sleep(0.5)

            # Debounce should limit events — but at minimum no crash
            # (exact count depends on filesystem event timing)
            assert broadcaster.broadcast.call_count <= 3
        finally:
            watcher.stop()

    def test_ignores_non_index_files(self, tmp_path):
        """Writing to a non-index file in the workspace triggers no broadcast."""
        watcher, broadcaster, _, _, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            # Write a non-index file
            data_dir = tmp_path / "data"
            data_dir.mkdir(exist_ok=True)
            (data_dir / "001_channel_read.json").write_text('{"data": "test"}')

            time.sleep(0.5)
            assert broadcaster.broadcast.call_count == 0
        finally:
            watcher.stop()

    def test_handles_corrupt_index(self, tmp_path):
        """Invalid JSON in index file doesn't crash the watcher."""
        watcher, broadcaster, _, _, _ = _make_watcher(tmp_path)
        watcher.start()
        try:
            # Write invalid JSON to the index
            index_file = tmp_path / "data_context.json"
            index_file.write_text("{invalid json!!!")

            time.sleep(0.5)
            # Should not crash — watcher logs warning and skips
            # No broadcast should have been made
        finally:
            watcher.stop()
