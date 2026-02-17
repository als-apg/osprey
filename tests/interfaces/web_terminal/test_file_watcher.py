"""Tests for file watcher and SSE broadcaster."""

from __future__ import annotations

import asyncio
import time

from osprey.interfaces.web_terminal.file_watcher import (
    FileEventBroadcaster,
    WorkspaceWatcher,
)


class TestFileEventBroadcaster:
    def test_subscribe_returns_queue(self):
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()
        assert isinstance(q, asyncio.Queue)

    def test_broadcast_delivers_to_subscribers(self):
        broadcaster = FileEventBroadcaster()
        q1 = broadcaster.subscribe()
        q2 = broadcaster.subscribe()

        broadcaster.broadcast({"type": "created", "path": "test.py"})

        assert not q1.empty()
        assert not q2.empty()
        assert q1.get_nowait()["path"] == "test.py"
        assert q2.get_nowait()["path"] == "test.py"

    def test_unsubscribe_removes_queue(self):
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()
        broadcaster.unsubscribe(q)

        broadcaster.broadcast({"type": "modified", "path": "test.py"})
        assert q.empty()

    def test_unsubscribe_nonexistent_is_safe(self):
        broadcaster = FileEventBroadcaster()
        q = asyncio.Queue()
        # Should not raise
        broadcaster.unsubscribe(q)

    def test_broadcast_drops_on_full_queue(self):
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()

        # Fill the queue (maxsize=64)
        for i in range(70):
            broadcaster.broadcast({"type": "modified", "path": f"file_{i}.py"})

        # Queue should be at max capacity, not overflowing
        assert q.qsize() <= 64


class TestWorkspaceWatcher:
    def test_start_creates_directory_if_missing(self, tmp_path):
        workspace = tmp_path / "new_workspace"
        broadcaster = FileEventBroadcaster()
        watcher = WorkspaceWatcher(workspace, broadcaster)

        watcher.start()
        try:
            assert workspace.exists()
        finally:
            watcher.stop()

    def test_start_and_stop(self, tmp_path):
        broadcaster = FileEventBroadcaster()
        watcher = WorkspaceWatcher(tmp_path, broadcaster)
        watcher.start()
        watcher.stop()
        # Should not raise on double stop
        watcher.stop()

    def test_detects_file_creation(self, tmp_path):
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()
        watcher = WorkspaceWatcher(tmp_path, broadcaster)
        watcher.start()

        try:
            # Create a file
            (tmp_path / "new_file.txt").write_text("hello")

            # Wait for event (watchdog is async, give it time)
            deadline = time.monotonic() + 3
            events = []
            while time.monotonic() < deadline:
                try:
                    event = q.get_nowait()
                    events.append(event)
                    if any(e["path"] == "new_file.txt" for e in events):
                        break
                except asyncio.QueueEmpty:
                    time.sleep(0.1)

            assert any(e["path"] == "new_file.txt" for e in events)
        finally:
            watcher.stop()

    def test_ignores_git_directory(self, tmp_path):
        broadcaster = FileEventBroadcaster()
        q = broadcaster.subscribe()
        watcher = WorkspaceWatcher(tmp_path, broadcaster)
        watcher.start()

        try:
            # Create files in .git (should be ignored)
            git_dir = tmp_path / ".git"
            git_dir.mkdir()
            (git_dir / "HEAD").write_text("ref: refs/heads/main")

            time.sleep(0.5)
            events = []
            while not q.empty():
                try:
                    events.append(q.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # Should not have events for .git paths
            git_events = [e for e in events if ".git" in e.get("path", "")]
            assert len(git_events) == 0
        finally:
            watcher.stop()
