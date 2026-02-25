"""File system watcher and SSE broadcaster for workspace changes.

FileEventBroadcaster manages per-client asyncio queues for SSE push.
WorkspaceWatcher uses watchdog to detect file changes and broadcast them.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


class FileEventBroadcaster:
    """Manages per-client asyncio.Queue instances for SSE push."""

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[dict]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=64)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        with self._lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass

    def broadcast(self, data: dict) -> None:
        """Push data to all connected SSE clients (called from sync context)."""
        with self._lock:
            for q in self._queues:
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    pass


# Patterns to ignore in file watching
_IGNORE_PATTERNS = {".git", "__pycache__", ".DS_Store", "_notebook_cache"}
_IGNORE_EXTENSIONS = {".pyc", ".pyo"}


class _WorkspaceHandler(FileSystemEventHandler):
    """Watchdog handler that filters and debounces file events."""

    def __init__(self, workspace_dir: Path, broadcaster: FileEventBroadcaster) -> None:
        self._workspace_dir = workspace_dir
        self._broadcaster = broadcaster
        self._last_event: dict[str, float] = {}
        self._debounce_seconds = 0.1

    def on_any_event(self, event: FileSystemEvent) -> None:
        src_path = Path(event.src_path)

        # Filter ignored paths
        for part in src_path.parts:
            if part in _IGNORE_PATTERNS:
                return
        if src_path.suffix in _IGNORE_EXTENSIONS:
            return

        # Debounce: skip if same path fired within 100ms
        now = time.monotonic()
        key = str(src_path)
        last = self._last_event.get(key, 0)
        if now - last < self._debounce_seconds:
            return
        self._last_event[key] = now

        # Map watchdog event types to our simpler types
        event_type_map = {
            "created": "created",
            "modified": "modified",
            "deleted": "deleted",
            "moved": "modified",
            "closed": "modified",
        }
        simple_type = event_type_map.get(event.event_type)
        if simple_type is None:
            return

        try:
            relative = src_path.relative_to(self._workspace_dir)
        except ValueError:
            return

        self._broadcaster.broadcast(
            {
                "type": simple_type,
                "path": str(relative),
                "is_dir": event.is_directory,
            }
        )


class WorkspaceWatcher:
    """Watches a workspace directory for file changes using watchdog."""

    def __init__(self, workspace_dir: Path, broadcaster: FileEventBroadcaster) -> None:
        self._workspace_dir = workspace_dir
        self._broadcaster = broadcaster
        self._observer: Observer | None = None

    def start(self) -> None:
        """Start watching the workspace directory."""
        if not self._workspace_dir.exists():
            self._workspace_dir.mkdir(parents=True, exist_ok=True)

        handler = _WorkspaceHandler(self._workspace_dir, self._broadcaster)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._workspace_dir), recursive=True)
        self._observer.daemon = True
        self._observer.start()

    def stop(self) -> None:
        """Stop the file watcher."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
