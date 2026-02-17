"""File-system watcher for cross-process store index changes.

Watches the three store index files (data_context.json, artifacts.json,
memories.json) for modifications by external processes and broadcasts
SSE events for any added or deleted entries.

Same-process saves already update the store's ``_entries`` in memory, so
the diff naturally produces nothing — no duplicate events.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("osprey.interfaces.artifacts.store_watcher")


class _IndexFileHandler(FileSystemEventHandler):
    """Watchdog handler that reacts to modifications of known index files."""

    def __init__(
        self,
        index_configs: dict[str, dict[str, Any]],
        broadcaster: Any,
    ) -> None:
        """
        Args:
            index_configs: Map from index filename to config dict with keys:
                - ``store``: The store instance (DataContext, MemoryStore, etc.)
                - ``id_attr``: Attribute name for the entry ID (e.g. ``"id"``)
                - ``event_type``: SSE event type for new entries (e.g. ``"context"``)
                - ``delete_type``: SSE event type for deleted entries
                - ``to_dict``: Whether entries have a ``.to_dict()`` method
            broadcaster: ``_SSEBroadcaster`` instance with ``.broadcast(data)``
        """
        self._index_configs = index_configs
        self._broadcaster = broadcaster
        self._last_event: dict[str, float] = {}
        self._debounce_seconds = 0.1
        # Snapshot known entry IDs per store
        self._known_ids: dict[str, set] = {}
        for filename, cfg in index_configs.items():
            store = cfg["store"]
            id_attr = cfg["id_attr"]
            self._known_ids[filename] = {
                getattr(e, id_attr) for e in store._entries
            }

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle(event)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle(event)

    def _handle(self, event: FileSystemEvent) -> None:
        src_path = Path(event.src_path)
        filename = src_path.name

        if filename not in self._index_configs:
            return

        # Debounce: skip if same path fired within 100ms
        now = time.monotonic()
        key = str(src_path)
        last = self._last_event.get(key, 0)
        if now - last < self._debounce_seconds:
            return
        self._last_event[key] = now

        cfg = self._index_configs[filename]
        store = cfg["store"]
        id_attr = cfg["id_attr"]
        event_type = cfg["event_type"]
        delete_type = cfg["delete_type"]

        old_ids = self._known_ids[filename]

        try:
            store._load_index()
        except Exception:
            logger.warning("Failed to reload %s index; skipping", filename, exc_info=True)
            return

        new_ids = {getattr(e, id_attr) for e in store._entries}

        # Broadcast additions
        added = new_ids - old_ids
        for entry in store._entries:
            if getattr(entry, id_attr) in added:
                self._broadcaster.broadcast({
                    "type": event_type,
                    **entry.to_dict(),
                })

        # Broadcast deletions
        removed = old_ids - new_ids
        for entry_id in removed:
            self._broadcaster.broadcast({
                "type": delete_type,
                "id": entry_id,
            })

        self._known_ids[filename] = new_ids


class StoreIndexWatcher:
    """Watches store index files for cross-process changes and broadcasts SSE events."""

    def __init__(
        self,
        workspace_root: Path,
        broadcaster: Any,
        context_store: Any,
        artifact_store: Any,
        memory_store: Any,
    ) -> None:
        self._workspace_root = workspace_root
        self._broadcaster = broadcaster
        self._observer: Observer | None = None

        self._index_configs: dict[str, dict[str, Any]] = {
            "data_context.json": {
                "store": context_store,
                "id_attr": "id",
                "event_type": "context",
                "delete_type": "context_deleted",
            },
            "artifacts.json": {
                "store": artifact_store,
                "id_attr": "id",
                "event_type": "artifact",
                "delete_type": "artifact_deleted",
            },
            "memories.json": {
                "store": memory_store,
                "id_attr": "id",
                "event_type": "memory",
                "delete_type": "memory_deleted",
            },
        }

        # Directories to watch — each index file lives in a different dir
        self._watch_dirs: dict[str, Path] = {
            "data_context.json": workspace_root,
            "artifacts.json": workspace_root / "artifacts",
            "memories.json": workspace_root / "memory",
        }

    def start(self) -> None:
        """Start watching index files for changes."""
        handler = _IndexFileHandler(self._index_configs, self._broadcaster)
        self._observer = Observer()

        # Schedule a watch on each directory that contains an index file
        watched = set()
        for filename, dir_path in self._watch_dirs.items():
            dir_str = str(dir_path)
            if dir_str not in watched:
                dir_path.mkdir(parents=True, exist_ok=True)
                self._observer.schedule(handler, dir_str, recursive=False)
                watched.add(dir_str)

        self._observer.daemon = True
        self._observer.start()

    def stop(self) -> None:
        """Stop the file watcher."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
