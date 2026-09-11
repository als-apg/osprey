"""File-system watcher for cross-process store index changes.

Watches the store index file (artifacts.json) for modifications by external
processes and broadcasts SSE events for any added or deleted entries.

Same-process saves already update the store's ``_entries`` in memory, so
the diff naturally produces nothing — no duplicate events.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from osprey.interfaces.fs_watch import ChangeStamp, ObserverFactory, one_level_listing

logger = logging.getLogger("osprey.interfaces.artifacts.store_watcher")


def _change_stamp(path: Path) -> tuple[int, int] | None:
    """``(mtime_ns, size)`` of *path*, or ``None`` when it is not there.

    Enough to tell one write from the next without reading the file: two
    reports of the same write carry the same stamp, and a second write moves
    it.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


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
                - ``store``: The store instance (e.g. ArtifactStore)
                - ``id_attr``: Attribute name for the entry ID (e.g. ``"id"``)
                - ``event_type``: SSE event type for new entries (e.g. ``"context"``)
                - ``delete_type``: SSE event type for deleted entries
                - ``to_dict``: Whether entries have a ``.to_dict()`` method
            broadcaster: ``_SSEBroadcaster`` instance with ``.broadcast(data)``
        """
        self._index_configs = index_configs
        self._broadcaster = broadcaster
        self._last_event: dict[str, float] = {}
        self._last_stamp: dict[str, tuple[int, int] | None] = {}
        self._debounce_seconds = 0.1
        self._listings: dict[str, dict[str, ChangeStamp]] = {}
        # Snapshot known entry IDs per store
        self._known_ids: dict[str, set] = {}
        for filename, cfg in index_configs.items():
            store = cfg["store"]
            id_attr = cfg["id_attr"]
            self._known_ids[filename] = {getattr(e, id_attr) for e in store._entries}

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            self._rescan_directory(event)
            return
        self._handle(event)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        # A directory that is gone keeps no listing. The rescan below evicts
        # one only when a *frame* arrives for a path that has already gone;
        # a directory removed the ordinary way is announced by this event and
        # by no other, so without this the entry would outlive the directory.
        if event.is_directory:
            self._listings.pop(str(Path(os.fsdecode(event.src_path))), None)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # Atomic index writes (tempfile + ``os.replace``) arrive as a move whose
        # destination is the index file — on Linux inotify this is the only
        # event delivered, never ``on_modified``/``on_created`` — so route on the
        # destination path.
        dest_path = getattr(event, "dest_path", "")
        self._handle(event, path=str(dest_path) if dest_path else None)

    def _rescan_directory(self, event: FileSystemEvent) -> None:
        """Read a directory frame as the index write it stands for.

        FSEvents coalesces: a write to ``artifacts.json`` can arrive as one
        ``modified`` event on the directory holding it, with no per-file event
        behind it. A directory frame carries no per-file event, so returning on
        it drops the index write. The directory is listed one level deep
        instead, compared with the listing last taken of it, and every name
        that differs is routed through :meth:`_handle` exactly as a per-file
        event would have been. Only the index filenames this watcher was given
        do anything there, so the rescan cannot broadcast anything a per-file
        event would not have.

        One level, because that is what the frame names: a write in a
        subdirectory produces its own frame for its own directory.

        One listing is kept per directory that still exists: one found already
        gone is dropped here rather than remembered as empty, and one removed
        the ordinary way is dropped by :meth:`on_deleted`.
        """
        directory = Path(os.fsdecode(event.src_path))
        key = str(directory)
        previous = self._listings.get(key)
        current = one_level_listing(directory)
        if directory.is_dir():
            self._listings[key] = current
        else:
            self._listings.pop(key, None)

        changed = [
            name
            for name, stamp in current.items()
            if previous is None or previous.get(name) != stamp
        ]
        changed.extend(sorted((previous or {}).keys() - current.keys()))
        for name in changed:
            if name in self._index_configs:
                self._handle(event, path=str(directory / name))

    def _handle(self, event: FileSystemEvent, path: str | None = None) -> None:
        src_path = Path(path) if path is not None else Path(event.src_path)
        filename = src_path.name

        if filename not in self._index_configs:
            return

        # Debounce: skip a repeat report of the index this handler last read.
        # A directory frame and the per-file event for one write arrive
        # milliseconds apart and a frame delivered before the index is on disk
        # reads it unchanged, so closing the window on the clock alone would
        # drop the event that did carry the entry — and nothing follows it. The
        # window closes on a change already read instead: same stamp, same
        # read.
        now = time.monotonic()
        key = str(src_path)
        stamp = _change_stamp(src_path)
        within_window = now - self._last_event.get(key, 0) < self._debounce_seconds
        if within_window and key in self._last_stamp and self._last_stamp[key] == stamp:
            return

        cfg = self._index_configs[filename]
        store = cfg["store"]
        id_attr = cfg["id_attr"]
        event_type = cfg["event_type"]
        delete_type = cfg["delete_type"]

        old_ids = self._known_ids[filename]

        try:
            store._load_index()
        except Exception:
            # No slot claimed: a half-written index that parses on the next
            # frame must not have been debounced away by the attempt that
            # failed on it.
            logger.warning("Failed to reload %s index; skipping", filename, exc_info=True)
            return

        self._last_event[key] = now
        self._last_stamp[key] = stamp

        new_ids = {getattr(e, id_attr) for e in store._entries}

        # Broadcast additions
        added = new_ids - old_ids
        for entry in store._entries:
            if getattr(entry, id_attr) in added:
                self._broadcaster.broadcast(
                    {
                        "type": event_type,
                        **entry.to_dict(),
                    }
                )

        # Broadcast deletions
        removed = old_ids - new_ids
        for entry_id in removed:
            self._broadcaster.broadcast(
                {
                    "type": delete_type,
                    "id": entry_id,
                }
            )

        self._known_ids[filename] = new_ids


class StoreIndexWatcher:
    """Watches store index files for cross-process changes and broadcasts SSE events."""

    def __init__(
        self,
        workspace_root: Path,
        broadcaster: Any,
        artifact_store: Any,
        *,
        observer_factory: ObserverFactory = Observer,
    ) -> None:
        """
        Args:
            workspace_root: Deployment workspace the index files live under.
            broadcaster: SSE broadcaster the index diffs are published to.
            artifact_store: Store whose index this watcher reloads.
            observer_factory: Builds the watchdog observer — see
                :data:`ObserverFactory`.
        """
        self._workspace_root = workspace_root
        self._broadcaster = broadcaster
        self._observer_factory = observer_factory
        self._observer: BaseObserver | None = None

        self._index_configs: dict[str, dict[str, Any]] = {
            "artifacts.json": {
                "store": artifact_store,
                "id_attr": "id",
                "event_type": "artifact",
                "delete_type": "artifact_deleted",
            },
        }

        # Directories to watch — each index file lives in a different dir
        self._watch_dirs: dict[str, Path] = {
            "artifacts.json": workspace_root / "artifacts",
        }

    def start(self) -> None:
        """Start watching index files for changes."""
        handler = _IndexFileHandler(self._index_configs, self._broadcaster)
        self._observer = self._observer_factory()

        # Schedule a watch on each directory that contains an index file
        watched = set()
        for _filename, dir_path in self._watch_dirs.items():
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
