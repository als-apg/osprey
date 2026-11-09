"""File-system watcher for cross-process store index changes.

Watches the store index file (artifacts.json) for modifications by external
processes and broadcasts SSE events for every entry that was added, changed or
removed there, plus the current head of every run group such a change touched.

The baseline is a content hash per entry id, so a rewrite that changes an
existing entry (a settle, a turn-end transcript stamp) is announced as an
update, and a re-read of an unchanged index announces nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from watchdog.events import DirModifiedEvent, FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from osprey.interfaces.fs_watch import (
    ChangeStamp,
    ObserverFactory,
    Reconciler,
    entry_stamp,
    evict_subtree,
    reconcile_interval_seconds,
    reconcile_targets,
    refresh_listing,
)

logger = logging.getLogger("osprey.interfaces.artifacts.store_watcher")


class _Known:
    """What the handler last saw of one entry: its content hash and run group."""

    __slots__ = ("digest", "exec_id")

    def __init__(self, digest: str, exec_id: str) -> None:
        self.digest = digest
        self.exec_id = exec_id


def _content_hash(entry: Any) -> str:
    """A stable digest of everything *entry* stores in the index.

    Hashes the entry's fields as they are, not ``to_dict()``: that deep-copies
    and sanitizes every nested value, and this runs over the whole index on
    every write another process makes to it, so on a store of thousands of
    entries the copy is most of the cost. Any field that changes still changes
    the digest.
    """
    encoded = json.dumps(vars(entry), sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _snapshot(store: Any, id_attr: str) -> dict[str, _Known]:
    """Every entry of *store* mapped by id to its :class:`_Known` record."""
    return {
        getattr(e, id_attr): _Known(_content_hash(e), getattr(e, "exec_id", "") or "")
        for e in store._entries
    }


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
                - ``update_type``: SSE event type for changed entries and for
                  the re-emitted head of a run group a change touched
                - ``payload``: Callable turning an entry into the dict of
                  fields its SSE event carries after ``type``
            broadcaster: ``_SSEBroadcaster`` instance with ``.broadcast(data)``
        """
        self._index_configs = index_configs
        self._broadcaster = broadcaster
        self._last_event: dict[str, float] = {}
        self._last_stamp: dict[str, ChangeStamp | None] = {}
        self._debounce_seconds = 0.1
        self._listings: dict[str, dict[str, ChangeStamp]] = {}
        # The listing map has two writers: the observer's emitter thread and the
        # reconciliation thread. :meth:`_rescan_directory` reads a listing,
        # scans the directory and writes it back before routing what differs,
        # and two of those interleaved would route a change against a listing
        # that already holds it. Re-entrant because a reconciliation pass enters
        # this handler through :meth:`on_modified` and so takes the lock twice.
        self._dispatch_lock = threading.RLock()
        # Snapshot each store's entries: id -> content hash and run group.
        self._known: dict[str, dict[str, _Known]] = {
            filename: _snapshot(cfg["store"], cfg["id_attr"])
            for filename, cfg in index_configs.items()
        }

    def on_modified(self, event: FileSystemEvent) -> None:
        with self._dispatch_lock:
            if event.is_directory:
                self._rescan_directory(event)
                return
            self._handle(event)

    def on_created(self, event: FileSystemEvent) -> None:
        with self._dispatch_lock:
            if event.is_directory:
                return
            self._handle(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        # A directory that is gone keeps no listing, and neither does anything
        # that was under it — whether it left by rename, which :meth:`on_moved`
        # reads, or by deletion. What the map holds is decided by the tree
        # rather than by how finely the removal was reported, so it stays
        # bounded by the tree rather than by the watcher's lifetime.
        with self._dispatch_lock:
            if event.is_directory:
                evict_subtree(self._listings, Path(os.fsdecode(event.src_path)))

    def on_moved(self, event: FileSystemEvent) -> None:
        with self._dispatch_lock:
            self._dispatch_move(event)

    def _dispatch_move(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            # A directory that leaves by rename. Nothing is routed below: a
            # moved directory is not an index write. Its listing goes, and so
            # does every one beneath it, which no later event would evict. A
            # destination inside the watched tree is evicted as well, so that
            # it is rebuilt by its next frame rather than answered from
            # whatever stood there before.
            evict_subtree(self._listings, Path(os.fsdecode(event.src_path)))
            dest_path = getattr(event, "dest_path", "")
            if dest_path:
                evict_subtree(self._listings, Path(os.fsdecode(dest_path)))
            return
        # Atomic index writes (tempfile + ``os.replace``) arrive as a move whose
        # destination is the index file — on Linux inotify this is the only
        # event delivered, never ``on_modified``/``on_created`` — so route on the
        # destination path.
        dest_path = getattr(event, "dest_path", "")
        self._handle(event, path=str(dest_path) if dest_path else None)

    def reconcile(self, roots: Sequence[Path]) -> None:
        """Re-read the watched directories and route what the stream did not.

        The notification stream is the fast trigger and the only one that can go
        quiet. This is the slow one: each directory in *roots*, then every
        directory this handler already holds a listing for, is dispatched as the
        coalesced directory frame the stream owes it — through
        :meth:`on_modified`, so the pass is read as exactly the frame
        :meth:`_rescan_directory` already documents and can route nothing a
        per-file event would not have routed.

        Nothing is primed first, and nothing needs to be: this handler's
        baseline is the per-entry content-hash snapshot ``__init__`` takes from
        each store, and :meth:`_handle` broadcasts only entries that differ from
        it. A first pass that re-reads an unchanged index therefore announces
        nothing. That diff is also what makes a pass harmless after the stream
        has delivered an index write: the hashes match, so nothing is broadcast
        twice.

        A pass reads no directory below a watch directory, and there is nothing
        there for it to find: this handler routes only the index filenames it
        was given, and each of those lives in a watch directory by
        construction. A pass that listed a subdirectory would be reading a
        directory this watcher's own observer was never asked about, and would
        spend a listing per interval on it for as long as it kept changing.

        The sequence is built first because dispatching mutates the map.
        """
        with self._dispatch_lock:
            for directory in reconcile_targets(roots, self._listings):
                self.on_modified(DirModifiedEvent(str(directory)))

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

        The map is left to :func:`~osprey.interfaces.fs_watch.refresh_listing`,
        which keeps one listing per directory that still exists, so one found
        already gone is dropped there rather than remembered as empty. One
        removed the ordinary way is dropped by :meth:`on_deleted` instead.
        """
        directory = Path(os.fsdecode(event.src_path))
        previous, current = refresh_listing(self._listings, directory)

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
        src_path = Path(path) if path is not None else Path(os.fsdecode(event.src_path))
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
        stamp = entry_stamp(src_path)
        within_window = now - self._last_event.get(key, 0) < self._debounce_seconds
        if within_window and key in self._last_stamp and self._last_stamp[key] == stamp:
            return

        cfg = self._index_configs[filename]
        store = cfg["store"]
        id_attr = cfg["id_attr"]

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

        old = self._known[filename]
        new = _snapshot(store, id_attr)
        self._known[filename] = new
        self._broadcast_diff(cfg, old, new)

    def _broadcast_diff(
        self, cfg: dict[str, Any], old: dict[str, _Known], new: dict[str, _Known]
    ) -> None:
        """Announce what differs between two snapshots of one index.

        Added ids go out as ``event_type`` and changed entries as
        ``update_type``, both with the full payload; removed ids go out as
        ``delete_type``. Every run group an added, changed or removed entry
        belongs to then has its current head re-emitted as ``update_type``,
        because a member's change can change the head's payload (its status,
        its parts, or which entry heads the group at all). A head already
        announced in this pass is not announced twice, and a group with no
        members left has nothing to re-emit.
        """
        store = cfg["store"]
        id_attr = cfg["id_attr"]
        update_type = cfg.get("update_type", "artifact_updated")
        by_id = {getattr(e, id_attr): e for e in store._entries}

        added = [i for i in new if i not in old]
        changed = [i for i in new if i in old and new[i].digest != old[i].digest]
        removed = [i for i in old if i not in new]

        touched: dict[str, None] = {}
        for entry_id in (*added, *changed):
            if new[entry_id].exec_id:
                touched[new[entry_id].exec_id] = None
        for entry_id in removed:
            if old[entry_id].exec_id:
                touched[old[entry_id].exec_id] = None

        announced: set[str] = set()
        for entry_id in added:
            self._broadcaster.broadcast(
                {"type": cfg["event_type"], **cfg["payload"](by_id[entry_id])}
            )
            announced.add(entry_id)
        for entry_id in changed:
            self._broadcaster.broadcast({"type": update_type, **cfg["payload"](by_id[entry_id])})
            announced.add(entry_id)
        for entry_id in removed:
            self._broadcaster.broadcast({"type": cfg["delete_type"], "id": entry_id})

        run_group = getattr(store, "run_group", None)
        if run_group is None:
            return
        for exec_id in touched:
            group = run_group(exec_id)
            head_id = group.head_id if group is not None else ""
            if not head_id or head_id in announced or head_id not in by_id:
                continue
            self._broadcaster.broadcast({"type": update_type, **cfg["payload"](by_id[head_id])})
            announced.add(head_id)


class StoreIndexWatcher:
    """Watches store index files for cross-process changes and broadcasts SSE events."""

    def __init__(
        self,
        workspace_root: Path,
        broadcaster: Any,
        artifact_store: Any,
        *,
        payload: Callable[[Any], dict[str, Any]],
        observer_factory: ObserverFactory = Observer,
    ) -> None:
        """
        Args:
            workspace_root: Deployment workspace the index files live under.
            broadcaster: SSE broadcaster the index diffs are published to.
            artifact_store: Store whose index this watcher reloads.
            observer_factory: Builds the watchdog observer — see
                :data:`ObserverFactory`.
            payload: Turns an added or changed entry into the fields its SSE frame carries
                after ``type``. The gallery passes the same function its
                same-process frames use, so an entry another process saved
                reaches the browser with the same fields, ``run_record``
                included, as one this process saved.
        """
        self._broadcaster = broadcaster
        self._observer_factory = observer_factory
        self._observer: BaseObserver | None = None
        self._handler: _IndexFileHandler | None = None
        self._reconciler: Reconciler | None = None

        self._index_configs: dict[str, dict[str, Any]] = {
            "artifacts.json": {
                "store": artifact_store,
                "id_attr": "id",
                "event_type": "artifact",
                "delete_type": "artifact_deleted",
                "update_type": "artifact_updated",
                "payload": payload,
            },
        }

        # Directories to watch — each index file lives in a different dir
        self._watch_dirs: dict[str, Path] = {
            "artifacts.json": workspace_root / "artifacts",
        }

    def start(self) -> None:
        """Start watching index files for changes.

        Two triggers: the observer, and the reconciliation pass that re-reads
        the watched directories on an interval so an index write the
        notification stream never delivers still reaches the gallery.
        """
        handler = _IndexFileHandler(self._index_configs, self._broadcaster)
        self._handler = handler
        self._observer = self._observer_factory()

        # Schedule a watch on each directory that contains an index file
        watched = set()
        for dir_path in self._watch_dirs.values():
            dir_str = str(dir_path)
            if dir_str not in watched:
                dir_path.mkdir(parents=True, exist_ok=True)
                self._observer.schedule(handler, dir_str, recursive=False)
                watched.add(dir_str)

        self._observer.daemon = True
        self._observer.start()

        roots = tuple(Path(dir_str) for dir_str in sorted(watched))
        self._reconciler = Reconciler(
            reconcile_interval_seconds(), lambda: handler.reconcile(roots)
        )
        self._reconciler.start()

    def stop(self) -> None:
        """Stop the file watcher."""
        if self._reconciler is not None:
            self._reconciler.stop()
            self._reconciler = None
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        self._handler = None
