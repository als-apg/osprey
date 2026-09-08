"""File system watcher and SSE broadcaster for workspace changes.

FileEventBroadcaster manages per-client asyncio queues for SSE push.
WorkspaceWatcher uses watchdog to detect file changes and broadcast them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path, PurePath

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


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


def _one_level_listing(directory: Path) -> dict[str, tuple[bool, int, int]]:
    """One level of *directory* as ``{name: (is_dir, mtime_ns, size)}``.

    The tuple is a change stamp, not metadata anyone reads: two listings of the
    same directory differ at exactly the names that were added, removed, or
    written to. A directory that has gone away between the frame and the scan
    lists as empty rather than raising — the frame is a report about the past.
    """
    listing: dict[str, tuple[bool, int, int]] = {}
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return listing
    for entry in entries:
        try:
            stat = entry.stat()
            listing[entry.name] = (entry.is_dir(), stat.st_mtime_ns, stat.st_size)
        except OSError:  # vanished mid-scan
            continue
    return listing


def filesystem_is_case_insensitive(directory: Path) -> bool:
    """Whether *directory* sits on a filesystem that matches names case-insensitively.

    Probed by re-spelling a name that already exists rather than by writing a
    probe file: this module watches the user's workspace, and creating a file
    there would fire the very events it exists to filter.

    The probe looks **inside** *directory* first, re-spelling one of its
    children. That is the only place that measures the right filesystem: a
    workspace which is itself a mount point — the ``{user}-agent-data`` volume
    this project ships — sits on a different filesystem from its parent. Only
    when the directory is empty, unreadable, or holds nothing with a cased
    character does it fall back to re-spelling the directory's own name in its
    parent, which describes the **parent's** filesystem and may therefore be
    wrong for a mount point.

    ``False`` means "compare path segments exactly", which is merely the prior
    behavior — it is *not* a safe default. On a host that really is
    case-insensitive, a ``False`` here reopens the bypass this probe exists to
    close, so it works to find a usable name before giving up.
    """
    try:
        cased_child = next(
            (child.name for child in directory.iterdir() if child.name.swapcase() != child.name),
            None,
        )
    except OSError:
        cased_child = None

    if cased_child is not None:
        try:
            return (directory / cased_child.swapcase()).exists()
        except OSError:
            return False

    flipped = directory.name.swapcase()
    if flipped == directory.name:
        return False
    try:
        return (directory.parent / flipped).exists()
    except OSError:
        return False


def resolve_store_rel(store_dir: Path, workspace_dir: Path) -> PurePath | None:
    """Workspace-relative location of one server-side store, for the watcher.

    Called once per store the watched tree may contain — the feedback records
    and the per-user bar layout both land under the agent-data root — and the
    results are handed to the watcher together as ``concealed``.

    ``None`` when the store lies outside the watched tree — the
    ``web_terminal.watch_dir`` case — where the watcher has nothing to conceal.
    A bare ``relative_to`` would raise there and abort startup.

    On a case-insensitive filesystem the containment test is case-folded:
    ``resolve_shared_data_root()`` and ``watch_dir`` can spell one directory
    two ways, and an exact comparison would then yield ``None`` and silently
    disable concealment altogether.

    Also ``None`` when the store *is* the watched tree. That relative path has
    no segments, which every path in the tree trivially starts with — the
    handler would drop every event and silently black out the file panel.
    There is no meaningful concealment to do in that configuration.
    """
    if store_dir.is_relative_to(workspace_dir):
        store_rel = store_dir.relative_to(workspace_dir)
    else:
        if not filesystem_is_case_insensitive(workspace_dir):
            return None
        workspace_parts = tuple(part.casefold() for part in workspace_dir.parts)
        store_parts = store_dir.parts
        head = tuple(part.casefold() for part in store_parts[: len(workspace_parts)])
        if head != workspace_parts:
            return None
        store_rel = PurePath(*store_parts[len(workspace_parts) :])

    if not store_rel.parts:
        logger.warning(
            "The store (%s) is the watched workspace itself; concealing it "
            "would drop every file event, so the watcher is left unfiltered",
            store_dir,
        )
        return None
    return store_rel


class _WorkspaceHandler(FileSystemEventHandler):
    """Watchdog handler that filters and debounces file events.

    ``concealed`` holds the workspace-relative location of every server-side
    store the watched tree contains — the feedback records and the per-user bar
    layout today — and is empty when they all lie outside it (nothing to
    conceal). Events at or below any member are dropped, so neither filing
    feedback nor saving a bar arrangement announces itself to every connected
    browser.

    A collection rather than one path because the stores are siblings under one
    agent-data root and grow by addition: each is resolved by
    :func:`resolve_store_rel` and the results are passed here together.

    The check is deliberately anchored at the workspace root and compares
    whole path segments: a ``sessions/<id>/feedback/`` directory elsewhere in
    the tree is ordinary workspace content and still broadcasts. That is also
    why the stores are not spelled into ``_IGNORE_PATTERNS``, which matches any
    segment at any depth. A member with no segments — a store that *is* the
    watched tree — is discarded rather than honoured: every path trivially
    starts with it, so it would black out the file panel entirely.

    On a case-insensitive filesystem those segments are compared case-folded,
    probed once here at construction. ``mkdir(exist_ok=True)`` against
    ``feedback`` succeeds silently when ``Feedback`` already exists, so records
    can genuinely arrive under a spelling that an exact comparison would miss
    and broadcast to every connected browser. Folding cannot produce a false
    positive there, because a distinct sibling ``Feedback/`` cannot coexist
    with the store on such a filesystem.
    """

    def __init__(
        self,
        workspace_dir: Path,
        broadcaster: FileEventBroadcaster,
        *,
        concealed: Sequence[PurePath] = (),
    ) -> None:
        self._workspace_dir = workspace_dir
        self._broadcaster = broadcaster
        self._concealed = tuple(rel for rel in concealed if rel.parts)
        self._fold_case = bool(self._concealed) and filesystem_is_case_insensitive(workspace_dir)
        self._concealed_parts = tuple(
            tuple(part.casefold() for part in rel.parts) if self._fold_case else rel.parts
            for rel in self._concealed
        )
        self._last_event: dict[str, float] = {}
        self._debounce_seconds = 0.1
        self._listings: dict[str, dict[str, tuple[bool, int, int]]] = {}

    def on_any_event(self, event: FileSystemEvent) -> None:
        src_path = Path(event.src_path)

        # Filter ignored paths
        if self._is_ignored(src_path):
            return

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

        # Conceal the server-side stores: drop the event rather than broadcast it.
        if self._is_concealed(relative):
            return

        # A coalesced frame: FSEvents may report a burst of writes inside a
        # directory as one ``modified`` event on the directory itself, with no
        # per-file event behind it. Forwarded as-is it says a directory changed
        # and nothing about what is in it, so the listing on the other end
        # never learns about the file. Say what changed instead.
        #
        # The frame skips the path debounce below: it is the whole report of
        # the change, so a frame dropped on the clock is a change lost rather
        # than a duplicate suppressed. What keeps a repeat frame quiet is the
        # listing diff, which announces only names that differ, and every name
        # it does announce claims its own slot.
        if event.is_directory and simple_type == "modified":
            self._broadcast_directory_diff(src_path, relative)
            return

        # Debounce: skip duplicate events for the same path within 100ms
        if not self._claim_debounce_slot(str(src_path)):
            return

        self._broadcaster.broadcast(
            {
                "type": simple_type,
                "path": str(relative),
                "is_dir": event.is_directory,
            }
        )

    def _is_ignored(self, path: Path) -> bool:
        """Whether *path* is one of the paths the file panel never shows."""
        return any(part in _IGNORE_PATTERNS for part in path.parts) or (
            path.suffix in _IGNORE_EXTENSIONS
        )

    def _is_concealed(self, relative: Path) -> bool:
        """Whether *relative* is at or below one of the server-side stores."""
        if not self._concealed_parts:
            return False
        parts = relative.parts
        if self._fold_case:
            parts = tuple(part.casefold() for part in parts)
        return any(parts[: len(store)] == store for store in self._concealed_parts)

    def _broadcast_directory_diff(self, directory: Path, relative: Path) -> None:
        """Broadcast what one level of *directory* holds that it did not before.

        The rescan stops at one level because that is what the frame names: a
        write deeper in the tree produces its own frame for its own directory,
        and recursing here would turn a single coalesced event into a walk of
        the whole workspace.

        The first frame for a directory has no earlier listing to compare
        against, so its contents are announced as ``created``. A file panel
        converges on what is there either way, and re-announcing a file that
        was already listed costs a redundant frame — where staying silent
        would lose the change the frame was reporting.

        One listing is kept per directory that still exists: a directory that
        has gone away is dropped after its contents are announced as deleted,
        so a workspace that churns cannot grow the map without bound.
        """
        key = str(directory)
        previous = self._listings.get(key)
        current = _one_level_listing(directory)
        if directory.is_dir():
            self._listings[key] = current
        else:
            self._listings.pop(key, None)

        if previous is None:
            for name, stamp in current.items():
                self._broadcast_child(directory, relative, name, "created", stamp[0])
            return

        for name, stamp in current.items():
            if name not in previous:
                self._broadcast_child(directory, relative, name, "created", stamp[0])
            elif previous[name] != stamp:
                self._broadcast_child(directory, relative, name, "modified", stamp[0])
        for name in sorted(previous.keys() - current.keys()):
            self._broadcast_child(directory, relative, name, "deleted", previous[name][0])

    def _broadcast_child(
        self, directory: Path, relative: Path, name: str, simple_type: str, is_dir: bool
    ) -> None:
        """Broadcast one child of a rescanned directory, ignore rules applied.

        The rescan reaches children the per-file path never filtered, so the
        same two predicates run again here — a rescan is not a way past the
        ignore list or past concealment.

        The child shares one debounce slot with its own per-file event: a write
        delivers both that event and a frame for the parent directory, and
        whichever arrives first is the one that announces the change.
        """
        child = directory / name
        if self._is_ignored(child):
            return
        child_relative = relative / name
        if self._is_concealed(child_relative):
            return
        if not self._claim_debounce_slot(str(child)):
            return
        self._broadcaster.broadcast(
            {"type": simple_type, "path": str(child_relative), "is_dir": is_dir}
        )

    def _claim_debounce_slot(self, key: str) -> bool:
        """Whether *key* may be announced now, claiming its slot if so.

        One slot per path, held for the debounce window: the pair of events a
        single write produces — the per-file event and the frame for its parent
        directory — describes one change, and the first of the two to arrive is
        the one that announces it.
        """
        now = time.monotonic()
        if now - self._last_event.get(key, 0) < self._debounce_seconds:
            return False
        self._last_event[key] = now
        return True


class WorkspaceWatcher:
    """Watches a workspace directory for file changes using watchdog.

    ``concealed`` is passed through to the handler: the workspace-relative
    paths of the server-side stores whose events are dropped, empty when they
    all sit outside the watched tree and nothing needs concealing.
    """

    def __init__(
        self,
        workspace_dir: Path,
        broadcaster: FileEventBroadcaster,
        *,
        concealed: Sequence[PurePath] = (),
    ) -> None:
        self._workspace_dir = workspace_dir
        self._broadcaster = broadcaster
        self._concealed = tuple(concealed)
        self._observer: Observer | None = None

    def start(self) -> None:
        """Start watching the workspace directory."""
        if not self._workspace_dir.exists():
            self._workspace_dir.mkdir(parents=True, exist_ok=True)

        handler = _WorkspaceHandler(
            self._workspace_dir, self._broadcaster, concealed=self._concealed
        )
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
