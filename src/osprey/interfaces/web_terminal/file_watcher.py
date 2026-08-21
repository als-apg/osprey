"""File system watcher and SSE broadcaster for workspace changes.

FileEventBroadcaster manages per-client asyncio queues for SSE push.
WorkspaceWatcher uses watchdog to detect file changes and broadcast them.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
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


def resolve_store_rel(feedback_dir: Path, workspace_dir: Path) -> PurePath | None:
    """Workspace-relative location of the feedback store, for the watcher.

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
    if feedback_dir.is_relative_to(workspace_dir):
        store_rel = feedback_dir.relative_to(workspace_dir)
    else:
        if not filesystem_is_case_insensitive(workspace_dir):
            return None
        workspace_parts = tuple(part.casefold() for part in workspace_dir.parts)
        store_parts = feedback_dir.parts
        head = tuple(part.casefold() for part in store_parts[: len(workspace_parts)])
        if head != workspace_parts:
            return None
        store_rel = PurePath(*store_parts[len(workspace_parts) :])

    if not store_rel.parts:
        logger.warning(
            "The feedback store (%s) is the watched workspace itself; concealing it "
            "would drop every file event, so the watcher is left unfiltered",
            feedback_dir,
        )
        return None
    return store_rel


class _WorkspaceHandler(FileSystemEventHandler):
    """Watchdog handler that filters and debounces file events.

    ``feedback_rel`` is the workspace-relative location of the feedback
    store, or ``None`` when the store lies outside the watched tree (nothing
    to conceal). Events at or below it are dropped, so filing feedback never
    announces itself to every connected browser.

    The check is deliberately anchored at the workspace root and compares
    whole path segments: a ``sessions/<id>/feedback/`` directory elsewhere in
    the tree is ordinary workspace content and still broadcasts. That is also
    why the store is not spelled into ``_IGNORE_PATTERNS``, which matches any
    segment at any depth.

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
        feedback_rel: PurePath | None = None,
    ) -> None:
        self._workspace_dir = workspace_dir
        self._broadcaster = broadcaster
        self._feedback_rel = feedback_rel
        self._fold_case = feedback_rel is not None and filesystem_is_case_insensitive(workspace_dir)
        store_parts = feedback_rel.parts if feedback_rel is not None else ()
        self._store_parts = (
            tuple(part.casefold() for part in store_parts) if self._fold_case else store_parts
        )
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

        # Debounce: skip duplicate events for the same path within 100ms
        now = time.monotonic()
        key = str(src_path)
        last = self._last_event.get(key, 0)
        if now - last < self._debounce_seconds:
            return
        self._last_event[key] = now

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

        # Conceal the feedback store: drop the event rather than broadcast it.
        if self._feedback_rel is not None:
            head = relative.parts[: len(self._store_parts)]
            if self._fold_case:
                head = tuple(part.casefold() for part in head)
            if head == self._store_parts:
                return

        self._broadcaster.broadcast(
            {
                "type": simple_type,
                "path": str(relative),
                "is_dir": event.is_directory,
            }
        )


class WorkspaceWatcher:
    """Watches a workspace directory for file changes using watchdog.

    ``feedback_rel`` is passed through to the handler: the workspace-relative
    path of the feedback store, whose events are dropped, or ``None`` when
    the store sits outside the watched tree and nothing needs concealing.
    """

    def __init__(
        self,
        workspace_dir: Path,
        broadcaster: FileEventBroadcaster,
        *,
        feedback_rel: PurePath | None = None,
    ) -> None:
        self._workspace_dir = workspace_dir
        self._broadcaster = broadcaster
        self._feedback_rel = feedback_rel
        self._observer: Observer | None = None

    def start(self) -> None:
        """Start watching the workspace directory."""
        if not self._workspace_dir.exists():
            self._workspace_dir.mkdir(parents=True, exist_ok=True)

        handler = _WorkspaceHandler(
            self._workspace_dir, self._broadcaster, feedback_rel=self._feedback_rel
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
