"""Base class for file-backed indexed stores.

Provides the shared infrastructure used by :class:`ArtifactStore`:

- File-backed JSON index with mtime-based staleness detection
- Directory management (``_ensure_dirs``)
- Listener notification pattern
- Module-level singleton lifecycle helpers
"""

from __future__ import annotations

import fcntl
import json
import logging
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

logger = logging.getLogger("osprey.stores.base_store")

INDEX_VERSION = 1

T = TypeVar("T")


class BaseStore(Generic[T]):
    """Abstract base for a file-backed, JSON-indexed store.

    Subclasses must set:
      - ``_store_name``: Human-readable name (for logging).
      - ``_subdir``: Subdirectory within the workspace root.
      - ``_index_filename``: Name of the JSON index file.

    And implement:
      - ``_entry_from_dict(d: dict) -> T``: Construct an entry from a dict.
      - ``_entry_to_dict(entry: T) -> dict``: Serialize an entry to a dict.

    Optionally override:
      - ``_parse_index_data(data)`` for custom migration logic (e.g. legacy format migration).
      - ``_build_index_data()`` for custom index envelope fields.
      - ``_post_load_index()`` for post-load hooks (e.g. setting ``_next_id``).
    """

    _store_name: str = "store"
    _subdir: str = ""
    _index_filename: str = "index.json"

    # Each subclass gets its own listener list via __init_subclass__
    _listeners: list[Callable[[Any], None]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._listeners = []

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._workspace = workspace_root or (Path.cwd() / "osprey-workspace")
        self._store_dir = self._workspace / self._subdir if self._subdir else self._workspace
        self._index_file = self._store_dir / self._index_filename
        self._entries: list[T] = []
        self._index_mtime: float = 0.0
        self._load_index()

    def _entry_from_dict(self, d: dict) -> T:
        raise NotImplementedError

    def _entry_to_dict(self, entry: T) -> dict:
        raise NotImplementedError

    def _ensure_dirs(self) -> None:
        """Create the store directory if it doesn't exist."""
        self._store_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _lock_file(self) -> Path:
        """Path to the advisory lock file adjacent to the index."""
        return self._index_file.with_suffix(".lock")

    @contextmanager
    def _with_index_lock(self):
        """Acquire an exclusive lock, reload the index, and yield.

        All mutating operations (save, delete, update) should wrap their
        critical section under this lock to prevent cross-process TOCTOU
        races on the shared index file.
        """
        self._ensure_dirs()
        fd = open(self._lock_file, "w")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
            self._load_index()  # Always reload under lock
            yield
        finally:
            fd.close()  # Releases lock

    def _load_index(self) -> None:
        """Load the index from disk, if present."""
        if self._index_file.exists():
            try:
                self._index_mtime = self._index_file.stat().st_mtime
                with open(self._index_file) as f:
                    data = json.load(f)
                self._entries = self._parse_index_data(data)
                self._post_load_index()
            except Exception:
                logger.warning("Could not load %s index; starting fresh", self._store_name)
                self._entries = []
                self._post_load_index()

    def _parse_index_data(self, data: Any) -> list[T]:
        """Parse loaded JSON into entries.

        Override for custom migration logic (e.g. legacy format migration).
        Default: read from ``data["entries"]``.
        """
        return [self._entry_from_dict(d) for d in data.get("entries", [])]

    def _post_load_index(self) -> None:
        """Hook called after index loading. Override to set ``_next_id``, etc."""

    def _refresh_if_stale(self) -> None:
        """Reload the index from disk if another process has updated it."""
        try:
            if self._index_file.exists():
                mtime = self._index_file.stat().st_mtime
                if mtime > self._index_mtime:
                    self._load_index()
        except OSError:
            pass

    def _save_index(self) -> None:
        """Persist the index to disk."""
        self._ensure_dirs()
        index_data = self._build_index_data()
        with open(self._index_file, "w") as f:
            json.dump(index_data, f, indent=2, default=str)
        self._index_mtime = self._index_file.stat().st_mtime

    def _build_index_data(self) -> dict:
        """Build the index dict for serialization.

        Override to add custom envelope fields.
        """
        return {
            "version": INDEX_VERSION,
            "updated": datetime.now(UTC).isoformat(),
            "entry_count": len(self._entries),
            "entries": [self._entry_to_dict(e) for e in self._entries],
        }

    @classmethod
    def register_listener(cls, fn: Callable[[Any], None]) -> None:
        """Register a callback invoked after every entry save."""
        cls._listeners.append(fn)

    @classmethod
    def unregister_listener(cls, fn: Callable[[Any], None]) -> None:
        """Remove a previously registered listener."""
        cls._listeners.remove(fn)

    def _notify_listeners(self, entry: T) -> None:
        """Notify all registered listeners of a new entry."""
        for fn in self.__class__._listeners:
            try:
                fn(entry)
            except Exception:
                logger.debug("%s listener failed", self._store_name, exc_info=True)

    def update_entry_metadata(self, entry_id: str, **kwargs: Any) -> T | None:
        """Set metadata attributes on an entry and persist the index.

        Usage::

            store.update_entry_metadata(entry.id, category="document", source_agent="data-visualizer")
        """
        with self._with_index_lock():
            for e in self._entries:
                if e.id == entry_id:
                    for key, value in kwargs.items():
                        if not hasattr(e, key):
                            raise AttributeError(f"{type(e).__name__} has no attribute {key!r}")
                        setattr(e, key, value)
                    self._save_index()
                    return e
        return None
