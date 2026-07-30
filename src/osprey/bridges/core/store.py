"""Lock-guarded, crash-safe JSON file store shared by the bridges' state stores.

Both :class:`~osprey.bridges.core.dedup.DedupStore` and
:class:`~osprey.bridges.core.history.HistoryStore` persist a small dict to a JSON file
on the bridge-owned volume. The durability pattern is identical and subtle
enough to keep in ONE place:

  * a bridge dispatches callbacks on a thread pool (e.g. google-cloud-pubsub),
    so a single shared store is mutated by many threads. Every read/modify/write
    (and the JSON dump, which iterates the dict) happens under one lock.
  * Writes are atomic: dump to a tmp file created in the SAME directory as the
    target (``os.replace`` is only atomic within one filesystem), then rename.
    A failed dump unlinks the tmp file and re-raises.
  * A missing or corrupt file loads as empty — the store must never crash the
    bridge at startup.

Subclasses override :meth:`_coerce` to validate/normalize the loaded mapping.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any


class JsonFileStore:
    def __init__(self, path: str):
        self.path = path
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._load()

    def _coerce(self, loaded: dict) -> dict:
        """Hook: filter/normalize the freshly loaded mapping. Default: as-is."""
        return loaded

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self._data = self._coerce(loaded)
        except (FileNotFoundError, ValueError):
            # ValueError covers BOTH JSONDecodeError and UnicodeDecodeError —
            # any unparseable file loads as empty. Deliberately NOT OSError:
            # an unreadable volume must fail loudly, because silently starting
            # with an empty dedup store would re-dispatch answered questions.
            self._data = {}

    def _flush_locked(self) -> None:
        """Persist the store. Caller MUST hold ``self._lock`` — the lock is
        what keeps concurrent mutation from racing the JSON dump."""
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        snapshot = dict(self._data)
        fd, tmp = tempfile.mkstemp(
            dir=directory, prefix=f".{os.path.basename(self.path)}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
