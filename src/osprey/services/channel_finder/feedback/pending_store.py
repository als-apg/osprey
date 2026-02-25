"""
Pending Review Store for Channel Finder Feedback

Stores agent-captured search results awaiting operator review.
File-backed JSON with cross-process locking via fcntl.flock.
"""

import fcntl
import json
import logging
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STORE_VERSION = 1
MAX_ITEMS = 500


class PendingReviewStore:
    """
    Stores channel finder search results captured by the PostToolUse hook,
    awaiting operator approval or dismissal in the web UI.

    File format::

        {
            "version": 1,
            "items": {
                "<uuid>": {
                    "id": "<uuid>",
                    "query": "show me magnets",
                    "facility": "ALS",
                    "tool_name": "mcp__channel-finder__build_channels",
                    "tool_response": "...",
                    "channel_count": 42,
                    "selections": {"system": "MAG"},
                    "session_id": "abc123",
                    "transcript_path": "/path/to/transcript",
                    "captured_at": "2026-02-23T..."
                }
            }
        }

    Items are capped at MAX_ITEMS (500). When the cap is exceeded, the oldest
    items (by captured_at) are evicted.
    """

    def __init__(self, store_path: str | Path) -> None:
        self._path = Path(store_path)
        self._lock_file = self._path.with_suffix(".lock")
        self._data: dict[str, Any] = {"version": STORE_VERSION, "items": {}}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(self, item: dict) -> str:
        """Capture a new pending review item.

        Args:
            item: Dict with keys like query, facility, tool_name,
                  tool_response, channel_count, selections, session_id,
                  transcript_path, user_prompt.

        Returns:
            UUID string for the new item.
        """
        item_id = str(uuid.uuid4())

        with self._locked():
            self._data["items"][item_id] = {
                "id": item_id,
                "query": item.get("query", ""),
                "facility": item.get("facility", ""),
                "tool_name": item.get("tool_name", ""),
                "tool_response": item.get("tool_response", ""),
                "channel_count": item.get("channel_count", 0),
                "selections": item.get("selections", {}),
                "session_id": item.get("session_id", ""),
                "transcript_path": item.get("transcript_path", ""),
                "agent_task": item.get("agent_task", ""),
                "captured_at": datetime.now(UTC).isoformat(),
            }

            # Evict oldest if over cap
            self._evict_if_needed()
            self._save()

        return item_id

    def list_items(self) -> list[dict]:
        """Return all pending items sorted by captured_at descending (newest first)."""
        self._load()
        items = list(self._data["items"].values())
        items.sort(key=lambda x: x.get("captured_at", ""), reverse=True)
        return items

    def get_item(self, item_id: str) -> dict | None:
        """Return a single item by ID, or None if not found."""
        self._load()
        return self._data["items"].get(item_id)

    def delete(self, item_id: str) -> bool:
        """Delete a single item by ID.

        Returns:
            True if found and deleted, False otherwise.
        """
        with self._locked():
            if item_id in self._data["items"]:
                del self._data["items"][item_id]
                self._save()
                return True
            return False

    def clear(self) -> None:
        """Remove all pending review items."""
        with self._locked():
            self._data = {"version": STORE_VERSION, "items": {}}
            self._save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict oldest items if count exceeds MAX_ITEMS."""
        items = self._data["items"]
        if len(items) <= MAX_ITEMS:
            return

        # Sort by captured_at ascending (oldest first), evict extras
        sorted_ids = sorted(
            items.keys(),
            key=lambda k: items[k].get("captured_at", ""),
        )
        evict_count = len(items) - MAX_ITEMS
        for item_id in sorted_ids[:evict_count]:
            del items[item_id]

    # ------------------------------------------------------------------
    # File I/O with locking
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load data from file (no lock required for reads)."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load pending review store: {e}")
                self._data = {"version": STORE_VERSION, "items": {}}
        else:
            self._data = {"version": STORE_VERSION, "items": {}}

    def _save(self) -> None:
        """Save data to file atomically (must be called within _locked context)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2)
        tmp_path.replace(self._path)

    @contextmanager
    def _locked(self):
        """Acquire exclusive file lock for writes."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(self._lock_file, "w")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
            self._load()  # Always reload under lock
            yield
        finally:
            fd.close()
