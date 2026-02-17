"""Memory storage for OSPREY MCP tools.

Provides persistent memory management for notes and pins created during
analysis sessions.  Memories are stored in ``osprey-workspace/memory/``
and indexed in ``memories.json``.

Two memory types are supported:
  1. **note** — free-form text observations, procedures, or findings.
  2. **pin** — annotations linked to a specific artifact or data context
     entry, providing a "bookmark with commentary" for important results.

Tags, importance levels, and optional cross-links to artifacts / data
context entries allow structured recall across sessions.

Backward-compatible with the legacy flat-list format previously used by
the ``memory_save`` / ``memory_recall`` tools.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.mcp_server.base_store import BaseStore

logger = logging.getLogger("osprey.mcp_server.memory_store")


# ---------------------------------------------------------------------------
# Backward-compatible listener API
# ---------------------------------------------------------------------------


def register_memory_listener(fn: Callable[[MemoryEntry], None]) -> None:
    """Register a callback invoked after every memory entry save."""
    MemoryStore.register_listener(fn)


def unregister_memory_listener(fn: Callable[[MemoryEntry], None]) -> None:
    """Remove a previously registered listener."""
    MemoryStore.unregister_listener(fn)


# ---------------------------------------------------------------------------
# Memory entry dataclass
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """Metadata for a single memory entry stored on disk.

    Supports two memory types:
      - ``"note"`` — standalone observation or finding.
      - ``"pin"`` — annotation linked to an artifact or data context entry.
    """

    id: int
    memory_type: str  # "note" | "pin"
    content: str  # Note body or pin annotation
    tags: list[str] = field(default_factory=list)  # e.g. ["procedure", "beam"]
    importance: str = "normal"  # "normal" | "important"
    timestamp: str = ""  # ISO 8601
    linked_artifact_id: str | None = None  # Pin -> ArtifactEntry.id
    linked_context_id: int | None = None  # Pin -> DataContextEntry.id
    linked_label: str | None = None  # Display label for link target
    category: str | None = None  # Backward compat with old memories

    def to_dict(self) -> dict[str, Any]:
        """Serialise for inclusion in the index file."""
        return asdict(self)

    def to_tool_response(self) -> dict[str, Any]:
        """Build the compact response returned to Claude from a tool call."""
        resp: dict[str, Any] = {
            "status": "success",
            "memory_id": self.id,
            "memory_type": self.memory_type,
            "content": self.content,
            "tags": self.tags,
            "importance": self.importance,
            "timestamp": self.timestamp,
        }
        if self.linked_artifact_id is not None:
            resp["linked_artifact_id"] = self.linked_artifact_id
        if self.linked_context_id is not None:
            resp["linked_context_id"] = self.linked_context_id
        if self.linked_label is not None:
            resp["linked_label"] = self.linked_label
        return resp


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore(BaseStore[MemoryEntry]):
    """Manages memory entries and the memories.json index."""

    _store_name = "memory"
    _subdir = "memory"
    _index_filename = "memories.json"

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._next_id: int = 1
        super().__init__(workspace_root)

    def _entry_from_dict(self, d: dict) -> MemoryEntry:
        return MemoryEntry(**d)

    def _entry_to_dict(self, entry: MemoryEntry) -> dict:
        return entry.to_dict()

    def _post_load_index(self) -> None:
        if self._entries:
            self._next_id = max(e.id for e in self._entries) + 1
        else:
            self._next_id = 1

    def _parse_index_data(self, data: Any) -> list[MemoryEntry]:
        """Parse index data, auto-migrating the legacy flat-list format."""
        # --- Backward compatibility: detect old flat-list format ---
        if isinstance(data, list):
            logger.info("Migrating %d legacy memory entries to enriched format", len(data))
            entries = []
            for raw in data:
                old_category = raw.get("category")
                entry = MemoryEntry(
                    id=raw.get("id", 0),
                    memory_type="note",
                    content=raw.get("content", ""),
                    tags=[old_category] if old_category else [],
                    importance="normal",
                    timestamp=raw.get("timestamp", ""),
                    category=old_category,
                )
                entries.append(entry)
            # Persist the migrated format immediately
            self._entries = entries
            self._post_load_index()
            self._save_index()
            return entries

        # --- Standard envelope format ---
        return [self._entry_from_dict(d) for d in data.get("entries", [])]

    # ---- public API --------------------------------------------------------

    def save(
        self,
        memory_type: str,
        content: str,
        tags: list[str] | None = None,
        importance: str = "normal",
        linked_artifact_id: str | None = None,
        linked_context_id: int | None = None,
        linked_label: str | None = None,
        category: str | None = None,
    ) -> MemoryEntry:
        """Create a new memory entry and persist it.

        Args:
            memory_type: ``"note"`` or ``"pin"``.
            content: Note body or pin annotation text.
            tags: Classification tags (e.g. ``["procedure", "beam"]``).
            importance: ``"normal"`` or ``"important"``.
            linked_artifact_id: For pins — the linked artifact ID.
            linked_context_id: For pins — the linked data context entry ID.
            linked_label: Human-readable label for the link target.
            category: Legacy category field (backward compat).

        Returns:
            The newly created :class:`MemoryEntry`.
        """
        with self._with_index_lock():
            entry_id = self._next_id
            self._next_id += 1
            now = datetime.now(UTC).isoformat()

            entry = MemoryEntry(
                id=entry_id,
                memory_type=memory_type,
                content=content,
                tags=tags or [],
                importance=importance,
                timestamp=now,
                linked_artifact_id=linked_artifact_id,
                linked_context_id=linked_context_id,
                linked_label=linked_label,
                category=category,
            )
            self._entries.append(entry)
            self._save_index()

        self._notify_listeners(entry)
        return entry

    def list_entries(
        self,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        importance: str | None = None,
        search: str | None = None,
        last_n: int | None = None,
    ) -> list[MemoryEntry]:
        """List memory entries, optionally filtered.

        Args:
            memory_type: Only return entries of this type (``"note"`` / ``"pin"``).
            tags: Only return entries matching *any* of these tags.
            importance: Only return entries with this importance level.
            search: Free-text search across content and tags.
            last_n: Return only the most recent *n* entries.
        """
        self._refresh_if_stale()
        entries = list(self._entries)

        if memory_type:
            entries = [e for e in entries if e.memory_type == memory_type]
        if tags:
            tag_set = set(tags)
            entries = [e for e in entries if tag_set & set(e.tags)]
        if importance:
            entries = [e for e in entries if e.importance == importance]
        if search:
            q = search.lower()
            entries = [
                e for e in entries if q in e.content.lower() or any(q in t.lower() for t in e.tags)
            ]
        if last_n is not None:
            entries = entries[-last_n:]

        return entries

    def get_entry(self, entry_id: int) -> MemoryEntry | None:
        """Look up a single entry by ID."""
        self._refresh_if_stale()
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    def update_entry(self, entry_id: int, **fields: Any) -> MemoryEntry | None:
        """Update mutable fields on an existing memory entry.

        Supported fields: ``content``, ``tags``, ``importance``.

        Returns:
            The updated :class:`MemoryEntry`, or ``None`` if not found.
        """
        with self._with_index_lock():
            allowed = {"content", "tags", "importance"}
            for e in self._entries:
                if e.id == entry_id:
                    for key, value in fields.items():
                        if key in allowed:
                            setattr(e, key, value)
                    self._save_index()
                    return e
        return None

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a memory entry by ID.

        Returns:
            ``True`` if the entry was found and deleted, ``False`` otherwise.
        """
        with self._with_index_lock():
            for i, e in enumerate(self._entries):
                if e.id == entry_id:
                    del self._entries[i]
                    self._save_index()
                    return True
        return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Return the module-level MemoryStore singleton (lazy-initialised)."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


def initialize_memory_store(workspace_root: Path | None = None) -> MemoryStore:
    """(Re-)initialise the MemoryStore singleton with an explicit workspace root."""
    global _memory_store
    _memory_store = MemoryStore(workspace_root=workspace_root)
    return _memory_store


def reset_memory_store() -> None:
    """Reset the singleton — used between tests."""
    global _memory_store
    _memory_store = None
