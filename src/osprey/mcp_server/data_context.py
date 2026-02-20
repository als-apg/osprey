"""Data Context management for OSPREY MCP tools.

Provides a centralized system for managing tool output data. Every MCP tool
saves its output to a data file and registers a compact entry in a central
index (data_context.json). This mirrors the CapabilityContext pattern from
the main OSPREY framework, adapted for file-based MCP communication.

The index serves as Claude's "table of contents" for all collected data.
Tools return only compact summaries; full data lives on disk.

Design inspired by:
    - ``osprey.context.base.CapabilityContext.get_summary()``
    - ``osprey.context.base.CapabilityContext.get_access_details()``
    - ``osprey.context.context_manager.ContextManager.get_summaries()``
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.mcp_server.base_store import BaseStore

logger = logging.getLogger("osprey.mcp_server.data_context")


# ---------------------------------------------------------------------------
# Backward-compatible listener API
# ---------------------------------------------------------------------------


def register_context_listener(fn: Callable[[DataContextEntry], None]) -> None:
    """Register a callback invoked after every context entry save."""
    DataContext.register_listener(fn)


def unregister_context_listener(fn: Callable[[DataContextEntry], None]) -> None:
    """Remove a previously registered listener."""
    DataContext.unregister_listener(fn)


@dataclass
class DataContextEntry:
    """One entry in the data context index.

    Mirrors the two-method pattern from ``CapabilityContext``:
      - ``summary`` corresponds to ``get_summary()`` — compact stats for the LLM.
      - ``access_details`` corresponds to ``get_access_details()`` — describes the
        file contents and structure so the LLM knows what it will find.
    """

    id: int
    tool: str
    timestamp: str
    description: str
    summary: dict[str, Any]
    access_details: dict[str, Any]
    data_file: str
    data_type: str
    size_bytes: int
    source_agent: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise for inclusion in the index file."""
        return asdict(self)

    def to_tool_response(self) -> dict[str, Any]:
        """Build the compact response returned to Claude from a tool call.

        This is the *only* thing Claude sees inline after a tool invocation.
        Full data is at ``data_file``.
        """
        return {
            "status": "success",
            "context_entry_id": self.id,
            "description": self.description,
            "summary": self.summary,
            "access_details": self.access_details,
            "data_file": self.data_file,
            "hint": (
                "Use data_context_list to see all collected data. "
                "Read the data file directly for full details."
            ),
        }


class DataContext(BaseStore[DataContextEntry]):
    """Singleton manager for the OSPREY MCP data context.

    Manages ``osprey-workspace/data_context.json`` (the index) and
    ``osprey-workspace/data/`` (the data files).

    Analogous to ``ContextManager`` in the main OSPREY framework, but
    simplified for single-process, file-based MCP operation.
    """

    _store_name = "data context"
    _subdir = ""  # Index lives at workspace root
    _index_filename = "data_context.json"

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._next_id: int = 1
        super().__init__(workspace_root)
        self._data_dir = self._workspace / "data"

    def _entry_from_dict(self, d: dict) -> DataContextEntry:
        return DataContextEntry(**d)

    def _entry_to_dict(self, entry: DataContextEntry) -> dict:
        return entry.to_dict()

    def _post_load_index(self) -> None:
        if self._entries:
            self._next_id = max(e.id for e in self._entries) + 1
        else:
            self._next_id = 1

    def _ensure_dirs(self) -> None:
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _build_index_data(self) -> dict:
        data = super()._build_index_data()
        data["created"] = (
            self._entries[0].timestamp if self._entries else datetime.now(UTC).isoformat()
        )
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        tool: str,
        data: Any,
        description: str,
        summary: dict[str, Any],
        access_details: dict[str, Any],
        data_type: str,
        source_agent: str = "",
    ) -> DataContextEntry:
        """Save tool output to a data file and register it in the index.

        Args:
            tool: Originating MCP tool name (e.g. ``"archiver_read"``).
            data: Full payload to serialise into the data file.
            description: One-liner for the index (human / LLM readable).
            summary: Compact stats dict (what ``get_summary()`` would return).
            access_details: Describes the file contents / structure
                            (what ``get_access_details()`` would return).
            data_type: Category tag — ``"timeseries"``, ``"channel_values"``,
                       ``"search_results"``, ``"code_output"``, etc.

        Returns:
            The newly created :class:`DataContextEntry`.
        """
        from osprey.mcp_server.type_registry import valid_data_type_keys

        if data_type not in valid_data_type_keys():
            logger.warning("Unregistered data_type %r — add to type_registry.py", data_type)

        with self._with_index_lock():
            entry_id = self._next_id
            self._next_id += 1
            now = datetime.now(UTC).isoformat()

            # ---- Write data file ----
            filename = f"{entry_id:03d}_{tool}.json"
            filepath = self._data_dir / filename

            metadata = {
                "context_entry_id": entry_id,
                "tool": tool,
                "timestamp": now,
                "description": description,
            }
            if source_agent:
                metadata["source_agent"] = source_agent
            file_payload = {
                "_osprey_metadata": metadata,
                "data": data,
            }
            with open(filepath, "w") as f:
                json.dump(file_payload, f, indent=2, default=str)

            size_bytes = filepath.stat().st_size

            # ---- Create & register index entry ----
            entry = DataContextEntry(
                id=entry_id,
                tool=tool,
                timestamp=now,
                description=description,
                summary=summary,
                access_details=access_details,
                data_file=str(filepath),
                data_type=data_type,
                size_bytes=size_bytes,
                source_agent=source_agent,
            )
            self._entries.append(entry)
            self._save_index()

        self._notify_listeners(entry)
        return entry

    def list_entries(
        self,
        tool_filter: str | None = None,
        data_type_filter: str | None = None,
        last_n: int | None = None,
        search: str | None = None,
        source_agent_filter: str | None = None,
    ) -> list[DataContextEntry]:
        """List context entries, optionally filtered.

        Args:
            tool_filter: Only return entries from this tool.
            data_type_filter: Only return entries with this data type.
            last_n: Return only the most recent *n* entries.
            search: Free-text search across description and tool name.
            source_agent_filter: Only return entries from this agent.
        """
        self._refresh_if_stale()
        entries = list(self._entries)

        if tool_filter:
            entries = [e for e in entries if e.tool == tool_filter]
        if data_type_filter:
            entries = [e for e in entries if e.data_type == data_type_filter]
        if source_agent_filter:
            entries = [e for e in entries if e.source_agent == source_agent_filter]
        if search:
            q = search.lower()
            entries = [e for e in entries if q in e.description.lower() or q in e.tool.lower()]
        if last_n is not None:
            entries = entries[-last_n:]

        return entries

    def get_entry(self, entry_id: int) -> DataContextEntry | None:
        """Look up a single entry by ID."""
        self._refresh_if_stale()
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a data context entry by ID, removing both the index entry and data file.

        Returns:
            ``True`` if the entry was found and deleted, ``False`` otherwise.
        """
        with self._with_index_lock():
            for i, e in enumerate(self._entries):
                if e.id == entry_id:
                    # Delete data file
                    filepath = Path(e.data_file)
                    if filepath.exists():
                        filepath.unlink()
                    del self._entries[i]
                    self._save_index()
                    return True
        return False

    def get_file_path(self, entry_id: int) -> Path | None:
        """Return the data file path for an entry, or None if not found/missing."""
        entry = self.get_entry(entry_id)
        if entry is None:
            return None
        path = Path(entry.data_file)
        return path if path.exists() else None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_data_context: DataContext | None = None


def get_data_context() -> DataContext:
    """Return the module-level DataContext singleton (lazy-initialised)."""
    global _data_context
    if _data_context is None:
        _data_context = DataContext()
    return _data_context


def initialize_data_context(workspace_root: Path | None = None) -> DataContext:
    """(Re-)initialise the DataContext singleton with an explicit workspace root."""
    global _data_context
    _data_context = DataContext(workspace_root=workspace_root)
    return _data_context


def reset_data_context() -> None:
    """Reset the singleton — used between tests."""
    global _data_context
    _data_context = None
