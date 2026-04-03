"""Artifact storage for OSPREY MCP tools.

Manages interactive artifacts (plots, tables, HTML, markdown) produced by
Claude during analysis sessions.  Artifacts are stored in
``_agent_data/artifacts/`` and served by the Artifact Server gallery.

Two entry points create artifacts:
  1. ``save_artifact()`` — injected into ``execute`` namespace
  2. ``artifact_save`` — standalone MCP tool for files / inline content

This module provides the low-level storage layer used by both.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.stores.base_store import BaseStore

logger = logging.getLogger("osprey.stores.artifact_store")


def register_artifact_listener(fn: Callable[[ArtifactEntry], None]) -> None:
    """Register a callback invoked after every artifact save."""
    ArtifactStore.register_listener(fn)


def unregister_artifact_listener(fn: Callable[[ArtifactEntry], None]) -> None:
    """Remove a previously registered listener."""
    ArtifactStore.unregister_listener(fn)


@dataclass
class ArtifactEntry:
    """Metadata for a single artifact stored on disk."""

    id: str
    artifact_type: str
    title: str
    description: str
    filename: str
    mime_type: str
    size_bytes: int
    timestamp: str
    tool_source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    pinned: bool = False
    # --- Unified artifact fields (replaces DataContext) ---
    category: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    access_details: dict[str, Any] = field(default_factory=dict)
    data_file: str = ""
    source_agent: str = ""
    session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_tool_response(self, gallery_url: str | None = None) -> dict[str, Any]:
        """Compact response returned to Claude after artifact creation.

        When ``summary`` / ``access_details`` are populated (data artifacts),
        the response includes them so Claude sees the same compact info that
        DataContext.to_tool_response() used to provide.
        """
        resp: dict[str, Any] = {
            "status": "success",
            "artifact_id": self.id,
            "title": self.title,
            "artifact_type": self.artifact_type,
            "size_bytes": self.size_bytes,
            "pinned": self.pinned,
        }
        if self.category:
            resp["category"] = self.category
        if self.summary:
            resp["summary"] = self.summary
        if self.access_details:
            resp["access_details"] = self.access_details
        if self.data_file:
            resp["data_file"] = self.data_file
        if gallery_url:
            resp["gallery_url"] = gallery_url
        return resp


def _serialize_object(obj: Any, title: str) -> tuple[bytes, str, str, str]:
    """Detect the type of *obj* and serialise it to bytes.

    Returns:
        (content_bytes, artifact_type, filename_ext, mime_type)
    """
    slug = _slugify(title)

    # --- Plotly Figure ---
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        if isinstance(obj, go.Figure):
            html = obj.to_html(include_plotlyjs="cdn", full_html=True)
            return html.encode(), "plot_html", f"{slug}.html", "text/html"
    except ImportError:
        pass

    # --- Matplotlib Figure ---
    try:
        import matplotlib.figure  # type: ignore[import-untyped]

        if isinstance(obj, matplotlib.figure.Figure):
            import io

            buf = io.BytesIO()
            obj.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            return buf.read(), "plot_png", f"{slug}.png", "image/png"
    except ImportError:
        pass

    # --- Pandas DataFrame ---
    try:
        import pandas as pd  # type: ignore[import-untyped]

        if isinstance(obj, pd.DataFrame):
            html = obj.to_html(classes="artifact-table", border=0)
            return html.encode(), "table_html", f"{slug}.html", "text/html"
    except ImportError:
        pass

    # --- str ---
    if isinstance(obj, str):
        if obj.lstrip().startswith(("<", "<!")) and "</" in obj:
            return obj.encode(), "html", f"{slug}.html", "text/html"
        return obj.encode(), "markdown", f"{slug}.md", "text/markdown"

    # --- dict / list ---
    if isinstance(obj, (dict, list)):
        return (
            json.dumps(obj, indent=2, default=str).encode(),
            "json",
            f"{slug}.json",
            "application/json",
        )

    # --- bytes ---
    if isinstance(obj, bytes):
        return obj, "binary", f"{slug}.bin", "application/octet-stream"

    # Fallback: repr as text
    return repr(obj).encode(), "text", f"{slug}.txt", "text/plain"


def _slugify(title: str) -> str:
    """Convert title to a filesystem-safe slug."""
    slug = title.lower().strip()
    slug = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in slug)
    slug = slug.replace(" ", "_")
    return slug[:60] or "artifact"


def _guess_mime(path: Path) -> str:
    """Guess MIME type from file extension."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _artifact_type_from_mime(mime: str) -> str:
    """Map MIME type to an artifact_type string."""
    if mime.startswith("image/"):
        return "image"
    if mime == "text/html":
        return "html"
    if mime in ("text/markdown", "text/x-markdown"):
        return "markdown"
    if mime == "application/json":
        return "json"
    if mime.startswith("text/"):
        return "text"
    return "file"


class ArtifactStore(BaseStore[ArtifactEntry]):
    """Manages artifact files and the artifacts.json index."""

    _store_name = "artifacts"
    _subdir = "artifacts"
    _index_filename = "artifacts.json"

    def _entry_from_dict(self, d: dict) -> ArtifactEntry:
        # Gracefully handle old index files missing fields
        d.pop("highlighted", None)  # Removed concept; strip from old indexes
        d.setdefault("pinned", False)
        d.setdefault("category", "")
        d.setdefault("summary", {})
        d.setdefault("access_details", {})
        d.setdefault("data_file", "")
        d.setdefault("source_agent", "")
        d.setdefault("session_id", "")
        return ArtifactEntry(**d)

    def _entry_to_dict(self, entry: ArtifactEntry) -> dict:
        return entry.to_dict()

    def _make_id(self) -> str:
        return uuid.uuid4().hex[:12]

    @property
    def artifact_dir(self) -> Path:
        return self._store_dir

    # ---- public API --------------------------------------------------------

    def save_file(
        self,
        file_content: bytes,
        filename: str,
        artifact_type: str,
        title: str,
        description: str = "",
        mime_type: str = "application/octet-stream",
        tool_source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactEntry:
        """Low-level save: write raw bytes to disk and register in the index."""
        with self._with_index_lock():
            art_id = self._make_id()
            # Prefix filename with id to avoid collisions
            safe_filename = f"{art_id}_{filename}"
            filepath = self._store_dir / safe_filename
            filepath.write_bytes(file_content)

            entry = ArtifactEntry(
                id=art_id,
                artifact_type=artifact_type,
                title=title,
                description=description,
                filename=safe_filename,
                mime_type=mime_type,
                size_bytes=len(file_content),
                timestamp=datetime.now(UTC).isoformat(),
                tool_source=tool_source,
                metadata=metadata or {},
                session_id=os.environ.get("OSPREY_SESSION_ID", ""),
            )
            self._entries.append(entry)
            self._save_index()

        self._notify_listeners(entry)

        # Auto-launch artifact server on first save
        try:
            from osprey.infrastructure.server_launcher import ensure_artifact_server

            ensure_artifact_server()
        except Exception as exc:
            logger.warning("Artifact server auto-launch failed: %s", exc, exc_info=True)

        return entry

    def save_object(
        self,
        obj: Any,
        title: str,
        description: str = "",
        artifact_type: str | None = None,
        tool_source: str = "execute",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactEntry:
        """Smart-serialize a Python object and save it.

        Detects Plotly figures, matplotlib figures, DataFrames, strings, dicts.
        """
        content, detected_type, filename, mime = _serialize_object(obj, title)

        return self.save_file(
            file_content=content,
            filename=filename,
            artifact_type=artifact_type or detected_type,
            title=title,
            description=description,
            mime_type=mime,
            tool_source=tool_source,
            metadata=metadata,
        )

    def save_from_path(
        self,
        source_path: str | Path,
        title: str,
        description: str = "",
        artifact_type: str | None = None,
        tool_source: str = "artifact_save",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactEntry:
        """Copy an existing file into the artifact store."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        mime = _guess_mime(source)
        a_type = artifact_type or _artifact_type_from_mime(mime)

        content = source.read_bytes()
        return self.save_file(
            file_content=content,
            filename=source.name,
            artifact_type=a_type,
            title=title,
            description=description,
            mime_type=mime,
            tool_source=tool_source,
            metadata=metadata,
        )

    def save_data(
        self,
        tool: str,
        data: Any,
        title: str,
        description: str = "",
        summary: dict[str, Any] | None = None,
        access_details: dict[str, Any] | None = None,
        category: str = "",
        source_agent: str = "",
        artifact_type: str = "json",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactEntry:
        """Save structured data as an artifact.  Replaces ``DataContext.save()``.

        Writes raw JSON (no envelope) to ``artifacts/{id}_{tool}.json`` and
        registers the entry in the unified artifact index.
        """
        from osprey.stores.type_registry import valid_category_keys

        if category and category not in valid_category_keys():
            logger.warning("Unregistered category %r — add to type_registry.py", category)

        content = json.dumps(data, indent=2, default=str).encode()
        filename = f"{tool}.json"

        with self._with_index_lock():
            art_id = self._make_id()
            safe_filename = f"{art_id}_{filename}"
            filepath = self._store_dir / safe_filename
            filepath.write_bytes(content)

            entry = ArtifactEntry(
                id=art_id,
                artifact_type=artifact_type,
                title=title,
                description=description,
                filename=safe_filename,
                mime_type="application/json",
                size_bytes=len(content),
                timestamp=datetime.now(UTC).isoformat(),
                tool_source=tool,
                metadata=metadata or {},
                category=category,
                summary=summary or {},
                access_details=access_details or {},
                data_file=safe_filename,
                source_agent=source_agent,
                session_id=os.environ.get("OSPREY_SESSION_ID", ""),
            )
            self._entries.append(entry)
            self._save_index()

        self._notify_listeners(entry)

        try:
            from osprey.infrastructure.server_launcher import ensure_artifact_server

            ensure_artifact_server()
        except Exception as exc:
            logger.warning("Artifact server auto-launch failed: %s", exc, exc_info=True)

        return entry

    def list_entries(
        self,
        type_filter: str | None = None,
        search: str | None = None,
        pinned: bool | None = None,
        category_filter: str | None = None,
        tool_filter: str | None = None,
        source_agent_filter: str | None = None,
        session_filter: str | None = None,
        last_n: int | None = None,
    ) -> list[ArtifactEntry]:
        """Return artifact entries, optionally filtered."""
        self._refresh_if_stale()
        entries = list(self._entries)
        if type_filter:
            entries = [e for e in entries if e.artifact_type == type_filter]
        if category_filter:
            entries = [e for e in entries if e.category == category_filter]
        if tool_filter:
            entries = [e for e in entries if e.tool_source == tool_filter]
        if source_agent_filter:
            entries = [e for e in entries if e.source_agent == source_agent_filter]
        if session_filter:
            entries = [e for e in entries if e.session_id == session_filter or not e.session_id]
        if pinned is not None:
            entries = [e for e in entries if e.pinned == pinned]
        if search:
            q = search.lower()
            entries = [e for e in entries if q in e.title.lower() or q in e.description.lower()]
        if last_n is not None:
            entries = entries[-last_n:]
        return entries

    def set_pinned(self, artifact_id: str, pinned: bool = True) -> ArtifactEntry | None:
        """Toggle the pinned flag on an artifact. Returns the updated entry or None."""
        with self._with_index_lock():
            for e in self._entries:
                if e.id == artifact_id:
                    e.pinned = pinned
                    self._save_index()
                    return e
        return None

    def get_entry(self, artifact_id: str) -> ArtifactEntry | None:
        self._refresh_if_stale()
        for e in self._entries:
            if e.id == artifact_id:
                return e
        return None

    def delete_entry(self, artifact_id: str) -> bool:
        """Delete an artifact by ID, removing both the index entry and physical file.

        Returns:
            ``True`` if the entry was found and deleted, ``False`` otherwise.
        """
        with self._with_index_lock():
            for i, e in enumerate(self._entries):
                if e.id == artifact_id:
                    # Delete physical file
                    filepath = self._store_dir / e.filename
                    if filepath.exists():
                        filepath.unlink()
                    del self._entries[i]
                    self._save_index()
                    return True
        return False

    def get_file_path(self, artifact_id: str) -> Path | None:
        entry = self.get_entry(artifact_id)
        if entry is None:
            return None
        return self._store_dir / entry.filename


_artifact_store: ArtifactStore | None = None


def get_artifact_store() -> ArtifactStore:
    """Return the module-level ArtifactStore singleton (lazy-initialised)."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store


def initialize_artifact_store(workspace_root: Path | None = None) -> ArtifactStore:
    """(Re-)initialise the ArtifactStore singleton with an explicit workspace root."""
    global _artifact_store
    _artifact_store = ArtifactStore(workspace_root=workspace_root)
    return _artifact_store


def reset_artifact_store() -> None:
    """Reset the singleton — used between tests."""
    global _artifact_store
    _artifact_store = None
