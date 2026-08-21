"""Artifact storage for OSPREY MCP tools.

Manages interactive artifacts (plots, tables, HTML, markdown) produced by
Claude during analysis sessions.  Artifacts are stored under the agent-data
root (``agent_data.base_dir``) in ``artifacts/``, and served by the Artifact
Server gallery.

Two entry points create artifacts:
  1. ``save_artifact()`` — injected into ``execute`` namespace
  2. ``artifact_save`` — standalone MCP tool for files / inline content

This module provides the low-level storage layer used by both.
"""

from __future__ import annotations

import contextvars
import json
import logging
import mimetypes
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.stores.base_store import BaseStore, _sanitize_for_json

logger = logging.getLogger("osprey.stores.artifact_store")

#: Who is performing the current store mutation. The store itself cannot tell
#: an agent tool call from a gallery click or a retention sweep — the frames a
#: listener emits from these events are *agent* activity, so non-agent callers
#: declare themselves via :func:`artifact_mutation_actor` and listeners read
#: :func:`current_artifact_mutation_actor` at event time. Defaults to "agent"
#: because every MCP-tool path mutates the store on the agent's behalf.
_MUTATION_ACTOR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "artifact_mutation_actor", default="agent"
)


@contextmanager
def artifact_mutation_actor(actor: str) -> Iterator[None]:
    """Attribute store mutations in this scope to *actor* ("human", "system").

    Listener callbacks fire synchronously inside the mutation call, so the
    scope only needs to cover the ``save_*``/``delete_*`` call itself.
    """
    token = _MUTATION_ACTOR.set(actor)
    try:
        yield
    finally:
        _MUTATION_ACTOR.reset(token)


def current_artifact_mutation_actor() -> str:
    """Return the actor of the mutation currently firing listeners."""
    return _MUTATION_ACTOR.get()


def register_artifact_listener(fn: Callable[[ArtifactEntry], None]) -> None:
    """Register a callback invoked after every artifact save."""
    ArtifactStore.register_listener(fn)


def unregister_artifact_listener(fn: Callable[[ArtifactEntry], None]) -> None:
    """Remove a previously registered listener."""
    ArtifactStore.unregister_listener(fn)


def register_artifact_delete_listener(fn: Callable[[ArtifactEntry], None]) -> None:
    """Register a callback invoked after every artifact delete."""
    ArtifactStore.register_delete_listener(fn)


def unregister_artifact_delete_listener(fn: Callable[[ArtifactEntry], None]) -> None:
    """Remove a previously registered delete listener."""
    ArtifactStore.unregister_delete_listener(fn)


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
    run_id: str = ""
    """Dispatch run that produced this artifact (``OSPREY_DISPATCH_RUN_ID``),
    empty for artifacts made outside a dispatch run.

    Deliberately separate from ``session_id``: a session id is not a run id —
    ``OSPREY_SESSION_ID`` tags an interactive session's artifacts, while a
    dispatch run needs its own scope so the worker can report and serve
    exactly this run's artifacts. The store itself always lives on the shared
    data root; both fields are index-level tags."""
    origin: str = ""
    """Provenance of the artifact within its run. Empty (the default) for the
    agent's own output. ``"input"`` marks a caller-supplied file ingested into
    the run *before* the agent starts: such entries are excluded from the run's
    produced-artifact listing yet remain reachable through the created-by byte
    route, so an input image can be served back without appearing as something
    the agent generated."""

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_for_json(asdict(self))

    def to_tool_response(self, gallery_url: str | None = None) -> dict[str, Any]:
        """Compact response returned to Claude after artifact creation.

        When ``summary`` / ``access_details`` are populated (data artifacts),
        the response includes them so Claude sees the shape of the data without
        a second read.
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


def serialize_object(obj: Any, title: str) -> tuple[bytes, str, str, str]:
    """Detect the type of *obj* and serialise it to bytes.

    The one serializer behind every "save this object" path: the in-process
    :meth:`ArtifactStore.save_object` and the ``save_artifact()`` injected into
    execution subprocesses (:mod:`osprey.stores.artifact_manifest`).

    Returns:
        (content_bytes, artifact_type, filename, mime_type)
    """
    slug = _slugify(title)

    # --- Plotly Figure ---
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        if isinstance(obj, go.Figure):
            html = obj.to_html(include_plotlyjs=False, full_html=True)
            return html.encode(), "plot_html", f"{slug}.html", "text/html"
    except ImportError:
        pass

    # --- Bokeh Model (layout, figure, widget, ...) ---
    try:
        import bokeh.model  # type: ignore[import-untyped]

        if isinstance(obj, bokeh.model.Model):
            from bokeh.embed import file_html
            from bokeh.resources import INLINE

            html = file_html(obj, resources=INLINE, title=title)
            return html.encode(), "dashboard_html", f"{slug}.html", "text/html"
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

    # --- numpy ndarray ---
    # Last of the type branches on purpose: an ndarray matches none of the
    # checks above, so every already-supported type keeps the exact sniffing
    # order it had. Without this an array reaches the repr() fallback below and
    # is stored as a lossy text summary instead of loadable data.
    #
    # Object-dtype arrays are deliberately NOT caught here: ``np.save`` cannot
    # write them without pickle, and the store never enables pickle, so they
    # keep falling through to the repr() fallback.
    try:
        import numpy as np

        if isinstance(obj, np.ndarray) and not obj.dtype.hasobject:
            import io

            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return buf.getvalue(), "file", f"{slug}.npy", "application/octet-stream"
    except ImportError:
        pass

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
        d.setdefault("run_id", "")
        d.setdefault("origin", "")
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
        category: str = "",
        run_id: str | None = None,
        origin: str = "",
    ) -> ArtifactEntry:
        """Low-level save: write raw bytes to disk and register in the index.

        ``run_id`` defaults to the ``OSPREY_DISPATCH_RUN_ID`` env var (how the
        dispatched agent's own tools tag their output). A caller that writes on
        behalf of a run whose id is not in its environment — e.g. the worker
        ingesting input files before the agent starts — passes it explicitly.
        ``origin`` tags the entry's provenance within the run (``"input"`` for
        an ingested caller file; empty for produced output)."""
        if category:
            from osprey.stores.type_registry import valid_category_keys

            if category not in valid_category_keys():
                logger.warning("Unregistered category %r — add to type_registry.py", category)

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
                category=category,
                session_id=os.environ.get("OSPREY_SESSION_ID", ""),
                run_id=(
                    run_id if run_id is not None else os.environ.get("OSPREY_DISPATCH_RUN_ID", "")
                ),
                origin=origin,
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
        category: str = "",
    ) -> ArtifactEntry:
        """Smart-serialize a Python object and save it.

        Detects Plotly figures, matplotlib figures, DataFrames, strings, dicts
        and numpy arrays (saved as ``.npy``).
        """
        content, detected_type, filename, mime = serialize_object(obj, title)

        return self.save_file(
            file_content=content,
            filename=filename,
            artifact_type=artifact_type or detected_type,
            title=title,
            description=description,
            mime_type=mime,
            tool_source=tool_source,
            metadata=metadata,
            category=category,
        )

    def save_from_path(
        self,
        source_path: str | Path,
        title: str,
        description: str = "",
        artifact_type: str | None = None,
        tool_source: str = "artifact_save",
        metadata: dict[str, Any] | None = None,
        category: str = "",
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
            category=category,
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

            # data_file is the agent-facing pointer: a path relative to the
            # repo root, which is where agent code's working directory is, so
            # the agent can pass it directly to ``open()``.
            # Falls back to a bare filename if the workspace layout doesn't
            # support a clean relative path (defensive — should not happen).
            try:
                agent_path = str(filepath.relative_to(self.repo_root))
            except ValueError:
                agent_path = safe_filename

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
                data_file=agent_path,
                source_agent=source_agent,
                session_id=os.environ.get("OSPREY_SESSION_ID", ""),
                run_id=os.environ.get("OSPREY_DISPATCH_RUN_ID", ""),
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

    def save_channel_reading(
        self,
        png_bytes: bytes,
        npy_bytes: bytes,
        title: str,
        summary: dict[str, Any] | None = None,
        access_details: dict[str, Any] | None = None,
        category: str = "channel_values",
        metadata: dict[str, Any] | None = None,
        description: str = "",
        tool_source: str = "channel_read",
        source_agent: str = "",
    ) -> ArtifactEntry:
        """Save a multi-dimensional channel reading as a PNG plus its raw array.

        The two-file shape, and the only reason this method exists next to
        :meth:`save_data`: an image-shaped reading is stored twice — a rendered
        PNG the gallery displays directly (the primary file) and the raw
        ``.npy`` payload beside it (``data_file``), which is what analysis code
        actually loads. One-dimensional readings do not need this; they go
        through :meth:`save_data` as a JSON series.

        Both blobs arrive ready: the caller renders the PNG *before* calling, so
        nothing inside the locked section can fail halfway through a render.
        Inside the lock the write order is ``.npy`` → PNG → index, and any
        failure unlinks whatever was already written before re-raising — a
        partial save must never strand a file no delete path can collect.

        ``size_bytes`` counts both files, because the entry owns both; the
        ``.npy`` half is normally the larger one by far, and a size that named
        only the thumbnail-sized PNG would misreport the entry's footprint to
        anything accounting for disk.

        Args:
            png_bytes: Rendered image of the reading — the primary file.
            npy_bytes: Serialized ``numpy`` array — the companion payload.
            title: Human-facing artifact title.
            summary: Compact shape/statistics description for the agent.
            access_details: How to load the payload back (agent-facing).
            category: Registered artifact category; defaults to the reserved
                ``channel_values`` slot.
            metadata: Free-form index metadata. ``metadata["channel"]`` is
                REQUIRED — the channel address is the only queryable key that
                selects an address's readings, so retention pruning needs it.
            description: Optional longer description.
            tool_source: Tool that produced the reading.
            source_agent: Subagent that produced the reading, when applicable.

        Returns:
            The registered :class:`ArtifactEntry`.

        Raises:
            ValueError: If ``metadata["channel"]`` is missing or empty.
        """
        from osprey.stores.type_registry import valid_category_keys

        if category and category not in valid_category_keys():
            logger.warning("Unregistered category %r — add to type_registry.py", category)

        metadata = dict(metadata or {})
        channel = metadata.get("channel")
        if not channel:
            raise ValueError(
                "save_channel_reading requires metadata['channel']: the channel "
                "address is the queryable key retention pruning selects on, and "
                "ArtifactEntry has no other field that records it."
            )

        slug = _slugify(str(channel))

        with self._with_index_lock():
            art_id = self._make_id()
            png_path = self._store_dir / f"{art_id}_{slug}.png"
            npy_path = self._store_dir / f"{art_id}_{slug}.npy"

            written: list[Path] = []
            appended = False
            try:
                npy_path.write_bytes(npy_bytes)
                written.append(npy_path)
                png_path.write_bytes(png_bytes)
                written.append(png_path)

                # data_file is the agent-facing pointer: a path relative to the
                # repo root, which is where agent code's working directory is,
                # so the agent can pass it straight to ``numpy.load()``. Same
                # convention (and same defensive fallback) as ``save_data``.
                try:
                    agent_path = str(npy_path.relative_to(self.repo_root))
                except ValueError:
                    agent_path = npy_path.name

                entry = ArtifactEntry(
                    id=art_id,
                    artifact_type="image",
                    title=title,
                    description=description,
                    filename=png_path.name,
                    mime_type="image/png",
                    size_bytes=len(png_bytes) + len(npy_bytes),
                    timestamp=datetime.now(UTC).isoformat(),
                    tool_source=tool_source,
                    metadata=metadata,
                    category=category,
                    summary=summary or {},
                    access_details=access_details or {},
                    data_file=agent_path,
                    source_agent=source_agent,
                    session_id=os.environ.get("OSPREY_SESSION_ID", ""),
                    run_id=os.environ.get("OSPREY_DISPATCH_RUN_ID", ""),
                )
                self._entries.append(entry)
                appended = True
                self._save_index()
            except BaseException:
                if appended:
                    self._entries[:] = [e for e in self._entries if e.id != art_id]
                for path in written:
                    with suppress(OSError):
                        path.unlink()
                raise

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
        run_filter: str | None = None,
        last_n: int | None = None,
    ) -> list[ArtifactEntry]:
        """Return artifact entries, optionally filtered.

        ``run_filter`` is the strict dispatch-run isolation boundary — see the
        note on its clause below. ``session_filter`` is deliberately NOT such a
        boundary (it also matches untagged artifacts).
        """
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
            # NOT an isolation boundary: this OR-empty clause also matches
            # untagged (empty session_id) artifacts. Dispatch callers needing
            # strict per-run isolation must use ``run_filter`` / ``get_run_entry``
            # instead, which never match empty tags.
            entries = [e for e in entries if e.session_id == session_filter or not e.session_id]
        if run_filter is not None:
            # Strict created-by isolation: an artifact belongs to a run only if
            # it was tagged with that run's id at WRITE time. Deliberately does
            # NOT match empty run_ids, so untagged/legacy artifacts are never
            # attributed to any dispatch run. An empty filter matches nothing.
            if run_filter == "":
                return []
            entries = [e for e in entries if e.run_id == run_filter]
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

    def get_run_entry(self, run_id: str, artifact_id: str) -> ArtifactEntry | None:
        """Return the artifact only if the run created it — the cross-run gate.

        The single authoritative isolation predicate: an entry resolves only
        when its write-time ``run_id`` tag is non-empty and exactly equals
        ``run_id``. An empty ``run_id`` argument, an untagged entry, or a tag
        mismatch all return ``None`` — indistinguishable from an unknown id, so
        callers cannot use this as a cross-run existence oracle.
        """
        entry = self.get_entry(artifact_id)
        if entry is None or not entry.run_id or entry.run_id != run_id:
            return None
        return entry

    def _entry_data_file(self, entry: ArtifactEntry) -> Path | None:
        """Return the entry's *second* file, when it really has one.

        A two-file entry (an image plus the raw payload it was rendered from,
        say) records the companion file in ``data_file``. Single-file entries
        set ``data_file`` too, but only as an agent-facing pointer at the
        primary file — so a companion counts only when ``data_file`` is set,
        resolves to something other than ``filename``, and lands *inside* the
        store directory. That last condition is the guard that keeps a
        hand-edited, imported or hostile index from turning a delete into an
        arbitrary unlink somewhere else on the filesystem.

        Returns:
            The resolved companion path, or ``None`` when the entry has none.
        """
        if not entry.data_file:
            return None

        primary = (self._store_dir / entry.filename).resolve()
        store_dir = self._store_dir.resolve()
        raw = Path(entry.data_file)

        # ``data_file`` is written repo-root-relative (see ``save_data``), with
        # a bare-filename fallback; try both anchors, absolute paths as given.
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            try:
                candidates.append(self.repo_root / raw)
            except Exception:
                logger.debug("Could not anchor data_file at the repo root", exc_info=True)
            candidates.append(self._store_dir / raw)

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if resolved == primary or not resolved.is_relative_to(store_dir):
                    continue
                if resolved.exists():
                    return resolved
            except OSError:
                continue
        return None

    def _unlink_entry_files(self, entry: ArtifactEntry) -> None:
        """Remove every file an entry owns: the primary one and any companion."""
        data_file = self._entry_data_file(entry)
        filepath = self._store_dir / entry.filename
        if filepath.exists():
            filepath.unlink()
        if data_file is not None:
            with suppress(OSError):
                data_file.unlink()

    def _delete_entry_locked(
        self, artifact_id: str, *, persist: bool = True
    ) -> ArtifactEntry | None:
        """Delete one entry from *inside* an already-held index lock.

        The lock-free half of :meth:`delete_entry`, for callers that are
        already in a ``_with_index_lock()`` critical section (a save that
        prunes its own retention window, for instance). Calling the public
        method from there would self-deadlock: ``_with_index_lock`` takes an
        ``flock`` on a freshly opened descriptor, and ``flock`` is per
        open-file-description — the in-process ``RLock`` is reentrant, the file
        lock is not, so the second acquisition waits on the first forever.

        Delete listeners are deliberately *not* fired here; the caller fires
        them after leaving the lock, exactly as :meth:`delete_entry` does.

        Args:
            artifact_id: Id of the entry to remove.
            persist: Whether to rewrite the index. Pass ``False`` when the
                caller saves the index once for a batch of mutations.

        Returns:
            The removed entry, or ``None`` when no entry matched.
        """
        for i, e in enumerate(self._entries):
            if e.id == artifact_id:
                self._unlink_entry_files(e)
                del self._entries[i]
                if persist:
                    self._save_index()
                return e
        return None

    def delete_entry(self, artifact_id: str) -> bool:
        """Delete an artifact by ID, removing the index entry and its file(s).

        Returns:
            ``True`` if the entry was found and deleted, ``False`` otherwise.
        """
        with self._with_index_lock():
            deleted = self._delete_entry_locked(artifact_id)

        if deleted is None:
            return False

        self._notify_delete_listeners(deleted)
        return True

    def _prune_channel_readings_locked(
        self, address: str, keep: int, *, persist: bool = True
    ) -> list[ArtifactEntry]:
        """Trim *address*'s reading window from inside an already-held index lock.

        The lock-free half of :meth:`prune_channel_readings`, so a save that
        wants to prune its own window can do it without releasing the lock it
        already holds (``_with_index_lock`` is not reentrant across processes —
        see :meth:`_delete_entry_locked`).

        Each removal goes through ``_delete_entry_locked(persist=False)`` and
        the index is rewritten once at the end, so a sweep that drops ten
        entries costs one index write, not ten.

        Delete listeners are deliberately *not* fired here: the caller fires
        them after leaving the lock, so a listener that calls back into the
        store cannot deadlock.

        Args:
            address: Channel address to match against ``metadata["channel"]``.
            keep: Window size. ``0`` or negative keeps everything.
            persist: Whether to rewrite the index when anything was pruned.
                Pass ``False`` only if the caller persists afterwards anyway.

        Returns:
            The removed entries, oldest first (empty when nothing was pruned).
        """
        if keep <= 0 or not address:
            return []

        # Pinned entries are skipped on both counts: they are never pruned, and
        # they never occupy a slot in the window either. Counting them would
        # let ``keep`` pinned frames permanently starve the rolling window.
        #
        # Order is the index's insertion order, which is the store's house
        # convention for chronology (``list_entries(last_n=...)`` takes the
        # tail for the same reason): entries are only ever appended, so the
        # oldest reading for an address is the earliest surviving match.
        candidates = [
            e
            for e in self._entries
            if e.category == "channel_values"
            and not e.pinned
            and e.metadata.get("channel") == address
        ]
        excess = len(candidates) - keep
        if excess <= 0:
            return []

        pruned: list[ArtifactEntry] = []
        for entry in candidates[:excess]:
            removed = self._delete_entry_locked(entry.id, persist=False)
            if removed is not None:
                pruned.append(removed)

        if pruned and persist:
            self._save_index()
        return pruned

    def prune_channel_readings(self, address: str, keep: int) -> list[ArtifactEntry]:
        """Keep only the newest *keep* unpinned readings of one channel.

        The retention half of the channel-read artifact path: an
        approval-free read in a polling loop would otherwise grow the gallery
        without bound. Entries qualify when their category is
        ``channel_values`` and their ``metadata["channel"]`` equals *address*,
        which covers both shapes a reading is stored in — the JSON series from
        :meth:`save_data` and the PNG-plus-``.npy`` pair from
        :meth:`save_channel_reading`.

        The store stays config-agnostic: the window size is the caller's to
        supply, read from configuration by the tool that saves the reading.
        Call this right after a successful save.

        Pinned entries are exempt in both directions — never pruned, and never
        counted against the window, so pinning a frame cannot shrink the
        rolling window to nothing.

        The whole sweep is attributed to the ``"system"`` actor
        (:func:`artifact_mutation_actor`), so activity feeds never show a
        retention prune as something the agent did.

        Args:
            address: Channel address, matched exactly against
                ``metadata["channel"]``.
            keep: How many unpinned readings of this channel to keep. ``0``
                or negative means keep everything (the sweep is a no-op).

        Returns:
            The removed entries, oldest first (empty when nothing was pruned).
        """
        if keep <= 0 or not address:
            return []

        with artifact_mutation_actor("system"):
            with self._with_index_lock():
                pruned = self._prune_channel_readings_locked(address, keep)

            # Outside the lock: a listener is free to call back into the store.
            for entry in pruned:
                self._notify_delete_listeners(entry)

        return pruned

    def _delete_where(self, predicate: Callable[[ArtifactEntry], bool]) -> list[ArtifactEntry]:
        """Delete every entry matching *predicate* in one atomic operation.

        Unlinks the matching files and rewrites the index in a single locked
        block, then fires the delete listener once per removed entry outside
        the lock. Returns the removed entries.
        """
        with self._with_index_lock():
            doomed = [e for e in self._entries if predicate(e)]
            for e in doomed:
                self._unlink_entry_files(e)
            if doomed:
                survivors = [e for e in self._entries if not predicate(e)]
                self._entries[:] = survivors
                self._save_index()

        for entry in doomed:
            self._notify_delete_listeners(entry)

        return doomed

    def delete_category(self, category: str) -> list[ArtifactEntry]:
        """Delete every artifact whose ``category`` equals *category*.

        The scoped destructive path: it honours exactly the partition the read
        path honours (:meth:`list_entries` ``category_filter``), so clearing
        one category never touches another. Pass ``""`` to delete only
        uncategorised entries.

        Deletion is not recoverable — files are unlinked outright.

        Args:
            category: Category key to delete. ``""`` selects uncategorised
                entries.

        Returns:
            The removed entries (empty when nothing matched).
        """
        return self._delete_where(lambda e: e.category == category)

    def delete_everything(self) -> list[ArtifactEntry]:
        """Delete EVERY artifact in the store, across all categories.

        Named for its blast radius. This is the whole-store clear, not a
        scoped delete: it destroys archiver datasets, run results and data
        files written by :meth:`save_data` alongside plots, documents and
        screenshots, and the files are unlinked outright — there is no trash.
        Callers that mean a single partition must use :meth:`delete_category`.

        Returns:
            Every removed entry.
        """
        return self._delete_where(lambda e: True)

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
