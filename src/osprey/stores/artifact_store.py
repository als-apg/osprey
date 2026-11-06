"""Artifact storage for OSPREY MCP tools.

Manages interactive artifacts (plots, tables, HTML, markdown) produced by
Claude during analysis sessions.  Artifacts are stored under the agent-data
root (``agent_data.base_dir``) in ``artifacts/``, and served by the Artifact
Server gallery.

Two entry points create artifacts:
  1. ``save_artifact()`` — injected into execution subprocesses for live
     Python objects (see :mod:`osprey.stores.artifact_manifest`)
  2. ``artifact_register`` — MCP tool for files on disk / literal text

This module provides the low-level storage layer used by both.
"""

from __future__ import annotations

import base64
import contextvars
import json
import logging
import mimetypes
import os
import uuid
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from osprey.stores.artifact_forms import (
    RUN_RECORD_KEY,
    extension_key,
    form_of,
    is_run_record,
)
from osprey.stores.base_store import BaseStore, _sanitize_for_json
from osprey.stores.session_ledger import aware_time
from osprey_connectors.safety_file import atomic_replace

if TYPE_CHECKING:
    from osprey.stores.artifact_scope import SessionScope

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


@dataclass(frozen=True)
class ArtifactPage:
    """One page of a newest-first artifact listing.

    Attributes:
        entries: The entries of this page, in the order they are drawn
            (newest first).
        total: How many entries the filters admit across all pages.
        next_cursor: The token that fetches the page after this one, or
            ``None`` when this is the last page.
    """

    entries: list[ArtifactEntry]
    total: int
    next_cursor: str | None


def _page_key(entry: ArtifactEntry) -> tuple[str, str]:
    # The listing's order is newest first by timestamp, with the id breaking a
    # tie so two artifacts saved in the same microsecond still have one stable
    # position.
    return (entry.timestamp, entry.id)


# The page cursor is base64url so a timestamp's ``:`` and ``+`` survive a query
# string untouched, and opaque so no caller builds one by hand. It carries no
# authority and needs none: it names a position in the caller's own filtered
# view, so a tampered token yields a different slice of data that caller may
# already read.
def _encode_cursor(entry: ArtifactEntry) -> str:
    payload = json.dumps({"timestamp": entry.timestamp, "id": entry.id})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def _decode_cursor(token: str) -> tuple[str, str]:
    try:
        data = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"not a listing cursor: {token!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"not a listing cursor: {token!r}")
    timestamp = data.get("timestamp")
    entry_id = data.get("id")
    if not isinstance(timestamp, str) or not isinstance(entry_id, str):
        raise ValueError(f"not a listing cursor: {token!r}")
    return (timestamp, entry_id)


def page_of(entries: list[ArtifactEntry], limit: int, cursor: str | None = None) -> ArtifactPage:
    """One newest-first page of *entries*.

    The entries are ordered by ``(timestamp, id)`` descending; ``cursor`` (a
    previous page's ``next_cursor``) resumes strictly after the entry it names.
    The cursor is a key, not an offset, so an artifact saved or deleted between
    two requests does not shift the next page. A cursor past every entry yields
    an empty page rather than an error.

    Raises:
        ValueError: ``cursor`` is not a token a listing issued.
    """
    ordered = sorted(entries, key=_page_key, reverse=True)
    total = len(ordered)
    start = 0
    if cursor is not None:
        key = _decode_cursor(cursor)
        start = sum(1 for e in ordered if _page_key(e) >= key)
    page = ordered[start : start + limit]
    next_cursor = _encode_cursor(page[-1]) if page and total > start + len(page) else None
    return ArtifactPage(entries=page, total=total, next_cursor=next_cursor)


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
    the agent generated. ``"watchdog"`` (:data:`WATCHDOG_ORIGIN`) marks a
    watchdog judge's artifact, which :func:`operator_visible` hides. A save with
    no explicit origin takes ``OSPREY_ARTIFACT_ORIGIN``."""
    exec_id: str = ""
    """The Claude Code ``tool_use_id`` of the script run this artifact is part of
    (preview, code, output or result), empty when it belongs to no run. Entries
    sharing a non-empty ``exec_id`` form one run group (:func:`classify`)."""
    kind: str = ""
    """The registered extension type this artifact is (``osprey.stores.type_registry``),
    empty for none. ``artifact_type`` stays the file format."""

    def to_dict(self) -> dict[str, Any]:
        # ``_sanitize_for_json`` already rebuilds every nested dict and list, so
        # ``asdict``'s deep copy before it only doubled the cost of every index
        # write (each one serialises the whole store).
        sanitized: dict[str, Any] = _sanitize_for_json(dict(vars(self)))
        return sanitized

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
        if self.kind:
            resp["kind"] = self.kind
        if self.exec_id:
            resp["exec_id"] = self.exec_id
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
        import plotly.graph_objects as go

        if isinstance(obj, go.Figure):
            html = obj.to_html(include_plotlyjs=False, full_html=True)
            return html.encode(), "plot_html", f"{slug}.html", "text/html"
    except ImportError:
        pass

    # --- Bokeh Model (layout, figure, widget, ...) ---
    try:
        import bokeh.model

        if isinstance(obj, bokeh.model.Model):
            from bokeh.embed import file_html
            from bokeh.resources import INLINE

            html = file_html(obj, resources=INLINE, title=title)
            return html.encode(), "dashboard_html", f"{slug}.html", "text/html"
    except ImportError:
        pass

    # --- Matplotlib Figure ---
    try:
        import matplotlib.figure

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
        import pandas as pd

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


EXAMPLE_ORIGIN = "demo"
"""``origin`` of an artifact shipped with OSPREY rather than produced in a
run: the example the gallery seeds into an empty workspace. Listed by the
gallery, hidden from every agent-facing listing (``exclude_examples``)."""

EXAMPLE_REMOVED_SENTINEL = ".example-removed"
"""File in the artifact directory written when an example is deleted, so no
later launch seeds it again. Removing the file brings the example back."""

WATCHDOG_ORIGIN = "watchdog"
"""``origin`` of an artifact a watchdog judge run saved (its env overlay sets
``OSPREY_ARTIFACT_ORIGIN``). Hidden from the operator's tree by
:func:`operator_visible`."""

ARTIFACT_ORIGIN_ENV = "OSPREY_ARTIFACT_ORIGIN"
"""Env var a save without an explicit ``origin`` takes its origin from."""

CATEGORY_DEPRECATION = (
    "The `category` argument of the artifact save APIs is deprecated and ignored; "
    "artifacts are grouped by form (and an extension type's `kind`) instead"
)
"""The ``DeprecationWarning`` a save with a non-empty ``category`` raises."""


def _warn_category(category: str) -> None:
    """Warn that ``category`` is deprecated when a caller passed one."""
    if category:
        warnings.warn(CATEGORY_DEPRECATION, DeprecationWarning, stacklevel=3)


def _resolved_origin(origin: str) -> str:
    """An explicit *origin*, else ``OSPREY_ARTIFACT_ORIGIN``, else ``""``."""
    return origin or os.environ.get(ARTIFACT_ORIGIN_ENV, "")


def _stamped_metadata(metadata: Mapping[str, Any] | None, now: str) -> dict[str, Any]:
    """A new entry's metadata with the first-save stamps.

    ``first_ts`` is the first save's time (ownership reads it; a touch never
    refreshes it). ``call_id`` is the tool call in scope, as provenance only.
    A value the caller already set wins.
    """
    from osprey.audit.call import current_tool_use_id

    stamped = dict(metadata or {})
    stamped.setdefault("first_ts", now)
    call_id = current_tool_use_id()
    if call_id is not None:
        stamped.setdefault("call_id", call_id)
    return stamped


def _is_channel_reading(entry: ArtifactEntry) -> bool:
    """Whether *entry* is a channel reading retention may prune.

    New readings are told by their writer (``tool_source="channel_read"``);
    readings saved before category retired still carry ``channel_values``.
    """
    return entry.tool_source == "channel_read" or entry.category == "channel_values"


# ---- run groups ------------------------------------------------------------

RunRole = Literal["preview", "code", "output", "result"]
RunStatus = Literal["succeeded", "failed", "not_run"]

RUN_ROLES: frozenset[str] = frozenset({"preview", "code", "output", "result"})
"""Every value ``metadata.run_part`` may take."""

RUN_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "not_run"})
"""Every value a result head's ``metadata.run_status`` may take."""

_EXECUTOR_TOOLS = frozenset({"execute", "execute_file"})


@dataclass(frozen=True)
class RunGroup:
    """How one ``exec_id`` group's entries are read: its head, status and roles.

    Attributes:
        head_id: The entry listed as the run (``runs``); ``""`` only for a
            group with no result, preview or notebook (outputs alone).
        run_status: ``succeeded``/``failed``/``not_run`` from stored state, or
            ``None`` while the run is in flight.
        roles: Every member's id mapped to its role.
    """

    head_id: str
    run_status: RunStatus | None
    roles: dict[str, RunRole]

    @property
    def in_flight(self) -> bool:
        """Whether no stored head or settle has finished the run yet."""
        return self.run_status is None


def _is_legacy_result(entry: ArtifactEntry) -> bool:
    """A result record the executor saved before ``run_part`` existed."""
    if entry.tool_source not in _EXECUTOR_TOOLS:
        return False
    if entry.category == "code_output":
        return True
    return entry.artifact_type == "json" and entry.metadata.get(RUN_RECORD_KEY) is True


def run_role(entry: ArtifactEntry) -> RunRole:
    """The role *entry* plays in a run: its ``metadata.run_part``, else the legacy shape.

    Legacy shapes: the executor's result record → ``result``, the approval
    hook's review → ``preview``, the executor's ``Notebook: …`` → ``code``;
    anything else → ``output``.
    """
    part = entry.metadata.get("run_part")
    if part in RUN_ROLES:
        return cast(RunRole, part)
    if _is_legacy_result(entry):
        return "result"
    if entry.tool_source == "osprey_approval":
        return "preview"
    if (
        entry.artifact_type == "notebook"
        and entry.tool_source in _EXECUTOR_TOOLS
        and entry.title.startswith("Notebook: ")
    ):
        return "code"
    return "output"


def _result_status(entry: ArtifactEntry) -> RunStatus:
    """A result head's status: its stored ``run_status``, else what its result says."""
    stored = entry.metadata.get("run_status")
    if stored in RUN_STATUSES:
        return cast(RunStatus, stored)
    summary = entry.summary or {}
    if summary.get("has_errors") is True or summary.get("status") == "Failed":
        return "failed"
    return "succeeded"


def _latest(entries: Sequence[ArtifactEntry]) -> ArtifactEntry | None:
    return max(entries, key=_page_key) if entries else None


def classify(group: Iterable[ArtifactEntry]) -> RunGroup:
    """Read one ``exec_id`` group from stored state alone.

    Precedence: a result record is the head with its stored status (a legacy
    result: ``succeeded`` unless it says failed); else a preview settled
    ``not_run`` or ``failed`` is the head with that status; else the run is in
    flight — its preview, else its notebook, heads it with ``run_status=None``.
    Among several candidates of one kind the newest wins.
    """
    members = list(group)
    roles: dict[str, RunRole] = {e.id: run_role(e) for e in members}

    def of(role: str) -> list[ArtifactEntry]:
        return [e for e in members if roles[e.id] == role]

    result = _latest(of("result"))
    if result is not None:
        return RunGroup(result.id, _result_status(result), roles)
    previews = of("preview")
    settled = _latest([e for e in previews if e.metadata.get("settled") in ("not_run", "failed")])
    if settled is not None:
        return RunGroup(settled.id, settled.metadata["settled"], roles)
    head = _latest(previews) or _latest(of("code"))
    return RunGroup(head.id if head is not None else "", None, roles)


def run_parts(
    run_group: RunGroup, members: Iterable[ArtifactEntry], registry: Any = None
) -> list[dict[str, Any]]:
    """The parts a run head lists: ``[{id, role, title, form}]``, oldest first.

    Every one of the group's *members* but the head; ``form`` is the part's
    intrinsic form, except that an output takes the form it lists under on its
    own row (:func:`listed_form`), so an extension ``kind`` keeps its key.
    """
    return [
        {
            "id": e.id,
            "role": run_group.roles[e.id],
            "title": e.title,
            "form": (
                listed_form(e, run_group, registry) if run_group.roles[e.id] == "output" else None
            )
            or form_of(e, registry),
        }
        for e in sorted(members, key=_page_key)
        if e.id != run_group.head_id
    ]


def listed_form(
    entry: ArtifactEntry, run_group: RunGroup | None, registry: Any = None
) -> str | None:
    """The form *entry* is listed under, or ``None`` for a run's part.

    In a group: the head → ``runs``; a preview, notebook or second result → a
    part (``None``); an output → its intrinsic form (it is also a part of its
    run). Outside a group (``exec_id == ""``): a result record → ``runs``, a
    legacy preview or notebook → ``documents``, anything else its intrinsic form.

    An entry whose ``kind`` is an extension key in *registry* (it declares an
    eligible base form) lists under that ``kind`` instead — unless it is a run
    head (``runs``) or a run part (``None``), which stay as they are.
    """
    listed = _listed_base_form(entry, run_group, registry)
    if listed is None or listed == "runs":
        return listed
    return extension_key(entry, registry) or listed


def _listed_base_form(
    entry: ArtifactEntry, run_group: RunGroup | None, registry: Any
) -> str | None:
    """:func:`listed_form` before an extension ``kind`` is applied."""
    role = run_role(entry)
    if not entry.exec_id or run_group is None or entry.id not in run_group.roles:
        if role == "result":
            return "runs"
        if role in ("preview", "code"):
            return "documents"
        return form_of(entry, registry)
    if entry.id == run_group.head_id:
        return "runs"
    if run_group.roles[entry.id] == "output":
        return form_of(entry, registry)
    return None


def operator_visible(entry: ArtifactEntry) -> bool:
    """Whether *entry* belongs in the operator's session tree, counts and Earlier.

    Dispatch-run artifacts (non-empty ``run_id``) and watchdog judge artifacts
    (``origin == "watchdog"``) do not; the palette search still finds them.
    """
    return not entry.run_id and entry.origin != WATCHDOG_ORIGIN


def _reuse_rule(scope: SessionScope) -> Callable[[ArtifactEntry], bool]:
    """Which existing entries a sha256 save under *scope* may reuse (contract C4)."""
    if scope.kind in ("web", "cli"):
        from osprey.stores.artifact_scope import owned_by

        return lambda entry: owned_by(entry, scope)
    if scope.kind == "none":
        return untagged
    return lambda entry: False


def untagged(entry: ArtifactEntry) -> bool:
    """Whether *entry* is operator-visible and names no session: no ``session_id``
    and no stamped ``transcript_id``, so no ledger can give it an owner."""
    return (
        operator_visible(entry) and not entry.session_id and not entry.metadata.get("transcript_id")
    )


# ---- ownership -------------------------------------------------------------


class Ledger(Protocol):
    """The read side of the session ledger (``stores/session_ledger.py``).

    ``rows`` are ``(session_key, transcript_id, from_ts)`` in append order,
    as tuples or objects with those attributes; ``from_ts`` is an aware
    ISO-8601 string or datetime.
    """

    @property
    def rows(self) -> Sequence[Any]: ...


def _row_fields(row: Any) -> tuple[str, str, Any]:
    if isinstance(row, (tuple, list)):
        return row[0], row[1], row[2]
    return row.session_key, row.transcript_id, row.from_ts


def owner_of(entry: ArtifactEntry, ledger: Ledger | None) -> str | None:
    """The transcript that owns *entry*, or ``None`` (it is listed under Earlier).

    1. ``metadata.transcript_id`` when stamped;
    2. else, when ``entry.session_id`` is a ledger key, the transcript of that
       key's latest row with ``from_ts <= first_ts``, or of its first row when
       none is (a key's first transcript has an open lower bound). ``first_ts``
       is ``metadata.first_ts``, falling back to ``timestamp``;
    3. else ``None``.

    Times are compared as aware datetimes, never as strings.
    """
    stamped = entry.metadata.get("transcript_id")
    if isinstance(stamped, str) and stamped:
        return stamped
    if ledger is None or not entry.session_id:
        return None
    rows = [r for r in (_row_fields(row) for row in ledger.rows) if r[0] == entry.session_id]
    if not rows:
        return None
    first_ts = aware_time(entry.metadata.get("first_ts")) or aware_time(entry.timestamp)
    owner = rows[0][1]
    if first_ts is None:
        return owner
    best: datetime | None = None
    for _key, transcript_id, from_ts in rows:
        start = aware_time(from_ts)
        if start is None or start > first_ts:
            continue
        if best is None or start >= best:
            best, owner = start, transcript_id
    return owner


def _window_bound(value: str | datetime | None, name: str) -> datetime | None:
    if value is None:
        return None
    parsed = aware_time(value)
    if parsed is None:
        raise ValueError(f"{name} must be an aware ISO-8601 time: {value!r}")
    return parsed


def in_window(
    entry: ArtifactEntry, after: str | datetime | None, before: str | datetime | None
) -> bool:
    """Whether *entry*'s ``timestamp`` lies in the half-open window ``[after, before)``.

    Raises:
        ValueError: A bound is not an aware ISO-8601 time.
    """
    low = _window_bound(after, "after")
    high = _window_bound(before, "before")
    ts = aware_time(entry.timestamp)
    if ts is None:
        return low is None and high is None
    return (low is None or ts >= low) and (high is None or ts < high)


class ArtifactStore(BaseStore[ArtifactEntry]):
    """Manages artifact files and the artifacts.json index."""

    _store_name = "artifacts"
    _subdir = "artifacts"
    _index_filename = "artifacts.json"

    def __init__(self, workspace_root: Path | None = None, *, auto_launch: bool = True) -> None:
        """Build the store.

        Args:
            workspace_root: Root the store's files live under. Resolved from the
                config when omitted.
            auto_launch: Whether a save starts the artifact gallery server. A
                save from a long-lived process launches it, so the artifact is
                viewable at once. A caller whose process ends right after the
                save turns it off: a server thread started there dies with the
                process and serves nobody.
        """
        self._auto_launch = auto_launch
        super().__init__(workspace_root)

    def _launch_gallery(self) -> None:
        """Start the artifact gallery server, unless this store opted out."""
        if not self._auto_launch:
            return
        try:
            from osprey.infrastructure.server_launcher import ensure_web_server

            ensure_web_server("artifact")
        except Exception as exc:
            logger.warning("Artifact server auto-launch failed: %s", exc, exc_info=True)

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
        d.setdefault("exec_id", "")
        d.setdefault("kind", "")
        return ArtifactEntry(**d)

    def _entry_to_dict(self, entry: ArtifactEntry) -> dict:
        return entry.to_dict()

    def _session_tag(self) -> str:
        """The default ``session_id`` of a save from this process (C2 ``session_tag``).

        Reads the environment and the CLI pointer under this store's workspace
        root, never the ledger, so it is safe under the index lock.
        """
        from osprey.stores.artifact_scope import session_tag

        return session_tag(root=self._workspace)

    def _make_id(self) -> str:
        return uuid.uuid4().hex[:12]

    @property
    def artifact_dir(self) -> Path:
        return self._store_dir

    @property
    def example_removed(self) -> bool:
        """Whether someone deleted the shipped example from this store."""
        return (self._store_dir / EXAMPLE_REMOVED_SENTINEL).exists()

    def _mark_example_removed(self) -> None:
        with suppress(OSError):
            (self._store_dir / EXAMPLE_REMOVED_SENTINEL).touch()

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
        session_id: str | None = None,
        *,
        exec_id: str = "",
        kind: str = "",
    ) -> ArtifactEntry:
        """Low-level save: write raw bytes to disk and register in the index.

        ``run_id`` defaults to the ``OSPREY_DISPATCH_RUN_ID`` env var (how the
        dispatched agent's own tools tag their output). A caller that writes on
        behalf of a run whose id is not in its environment — e.g. the worker
        ingesting input files before the agent starts — passes it explicitly.
        ``session_id`` defaults to :func:`~osprey.stores.artifact_scope.session_tag`
        (the web key, ``cli:<transcript>``, else ``""``); pass ``""`` for an
        entry that belongs to no session (the shipped example).
        ``origin`` tags the entry's provenance within the run (``"input"`` for
        an ingested caller file, :data:`EXAMPLE_ORIGIN` for the shipped
        example; empty for produced output). An empty ``origin`` takes
        ``OSPREY_ARTIFACT_ORIGIN``. ``exec_id`` links the entry into a run group;
        ``kind`` names its extension type. ``category`` is deprecated and
        ignored."""
        _warn_category(category)
        with self._with_index_lock():
            entry = self._save_file_locked(
                file_content,
                filename,
                artifact_type,
                title,
                description=description,
                mime_type=mime_type,
                tool_source=tool_source,
                metadata=metadata,
                run_id=run_id,
                origin=origin,
                session_id=session_id,
                exec_id=exec_id,
                kind=kind,
            )

        self._notify_listeners(entry)

        self._launch_gallery()

        return entry

    def _save_file_locked(
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
        session_id: str | None = None,
        *,
        exec_id: str = "",
        kind: str = "",
    ) -> ArtifactEntry:
        """Write the bytes and the index row and save the index.

        The caller holds :meth:`_with_index_lock` (the flock is not reentrant)
        and fires the listeners after releasing it. ``category`` is accepted
        for old ``save_kwargs`` and never stored.
        """
        del category  # deprecated and ignored; the public callers warned
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
            timestamp=(now := datetime.now(UTC).isoformat()),
            tool_source=tool_source,
            metadata=_stamped_metadata(metadata, now),
            session_id=(session_id if session_id is not None else self._session_tag()),
            run_id=(run_id if run_id is not None else os.environ.get("OSPREY_DISPATCH_RUN_ID", "")),
            origin=_resolved_origin(origin),
            exec_id=exec_id,
            kind=kind,
        )
        self._entries.append(entry)
        self._save_index()
        return entry

    def save_or_touch_by_sha256(
        self,
        sha256: str,
        *,
        origin: str,
        save_kwargs: dict[str, Any],
        scope: SessionScope,
    ) -> ArtifactEntry:
        """The entry of *origin* holding bytes with this *sha256*, saved once per scope.

        A match is an entry with this ``origin`` and ``metadata["sha256"]``.
        Under one index lock, the newest match that qualifies for *scope* gets a
        fresh timestamp (so retention keeps it as long as it is still being
        seen) and nothing else — ``metadata.first_ts`` keeps the first save's
        time, which ownership reads. Otherwise ``save_kwargs`` (the
        :meth:`save_file` arguments) are saved as a new entry with *origin*.
        A match qualifies for a ``web`` or ``cli`` scope when the scope owns
        it; for a ``none`` scope when it is operator-visible and untagged (no
        ``session_id``, no stamped ``transcript_id``); for a ``dispatch`` scope
        never. Lookup and save share the lock, so two concurrent callers with
        the same bytes and scope produce one entry. Listeners fire for a new
        entry only.
        """
        kwargs = {k: v for k, v in save_kwargs.items() if k != "origin"}
        _warn_category(kwargs.pop("category", ""))
        qualifies = _reuse_rule(scope)
        with self._with_index_lock():
            for e in reversed(self._entries):
                if e.origin == origin and e.metadata.get("sha256") == sha256 and qualifies(e):
                    e.timestamp = datetime.now(UTC).isoformat()
                    self._save_index()
                    return e
            created = self._save_file_locked(**kwargs, origin=origin)

        self._notify_listeners(created)
        self._launch_gallery()
        return created

    def save_object(
        self,
        obj: Any,
        title: str,
        description: str = "",
        artifact_type: str | None = None,
        tool_source: str = "execute",
        metadata: dict[str, Any] | None = None,
        category: str = "",
        *,
        exec_id: str = "",
        kind: str = "",
        origin: str = "",
    ) -> ArtifactEntry:
        """Smart-serialize a Python object and save it.

        Detects Plotly figures, matplotlib figures, DataFrames, strings, dicts
        and numpy arrays (saved as ``.npy``). ``category`` is deprecated and
        ignored.
        """
        _warn_category(category)
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
            origin=origin,
            exec_id=exec_id,
            kind=kind,
        )

    def save_from_path(
        self,
        source_path: str | Path,
        title: str,
        description: str = "",
        artifact_type: str | None = None,
        tool_source: str = "artifact_register",
        metadata: dict[str, Any] | None = None,
        category: str = "",
        *,
        exec_id: str = "",
        kind: str = "",
        origin: str = "",
    ) -> ArtifactEntry:
        """Copy an existing file into the artifact store. ``category`` is deprecated and ignored."""
        _warn_category(category)
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
            origin=origin,
            exec_id=exec_id,
            kind=kind,
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
        *,
        exec_id: str = "",
        kind: str = "",
        origin: str = "",
    ) -> ArtifactEntry:
        """Save structured data as an artifact.  Replaces ``DataContext.save()``.

        Writes raw JSON (no envelope) to ``artifacts/{id}_{tool}.json`` and
        registers the entry in the unified artifact index. ``category`` is
        deprecated and ignored.
        """
        _warn_category(category)
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
                timestamp=(now := datetime.now(UTC).isoformat()),
                tool_source=tool,
                metadata=_stamped_metadata(metadata, now),
                summary=summary or {},
                access_details=access_details or {},
                data_file=agent_path,
                source_agent=source_agent,
                session_id=self._session_tag(),
                run_id=os.environ.get("OSPREY_DISPATCH_RUN_ID", ""),
                origin=_resolved_origin(origin),
                exec_id=exec_id,
                kind=kind,
            )
            self._entries.append(entry)
            self._save_index()

        self._notify_listeners(entry)

        self._launch_gallery()

        return entry

    def save_channel_reading(
        self,
        png_bytes: bytes,
        npy_bytes: bytes,
        title: str,
        summary: dict[str, Any] | None = None,
        access_details: dict[str, Any] | None = None,
        category: str = "",
        metadata: dict[str, Any] | None = None,
        description: str = "",
        tool_source: str = "channel_read",
        source_agent: str = "",
        *,
        exec_id: str = "",
        kind: str = "",
        origin: str = "",
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
            category: Deprecated and ignored.
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
        _warn_category(category)
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
                    timestamp=(now := datetime.now(UTC).isoformat()),
                    tool_source=tool_source,
                    metadata=_stamped_metadata(metadata, now),
                    summary=summary or {},
                    access_details=access_details or {},
                    data_file=agent_path,
                    source_agent=source_agent,
                    session_id=self._session_tag(),
                    run_id=os.environ.get("OSPREY_DISPATCH_RUN_ID", ""),
                    origin=_resolved_origin(origin),
                    exec_id=exec_id,
                    kind=kind,
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

        self._launch_gallery()

        return entry

    def list_entries(
        self,
        type_filter: str | None = None,
        search: str | None = None,
        pinned: bool | None = None,
        tool_filter: str | None = None,
        run_filter: str | None = None,
        run_record: bool | None = None,
        exclude_examples: bool = False,
        *,
        form_filter: str | None = None,
        after: str | datetime | None = None,
        before: str | datetime | None = None,
        registry: Any = None,
    ) -> list[ArtifactEntry]:
        """Return artifact entries, optionally filtered.

        ``run_filter`` is the strict dispatch-run isolation boundary — see the
        note on its clause below.

        ``run_record`` keeps run records (``True``) or everything else
        (``False``) by :func:`is_run_record`; it has nothing to do with
        ``run_filter``, which names a dispatch run.

        ``exclude_examples`` drops the shipped example (``origin ==
        EXAMPLE_ORIGIN``). Every listing the agent reads passes it: the
        example is for the person at the gallery, never something the agent
        produced or may cite.

        ``search`` is a case-insensitive substring match on the title, file
        name, description and artifact type.

        ``form_filter`` keeps the entries listed under that form
        (:func:`listed_form`, with run groups read from the whole index;
        ``registry`` defaults to the process's extension registry). ``after`` /
        ``before`` keep the half-open time window ``[after, before)`` (aware
        ISO-8601; :func:`in_window`).
        """
        self._refresh_if_stale()
        entries = list(self._entries)
        if after is not None or before is not None:
            entries = [e for e in entries if in_window(e, after, before)]
        if form_filter:
            entries = self._filter_form(entries, form_filter, registry)
        if exclude_examples:
            entries = [e for e in entries if e.origin != EXAMPLE_ORIGIN]
        if type_filter:
            entries = [e for e in entries if e.artifact_type == type_filter]
        if tool_filter:
            entries = [e for e in entries if e.tool_source == tool_filter]
        if run_filter is not None:
            # Strict created-by isolation: an artifact belongs to a run only if
            # it was tagged with that run's id at WRITE time. Deliberately does
            # NOT match empty run_ids, so untagged/legacy artifacts are never
            # attributed to any dispatch run. An empty filter matches nothing.
            if run_filter == "":
                return []
            entries = [e for e in entries if e.run_id == run_filter]
        if run_record is not None:
            entries = [e for e in entries if is_run_record(e) == run_record]
        if pinned is not None:
            entries = [e for e in entries if e.pinned == pinned]
        if search:
            # These are the four fields the gallery's filter box offers; the box
            # and this clause must admit the same artifacts, or a paged listing
            # hides matches the box promises.
            q = search.lower()
            entries = [
                e
                for e in entries
                if q in e.title.lower()
                or q in e.filename.lower()
                or q in e.description.lower()
                or q in e.artifact_type.lower()
            ]
        return entries

    def page_entries(
        self,
        *,
        limit: int,
        cursor: str | None = None,
        type_filter: str | None = None,
        search: str | None = None,
        pinned: bool | None = None,
        tool_filter: str | None = None,
        run_filter: str | None = None,
        run_record: bool | None = None,
        exclude_examples: bool = False,
        form_filter: str | None = None,
        after: str | datetime | None = None,
        before: str | datetime | None = None,
        registry: Any = None,
    ) -> ArtifactPage:
        """Return one newest-first page of the entries the filters admit.

        Filtering is exactly :meth:`list_entries`'s; ordering and the cursor are
        :func:`page_of`'s.

        Raises:
            ValueError: ``cursor`` is not a token this method issued.
        """
        entries = self.list_entries(
            type_filter=type_filter,
            search=search,
            pinned=pinned,
            tool_filter=tool_filter,
            run_filter=run_filter,
            run_record=run_record,
            exclude_examples=exclude_examples,
            form_filter=form_filter,
            after=after,
            before=before,
            registry=registry,
        )
        return page_of(entries, limit, cursor)

    def set_pinned(
        self, artifact_id: str, pinned: bool = True, *, set_in: str | None = None
    ) -> ArtifactEntry | None:
        """Toggle the pinned flag on an artifact. Returns the updated entry or None.

        A pin records the transcript it was set in as ``metadata["pinned_in"]``;
        re-pinning moves that record, and a pin with ``set_in=None`` leaves none.
        Unpinning clears both the flag and the record. The record lives in
        ``metadata`` so an older reader of the index still loads the entry.
        """
        with self._with_index_lock():
            for e in self._entries:
                if e.id == artifact_id:
                    e.pinned = pinned
                    metadata = {k: v for k, v in e.metadata.items() if k != "pinned_in"}
                    if pinned and set_in is not None:
                        metadata["pinned_in"] = set_in
                    e.metadata = metadata
                    self._save_index()
                    return e
        return None

    def replace_file(
        self,
        artifact_id: str,
        file_content: bytes,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactEntry | None:
        """Rewrite one artifact's bytes in place, and its description and metadata if given.

        The entry keeps its id, filename, title and timestamp, so it stays where it sits
        in the gallery. The bytes are replaced whole through ``atomic_replace`` under the
        index lock, so a reader sees the old file or the new one, never a partial file.
        ``metadata`` is merged into the entry's existing metadata. Returns the updated
        entry, or None when no entry has this id.
        """
        with self._with_index_lock():
            entry = next((e for e in self._entries if e.id == artifact_id), None)
            if entry is None:
                return None
            path = self._store_dir / entry.filename
            atomic_replace(path, file_content, mode=path.stat().st_mode & 0o777, sync="best_effort")
            entry.size_bytes = len(file_content)
            if description is not None:
                entry.description = description
            if metadata is not None:
                entry.metadata = {**entry.metadata, **metadata}
            self._save_index()
            return entry

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
        if entry.origin == EXAMPLE_ORIGIN:
            # Every delete path (single, category, whole store) comes through
            # here, so a removed example stays removed whichever one ran.
            self._mark_example_removed()

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
        # convention for chronology: entries are only ever appended, so the
        # oldest reading for an address is the earliest surviving match.
        candidates = [
            e
            for e in self._entries
            if _is_channel_reading(e) and not e.pinned and e.metadata.get("channel") == address
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
        without bound. Entries qualify when they are channel readings
        (``tool_source == "channel_read"``, or the legacy ``channel_values``
        category) and their ``metadata["channel"]`` equals *address*,
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

        The scoped destructive path for entries saved before categories
        retired: clearing one category never touches another. Pass ``""`` to
        delete only uncategorised entries.

        Deletion is not recoverable — files are unlinked outright.

        Args:
            category: Category key to delete. ``""`` selects uncategorised
                entries.

        Returns:
            The removed entries (empty when nothing matched).
        """
        return self._delete_where(lambda e: e.category == category)

    def delete_entries(self, artifact_ids: Iterable[str]) -> list[ArtifactEntry]:
        """Delete the artifacts with these ids in one atomic operation.

        Ids the store does not hold are ignored. Deletion is not recoverable —
        files are unlinked outright.

        Returns:
            The removed entries, in index order.
        """
        wanted = set(artifact_ids)
        if not wanted:
            return []
        return self._delete_where(lambda e: e.id in wanted)

    # ---- run groups and forms ----------------------------------------------

    def run_members(self, exec_id: str) -> list[ArtifactEntry]:
        """The entries of run group *exec_id*, in index order (none for ``""``)."""
        if not exec_id:
            return []
        self._refresh_if_stale()
        return [e for e in self._entries if e.exec_id == exec_id]

    def run_group(self, exec_id: str) -> RunGroup | None:
        """:func:`classify` of group *exec_id*, or ``None`` when it has no entries."""
        members = self.run_members(exec_id)
        return classify(members) if members else None

    def run_groups(self) -> dict[str, RunGroup]:
        """Every run group in the store, classified."""
        return _groups_of(self.list_entries())

    def _filter_form(
        self, entries: list[ArtifactEntry], form: str, registry: Any
    ) -> list[ArtifactEntry]:
        groups = _groups_of(self._entries)
        registry = registry_or_default(registry)
        return [e for e in entries if listed_form(e, groups.get(e.exec_id), registry) == form]

    def delete_with_group(self, artifact_id: str) -> list[ArtifactEntry]:
        """Delete an artifact; deleting a run's head deletes its whole group.

        A stored or in-flight head takes every entry sharing its ``exec_id``;
        any other entry (an output, a part, an ungrouped artifact) goes alone.

        Returns:
            The removed entries (empty when the id is unknown).
        """
        entry = self.get_entry(artifact_id)
        if entry is None:
            return []
        if entry.exec_id:
            group = self.run_group(entry.exec_id)
            if group is not None and group.head_id == entry.id:
                exec_id = entry.exec_id
                return self._delete_where(lambda e: e.exec_id == exec_id)
        return self._delete_where(lambda e: e.id == artifact_id)

    def delete_owned(
        self, scope: SessionScope, form: str | None = None, registry: Any = None
    ) -> list[ArtifactEntry]:
        """Delete what *scope* owns (:func:`~osprey.stores.artifact_scope.owned_by`).

        ``form=None`` deletes everything the scope owns, run parts included.
        ``"runs"`` deletes each owned run head with the owned members of its
        group. Any other form or extension key deletes the owned entries listed
        under it. An entry another session owns is never touched, even inside a
        run group split by a ``/clear``; a scope with no transcript deletes
        nothing. Runs are classified over the whole index under the delete's own
        lock; delete listeners fire after it.

        Returns:
            The removed entries, in index order.
        """
        from osprey.stores.artifact_scope import owned_by

        if scope.transcript is None:
            return []
        if form is not None:
            registry = registry_or_default(registry)
        with self._with_index_lock():
            owned = [e for e in self._entries if owned_by(e, scope)]
            if form is None:
                gone = {e.id for e in owned}
            else:
                groups = _groups_of(self._entries)
                gone = {
                    e.id for e in owned if listed_form(e, groups.get(e.exec_id), registry) == form
                }
                heads = {exec_id for exec_id, group in groups.items() if group.head_id in gone}
                gone |= {e.id for e in owned if e.exec_id in heads}
            doomed = [e for e in self._entries if e.id in gone]
            for e in doomed:
                self._unlink_entry_files(e)
            if doomed:
                self._entries[:] = [e for e in self._entries if e.id not in gone]
                self._save_index()

        for entry in doomed:
            self._notify_delete_listeners(entry)
        return doomed

    def get_file_path(self, artifact_id: str) -> Path | None:
        entry = self.get_entry(artifact_id)
        if entry is None:
            return None
        return self._store_dir / entry.filename


def _groups_of(entries: Iterable[ArtifactEntry]) -> dict[str, RunGroup]:
    """Every non-empty ``exec_id`` among *entries*, classified."""
    members: dict[str, list[ArtifactEntry]] = {}
    for e in entries:
        if e.exec_id:
            members.setdefault(e.exec_id, []).append(e)
    return {exec_id: classify(group) for exec_id, group in members.items()}


def registry_or_default(registry: Any = None) -> Any:
    """*registry*, else the process's extension registry (empty on failure)."""
    if registry is not None:
        return registry
    try:
        from osprey.stores.type_registry import get_extension_types

        return get_extension_types()
    except Exception:  # a broken registry never breaks a listing
        logger.debug("Extension registry unavailable", exc_info=True)
        return None


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
