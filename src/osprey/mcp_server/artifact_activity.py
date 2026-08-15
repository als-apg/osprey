"""Artifact-mutation activity emits for the Web Terminal.

Every artifact save and delete passes through :class:`~osprey.stores.artifact_store.ArtifactStore`,
which fires the listener seam registered by
:func:`osprey.stores.artifact_store.register_artifact_listener` /
:func:`~osprey.stores.artifact_store.register_artifact_delete_listener`. This
module is the single subscriber that turns those store events into
``/api/agent-activity`` frames, so artifact visibility does not have to be
re-implemented in every tool that writes one.

Three properties shape the implementation:

**Only agent mutations emit.** The store fires the same listeners for a human
deleting from the gallery UI and for the dispatch worker's retention sweep;
those callers tag themselves via
:func:`osprey.stores.artifact_store.artifact_mutation_actor` and their events
are dropped here — the frames are *agent* activity.

**The caller is never blocked.** ``notify_agent_activity`` performs a blocking
HTTP POST, and ``ArtifactStore.delete_everything`` fires the delete listener once per
removed entry — a "clear the gallery" call would otherwise stall for one POST
per artifact, on whatever thread (or event loop) invoked it. The listener
callbacks therefore only enqueue onto a :class:`queue.Queue`; a single daemon
worker thread drains the queue and does the POSTs.

**Bookkeeping never emits.** An ``execute`` run auto-saves a notebook artifact
and a ``code_output`` record alongside whatever the code deliberately produced.
Emitting for those would turn one agent action into three strip entries, so
they are filtered out here; the run's own channel emit is that action's signal.
Deliberate saves — figures, ``save_artifact()``, the ``create_*`` tools,
``archiver_read`` / ``phoebus_snapshot`` persists — emit one frame per entry.
Accepted edge: a notebook the agent saves ON PURPOSE from inside an ``execute``
run is indistinguishable from the auto-save (same type, same ``tool_source``)
and is silently dropped. A missed frame for a rare case beats three frames for
every run; the run's own emit still shows the activity.

Headless dispatch agents register these listeners too. Their web terminal does
not exist, so each notify fails and is swallowed by ``notify_agent_activity``
(bounded at roughly a second, and only ever on the worker thread).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

from osprey.mcp_server.http import notify_agent_activity
from osprey.stores.artifact_store import current_artifact_mutation_actor

if TYPE_CHECKING:
    from osprey.stores.artifact_store import ArtifactEntry

logger = logging.getLogger("osprey.mcp_server.artifact_activity")

#: Activity kind for the ``/api/agent-activity`` route. Unknown kinds are
#: rejected server-side with a 422 that ``notify_agent_activity`` swallows, so
#: this string is asserted in the tests rather than trusted by eye.
_KIND = "artifact"

#: Tool names reported for the two mutation types.
_SAVE_TOOL = "artifact_save"
_DELETE_TOOL = "artifact_delete"

#: ``tool_source`` values whose notebook artifacts are auto-saved bookkeeping.
_NOTEBOOK_BOOKKEEPING_SOURCES = frozenset({"execute", "execute_file"})

#: Category of the per-execution result record the python executor writes.
_BOOKKEEPING_CATEGORY = "code_output"

#: Backlog bound. When the web terminal is unreachable each notify costs up to
#: a second, so a bulk delete can outrun the worker; the frames are
#: ephemeral UI signals (the browser-side history ring holds 50) and dropping
#: the overflow is strictly better than growing a queue nobody will read.
_MAX_PENDING = 256

_pending: queue.Queue[tuple[str, ArtifactEntry]] = queue.Queue(maxsize=_MAX_PENDING)
_state_lock = threading.Lock()
_worker: threading.Thread | None = None
_registered = False


def _is_bookkeeping(entry: ArtifactEntry) -> bool:
    """Return True when *entry* is a by-product of a run rather than an action."""
    if entry.category == _BOOKKEEPING_CATEGORY:
        return True
    return entry.artifact_type == "notebook" and entry.tool_source in _NOTEBOOK_BOOKKEEPING_SOURCES


def _detail_for(entry: ArtifactEntry) -> str:
    """Human-readable subject for the activity frame."""
    title = entry.title or entry.id
    return f"{title} ({entry.tool_source})" if entry.tool_source else title


def _drain_pending() -> None:
    """Worker loop: POST one activity frame per queued store event."""
    while True:
        tool, entry = _pending.get()
        try:
            notify_agent_activity(tool=tool, kind=_KIND, detail=_detail_for(entry))
        except Exception:
            # notify_agent_activity swallows its own failures; this guard only
            # keeps an unforeseen error from killing the worker for the process.
            logger.debug("artifact activity notify failed", exc_info=True)
        finally:
            _pending.task_done()


def _enqueue(tool: str, entry: ArtifactEntry) -> None:
    # Frames are *agent* activity. A gallery click or a retention sweep fires
    # the same store listener; those callers tag themselves via
    # artifact_mutation_actor and must not be reported as agent actions. The
    # actor is only valid here, on the mutating caller's own context — the
    # worker thread that POSTs later would always see the default.
    if current_artifact_mutation_actor() != "agent":
        return
    if _is_bookkeeping(entry):
        return
    try:
        _pending.put_nowait((tool, entry))
    except queue.Full:
        # Never wait for room: the caller is a store mutation, not a reporter.
        logger.debug("artifact activity backlog full — dropping %s frame", tool)


def _on_artifact_saved(entry: ArtifactEntry) -> None:
    """Store save listener — enqueue only, never POST on the caller's thread."""
    _enqueue(_SAVE_TOOL, entry)


def _on_artifact_deleted(entry: ArtifactEntry) -> None:
    """Store delete listener — fires once per entry, bulk deletes included."""
    _enqueue(_DELETE_TOOL, entry)


def register_artifact_activity_listeners() -> None:
    """Subscribe to artifact saves and deletes, at most once per process.

    ``ArtifactStore.register_listener`` appends unconditionally and the store
    listener lists live on the class, so a second call would double every
    frame. MCP server startup runs repeatedly in-process under test, hence the
    flag guard rather than a bare register.
    """
    global _worker, _registered

    from osprey.stores.artifact_store import (
        register_artifact_delete_listener,
        register_artifact_listener,
    )

    with _state_lock:
        if _registered:
            return
        register_artifact_listener(_on_artifact_saved)
        register_artifact_delete_listener(_on_artifact_deleted)
        _registered = True
        if _worker is None:
            _worker = threading.Thread(target=_drain_pending, name="artifact-activity", daemon=True)
            _worker.start()


def unregister_artifact_activity_listeners() -> None:
    """Unsubscribe from the store. No-op when not registered.

    The worker thread is left running: it is a daemon parked on an empty queue,
    and keeping it lets a later re-registration reuse it.
    """
    global _registered

    from osprey.stores.artifact_store import (
        unregister_artifact_delete_listener,
        unregister_artifact_listener,
    )

    with _state_lock:
        if not _registered:
            return
        unregister_artifact_listener(_on_artifact_saved)
        unregister_artifact_delete_listener(_on_artifact_deleted)
        _registered = False
