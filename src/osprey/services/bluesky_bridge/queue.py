"""The bridge's queue surface: REST + SSE over the queueserver-backed queue.

Transport shape mirrors `draft.py`: an `APIRouter` of REST routes plus a
`GET /queue/events` SSE stream with
hello/heartbeat/disconnect-on-overflow semantics. Unlike the draft, the state
itself lives in the RE manager (and the Redis it persists to) — this module
owns NO queue state beyond `_last_summary`, the last-status cache the SSE
poller diffs against. Everything the routes report is fetched through the
stateless :class:`~.queue_backend.QueueBackend` at request time.

**Arming policy.** Nothing here moves hardware directly, but two
operations arm hardware motion and are gated on the launch token
(`security.verify_launch_token`):

- ``POST /queue/start`` — always token-gated; the only way execution ever
  begins (qserver autostart/loop stays disabled on the manager, so the bridge
  originates every start).
- ``POST /queue/items`` while the manager is running, starting, or reporting
  autostart enabled (`_requires_arming` — the flag is observed, never assumed
  off) — an item added to a draining queue will execute without any further
  human action, so it needs the same token. Enqueuing onto an *idle* queue is
  ungated: the item just sits there until an armed start.
- ``POST /queue/stop`` with ``cancel: true`` — withdrawing a pending stop
  reverses a human's halt and lets the queue keep draining, so it carries the
  same token; the plain stop half stays ungated (halting is always allowed).

``POST /queue/abort`` — the emergency halt for a plan ALREADY moving hardware
— is ungated on the same principle as the plain stop, and more strictly: it
declares no token header at all, so there is nothing here for a later edit to
gate by accident. A halt that can be refused for a policy reason is a halt with
a failure mode. It is also deliberately not capability-gated: a deployment that
somehow has a plan running is a deployment that must be able to stop it,
whatever its capability record says.

The gate is race-free by construction, twice over. First, ``_arming_lock``
serializes the enqueue critical section {status-check + add + re-check} and
the start critical section {idle-check + interruption-gate + session-gate +
start}, so a bridge-side start can
never land between an unarmed caller's status check and its add (environment
open runs before the lock — it starts nothing and can take tens of seconds).
Second — defense in depth against anything that starts the queue outside this
process — an unarmed add re-checks the manager state *after* the add with
``status(reload=True)`` (the client caches status for ~0.5 s; a stale read
here would be a safety bug, not a cosmetic one) and, if the manager
transitioned, removes the just-added item and refuses (403/503, no item).

**Enqueue-from-pinned-draft.** ``POST /queue/items`` is the successor of the
direct launch-from-draft flow and keeps its guarantees unchanged: the body
pins a ``draft_revision``; `draft.check_launchable` snapshots and RESERVES
that revision once per process lifetime (concurrent duplicates 409); the
validation gate (`validation._validate_launchable_request` — session plans
need a current passing record) runs before anything reaches the manager; any
failure releases the reservation so the revision stays enqueueable; success
consumes it via `draft.record_and_broadcast_launch`. The OSPREY run id is
threaded through the item's metadata (`queue_backend.RUN_ID_META_KEY`) so
start documents, live rows, and Tiled results all key back to it. One check is
new rather than inherited: a plan whose role-typed channel parameters name
something the worker's namespace lacks is refused at add time
(`_check_devices_exist`), because that mistake would otherwise only surface as
a failed run.

A deployment whose capability record says it cannot execute (mock connector,
unreadable config, no manager) refuses enqueue outright — a browse-only
deployment must never hold queue items it can never run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, NoReturn

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import document_plane, draft, runs
from .plan_fields import MOVABLE_ROLE, READABLE_ROLE, collect_channels
from .plan_types import PlanSpec
from .queue_backend import (
    PLAN_META_KEY,
    REASON_MANAGER_NOT_CONFIGURED,
    REASON_MANAGER_UNREACHABLE,
    AbortPauseTimeoutError,
    EnvironmentUnavailableError,
    ExecutionUnavailableError,
    NothingRunningError,
    QueueBackend,
    QueueBackendError,
    QueueRequestRejectedError,
    QueueUnavailableError,
    is_queue_active,
    run_id_of,
)
from .security import verify_launch_token
from .session_upload import (
    SessionPlanNotReadyError,
    check_session_plan_ready,
    check_session_plans_ready,
)
from .validation import _validate_launchable_request

logger = logging.getLogger("osprey.services.bluesky_bridge.queue")

router = APIRouter()

# Same SSE plumbing constants as `draft.py` (see that module for rationale).
_QUEUE_MAXSIZE = 64
_HEARTBEAT_INTERVAL_S = 15.0
_DISCONNECT = object()

# How often the SSE poller re-reads manager status while subscribers exist.
# 1 s matches the panel freshness target: a queue change is visible in ≤ 1 s.
_POLL_INTERVAL_S = 1.0

# The status-document keys the SSE poller diffs on and every route reports.
# Any queue mutation moves `plan_queue_uid`, any history append moves
# `plan_history_uid`, and state transitions move `manager_state` /
# `running_item_uid` — so a summary-equality check is a complete change
# detector without ever caching the item list itself.
_SUMMARY_KEYS = (
    "manager_state",
    "worker_environment_exists",
    "items_in_queue",
    "items_in_history",
    "running_item_uid",
    "plan_queue_uid",
    "plan_history_uid",
    "queue_stop_pending",
    # Observed, never assumed: the bridge keeps autostart disabled, so a True
    # here is an out-of-band arming panels and MCP tools should surface —
    # and the enqueue pre-check treats it as armed (see `_requires_arming`).
    "queue_autostart_enabled",
)

# ---------------------------------------------------------------------------
# Module state: SSE plumbing plus the one lock the arming policy needs.
# The backend itself is NOT held here — `app.py`'s `get_queue_backend()` is
# the process's single backend accessor (shared with the capability surface),
# fetched lazily per call to avoid a module-level import cycle with `app.py`.
# `_last_summary` is the ONLY queue-shaped thing this module ever caches —
# the diff baseline for SSE frames, never served to a client as truth.
# ---------------------------------------------------------------------------
_arming_lock = asyncio.Lock()
_subscribers: set[asyncio.Queue[Any]] = set()
_poller_task: asyncio.Task[None] | None = None
_change_event = asyncio.Event()
_last_summary: dict[str, Any] | None = None


def _get_backend() -> QueueBackend:
    """The process's single `QueueBackend`, via `app.py`'s shared accessor.

    Imported lazily because `app.py` imports this module at load time to
    mount the router; a module-level back-import would be a cycle.
    """
    from .app import get_queue_backend

    return get_queue_backend()


def _clear() -> None:
    """Reset all module state (test isolation only; mirrors `draft._clear`).

    Recreates the asyncio primitives rather than merely resetting them:
    a lock or event that saw contention binds to the event loop that awaited
    it, and the next test may run on a fresh loop. The backend singleton
    lives in `app.py` (`set_queue_backend(None)` resets it) — not here.
    """
    global _last_summary, _arming_lock, _change_event
    _stop_poller()
    _last_summary = None
    _subscribers.clear()
    _arming_lock = asyncio.Lock()
    _change_event = asyncio.Event()


async def shutdown() -> None:
    """Stop the SSE poller (lifespan shutdown). The backend is closed by its owner."""
    _stop_poller()


def _http_error(exc: QueueBackendError) -> HTTPException:
    """Map one typed backend failure to the HTTP refusal the wire contract promises.

    - `ExecutionUnavailableError` carries the capability record in the body so
      the caller sees the same reason/remediation the status surface
      publishes. Manager-side reasons are 503 (deploy the manager / bring it
      back and retry); connector/config reasons are 409 (the deployment
      itself cannot execute — retrying changes nothing).
    - `QueueUnavailableError` / `EnvironmentUnavailableError` /
      `AbortPauseTimeoutError` are 503 — genuinely retryable. The abort timeout
      belongs in that bucket rather than getting a status of its own: the
      operator's next move is to try again (or reach for out-of-band means),
      and every client already branches on `detail.code`, so a fourth status
      would only be one more thing to learn.
    - `QueueRequestRejectedError` / `NothingRunningError` are 409 — the manager
      answered and the request does not apply.
    """
    detail: dict[str, Any] = {"code": exc.reason, "detail": str(exc)}
    if isinstance(exc, ExecutionUnavailableError):
        detail["capability"] = exc.capability.to_dict()
        retryable = exc.reason in (REASON_MANAGER_UNREACHABLE, REASON_MANAGER_NOT_CONFIGURED)
        return HTTPException(status_code=503 if retryable else 409, detail=detail)
    if isinstance(
        exc, (QueueUnavailableError, EnvironmentUnavailableError, AbortPauseTimeoutError)
    ):
        return HTTPException(status_code=503, detail=detail)
    if isinstance(exc, (QueueRequestRejectedError, NothingRunningError)):
        return HTTPException(status_code=409, detail=detail)
    return HTTPException(status_code=500, detail=detail)


def _token_is_valid(header: str) -> bool:
    """Whether ``header`` carries the armed launch token (never raises)."""
    try:
        verify_launch_token(header)
    except HTTPException:
        return False
    return True


def _require_arming_token(header: str, action: str) -> None:
    """`verify_launch_token`, re-dressed in the queue surface's refusal shape.

    `verify_launch_token` raises a bare-string detail; every refusal this
    module puts on the wire is a dict whose ``code`` is the machine-readable
    branch key (panels read ``body.detail.code``, MCP tools
    ``body["detail"]["code"]``). The status code is passed through untouched —
    503 when no token is configured at all, 403 on a missing/wrong header —
    and ``action`` names the operation being refused so the sentence reads on
    its own. The enqueue path has its own builder (`_refuse_unarmed`), which
    additionally reports ``manager_state`` and any stranded item.
    """
    try:
        verify_launch_token(header)
    except HTTPException as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "code": "launch_token_required",
                "detail": f"{action} requires the launch token: {exc.detail}",
            },
        ) from exc


def _requires_arming(status: dict[str, Any]) -> bool:
    """Whether adding an item right now is an armed operation.

    True while a plan is under way (or about to be), and ALSO whenever the
    manager reports autostart enabled: the bridge never enables autostart —
    every start originates here — so seeing it on means something armed the
    manager out-of-band, and an item added to even an idle queue would drain
    without any further human action. Observing the flag instead of assuming
    it closes that silent-collapse mode (defense in depth, not a supported
    configuration).
    """
    return is_queue_active(status) or bool(status.get("queue_autostart_enabled"))


def _refuse_unarmed(
    header: str, status: dict[str, Any], *, stranded_uid: str | None = None
) -> NoReturn:
    """Raise the machine-readable enqueue-while-armed refusal (403 or 503).

    Delegates the status-code choice to `verify_launch_token` (503 when no
    token is configured at all, 403 on a missing/wrong header) and wraps its
    message so clients can branch on ``code`` without parsing prose.

    ``stranded_uid`` names an item this refusal tried and FAILED to withdraw
    (`_remove_item_best_effort` returned False): the refusal then carries
    ``item_left_behind: true`` + ``item_uid`` so the wire — not only a
    container log — witnesses that an unarmed item remains in an armed queue.
    """
    try:
        verify_launch_token(header)
    except HTTPException as exc:
        detail: dict[str, Any] = {
            "code": "launch_token_required",
            "detail": (
                "the queue is running, starting, or set to autostart, so adding an "
                f"item requires the launch token: {exc.detail}"
            ),
            "manager_state": status.get("manager_state"),
        }
        if stranded_uid is not None:
            detail["item_left_behind"] = True
            detail["item_uid"] = stranded_uid
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc
    raise RuntimeError("_refuse_unarmed called with a valid token")  # pragma: no cover


def _refuse_interrupted_item(item: dict[str, Any]) -> NoReturn:
    """Raise the 409 that stops an emergency-aborted plan from silently re-running.

    Upstream does not simply discard a plan the worker interrupted: it records
    the run in history AND pushes a copy back to the FRONT of the queue under a
    new ``item_uid``, keeping the ``result`` block (see
    `runs.is_requeued_after_interruption`). Nothing in that mechanism is
    configurable. So the queue an operator leaves behind after hitting the
    emergency Abort is a queue whose HEAD is the plan they just stopped — and
    the next armed start would put it straight back on the hardware, with no
    fresh human decision anywhere in the loop.

    Refusing the start is deliberately where this is closed, not the abort:
    the halt itself stays completely ungated (a halt with a failure mode is not
    a halt), and the arming action — which already requires the launch token —
    is the one that must not proceed on an ambiguous queue.

    There is exactly ONE way out, and the refusal says so: this gate is
    stateless and re-reads the queue on every start, so it refuses EVERY start
    while the requeued copy carries its ``result``. Removing that copy is the
    only thing that unblocks the queue. Re-running the plan is a SECOND step
    taken after the removal — re-stage it through the draft and enqueue it
    again — never an alternative to it.

    409, not 503: nothing is broken and retrying unchanged will not help — a
    human has to choose.
    """
    result = item.get("result")
    exit_status = result.get("exit_status") if isinstance(result, dict) else None
    uid = item.get("item_uid")
    name = item.get("name")
    raise HTTPException(
        status_code=409,
        detail={
            "code": "interrupted_item_in_queue",
            "detail": (
                f"the queue still holds {name!r}, a plan that already ran and ended "
                f"{exit_status!r}; the queue server puts an interrupted plan back at the "
                "front of the queue, so starting now would re-run it. Every start is "
                "refused while that copy is queued, so removing it with "
                f"DELETE /queue/items/{uid} is the only way on. To run the plan again, "
                "remove it first and then re-stage it through the draft and enqueue it "
                "afresh, to run it deliberately."
            ),
            "item_uid": uid,
            "plan": name,
            "exit_status": exit_status,
        },
    )


def _refuse_manager_not_idle(running_item: dict[str, Any]) -> NoReturn:
    """Raise the 409 that refuses a start while a plan is already in motion.

    Behaviour-preserving on its face: the RE manager accepts ``queue_start``
    only from an idle state, so a start issued while an item is running was
    already going to be refused — as an opaque ``queue_request_rejected``
    relayed from upstream. Answering it here instead costs nothing and buys a
    machine-readable code (and a round trip).

    The reason it EXISTS, though, is `_refuse_interrupted_item`'s residual
    race. That gate reads one snapshot and asks whether any item in it carries
    a ``result`` from an earlier interruption. A RUNNING item never does — it
    has not finished — so a snapshot holding one passes that check, and if the
    plan then aborts or fails in the gap before ``backend.start()``, upstream
    requeues the result-bearing copy to the FRONT and the start drains exactly
    the item the gate exists to stop.

    Refusing every snapshot that holds a running item closes that structurally
    rather than by narrowing the window: after this check, no snapshot that
    reaches ``start()`` has anything in flight that could become an interrupted
    requeue while the arming lock is held. The interruption check is then
    exhaustive over every snapshot that can reach a start.

    409, not 503: nothing is broken, and the queue is already doing the thing a
    start would ask for.
    """
    uid = running_item.get("item_uid")
    name = running_item.get("name")
    raise HTTPException(
        status_code=409,
        detail={
            "code": "manager_not_idle",
            "detail": (
                f"the queue is already running {name!r}, so there is nothing for a start "
                "to do — the queue moves on to the next item by itself when this one "
                "finishes. Poll GET /queue to follow it, or POST /queue/abort to halt "
                "the plan in motion."
            ),
            "item_uid": uid,
            "plan": name,
        },
    )


def _session_refusal(exc: SessionPlanNotReadyError) -> HTTPException:
    """The 409 a non-admissible session plan earns on any armed path.

    Same dict-detail convention as every other queue refusal (``code`` is the
    machine-readable branch key); ``plan`` names the offending item so a
    queue-start refusal is actionable without re-deriving which item blocked
    it. Caller-fixable (re-validate), so 409 — never 503. The error's native
    ``to_dict()`` keys (``reason``/``detail``/``plan``) ride along verbatim,
    with the same value repeated under the queue surface's canonical ``code``
    key so clients branch on one name everywhere.
    """
    return HTTPException(status_code=409, detail={**exc.to_dict(), "code": exc.reason})


# ---------------------------------------------------------------------------
# Add-time channel pre-check
# ---------------------------------------------------------------------------
# A plan resolves its channel parameters by string name against the worker's
# namespace, and it does so on the run's FIRST iteration — so a name the worker
# has no device for is not caught by any schema, and surfaces only as a failed
# run after an enqueue and a start. Checking the names against the worker's own
# `devices_allowed` before the add turns that into one legible refusal at the
# moment the name was chosen.
#
# WHICH strings in the params are channel names is the plan's own declaration,
# never a guess this module makes: an author annotates each channel field with
# the role the plan gives it (movable / readable), the loader records what the
# schema declared on `PlanSpec.roles`, and `plan_fields.collect_channels` reads
# the supplied names back out of one params dict for one role. Field names match
# EXACTLY — a field named ``setpoint`` supplies channels only from ``setpoint``
# — so nothing here has to know, or approximate, any plan's parameter shape.


def _referenced_channel_names(spec: PlanSpec[Any], plan_args: Any) -> tuple[set[str], set[str]]:
    """The movable and readable channel names ``plan_args`` supplies, as two sets.

    Both come from `plan_fields.collect_channels`, whose matching rule is the
    NEAREST enclosing field name: the enclosing name is rebound at every mapping
    level and carried unchanged across list levels, so a string is a channel
    name only when the field *immediately* above it declared a role. Nesting is
    found (`grid_scan` carries a movable channel as ``axes[].setpoint``) without
    this module knowing that shape.

    That "nearest" is a safety property, not a style choice. For a plan whose
    role-typed field holds objects — ``{"axes": [{"setpoint": "COR1", "mode":
    "fast"}]}`` — a sticky "anywhere under a role-typed field" rule would collect
    ``"fast"`` too and refuse a perfectly good enqueue over a mode string: a
    false refusal no agent can fix, since nothing it changes about the channel
    name makes ``"fast"`` a device. The rebinding rule makes that shape a MISS
    instead, and a miss leaves the worker as the enforcement point — which is
    the direction this check must always be wrong in.

    The two roles are kept apart rather than merged because they are not the
    same mistake: a movable name the worker lacks is the one the refusal is
    built around (it would have armed hardware against nothing), and it is the
    name the sentence leads with when both roles carry an unknown one.
    """
    return (
        set(collect_channels(spec.schema, plan_args, MOVABLE_ROLE)),
        set(collect_channels(spec.schema, plan_args, READABLE_ROLE)),
    )


# How many device names the refusal SENTENCE lists before summarizing the
# rest. A real facility worker builds hundreds, and the sentence is prose an
# agent and an operator read — the complete set is always on the wire
# structurally in ``available_devices``, so nothing is lost by capping the
# readable copy.
_SENTENCE_DEVICE_LIMIT = 20


def _available_devices_phrase(available: set[str]) -> str:
    """The refusal sentence's device list, capped at `_SENTENCE_DEVICE_LIMIT`."""
    names = sorted(available)
    if len(names) <= _SENTENCE_DEVICE_LIMIT:
        return f"{names}"
    shown = names[:_SENTENCE_DEVICE_LIMIT]
    return f"{shown} (+{len(names) - len(shown)} more; full list in available_devices)"


def _refuse_unknown_devices(
    plan_name: str,
    unknown_movable: set[str],
    unknown_readable: set[str],
    available: set[str],
    *,
    session_tier: bool,
) -> NoReturn:
    """Raise the 400 for an enqueue naming a device this worker did not build.

    The sentence is the worker's own (`qserver_startup.py`'s plan wrapper and
    `session_upload.py`'s raise it when the name reaches the RunEngine), down
    to which noun it opens with: the session-plan wrapper says "session plan
    {name}", the catalog wrapper says "plan {name}", and this refusal matches
    whichever one would have raised. Whichever layer catches the mistake, the
    operator and the agent read the same sentence about the same event.

    Three deliberate differences from the run-time version, all additive:
    ``devices`` carries EVERY unknown name across both roles (the run-time raise
    can only ever report the first one it tripped over), the in-sentence device
    list is capped — `available_devices` carries it whole — and the one name the
    sentence quotes is drawn from the movable names when there are any, because
    that is the reading under which a start would have driven hardware toward a
    channel the worker never built.

    400, not 409: the request itself names something that does not exist, and
    the fix is in the caller's hands — pick a name `GET /devices` lists.
    """
    unknown = unknown_movable | unknown_readable
    first = sorted(unknown_movable or unknown_readable)[0]
    label = "session plan" if session_tier else "plan"
    raise HTTPException(
        status_code=400,
        detail={
            "code": "unknown_device",
            "detail": (
                f"{label} {plan_name!r} referenced device {first!r}, which this worker "
                f"did not build; available devices: {_available_devices_phrase(available)}"
            ),
            "plan": plan_name,
            "devices": sorted(unknown),
            "available_devices": sorted(available),
        },
    )


async def _check_devices_exist(backend: QueueBackend, plan_name: str, plan_args: Any) -> None:
    """Refuse the enqueue if it names a device the worker's namespace lacks.

    Deliberately narrow, and fail-open at every step where the answer is not
    certain: a plan whose schema declares no channel roles at all, params
    carrying nothing under a role-typed field, a manager that will not answer,
    and a manager reporting no devices at all all pass straight through. The
    worker remains the enforcement point — this only moves the *legible* cases
    earlier, and an availability failure here must never cost an operator an
    enqueue that would have run.

    **Both roles are checked.** A movable name the worker lacks is the refusal's
    reason for existing (SC-7): the plan would have been armed to drive a
    channel that does not exist. A readable name the worker lacks is refused on
    exactly the same certainty — the run's very first read would raise on it —
    and refusing it is what this check already did before roles were declared,
    so nothing is loosened here and nothing is tightened. The two sets stay
    apart because the refusal sentence leads with a movable name when there is
    one; see `_refuse_unknown_devices`.
    """
    from .plan_loader import get_facility_plans

    try:
        # `get_facility_plans()` re-scans the session-plan directory on every
        # call, so it runs off the loop (`draft._resolve_plan_schema` states the
        # house rule). Wrapped because a plan-directory read failure must skip
        # the pre-check, never 500 an enqueue that would otherwise have run.
        facility_plans = await asyncio.to_thread(get_facility_plans)
    except Exception as exc:
        logger.warning("device pre-check skipped; the plan registry is unreadable: %s", exc)
        return

    spec = facility_plans.plans.get(plan_name)
    if spec is None or not spec.roles:
        # No declaration, nothing to say: which params are channel names is the
        # plan's to state, and a plan that states nothing is one this check has
        # no honest opinion about. (`roles` is what the loader recorded off the
        # schema, so this costs no re-walk.)
        return

    movable, readable = _referenced_channel_names(spec, plan_args)
    if not movable and not readable:
        return

    try:
        reply = await backend.devices_allowed()
    except Exception as exc:
        # Availability over the pre-check: the manager being unreachable is
        # already reported by the add itself, and a convenience gate must not
        # be the thing that refuses an enqueue.
        logger.warning("device pre-check skipped; the worker's device list is unreadable: %s", exc)
        return

    allowed = reply.get("devices_allowed")
    if not isinstance(allowed, dict) or not allowed:
        # A worker reporting no devices at all is a worker whose environment is
        # not up yet, not a worker on which every name is wrong.
        return

    built = set(allowed)
    unknown_movable = movable - built
    unknown_readable = readable - built
    if unknown_movable or unknown_readable:
        session_tier = spec.provenance in ("session", "unreviewed")
        _refuse_unknown_devices(
            plan_name, unknown_movable, unknown_readable, built, session_tier=session_tier
        )


# ---------------------------------------------------------------------------
# Status summaries + SSE plumbing
# ---------------------------------------------------------------------------


def _status_summary(status: dict[str, Any]) -> dict[str, Any]:
    """The bounded, diffable projection of a manager status document."""
    summary: dict[str, Any] = {"available": True}
    for key in _SUMMARY_KEYS:
        summary[key] = status.get(key)
    return summary


def _unavailable_summary(reason: str) -> dict[str, Any]:
    """The summary published when the manager (or backend) cannot be read."""
    summary: dict[str, Any] = {"available": False, "reason": reason}
    for key in _SUMMARY_KEYS:
        summary[key] = None
    return summary


async def _current_summary() -> dict[str, Any]:
    try:
        # reload=True: the poller exists to make ≤1 s freshness true, and the
        # client's ~0.5 s status cache would quietly halve that.
        return _status_summary(await _get_backend().status(reload=True))
    except QueueBackendError as exc:
        return _unavailable_summary(exc.reason)


def _with_progress(running_item: Any) -> Any:
    """Attach the running item's progress record, when known.

    ``progress`` keys on the OSPREY run id the enqueue path threaded into the
    item's metadata; an item without one (enqueued out-of-band) or with no
    recorded rows/denominator passes through untouched — never a fabricated
    ``0.0``.
    """
    if not isinstance(running_item, dict) or not running_item:
        return None
    run_id = run_id_of(running_item)
    if run_id is None:
        return running_item
    progress = document_plane.progress(run_id)
    if progress is None:
        return running_item
    return {**running_item, "progress": progress}


def _public_item(item: Any) -> Any:
    """One queue item as the bridge relays it: the plan-identity stamp removed.

    The enqueue path stamps the plan's own name and kwargs into the item's
    metadata (`queue_backend.PLAN_META_KEY`) so that a run's figure can be
    rendered as the plan it was, from the run's start document alone. On a
    queue READ that stamp is pure duplication — the item already carries its
    ``name`` and ``kwargs`` at top level — and the queue is the one surface
    panels both poll and stream continuously, so it is dropped here rather
    than doubling every item's params on every frame.

    Only that key goes. ``RUN_ID_META_KEY`` stays, because it is what joins a
    queue row to its run (`itemRunId()` in `queue-client.js` reads it), and
    any other metadata is somebody else's — an item enqueued out of band
    passes through untouched.
    """
    if not isinstance(item, dict):
        return item
    meta = item.get("meta")
    if not isinstance(meta, dict) or PLAN_META_KEY not in meta:
        return item
    return {**item, "meta": {key: value for key, value in meta.items() if key != PLAN_META_KEY}}


async def _frame_from(summary: dict[str, Any], frame_type: str) -> dict[str, Any]:
    """A full snapshot frame: summary plus the current items, freshly fetched."""
    items: list[Any] = []
    running_item: Any = None
    if summary.get("available"):
        try:
            queue_state = await _get_backend().items()
            items = [_public_item(item) for item in (queue_state.get("items") or [])]
            running_item = _public_item(_with_progress(queue_state.get("running_item")))
        except QueueBackendError as exc:  # pragma: no cover - narrow race
            logger.debug("queue item fetch failed after a good status read: %s", exc)
    return {"type": frame_type, "status": summary, "items": items, "running_item": running_item}


def _format_sse(frame: dict[str, Any]) -> str:
    return f"data: {json.dumps(frame)}\n\n"


def _broadcast(frame: dict[str, Any]) -> None:
    """Push *frame* to every subscriber; disconnect-on-overflow like `draft.py`."""
    for subscriber in list(_subscribers):
        try:
            subscriber.put_nowait(frame)
        except asyncio.QueueFull:
            _subscribers.discard(subscriber)
            while True:
                try:
                    subscriber.get_nowait()
                except asyncio.QueueEmpty:
                    break
            try:
                subscriber.put_nowait(_DISCONNECT)
            except asyncio.QueueFull:  # pragma: no cover - defensive only
                pass


def _notify_change() -> None:
    """Wake the SSE poller early: a bridge-side mutation just changed the queue."""
    _change_event.set()


async def _poll_once() -> None:
    global _last_summary
    summary = await _current_summary()
    if summary == _last_summary:
        return
    frame = await _frame_from(summary, "queue")
    _last_summary = summary
    _broadcast(frame)


async def _poll_loop() -> None:
    while True:
        try:
            await asyncio.wait_for(_change_event.wait(), timeout=_POLL_INTERVAL_S)
        except TimeoutError:
            pass
        _change_event.clear()
        try:
            await _poll_once()
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception:  # pragma: no cover - keep the poller alive
            logger.exception("queue SSE poller iteration failed")


def _ensure_poller() -> None:
    global _poller_task
    if _poller_task is None or _poller_task.done():
        _poller_task = asyncio.get_running_loop().create_task(_poll_loop())


def _stop_poller() -> None:
    global _poller_task
    if _poller_task is not None:
        _poller_task.cancel()
        _poller_task = None


async def _subscribe() -> tuple[asyncio.Queue[Any], dict[str, Any]]:
    """Register a new SSE subscriber and build its hello frame.

    Resets the diff cache so the poller re-broadcasts a full snapshot on its
    next tick — the cheap way to close the register/hello race window: any
    change that lands while the hello is being fetched reaches the new
    subscriber (and everyone else, as one duplicate full-snapshot frame,
    which full-snapshot clients absorb idempotently) within one poll tick.
    """
    global _last_summary
    subscriber: asyncio.Queue[Any] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    _subscribers.add(subscriber)
    _last_summary = None
    _ensure_poller()
    hello = await _frame_from(await _current_summary(), "hello")
    return subscriber, hello


async def _unsubscribe(subscriber: asyncio.Queue[Any]) -> None:
    _subscribers.discard(subscriber)
    if not _subscribers:
        _stop_poller()


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class QueueAddRequest(BaseModel):
    """Body for `POST /queue/items`: enqueue the shared draft at a pinned revision.

    The enqueued ``plan_name``/``plan_args`` come exclusively from the
    server-side draft snapshot taken at exactly ``draft_revision`` — never
    from this body (same contract as the launch-from-draft flow it replaces).
    """

    draft_revision: int


class QueueMoveRequest(BaseModel):
    """Body for `POST /queue/items/{uid}/move`: exactly one destination.

    Per queueserver: ``pos_dest`` is ``"front"``, ``"back"``, or an index;
    ``before_uid``/``after_uid`` place relative to another item.
    """

    pos_dest: int | str | None = None
    before_uid: str | None = None
    after_uid: str | None = None


class QueueStopRequest(BaseModel):
    """Body for `POST /queue/stop`. ``cancel: true`` withdraws a pending stop."""

    cancel: bool = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/queue")
async def get_queue() -> dict[str, Any]:
    """The queue as the manager holds it: pending items, running item, status summary.

    Items are relayed through `_public_item`, which drops the enqueue path's
    plan-identity stamp — it duplicates the item's own name and kwargs, and
    the run id panels join on rides through untouched.
    """
    backend = _get_backend()
    try:
        status = await backend.status()
        queue_state = await backend.items()
    except QueueBackendError as exc:
        raise _http_error(exc) from exc
    return {
        "status": _status_summary(status),
        "items": [_public_item(item) for item in (queue_state.get("items") or [])],
        "running_item": _public_item(_with_progress(queue_state.get("running_item"))),
    }


@router.post("/queue/items")
async def add_queue_item(
    body: QueueAddRequest, x_launch_token: str = Header(default="")
) -> dict[str, Any]:
    """Enqueue the shared draft at a pinned revision.

    Sequence, and why each step sits where it does (module docstring has the
    full arming rationale):

    1. Capability refusal first: a deployment that cannot execute plans never
       holds queue items, and never consumes the caller's draft revision.
    2. `draft.check_launchable` reserves the pinned revision (409 on a stale
       or already-consumed revision — same dict-detail shape as the draft
       routes). Every later failure releases the reservation, so a refused
       enqueue never burns the revision.
    3. The validation gate runs unchanged (session plans need a current
       passing record; sync file I/O, so it runs in a thread) before anything
       reaches the manager.
    4. `_check_devices_exist` refuses (400) a plan whose role-typed channel
       parameters name something the worker's namespace lacks — the one class
       of mistake the schema cannot catch and the run would otherwise report on
       its first iteration, after a start.
    5. Under ``_arming_lock``: status pre-check (active + unarmed → refuse
       before adding), the session-plan admissibility re-check
       (`session_upload.check_session_plan_ready` — 409 for a session plan
       with no current passing record in the live namespace), the add itself,
       then — for unarmed callers only — the post-add ``status(reload=True)``
       re-check that removes the item and refuses if the manager transitioned
       underneath the add.
    6. Only then is the reservation consumed and the draft's ``launched`` frame
       broadcast; the SSE poller is nudged so panels see the new item
       immediately. Nothing here records a progress denominator: a run declares
       its own point count in its start document, which `document_plane` reads
       when the run begins. A queued item that has not started therefore has no
       denominator — an accepted gap, and an honest one, since the number a plan
       will actually produce is the plan's to state and not this route's to
       infer from the enqueued parameters.
    """
    backend = _get_backend()

    try:
        capability = await backend.capability()
    except QueueBackendError as exc:
        raise _http_error(exc) from exc
    if not capability.can_execute:
        raise _http_error(ExecutionUnavailableError(capability))

    armed = _token_is_valid(x_launch_token)

    checked = await draft.check_launchable(body.draft_revision)
    if isinstance(checked, draft.LaunchRejected):
        raise HTTPException(
            status_code=409,
            detail={"code": checked.code, "detail": checked.detail, "revision": checked.revision},
        )

    run_id = uuid.uuid4().hex
    item = {
        "item_type": "plan",
        "name": checked.plan_name,
        "kwargs": dict(checked.plan_args),
    }

    enqueued = False
    result: dict[str, Any] = {}
    try:
        # Unchanged validation gate (session-plan passing record); the
        # snapshot is request-shaped (`plan_name` attribute), which is all
        # `_validate_launchable_request` reads.
        await asyncio.to_thread(_validate_launchable_request, checked)

        # Channel names resolve to devices in the worker, on the run's first
        # iteration, so an unresolvable one is checked here or not at all until
        # a failed run.
        # Outside the arming lock: it is a manager read that starts nothing.
        await _check_devices_exist(backend, checked.plan_name, checked.plan_args)

        try:
            async with _arming_lock:
                status = await backend.status(reload=True)
                if not armed and _requires_arming(status):
                    _refuse_unarmed(x_launch_token, status)

                # Session-plan re-check, inside the lock so
                # the admissibility answer and the add are atomic relative to
                # starts. A no-op for catalog plans; for a session plan it
                # requires a current passing record AND the validated bytes in
                # the live worker namespace — refusing, never repairing.
                try:
                    await check_session_plan_ready(checked.plan_name)
                except SessionPlanNotReadyError as exc:
                    raise _session_refusal(exc) from exc

                result = await backend.add_item(item, run_id=run_id)

                if not armed:
                    added_uid = _added_item_uid(result)
                    try:
                        recheck = await backend.status(reload=True)
                    except QueueBackendError as exc:
                        # Fail closed: with the manager state unknowable, an
                        # unarmed item must not stay behind — and if the
                        # withdrawal itself fails, the wire says so.
                        removed = await _remove_item_best_effort(backend, added_uid)
                        error = _http_error(exc)
                        if not removed:
                            error.detail["item_left_behind"] = True
                            error.detail["item_uid"] = added_uid
                        raise error from exc
                    if _requires_arming(recheck):
                        removed = await _remove_item_best_effort(backend, added_uid)
                        _refuse_unarmed(
                            x_launch_token,
                            recheck,
                            stranded_uid=None if removed else added_uid,
                        )
        except QueueBackendError as exc:
            raise _http_error(exc) from exc
        enqueued = True
    finally:
        if not enqueued:
            await draft.release_launch(checked.revision)

    # Post-enqueue bookkeeping. The item is already in the queue here, so a
    # failure in this block must not strand the revision in the in-flight
    # reservation set forever — release it, log loudly, and surface the error.
    try:
        await draft.record_and_broadcast_launch(run_id=run_id, revision=checked.revision)
    except Exception:
        logger.exception("post-enqueue bookkeeping failed for run %s", run_id)
        await draft.release_launch(checked.revision)
        raise
    _notify_change()

    returned_item = result.get("item")
    return {
        "run_id": run_id,
        "revision": checked.revision,
        "item": returned_item if isinstance(returned_item, dict) else None,
    }


def _added_item_uid(add_result: dict[str, Any]) -> str | None:
    """The server-assigned uid off an ``item_add`` response, if it carried one."""
    item = add_result.get("item")
    uid = item.get("item_uid") if isinstance(item, dict) else None
    return uid if isinstance(uid, str) and uid else None


async def _remove_item_best_effort(backend: QueueBackend, uid: str | None) -> bool:
    """Undo an unarmed add the re-check refused. Never raises.

    Returns True only when the item is verifiably gone. False means an
    unarmed item may remain in an armed queue — the lock guarantees no
    *bridge*-side start raced this add, so reaching that state implies an
    out-of-band actor. The refusal still stands; the caller puts
    ``item_left_behind`` on the wire so the stranded item has a witness
    beyond this log line.
    """
    if not uid:
        logger.error(
            "cannot undo an unarmed enqueue: the manager's add response carried no item_uid"
        )
        return False
    try:
        await backend.remove(uid)
    except QueueBackendError as exc:
        logger.error("failed to remove item %s after the arming re-check refused it: %s", uid, exc)
        return False
    return True


@router.post("/queue/items/{uid}/move")
async def move_queue_item(uid: str, body: QueueMoveRequest) -> dict[str, Any]:
    """Move one queued item. Ungated: reordering pending work arms nothing."""
    backend = _get_backend()
    try:
        result = await backend.reorder(
            uid, pos_dest=body.pos_dest, before_uid=body.before_uid, after_uid=body.after_uid
        )
    except QueueBackendError as exc:
        raise _http_error(exc) from exc
    _notify_change()
    moved_item = result.get("item")
    return {"moved": True, "item": moved_item if isinstance(moved_item, dict) else None}


@router.delete("/queue/items/{uid}")
async def remove_queue_item(uid: str) -> dict[str, Any]:
    """Drop one queued item. Ungated: removing pending work arms nothing."""
    backend = _get_backend()
    try:
        result = await backend.remove(uid)
    except QueueBackendError as exc:
        raise _http_error(exc) from exc
    _notify_change()
    removed_item = result.get("item")
    return {"removed": True, "item": removed_item if isinstance(removed_item, dict) else None}


@router.post("/queue/start")
async def start_queue(x_launch_token: str = Header(default="")) -> dict[str, Any]:
    """Start draining the queue. Token-gated — this is the arming action.

    `verify_launch_token` runs before ANY state is touched (503 unarmed, 403
    mismatch). `ensure_environment_for_execute` runs next, OUTSIDE the lock:
    it refuses browse-only deployments outright (capability record on the
    wire) and brings the worker environment up when this deployment should be
    able to execute but the environment is closed — which can take tens of
    seconds on a cold worker, and opening an environment starts nothing, so
    holding the arming lock across it would only park every concurrent
    enqueue for no safety gain. The critical section — shared with the
    enqueue path so an unarmed add can never interleave between its status
    check and this start — is {idle check + interrupted-item check +
    session-plan re-check + start}, in that order:

    - a queue with a plan already in motion is refused outright (409
      ``manager_not_idle`` — see `_refuse_manager_not_idle`), which is also
      what makes the next check exhaustive;
    - every item the start would drain is checked for a ``result`` left by an
      earlier interruption (409 ``interrupted_item_in_queue`` — see
      `_refuse_interrupted_item`; this is what stops an emergency-aborted plan
      re-running by itself);
    - and re-checked for admissibility (`check_session_plans_ready`; one stale
      session plan refuses the whole start, all-or-nothing).

    Only then does the queue start. If the environment closes again in the gap,
    the manager refuses the start (409/503) — fail-closed either way.
    """
    _require_arming_token(x_launch_token, "starting the queue")
    backend = _get_backend()
    try:
        await backend.ensure_environment_for_execute()
        async with _arming_lock:
            queue_state = await backend.items()

            # Nothing may already be in motion (see `_refuse_manager_not_idle`).
            # First, because everything below reasons about a queue that is not
            # moving: this is what makes the interruption check exhaustive
            # rather than merely likely.
            running_item = queue_state.get("running_item")
            if isinstance(running_item, dict) and running_item:
                _refuse_manager_not_idle(running_item)

            # An interrupted plan is still sitting in this queue (see
            # `_refuse_interrupted_item`). Checked before the session gate and
            # before the start, inside the same critical section, so the answer
            # cannot go stale between the look and the start.
            drained = list(queue_state.get("items") or [])
            for item in drained:
                if runs.is_requeued_after_interruption(item):
                    _refuse_interrupted_item(item)

            # Session-plan re-check over EVERY item this start would drain — a
            # queue holding one stale session plan does not start. One
            # allowed-plan round trip for the whole queue. The pending items ARE
            # the whole drain set here: a running item would already have been
            # refused above.
            names = [
                item["name"]
                for item in drained
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]
            try:
                await check_session_plans_ready(names)
            except SessionPlanNotReadyError as exc:
                raise _session_refusal(exc) from exc

            result = await backend.start()
    except QueueBackendError as exc:
        raise _http_error(exc) from exc
    _notify_change()
    return {"started": True, "msg": str(result.get("msg") or "")}


@router.post("/queue/stop")
async def stop_queue(
    body: QueueStopRequest | None = None, x_launch_token: str = Header(default="")
) -> dict[str, Any]:
    """Stop the queue after the running item finishes.

    Deliberately asymmetric. A plain stop (``cancel`` false or absent) is
    ungated — halting is always allowed. ``cancel: true`` WITHDRAWS a pending
    stop and lets the queue keep draining toward hardware: that reverses a
    human's halt, which on a shared loopback port (see `security.py`'s threat
    model) is an arming action, so it requires the launch token exactly like
    a start (503 unarmed / 403 mismatch, before anything is touched).

    This route does NOT touch the item already in motion — that is
    `POST /queue/abort`.
    """
    cancel = bool(body.cancel) if body is not None else False
    if cancel:
        _require_arming_token(x_launch_token, "withdrawing a pending stop")
    backend = _get_backend()
    try:
        result = await backend.stop(cancel=cancel)
    except QueueBackendError as exc:
        raise _http_error(exc) from exc
    _notify_change()
    return {"stop_pending": not cancel, "msg": str(result.get("msg") or "")}


@router.post("/queue/abort")
async def abort_running_plan() -> dict[str, Any]:
    """Abort the plan running RIGHT NOW. Completely ungated.

    The emergency halt, and the only OSPREY surface that stops a plan already
    moving hardware (`POST /queue/stop` halts the queue only after the running
    item finishes). No launch token, no capability check, no arming lock: this
    route declares no token header at all, so it cannot be gated by a later
    edit that adds one "for consistency", and it takes no body, so there is no
    parameter whose validation could refuse a halt.

    The composition — immediate pause, bounded poll for the paused state, then
    abort — lives in `QueueBackend.abort`, which documents why each step is
    where it is. Every outcome is machine-readable and honest about what did
    and did not happen:

    - 200 ``{aborted: true, abort_pending, paused_first, manager_state, msg}``.
      ``abort_pending`` true means the manager accepted the abort and is still
      unwinding, not that it failed.
    - 409 ``nothing_running`` — no plan was under way (or it ended while the
      abort was being prepared). Nothing was stopped because there was nothing
      to stop.
    - 503 ``abort_pause_timeout`` — the Run Engine never paused, so NOTHING was
      aborted and the plan may still be running. Never dressed up as a halt.
    - 503 ``manager_unreachable`` — the manager did not answer at all.
    - 409 ``queue_request_rejected`` — the manager refused the abort itself.

    WHERE THE ABORTED PLAN ENDS UP, which is not where the wording above might
    suggest: the queue is left stopped, but the ITEM is not discarded. Upstream
    records the run in history with ``exit_status: "aborted"`` AND pushes a copy
    of the item back to the FRONT of the queue under a new ``item_uid``, so an
    operator can decide whether to run it again. That is not configurable. Two
    consequences, both handled rather than hidden:

    - `GET /runs` reports that run as ``stopped``, not ``pending``, because
      `runs._queue_status` projects a queued item's own ``result`` through the
      history mapping — the requeued copy would otherwise shadow the history
      entry and publish a just-aborted plan as work still to come.
    - `POST /queue/start` REFUSES while such an item is in the queue (409
      ``interrupted_item_in_queue``), so the plan a human emergency-stopped
      cannot go back on the hardware without a fresh, explicit decision. The
      abort itself stays ungated; the arming action is where the ambiguity is
      resolved.

    The SSE poller is nudged as usual; an abort moves ``manager_state``,
    ``running_item_uid`` and the history keys, so the change reaches panels
    within one tick either way.
    """
    backend = _get_backend()
    try:
        result = await backend.abort()
    except QueueBackendError as exc:
        # Even a refusal changed nothing but may have moved the manager (a
        # pause can land before the plan ends); let the poller re-read.
        _notify_change()
        raise _http_error(exc) from exc
    _notify_change()
    return result


@router.get("/queue/events")
async def queue_events() -> StreamingResponse:
    """SSE stream of queue changes: a hello snapshot on connect, then live frames.

    Frame types: ``hello`` (full snapshot, once on connect) and ``queue``
    (full snapshot on every detected change — the poller diffs the status
    summary, so every add/move/remove/start/stop/history append yields one
    frame). Bridge-side mutations wake the poller immediately; out-of-band
    changes (the manager draining the queue) surface within one poll tick
    (~1 s). Heartbeats and the disconnect-on-overflow slow-consumer policy
    are exactly `draft.py`'s.
    """
    subscriber, hello = await _subscribe()

    async def _generate():
        try:
            yield _format_sse(hello)
            while True:
                try:
                    frame = await asyncio.wait_for(subscriber.get(), timeout=_HEARTBEAT_INTERVAL_S)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if frame is _DISCONNECT:
                    break
                yield _format_sse(frame)
        finally:
            await _unsubscribe(subscriber)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
