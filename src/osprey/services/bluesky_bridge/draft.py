"""Server-held shared plan draft: the agent's and the human's collaborative
scratch pad for composing one Bluesky scan plan before launch (PROPOSAL.md
"Bridge draft module").

One in-process singleton draft (`{plan_name, plan_args, updated_by,
updated_at}`) plus a process-lifetime monotonic revision counter, guarded by
one `asyncio.Lock` so read-modify-broadcast is atomic and revision order
always equals SSE frame order. This module is intentionally its own
`APIRouter` (mounted into `app.py` via `include_router`) rather than more
inline `@app.<verb>` routes — the draft's state, lock, and broadcaster are
self-contained and don't need anything else `app.py` owns.

Per-field value validation (`PATCH /draft`'s `plan_args_patch`) uses
``TypeAdapter(Annotated[field.annotation, field])`` — the field's own
``FieldInfo`` instance as the ``Annotated`` metadata directly, NOT unpacked
with a leading ``*``. The starred form (``Annotated[field.annotation,
*field.metadata]``) raises ``TypeError`` for any field with no
``Field()``-level constraints (its ``metadata`` list is empty, and
``Annotated`` requires at least one metadata argument) — verified against
pydantic 2.12.5. Passing the ``FieldInfo`` itself works uniformly whether or
not the field carries constraints, which is why every per-field validation
call in this module goes through :func:`_validate_field` rather than
resolving a fresh adapter ad hoc.

Every coerced value is additionally passed back through
``TypeAdapter.dump_python(..., mode="json")`` before being stored. Without
this, a field typed as a nested pydantic model (e.g. `grid_scan`'s
``axes: list[GridAxis]``) would store raw model instances in ``plan_args``:
not JSON-serializable by the plain ``json.dumps`` the SSE broadcaster uses,
and not comparable by ordinary ``==`` in the exact same way plain dicts are
for :func:`_diff_keys`. Dumping to JSON-safe primitives up front keeps the
draft's on-the-wire and in-memory representations identical.

Plan-model resolution (`get_facility_plans()`) does synchronous file I/O — it
re-scans the session-plan directory on every call (see `plan_loader.py`).
Whenever a `PATCH` names a `plan_name`, that resolution runs via
``asyncio.to_thread`` *before* the draft lock is acquired, and the resolved
schema class is cached on the new draft object; every subsequent per-field
validation (until the next `plan_name` change) reads that cached class only
— no further registry calls. This module never copies the artifacts
interface's `_SSEBroadcaster` cross-thread ``put_nowait`` pattern
(`interfaces/artifacts/app.py`): every mutation here happens inside an
``async def`` route already running on the event loop, so a synchronous
``put_nowait`` from within the same coroutine, under the same lock that just
bumped the revision, is safe and requires no cross-thread handoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, TypeAdapter, ValidationError

logger = logging.getLogger("osprey.services.bluesky_bridge.draft")

router = APIRouter()

# Per-subscriber SSE queue depth. Generous enough that a normally-paced
# client (one frame per human/agent edit) never overflows it, but bounded —
# see `_broadcast_locked`'s disconnect-on-overflow behavior.
_QUEUE_MAXSIZE = 64

# How often the SSE stream emits a bare comment frame to keep the connection
# alive through intermediary timeouts (the sidecar relay's own per-request
# `httpx.Timeout(None, connect=5.0)` disables its *read* timeout, but nothing
# stops an even further upstream proxy from closing an apparently-idle
# connection).
_HEARTBEAT_INTERVAL_S = 15.0

# Sentinel pushed onto a subscriber's queue in place of a frame it couldn't
# hold (see `_broadcast_locked`) — the SSE generator treats receiving this
# object (identity check, never equality) as "close the stream now", forcing
# the client to reconnect and resync via a fresh hello frame rather than
# silently missing frames.
_DISCONNECT = object()

_MISSING = object()


@dataclass
class _Draft:
    """The singleton draft's fields, plus the plan's cached schema class.

    ``model`` is resolved once — off the event loop, before the draft lock —
    whenever a `PATCH` sets `plan_name` (see module docstring), and reused
    for every later per-field validation until the plan changes again. It is
    deliberately excluded from :meth:`to_dict` (never serialized to a client;
    it exists purely to save a `get_facility_plans()` call per field).
    """

    plan_name: str
    plan_args: dict[str, Any]
    model: type[BaseModel]
    updated_by: str | None
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "plan_args": dict(self.plan_args),
            "updated_by": self.updated_by,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class LaunchSnapshot:
    """A launchable snapshot of the draft, pinned at `revision`.

    Returned by :func:`check_launchable` when the caller's pinned
    `draft_revision` both matches the current draft AND names a revision that
    hasn't already been launched. The caller mints and launches the run
    *outside* the draft lock from this snapshot, then records the launch (and
    broadcasts it) via :func:`record_and_broadcast_launch` on success — the
    draft lock is never held across the launch itself.

    `plan_args` is a defensive copy taken under the lock, so a concurrent
    `PATCH` mutating the live draft can never alter what the caller launches.
    """

    plan_name: str
    plan_args: dict[str, Any]
    revision: int


@dataclass(frozen=True)
class LaunchRejected:
    """Why a pinned `draft_revision` can't be launched (a typed refusal).

    Returned by :func:`check_launchable` instead of a snapshot so the caller
    maps each cause to a distinct `409` without minting or launching
    anything. `code` is one of:

    - ``"stale_draft_revision"`` — no draft exists, or the pinned revision
      doesn't equal the current `revision` (someone edited or cleared the
      draft since the caller last read it).
    - ``"draft_revision_already_launched"`` — this exact revision was already
      launched, OR another caller's launch of it is in flight right now (a
      reservation held in `_launching`, see :func:`check_launchable`); the
      guard against a replayed or concurrent launch firing a duplicate
      hardware scan. One code for both on purpose: the caller's remedy is
      identical (edit the draft to bump the revision, or resync), and clients
      already branch on exactly two codes — the two situations differ only in
      wording of `detail`. A later `PATCH` bumps the revision and re-arms
      launch.

    `revision` is always the *current* server revision, so the caller can
    hand the client a fresh baseline to resync against.
    """

    code: str
    detail: str
    revision: int


# ---------------------------------------------------------------------------
# Module state: one draft, one process-lifetime monotonic revision counter,
# one lock guarding both plus the subscriber set. `revision` is intentionally
# NOT a field on `_Draft` — `GET /draft` must report it even when `draft` is
# `None` (symmetric with the hello frame), so it lives outside the object
# that can itself become `None`.
#
# `_last_launched_revision` lives alongside `_revision` under the same lock:
# it is the revision most recently launched via `POST /draft/run` (0 = none
# launched this process). `check_launchable` refuses to re-launch a revision
# equal to it, so a replayed launch can't fire a second hardware scan; a
# `PATCH` that bumps `_revision` past it re-arms launch. Like `_revision` it
# is monotonic in practice and never reset for real drafts — only `_clear()`
# (test isolation) resets it.
#
# `_launching` holds revisions whose launch is IN FLIGHT: reserved by
# `check_launchable` in the same critical section that returns the snapshot,
# and released by exactly one of `record_and_broadcast_launch` (success) or
# `release_launch` (any failure after the check). Reserving at the HEAD of
# the unlocked mint/launch window — not committing at its tail — is what
# makes "exactly one launch per revision" hold for CONCURRENT callers, not
# just sequential replays: a second `check_launchable` at the same revision
# finds the reservation and refuses while the first launch is still running.
# ---------------------------------------------------------------------------
_draft: _Draft | None = None
_revision: int = 0
_last_launched_revision: int = 0
_launching: set[int] = set()
_lock = asyncio.Lock()
_subscribers: set[asyncio.Queue[Any]] = set()


def _clear() -> None:
    """Reset all module-level draft state (test isolation only).

    Mirrors `live_rows.py`'s `_clear()` test hook. Also drops any still-
    registered SSE subscriber queues — a leftover subscriber from a prior
    test's un-closed stream must never receive frames meant for the next
    test's isolated state.
    """
    global _draft, _revision, _last_launched_revision
    _draft = None
    _revision = 0
    _last_launched_revision = 0
    _launching.clear()
    _subscribers.clear()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _diff_keys(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Sorted keys whose value differs between *old* and *new* plan_args.

    A key present in one dict and absent in the other counts as changed too
    — this is what lets one helper serve merge patches, `remove`, and a full
    plan_name replacement's wholesale plan_args swap without special-casing
    any of them: "changed" is always "value comparison against the current
    draft", never derived from a patch's key list.
    """
    keys = set(old) | set(new)
    return sorted(key for key in keys if old.get(key, _MISSING) != new.get(key, _MISSING))


def _validate_field(model: type[BaseModel], key: str, value: Any) -> Any:
    """Coerce and validate one `plan_args` value against ``model``'s field ``key``.

    Raises `HTTPException(422, ...)` for an unknown field name or a value
    that fails the field's type/`Field()` constraints. The FieldInfo itself
    is passed as `Annotated` metadata (never unpacked with `*`) — see the
    module docstring for why. The returned value is dumped back through
    `TypeAdapter.dump_python(mode="json")` so the draft only ever stores
    JSON-safe primitives, never live pydantic model instances.
    """
    field = model.model_fields.get(key)
    if field is None:
        raise HTTPException(
            status_code=422,
            detail={"field": key, "error": f"unknown field {key!r} for this plan"},
        )
    adapter: TypeAdapter[Any] = TypeAdapter(Annotated[field.annotation, field])
    try:
        coerced = adapter.validate_python(value)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"field": key, "error": str(exc)}) from exc
    return adapter.dump_python(coerced, mode="json")


async def _resolve_plan_schema(plan_name: str) -> type[BaseModel]:
    """Off-loop registry lookup for ``plan_name``'s parameter schema.

    `get_facility_plans()` does synchronous file I/O (it re-scans the
    session-plan directory on every call); running it via `asyncio.to_thread`
    keeps the event loop free while it does so. Callers run this BEFORE
    acquiring `_lock` (module docstring) — the lock only ever guards the
    in-memory mutation and broadcast, never this lookup.

    Raises `HTTPException(422, ...)` if ``plan_name`` isn't registered.
    """
    from .plan_loader import get_facility_plans

    facility_plans = await asyncio.to_thread(get_facility_plans)
    spec = facility_plans.plans.get(plan_name)
    if spec is None:
        raise HTTPException(status_code=422, detail=f"unknown plan {plan_name!r}")
    return spec.schema


def _format_sse(frame: dict[str, Any]) -> str:
    return f"data: {json.dumps(frame)}\n\n"


def _broadcast_locked(frame: dict[str, Any]) -> None:
    """Push *frame* to every subscriber queue. MUST be called while holding `_lock`.

    On a full queue, this disconnects that subscriber instead of silently
    dropping the frame (PROPOSAL.md "Slow consumers"): the queue is drained
    and a `_DISCONNECT` sentinel is enqueued in its place, so the SSE
    generator wakes up, closes the stream, and the client reconnects to a
    fresh hello resync — never a client that silently missed frames.
    """
    for queue in list(_subscribers):
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            _subscribers.discard(queue)
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            try:
                queue.put_nowait(_DISCONNECT)
            except asyncio.QueueFull:  # pragma: no cover - defensive only
                pass


async def _subscribe() -> tuple[asyncio.Queue[Any], dict[str, Any]]:
    """Register a new SSE subscriber and take its hello snapshot atomically.

    Both happen under `_lock`, in the same critical section — a mutation
    that runs its own `_broadcast_locked` (also lock-guarded) can never land
    between "this subscriber's queue exists" and "this subscriber's hello
    snapshot was read", so a subscriber never gets a hello reflecting a
    stale state while another client already has a newer frame in hand.
    """
    async with _lock:
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        _subscribers.add(queue)
        hello = {
            "type": "hello",
            "draft": _draft.to_dict() if _draft is not None else None,
            "revision": _revision,
        }
    return queue, hello


async def _unsubscribe(queue: asyncio.Queue[Any]) -> None:
    async with _lock:
        _subscribers.discard(queue)


# ---------------------------------------------------------------------------
# Launch state: the substrate for `POST /draft/run` (the bridge's launch
# primitive). The draft lock is NEVER held across `registry.add`/`do_launch`,
# so the launch flow is deliberately split into lock-guarded steps with the
# mint/launch happening between them, outside the lock:
#
#   snapshot = await check_launchable(draft_revision)   # under _lock; RESERVES
#   ... mint + launch the run in a threadpool ...       # NO lock held
#   await record_and_broadcast_launch(run_id, rev)      # under _lock; consumes
#   # ...or, on ANY failure after a successful check:
#   await release_launch(rev)                           # under _lock; releases
#
# `check_launchable` mints nothing on failure — a stale, already-launched, or
# currently-launching revision returns a typed `LaunchRejected` the caller
# turns into a `409`. On success it RESERVES the revision in `_launching`
# within the same critical section, so a concurrent second caller at the same
# revision is refused while the first launch is still in flight — the
# exclusivity token is taken at the head of the unlocked window, never
# committed only at its tail. The caller owes exactly one matching
# `record_and_broadcast_launch` (success) or `release_launch` (failure).
# ---------------------------------------------------------------------------


async def check_launchable(draft_revision: int) -> LaunchSnapshot | LaunchRejected:
    """Atomically snapshot the draft — and reserve the launch — at a pinned `draft_revision`.

    Under `_lock`, in one critical section, checks that (a) a draft exists
    and its revision equals `draft_revision`, (b) that revision hasn't
    already been launched, and (c) no other caller's launch of it is
    currently in flight. On success it adds the revision to `_launching`
    (the in-flight reservation) and returns a :class:`LaunchSnapshot` (with
    a defensive copy of `plan_args`) the caller launches *outside* the lock;
    the caller must then settle the reservation with exactly one of
    :func:`record_and_broadcast_launch` (success) or :func:`release_launch`
    (failure). On any failure this returns a typed :class:`LaunchRejected`,
    having minted and reserved nothing.

    Check (c) shares check (b)'s ``draft_revision_already_launched`` code
    (see :class:`LaunchRejected` for why) with in-flight-specific wording:
    without the reservation, two concurrent callers pinning the same
    revision could both pass (b) — neither has recorded yet — and each fire
    a real hardware scan; the guard would only defeat sequential replays.

    Taking the snapshot, all three checks, and the reservation in the same
    critical section is what guarantees the caller launches exactly the
    plan_args that were current at `draft_revision`, exactly once: a
    concurrent `PATCH` either lands entirely before this call (so the
    revision it reads already reflects the edit and a caller pinning the old
    revision gets `stale_draft_revision`) or entirely after it (so this
    snapshot is untouched), and a concurrent launch either reserved first
    (so this call is refused) or sees this call's reservation.
    """
    async with _lock:
        if _draft is None or _revision != draft_revision:
            return LaunchRejected(
                code="stale_draft_revision",
                detail=(
                    f"pinned draft_revision {draft_revision} does not match the "
                    f"current draft revision {_revision}"
                ),
                revision=_revision,
            )
        if _revision == _last_launched_revision:
            return LaunchRejected(
                code="draft_revision_already_launched",
                detail=(
                    f"draft revision {_revision} was already launched; edit the "
                    "draft (bumping its revision) to launch again"
                ),
                revision=_revision,
            )
        if _revision in _launching:
            return LaunchRejected(
                code="draft_revision_already_launched",
                detail=(
                    f"a launch of draft revision {_revision} is already in "
                    "progress; edit the draft (bumping its revision) to launch "
                    "a different plan"
                ),
                revision=_revision,
            )
        _launching.add(_revision)
        return LaunchSnapshot(
            plan_name=_draft.plan_name,
            plan_args=dict(_draft.plan_args),
            revision=_revision,
        )


async def record_and_broadcast_launch(*, run_id: str, revision: int) -> None:
    """Record a successful launch and announce it to subscribers, atomically.

    Settles `check_launchable`'s in-flight reservation (``_launching.discard``
    — tolerant of a caller that never reserved, e.g. white-box tests driving
    this directly), sets `_last_launched_revision = revision` (arming the
    duplicate-launch guard for that revision), and broadcasts a ``launched``
    frame (``{type, run_id, revision}``) to every SSE subscriber — all under
    `_lock` in one critical section: the "this revision is now launched"
    state and the frame announcing it can never be observed out of order, and
    the broadcast reuses the same `_broadcast_locked` mechanism as `change`
    frames.

    Called only *after* the run was successfully minted and launched outside
    the lock (see the module comment above); `revision` is the pinned
    revision from the :class:`LaunchSnapshot` that authorized this launch.
    """
    global _last_launched_revision
    async with _lock:
        _launching.discard(revision)
        _last_launched_revision = revision
        _broadcast_locked({"type": "launched", "run_id": run_id, "revision": revision})


async def release_launch(revision: int) -> None:
    """Release `check_launchable`'s in-flight reservation WITHOUT recording a launch.

    The failure-path counterpart of :func:`record_and_broadcast_launch`: the
    caller's mint/launch failed after a successful check, so the reservation
    is dropped and — because `_last_launched_revision` is left untouched —
    the same revision is immediately launchable again (a failed launch never
    consumes the revision). Idempotent, and tolerant of a revision that was
    never reserved.
    """
    async with _lock:
        _launching.discard(revision)


class PatchDraftRequest(BaseModel):
    """Body for `PATCH /draft`.

    ``client_id`` is required (never optional) — it is both the frame
    ``origin`` other subscribers use for echo suppression, and the
    ``updated_by`` stamp on the draft itself. The MCP draft tools always send
    the fixed id ``"mcp-agent"``; the plan panel sends its own per-tab id.
    """

    plan_args_patch: dict[str, Any] | None = None
    remove: list[str] | None = None
    plan_name: str | None = None
    expected_plan_name: str | None = None
    client_id: str


@router.get("/draft")
async def get_draft() -> dict[str, Any]:
    """Current draft (or `null`) plus the process-monotonic revision.

    Always `200` — never `404` — and `revision` is present even when
    `draft` is `null`, so a client can always resync to a baseline
    (symmetric with the SSE hello frame).
    """
    async with _lock:
        return {
            "draft": _draft.to_dict() if _draft is not None else None,
            "revision": _revision,
        }


@router.patch("/draft")
async def patch_draft(body: PatchDraftRequest) -> dict[str, Any]:
    """Merge/replace the draft's `plan_args`, or set/switch its `plan_name`.

    See the module docstring for the per-field validation and plan-
    resolution contract. Response body is always
    ``{revision, changed, plan_name}`` — including on a no-op patch (no
    value actually changed): the caller can always read the current
    revision/plan_name off the response without a follow-up `GET`.

    Raises:
        HTTPException(422): unknown `plan_name`, an unknown `plan_args_patch`
            key, or a value that fails its field's validation.
        HTTPException(409): `expected_plan_name` doesn't match the draft's
            current `plan_name` (``code: "plan_name_mismatch"``), or there is
            no existing draft and no `plan_name` was given to create one
            (``code: "no_draft"``).
    """
    global _draft, _revision

    resolved_model: type[BaseModel] | None = None
    if body.plan_name is not None:
        # Off-loop, before the lock (module docstring): `get_facility_plans()`
        # does sync file I/O and must never block the event loop, let alone
        # run while `_lock` is held.
        resolved_model = await _resolve_plan_schema(body.plan_name)

    async with _lock:
        current_plan_name = _draft.plan_name if _draft is not None else None

        if body.expected_plan_name is not None and body.expected_plan_name != current_plan_name:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "plan_name_mismatch",
                    "detail": (
                        f"expected draft plan_name {body.expected_plan_name!r}, "
                        f"but the current draft's plan_name is {current_plan_name!r}"
                    ),
                },
            )

        if _draft is None and body.plan_name is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "no_draft",
                    "detail": "no draft exists; provide plan_name to create one",
                },
            )

        plan_name_changing = body.plan_name is not None and body.plan_name != current_plan_name

        if body.plan_name is not None:
            # Resolve which model *would* back the draft if this patch isn't
            # a true no-op. A reaffirming plan_name PATCH that changes
            # nothing still hits the no-op early-return below, which keeps
            # the previously cached model — the swap to `resolved_model`
            # here only actually lands when the patch changes something (a
            # merge-only patch with no plan_name never triggers a resolve at
            # all — module docstring's "no registry calls" guarantee holds
            # for that path).
            model = resolved_model
            assert model is not None
        else:
            assert _draft is not None  # guaranteed by the no_draft check above
            model = _draft.model

        if plan_name_changing:
            old_args: dict[str, Any] = dict(_draft.plan_args) if _draft is not None else {}
            new_args: dict[str, Any] = {}
        else:
            assert _draft is not None
            old_args = dict(_draft.plan_args)
            new_args = dict(old_args)

        for key, raw_value in (body.plan_args_patch or {}).items():
            new_args[key] = _validate_field(model, key, raw_value)

        for key in body.remove or []:
            new_args.pop(key, None)

        changed = _diff_keys(old_args, new_args)

        if not plan_name_changing and not changed:
            # True no-op (PROPOSAL.md: "A PATCH that changes nothing is a
            # no-op: no revision bump, no frame"). `current_plan_name` here
            # is `_draft.plan_name`, unchanged.
            return {"revision": _revision, "changed": [], "plan_name": current_plan_name}

        new_plan_name = body.plan_name if body.plan_name is not None else current_plan_name
        assert new_plan_name is not None
        _draft = _Draft(
            plan_name=new_plan_name,
            plan_args=new_args,
            model=model,
            updated_by=body.client_id,
            updated_at=_now_iso(),
        )
        _revision += 1

        frame = {
            "type": "plan-change" if plan_name_changing else "change",
            "draft": _draft.to_dict(),
            "changed": changed,
            "revision": _revision,
            "origin": body.client_id,
        }
        _broadcast_locked(frame)

        return {"revision": _revision, "changed": changed, "plan_name": _draft.plan_name}


@router.delete("/draft")
async def delete_draft(client_id: str | None = None) -> dict[str, Any]:
    """Clear the draft. The sole clear path; idempotent.

    A no-op (`200`, ``cleared: false``, no revision bump, no SSE frame) when
    no draft exists — calling this twice in a row is always safe. When a
    draft IS cleared, the revision still bumps (never resets): a
    `draft_revision` pinned from before the clear must never match a later
    rebuilt draft.
    """
    global _draft, _revision
    async with _lock:
        if _draft is None:
            return {"revision": _revision, "cleared": False}

        removed_keys = sorted(_draft.plan_args.keys())
        _draft = None
        _revision += 1

        frame = {
            "type": "clear",
            "draft": None,
            "changed": removed_keys,
            "revision": _revision,
            "origin": client_id,
        }
        _broadcast_locked(frame)

        return {"revision": _revision, "cleared": True}


@router.get("/draft/events")
async def draft_events() -> StreamingResponse:
    """SSE stream of draft changes: a hello frame on connect, then live frames.

    Frame types: ``hello`` (full draft + revision, sent once on connect),
    ``change`` (merge/remove patch), ``clear`` (`DELETE /draft`), and
    ``plan-change`` (a `PATCH` that set/switched `plan_name`). A bare `:
    heartbeat` comment is emitted roughly every 15 s of inactivity to keep
    intermediary proxies from treating the connection as idle-dead.

    On a full subscriber queue (a client falling behind), the stream is
    closed rather than silently dropping frames — see `_broadcast_locked`.
    """
    queue, hello = await _subscribe()

    async def _generate():
        try:
            yield _format_sse(hello)
            while True:
                try:
                    frame = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL_S)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if frame is _DISCONNECT:
                    break
                yield _format_sse(frame)
        finally:
            await _unsubscribe(queue)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
