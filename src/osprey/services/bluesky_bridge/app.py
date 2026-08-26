"""The Bluesky bridge's FastAPI app: the HTTP surface in front of the queue.

Two processes, one machine: this app runs in a separate container from
OSPREY's own venv, reachable only over HTTP plus the ``X-Launch-Token``
header.

The bridge does not run plans. Execution belongs to the queueserver worker
(``qserver_startup.py``), which owns the RunEngine and every device; this
process composes and validates plans, enqueues them through ``queue.py``,
projects the manager's own item state back out as run records (``runs.py``),
and serves the data the document plane buffers. That division is what keeps
this module import-clean of bluesky/ophyd/tiled (``_BRIDGE_ONLY_MODULES``) —
there is no in-process runner left to import them for.

The direct-execute routes that predate the queue (``POST /runs``,
``POST /runs/{id}/launch``, ``POST /draft/run``, ``POST /runs/{id}/stop``)
still answer, with a single machine-readable ``use_the_queue`` refusal naming
the route that replaced each of them — a removed route would 404 and tell a
caller nothing about where its capability went.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import ValidationError

from . import analysis, document_plane, draft, figure_cache, live_rows, queue, runs
from .figure import (
    DEFAULT_MAX_POINTS,
    REASON_NO_RENDER,
    REASON_PARAMS_MISMATCH,
    REASON_PLAN_IDENTITY_UNAVAILABLE,
    REASON_RENDER_FAILED,
    REASON_RENDER_NOT_SUPPORTED_FOR_SESSION_PLANS,
    REASON_SOURCE_UNAVAILABLE,
    BarsMark,
    Figure,
    LinesMark,
    Point,
    RowWindow,
    Series,
    decimate,
    default_figure,
    rows_from_columnar,
)
from .models import (
    PlanSessionWriteRequest,
    PlanValidateRequest,
)
from .plan_fields import MOVABLE_ROLE, READABLE_ROLE, collect_channels
from .plan_metadata import extract_plan_name_from_source
from .plan_types import PlanSpec, Provenance
from .plan_validation import hash_plan_body, validate_plan
from .queue_backend import (
    PLAN_META_KEY,
    FunctionFailedError,
    FunctionTimeoutError,
    QueueBackendError,
    QueueUnavailableError,
)
from .session_dir import resolve_session_plan_dir
from .session_upload import get_session_uploader, upload_after_validation
from .validation import _assert_limits_readable_if_writable
from .validation_record import validation_records

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger("osprey.services.bluesky_bridge.app")

# Root package names this module — and therefore the bridge's import path —
# must never pull in. The bluesky stack lives in the queueserver worker's
# process, not here; `test_app_import_clean.py` enforces the boundary by
# importing this app in a subprocess and checking `sys.modules`.
#
# `bluesky_tiled_plugins` is named on its own rather than left to the fact that
# it imports `tiled` transitively: the guard has to fail when *this* module
# grows an eager import of it, not only for as long as the plugin keeps a
# top-level `tiled` import of its own.
_BRIDGE_ONLY_MODULES = {"bluesky", "bluesky_tiled_plugins", "ophyd", "ophyd_async", "tiled"}

# The durable catalog the worker's `TiledWriter` persists runs into; read here
# only to serve `GET /runs/{id}/data` once a run's live buffer is gone. The
# worker names the same two variables for its writing half
# (`qserver_startup.py`) — the two halves are wired by the compose spec, which
# is the one place both are set.
_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"

# The refusal code every retired direct-execute route answers with. One code
# for all of them: the answer is always "this capability belongs to the queue",
# and the per-route sentence in `detail` says which route owns it.
_USE_THE_QUEUE = "use_the_queue"


def _use_the_queue(detail: str) -> HTTPException:
    """The machine-readable refusal a retired direct-execute route raises.

    Same ``{"code", "detail"}`` body shape as every queue refusal
    (``queue.py``), so a caller branches on ``detail.code`` uniformly rather
    than special-casing these four routes. 410 Gone, not 404: the route is
    still here and still answering — what is gone is the in-process execution
    behind it.
    """
    return HTTPException(status_code=410, detail={"code": _USE_THE_QUEUE, "detail": detail})


# The bridge's single `QueueBackend` — one queueserver client handle for the
# whole process, built lazily on first use so no route ever constructs its own.
# `None` until then; on a browse-only deployment the backend it builds holds no
# manager at all (`QSERVER_ZMQ_CONTROL_ADDRESS` unset).
_queue_backend: Any | None = None


def get_queue_backend() -> Any:
    """The bridge's single `QueueBackend`, built from the compose env on first use.

    Every route that speaks to the queue server comes through here, so the
    process holds exactly one 0MQ handle. The import is deliberately lazy:
    `app.py`'s module-level import graph stays exactly as it was.
    """
    global _queue_backend
    if _queue_backend is None:
        from .queue_backend import QueueBackend

        _queue_backend = QueueBackend.from_env()
    return _queue_backend


def set_queue_backend(backend: Any | None) -> None:
    """Override the backend `get_queue_backend` returns.

    Passing `None` clears it, so the next `get_queue_backend()` rebuilds from
    the environment as it stands then.
    """
    global _queue_backend
    _queue_backend = backend


async def _open_environment_at_startup() -> None:
    """Bring the qserver worker environment up at startup, off the serve path.

    Environment ownership is the bridge's: on a deployment whose capability
    says it can execute, `ensure_environment`
    opens the worker (bounded retry — device connect can take tens of
    seconds, which is why this runs as a background task rather than blocking
    readiness); on a browse-only deployment it opens nothing, and a closed
    environment is the healthy steady state. Failures are logged, never
    fatal: the start route re-runs `ensure_environment_for_execute` on every
    armed start, so a slow or failed startup open self-heals there.

    Opening the environment builds the worker namespace from the startup
    module, which knows nothing about session plans — so every open is also
    the moment the validated session set has to go back in. `sync_namespace`
    (`session_upload.py`) re-uploads whatever is missing; it is idempotent,
    and a no-op when the environment stayed closed or nothing is validated
    (the normal shape of a fresh start, where the in-memory validation
    records are empty anyway).
    """
    from .queue_backend import QueueBackendError

    try:
        await get_queue_backend().ensure_environment()
        await get_session_uploader().sync_namespace()
    except QueueBackendError as exc:
        logger.warning("worker environment did not open at startup: %s", exc)
    except Exception:
        logger.exception("startup environment open failed unexpectedly")


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Bring up the process-wide plumbing the bridge owns, and tear it down once.

    Three things, none of which involves running a plan (execution belongs to
    the queueserver worker):

    - The write-safety startup guard. `_assert_limits_readable_if_writable`
      fail-OPEN refuses startup (raises) only if writes are enabled, limits
      checking is enabled, and the limits database can't be read — every other
      combination, including writes disabled entirely, starts normally. It runs
      unconditionally here, never behind a device-wiring flag such as
      `BLUESKY_DEVICES_FILE`: the posture it guards against is a property of
      the project config, not of whether this deployment wires up any devices
      at all, and a gated guard leaves whole classes of deployment unchecked.
    - The document plane: the 0MQ proxy the queueserver's Publisher connects
      to, and the dispatcher that turns that stream into live rows.
      Unconfigured is a no-op; see `document_plane.start_from_env`.
    - The worker environment, opened in the background, and the queue
      plumbing's single teardown.
    """
    from osprey.utils.logger import configure_logging

    # The bridge is launched as `uvicorn ...:app`, so it passes through no
    # Osprey entry point. Configuring here — on serve, never on import — keeps
    # the startup breadcrumbs below visible in `docker logs` without turning
    # importing this module into a logging side effect.
    configure_logging()

    # The one startup posture the bridge refuses to come up in (writable +
    # limits checking on + an unreadable limits database), before anything
    # else is brought up.
    _assert_limits_readable_if_writable()

    # The document plane's 0MQ proxy — the binding element the queueserver's
    # Publisher connects to, and the RemoteDispatcher that turns that stream
    # back into live rows. Unconfigured is a no-op; see
    # `document_plane.start_from_env`.
    document_plane.start_from_env()

    # Environment ownership — kick the startup open in the
    # background (never blocking readiness) and tear the queue plumbing down
    # exactly once on shutdown. Only when no backend was injected before
    # startup: a pre-set backend means a test (or bespoke deploy wiring) owns
    # the backend's whole lifecycle — environment AND close — and an
    # unasked-for probe or a shutdown close() of another owner's handle would
    # be interference, not help. In production nothing runs before the
    # lifespan, so the backend is always un-built here, the open always
    # happens, and whatever `get_queue_backend()` lazily builds while serving
    # is this lifespan's to close.
    backend_injected_before_startup = _queue_backend is not None
    env_open_task: asyncio.Task[None] | None = None
    if not backend_injected_before_startup:
        env_open_task = asyncio.create_task(_open_environment_at_startup())

    yield

    if env_open_task is not None:
        env_open_task.cancel()
        with suppress(asyncio.CancelledError):
            await env_open_task
    await queue.shutdown()
    document_plane.shutdown()
    if _queue_backend is not None and not backend_injected_before_startup:
        await _queue_backend.close()
        set_queue_backend(None)


app = FastAPI(title="OSPREY Bluesky Bridge", lifespan=_lifespan)

# The shared plan draft's routes (`GET`/`PATCH`/`DELETE /draft`, `GET
# /draft/events`) live in their own self-contained module — state, lock, and
# SSE broadcaster all belong together, and don't need anything else this
# module owns. First `include_router` precedent in this app; every other
# route here is still an inline `@app.<verb>`. `POST /draft/run` is NOT part
# of that router: it is one of the retired direct-execute routes this module
# answers with a `use_the_queue` refusal, below.
app.include_router(draft.router)

# The queue surface (`GET /queue`, `POST /queue/items`, move/remove, token-
# gated `POST /queue/start`, `POST /queue/stop`, `GET /queue/events` SSE) is
# its own self-contained module for the same reason the draft is: its arming
# lock, SSE poller, and last-status cache belong together. It reaches back
# into this module only through `get_queue_backend()`, the process's single
# backend accessor.
app.include_router(queue.router)


async def _capability_dict() -> dict[str, Any]:
    """This deployment's capability record as the JSON object `/health` publishes.

    `QueueBackend.capability` is already fail-closed for every situation it can
    name — an unreadable project config, a connector that cannot drive Channel
    Access, an unconfigured manager, an unanswering one — each with its own
    reason code. This wrapper covers the remainder: if building the backend or
    asking it raises anything at all, the answer is still "no", reported as
    ``manager_unreachable``, which is precisely what that code means here (the
    bridge could not confirm a working queue server). The underlying error goes
    in ``detail`` so the operator sees the real cause rather than a shrug, and
    the route still answers 200 — the container healthcheck must not go red
    over a capability probe.

    The hand-built record below is the ONE capability object on the wire that
    does not come out of ``Capability.to_dict``, so it has to carry the same
    keys — a consumer that parses the fallback differently from every other
    record is a consumer that breaks exactly when the bridge is already in
    trouble. The lane identity in particular is still perfectly knowable here:
    it comes from render-time facts (`resolve_lane_identity`, which never
    raises), not from the probe that just failed.
    """
    from .queue_backend import REASON_MANAGER_UNREACHABLE, resolve_lane_identity

    try:
        return (await get_queue_backend().capability()).to_dict()
    except Exception as exc:
        logger.warning("capability probe failed; reporting cannot-execute: %s", exc)
        lane, lane_target = resolve_lane_identity()
        return {
            "can_execute": False,
            "reason": REASON_MANAGER_UNREACHABLE,
            "detail": f"The bridge could not determine whether plans can execute: {exc}",
            "lane": lane,
            "lane_target": lane_target,
        }


@app.get("/health")
async def health() -> dict:
    """Liveness, plus whether this deployment can actually execute plans.

    The capability record rides on the existing status surface rather than a
    route of its own, so no consumer has to learn a second endpoint to find out
    what the bridge in front of it can do:

    ``{"status": "ok", "capability": {"can_execute", "reason", "detail",
    "lane", "lane_target"}}``

    ``can_execute`` is the answer; ``reason`` is one of the machine-readable
    ``REASON_*`` codes in `queue_backend.py`, which panels and MCP tools branch
    on; ``detail`` is the operator-facing sentence, and for a browse-only mock
    deployment it names the exact command that flips it. ``lane`` and
    ``lane_target`` say which plan lane answered and which control target it
    serves — render-time facts, identical before and after a session switch,
    which the host is what composes against the session's own target.
    ``status: "ok"`` means
    only that this process is up — it is deliberately independent of
    ``can_execute``, because a browse-only deployment is a healthy deployment.
    """
    return {"status": "ok", "capability": await _capability_dict()}


async def _manager_view() -> tuple[Any, list[Any], list[Any]]:
    """The three manager documents every run record is projected from.

    Fetched together so one route answer is one consistent picture: the running
    item, the pending queue, and the history. `runs.py` does the projection —
    this function is only the I/O.
    """
    backend = get_queue_backend()
    queue_state = await backend.items()
    history = await backend.history()
    return (
        queue_state.get("running_item"),
        list(queue_state.get("items") or []),
        list(history.get("items") or []),
    )


@app.post("/runs")
def create_run() -> dict:
    """Retired: the bridge mints no launch intent of its own.

    Minting a run id here and launching it in a second call would leave an
    intermediate state nothing owns. Both halves belong to `POST /queue/items`,
    which mints the id AND enqueues in one armed, race-checked step.
    """
    raise _use_the_queue(
        "The bridge no longer records launch intents separately from execution. "
        "Enqueue the draft with POST /queue/items, which mints the run id and "
        "returns it."
    )


@app.get("/runs")
async def list_runs(limit: int = 20) -> list[dict]:
    """Runs OSPREY has enqueued, most relevant first: running, pending, then history.

    Derived live from the queue server rather than from any state this process
    keeps, so the answer survives a bridge restart and cannot drift from what
    the manager is actually holding. Items enqueued out-of-band carry no OSPREY
    run id and are absent here; `GET /queue` shows the manager's queue whole.
    """
    try:
        running_item, queue_items, history_items = await _manager_view()
    except QueueBackendError as exc:
        # The queue surface owns this mapping; duplicating it here is exactly
        # how two halves of one wire contract drift apart.
        raise queue._http_error(exc) from exc
    return runs.list_records(
        running_item=running_item,
        queue_items=queue_items,
        history_items=history_items,
        limit=limit,
    )


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    """One run's record, including its plan name and the params it was enqueued with.

    404 when the manager knows no such run — which is also the honest answer
    for a run whose history the manager has since rotated away. Note that
    `GET /runs/{id}/data` can still serve a run this route 404s: run *data* is
    durable in Tiled long after the manager has forgotten the item.
    """
    try:
        running_item, queue_items, history_items = await _manager_view()
    except QueueBackendError as exc:
        raise queue._http_error(exc) from exc
    record = runs.find_record(
        run_id,
        running_item=running_item,
        queue_items=queue_items,
        history_items=history_items,
    )
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return record


@app.post("/runs/{run_id}/launch")
def launch_run(run_id: str) -> dict:
    """Retired: launching a pre-minted run in-process is gone.

    Execution is the queueserver worker's, and the only way into it is the
    queue: `POST /queue/items` (enqueue) plus the token-gated `POST
    /queue/start` (arm and drain).
    """
    raise _use_the_queue(
        "Plans now execute in the queue server, not in the bridge. Enqueue the "
        "draft with POST /queue/items, then arm the queue with POST /queue/start."
    )


@app.post("/draft/run")
def launch_draft_run() -> dict:
    """Retired: launching the shared draft directly is gone.

    `POST /queue/items` carries every guarantee a direct launch would — the
    pinned `draft_revision` and the reservation that stops two callers racing
    the same revision onto hardware — while putting the plan in a durable,
    serialized queue instead of a thread in this process. The launch
    token still gates every enqueue that can actually start something: one onto
    a draining queue, or onto a queue armed to autostart.
    """
    raise _use_the_queue(
        "The shared draft is no longer launched directly. Enqueue it with "
        "POST /queue/items (same pinned draft_revision and launch token), then "
        "arm the queue with POST /queue/start."
    )


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    """Retired: stopping is a queue operation.

    This route can only stop a plan running inside the bridge process, and none
    do. Which of the two queue operations a caller wants depends on what they
    need stopped: `POST /queue/stop` halts the queue AFTER
    the running item finishes, and `POST /queue/abort` aborts the item already
    in motion. Neither is token-gated, on the same principle — halting is
    always allowed.
    """
    raise _use_the_queue(
        "Plans run in the queue server, not in the bridge. Halt the queue after the "
        "running item with POST /queue/stop, or abort the running plan now with "
        "POST /queue/abort."
    )


@app.get("/plans")
def list_plans() -> list:
    """Registered plans: `plan_loader.get_facility_plans()`'s trust-resolved set.

    `plan_loader.py` is the sole plan registry — a layered directory scan
    (`shipped`/`preset`/`facility`/`session`) plus the legacy single-module
    facility-injection contract, merged fail-closed by trust tier (see that
    module's docstring). It is import-clean of bluesky, so this route never
    needs a guarded import.

    Each entry (`PlanSpec.to_dict()`) carries `metadata` (the plan's
    authoring-declared `PLAN_METADATA`, or `None` if it doesn't author one)
    and `provenance` (its loader-assigned trust tier) alongside
    `name`/`description`/`schema` — see `plan_types.py`.
    """
    from .plan_loader import get_facility_plans

    return [spec.to_dict() for spec in get_facility_plans().plans.values()]


# The per-device keys `GET /devices` republishes from the manager's own device
# description. The rest of what it carries (`classname`, `module`, the
# component tree) is worker-internal detail nothing can be done with from
# outside the worker; these three protocol flags are the part that answers the
# question a caller actually has — whether a device can be driven as a
# setpoint or only read as a readback. Absent keys stay absent rather than
# defaulting to False: "the manager did not say" is not "no".
_DEVICE_FLAG_KEYS = ("is_movable", "is_readable", "is_flyable")


def _device_entry(name: str, description: Any) -> dict[str, Any]:
    """One `GET /devices` entry: the name, plus whichever flags the manager gave it."""
    if not isinstance(description, dict):
        return {"name": name}
    return {
        "name": name,
        **{key: description[key] for key in _DEVICE_FLAG_KEYS if key in description},
    }


@app.get("/devices")
async def list_devices() -> list[dict]:
    """Devices the queueserver worker built, by the name plans resolve them under.

    The companion of `GET /plans`: a plan's device parameters carry device
    *names* as strings, and this is the set those names must come from. A name
    absent here is a device the worker does not have, and a plan naming it
    fails on the run's first iteration — after an enqueue and a start — so this
    route is what turns picking a device into a lookup rather than a guess.

    Each entry is `{"name", ...}` plus whatever of `is_movable`/`is_readable`/
    `is_flyable` the manager reported for it, which is how a caller tells a
    drivable setpoint from a read-only readback.
    """
    try:
        reply = await get_queue_backend().devices_allowed()
    except QueueBackendError as exc:
        # Same mapping as every other manager-backed read (see `list_runs`).
        raise queue._http_error(exc) from exc
    allowed = reply.get("devices_allowed")
    if not isinstance(allowed, dict):
        return []
    return [_device_entry(name, description) for name, description in sorted(allowed.items())]


# ---------------------------------------------------------------------------
# Session-plan authoring + validation
# ---------------------------------------------------------------------------
# A valid Python identifier: the sanitized name doubles as the on-disk file
# stem (`<name>.py`) and the `PLAN_METADATA["name"]` value, so this also rules
# out path traversal (`../`, absolute paths, path separators) in one check.
# Anchored with `\Z`, NOT `$` — `$` matches at end-of-string OR just before a
# single trailing "\n", so `"foo\n"` would otherwise pass this check while
# still not being a valid identifier.
#
# A LEADING UNDERSCORE is excluded, which makes this narrower than "identifier":
# the queueserver manager's permissions forbid `:^_` plans (private names are
# never exposed), so `_foo` would author and upload perfectly well and then be
# permanently unenqueueable — the worker would hold it while `plans_allowed`
# never listed it, and the session-plan gate would refuse it forever with
# "its upload did not land". Refusing the name up front turns a permanent
# mystery refusal into one legible 400 at authoring time.
_PLAN_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*\Z")

# A generous bound well above any real plan name — exists only so an
# absurdly long name fails closed here (400) rather than surfacing as an
# unhandled OSError from `Path.write_text` (some filesystems reject a
# filename this long outright, which would otherwise 500).
_MAX_PLAN_NAME_LENGTH = 100

# Neither `/plans/session` nor `/plans/validate` is gated on
# `BLUESKY_LAUNCH_TOKEN` (`security.py`) — that token authenticates network
# callers to the two launch routes only, and both these routes MUST keep
# working with writes disabled: authoring and validating a plan
# body never touches a device (the validator's stage-3 dry run drives mock
# devices only, in a subprocess with `EPICS_CA_*` neutralized — see
# `plan_validation.py`). Their protection is the bridge's loopback-only bind
# (see the compose template) plus the MCP-side approval hook
# (`registry/mcp.py`'s `write_plan`/`validate_plan` tiers) —
# not a token gate.


def _sanitize_plan_name(name: str) -> str:
    """Validate ``name`` as a safe plan name, or raise 400.

    Enforced as a Python identifier that does not begin with an underscore
    (not merely "no path separators") because the same string is written into
    the generated ``PLAN_METADATA["name"]`` block as a plain literal, used
    verbatim as the on-disk file stem, and — for a session plan — becomes the
    name the queueserver worker exposes, where a leading underscore is
    permanently unenqueueable (see `_PLAN_NAME_RE`).
    Length-checked FIRST, before the regex echoes ``name`` back in the error
    detail — an oversized name fails closed on its length alone rather than
    being quoted in full into an HTTPException detail.
    """
    if len(name) > _MAX_PLAN_NAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"invalid plan name: exceeds the {_MAX_PLAN_NAME_LENGTH}-character limit",
        )
    if not _PLAN_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=(
                f"invalid plan name {name!r}: must be a valid Python identifier "
                f"that does not begin with an underscore"
            ),
        )
    return name


@app.post("/plans/session")
def write_session_plan(request: PlanSessionWriteRequest) -> dict:
    """Author a session-tier plan file. NEVER imports or execs it.

    Assembles the final file content as ONE string — a generated
    `PLAN_METADATA = {...}` block carrying exactly the three declared fields
    (``name``/``description``/``writes``), followed by the author's own
    ``body`` — and writes exactly that string to ``resolve_session_plan_dir()/<name>.py``,
    overwriting any existing file of the same name (a name reused for
    different content is a re-authoring: its hash changes, so any prior
    validation record no longer matches — the file becomes unvalidated again
    until `POST /plans/validate` is called on it, which is the correct
    fail-closed behavior).

    Returns the plan name and `hash_plan_body` of the EXACT bytes written —
    the same bytes `POST /plans/validate` re-reads and hashes, and the same
    bytes the load and enqueue gates re-hash from disk when checking for a
    passing validation record.
    """
    name = _sanitize_plan_name(request.name)
    metadata = {
        "name": name,
        "description": request.description,
        "writes": request.writes,
    }
    final_content = f"PLAN_METADATA = {metadata!r}\n\n{request.body}"

    plan_path = resolve_session_plan_dir() / f"{name}.py"
    plan_path.write_text(final_content, encoding="utf-8")

    return {"name": name, "content_hash": hash_plan_body(final_content)}


@app.post("/plans/validate")
async def validate_session_plan(request: PlanValidateRequest) -> dict:
    """Validate the CURRENT on-disk content of a session plan file.

    Reads the file `POST /plans/session` wrote (never a separately-passed
    body) so "validated bytes == file bytes" is structural, not a caller
    convention. Runs `validate_plan`'s three ordered stages; on a
    pass, records the content hash in `validation_records` so the session-layer
    load gate and the enqueue gate will admit this exact file content.

    A pass is also the ONLY thing that puts a session plan in the queueserver
    worker's namespace: `upload_after_validation` (`session_upload.py`) runs
    the qserver script upload for exactly the bytes that just passed. It
    reports rather than raises, and its outcome rides along as ``upload`` —
    a deployment with no queue server validates plans perfectly well and
    simply has nowhere to upload them, so an upload that did not happen must
    never turn a PASS into an error. It is not a safety gap either: enqueue
    and queue-start re-check namespace presence and refuse on their own.

    Raises 404 if no session plan named ``request.name`` has been written.
    """
    name = _sanitize_plan_name(request.name)
    plan_path = resolve_session_plan_dir() / f"{name}.py"
    if not plan_path.is_file():
        raise HTTPException(status_code=404, detail=f"unknown session plan {name!r}")

    content = plan_path.read_text(encoding="utf-8")
    result = await validate_plan(
        content,
        plan_name=name,
        sample_args=request.sample_args,
        dry_run_timeout=request.dry_run_timeout,
    )
    upload: dict[str, Any] = {"uploaded": False, "reason": None, "detail": None}
    if result.passed:
        validation_records.record(result.content_hash)
        upload = await upload_after_validation(name)

    return {
        "passed": result.passed,
        "reasons": result.reasons,
        "content_hash": result.content_hash,
        "upload": upload,
    }


@app.delete("/plans/session/{name}")
def delete_session_plan(name: str) -> dict:
    """Retire an authored session-tier plan file. Reaches NO hardware.

    The counterpart of `POST /plans/session`, and the only way to un-author a
    plan short of restarting the bridge — session state is ephemeral by design
    (`session_dir.py`), but "ephemeral" only ever meant "lost on restart",
    which left a plan authored once listed in `GET /plans` for the life of the
    container. Ungated for the same reason the write and validate routes are:
    removing a file touches no device, and `BLUESKY_LAUNCH_TOKEN`
    authenticates the two launch routes only.

    Scope is exactly one directory. The name resolves inside
    `resolve_session_plan_dir()` and nowhere else, so the read-only
    `shipped`/`preset`/`facility` layers are unreachable from here — a name
    belonging to one of those simply has no session file behind it and
    answers `deleted: false`.

    The plan leaves `GET /plans` immediately: `plan_loader` re-scans the
    directory layers on every call rather than caching them, so every surface
    that resolves a name through the registry (the draft's plan-name check,
    the load gate, the enqueue gate) stops finding it with no restart. Work
    already under way is untouched — a queued or running item resolved its
    plan out of the queueserver worker's namespace when it was enqueued, and
    that namespace is not this route's to edit.

    Any passing validation record for the removed bytes is deliberately left
    alone. Records are keyed by content hash, not by name, so an orphan can
    only ever be matched again by re-authoring byte-identical content — which
    is the same file the record was made for, and correctly still validated.

    Returns:
        `{"name", "deleted"}` — `deleted` is False when there was no file to
        remove. Idempotent like `DELETE /draft`: a caller that wants the name
        gone gets that outcome either way, and never has to tell "I removed
        it" apart from "it was already absent".
    """
    name = _sanitize_plan_name(name)
    plan_path = resolve_session_plan_dir() / f"{name}.py"
    # missing_ok rather than an is_file() precondition: the check and the
    # unlink cannot be made atomic, and losing that race to a concurrent
    # delete still leaves the caller in the state it asked for.
    existed = plan_path.is_file()
    plan_path.unlink(missing_ok=True)
    return {"name": name, "deleted": existed}


# ---------------------------------------------------------------------------
# Plan source rendering: backs the launch-approval hook's
# human-legible plan excerpt — the human backstop for the plan validator's
# documented, accepted obfuscation residual (see `plan_validation.py`'s
# module docstring). Read-only: the lookup itself never execs anything, only
# reads file text already sitting on disk — it does consult the plan registry
# (`plan_loader.get_facility_plans`) to break a same-name collision the way the
# loader broke it, which is the registry the bridge builds at startup and every
# `GET /plans` already returns, not a new execution surface.
# ---------------------------------------------------------------------------

_SOURCE_TRUNCATE_CHARS = 4000  # default: a few KB, enough for a human skim
# Hard ceiling for an explicit `max_chars` ask (the plan panel's Source tab
# requests full source this way). Far above any real plan file, but still a
# bound — the response can never grow unbounded with the file.
_SOURCE_TRUNCATE_CHARS_MAX = 200_000


def _find_layer_source_path(name: str) -> tuple[Any, Provenance] | None:
    """Best-effort locate the on-disk file behind a shipped/preset/facility plan.

    Directory-layer files are keyed by their declared ``PLAN_METADATA["name"]``,
    not necessarily their filename — so each candidate file's declared name is
    read with `extract_plan_name_from_source`, the shared source-only reader
    for that one key (it parses the file, never imports or execs it).

    Identity, not validity: this scan asks only *which file declares this
    name*, and so must be exactly as permissive as an import. It deliberately
    does not use the full-contract reader
    (`extract_plan_metadata_from_source`), because a file the loader accepts
    and registers can still fail that stricter read — a ``PLAN_METADATA`` whose
    ``description`` is a module-level constant rather than a literal, say — and
    404ing on such a plan would strip the source excerpt out of the human
    approval prompt for a plan that is registered and launchable.

    A file whose declared name cannot be read at all is skipped and the scan
    continues to the next file and the next layer — never fatal. Such a file
    cannot be the plan being asked about under any reading of it, so skipping
    it decides nothing; it is logged at debug level because the route reports
    only a bare 404.

    When several layers declare the same plan name, the file served has to be
    the file that would actually run — anything else shows an operator the
    source of a file that is not the one about to execute, which is worse at an
    approval gate than showing nothing. For a REGISTERED name the collision is
    resolved by the registry itself: the candidate whose tier equals the
    registered spec's ``provenance`` wins, so this agrees with
    `plan_loader._register_plan` by reading its decision rather than re-deriving
    it. Re-deriving it would not be equivalent — `_register_plan` only ever sees
    files that loaded, and this scan cannot tell a loaded file from a
    quarantined one, so a quarantined facility file shadowing a loadable shipped
    one would win here while losing in the registry.

    For an UNREGISTERED name the scan falls back to highest trust seen, last
    match winning. That is the quarantined-file case — nothing registered the
    name, and an operator debugging why their facility plan never appeared needs
    exactly that file's text — so the fallback keeps an unregistered file's
    source reachable instead of 404ing it.

    Plans supplied through the legacy ``BLUESKY_PLAN_MODULE`` contract are
    outside this scan either way: they register at `facility` rank but live in
    no scanned directory, so no candidate file can match them. Pre-existing, and
    unchanged by the registry tie-break.

    Returns `None` for a plan with no backing file at all, or a name this scan
    can't locate; the route degrades to a 404 either way.
    """
    from .plan_loader import (
        _TRUST_ORDER,
        _iter_plan_files,
        _resolve_plan_dir_layers,
        get_facility_plans,
    )

    registered = get_facility_plans().plans.get(name)
    registered_provenance = registered.provenance if registered is not None else None

    matched: tuple[Any, Provenance] | None = None
    found: tuple[Any, Provenance] | None = None
    found_rank = -1
    for directory, provenance in _resolve_plan_dir_layers():
        # The session tier is resolved by filename before this scan is reached,
        # and must never be served as if it carried a directory tier's trust.
        if provenance == "session":
            continue
        rank = _TRUST_ORDER[provenance]
        for path in _iter_plan_files(directory):
            declared = extract_plan_name_from_source(path)
            if declared is None:
                logger.debug("plan source scan: skipping %s (no readable plan name)", path)
                continue
            if declared != name:
                continue
            if provenance == registered_provenance:
                # Last match within the registered tier wins: same-tier
                # directories are scanned in the loader's own order, and
                # `_register_plan` lets the later definition win there too.
                matched = (path, provenance)
            # `>=`, not `>`: an equal-trust redefinition lets the later-scanned
            # file win, which is `_register_plan`'s rule for that case too.
            if rank >= found_rank:
                found, found_rank = (path, provenance), rank
    return matched if matched is not None else found


@app.get("/plans/{name}/source")
def get_plan_source(
    name: str,
    max_chars: int = Query(default=_SOURCE_TRUNCATE_CHARS, ge=1, le=_SOURCE_TRUNCATE_CHARS_MAX),
) -> dict:
    """Truncated source text for one plan — the approval hook's data source
    for showing a human what the plan they are approving would actually run.

    ``max_chars`` bounds the returned source text. The default stays at the
    approval hook's skim size (the hook embeds the response verbatim in its
    prompt, so its excerpt must stay small); a caller that needs the full
    text — the plan panel's Source tab — asks for more explicitly. The ask is
    itself capped (422 beyond ``_SOURCE_TRUNCATE_CHARS_MAX``) so the response
    stays bounded no matter what the client sends.

    A session-tier file is looked up directly: its filename IS its name (see
    `write_session_plan`). Its ``validated`` flag reflects the SAME
    `hash_plan_body`/`validation_records` check the load and enqueue gates use,
    computed fresh from the file's CURRENT content — never cached — so a
    re-authored file that invalidates a prior pass is reported honestly, even
    if that leaves it quarantined out of `GET /plans` entirely.

    A shipped/preset/facility file is located by `_find_layer_source_path`
    (best-effort) and reported ``validated=True`` unconditionally — those
    tiers carry no validation-record gate; they are operator-trusted by
    construction, not by a passing record.

    Raises 404 if no file can be located for ``name`` in any tier.
    """
    name = _sanitize_plan_name(name)
    session_path = resolve_session_plan_dir() / f"{name}.py"
    provenance: Provenance
    if session_path.is_file():
        content = session_path.read_text(encoding="utf-8")
        validated = validation_records.has_passing_record(hash_plan_body(content))
        provenance = "session"
    else:
        found = _find_layer_source_path(name)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no source file found for plan {name!r}")
        path, provenance = found
        content = path.read_text(encoding="utf-8")
        validated = True

    truncated_content = content[:max_chars]
    return {
        "name": name,
        "provenance": provenance,
        "validated": validated,
        "truncated": len(truncated_content) < len(content),
        "source": truncated_content,
    }


# ---------------------------------------------------------------------------
# Read-only pre-flight: what running a plan would actually do
# ---------------------------------------------------------------------------
# The other half of the launch-approval hook's evidence. `GET /plans/{name}/source`
# shows a human the code they are approving; this route shows them its
# consequences — the channels the plan declares it will move and read, and the
# values it would drive them to, for the exact parameters about to be launched.
#
# Nothing here moves anything. The worker builds the plan and walks its
# instruction stream without a RunEngine consuming it (`qserver_startup.py`'s
# `preview_plan`), which reads the trajectory without driving it, and this
# process only relays the summary.

# The worker-namespace function that produces the trajectory summary. Called by
# name, and permitted by name in the manager's function permissions — the one
# deliberate exception in an otherwise deny-all list.
_PREVIEW_FUNCTION = "preview_plan"

# Length bound on any explanatory sentence this route puts on the wire. The
# approval hook embeds the response verbatim in a prompt a human reads, so an
# unbounded relay — a worker traceback, a deeply nested validation report —
# must not be able to bury the summary it accompanies.
_PREVIEW_DETAIL_CHARS = 2000

# Why no trajectory could be summarized. Every one of them is a 200 answer
# carrying the same payload shape as a successful summary, so the approval hook
# reads `ok` and renders either the trajectory or the reason it has none —
# never a status code, and never an error branch of its own.
PREVIEW_REASON_UNKNOWN_PLAN = "unknown_plan"
PREVIEW_REASON_CATALOG_UNAVAILABLE = "plan_catalog_unavailable"
PREVIEW_REASON_QUEUE_UNREACHABLE = "queue_unreachable"
PREVIEW_REASON_REFUSED = "preview_refused"
PREVIEW_REASON_TIMED_OUT = "preview_timed_out"
PREVIEW_REASON_FAILED = "preview_failed"
PREVIEW_REASON_PLAN_ERROR = "plan_error"


def _declared_channels(spec: PlanSpec[Any], params: Any) -> list[dict[str, str]]:
    """The channels *params* supplies, each labelled with the role the plan gave it.

    The plan's own declaration is the only source: `PlanSpec.roles` is what its
    schema declared, and `plan_fields.collect_channels` reads back which names
    the supplied parameters put in those fields. A plan that declares no channel
    roles at all contributes nothing rather than having its parameters guessed
    at by field name.

    Movable entries come first — the channels a launch would drive are what an
    approver needs to see before the ones it would merely record.
    """
    if not spec.roles:
        return []
    return [
        {"channel": name, "role": role}
        for role in (MOVABLE_ROLE, READABLE_ROLE)
        for name in collect_channels(spec.schema, params, role)
    ]


def _preview_params(body: bytes) -> dict[str, Any] | None:
    """The parameters a request body carries, or ``None`` if it carries none.

    The body is read and parsed here rather than declared as a typed argument,
    because a typed one would answer a malformed or non-object body with
    FastAPI's own 422 — the one answer this route promises never to give. No
    body at all, and an explicit ``null``, both mean "no parameters": a plan
    whose parameters all default is previewed by name alone.
    """
    if not body.strip():
        return {}
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if parsed is None:
        return {}
    return parsed if isinstance(parsed, dict) else None


def _preview_unavailable(
    name: str, channels: list[dict[str, str]], reason: str, detail: str
) -> dict[str, Any]:
    """The answer when no trajectory could be summarized: same keys, no moves.

    Deliberately identical in shape to a successful summary. Whatever went
    wrong, the caller reads one payload — and keeps whatever channel list could
    still be derived, because "these are the channels this launch declares it
    would move" stays true and stays worth showing even when the trajectory
    behind them is unavailable.
    """
    return {
        "ok": False,
        "plan": name,
        "channels": channels,
        "moves": [],
        "total_moves": 0,
        "truncated": False,
        "move_cap": None,
        "reason": reason,
        "detail": detail[:_PREVIEW_DETAIL_CHARS],
    }


@app.post(
    "/plans/{name}/preview",
    openapi_extra={
        "requestBody": {
            "required": False,
            "content": {"application/json": {"schema": {"type": "object"}}},
        }
    },
)
async def preview_plan(name: str, request: Request) -> dict:
    """The moves running ``name`` with the posted parameters would make. Moves nothing.

    POST, and the request body is the plan's parameters exactly as the queue
    item carries them — one nested JSON object, the same mapping
    `POST /queue/items` would enqueue. A GET could only carry that shape as
    JSON packed into a query string, which every caller would then have to
    encode by hand; every route on this app that takes a nested object takes it
    as a body, and every one that takes query parameters takes scalars.

    Total by construction, exactly like `GET /runs/{id}/figure`: this answer is
    read by the launch-approval gate, and a gate that errors is a gate that
    stops a human from seeing what they are about to approve. So there is one
    payload shape and one 200 — ``ok`` says whether a trajectory was summarized,
    and when it was not, ``reason`` says why in a machine-readable word and
    ``detail`` in a sentence:

    - ``unknown_plan`` — no plan of that name is registered. Not a 404: an
      unknown name is one more reason the gate has no trajectory to show, and
      it arrives in the same shape as every other.
    - ``plan_catalog_unavailable`` — the plan registry itself could not be read.
    - ``queue_unreachable`` — no queue server could be asked to walk the plan:
      none is configured, none answered, or the connection to one could not be
      built from this deployment's settings at all.
    - ``preview_refused`` — the queue server answered and refused the call. Its
      own sentence is relayed in ``detail``: the pre-flight function may not be
      permitted for this deployment, or the worker may not be able to take the
      call right now. The two are not told apart here, because the queue server
      does not distinguish them in any way this process could read without
      pattern-matching its prose.
    - ``preview_timed_out`` — the call was accepted but had not finished inside
      the backend's budget.
    - ``preview_failed`` — the worker ran the pre-flight and the call itself
      failed, or answered with something that is not a summary.
    - ``plan_error`` — the parameters are what could not produce a plan:
      parameters that do not validate, a channel the worker has no device for,
      anything the plan itself raises. ``detail`` carries the worker's own
      reason, which is the same reason a launch would fail with. A body that is
      not a JSON object at all is reported here too — it is the same fault one
      step earlier, and it is answered before the plan is even looked up, so it
      is what a request with both a bad body and an unknown name reports.

    On success the payload carries the worker's summary — ``moves`` (each
    ``{"channel", "target"}``, in the order the run would make them),
    ``total_moves`` (the exact count, whether or not the list holds them all),
    ``truncated``, and the ``move_cap`` the worker collected up to — plus
    ``channels``, this process's own reading of the plan's role declaration
    against these parameters. ``channels`` is present on every answer, success
    or not, whenever the plan is known.
    """
    from .plan_loader import get_facility_plans

    plan_params = _preview_params(await request.body())
    if plan_params is None:
        return _preview_unavailable(
            name,
            [],
            PREVIEW_REASON_PLAN_ERROR,
            "The plan parameters must be a JSON object; this request body is not one.",
        )

    try:
        # Re-scans the session-plan directory on every call, so it runs off the
        # loop (`draft._resolve_plan_schema` states the house rule).
        facility_plans = await asyncio.to_thread(get_facility_plans)
    except Exception as exc:
        logger.warning("pre-flight: the plan registry is unreadable: %s", exc)
        return _preview_unavailable(
            name,
            [],
            PREVIEW_REASON_CATALOG_UNAVAILABLE,
            f"The plan registry could not be read: {exc}",
        )

    spec = facility_plans.plans.get(name)
    if spec is None:
        return _preview_unavailable(
            name, [], PREVIEW_REASON_UNKNOWN_PLAN, f"No plan named {name!r} is registered."
        )

    try:
        channels = _declared_channels(spec, plan_params)
    except Exception:
        # The parameters are whatever the caller sent, and reading a channel
        # list out of them is a description of the launch — a description must
        # never be the thing that costs an approver their summary.
        logger.warning("pre-flight: reading %r's channel declaration failed", name, exc_info=True)
        channels = []

    try:
        payload = await get_queue_backend().function_execute(_PREVIEW_FUNCTION, name, plan_params)
    except QueueUnavailableError as exc:
        return _preview_unavailable(name, channels, PREVIEW_REASON_QUEUE_UNREACHABLE, str(exc))
    except FunctionTimeoutError as exc:
        return _preview_unavailable(name, channels, PREVIEW_REASON_TIMED_OUT, str(exc))
    except FunctionFailedError as exc:
        return _preview_unavailable(name, channels, PREVIEW_REASON_FAILED, str(exc))
    except QueueBackendError as exc:
        return _preview_unavailable(name, channels, PREVIEW_REASON_REFUSED, str(exc))
    except Exception as exc:
        # The rungs above name what `queue_backend` translates, and a typed
        # ladder is only as total as its base class: building the connection
        # from a malformed environment happens before any translation exists,
        # and upstream's client raises types of its own that the backend does
        # not rewrite. Neither may reach an approver as a 500, so the ladder
        # ends on a rung that cannot be fallen off.
        logger.warning("pre-flight: the queue call for %r failed", name, exc_info=True)
        return _preview_unavailable(
            name,
            channels,
            PREVIEW_REASON_QUEUE_UNREACHABLE,
            f"The queue server could not be reached: {exc}",
        )

    if not isinstance(payload, dict):
        return _preview_unavailable(
            name,
            channels,
            PREVIEW_REASON_FAILED,
            f"The pre-flight answered with {type(payload).__name__}, not a summary.",
        )
    if not payload.get("ok"):
        return _preview_unavailable(
            name,
            channels,
            PREVIEW_REASON_PLAN_ERROR,
            str(payload.get("error") or "The pre-flight reported no reason."),
        )

    moves = payload.get("moves")
    total_moves = payload.get("total_moves")
    # `bool` is the one `int` subclass that is never a count, so it is excluded
    # explicitly rather than admitted by `isinstance`.
    counted = isinstance(total_moves, int) and not isinstance(total_moves, bool)
    if not isinstance(moves, list) or not counted:
        # A summary missing its moves or its exact count is not a summary, and
        # relaying it half-formed would let the gate show an approver a move
        # count that is not one.
        return _preview_unavailable(
            name,
            channels,
            PREVIEW_REASON_FAILED,
            "The pre-flight answered without the moves it would make.",
        )

    return {
        "ok": True,
        "plan": name,
        "channels": channels,
        "moves": moves,
        "total_moves": total_moves,
        "truncated": bool(payload.get("truncated")),
        "move_cap": payload.get("move_cap"),
        "reason": None,
        "detail": None,
    }


def _window(
    columns: list[str],
    rows: list[Any],
    total_seen: int,
    max_rows: int,
    offset: int | None,
    tail: bool,
) -> dict[str, Any]:
    """Compute a bounded, paginated window over a row buffer.

    Shared by every data source `get_run_data` serves from (today: the live
    buffer; later: Tiled) — one implementation is what makes pagination
    parity across sources structural rather than something tests have to
    police across copies.

    ``row_count`` is `total_seen` — the *true* total rows the run has
    produced, even if that's more than what's physically passed in via
    ``rows`` — not ``len(rows)``. ``truncated`` reflects whether this
    response's window omits any of the passed-in rows, or whether more rows
    exist than were passed in at all.
    """
    stored_count = len(rows)
    max_rows = max(0, max_rows)
    skip = max(0, offset) if offset is not None else 0

    if tail:
        end = max(0, stored_count - skip)
        start = max(0, end - max_rows)
    else:
        start = skip
        end = start + max_rows
    window = rows[start:end]

    truncated = start > 0 or end < stored_count or total_seen > stored_count

    return {
        "columns": list(columns),
        "rows": window,
        "row_count": total_seen,
        "truncated": truncated,
    }


def _tiled_client() -> Any | None:
    """A client for the configured Tiled catalog, or ``None`` when unconfigured.

    The runs this catalog holds were written by ``TiledWriter``, which stamps
    each one with the ``BlueskyRun``/``BlueskyEventStream`` specs. Installing
    `bluesky-tiled-plugins` registers the client classes those specs dispatch
    to, so every run this function's client hands back is a `BlueskyRun` — a
    client that knows the catalog's own layout — rather than a bare container
    the bridge would have to walk itself.

    ``BLUESKY_TILED_API_KEY`` is read with ``.get``, never a bare subscript: a
    token-less catalog is a working configuration (`from_uri` accepts
    ``api_key=None``), not a ``KeyError`` on the first read that needs it.

    `tiled` is imported here, never at module level, so `app.py` stays
    import-clean of it (`_BRIDGE_ONLY_MODULES`) even when Tiled *is* configured
    for this deploy.
    """
    uri = os.environ.get(_TILED_URI_ENV)
    if not uri:
        logger.info(
            "tiled read: %s is unset; Tiled is not configured for this deploy", _TILED_URI_ENV
        )
        return None

    from tiled.client import from_uri

    return from_uri(uri, api_key=os.environ.get(_TILED_API_KEY_ENV))


def _newest_run_node(client: Any, run_id: str) -> Any | None:
    """The run for *run_id* — the newest ``start.time`` when several match.

    A re-run of an interrupted item records a second start document under the
    same OSPREY run id, so the search can legitimately return several runs;
    ``matches[0]`` would leave the choice to whatever order the server answers
    in. The newest start is the run the id currently means — the same "latest
    occupant wins" rule the live buffer applies by overwriting on start.

    The start doc `TiledWriter.start` records lives under ``metadata["start"]``
    on the run container — a bare ``Key("osprey_run_id")`` matches nothing.
    ``Key`` comes from the plugin's query surface, which re-exports Tiled's own:
    one import site for everything this module asks the catalog.
    """
    from bluesky_tiled_plugins.queries import Key

    matches = list(client.search(Key("start.osprey_run_id") == run_id).values())
    if not matches:
        return None

    def _start_time(run: Any) -> float:
        time = dict(run.metadata).get("start", {}).get("time")
        return time if isinstance(time, int | float) else float("-inf")

    return max(matches, key=_start_time)


def _row_part_columns(stream: Any) -> list[str]:
    """The stream's table-family column names, in the order the catalog stores them.

    Answered from structure metadata alone — ``structure_family`` and, for a
    table, ``columns`` both come off the item the container already holds, so
    naming a part here costs no payload fetch. That is the whole point: a
    stream's non-table parts are its external arrays (a waveform, an image
    stack), and naming one in a read downloads the entire array before the row
    filter in `_primary_table` throws it away. Reading a run must not cost the
    frames the run happens to carry.

    ``structure_family`` is compared against the bare string ``"table"``
    because Tiled's `StructureFamily` is a ``str`` enum — matching on the value
    keeps `tiled` out of this module's import path (`_BRIDGE_ONLY_MODULES`)
    without a lazy import that buys nothing.

    Order is the container's own part order, then each table's declared column
    order — the order the run was written in. `CompositeClient.read` builds its
    variables from a ``set``, so the read's own order varies with the process's
    string hash salt; carrying the declared order through here is what keeps
    the column list a caller sees stable across bridge restarts, and equal to
    the order the live path served the same run in.
    """
    columns: list[str] = []
    for _name, part in stream.base.items():
        if part.structure_family == "table":
            columns.extend(part.columns)
    return columns


def _primary_table(run: Any) -> Any:
    """The run's ``primary`` stream as a row table, read through the plugin.

    The stream client `BlueskyRun` hands back knows where the stream's own
    parts live, so this asks the stream to read itself, and asks its parts what
    they ARE (`_row_part_columns`) rather than hard-coding the name of the one
    that holds the rows. ``.base["internal"]`` is the layout this used to
    depend on from here; a catalog that stored the rows under a different part
    name would still read correctly now.

    The read names the table parts' columns explicitly (`_row_part_columns`)
    rather than leaning on the stream's default projection: the projection this
    bridge serves is `_data_columns`, and a source read that had already
    dropped columns would leave ``seq_num`` — the ordering key the figure path
    sorts on — unavailable.

    Row-shapedness is then decided per variable — one dimension, and as long as
    the row axis — never from whether the read gave every variable the same
    dimension NAME. When a composite read finds parts of differing length it
    privately renames every variable's dimension, and a rule that compared dim
    names would collapse to ``seq_num`` alone and serve a run's real rows as
    ``columns: []``: a silent empty answer a caller cannot tell from a genuinely
    column-less run. The frame is built from the arrays directly for the same
    reason — ``to_dataframe()`` would cross-join variables whose dims were
    renamed apart.

    A stream with no table part at all reads as an empty frame, not an
    exception: it is a real, if column-less, run.

    The read runs with dask's ``dataframe.convert-string`` turned OFF, and that
    is load-bearing rather than tuning. Tiled's table client fetches each
    partition as Arrow and assembles them through dask; with the option on
    (dask's default) dask casts every object-dtype column to a string dtype on
    the way out. For a text column that is a no-op, but a waveform signal is
    stored as an Arrow ``list<double>`` whose cells are arrays, and ``astype``
    on an array is ``str`` of it — so the whole reading arrives as the TEXT
    ``'[0. 1. 2. 3. 4. 5. 6. 7.]'``, with the numbers gone before this module
    ever sees them. Measured against the pinned server, not reasoned: the same
    read with the option off hands back the arrays. Scoped to this read as a
    context manager, so nothing else in the process inherits it.
    """
    import dask
    import pandas as pd

    stream = run["primary"]
    declared = _row_part_columns(stream)
    if not declared:
        return pd.DataFrame()

    with dask.config.set({"dataframe.convert-string": False}):
        dataset = stream.read(variables=declared)
    lengths = {
        name: dataset[name].values.shape[0]
        for name in declared
        if name in dataset and dataset[name].values.ndim == 1
    }
    if not lengths:
        return pd.DataFrame()

    row_axis = next(
        (lengths[name] for name in ("seq_num", "time") if name in lengths),
        next(iter(lengths.values())),
    )
    return pd.DataFrame(
        {name: dataset[name].values for name, length in lengths.items() if length == row_axis}
    )


def _data_columns(table: Any) -> list[str]:
    """The stored table's data columns, projected onto the live buffer's set.

    Tiled's stored rows carry ``seq_num``, ``time``, and per-signal ``ts_*``
    timestamp columns the live buffer never had (see `LiveRowRecorder`). The
    Tiled read projects them away HERE, so "a run replayed from Tiled has the
    identical column set it had live" is one rule in one place — the same
    reason the data and figure routes share `_tiled_run_snapshot` rather than
    each keeping a filter that happens to agree with the other.
    """
    return [c for c in table.columns if c != "seq_num" and c != "time" and not c.startswith("ts_")]


def _json_cell(value: Any) -> Any:
    """One stored cell as something JSON can carry losslessly.

    Only ever called for cells that came out of an object-dtype column, which
    is the only place a non-scalar reading can be. A waveform signal is stored
    as an arrow ``list<double>`` column and reads back as a one-dimensional
    object array whose every cell is itself a `numpy.ndarray`; ``tolist()``
    turns one into a plain nested list of Python floats, which is what a JSON
    array is made of. Anything else in an object column — a string status, a
    ``None``, a Python float — has no ``tolist`` and passes through untouched.

    ``tolist()`` rather than ``list()`` on purpose: it recurses (an image cell
    is a nested list, not a list of arrays) and it converts numpy scalars to
    Python ones, so nothing downstream has to know a numpy type ever existed.
    """
    to_list = getattr(value, "tolist", None)
    return to_list() if callable(to_list) else value


def _json_rows(frame: Any) -> list[list[Any]]:
    """The frame's rows as JSON-ready lists — arrays become arrays, not text.

    ``frame.values.tolist()`` alone is right for a frame of numbers and wrong
    for one carrying a waveform: a mixed frame's ``.values`` is object-dtype, so
    each waveform cell survives as a `numpy.ndarray`, and an ndarray reaching
    the response encoder comes out as the TEXT ``'[0. 1. 2.]'`` — not a JSON
    array, not round-trippable, and indistinguishable from a signal that really
    did record that string. So the array cells are converted here, at the one
    place rows are built, rather than left for whatever encoder happens to see
    them last.

    Which cells those are is decided per COLUMN, from the frame's own dtypes,
    not per cell: a scalar-only run is the common case and this route is polled
    at 1 Hz over runs that reach six figures of rows, so a run with no object
    column pays nothing beyond the ``tolist()`` it already paid.

    NaN and infinity inside a waveform end up as Python floats, exactly as they
    do in a scalar column, and the response encoder renders both as ``null``
    either way — the same value a caller reads for a scalar the readback could
    not measure. A waveform is not a special case in the wire format; it is a
    cell whose value happens to be a list.
    """
    rows: list[list[Any]] = frame.values.tolist()
    object_columns = [
        position
        for position, dtype in enumerate(frame.dtypes)
        if dtype == object  # noqa: E721
    ]
    if not object_columns:
        return rows
    for row in rows:
        for position in object_columns:
            row[position] = _json_cell(row[position])
    return rows


def _tiled_run_snapshot(run_id: str) -> dict[str, Any] | None:
    """One run's full stored state, read from the durable Tiled catalog.

    The single Tiled read in this module: `_from_tiled` windows this for the
    data route, `get_run_figure` composes a figure from it, and they see the
    same rows in the same order flagged the same way BECAUSE it is one read and
    not two. A run served from Tiled must not look settled on one route and
    in-flight on the other, or ordered on one and shuffled on the other.

    Two situations fall through the live path in `get_run_data` and land here: a
    run with no live buffer at all (a bridge restart drops every buffer, so a
    run that started before it — even one still executing — has nothing in
    memory), and one whose buffer was evicted past `live_rows._MAX_RUNS`. The
    search keys on `osprey_run_id`, the durable stamp the enqueue path threads
    into the item metadata and the worker records onto the start document.

    Every stored row is returned, never a window: a figure is computed over the
    whole run (a windowed ORM fit is silently wrong), and the data route does
    its own bounding afterwards. Rows come back sorted by ``seq_num`` — emission
    order, however the catalog chose to return them — and projected onto the
    live-buffer column set exactly as the live path serves it. Cells are plain
    JSON values: a scalar reading is a number, a waveform reading is a list of
    numbers (`_json_rows`), and neither is ever text describing itself.

    ``partial`` is the absence of a stop document, the same signal ``partial``
    means everywhere else, and never the run record: a run still executing
    after a bridge restart reads as partial from the catalog alone. ``plan`` is
    the stamp the enqueue path recorded onto the start document, which is what
    tells both routes which channels the run declared — the figure route to
    pick its axes, the data route to compute the statistics over them.

    Returns `None` when Tiled is unconfigured (`BLUESKY_TILED_URI` unset —
    logged, not an error) or when no run in the catalog matches `run_id`; both
    callers turn either into the same 404.

    `tiled` and its Bluesky client plugin are imported below the routes, never
    at module level, so `app.py` stays import-clean of them
    (`_BRIDGE_ONLY_MODULES`) even when Tiled *is* configured for this deploy.
    """
    client = _tiled_client()
    if client is None:
        return None
    run_node = _newest_run_node(client, run_id)
    if run_node is None:
        return None

    metadata = dict(run_node.metadata)
    start = metadata.get("start") or {}

    if "primary" not in run_node:
        # Start doc landed but no Event ever arrived (e.g. a plan that
        # errored before its first point) — the run is real, so this is the
        # "nothing to read yet" shape, not a 404. Deliberately a membership
        # check on `"primary"` alone, never a `try`/`except KeyError` around
        # the read below: a stream whose own parts cannot be resolved raises
        # `KeyError` too, and a broad guard here would silently answer a
        # broken catalog with this empty-but-successful shape.
        columns: list[str] = []
        rows: list[Any] = []
        total_seen = 0
    else:
        table = _primary_table(run_node)
        if "seq_num" in table.columns:
            table = table.sort_values("seq_num", kind="stable")
        columns = _data_columns(table)
        rows = _json_rows(table[columns])
        total_seen = len(table)

    return {
        "columns": columns,
        "rows": rows,
        "total_seen": total_seen,
        "partial": metadata.get("stop") is None,
        "plan": start.get(PLAN_META_KEY),
        "run_uid": start.get("uid"),
    }


def _compute_analysis(
    columns: list[str], rows: list[Any], plan_stamp: Any
) -> tuple[dict[str, Any], bool]:
    """One settled run's peak statistics, plus whether they may be cached.

    Returns ``(payload, cacheable)``. ``cacheable`` is False only when the
    absence describes a *transient* failure of this process rather than a truth
    about the run — today, a plan catalog that raised — so it never sticks to a
    settled run; the figure route draws the same distinction for the same
    reason.

    Which channels the statistics are computed for comes from the plan the run
    was stamped with, read through the same `_role_channels` the default figure
    reads: the movable channel is the independent variable, the readable ones
    are what is fitted against it. A run with no usable stamp, or whose stamped
    name has no current owner in the catalog, has no declaration to read and
    comes back absent with `analysis.REASON_PLAN_IDENTITY_UNAVAILABLE` — the
    same word the figure route uses for the same situation.
    """
    plan_name = _stamp_name(plan_stamp)
    if plan_name is None:
        return analysis.absent(analysis.REASON_PLAN_IDENTITY_UNAVAILABLE), True

    try:
        from .plan_loader import get_facility_plans

        spec = get_facility_plans().plans.get(plan_name)
    except Exception:
        logger.warning(
            "analysis: plan catalog unavailable; this run's payload carries no statistics",
            exc_info=True,
        )
        return analysis.absent(analysis.REASON_STATISTICS_UNAVAILABLE), False
    if spec is None:
        return analysis.absent(analysis.REASON_PLAN_IDENTITY_UNAVAILABLE), True

    movable, readable = _role_channels(spec, plan_stamp.get("kwargs"))
    return analysis.analyze(columns, rows, movable, readable), True


def _run_analysis(
    *,
    run_id: str,
    columns: list[str],
    rows: list[Any],
    total_seen: int,
    partial: bool,
    plan_stamp: Any,
    source: str,
    generation: int,
) -> dict[str, Any]:
    """The ``analysis`` block `get_run_data` serves for one run. Never raises.

    Computed over the rows the SOURCE holds — every row, not the window the
    caller asked for, because statistics over a page of a run describe the
    page. A run still producing rows carries
    `analysis.REASON_RUN_IN_PROGRESS` and costs nothing: its peak would move
    with every poll, and settledness here is the source's own verdict, exactly
    as it is for the figure route.

    Settled results are held in `analysis`' own cache, keyed by the same
    per-run generation `figure_cache` compares against, so a polled settled run
    computes its statistics once.
    """
    if partial:
        return analysis.absent(analysis.REASON_RUN_IN_PROGRESS)

    hit = analysis.cached(run_id, source, generation, total_seen)
    if hit is not None:
        return hit

    payload, cacheable = _compute_analysis(columns, rows, plan_stamp)
    if cacheable:
        analysis.store(run_id, source, generation, total_seen, payload)
    return payload


def _data_response(
    held: Mapping[str, Any],
    *,
    run_id: str,
    run_uid: str | None,
    source: str,
    generation: int,
    max_rows: int,
    offset: int | None,
    tail: bool,
) -> dict[str, Any]:
    """`get_run_data`'s body, built from whichever source is holding the run.

    *held* is everything one source knows about the run — ``columns``,
    ``rows``, ``total_seen``, ``partial`` and the ``plan`` stamp — as the live
    buffer (`live_rows.get`) and the catalog snapshot (`_tiled_run_snapshot`)
    both spell it. That shared spelling is what lets one function shape both
    answers, and shaping both here is the point: the route's promise is that a
    run reads the same whichever source served it, and a promise kept by two
    copies of the same code is kept only until one of them is edited.

    Three rules the two sources must not diverge on:

    - ``run_uid`` is always present, carrying ``None`` when the source does not
      know it, so a caller never has to tell "unknown" from "omitted".
    - ``partial`` is set only when true. Its ABSENCE is how a caller reads
      "settled", so a source that spelled it ``partial: false`` would be
      answering a different question than the other one.
    - ``analysis`` is computed over ALL the rows the source holds, never the
      window `_window` cuts — statistics over a page of a run describe the
      page.

    *generation* is the caller's, read BEFORE the source, as the figure route
    reads it: an eviction landing between the two reads then leaves a cache
    entry that can never match again rather than one holding the previous
    occupant's peak.
    """
    result: dict[str, Any] = {"run_uid": run_uid}
    result.update(
        _window(held["columns"], held["rows"], held["total_seen"], max_rows, offset, tail)
    )
    if held["partial"]:
        result["partial"] = True
    result["analysis"] = _run_analysis(
        run_id=run_id,
        columns=held["columns"],
        rows=held["rows"],
        total_seen=held["total_seen"],
        partial=bool(held["partial"]),
        plan_stamp=held["plan"],
        source=source,
        generation=generation,
    )
    return result


def _from_tiled(
    run_id: str, max_rows: int, offset: int | None, tail: bool
) -> dict[str, Any] | None:
    """Serve `get_run_data` from Tiled once a run's live buffer is gone.

    A window over `_tiled_run_snapshot` and nothing else — the reading, the
    ordering and the ``partial`` verdict all belong to the shared snapshot, so
    the two Tiled-backed routes cannot answer differently about the same run.
    The body is then shaped by `_data_response`, the same function the live
    branch uses, which is what keeps the two sources' answers the same shape.
    """
    generation = figure_cache.snapshot_generation(run_id)
    snapshot = _tiled_run_snapshot(run_id)
    if snapshot is None:
        return None

    return _data_response(
        snapshot,
        run_id=run_id,
        run_uid=snapshot["run_uid"],
        source="tiled",
        generation=generation,
        max_rows=max_rows,
        offset=offset,
        tail=tail,
    )


@app.get("/runs/{run_id}/data")
def get_run_data(
    run_id: str, max_rows: int = 100, offset: int | None = None, tail: bool = False
) -> dict:
    """Read a bounded window of a run's recorded data — dual-source.

    Row-bounded by design — this never returns an unbounded table. Prefers the
    live-row buffer the document plane fills (see `live_rows.py`) whenever it
    has one: ``partial: true`` while the run is still filling in (before its
    stop doc lands), permanently readable once it's marked completed (see
    `live_rows.py`'s retention bound). ``row_count`` is the *true* total rows
    the run has produced so far, even if that's more than what's physically
    stored — ``truncated`` reflects whether this response's window omits any
    of them.

    ``run_id`` is looked up in the buffer directly, with no indirection through
    any record this process keeps. That one lookup covers both keyings by
    construction: the document plane buffers a queue-executed run under its
    OSPREY run id, while a run reaching a bare recorder with no such id is
    buffered under the RunEngine's own uid (`live_rows.LiveRowRecorder`), and
    either identity is a key a caller can pass here.

    Falls back to `_from_tiled` whenever there is no live buffer to serve. The
    fallback trigger is always ``buf is None`` — a present-but-empty buffer
    (``partial: true``, zero rows) is a real in-flight run and stays on the
    live path; checking "falsy rows" instead would incorrectly divert a running
    plan to Tiled before it has ever written anything there. A run still
    executing after a bridge restart legitimately has no buffer and is served
    from Tiled; that is correct, not a gap.

    Raises 404 when neither source has the run — the MCP `get_run_data` tool
    maps 404 to `unknown_run`, and a 200-empty response would make a
    nonexistent run look like a valid empty run. A deployment with no Tiled at
    all reaches that same 404 for a run it never buffered, which is honest: no
    source this bridge has knows the run.

    Raises 503 when Tiled IS configured and could not be reached. That is a
    different answer from the 404 on purpose, and the difference is the whole
    point: a 404 says the run does not exist, and saying that about a run whose
    catalog merely blinked would send a caller looking for a mistake it did not
    make. 503 says come back — the same line the export route draws between
    `tiled_unavailable` and its other refusals. The detail is a plain sentence,
    the shape every refusal on this route already uses, so the MCP tool renders
    it as `bluesky_bridge_error` with the sentence intact.

    ``run_uid`` is the RunEngine's own uid. Present on the Tiled path (it is on
    the stored start document) and ``None`` on the live path, where the bridge
    holds a buffer but no record of the uid the worker's RunEngine minted. The
    key is always present, so a consumer never has to tell "unknown" apart from
    "this response shape omits it".

    ``analysis`` is the run's peak statistics (`analysis.analyze`), computed
    over the whole run rather than this window and always present: a run that
    has none carries the reason it has none. See `_run_analysis`.
    """
    generation = figure_cache.snapshot_generation(run_id)
    buf = live_rows.get(run_id)

    if buf is not None:
        return _data_response(
            buf,
            run_id=run_id,
            run_uid=None,
            source="live",
            generation=generation,
            max_rows=max_rows,
            offset=offset,
            tail=tail,
        )

    try:
        tiled_result = _from_tiled(run_id, max_rows, offset, tail)
    except HTTPException:
        # Nothing raises one from in there today; passing it through keeps the
        # guard from ever swallowing a refusal that was deliberately named.
        raise
    except Exception as exc:
        # Every line of `_from_tiled`'s read reaches the Tiled server over the
        # network — `from_uri` performs a handshake GET, the search is a query,
        # and the stream membership test and part reads resolve against it — so
        # a catalog that is configured but down, restarting, behind a rotated
        # key, or simply slow raises from any of them. Without this the route
        # answers a routine outage with a bare 500 and the body `Internal
        # Server Error`.
        logger.warning(
            "data: the Tiled catalog could not be read for run %r", run_id, exc_info=True
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"The Tiled catalog is configured for this deployment but could not be "
                f"reached, so whether run {run_id!r} has stored data is unknown: {exc}"
            ),
        ) from exc

    if tiled_result is not None:
        return tiled_result

    raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")


# ---------------------------------------------------------------------------
# Run export
# ---------------------------------------------------------------------------

# The formats `GET /runs/{id}/export` offers, each mapped to the media type
# Tiled's own format GET is asked for. Both were verified against the pinned
# server image (`ghcr.io/bluesky/tiled:0.2.12`, the tag the compose template
# pins) rather than read off Tiled's serializer registry: the registry lists
# what the library *can* register, and the image's table family registers
# several entries — `application/x-hdf5` among them — whose GET answers 500.
# `application/json` and `application/json-seq` are registered and answer for a
# scalar-only table but 500 for one carrying a list-valued column, which any
# run with a waveform signal has. So this map holds only what the image serves
# for every table a run can produce: one text format and one binary one.
#
# Keys are the short aliases a caller passes as `?format=`; a media type is
# never accepted on the wire, so the set of answers this route can give is the
# set of keys here and cannot drift with whatever the server happens to
# register.
_EXPORT_FORMATS = {
    "csv": "text/csv",
    "parquet": "application/x-parquet",
}

# Why an export could not be produced. Same `{"code", "detail"}` refusal body
# every other refusal on this app uses (`queue.py`, `_use_the_queue`), so a
# caller branches on `detail.code` here exactly as it does there.
EXPORT_REASON_UNSUPPORTED_FORMAT = "unsupported_format"
EXPORT_REASON_TILED_UNAVAILABLE = "tiled_unavailable"
EXPORT_REASON_UNKNOWN_RUN = "unknown_run"
EXPORT_REASON_NOTHING_STORED = "nothing_stored"
EXPORT_REASON_AMBIGUOUS_TABLE = "ambiguous_table"
EXPORT_REASON_EXPORT_FAILED = "export_failed"


def _export_refused(status: int, code: str, detail: str) -> HTTPException:
    """The refusal an export answers with — always a body, never a bare status.

    A download route that fails has to say *why* in something a caller can
    branch on: the response body is the only place left, because the bytes it
    would otherwise be carrying are what is missing.
    """
    return HTTPException(status_code=status, detail={"code": code, "detail": detail})


# Everything a run id may contribute to a download filename. Deliberately a
# tiny allowlist rather than a blocklist of the characters that are known to
# hurt: the quoted-string form of `Content-Disposition` gives a `"` the power to
# close the parameter and start another, a bare CR or LF the power to end the
# header, and any byte outside latin-1 the power to raise inside the ASGI
# server's header encoder — and a blocklist is only ever as complete as the list
# of tricks its author had heard of.
_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _content_disposition(run_id: str, extension: str) -> str:
    """The download header for one run's export, built from a SANITIZED run id.

    The filename is a derivative of the run id, never the run id: every
    character outside ``[A-Za-z0-9._-]`` becomes an underscore before it reaches
    the header. `extension` needs no such treatment — it is only ever a key of
    `_EXPORT_FORMATS`, checked before this is called.

    OSPREY mints run ids as ``uuid4().hex`` (`queue.py`), which the allowlist
    passes through untouched, so this changes nothing for a run this deployment
    enqueued. It is written anyway because the catalog is SHARED infrastructure:
    anything else writing through `TiledWriter` — a facility script, a migration,
    a second tool — can stamp an arbitrary string as ``osprey_run_id``, and the
    moment one does, the id is untrusted input on its way into a response header.
    Safety that depends on an invariant held in another module is safety that
    lasts exactly until someone else writes to the same catalog, so the check
    lives here, at the point of use, where the dependency is visible.

    The RFC 5987 ``filename*`` form carries the id percent-encoded alongside the
    sanitized one, so a client that understands it saves the true id and one that
    does not still gets a valid, unambiguous name. Order matters: ``filename``
    comes first, because RFC 6266 has clients prefer ``filename*`` when both are
    present and the plain one is the fallback.
    """
    safe = _FILENAME_UNSAFE.sub("_", run_id)
    encoded = quote(f"{run_id}.{extension}", safe="")
    return f"attachment; filename=\"{safe}.{extension}\"; filename*=UTF-8''{encoded}"


def _resolve_stored_tables(run_id: str) -> list[Any]:
    """`_stored_table_node`'s catalog work: every read, none of the guarding.

    Split out so the guard around it can be one ``try`` over the whole
    resolution rather than a wrapper per call. Each step here reaches the
    catalog — building the client performs a handshake GET, the search is a
    query, and membership and part iteration both resolve against the server —
    so any of them can raise for a reason that has nothing to do with this run.
    Which of those raised is not a distinction a caller can act on; that
    judgement belongs to the caller of this function, which is why it makes
    none itself.

    The two ``None``/``not in`` refusals raised here are the opposite case:
    facts about the catalog's contents, learned by asking successfully.

    What comes back is every table-family part of the stream, in container
    order — not the one to export. `TiledWriter` writes exactly one of them
    (``internal``); naming the family rather than that key keeps the export
    reading whatever part actually holds the rows, exactly as
    `_row_part_columns` does on the read path. Deciding what a list of zero,
    one, or several means is `_stored_table_node`'s, not this function's.
    """
    client = _tiled_client()
    if client is None:
        raise _export_refused(
            503,
            EXPORT_REASON_TILED_UNAVAILABLE,
            "No Tiled catalog is configured for this deployment, so no run has "
            "stored data to export.",
        )

    run_node = _newest_run_node(client, run_id)
    if run_node is None:
        raise _export_refused(
            404, EXPORT_REASON_UNKNOWN_RUN, f"No run in the catalog matches {run_id!r}."
        )

    if "primary" not in run_node:
        raise _export_refused(
            409,
            EXPORT_REASON_NOTHING_STORED,
            f"Run {run_id!r} recorded no data — it has no 'primary' stream to export.",
        )

    return [
        part for _name, part in run_node["primary"].base.items() if part.structure_family == "table"
    ]


def _stored_table_node(run_id: str) -> Any:
    """The catalog node holding *run_id*'s stored rows. Raises a refusal, never a 500.

    Two kinds of failure, told apart because a caller acts on them differently:

    **The catalog answered, and its answer was "no".** No catalog configured, no
    run under that id, a run whose ``primary`` stream never landed, a stream with
    no table part — each a named refusal raised here or by
    `_resolve_stored_tables` and passed straight through.

    **The catalog did not answer at all.** Every line of the resolution reaches
    the Tiled server over the network — `from_uri` performs a handshake GET, the
    search is a query, and membership and part iteration both resolve against
    it — so a catalog that is configured but down, restarting, behind a rotated
    key, or simply slow raises from any of them. That is an ordinary operational
    state, not an exotic one, and it becomes 503 `tiled_unavailable` with the
    cause logged. The alternative is what this guard replaced: an unbranchable
    500 with a bare ``Internal Server Error`` body, on a route whose whole
    contract is that a caller can branch on ``detail.code``.

    503 rather than 502 for the outage, deliberately: 502 `export_failed` means
    "the catalog was reached and could not serialize this run", and collapsing
    the two would cost a caller the difference between "retry later" and "this
    run will never export". The figure route draws the same line one route over,
    wrapping `_tiled_run_snapshot` for exactly this reason.

    Only a table part is ever exported, never the run or the stream container:
    the pinned image's container family serves `application/json` alone, so a
    container-level format GET for CSV or parquet answers 406.

    Several table parts are refused rather than picked between, under their own
    `ambiguous_table` code — a caller told `nothing_stored` would act on the
    opposite of what happened, and this route's whole contract is that
    ``detail.code`` is branchable. `TiledWriter` writes one, so this cannot
    happen today — but the read path concatenates every table part's columns
    (`_row_part_columns`) while an export can serialize only one node, so
    silently taking the first would make the export a partial archive with
    nothing to say so, on the one route whose entire promise is completeness. A
    loud refusal on the day the assumption breaks is the point.
    """
    try:
        tables = _resolve_stored_tables(run_id)
    except HTTPException:
        # A refusal `_resolve_stored_tables` already named — the catalog answered.
        raise
    except Exception as exc:
        logger.warning(
            "export: the Tiled catalog could not be read for run %r", run_id, exc_info=True
        )
        raise _export_refused(
            503,
            EXPORT_REASON_TILED_UNAVAILABLE,
            f"The Tiled catalog could not be reached to look up this run: {exc}",
        ) from exc

    if not tables:
        raise _export_refused(
            409,
            EXPORT_REASON_NOTHING_STORED,
            f"Run {run_id!r} stored no row table, so there is nothing to serialize.",
        )
    if len(tables) > 1:
        raise _export_refused(
            409,
            EXPORT_REASON_AMBIGUOUS_TABLE,
            f"Run {run_id!r} stored {len(tables)} row tables; an export serializes one "
            "node, and serving part of an archive without saying so is not an option.",
        )
    return tables[0]


@app.get("/runs/{run_id}/export")
def export_run_data(run_id: str, format: str = "csv") -> Response:
    """Download one stored run's rows as a file. Serialized by Tiled, not here.

    The bytes come from the catalog's own format GET — this route resolves which
    node holds the run's rows, asks Tiled to serialize it, and relays the result.
    Nothing about the file's contents is decided in this process: no reformatting,
    no row limit, no column filter. A caller that wants the parsed, bounded,
    live-comparable view of a run reads `GET /runs/{id}/data`; this route is for
    getting the recorded run *out*, whole, into a tool that reads files.

    Two consequences of that division are worth stating, because they are the
    two ways this answer legitimately differs from the data route's:

    - **Every stored column is present**, including `seq_num`, `time`, and the
      per-signal `ts_*` timestamps that `_data_columns` projects away. Those
      columns exist in the archive and an export is the archive.
    - **Non-scalar signals survive.** A waveform is stored as a list-valued
      column, and Tiled serializes it — as a bracketed cell in CSV, as a native
      list column in parquet.

    ``format`` is a short alias, ``csv`` or ``parquet`` — the two the pinned
    server image serves for every table a run can produce (see
    `_EXPORT_FORMATS`). Anything else is refused by name rather than passed
    through, so an unsupported request is answered by this process with the list
    of what works instead of by the catalog with a 406.

    Total by construction, like the data and figure routes: every failure is a
    ``{"code", "detail"}`` body under a status that says which kind it is —
    400 `unsupported_format`, 404 `unknown_run`, 409 `nothing_stored`,
    409 `ambiguous_table`, 503 `tiled_unavailable`, 502 `export_failed`. The two
    409s are told apart because a caller acts on them differently — nothing to
    serialize versus more than one thing. The two network ones split the
    catalog's failures where a caller can act on the difference: 503 is "could
    not be reached at all, retry later" (`_stored_table_node` owns it, for the
    whole resolution), 502 is "was reached and could not serialize this run",
    which retrying will not fix. Both are catch-alls over an exception the
    catalog raised, because it is a separate process over a network and the ways
    that call can fail are not this module's to enumerate.

    Serving is unbounded by design and the payload is held in memory — more than
    once at peak, not once: Tiled's client reads the whole response body into
    `bytes` before writing it into this route's buffer, and the buffer is copied
    again on the way into the response. Roughly three times the file, briefly.
    That is affordable because a run is bounded, not a stream — the same bound the
    rest of this module relies on — but the multiplier is what to size against,
    not the file.

    ``tiled`` and its Bluesky client plugin stay imported below the routes
    (`_tiled_client`, `_newest_run_node`), so this route adds nothing to the
    module's import closure (`_BRIDGE_ONLY_MODULES`).
    """
    media_type = _EXPORT_FORMATS.get(format)
    if media_type is None:
        raise _export_refused(
            400,
            EXPORT_REASON_UNSUPPORTED_FORMAT,
            f"{format!r} is not an export format; this bridge serves "
            f"{', '.join(sorted(_EXPORT_FORMATS))}.",
        )

    node = _stored_table_node(run_id)

    buffer = io.BytesIO()
    try:
        node.export(buffer, format=media_type)
    except Exception as exc:
        logger.warning("export: serializing run %r as %s failed", run_id, format, exc_info=True)
        raise _export_refused(
            502,
            EXPORT_REASON_EXPORT_FAILED,
            f"The Tiled catalog could not serialize this run as {format}: {exc}",
        ) from exc

    return Response(
        content=buffer.getvalue(),
        media_type=media_type,
        headers={"Content-Disposition": _content_disposition(run_id, format)},
    )


# ---------------------------------------------------------------------------
# Run figures
# ---------------------------------------------------------------------------


class _CachedFigure(NamedTuple):
    """What the settled-figure cache stores: the served figure plus check data.

    `figure_cache` stores values opaquely, so the route wraps what it needs
    beside the figure: ``total_seen`` lets the live branch re-validate a hit
    against the buffer snapshot it already holds (a recycled run id whose
    eviction has not landed yet shows up as a row-count mismatch — a free
    check, costing no extra source read), and ``run_uid`` is carried on Tiled
    entries for logging only.
    """

    figure: Figure
    total_seen: int
    run_uid: str | None


def _decimate_figure(figure: Figure) -> Figure:
    """Bound every lines/bars series to `DEFAULT_MAX_POINTS`, truthfully flagged.

    Mutates *figure* in place and returns it. Only marks that actually exceed
    the budget are touched — an already-bounded mark keeps whatever
    ``decimated``/``source_points`` truth its author set, and a touched mark's
    aggregates are recomputed from its per-series truth. `BarsMark` carries no
    `Point` list, so its values ride through `decimate` as index-valued points
    and the kept indices select the surviving category/value pairs — same
    stride, same endpoint rule, and a ``None`` value survives as a gap.
    """
    for panel in figure.panels:
        mark = panel.mark
        if isinstance(mark, LinesMark):
            if not any(len(series.points) > DEFAULT_MAX_POINTS for series in mark.series):
                continue
            mark.series = [
                series
                if len(series.points) <= DEFAULT_MAX_POINTS
                else Series(
                    label=series.label,
                    **decimate(
                        series.points,
                        source_points=series.source_points,
                        decimated=series.decimated,
                    )._asdict(),
                )
                for series in mark.series
            ]
            mark.decimated = any(series.decimated for series in mark.series)
            mark.source_points = sum(series.source_points for series in mark.series)
        elif isinstance(mark, BarsMark) and len(mark.values) > DEFAULT_MAX_POINTS:
            indexed = [Point(x=float(i), y=value) for i, value in enumerate(mark.values)]
            thinned = decimate(indexed, source_points=mark.source_points, decimated=mark.decimated)
            kept = [int(point.x) for point in thinned.points]
            mark.categories = [mark.categories[i] for i in kept]
            mark.values = [mark.values[i] for i in kept]
            mark.decimated = thinned.decimated
            mark.source_points = thinned.source_points
    return figure


def _stamp_name(plan_stamp: Any) -> str | None:
    """The stamp's plan name, or ``None`` unless it is an actual string.

    The stamp is opaque data off a document: a malformed one (non-dict, or a
    name that is a list/number) must degrade to `plan_identity_unavailable`
    like any other unusable identity — never reach a catalog lookup or a cache
    key, where a non-str, possibly unhashable name would raise.
    """
    if not isinstance(plan_stamp, dict):
        return None
    name = plan_stamp.get("name")
    return name if isinstance(name, str) else None


def _role_channels(spec: PlanSpec[Any], params: Any) -> tuple[list[str], list[str]]:
    """The movable and the readable channels *params* supplies, as two lists.

    `_declared_channels`' counterpart for the read routes, which need the two
    roles apart rather than interleaved and labelled: the movable channels pick
    the default figure's x axis and the run's independent variable for
    `analysis`, the readable ones order the figure's panels and are what the
    statistics are computed for. Same single authority — `PlanSpec.roles` says
    whether the plan declared anything at all, `plan_fields.collect_channels`
    says which names the parameters put in the declared fields.

    Best-effort and never raises. Role context only chooses an x axis, so a run
    whose stamped parameters no longer validate — or a schema whose walk goes
    wrong — must still get its row-index figure rather than turn a route's
    promised 200 into a 500. Parameters are validated first when they still can
    be, so a channel supplied by a field default rather than by the stamp is
    seen; a stamp that no longer validates is walked exactly as it was stored.
    """
    if not spec.roles:
        return [], []
    try:
        walked: Any = params
        with suppress(ValidationError):
            walked = spec.schema.model_validate({} if params is None else params)
        return (
            collect_channels(spec.schema, walked, MOVABLE_ROLE),
            collect_channels(spec.schema, walked, READABLE_ROLE),
        )
    except Exception:
        logger.warning(
            "could not read plan %r's declared channels; the default figure falls back to "
            "row index and the run's payload carries no statistics",
            spec.name,
            exc_info=True,
        )
        return [], []


def _compose_figure(
    *,
    columns: list[str],
    rows: list[Any],
    total_seen: int,
    plan_stamp: Any,
    partial: bool,
    source: Literal["live", "tiled"],
) -> tuple[Figure, bool]:
    """One source snapshot in, one servable figure out — total, never raises.

    Returns ``(figure, cacheable)``. ``cacheable`` is False only when the
    figure describes a *transient* failure of this process rather than a truth
    about the run — today, a plan catalog that raised — so the route must not
    stick it to a settled run; every other outcome, genuine no-owner lookups
    included, caches normally.

    Every fallback is the default figure carrying its machine-readable
    ``reason``; ``partial`` and ``source`` are stamped from the SOURCE's truth
    on every path, plan-rendered figures included — a `render` is handed the
    `RowWindow`, which says how much of the run its rows are but not which
    store they came from or whether it is still producing them, so both of a
    rendered figure's own values are placeholders by convention (see
    `plans_core/orm.py`).

    Every default figure built once the run's plan is known is handed that
    plan's declared channels (`_role_channels`), so the stand-in view plots
    against the run's own movable rather than against row index. Reading the
    declaration cannot change which reason is served: the ladder is walked in
    the same order either way, and a declaration that cannot be read at all
    degrades to no role context — the row-index figure — never to a failure.

    The reason ladder, in the order it is walked:

    - ``plan_identity_unavailable`` — no plan stamp on the source, a stamp
      with no name (a run enqueued before stamping existed, or driven directly
      at the worker), or a malformed stamp whose name is not a string. The
      stamp is the ONLY identity source; the run record is never consulted.
    - ``no_render`` — the stamped name has no current owner in the plan
      catalog (removed or renamed since the run), its spec carries no
      ``render``, or the catalog itself failed to load (logged; the one
      non-cacheable case). Figures are always computed by the plan code
      *currently* owning the name.
    - ``render_not_supported_for_session_plans`` — the name resolves to a
      session/unreviewed spec. Decided on provenance, not on ``render is
      None``, so the authoring agent learns *why* rather than seeing a plain
      ``no_render``.
    - ``params_mismatch`` — the stamped kwargs no longer validate against the
      plan's current schema (schema drift since the run), or the source
      snapshot itself is structurally broken (`rows_from_columnar` refuses a
      window claiming more rows than its run, or ragged rows — same treatment:
      the stored shape no longer matches what the code expects).
    - ``render_failed`` — the plan's `render` raised, returned a non-`Figure`,
      or produced marks whose decimation truth is inconsistent.
    """
    try:
        window = rows_from_columnar(columns, rows, total_seen)
    except ValueError:
        logger.warning("figure: malformed row snapshot; serving the default figure", exc_info=True)
        window = RowWindow(
            rows=[], columns=list(columns), rows_complete=False, total_seen=max(total_seen, 0)
        )
        figure = _decimate_figure(
            default_figure(window, reason=REASON_PARAMS_MISMATCH, partial=partial, source=source)
        )
        return figure, True

    def _default(reason: str, roles: tuple[Sequence[str], Sequence[str]] = ((), ())) -> Figure:
        movable, readable = roles
        return _decimate_figure(
            default_figure(
                window,
                reason=reason,
                partial=partial,
                source=source,
                movable=movable,
                readable=readable,
            )
        )

    plan_name = _stamp_name(plan_stamp)
    if plan_name is None:
        return _default(REASON_PLAN_IDENTITY_UNAVAILABLE), True

    try:
        from .plan_loader import get_facility_plans

        spec = get_facility_plans().plans.get(plan_name)
    except Exception:
        # A failing catalog is this process's transient trouble, not a truth
        # about the run — hence cacheable=False, so it never sticks to a
        # settled run the way a genuine no-owner lookup legitimately would.
        logger.warning(
            "figure: plan catalog unavailable; serving the default figure", exc_info=True
        )
        return _default(REASON_NO_RENDER), False
    if spec is None:
        return _default(REASON_NO_RENDER), True

    # Every path below has a spec, so every default figure below is role-aware:
    # a session plan's view and a mismatched-params view get the run's own x
    # axis exactly as a shipped plan's does. The declaration is read off the
    # stamped kwargs, which is all a run that never rendered ever supplies.
    kwargs = plan_stamp.get("kwargs")
    roles = _role_channels(spec, kwargs)

    if spec.provenance in ("session", "unreviewed"):
        return _default(REASON_RENDER_NOT_SUPPORTED_FOR_SESSION_PLANS, roles), True
    if spec.render is None:
        return _default(REASON_NO_RENDER, roles), True

    try:
        params = spec.schema.model_validate({} if kwargs is None else kwargs)
    except ValidationError:
        return _default(REASON_PARAMS_MISMATCH, roles), True

    try:
        rendered = spec.render(window, params)
        if not isinstance(rendered, Figure):
            raise TypeError(f"render returned {type(rendered).__name__}, not a Figure")
        figure = _decimate_figure(
            rendered.model_copy(update={"partial": partial, "source": source})
        )
        return figure, True
    except Exception:
        logger.warning(
            "figure: plan %r render failed; serving the default figure",
            plan_name,
            exc_info=True,
        )
        return _default(REASON_RENDER_FAILED, roles), True


@app.get("/runs/{run_id}/figure")
def get_run_figure(run_id: str) -> dict:
    """A run's figure: its plan's own view of the rows, or the default view.

    Total over known runs — every run a DATA source knows has a figure, and
    every degradation is a 200 default figure with a machine-readable
    ``reason`` (this route is polled at 1 Hz; a flaky catalog must never
    become a 500). ``partial`` and ``source`` are always present. 404 only
    when neither the live buffer nor Tiled knows the run — byte-for-byte
    `get_run_data`'s 404; the run RECORD is never consulted anywhere here (it
    would need an awaited manager RPC from this sync def, could raise on a
    never-non-200 route, and would make /figure and /data disagree about a
    pending run).

    Settledness comes from the SOURCE only — the live buffer's ``partial``
    flag, or the stop document's presence in Tiled — never from the run
    record, whose terminal status can precede the last rows and survives a
    re-run of an interrupted item under the same run id. Settled figures are
    cached (`figure_cache`) so a second GET on a settled Tiled run issues zero
    Tiled calls; recycled run ids are handled by the document plane's
    invalidate-on-write eviction plus the generation compare-and-set on store.
    Live runs recompute per tick — ~50 ms end-to-end at facility scale, of
    which the vectorized fit is ~15 ms.

    A plain sync ``def`` (threadpool, like `get_run_data`) — no awaits, no
    single-flight; losers of a cache race simply recompute.
    """
    # Generation before the source snapshot: if an eviction (a new start doc
    # for this id) lands between the two reads, the CAS put below must lose.
    # Read the other way around, a snapshot of the OLD rows could be stored
    # under the NEW generation — a poisoned settled entry.
    generation = figure_cache.snapshot_generation(run_id)

    buf = live_rows.get(run_id)
    if buf is not None:
        plan_stamp = buf["plan"]
        partial = bool(buf["partial"])
        key = figure_cache.make_key(run_id, _stamp_name(plan_stamp), "live")
        if not partial:
            cached = figure_cache.get(key)
            if isinstance(cached, _CachedFigure) and cached.total_seen == buf["total_seen"]:
                return cached.figure.model_dump()
        figure, cacheable = _compose_figure(
            columns=buf["columns"],
            rows=buf["rows"],
            total_seen=buf["total_seen"],
            plan_stamp=plan_stamp,
            partial=partial,
            source="live",
        )
        if not partial and cacheable:
            figure_cache.put(key, _CachedFigure(figure, buf["total_seen"], None), generation)
        return figure.model_dump()

    cached = figure_cache.get_for_source(run_id, "tiled")
    if isinstance(cached, _CachedFigure):
        logger.debug(
            "figure: settled Tiled cache hit for run %r (run_uid=%s)", run_id, cached.run_uid
        )
        return cached.figure.model_dump()

    try:
        snapshot = _tiled_run_snapshot(run_id)
    except Exception:
        # `partial=True` because settledness is unknowable without the source:
        # the panel keeps its last good figure and keeps polling, which is the
        # behavior that recovers when the catalog does.
        logger.warning(
            "figure: Tiled read failed for run %r; serving the default figure",
            run_id,
            exc_info=True,
        )
        window = RowWindow(rows=[], columns=[], rows_complete=False, total_seen=0)
        return default_figure(
            window, reason=REASON_SOURCE_UNAVAILABLE, partial=True, source="tiled"
        ).model_dump()

    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")

    plan_stamp = snapshot["plan"]
    figure, cacheable = _compose_figure(
        columns=snapshot["columns"],
        rows=snapshot["rows"],
        total_seen=snapshot["total_seen"],
        plan_stamp=plan_stamp,
        partial=snapshot["partial"],
        source="tiled",
    )
    if not snapshot["partial"] and cacheable:
        figure_cache.put(
            figure_cache.make_key(run_id, _stamp_name(plan_stamp), "tiled"),
            _CachedFigure(figure, snapshot["total_seen"], snapshot["run_uid"]),
            generation,
        )
    return figure.model_dump()
