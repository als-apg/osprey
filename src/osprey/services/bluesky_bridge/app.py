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

import ast
import asyncio
import json
import logging
import os
import re
from collections.abc import Sequence
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import ValidationError

from . import document_plane, draft, figure_cache, live_rows, queue, runs
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
_BRIDGE_ONLY_MODULES = {"bluesky", "ophyd", "ophyd_async", "tiled"}

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
      unconditionally here, never behind a wiring flag such as
      `BLUESKY_EPICS_SUBSTRATE`: the posture it guards against is a property of
      the project config, not of how the bridge happens to be wired, and a
      gated guard leaves whole classes of deployment unchecked.
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
    """
    from .queue_backend import REASON_MANAGER_UNREACHABLE

    try:
        return (await get_queue_backend().capability()).to_dict()
    except Exception as exc:
        logger.warning("capability probe failed; reporting cannot-execute: %s", exc)
        return {
            "can_execute": False,
            "reason": REASON_MANAGER_UNREACHABLE,
            "detail": f"The bridge could not determine whether plans can execute: {exc}",
        }


@app.get("/health")
async def health() -> dict:
    """Liveness, plus whether this deployment can actually execute plans.

    The capability record rides on the existing status surface rather than a
    route of its own, so no consumer has to learn a second endpoint to find out
    what the bridge in front of it can do:

    ``{"status": "ok", "capability": {"can_execute", "reason", "detail"}}``

    ``can_execute`` is the answer; ``reason`` is one of the machine-readable
    ``REASON_*`` codes in `queue_backend.py`, which panels and MCP tools branch
    on; ``detail`` is the operator-facing sentence, and for a browse-only mock
    deployment it names the exact command that flips it. ``status: "ok"`` means
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
    """Registered scan plans: `plan_loader.get_facility_plans()`'s trust-resolved set.

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
# setpoint or only read as a detector. Absent keys stay absent rather than
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
    drivable setpoint from a read-only detector.
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


# ---------------------------------------------------------------------------
# Plan source rendering: backs the launch-approval hook's
# human-legible plan excerpt — the human backstop for the plan validator's
# documented, accepted obfuscation residual (see `plan_validation.py`'s
# module docstring). Read-only: never execs anything, only reads file text
# already sitting on disk.
# ---------------------------------------------------------------------------

_SOURCE_TRUNCATE_CHARS = 4000  # default: a few KB, enough for a human skim
# Hard ceiling for an explicit `max_chars` ask (the plan panel's Source tab
# requests full source this way). Far above any real plan file, but still a
# bound — the response can never grow unbounded with the file.
_SOURCE_TRUNCATE_CHARS_MAX = 200_000


def _find_layer_source_path(name: str) -> tuple[Any, Provenance] | None:
    """Best-effort locate the on-disk file behind a shipped/preset/facility plan.

    Directory-layer files are keyed by their declared ``PLAN_METADATA["name"]``,
    not necessarily their filename — so this parses each candidate file's
    source with ``ast`` (never execs it) purely to read the literal ``name``
    off its ``PLAN_METADATA`` dict. Returns `None` for a plan with no backing
    file at all, or a name this scan can't locate; the route degrades to a
    404 either way.
    """
    from .plan_loader import _iter_plan_files, _resolve_plan_dir_layers

    for directory, provenance in _resolve_plan_dir_layers():
        if provenance == "session":
            continue
        for path in _iter_plan_files(directory):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                if not any(
                    isinstance(target, ast.Name) and target.id == "PLAN_METADATA"
                    for target in node.targets
                ):
                    continue
                if not isinstance(node.value, ast.Dict):
                    continue
                for key, value in zip(node.value.keys, node.value.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "name"
                        and isinstance(value, ast.Constant)
                        and value.value == name
                    ):
                        return path, provenance
    return None


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
    """The catalog node for *run_id* — the newest ``start.time`` when several match.

    A re-run of an interrupted item records a second start document under the
    same OSPREY run id, so the search can legitimately return several nodes;
    ``matches[0]`` would leave the choice to whatever order the server answers
    in. The newest start is the run the id currently means — the same "latest
    occupant wins" rule the live buffer applies by overwriting on start.

    The start doc `TiledWriter.start` records lives under ``metadata["start"]``
    on the run container — a bare ``Key("osprey_run_id")`` matches nothing.
    """
    from tiled.queries import Key

    matches = list(client.search(Key("start.osprey_run_id") == run_id).values())
    if not matches:
        return None

    def _start_time(node: Any) -> float:
        time = dict(node.metadata).get("start", {}).get("time")
        return time if isinstance(time, int | float) else float("-inf")

    return max(matches, key=_start_time)


def _data_columns(table: Any) -> list[str]:
    """The stored table's data columns, projected onto the live buffer's set.

    Tiled's stored rows carry ``seq_num``, ``time``, and per-signal ``ts_*``
    timestamp columns the live buffer never had (see `LiveRowRecorder`). Both
    Tiled readers — `_from_tiled` for the data route, `_figure_source_from_tiled`
    for the figure route — project them away HERE, so "a run replayed from
    Tiled has the identical column set it had live" is one rule in one place
    rather than two filters that happen to agree.
    """
    return [c for c in table.columns if c != "seq_num" and c != "time" and not c.startswith("ts_")]


def _from_tiled(
    run_id: str, max_rows: int, offset: int | None, tail: bool
) -> dict[str, Any] | None:
    """Serve `get_run_data` from the durable Tiled catalog once a run's live buffer is gone.

    Two situations fall through the live path in `get_run_data` and land here: a run with
    no live buffer at all (a bridge restart drops every buffer, so a run that started
    before it — even one still executing — has nothing in memory), and one whose buffer was
    evicted past `live_rows._MAX_RUNS`. The search keys on `osprey_run_id`, the durable
    stamp the enqueue path threads into the item metadata and the worker records onto the
    start document.

    Returns `None` when Tiled is unconfigured (`BLUESKY_TILED_URI` unset — logged, not an
    error) or when no run in the catalog matches `run_id`; the caller turns either into a
    404.

    `tiled` is imported here, never at module level, so `app.py` stays import-clean of it
    (`_BRIDGE_ONLY_MODULES`) even when Tiled *is* configured for this deploy.
    """
    client = _tiled_client()
    if client is None:
        return None
    run_node = _newest_run_node(client, run_id)
    if run_node is None:
        return None

    run_uid = dict(run_node.metadata).get("start", {}).get("uid")

    if "primary" not in run_node:
        # Start doc landed but no Event ever arrived (e.g. a scan that
        # errored before its first point) — the run is real, so this is the
        # "nothing to read yet" shape, not a 404. Deliberately a membership
        # check on `"primary"` alone, never a `try`/`except KeyError` around
        # the whole traversal below: `CompositeClient.__getitem__` raises
        # `KeyError` for `"internal"` too (it exposes the table's *columns*,
        # not the table), so a broad guard here would silently convert a
        # wrong traversal into this empty-but-successful answer.
        columns: list[str] = []
        rows: list[Any] = []
        total_seen = 0
    else:
        # `run_node["primary"]` is a `CompositeClient`, whose keys are the
        # flattened column names; the appendable table itself hangs off its
        # `.base` container.
        internal_table = run_node["primary"].base["internal"]
        table = internal_table.read()
        columns = _data_columns(table)
        rows = table[columns].values.tolist()
        total_seen = len(table)

    result: dict[str, Any] = {"run_uid": run_uid}
    result.update(_window(columns, rows, total_seen, max_rows, offset, tail))
    return result


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
    nonexistent run look like a valid empty scan.

    ``run_uid`` is the RunEngine's own uid. Present on the Tiled path (it is on
    the stored start document) and ``None`` on the live path, where the bridge
    holds a buffer but no record of the uid the worker's RunEngine minted. The
    key is always present, so a consumer never has to tell "unknown" apart from
    "this response shape omits it".
    """
    buf = live_rows.get(run_id)

    if buf is not None:
        result: dict[str, Any] = {"run_uid": None}
        result.update(
            _window(buf["columns"], buf["rows"], buf["total_seen"], max_rows, offset, tail)
        )
        if buf["partial"]:
            result["partial"] = True
        return result

    tiled_result = _from_tiled(run_id, max_rows, offset, tail)
    if tiled_result is not None:
        return tiled_result

    raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")


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


def _figure_source_from_tiled(run_id: str) -> dict[str, Any] | None:
    """Read a run's FULL stored table from Tiled for the figure route.

    Unlike `_from_tiled`, which windows rows for `get_run_data`, this returns
    every stored row: a figure is computed over the whole run (a windowed ORM
    fit is silently wrong), sorted by ``seq_num`` so rows are in emission order
    however the catalog returns them, then projected onto the live-buffer
    column set exactly as the data path is.

    The figure route also needs the run's story, not just its rows: the plan
    stamp the enqueue path recorded onto the start document, and whether the
    run has settled — decided by the stop document's presence, the same signal
    ``partial`` means everywhere, never the run record.

    Returns ``None`` when Tiled is unconfigured or no run matches; the caller
    turns either into `get_run_data`'s exact 404.
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
        # Start doc landed but no Event ever arrived — a real, empty run. See
        # `_from_tiled` for why this is a membership check on `"primary"` and
        # never a broad `try`/`except KeyError` around the traversal.
        columns: list[str] = []
        rows: list[Any] = []
        total_seen = 0
    else:
        table = run_node["primary"].base["internal"].read()
        if "seq_num" in table.columns:
            table = table.sort_values("seq_num", kind="stable")
        columns = _data_columns(table)
        rows = table[columns].values.tolist()
        total_seen = len(table)

    return {
        "columns": columns,
        "rows": rows,
        "total_seen": total_seen,
        "partial": metadata.get("stop") is None,
        "plan": start.get(PLAN_META_KEY),
        "run_uid": start.get("uid"),
    }


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
        snapshot = _figure_source_from_tiled(run_id)
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
