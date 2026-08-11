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
import logging
import os
import re
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Query

from . import document_plane, draft, live_rows, queue, runs
from .models import (
    PlanSessionWriteRequest,
    PlanValidateRequest,
)
from .plan_types import Provenance
from .plan_validation import hash_plan_body, validate_plan
from .queue_backend import QueueBackendError
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
# for all of them: the answer is always "this capability moved to the queue",
# and the per-route sentence in `detail` says which route took it over.
_USE_THE_QUEUE = "use_the_queue"


def _use_the_queue(detail: str) -> HTTPException:
    """The machine-readable refusal a retired direct-execute route raises.

    Same ``{"code", "detail"}`` body shape as every queue refusal
    (``queue.py``), so a caller branches on ``detail.code`` uniformly rather
    than special-casing these four routes. 410 Gone, not 404: the route is
    still here and still answering — what is gone is the in-process execution
    it used to perform.
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
      unconditionally here. It used to sit inside the retired EPICS-substrate
      runner branch, which meant a deployment that never set
      `BLUESKY_EPICS_SUBSTRATE` was never checked at all; the posture it guards
      against is a property of the project config, not of how the bridge
      happens to be wired, so gating it on a wiring flag was always the wrong
      shape.
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
    """Retired: the bridge no longer mints launch intents of its own.

    A run id used to be minted here and launched in a second call. Both halves
    now belong to `POST /queue/items`, which mints the id AND enqueues in one
    armed, race-checked step — there is no intermediate state left for this
    route to create.
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
    queue. The two-step mint-then-launch flow this route completed collapsed
    into `POST /queue/items` (enqueue) plus the token-gated `POST /queue/start`
    (arm and drain).
    """
    raise _use_the_queue(
        "Plans now execute in the queue server, not in the bridge. Enqueue the "
        "draft with POST /queue/items, then arm the queue with POST /queue/start."
    )


@app.post("/draft/run")
def launch_draft_run() -> dict:
    """Retired: launching the shared draft directly is gone.

    `POST /queue/items` is the replacement and keeps everything this route
    guaranteed — the pinned `draft_revision` and the reservation that stops two
    callers racing the same revision onto hardware — while putting the plan in a
    durable, serialized queue instead of a thread in this process. The launch
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
    """Retired: stopping is a queue operation now.

    This route could only ever stop a plan running inside the bridge process,
    and none do. There are two replacements, and which one a caller wants
    depends on what they need stopped: `POST /queue/stop` halts the queue AFTER
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
    `PLAN_METADATA = {...}` block followed by the author's own ``body`` — and
    writes exactly that string to ``resolve_session_plan_dir()/<name>.py``,
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
        "category": request.category,
        "required_devices": list(request.required_devices),
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
    uri = os.environ.get(_TILED_URI_ENV)
    if not uri:
        logger.info(
            "_from_tiled: %s is unset; Tiled is not configured for this deploy", _TILED_URI_ENV
        )
        return None

    from tiled.client import from_uri
    from tiled.queries import Key

    client = from_uri(uri, api_key=os.environ[_TILED_API_KEY_ENV])
    # The start doc `TiledWriter.start` records lives under `metadata["start"]`
    # on the run container — a bare `Key("osprey_run_id")` matches nothing.
    matches = list(client.search(Key("start.osprey_run_id") == run_id).values())
    if not matches:
        return None
    run_node = matches[0]

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
        # Tiled's stored rows carry `seq_num`, `time`, and per-signal `ts_*`
        # timestamp columns the live buffer never had (see `LiveRowRecorder`)
        # — project those away so both sources return the identical column set.
        columns = [
            c for c in table.columns if c != "seq_num" and c != "time" and not c.startswith("ts_")
        ]
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
