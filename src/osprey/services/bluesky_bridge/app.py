"""The Bluesky bridge's FastAPI app: wires the run registry, launch gate, and
runner seam (``runs.py``, ``security.py``, ``plan_runner.py``) into HTTP routes.

Two processes, one machine (see PLAN.md's Technical Architecture): this app
runs in a separate container from OSPREY's own venv, reachable only over
HTTP plus the ``X-Launch-Token`` header. It stays import-clean of bluesky/
ophyd/tiled in Phase 1 — ``_runner_factory`` defaults to the no-op
``FakePlanRunner`` so this app is runnable and manually smoke-testable
(``GET /health``, even a real ``launch``) before the bluesky-backed
``PlanRunner`` exists. Real wiring swaps the factory via ``set_runner_factory``:
either a facility's own deploy code, or this module's own opt-in
``_lifespan`` hook — real EPICS devices (``BLUESKY_EPICS_SUBSTRATE``) or the
built-in deploy smoke demo (``BLUESKY_DEMO_RUNNER``) — see below.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Header, HTTPException, Query

from . import draft, live_rows
from .models import (
    DraftRunRequest,
    PlanSessionWriteRequest,
    PlanValidateRequest,
    RunRequest,
)
from .plan_runner import FakePlanRunner, PlanRunner
from .plan_types import Provenance
from .plan_validation import hash_plan_body, validate_plan
from .runner_wiring import (
    _BRIDGE_ONLY_MODULES,
    _DEMO_RUNNER_ENV,
    _EPICS_LIKE_CONNECTOR_TYPES,
    _EPICS_SUBSTRATE_ENV,
    _TILED_API_KEY_ENV,
    _TILED_URI_ENV,
    _build_tiled_writer_factory,
    _is_demo_runner_enabled,
    _is_epics_substrate_enabled,
    _resolve_control_system_type,
)
from .runs import Run, do_launch, registry
from .security import verify_launch_token
from .session_dir import resolve_session_plan_dir
from .validation import (
    _assert_limits_readable_if_writable,
    _launch_validation_gate,
    _request_field,
    _validate_launchable_request,
)
from .validation_record import validation_records

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger("osprey.services.bluesky_bridge.app")

# The `PlanRunner` implementation `do_launch` builds for every launch.
_runner_factory: Callable[[], PlanRunner] = FakePlanRunner

# Task 2.1: the bridge's single long-lived OSPREY connector — one Channel
# Access client for the process's whole lifetime, constructed by `_lifespan`
# only when `_is_epics_substrate_enabled()`, and disconnected exactly once on
# shutdown. `None` whenever the EPICS substrate isn't enabled, or before
# `_lifespan` has constructed it, or after shutdown has torn it down. Wired
# into the runner factory by task 2.2's `_epics_runner_factory` closure,
# which builds every scan device against this one connector.
_connector: Any | None = None


def get_connector() -> Any | None:
    """The bridge's single long-lived OSPREY connector, or `None` if unset.

    `None` whenever the EPICS substrate isn't enabled, before `_lifespan` has
    constructed it, or after shutdown has disconnected it. A later task's
    `_epics_runner_factory` closure reads this to build connector-backed
    devices.
    """
    return _connector


def set_runner_factory(factory: Callable[[], PlanRunner]) -> None:
    """Override the `PlanRunner` implementation `do_launch` builds.

    A real bluesky-backed factory (`_lifespan` below, or a facility's own
    deploy wiring) calls this instead of reaching into the private module
    global directly.
    """
    global _runner_factory
    _runner_factory = factory


async def _wire_epics_substrate_runner() -> Any | None:
    """Wire the real EPICS-substrate ``BlueskyPlanRunner`` (tasks 2.1/2.2/3.1/3.4).

    The body of ``_lifespan``'s EPICS-substrate branch: guard-import the
    bluesky extra, run the fail-OPEN limits preflight, construct and connect
    the bridge's single long-lived OSPREY connector, and register a runner
    factory whose devices are all connector-mediated.

    Returns the connected connector for ``_lifespan`` to store in the module
    global ``_connector`` (so shutdown can disconnect it exactly once), or
    ``None`` when the bluesky stack is not importable — the
    ``FakePlanRunner`` fallback, mirroring ``list_plans``'s guarded/lazy-import
    pattern.
    """
    try:
        from .devices import connector as connector_devices
        from .devices._specs_from_env import specs_from_env
        from .plan_runner_bluesky import BlueskyPlanRunner
    except ImportError as exc:
        root_name = (getattr(exc, "name", None) or "").split(".")[0]
        if root_name not in _BRIDGE_ONLY_MODULES:
            raise
        logger.warning(
            "%s is enabled but the bluesky stack is not importable "
            "(%s not found); falling back to FakePlanRunner",
            _EPICS_SUBSTRATE_ENV,
            exc.name,
        )
        return None

    # Task 3.1: fail-OPEN startup guard, before any connector/CA work
    # begins — refuses startup only for the one unsafe combination
    # (writable + limits checking enabled + limits database
    # unreadable). See `_assert_limits_readable_if_writable`'s
    # docstring for the full condition and why every other
    # combination starts normally.
    _assert_limits_readable_if_writable()

    # Task 3.4: construct the single long-lived OSPREY connector this
    # bridge holds for its whole process lifetime, built from the
    # project's `control_system.type` (Connector = the single
    # control-system interface) rather than a hardcoded
    # `virtual_accelerator` — one config line now flips the whole
    # Bluesky stack between the mock connector and real Channel Access.
    # `osprey.connectors.factory` and `epics_connector` are
    # import-safe even in a base install (pyepics is imported lazily
    # inside `EPICSConnector.connect()`), but the import stays inside
    # this already-guarded path regardless.
    #
    # For the EPICS-like types (`virtual_accelerator`/`epics`), the
    # `type_config` stays gateway-less (no "gateways" key) exactly as
    # before — this makes `connect()` skip the block that sets
    # process-wide `EPICS_CA_*` env, so the compose-inherited
    # `EPICS_CA_NAME_SERVERS` (pointing at the virtual accelerator or
    # real hardware) survives untouched (FR8/CF-1) — and it needs no
    # running CA server, so this is safe to do unconditionally at
    # startup. `control_system.type: virtual_accelerator` therefore
    # yields the exact same `type_config` this branch always built.
    #
    # For `mock`, the connector-mediated devices below (built by
    # `connector_devices.build_devices` from the real corrector/BPM
    # channel names) construct fine against the mock connector, but a
    # scan will NOT complete on it: the mock connector accepts writes,
    # yet its readbacks simulate a non-tracking base value rather than
    # tracking the setpoint, so a settle-verified corrector move
    # (`ConnectorSettable.set`) never sees its target and times out.
    # This is intentional — mock mode is for browsing/UI only; running
    # an actual scan requires a setpoint-tracking control system
    # (`virtual_accelerator` or `epics`), selected via
    # `control_system.type`.
    from osprey.connectors.factory import (
        ConnectorFactory,
        register_builtin_connectors,
    )

    control_system_type = _resolve_control_system_type()
    if control_system_type in _EPICS_LIKE_CONNECTOR_TYPES:
        connector_type_config: dict[str, Any] = {
            "type": control_system_type,
            "connector": {control_system_type: {"timeout": 5.0}},
        }
    else:
        # "mock" (the fail-safe default), or any other resolved type
        # the bridge doesn't special-case: forward the type name
        # through with no type-specific config (mock needs none) so
        # an unrecognized value surfaces as `ConnectorFactory`'s own
        # clear "Unknown control system type" error rather than being
        # silently mis-wired to a connector the operator didn't ask for.
        connector_type_config = {
            "type": control_system_type,
            "connector": {control_system_type: {}},
        }

    register_builtin_connectors()  # idempotent (CF-3); must run before create
    connector = await ConnectorFactory.create_control_system_connector(connector_type_config)
    logger.info(
        "%s is enabled: connected the bridge's single long-lived OSPREY connector "
        "(control_system.type=%s, %s)",
        _EPICS_SUBSTRATE_ENV,
        control_system_type,
        type(connector).__name__,
    )

    motors, detectors = specs_from_env(os.environ)

    def _epics_runner_factory() -> BlueskyPlanRunner:
        # `connector_devices.build_devices` is `async def` (it builds
        # connector-mediated devices) — `BlueskyPlanRunner._resolve_devices`
        # bridges that for us; passing the bare lambda here, not its
        # result. Every read and write these devices perform is
        # connector-mediated (`read_channel`/`write_channel_checked`) —
        # there is no raw Channel Access anywhere in this path. The closure
        # binds THE one long-lived connector constructed above, not a
        # re-fetch of the (possibly reassigned-on-shutdown) module global.
        #
        # `plans` is left unset (`None`), so `BlueskyPlanRunner.reinitialize`
        # resolves plan names through `_default_plan_registry()` —
        # `get_facility_plans().plans` (task 2.4), which re-scans and
        # re-gates the session/facility layers on every call. A
        # validated session or facility plan is therefore launchable
        # on this connector-mediated path exactly like the demo
        # runner factory; an unvalidated (or
        # validated-then-edited) one is simply absent from the
        # registry the next time this factory's runner resolves it —
        # fail-closed, with no separate gate needed here.
        return BlueskyPlanRunner(
            devices=lambda: connector_devices.build_devices(motors, detectors, connector),
            tiled_writer_factory=_build_tiled_writer_factory(),
        )

    set_runner_factory(_epics_runner_factory)
    if not motors and not detectors:
        # Substrate enabled but neither env var yielded a device: the
        # runner will connect nothing and every scan will have no data.
        # Almost always a misconfiguration (unset/empty
        # BLUESKY_EPICS_MOTORS / _DETECTORS), so surface it loudly.
        logger.warning(
            "%s is enabled but no devices were configured "
            "(BLUESKY_EPICS_MOTORS / BLUESKY_EPICS_DETECTORS are empty or unset); "
            "the substrate runner will connect nothing",
            _EPICS_SUBSTRATE_ENV,
        )
    else:
        logger.info(
            "%s is enabled: wired the EPICS substrate BlueskyPlanRunner "
            "(%d motor(s), %d detector(s))",
            _EPICS_SUBSTRATE_ENV,
            len(motors),
            len(detectors),
        )
    return connector


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Opt-in guarded startup: wire a real bluesky-backed `BlueskyPlanRunner`.

    Two mutually-exclusive opt-in branches, both gated on bluesky/ophyd-async
    being importable (mirrors ``list_plans``'s guarded/lazy-import pattern so
    the Phase-1 "app.py import-clean of bluesky" invariant holds whether the
    extra is absent or neither flag is set — either way ``_runner_factory``
    stays at its ``FakePlanRunner`` default):

    - `_is_epics_substrate_enabled()`: real EPICS devices (Channel Access
      clients of whatever IOC the deploy points at — a virtual accelerator or
      real hardware), built from an explicit PV list
      (`devices/_specs_from_env.py`). This is what a facility deploy (or the
      Phase 3 scenario benchmark) actually runs against.
    - `_is_demo_runner_enabled()`: mock ophyd-async devices, no CA at all —
      the ``osprey deploy`` smoke demo only ("does a run at all").

    If both flags are set, the EPICS substrate wins: an operator who asked
    for real EPICS must never silently get routed to the mock demo instead.

    Task 2.1: when the EPICS substrate branch runs, this also constructs and
    connects the bridge's single long-lived OSPREY connector (module global
    `_connector`, readable via `get_connector()`) — one Channel Access client
    for the whole process lifetime. Task 2.2 wires that same connector into
    `_epics_runner_factory`, so every scan device it builds is
    connector-mediated. The connector is disconnected exactly once after
    `yield`, on shutdown.

    Task 3.1: before any of that connector/CA work, the EPICS-substrate
    branch calls `_assert_limits_readable_if_writable`, which fail-OPEN
    refuses startup (raises) only if writes are enabled, limits checking is
    enabled, and the limits database can't be read — every other combination
    (including writes disabled entirely) starts normally.
    """
    global _connector
    epics_substrate_enabled = _is_epics_substrate_enabled()
    demo_runner_enabled = _is_demo_runner_enabled()
    if epics_substrate_enabled and demo_runner_enabled:
        logger.warning(
            "both %s and %s are set; %s takes precedence (wiring the real EPICS "
            "substrate runner, not the mock demo)",
            _EPICS_SUBSTRATE_ENV,
            _DEMO_RUNNER_ENV,
            _EPICS_SUBSTRATE_ENV,
        )

    if epics_substrate_enabled:
        _connector = await _wire_epics_substrate_runner()
    elif demo_runner_enabled:
        try:
            from .devices.mock import build_devices
            from .plan_runner_bluesky import BlueskyPlanRunner
        except ImportError as exc:
            root_name = (getattr(exc, "name", None) or "").split(".")[0]
            if root_name not in _BRIDGE_ONLY_MODULES:
                raise
            logger.warning(
                "%s is enabled but the bluesky stack is not importable "
                "(%s not found); falling back to FakePlanRunner",
                _DEMO_RUNNER_ENV,
                exc.name,
            )
        else:

            def _demo_runner_factory() -> BlueskyPlanRunner:
                # `build_devices` is `async def` (it connects ophyd-async
                # devices) — `BlueskyPlanRunner._resolve_devices` bridges that
                # for us; passing the bare callable here, not its result.
                return BlueskyPlanRunner(
                    devices=lambda: build_devices(),
                    tiled_writer_factory=_build_tiled_writer_factory(),
                )

            set_runner_factory(_demo_runner_factory)
            logger.info(
                "%s is enabled: wired the mock-devices demo BlueskyPlanRunner (deploy smoke demo)",
                _DEMO_RUNNER_ENV,
            )
    yield

    if _connector is not None:
        await _connector.disconnect()
        _connector = None


app = FastAPI(title="OSPREY Bluesky Bridge", lifespan=_lifespan)

# The shared plan draft's routes (`GET`/`PATCH`/`DELETE /draft`, `GET
# /draft/events`) live in their own self-contained module — state, lock, and
# SSE broadcaster all belong together, and don't need anything else this
# module owns. First `include_router` precedent in this app; every other
# route here is still an inline `@app.<verb>`. `POST /draft/run` is NOT part
# of that router: launching needs the run registry, launch gate, and runner
# factory this module owns, so it lives below as an inline route consuming
# `draft.py`'s launch-state primitives.
app.include_router(draft.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/runs")
def create_run(request: RunRequest) -> dict:
    """Record a launch *intent*. Never touches the runner seam."""
    return registry.add(request).to_dict()


@app.get("/runs")
def list_runs(limit: int = 20) -> list[dict]:
    """This bridge process's tracked runs, newest first (in-memory only)."""
    return [run.to_dict() for run in registry.list(limit=limit)]


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Run status, plus (when present) the intent's ``plan_name``/``plan_args``.

    ``Run.to_dict()`` itself carries neither field — both are read straight
    off the stored intent here via ``_request_field``, so callers that need
    to know *what* is being launched (e.g. task 2.6's launch-approval hook,
    resolving a bare ``run_id`` into a plan to render) don't need a second
    route.
    """
    run = registry.get(run_id)
    out = run.to_dict()
    request = run.request
    plan_name = _request_field(request, "plan_name")
    if plan_name is not None:
        out["plan_name"] = plan_name
        out["plan_args"] = _request_field(request, "plan_args", {})
    return out


@app.post("/runs/{run_id}/launch")
def launch_run(run_id: str, x_launch_token: str = Header(default="")) -> dict:
    """Launch a pending run as a real scan. Token-gated (see `security.py`).

    Callable only by holders of `BLUESKY_LAUNCH_TOKEN` — in practice, the
    `launch_run` MCP tool, whose own invocation already required a human
    approval prompt (PreToolUse) plus an in-tool `writes_enabled` re-check.
    `_launch_validation_gate` runs inside `do_launch`'s own lock, before any
    runner is built (task 2.5) — a session/unreviewed plan with no current
    passing validation record 409s here rather than surfacing downstream as a
    confusing "unknown plan" resolution failure.
    """
    verify_launch_token(x_launch_token)
    run = registry.get(run_id)
    launched_run = do_launch(run, _runner_factory, validator=_launch_validation_gate)
    # Only recorded once do_launch actually succeeds (it raises 409/500
    # otherwise) — a rejected launch attempt must not mark the run as
    # launched by anything.
    if launched_run.launched_by is None:
        launched_run.launched_by = "agent"
    return launched_run.to_dict()


def _mint_and_launch_draft_snapshot(snapshot: draft.LaunchSnapshot) -> Run:
    """Blocking mint + launch of a draft snapshot (runs in a threadpool).

    Order preserves two guarantees:

    - **Mint nothing on a validation-gate rejection**:
      `_validate_launchable_request` runs BEFORE `registry.add`, so a
      session plan whose current on-disk content lost its passing record
      409s with the registry untouched — unlike the two-step launch path,
      where the intent record pre-exists the gate by client action.
    - **Never an eternal pre-launch record**: once the run IS minted, any
      failure — `do_launch`'s own 500s, or the re-run gate catching a file
      edited in the window since the pre-mint check — stamps ``run.error``
      before re-raising, so the record reports ``error`` rather than sitting
      in a pre-launch state forever. `do_launch`'s validator stays wired
      (same defense-in-depth as the launch route); its rejection is the one
      failure `do_launch` raises without stamping the run itself.
    """
    request = RunRequest(plan_name=snapshot.plan_name, plan_args=snapshot.plan_args)

    _validate_launchable_request(request)

    run = registry.add(request)
    try:
        do_launch(run, _runner_factory, validator=_launch_validation_gate)
    except Exception as exc:
        with registry.lock:
            if not run.error:
                run.error = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
        raise
    # Only stamped once do_launch actually succeeds — mirrors `launch_run`'s
    # "a rejected launch attempt must not mark the run as launched" rule.
    if run.launched_by is None:
        run.launched_by = "draft"
    return run


@app.post("/draft/run")
async def launch_draft_run(
    request: DraftRunRequest, x_launch_token: str = Header(default="")
) -> dict:
    """Launch the shared plan draft at a pinned revision — the bridge's single
    launch-from-draft primitive (panel Launch and the agent's `launch_run`
    both land here). Token-gated exactly like `launch_run`.

    Sequence, and why each step sits where it does:

    1. `verify_launch_token` before ANY state is touched — an unarmed (503)
       or bad-token (403) caller never consumes launchability, mints nothing,
       and never even reads the draft.
    2. `draft.check_launchable` snapshots the draft and all three launch
       checks in one critical section under the draft lock, and — on
       success — RESERVES the revision in-flight, so a concurrent second
       POST at the same revision 409s instead of racing this one to a
       duplicate hardware scan (the reservation is taken at the head of the
       unlocked launch window, not committed at its tail). A typed
       :class:`draft.LaunchRejected` becomes a 409 whose ``detail`` carries
       ``code`` (``stale_draft_revision`` / ``draft_revision_already_launched``,
       same dict-detail convention as `PATCH /draft`'s 409s) plus the current
       ``revision`` as a fresh resync baseline. Nothing is minted.
    3. Mint + launch in a threadpool via `_mint_and_launch_draft_snapshot`
       (`do_launch` and the validation gate are blocking; the existing
       launch route gets its threadpool from being a sync ``def`` route,
       which this route can't be — it awaits the draft primitives). The draft
       lock is NOT held here, by construction: `check_launchable` and
       `record_and_broadcast_launch` are separate lock acquisitions with
       the launch in between. On ANY exit without a minted-and-launched run
       (gate 409, do_launch 500, even cancellation — hence ``finally`` with
       a success flag, not ``except``), `draft.release_launch` drops the
       reservation without recording it, so a failed launch never consumes
       the revision.
    4. Only after a successful launch, `draft.record_and_broadcast_launch`
       consumes the reservation, arms the duplicate-launch guard for this
       revision, and emits the ``launched`` SSE frame in one critical
       section.
    """
    verify_launch_token(x_launch_token)

    checked = await draft.check_launchable(request.draft_revision)
    if isinstance(checked, draft.LaunchRejected):
        raise HTTPException(
            status_code=409,
            detail={
                "code": checked.code,
                "detail": checked.detail,
                "revision": checked.revision,
            },
        )

    run: Run | None = None
    try:
        run = await asyncio.to_thread(_mint_and_launch_draft_snapshot, checked)
    finally:
        if run is None:
            await draft.release_launch(checked.revision)

    await draft.record_and_broadcast_launch(run_id=run.id, revision=checked.revision)
    return run.to_dict()


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    """Abort a running plan. Not token-gated — halting is always allowed.

    Coordinates with `do_launch`'s unlocked runner-build window under
    `registry.lock`: if a launch is concurrently mid-build (``launching``
    set, ``runner``/``launched`` not yet published), this just records
    ``stopped`` and `do_launch` itself stops the just-started runner once
    it re-checks `stopped` at publish time — see `runs.py`.
    """
    run = registry.get(run_id)
    with registry.lock:
        scanner_to_stop = run.runner if run.launched else None
        run.stopped = True
    if scanner_to_stop is not None:
        scanner_to_stop.stop_run_thread()
    return run.to_dict()


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


# ---------------------------------------------------------------------------
# Session-plan authoring + validation (task 2.3)
# ---------------------------------------------------------------------------
# A valid Python identifier: the sanitized name doubles as the on-disk file
# stem (`<name>.py`) and the `PLAN_METADATA["name"]` value, so this also rules
# out path traversal (`../`, absolute paths, path separators) in one check.
# Anchored with `\Z`, NOT `$` — `$` matches at end-of-string OR just before a
# single trailing "\n", so `"foo\n"` would otherwise pass this check while
# still not being a valid identifier.
_PLAN_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z")

# A generous bound well above any real plan name — exists only so an
# absurdly long name fails closed here (400) rather than surfacing as an
# unhandled OSError from `Path.write_text` (some filesystems reject a
# filename this long outright, which would otherwise 500).
_MAX_PLAN_NAME_LENGTH = 100

# Neither `/plans/session` nor `/plans/validate` is gated on
# `BLUESKY_LAUNCH_TOKEN` (`security.py`) — that token is deliberately
# unminted whenever writes are unsafe to arm (see
# `container_lifecycle._local_exec_arming_unsafe`), and both these routes
# MUST keep working with writes disabled: authoring and validating a plan
# body never touches a device (the validator's stage-3 dry run drives mock
# devices only, in a subprocess with `EPICS_CA_*` neutralized — see
# `plan_validation.py`). Their protection is the bridge's loopback-only bind
# (see the compose template) plus the MCP-side approval hook
# (`registry/mcp.py`'s `write_plan`/`validate_plan` tiers) —
# not a token gate.


def _sanitize_plan_name(name: str) -> str:
    """Validate ``name`` as a safe plan name, or raise 400.

    Enforced as a Python identifier (not merely "no path separators") because
    the same string is written into the generated ``PLAN_METADATA["name"]``
    block as a plain literal and used verbatim as the on-disk file stem.
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
            detail=f"invalid plan name {name!r}: must be a valid Python identifier",
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
    bytes task 2.4/2.5's gates re-hash from disk when checking for a passing
    validation record.
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
    pass, records the content hash in `validation_records` so task 2.4's load
    gate and task 2.5's launch gate will admit this exact file content.

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
    if result.passed:
        validation_records.record(result.content_hash)

    return {
        "passed": result.passed,
        "reasons": result.reasons,
        "content_hash": result.content_hash,
    }


# ---------------------------------------------------------------------------
# Plan source rendering (task 2.6): backs the launch-approval hook's
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
    """Truncated source text for one plan — the launch-approval hook's data
    source for rendering what a `launch_run` call would actually run.

    ``max_chars`` bounds the returned source text. The default stays at the
    approval hook's skim size (the hook embeds the response verbatim in its
    prompt, so its excerpt must stay small); a caller that needs the full
    text — the plan panel's Source tab — asks for more explicitly. The ask is
    itself capped (422 beyond ``_SOURCE_TRUNCATE_CHARS_MAX``) so the response
    stays bounded no matter what the client sends.

    A session-tier file is looked up directly: its filename IS its name (see
    `write_session_plan`). Its ``validated`` flag reflects the SAME
    `hash_plan_body`/`validation_records` check the load/launch gates use,
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

    Two situations fall through the live path in `get_run_data` and land here: a registry
    miss after a bridge restart (the whole in-memory registry — including `run.run_uid` —
    is gone, so the search below keys on `osprey_run_id`, the durable stamp `do_launch`
    writes into the start doc, never the lost `run_uid`), and a registry hit whose buffer
    was evicted past `live_rows._MAX_RUNS`.

    Returns `None` when Tiled is unconfigured (`BLUESKY_TILED_URI` unset — logged, not an
    error) or when no run in the catalog matches `run_id`; the caller turns either into a
    404, exactly like an unknown live `run_uid` today.

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
    """Read a bounded window of a run's recorded data — dual-source (task 3.3).

    Row-bounded by design — this never returns an unbounded table. Prefers the
    in-process live-row buffer (see `live_rows.py`) whenever it has one:
    ``partial: true`` while the run is still filling in (before its stop doc
    lands), permanently readable once it's marked completed (see
    `live_rows.py`'s retention bound). ``row_count`` is the *true* total rows
    the run has produced so far, even if that's more than what's physically
    stored — ``truncated`` reflects whether this response's window omits any
    of them.

    Falls back to `_from_tiled` (task 3.2) whenever there is no live buffer to
    serve — this is the SAME branch for two different situations: a registry
    miss (the whole in-memory registry, including `run.run_uid`, is gone after
    a bridge restart — so the fallback searches Tiled by `run_id` directly,
    never a `run_uid` that no longer exists to look up), and a registry hit
    whose buffer was evicted past `live_rows._MAX_RUNS`. The fallback trigger
    is always ``buf is None`` — a present-but-empty buffer (``partial: true``,
    zero rows) is a real in-flight run and stays on the live path; checking
    "falsy rows" instead would incorrectly divert a running plan to Tiled
    before it has ever written anything there.

    Raises 409 if the registry has the run but it has no `run_uid` yet (never
    launched, or launched but the scan hasn't emitted a start doc) — there is
    nothing to read from either source, so Tiled is never consulted for this
    case. Raises 404 when neither source has the run — the MCP `get_run_data`
    tool maps 404 to `unknown_run`, and a 200-empty response would make a
    nonexistent run look like a valid empty scan.
    """
    try:
        run = registry.get(run_id)
    except HTTPException:
        run = None

    buf = None
    run_uid: str | None = None
    if run is not None:
        run_uid = run.run_uid
        if run_uid is None:
            raise HTTPException(
                status_code=409, detail=f"run {run_id!r} has not started; no data yet"
            )
        buf = live_rows.get(run_uid)

    if buf is not None:
        result: dict[str, Any] = {"run_uid": run_uid}
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
