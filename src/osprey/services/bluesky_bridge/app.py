"""The Bluesky bridge's FastAPI app: wires the run registry, promote gate, and
runner seam (``runs.py``, ``security.py``, ``plan_runner.py``) into HTTP routes.

Two processes, one machine (see PLAN.md's Technical Architecture): this app
runs in a separate container from OSPREY's own venv, reachable only over
HTTP plus the ``X-Promote-Token`` header. It stays import-clean of bluesky/
ophyd/tiled in Phase 1 — ``_runner_factory`` defaults to the no-op
``FakePlanRunner`` so this app is runnable and manually smoke-testable
(``GET /health``, even a real ``promote``) before the bluesky-backed
``PlanRunner`` exists. Real wiring swaps the factory via ``set_runner_factory``:
either a facility's own deploy code, or this module's own opt-in
``_lifespan`` hook — real EPICS devices (``BLUESKY_EPICS_SUBSTRATE``) or the
built-in deploy smoke demo (``BLUESKY_DEMO_RUNNER``) — see below.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import live_rows
from .plan_runner import FakePlanRunner, PlanRunner
from .plan_types import Provenance
from .plan_validation import hash_plan_body, validate_plan
from .runs import Run, do_promote, registry
from .security import verify_promote_token
from .session_dir import resolve_session_plan_dir
from .validation_record import validation_records

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger("osprey.services.bluesky_bridge.app")

# Root package names `plans.py`'s `from .plans import BUILTIN_PLANS` (and the
# demo-runner lifespan hook below) are allowed to fail on — i.e. the bridge
# running without the `bluesky-bridge` extra installed. An ImportError naming
# anything else (e.g. a module missing an expected attribute, or an unrelated
# third-party import broke) is a genuine bug and must not be swallowed as
# "bluesky is just absent".
_BRIDGE_ONLY_MODULES = {"bluesky", "ophyd", "ophyd_async", "tiled"}

# Opt-in flag (task 2.14a): when truthy (see `_is_demo_runner_enabled`) AND
# bluesky/ophyd-async are importable, `_lifespan` wires a real bluesky-backed
# `BlueskyPlanRunner` against mock devices — the deploy smoke demo's PlanRunner
# (PLAN.md), never a facility's real-hardware wiring. The deploy template
# only renders this var at all when the demo runner is wanted (house
# convention, matching `container_lifecycle.py`'s `DEV_MODE="true"`), so
# "absent" is the off state — but the check itself accepts a few equivalent
# truthy spellings rather than one exact string, so neither half of this
# seam (this hook vs. whatever template/generator sets the var) can drift
# out of sync with the other again.
_DEMO_RUNNER_ENV = "BLUESKY_DEMO_RUNNER"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}

# Opt-in flag (task 2.3): when truthy (see `_is_epics_substrate_enabled`) AND
# bluesky/ophyd-async are importable, `_lifespan` wires a real bluesky-backed
# `BlueskyPlanRunner` against real EPICS devices — Channel Access clients of
# whatever IOC the deploy points at (a virtual accelerator, or real
# hardware), never mock devices. The PV list comes entirely from
# `BLUESKY_EPICS_MOTORS`/`BLUESKY_EPICS_DETECTORS` (see
# `devices/_specs_from_env.py`), never the VA manifest — this process cannot
# import that. If both this flag and `_DEMO_RUNNER_ENV` are set, this one
# wins (see `_lifespan`): an operator who explicitly asked for real EPICS
# must never silently get the mock demo instead.
_EPICS_SUBSTRATE_ENV = "BLUESKY_EPICS_SUBSTRATE"

# Task 2.5: when set, both runner-factory branches below subscribe a
# `TiledWriter` (via `_FaultIsolatedTiledWriter`, see `plan_runner_bluesky.py`) so
# run data survives a bridge restart. Orthogonal to `_DEMO_RUNNER_ENV`/
# `_EPICS_SUBSTRATE_ENV` — it augments whichever runner those two flags pick,
# rather than picking one itself. `BLUESKY_TILED_API_KEY` grants catalog
# access only, never promote authority — see `container_lifecycle.py`'s
# `_SERVICE_TOKEN_VARS`.
_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"

# The `PlanRunner` implementation `do_promote` builds for every promotion.
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
    """Override the `PlanRunner` implementation `do_promote` builds.

    A real bluesky-backed factory (`_lifespan` below, or a facility's own
    deploy wiring) calls this instead of reaching into the private module
    global directly.
    """
    global _runner_factory
    _runner_factory = factory


def _is_demo_runner_enabled() -> bool:
    """True if `BLUESKY_DEMO_RUNNER` is set to any of `_TRUTHY_VALUES` (case/whitespace-insensitive).

    Absent, empty, or any other value (e.g. "false") is off — deliberately
    liberal on the "on" spellings, but never guesses at "on" from an
    unrecognized value.
    """
    return os.environ.get(_DEMO_RUNNER_ENV, "").strip().lower() in _TRUTHY_VALUES


def _is_epics_substrate_enabled() -> bool:
    """True if `BLUESKY_EPICS_SUBSTRATE` is set to any of `_TRUTHY_VALUES`.

    Same liberal-on-"on"-spellings parsing as `_is_demo_runner_enabled` —
    see that function's docstring for why.
    """
    return os.environ.get(_EPICS_SUBSTRATE_ENV, "").strip().lower() in _TRUTHY_VALUES


# Connector types the EPICS-substrate branch knows how to build a gateway-less
# `type_config` for — real Channel Access, whether against a virtual
# accelerator soft-IOC or live hardware.
_EPICS_LIKE_CONNECTOR_TYPES = ("virtual_accelerator", "epics")


def _resolve_control_system_type() -> str:
    """Read `control_system.type` from the bridge's mounted project config.

    Single source of truth (Connector = the single control-system interface):
    one config line flips the whole Bluesky stack between the mock connector and
    real Channel Access (virtual accelerator or live hardware) — see the
    `control-assistant` preset's `config.control_system.type` comment.

    Fail-SAFE default: `"mock"` whenever the config can't be read at all (no
    project config context — most unit-test environments — or a transient
    lookup failure), never `"virtual_accelerator"`/`"epics"` — the mock
    connector never touches Channel Access, so an unreadable config can never
    silently connect to real hardware. Mirrors
    `_assert_limits_readable_if_writable`'s "no project config context ->
    treat as absent, don't block" handling of the same exception set.
    """
    from osprey.utils.config import get_config_value

    try:
        control_system_type = get_config_value("control_system.type", "mock")
    except (FileNotFoundError, KeyError, RuntimeError):
        return "mock"

    if not control_system_type or not isinstance(control_system_type, str):
        return "mock"
    return control_system_type


def _build_tiled_writer_factory() -> Callable[[], Any] | None:
    """Build the `tiled_writer_factory` `BlueskyPlanRunner` accepts, or `None` if Tiled is unconfigured.

    Reads `BLUESKY_TILED_URI` fresh on every call (never cached), so each
    promotion's `BlueskyPlanRunner` picks up the current env — matching
    `do_promote`'s "fresh runner per promotion" contract (`plan_runner_bluesky.py`'s
    `_FaultIsolatedTiledWriter` docstring: "no cross-run state to reset").
    `None` when the URI is unset: `BlueskyPlanRunner.__init__` treats that as "no
    Tiled subscription", identical to Phase 1's no-Tiled-server behavior.

    The returned closure imports `TiledWriter` from
    `bluesky.callbacks.tiled_writer` — NOT from `tiled` (TR2) — lazily,
    inside itself, so this module stays import-clean of both `bluesky` and
    `tiled` (`_BRIDGE_ONLY_MODULES`) even when a caller holds onto the
    returned factory without ever invoking it.
    """
    uri = os.environ.get(_TILED_URI_ENV)
    if not uri:
        return None

    def factory() -> Any:
        from bluesky.callbacks.tiled_writer import TiledWriter

        return TiledWriter.from_uri(uri, api_key=os.environ[_TILED_API_KEY_ENV])

    return factory


def _assert_limits_readable_if_writable() -> None:
    """Refuse startup if writes are enabled but the limits database can't be read.

    Fail-OPEN by design (task 3.1): this is the ONLY combination that refuses
    startup — ``control_system.writes_enabled`` AND
    ``control_system.limits_checking.enabled`` both true, AND the limits
    database is missing, unreadable, or unparseable. Every other combination
    starts normally: writes disabled (read-only posture) never even probes
    the database; writes enabled with limits checking disabled needs no
    database at all; writes enabled with a readable database is the healthy
    case. A writable deploy with no working limits enforcement is the one
    unsafe posture this guard exists to catch before any connector/CA work
    begins.

    Mirrors `LimitsValidator.from_config`'s ``database_path`` resolution
    (a relative path resolved against the ``CONFIG_FILE`` env var's directory
    when set, falling back to ``project_root`` otherwise — container-correct,
    since the deploy flattens ``project_root`` in as the HOST build path,
    while ``CONFIG_FILE`` points at the config actually mounted in-container),
    but probes readability via `LimitsValidator._load_limits_database`
    directly rather than calling `from_config` — `from_config` swallows every
    load failure to `None`, which would hide the exact failure this guard
    must detect and raise on.

    No project config context at all (e.g. running outside a configured
    OSPREY project — most unit-test environments) is treated the same way
    `LimitsValidator.from_config` treats it: nothing to probe, so this
    returns without blocking startup, rather than raising on the config
    lookup itself.

    Raises:
        RuntimeError: naming which condition failed (config keys, and
            whether the database path was configured/found/parseable) —
            never the database's file contents or any other secret value.
    """
    from osprey.utils.config import get_config_value

    try:
        writes_enabled = get_config_value("control_system.writes_enabled", False)
        limits_enabled = get_config_value("control_system.limits_checking.enabled", False)
        db_path = get_config_value("control_system.limits_checking.database_path", None)
        project_root = get_config_value("project_root", None)
    except (FileNotFoundError, KeyError, RuntimeError):
        return

    if not writes_enabled:
        return
    if not limits_enabled:
        return

    if not db_path or not isinstance(db_path, str):
        raise RuntimeError(
            "refusing to start writable: control_system.writes_enabled and "
            "control_system.limits_checking.enabled are both set, but "
            "control_system.limits_checking.database_path is not configured"
        )

    from osprey.connectors.control_system.limits_validator import LimitsValidator

    # Same relative-path resolution as `LimitsValidator.from_config`.
    db_path = LimitsValidator.resolve_database_path(db_path, project_root)

    try:
        LimitsValidator._load_limits_database(db_path)
    except Exception as exc:
        raise RuntimeError(
            "refusing to start writable: control_system.writes_enabled and "
            "control_system.limits_checking.enabled are both set, but the "
            "configured control_system.limits_checking.database_path could "
            "not be read or parsed"
        ) from exc


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
        try:
            from .devices import connector as connector_devices
            from .devices._specs_from_env import specs_from_env
            from .plan_runner_bluesky import BlueskyPlanRunner
        except ImportError as exc:
            root_name = (getattr(exc, "name", None) or "").split(".")[0]
            if root_name not in _BRIDGE_ONLY_MODULES:
                raise
            logger.warning(
                "%s is enabled but the bluesky-bridge extra is not installed "
                "(%s not found); falling back to FakePlanRunner",
                _EPICS_SUBSTRATE_ENV,
                exc.name,
            )
        else:
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
            # this already-guarded branch regardless.
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
            _connector = await ConnectorFactory.create_control_system_connector(
                connector_type_config
            )
            logger.info(
                "%s is enabled: connected the bridge's single long-lived OSPREY connector "
                "(control_system.type=%s, %s)",
                _EPICS_SUBSTRATE_ENV,
                control_system_type,
                type(_connector).__name__,
            )

            motors, detectors = specs_from_env(os.environ)

            # Bind the closure to THE one long-lived connector constructed
            # above, not a re-fetch of the (possibly reassigned-on-shutdown)
            # module global.
            connector = _connector

            def _epics_runner_factory() -> BlueskyPlanRunner:
                # `connector_devices.build_devices` is `async def` (it builds
                # connector-mediated devices) — `BlueskyPlanRunner._resolve_devices`
                # bridges that for us; passing the bare lambda here, not its
                # result. Every read and write these devices perform is
                # connector-mediated (`read_channel`/`write_channel_checked`) —
                # there is no raw Channel Access anywhere in this path.
                #
                # `plans` is left unset (`None`) rather than pinned to
                # `BUILTIN_PLANS`, so `BlueskyPlanRunner.reinitialize` resolves
                # plan names through `_default_plan_registry()` — built-ins
                # merged with `get_facility_plans().plans` (task 2.4), which
                # re-scans and re-gates the session/facility layers on every
                # call. A validated session or facility plan is therefore
                # launchable on this connector-mediated path exactly like the
                # demo runner factory below; an unvalidated (or
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
    elif demo_runner_enabled:
        try:
            from .devices.mock import build_devices
            from .plan_runner_bluesky import BlueskyPlanRunner
        except ImportError as exc:
            root_name = (getattr(exc, "name", None) or "").split(".")[0]
            if root_name not in _BRIDGE_ONLY_MODULES:
                raise
            logger.warning(
                "%s is enabled but the bluesky-bridge extra is not installed "
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


class RunRequest(BaseModel):
    """A scan launch intent (`POST /runs`).

    Intentionally generic: `plan_name` names a plan the registry (`plans.py`,
    plus any facility-injected plans from `plan_loader.py`) resolves, and
    `plan_args` is forwarded to the runner unmodified via `do_promote` ->
    `PlanRunner.reinitialize(run.request)`.
    """

    plan_name: str
    plan_args: dict[str, Any] = Field(default_factory=dict)


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
def get_run_status(run_id: str) -> dict:
    """Run status, plus (when present) the intent's ``plan_name``/``plan_args``.

    ``Run.to_dict()`` itself carries neither field — the lifecycle core
    (``runs.py``) treats ``request`` as opaque (see its own docstring). Both
    are read straight off the stored intent here, the same
    dict-or-attribute extraction ``_promote_validation_gate`` already uses,
    so callers that need to know *what* is being launched (e.g. task 2.6's
    launch-approval hook, resolving a bare ``run_id`` into a plan to render)
    don't need a second route.
    """
    run = registry.get(run_id)
    out = run.to_dict()
    request = run.request
    plan_name = (
        request.get("plan_name")
        if isinstance(request, dict)
        else getattr(request, "plan_name", None)
    )
    if plan_name is not None:
        out["plan_name"] = plan_name
        out["plan_args"] = (
            request.get("plan_args", {})
            if isinstance(request, dict)
            else getattr(request, "plan_args", {})
        )
    return out


def _promote_validation_gate(run: Run) -> None:
    """Refuse to promote a session/unreviewed plan with no CURRENT passing validation record.

    Defense-in-depth alongside task 2.4's session-layer LOAD gate
    (`plan_loader.py`'s `_load_plan_file`): that gate already keeps an
    unvalidated session/unreviewed file out of `get_facility_plans().plans`
    entirely, so in the common case this validator finds nothing to reject.
    It exists for the narrow race the load gate can't close on its own — the
    `PlanSpec` `get_facility_plans()` returned to resolve this run's
    `plan_name` moments earlier could be stale by the time promote runs (e.g.
    the session file was edited in between) — so this independently re-reads
    the file straight from `resolve_session_plan_dir()` and re-hashes its
    CURRENT content with `hash_plan_body`, the same normalization the record
    was keyed on, rather than trusting the earlier snapshot.

    Raises `HTTPException(409, ...)` for any plan name backed by a file in
    `resolve_session_plan_dir()` whose current content has no passing
    record — whether or not `get_facility_plans()` currently registers it.
    A name the load gate is quarantining *right now* for lacking a record
    resolves to no `PlanSpec` at all, but its file still exists under the
    session directory; treating that as `session` provenance too (rather than
    "not found") is what turns an already-quarantined plan's promote attempt
    into this clear 409 instead of a confusing "unknown plan" failure further
    downstream. A non-session provenance (`shipped`/`preset`/`facility`), or a
    name with neither a `PlanSpec` nor a session-dir file at all, is left
    alone — `PlanRunner.reinitialize`'s own "unknown plan" handling is the right
    place for the latter.
    """
    request = run.request
    plan_name = (
        request.get("plan_name")
        if isinstance(request, dict)
        else getattr(request, "plan_name", None)
    )
    if not plan_name:
        return

    from .plan_loader import get_facility_plans

    spec = get_facility_plans().plans.get(plan_name)
    plan_path = resolve_session_plan_dir() / f"{plan_name}.py"
    if spec is not None:
        is_session = spec.provenance in ("session", "unreviewed")
    else:
        is_session = plan_path.is_file()

    if not is_session or not plan_path.is_file():
        # Not a session-tier plan at all, or its file has since vanished —
        # either way there is nothing here to re-hash; `PlanRunner.reinitialize`
        # will hit its own "unknown plan" path if the name doesn't resolve.
        return

    content = plan_path.read_text(encoding="utf-8")
    if not validation_records.has_passing_record(hash_plan_body(content)):
        raise HTTPException(
            status_code=409,
            detail=(
                f"session plan {plan_name!r} has no passing validation record; "
                "validate it before launching"
            ),
        )


@app.post("/runs/{run_id}/promote")
def promote_run(run_id: str, x_promote_token: str = Header(default="")) -> dict:
    """Promote an intent to a real scan launch. Token-gated (see `security.py`).

    Callable only by holders of `BLUESKY_PROMOTE_TOKEN` — in practice, the
    `launch_run` MCP tool, whose own invocation already required a human
    approval prompt (PreToolUse) plus an in-tool `writes_enabled` re-check.
    `_promote_validation_gate` runs inside `do_promote`'s own lock, before any
    runner is built (task 2.5) — a session/unreviewed plan with no current
    passing validation record 409s here rather than surfacing downstream as a
    confusing "unknown plan" resolution failure.
    """
    verify_promote_token(x_promote_token)
    run = registry.get(run_id)
    promoted_run = do_promote(run, _runner_factory, validator=_promote_validation_gate)
    # Only recorded once do_promote actually succeeds (it raises 409/500
    # otherwise) — a rejected promote attempt must not mark the run as
    # launched by anything.
    if promoted_run.launched_by is None:
        promoted_run.launched_by = "agent"
    return promoted_run.to_dict()


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    """Abort a running plan. Not token-gated — halting is always allowed.

    Coordinates with `do_promote`'s unlocked runner-build window under
    `registry.lock`: if a promote is concurrently mid-build (``promoting``
    set, ``runner``/``promoted`` not yet published), this just records
    ``stopped`` and `do_promote` itself stops the just-started runner once
    it re-checks `stopped` at publish time — see `runs.py`.
    """
    run = registry.get(run_id)
    with registry.lock:
        scanner_to_stop = run.runner if run.promoted else None
        run.stopped = True
    if scanner_to_stop is not None:
        scanner_to_stop.stop_run_thread()
    return run.to_dict()


@app.get("/plans")
def list_plans() -> list:
    """Registered scan plans: built-ins merged with any facility-injected plans.

    A facility plan overrides a built-in of the same name. `plans.py` (the
    built-in set) imports bluesky, so it's a guarded/lazy import — this route
    degrades to facility-only (or `[]`) rather than 500ing when bluesky isn't
    installed. `plan_loader.py` (facility injection, task 2.4) is import-clean
    of bluesky, so facility-injected plans are always served regardless — see
    that module for how the plan module path is resolved.

    Each entry (`PlanSpec.to_dict()`) carries `metadata` (the plan's
    authoring-declared `PLAN_METADATA`, or `None` for a built-in that doesn't
    author one) and `provenance` (its loader-assigned trust tier) alongside
    `name`/`description`/`schema` — see `plan_types.py`.
    """
    from .plan_loader import get_facility_plans

    merged: dict[str, Any] = {}
    try:
        from .plans import BUILTIN_PLANS

        merged.update(BUILTIN_PLANS)
    except ImportError as exc:
        root_name = (getattr(exc, "name", None) or "").split(".")[0]
        if root_name not in _BRIDGE_ONLY_MODULES:
            raise
        logger.info(
            "GET /plans: built-in plan set unavailable (%s not installed); "
            "serving facility-injected plans only",
            exc.name,
        )
    merged.update(get_facility_plans().plans)

    return [spec.to_dict() for spec in merged.values()]


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
# `BLUESKY_PROMOTE_TOKEN` (`security.py`) — that token is deliberately
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


class PlanSessionWriteRequest(BaseModel):
    """Request body for `POST /plans/session`: author a session-tier plan file.

    ``body`` is the author's own source (``PARAMS`` + ``build_plan``, per the
    layered directory catalog's file contract) — it is never exec'd by this
    route. The remaining fields become the generated `PLAN_METADATA` block
    prepended to it; together they must satisfy `plan_metadata.PlanMetadata`'s
    contract once the session-tier load gate (task 2.4) parses the file.
    """

    name: str
    description: str = ""
    category: str
    required_devices: list[str] = Field(default_factory=list)
    writes: bool
    body: str


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


class PlanValidateRequest(BaseModel):
    """Request body for `POST /plans/validate`: validate a session plan by name.

    ``sample_args`` supplies the stage-3 dry run's `PARAMS` field values
    directly (the simpler of the two options `plan_validation.py`'s docstring
    calls out — deriving minimal samples from the `PARAMS` schema would need
    per-type generation logic this bridge does not otherwise have); omit it
    for a `PARAMS` with no required fields.
    """

    name: str
    sample_args: dict[str, Any] | None = None
    dry_run_timeout: float = 30.0


@app.post("/plans/validate")
async def validate_session_plan(request: PlanValidateRequest) -> dict:
    """Validate the CURRENT on-disk content of a session plan file.

    Reads the file `POST /plans/session` wrote (never a separately-passed
    body) so "validated bytes == file bytes" is structural, not a caller
    convention. Runs `validate_plan`'s three ordered stages; on a
    pass, records the content hash in `validation_records` so task 2.4's load
    gate and task 2.5's promote gate will admit this exact file content.

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

_SOURCE_TRUNCATE_CHARS = 4000  # a few KB: enough for a human skim, bounded


def _find_layer_source_path(name: str) -> tuple[Any, Provenance] | None:
    """Best-effort locate the on-disk file behind a shipped/preset/facility plan.

    Directory-layer files are keyed by their declared ``PLAN_METADATA["name"]``,
    not necessarily their filename (``plans_core/grid_scan.py`` declares
    ``"grid_scan_nd"``) — so this parses each candidate file's source with
    ``ast`` (never execs it) purely to read the literal ``name`` off its
    ``PLAN_METADATA`` dict. Returns `None` for a built-in with no backing
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
def get_plan_source(name: str) -> dict:
    """Truncated source text for one plan — the launch-approval hook's data
    source for rendering what a `launch_run` call would actually run.

    A session-tier file is looked up directly: its filename IS its name (see
    `write_session_plan`). Its ``validated`` flag reflects the SAME
    `hash_plan_body`/`validation_records` check the load/promote gates use,
    computed fresh from the file's CURRENT content — never cached — so a
    re-authored file that invalidates a prior pass is reported honestly, even
    if that leaves it quarantined out of `GET /plans` entirely.

    A shipped/preset/facility file is located by `_find_layer_source_path`
    (best-effort) and reported ``validated=True`` unconditionally — those
    tiers carry no validation-record gate; they are operator-trusted by
    construction, not by a passing record.

    Raises 404 if no file can be located for ``name`` in any tier (including
    a built-in with no backing file at all).
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

    truncated_content = content[:_SOURCE_TRUNCATE_CHARS]
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

    Shared by every data source `read_run_data` serves from (today: the live
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
    """Serve `read_run_data` from the durable Tiled catalog once a run's live buffer is gone.

    Two situations fall through the live path in `read_run_data` and land here: a registry
    miss after a bridge restart (the whole in-memory registry — including `run.run_uid` —
    is gone, so the search below keys on `osprey_run_id`, the durable stamp `do_promote`
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
def read_run_data(
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
    promoted, or promoted but the scan hasn't emitted a start doc) — there is
    nothing to read from either source, so Tiled is never consulted for this
    case. Raises 404 when neither source has the run — the MCP `read_run_data`
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
