"""The Bluesky bridge's FastAPI app: wires the run registry, promote gate, and
scanner seam (``runs.py``, ``security.py``, ``scanner.py``) into HTTP routes.

Two processes, one machine (see PLAN.md's Technical Architecture): this app
runs in a separate container from OSPREY's own venv, reachable only over
HTTP plus the ``X-Promote-Token`` header. It stays import-clean of bluesky/
ophyd/tiled in Phase 1 — ``_scanner_factory`` defaults to the no-op
``FakeScanner`` so this app is runnable and manually smoke-testable
(``GET /health``, even a real ``promote``) before the bluesky-backed
``Scanner`` exists. Real wiring swaps the factory via ``set_scanner_factory``:
either a facility's own deploy code, or this module's own opt-in
``_lifespan`` hook — real EPICS devices (``BLUESKY_EPICS_SUBSTRATE``) or the
built-in deploy smoke demo (``BLUESKY_DEMO_SCANNER``) — see below.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import live_rows
from .runs import do_promote, registry
from .scanner import FakeScanner, Scanner
from .security import verify_promote_token

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger("osprey.services.bluesky_bridge.app")

# Root package names `plans.py`'s `from .plans import BUILTIN_PLANS` (and the
# demo-scanner lifespan hook below) are allowed to fail on — i.e. the bridge
# running without the `bluesky-bridge` extra installed. An ImportError naming
# anything else (e.g. a module missing an expected attribute, or an unrelated
# third-party import broke) is a genuine bug and must not be swallowed as
# "bluesky is just absent".
_BRIDGE_ONLY_MODULES = {"bluesky", "ophyd", "ophyd_async", "tiled"}

# Opt-in flag (task 2.14a): when truthy (see `_is_demo_scanner_enabled`) AND
# bluesky/ophyd-async are importable, `_lifespan` wires a real bluesky-backed
# `BlueskyScanner` against mock devices — the deploy smoke demo's Scanner
# (PLAN.md), never a facility's real-hardware wiring. The deploy template
# only renders this var at all when the demo scanner is wanted (house
# convention, matching `container_lifecycle.py`'s `DEV_MODE="true"`), so
# "absent" is the off state — but the check itself accepts a few equivalent
# truthy spellings rather than one exact string, so neither half of this
# seam (this hook vs. whatever template/generator sets the var) can drift
# out of sync with the other again.
_DEMO_SCANNER_ENV = "BLUESKY_DEMO_SCANNER"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}

# Opt-in flag (task 2.3): when truthy (see `_is_epics_substrate_enabled`) AND
# bluesky/ophyd-async are importable, `_lifespan` wires a real bluesky-backed
# `BlueskyScanner` against real EPICS devices — Channel Access clients of
# whatever IOC the deploy points at (a virtual accelerator, or real
# hardware), never mock devices. The PV list comes entirely from
# `BLUESKY_EPICS_MOTORS`/`BLUESKY_EPICS_DETECTORS` (see
# `devices/_specs_from_env.py`), never the VA manifest — this process cannot
# import that. If both this flag and `_DEMO_SCANNER_ENV` are set, this one
# wins (see `_lifespan`): an operator who explicitly asked for real EPICS
# must never silently get the mock demo instead.
_EPICS_SUBSTRATE_ENV = "BLUESKY_EPICS_SUBSTRATE"

# Task 2.5: when set, both scanner-factory branches below subscribe a
# `TiledWriter` (via `_FaultIsolatedTiledWriter`, see `scanner_bluesky.py`) so
# scan data survives a bridge restart. Orthogonal to `_DEMO_SCANNER_ENV`/
# `_EPICS_SUBSTRATE_ENV` — it augments whichever scanner those two flags pick,
# rather than picking one itself. `BLUESKY_TILED_API_KEY` grants catalog
# access only, never promote authority — see `container_lifecycle.py`'s
# `_SERVICE_TOKEN_VARS`.
_TILED_URI_ENV = "BLUESKY_TILED_URI"
_TILED_API_KEY_ENV = "BLUESKY_TILED_API_KEY"

# The `Scanner` implementation `do_promote` builds for every promotion.
_scanner_factory: Callable[[], Scanner] = FakeScanner

# Task 2.1: the bridge's single long-lived OSPREY connector — one Channel
# Access client for the process's whole lifetime, constructed by `_lifespan`
# only when `_is_epics_substrate_enabled()`, and disconnected exactly once on
# shutdown. `None` whenever the EPICS substrate isn't enabled, or before
# `_lifespan` has constructed it, or after shutdown has torn it down. Wired
# into the scanner factory by task 2.2's `_epics_scanner_factory` closure,
# which builds every scan device against this one connector.
_connector: Any | None = None


def get_connector() -> Any | None:
    """The bridge's single long-lived OSPREY connector, or `None` if unset.

    `None` whenever the EPICS substrate isn't enabled, before `_lifespan` has
    constructed it, or after shutdown has disconnected it. A later task's
    `_epics_scanner_factory` closure reads this to build connector-backed
    devices.
    """
    return _connector


def set_scanner_factory(factory: Callable[[], Scanner]) -> None:
    """Override the `Scanner` implementation `do_promote` builds.

    A real bluesky-backed factory (`_lifespan` below, or a facility's own
    deploy wiring) calls this instead of reaching into the private module
    global directly.
    """
    global _scanner_factory
    _scanner_factory = factory


def _is_demo_scanner_enabled() -> bool:
    """True if `BLUESKY_DEMO_SCANNER` is set to any of `_TRUTHY_VALUES` (case/whitespace-insensitive).

    Absent, empty, or any other value (e.g. "false") is off — deliberately
    liberal on the "on" spellings, but never guesses at "on" from an
    unrecognized value.
    """
    return os.environ.get(_DEMO_SCANNER_ENV, "").strip().lower() in _TRUTHY_VALUES


def _is_epics_substrate_enabled() -> bool:
    """True if `BLUESKY_EPICS_SUBSTRATE` is set to any of `_TRUTHY_VALUES`.

    Same liberal-on-"on"-spellings parsing as `_is_demo_scanner_enabled` —
    see that function's docstring for why.
    """
    return os.environ.get(_EPICS_SUBSTRATE_ENV, "").strip().lower() in _TRUTHY_VALUES


def _build_tiled_writer_factory() -> Callable[[], Any] | None:
    """Build the `tiled_writer_factory` `BlueskyScanner` accepts, or `None` if Tiled is unconfigured.

    Reads `BLUESKY_TILED_URI` fresh on every call (never cached), so each
    promotion's `BlueskyScanner` picks up the current env — matching
    `do_promote`'s "fresh scanner per promotion" contract (`scanner_bluesky.py`'s
    `_FaultIsolatedTiledWriter` docstring: "no cross-run state to reset").
    `None` when the URI is unset: `BlueskyScanner.__init__` treats that as "no
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
    """Opt-in guarded startup: wire a real bluesky-backed `BlueskyScanner`.

    Two mutually-exclusive opt-in branches, both gated on bluesky/ophyd-async
    being importable (mirrors ``list_plans``'s guarded/lazy-import pattern so
    the Phase-1 "app.py import-clean of bluesky" invariant holds whether the
    extra is absent or neither flag is set — either way ``_scanner_factory``
    stays at its ``FakeScanner`` default):

    - `_is_epics_substrate_enabled()`: real EPICS devices (Channel Access
      clients of whatever IOC the deploy points at — a virtual accelerator or
      real hardware), built from an explicit PV list
      (`devices/_specs_from_env.py`). This is what a facility deploy (or the
      Phase 3 scenario benchmark) actually runs against.
    - `_is_demo_scanner_enabled()`: mock ophyd-async devices, no CA at all —
      the ``osprey deploy`` smoke demo only ("does a scan run at all").

    If both flags are set, the EPICS substrate wins: an operator who asked
    for real EPICS must never silently get routed to the mock demo instead.

    Task 2.1: when the EPICS substrate branch runs, this also constructs and
    connects the bridge's single long-lived OSPREY connector (module global
    `_connector`, readable via `get_connector()`) — one Channel Access client
    for the whole process lifetime. Task 2.2 wires that same connector into
    `_epics_scanner_factory`, so every scan device it builds is
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
    demo_scanner_enabled = _is_demo_scanner_enabled()
    if epics_substrate_enabled and demo_scanner_enabled:
        logger.warning(
            "both %s and %s are set; %s takes precedence (wiring the real EPICS "
            "substrate scanner, not the mock demo)",
            _EPICS_SUBSTRATE_ENV,
            _DEMO_SCANNER_ENV,
            _EPICS_SUBSTRATE_ENV,
        )

    if epics_substrate_enabled:
        try:
            from .devices import connector as connector_devices
            from .devices._specs_from_env import specs_from_env
            from .plans import BUILTIN_PLANS
            from .scanner_bluesky import BlueskyScanner
        except ImportError as exc:
            root_name = (getattr(exc, "name", None) or "").split(".")[0]
            if root_name not in _BRIDGE_ONLY_MODULES:
                raise
            logger.warning(
                "%s is enabled but the bluesky-bridge extra is not installed "
                "(%s not found); falling back to FakeScanner",
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

            # Task 2.1: construct the single long-lived OSPREY connector this
            # bridge holds for its whole process lifetime. `osprey.connectors.factory`
            # and `epics_connector` are import-safe even in a base install (pyepics
            # is imported lazily inside `EPICSConnector.connect()`), but the import
            # stays inside this already-guarded branch regardless. A gateway-less
            # `type_config` (no "gateways" key) makes `connect()` skip the block
            # that sets process-wide `EPICS_CA_*` env, so the compose-inherited
            # `EPICS_CA_NAME_SERVERS` (pointing at the virtual accelerator or real
            # hardware) survives untouched (FR8/CF-1) — and it needs no running CA
            # server, so this is safe to do unconditionally at startup.
            from osprey.connectors.factory import (
                ConnectorFactory,
                register_builtin_connectors,
            )

            register_builtin_connectors()  # idempotent (CF-3); must run before create
            _connector = await ConnectorFactory.create_control_system_connector(
                {
                    "type": "virtual_accelerator",
                    "connector": {"virtual_accelerator": {"timeout": 5.0}},
                }
            )
            logger.info(
                "%s is enabled: connected the bridge's single long-lived OSPREY connector (%s)",
                _EPICS_SUBSTRATE_ENV,
                type(_connector).__name__,
            )

            motors, detectors = specs_from_env(os.environ)

            # Bind the closure to THE one long-lived connector constructed
            # above, not a re-fetch of the (possibly reassigned-on-shutdown)
            # module global.
            connector = _connector

            def _epics_scanner_factory() -> BlueskyScanner:
                # `connector_devices.build_devices` is `async def` (it builds
                # connector-mediated devices) — `BlueskyScanner._resolve_devices`
                # bridges that for us; passing the bare lambda here, not its
                # result. Every read and write these devices perform is
                # connector-mediated (`read_channel`/`write_channel_checked`) —
                # there is no raw Channel Access anywhere in this path.
                return BlueskyScanner(
                    devices=lambda: connector_devices.build_devices(motors, detectors, connector),
                    plans=BUILTIN_PLANS,
                    tiled_writer_factory=_build_tiled_writer_factory(),
                )

            set_scanner_factory(_epics_scanner_factory)
            if not motors and not detectors:
                # Substrate enabled but neither env var yielded a device: the
                # scanner will connect nothing and every scan will have no data.
                # Almost always a misconfiguration (unset/empty
                # BLUESKY_EPICS_MOTORS / _DETECTORS), so surface it loudly.
                logger.warning(
                    "%s is enabled but no devices were configured "
                    "(BLUESKY_EPICS_MOTORS / BLUESKY_EPICS_DETECTORS are empty or unset); "
                    "the substrate scanner will connect nothing",
                    _EPICS_SUBSTRATE_ENV,
                )
            else:
                logger.info(
                    "%s is enabled: wired the EPICS substrate BlueskyScanner "
                    "(%d motor(s), %d detector(s))",
                    _EPICS_SUBSTRATE_ENV,
                    len(motors),
                    len(detectors),
                )
    elif demo_scanner_enabled:
        try:
            from .devices.mock import build_devices
            from .scanner_bluesky import BlueskyScanner
        except ImportError as exc:
            root_name = (getattr(exc, "name", None) or "").split(".")[0]
            if root_name not in _BRIDGE_ONLY_MODULES:
                raise
            logger.warning(
                "%s is enabled but the bluesky-bridge extra is not installed "
                "(%s not found); falling back to FakeScanner",
                _DEMO_SCANNER_ENV,
                exc.name,
            )
        else:

            def _demo_scanner_factory() -> BlueskyScanner:
                # `build_devices` is `async def` (it connects ophyd-async
                # devices) — `BlueskyScanner._resolve_devices` bridges that
                # for us; passing the bare callable here, not its result.
                return BlueskyScanner(
                    devices=lambda: build_devices(),
                    tiled_writer_factory=_build_tiled_writer_factory(),
                )

            set_scanner_factory(_demo_scanner_factory)
            logger.info(
                "%s is enabled: wired the mock-devices demo BlueskyScanner (deploy smoke demo)",
                _DEMO_SCANNER_ENV,
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
    `plan_args` is forwarded to the scanner unmodified via `do_promote` ->
    `Scanner.reinitialize(run.request)`.
    """

    plan_name: str
    plan_args: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/runs")
def create_run(request: RunRequest) -> dict:
    """Record a launch *intent*. Never touches the scanner seam."""
    return registry.add(request).to_dict()


@app.get("/runs")
def list_runs(limit: int = 20) -> list[dict]:
    """This bridge process's tracked runs, newest first (in-memory only)."""
    return [run.to_dict() for run in registry.list(limit=limit)]


@app.get("/runs/{run_id}")
def get_run_status(run_id: str) -> dict:
    return registry.get(run_id).to_dict()


@app.post("/runs/{run_id}/promote")
def promote_run(run_id: str, x_promote_token: str = Header(default="")) -> dict:
    """Promote an intent to a real scan launch. Token-gated (see `security.py`).

    Callable only by holders of `BLUESKY_PROMOTE_TOKEN` — in practice, the
    `launch_scan` MCP tool, whose own invocation already required a human
    approval prompt (PreToolUse) plus an in-tool `writes_enabled` re-check.
    """
    verify_promote_token(x_promote_token)
    run = registry.get(run_id)
    promoted_run = do_promote(run, _scanner_factory)
    # Only recorded once do_promote actually succeeds (it raises 409/500
    # otherwise) — a rejected promote attempt must not mark the run as
    # launched by anything.
    if promoted_run.launched_by is None:
        promoted_run.launched_by = "agent"
    return promoted_run.to_dict()


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    """Abort a running scan. Not token-gated — halting is always allowed.

    Coordinates with `do_promote`'s unlocked scanner-build window under
    `registry.lock`: if a promote is concurrently mid-build (``promoting``
    set, ``scanner``/``promoted`` not yet published), this just records
    ``stopped`` and `do_promote` itself stops the just-started scanner once
    it re-checks `stopped` at publish time — see `runs.py`.
    """
    run = registry.get(run_id)
    with registry.lock:
        scanner_to_stop = run.scanner if run.promoted else None
        run.stopped = True
    if scanner_to_stop is not None:
        scanner_to_stop.stop_scanning_thread()
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
    "falsy rows" instead would incorrectly divert a running scan to Tiled
    before it has ever written anything there.

    Raises 409 if the registry has the run but it has no `run_uid` yet (never
    promoted, or promoted but the scan hasn't emitted a start doc) — there is
    nothing to read from either source, so Tiled is never consulted for this
    case. Raises 404 when neither source has the run — the MCP `read_scan_data`
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
