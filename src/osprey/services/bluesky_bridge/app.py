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

# The `Scanner` implementation `do_promote` builds for every promotion.
_scanner_factory: Callable[[], Scanner] = FakeScanner


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
    """
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
            from .devices import epics
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
            motors, detectors = specs_from_env(os.environ)

            def _epics_scanner_factory() -> BlueskyScanner:
                # `epics.build_devices` is `async def` (it connects real CA
                # signals) — `BlueskyScanner._resolve_devices` bridges that
                # for us; passing the bare lambda here, not its result.
                return BlueskyScanner(
                    devices=lambda: epics.build_devices(motors, detectors),
                    plans=BUILTIN_PLANS,
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
                return BlueskyScanner(devices=lambda: build_devices())

            set_scanner_factory(_demo_scanner_factory)
            logger.info(
                "%s is enabled: wired the mock-devices demo BlueskyScanner (deploy smoke demo)",
                _DEMO_SCANNER_ENV,
            )
    yield


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


@app.get("/runs/{run_id}/data")
def read_run_data(
    run_id: str, max_rows: int = 100, offset: int | None = None, tail: bool = False
) -> dict:
    """Read a bounded window of a run's recorded data (see `live_rows.py`).

    Row-bounded by design — this never returns an unbounded table. Serves
    from the in-process live-row buffer, so reads work with no Tiled server:
    ``partial: true`` while the run is still filling in (before its stop doc
    lands), permanently readable once it's marked completed (see
    `live_rows.py`'s retention bound). ``row_count`` is the *true* total rows
    the run has produced so far, even if that's more than what's physically
    stored — ``truncated`` reflects whether this response's window omits any
    of them.

    Raises 409 if the run has no `run_uid` yet (never promoted, or promoted
    but the scan hasn't emitted a start doc) — there is nothing to read.
    """
    run = registry.get(run_id)
    run_uid = run.run_uid
    if run_uid is None:
        raise HTTPException(status_code=409, detail=f"run {run_id!r} has not started; no data yet")

    buf = live_rows.get(run_uid)
    if buf is None:
        return {"columns": [], "rows": []}

    rows = buf["rows"]
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

    truncated = start > 0 or end < stored_count or buf["total_seen"] > stored_count

    result: dict[str, Any] = {
        "run_uid": run_uid,
        "columns": list(buf["columns"]),
        "rows": window,
        "row_count": buf["total_seen"],
        "truncated": truncated,
    }
    if buf["partial"]:
        result["partial"] = True
    return result
