"""The real bluesky-backed ``Scanner``: a RunEngine in a daemon thread.

Implements the injected ``Scanner`` seam (``scanner.py``) that ``do_promote``
builds per promoted run. Imports bluesky, so this module lives behind the
optional ``osprey-framework[bluesky-bridge]`` extra — it must never be
imported from the lifecycle core's import path (``app.py``, ``runs.py``,
``scanner.py``, ``security.py`` stay import-clean); only a deploy wiring or a
bluesky-capable test imports this module directly.

Plan resolution happens entirely inside ``reinitialize()``: ``exec_config``
(the bridge's ``RunRequest`` — ``plan_name`` + ``plan_args``, or an equivalent
mapping) is resolved against a plan registry (by default the same built-ins +
facility-injected merge ``app.py``'s ``GET /plans`` route serves), its
``plan_args`` validated against that plan's pydantic schema, and the
resulting bluesky plan generator built and stored — all *before*
``start_scan_thread()`` ever touches a thread. That is what makes
``start_scan_thread()`` effectively all-or-nothing: the only way it can fail
is daemon-thread creation itself, not plan/device resolution (see
``runs.py``'s ``do_promote``, which also stops a partially-started scanner
defensively on any exception, as a second line of defense).

Live data: this scanner subscribes its own ``live_rows.LiveRowRecorder`` to
the RunEngine, so ``read_scan_data`` returns real buffered rows for a real
scan with no Tiled server needed. A ``TiledWriter`` subscription is optional
(best-effort — its absence, or failure to connect, never blocks a scan).

``error_message`` is the Scanner protocol's explicit terminal-error signal
(see ``scanner.py``): set on a plan-resolution failure in ``reinitialize()``
or on an uncaught exception from the RunEngine thread in ``start_scan_thread()``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from bluesky import RunEngine

from .live_rows import LiveRowRecorder
from .plan_types import PlanSpec

logger = logging.getLogger("osprey.services.bluesky_bridge.scanner_bluesky")

# A device mapping, or a zero-arg callable producing one (sync) or an
# awaitable of one (async) — mirrors `plan_loader.py`'s `get_devices() ->
# dict[str, Any]` contract, so the same device source shape works whether it
# comes from a facility plan module, the mock factory (`devices/mock.py`), or
# a plain pre-built dict (tests). The callable may be sync (a plain
# `Mapping[str, Any]` return) OR async (e.g. `devices.mock.build_devices`/
# `devices.epics.build_devices`, which connect ophyd-async devices and are
# `async def`) — `reinitialize()` bridges either via `_resolve_devices`.
DeviceSource = Mapping[str, Any] | Callable[[], "Mapping[str, Any] | Awaitable[Mapping[str, Any]]"]

# How long `_resolve_devices` waits for an async device factory to connect on
# `RE.loop` before giving up. Generous enough to cover real Channel Access
# connect timeouts (IOC startup, network hiccups), not so long that a launch
# request hangs indefinitely on a dead IOC.
CONNECT_TIMEOUT = 30.0


async def _await(awaitable: Awaitable[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Wrap an arbitrary awaitable as a coroutine `run_coroutine_threadsafe` accepts."""
    return await awaitable


def _default_plan_registry() -> dict[str, PlanSpec[Any]]:
    """Built-ins merged with facility-injected plans (mirrors app.py's `/plans` merge).

    Duplicated locally rather than imported from `app.py` (this module must
    never be imported by the lifecycle core, and `app.py` must never import
    this module either — the dependency only runs one way, from a deploy
    wiring or a test towards this module).
    """
    from .plan_loader import get_facility_plans

    merged: dict[str, PlanSpec[Any]] = {}
    try:
        from .plans import BUILTIN_PLANS

        merged.update(BUILTIN_PLANS)
    except ImportError:
        pass
    merged.update(get_facility_plans().plans)
    return merged


def _plan_name_and_args(exec_config: Any) -> tuple[str | None, dict[str, Any]]:
    """Extract ``(plan_name, plan_args)`` from either a `RunRequest` or a plain dict."""
    if isinstance(exec_config, Mapping):
        return exec_config.get("plan_name"), dict(exec_config.get("plan_args") or {})
    return getattr(exec_config, "plan_name", None), dict(
        getattr(exec_config, "plan_args", None) or {}
    )


class BlueskyScanner:
    """One promoted run's real bluesky RunEngine handle.

    A fresh instance is built per promotion (`do_promote`'s `scanner_factory`).
    ``devices``/``plans`` are injected rather than resolved from global state
    by default, so contract tests can supply mock devices (`devices/mock.py`)
    and the real built-in plan set without any facility injection wiring.
    """

    def __init__(
        self,
        devices: DeviceSource,
        plans: Mapping[str, PlanSpec[Any]] | None = None,
        *,
        tiled_writer_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._devices_source = devices
        self._plans = dict(plans) if plans is not None else None
        self._plan_gen: Any = None

        self.current_state: str = "idle"
        self.last_run_uid: str | None = None
        self.error_message: str | None = None

        # `context_managers=[]` disables the RunEngine's default SIGINT/SIGTSTP
        # handling, which only works on the main thread — required since
        # `RE(...)` runs from `self._thread`, not the process's main thread.
        self.RE = RunEngine(context_managers=[])
        self._recorder = LiveRowRecorder()
        self.RE.subscribe(self._on_document)

        if tiled_writer_factory is not None:
            try:
                self.RE.subscribe(tiled_writer_factory())
            except Exception:
                logger.warning(
                    "BlueskyScanner: TiledWriter subscription failed; continuing without it",
                    exc_info=True,
                )

        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._completion = 0.0

    def _on_document(self, name: str, doc: dict[str, Any]) -> None:
        """Capture `last_run_uid` from the start doc, then forward to the live-row buffer.

        Never raises: `dict.get` is safe, and `LiveRowRecorder.__call__` is
        already fully exception-wrapped (see `live_rows.py`) — a recorder bug
        must never reach the RunEngine thread running the actual scan.
        """
        if name == "start":
            self.last_run_uid = doc.get("uid")
        self._recorder(name, doc)

    def _resolve_devices(self, source: DeviceSource) -> Mapping[str, Any]:
        """Resolve a `DeviceSource` to a plain device mapping, bridging async factories.

        `plan_loader.py`'s facility contract calls `get_devices()` synchronously,
        but the actual device factories (`devices/mock.py`'s `build_devices`, and
        `devices/epics.py`'s ophyd-async equivalent) are `async def` — connecting
        a device is an async operation. aioca binds a device's CA monitors to
        whichever event loop is *running* when `Device.connect()` awaits them,
        and bluesky drives all signal I/O for a scan on `self.RE.loop` (running
        in the RunEngine's own daemon thread, started at `RunEngine.__init__`
        time — already alive here, well before `start_scan_thread()`). So an
        async factory must be awaited *on that loop*, not a throwaway one, or
        its monitors end up bound to a loop the scan itself never touches.
        `asyncio.run_coroutine_threadsafe` schedules the await onto `self.RE.loop`
        from whichever thread calls `reinitialize()` (the promote HTTP request
        thread) and blocks that thread for the result — the sync-`Mapping`
        branch below needs none of this, since there's nothing to connect.
        """
        result: Mapping[str, Any] | Awaitable[Mapping[str, Any]] = (
            source() if callable(source) else source
        )
        if inspect.isawaitable(result):
            future = asyncio.run_coroutine_threadsafe(_await(result), self.RE.loop)
            try:
                return future.result(timeout=CONNECT_TIMEOUT)
            except TimeoutError:
                future.cancel()
                raise
        return result

    def reinitialize(self, exec_config: Any) -> bool:
        """Resolve `exec_config`'s plan_name/plan_args into a ready-to-run plan generator.

        Every failure mode here — unknown plan, invalid plan_args, a plan
        callable that raises while resolving devices — returns False (and
        sets `error_message`) rather than raising, so a bad launch request
        never reaches `start_scan_thread()` at all.
        """
        plan_name, plan_args = _plan_name_and_args(exec_config)
        if not plan_name:
            self.current_state = "error"
            self.error_message = "exec_config has no plan_name"
            return False

        plans = self._plans if self._plans is not None else _default_plan_registry()
        spec = plans.get(plan_name)
        if spec is None:
            self.current_state = "error"
            self.error_message = f"unknown plan {plan_name!r}"
            return False

        try:
            params = spec.schema.model_validate(plan_args)
            devices = self._resolve_devices(self._devices_source)
            self._plan_gen = spec.plan(dict(devices), params)
        except Exception as exc:
            self.current_state = "error"
            self.error_message = f"failed to build plan {plan_name!r}: {exc}"
            return False

        self.current_state = "armed"
        self.error_message = None
        self._stop_requested = False
        self._completion = 0.0
        return True

    def start_scan_thread(self) -> None:
        """Run the plan built by `reinitialize()` in a daemon thread. Non-blocking.

        Deliberately minimal: all plan resolution/validation already happened
        in `reinitialize()`, so the only way this raises is thread creation
        itself failing — not a partially-built scan.
        """
        if self._plan_gen is None:
            raise RuntimeError("start_scan_thread() called before a successful reinitialize()")

        self.current_state = "running"
        self._thread = threading.Thread(target=self._run, name="bluesky-scanner", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """The daemon thread body: drive the RunEngine, classify how it ended.

        `RE.abort()` called from another thread (see `stop_scanning_thread`)
        interrupts `RE(...)` here — depending on bluesky version/timing this
        surfaces either as `RE(...)` returning normally (an "abort" exit
        status recorded in the stop doc) or as an exception out of `RE(...)`
        (e.g. `RunEngineInterrupted`, confirmed against a real RunEngine).
        Either way, `_stop_requested` (set before `RE.abort()` is called) is
        what distinguishes an intentional stop from a genuine plan failure —
        not the exception's presence.
        """
        try:
            self.RE(self._plan_gen)
        except Exception as exc:
            if self._stop_requested:
                self.current_state = "stopped"
                self._completion = 1.0
            else:
                logger.warning("BlueskyScanner: RunEngine plan failed", exc_info=True)
                self.error_message = str(exc)
                self.current_state = "error"
        else:
            self.current_state = "stopped" if self._stop_requested else "completed"
            self._completion = 1.0

    def stop_scanning_thread(self) -> None:
        """Abort the running scan. Safe to call even if not active.

        `RunEngine.abort()`/`.pause()`/`.resume()` are bluesky's documented
        cross-thread control primitives — calling `abort()` from a thread
        other than the one running `RE(...)` is the supported way to stop a
        RunEngine driven in a background thread (as this one is).
        """
        self._stop_requested = True
        try:
            self.RE.abort(reason="stop_scanning_thread called")
        except Exception:
            logger.debug("BlueskyScanner: RE.abort() raised (may not be running)", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def is_scanning_active(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def estimate_current_completion(self) -> float:
        return self._completion
