"""The real bluesky-backed ``PlanRunner``: a RunEngine in a daemon thread.

Implements the injected ``PlanRunner`` seam (``plan_runner.py``) that ``do_launch``
builds per launched run. Imports bluesky, so this module lives behind the
optional ``osprey-framework[bluesky-bridge]`` extra — it must never be
imported from the lifecycle core's import path (``app.py``, ``runs.py``,
``plan_runner.py``, ``security.py`` stay import-clean); only a deploy wiring or a
bluesky-capable test imports this module directly.

Plan resolution happens entirely inside ``reinitialize()``: ``exec_config``
(the bridge's ``RunRequest`` — ``plan_name`` + ``plan_args``, or an equivalent
mapping) is resolved against a plan registry (by default the same registry
``app.py``'s ``GET /plans`` route serves — see ``plan_loader.get_facility_plans``),
its ``plan_args`` validated against that plan's pydantic schema, and the
resulting bluesky plan generator built and stored — all *before*
``start_run_thread()`` ever touches a thread. That is what makes
``start_run_thread()`` effectively all-or-nothing: the only way it can fail
is daemon-thread creation itself, not plan/device resolution (see
``runs.py``'s ``do_launch``, which also stops a partially-started runner
defensively on any exception, as a second line of defense).

Live data: this runner subscribes its own ``live_rows.LiveRowRecorder`` to
the RunEngine, so ``get_run_data`` returns real buffered rows for a real
run with no Tiled server needed. A ``TiledWriter`` subscription is optional
(best-effort — its absence, or failure to connect, never blocks a run).

``error_message`` is the PlanRunner protocol's explicit terminal-error signal
(see ``plan_runner.py``): set on a plan-resolution failure in ``reinitialize()``
or on an uncaught exception from the RunEngine thread in ``start_run_thread()``.
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

logger = logging.getLogger("osprey.services.bluesky_bridge.plan_runner_bluesky")

# A device mapping, or a zero-arg callable producing one (sync) or an
# awaitable of one (async) — mirrors `plan_loader.py`'s `get_devices() ->
# dict[str, Any]` contract, so the same device source shape works whether it
# comes from a facility plan module, the mock factory (`devices/mock.py`), or
# a plain pre-built dict (tests). The callable may be sync (a plain
# `Mapping[str, Any]` return) OR async (e.g. `devices.mock.build_devices`/
# `devices.connector.build_devices`, which connect ophyd-async devices and
# are `async def`) — `reinitialize()` bridges either via `_resolve_devices`.
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
    """The bridge's sole plan registry (mirrors app.py's `/plans` route).

    Duplicated locally rather than imported from `app.py` (this module must
    never be imported by the lifecycle core, and `app.py` must never import
    this module either — the dependency only runs one way, from a deploy
    wiring or a test towards this module).
    """
    from .plan_loader import get_facility_plans

    return dict(get_facility_plans().plans)


def _plan_name_and_args(exec_config: Any) -> tuple[str | None, dict[str, Any]]:
    """Extract ``(plan_name, plan_args)`` from either a `RunRequest` or a plain dict."""
    if isinstance(exec_config, Mapping):
        return exec_config.get("plan_name"), dict(exec_config.get("plan_args") or {})
    return getattr(exec_config, "plan_name", None), dict(
        getattr(exec_config, "plan_args", None) or {}
    )


class _FaultIsolatedTiledWriter:
    """Wraps a `(name, doc)` callback so a Tiled outage degrades persistence, never a run.

    `TiledWriter` is a synchronous RunEngine callback, and bluesky's
    `RunEngine` does not swallow callback exceptions by default — an
    exception escaping `__call__` aborts the running plan. This wrapper
    catches any exception from the inner writer, latches `degraded = True`,
    logs once with `exc_info`, and never re-raises. Once latched, subsequent
    documents short-circuit without touching the inner writer again.

    Latch semantics are per-instance: `BlueskyPlanRunner` builds a fresh writer
    (and fresh wrapper) per launch, so there is no cross-run state to reset.
    """

    def __init__(self, inner: Callable[[str, dict[str, Any]], Any]) -> None:
        self._inner = inner
        self.degraded = False

    def __call__(self, name: str, doc: dict[str, Any]) -> None:
        if self.degraded:
            return
        try:
            self._inner(name, doc)
        except Exception:
            self.degraded = True
            logger.error(
                "BlueskyPlanRunner: TiledWriter failed on %r document; persistence degraded",
                name,
                exc_info=True,
            )


class BlueskyPlanRunner:
    """One launched run's real bluesky RunEngine handle.

    A fresh instance is built per launch (`do_launch`'s `runner_factory`).
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

        # `None` means Tiled is disabled entirely (no factory supplied) —
        # distinct from a wired-but-degraded writer (see `tiled_degraded`).
        self._tiled_writer: _FaultIsolatedTiledWriter | None = None
        if tiled_writer_factory is not None:
            # Pre-latch a no-op placeholder so `tiled_degraded` has something
            # to read even if `tiled_writer_factory()` itself raises below;
            # reassigned to wrap the real inner writer the moment it exists.
            self._tiled_writer = _FaultIsolatedTiledWriter(lambda name, doc: None)
            try:
                # BOTH construction and subscription are inside this guard: a
                # Tiled server down at launch time can fail either call, and
                # either failure must degrade persistence, never abort the
                # launch (FR4) — a raising `subscribe()` outside the guard
                # would propagate out of `__init__` -> `runner_factory()` ->
                # `do_launch`, turning a Tiled outage into a failed launch.
                inner_writer = tiled_writer_factory()
                self._tiled_writer = _FaultIsolatedTiledWriter(inner_writer)
                self.RE.subscribe(self._tiled_writer)
            except Exception:
                logger.warning(
                    "BlueskyPlanRunner: TiledWriter construction/subscription failed;"
                    " persistence degraded",
                    exc_info=True,
                )
                # Whichever wrapper is currently referenced (the placeholder,
                # if the factory itself raised, or the real one, if only
                # `subscribe` raised) is latched degraded either way — it is
                # already a no-op past this point even if `subscribe` had
                # partially registered it before raising.
                self._tiled_writer.degraded = True

        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._completion = 0.0

    @property
    def tiled_degraded(self) -> bool:
        """Whether Tiled persistence has failed (construction or per-document).

        `False` both when Tiled is disabled entirely (no `tiled_writer_factory`
        supplied) and while a wired writer is healthy — never `True` merely
        because Tiled is absent. Reflects the fault-isolated wrapper's latch
        once a writer is wired.
        """
        if self._tiled_writer is None:
            return False
        return self._tiled_writer.degraded

    def _on_document(self, name: str, doc: dict[str, Any]) -> None:
        """Capture `last_run_uid` from the start doc, then forward to the live-row buffer.

        Never raises: `dict.get` is safe, and `LiveRowRecorder.__call__` is
        already fully exception-wrapped (see `live_rows.py`) — a recorder bug
        must never reach the RunEngine thread running the actual run.
        """
        if name == "start":
            self.last_run_uid = doc.get("uid")
        self._recorder(name, doc)

    def _resolve_devices(self, source: DeviceSource) -> Mapping[str, Any]:
        """Resolve a `DeviceSource` to a plain device mapping, bridging async factories.

        `plan_loader.py`'s facility contract calls `get_devices()` synchronously,
        but the actual device factories (`devices/mock.py`'s `build_devices`, and
        `devices/connector.py`'s connector-mediated equivalent) are `async def`
        — connecting a device is an async operation, and the resulting devices'
        async `set()`/`read()` methods run on whichever event loop is *running*
        when they're awaited. bluesky drives all signal I/O for a run on
        `self.RE.loop` (running in the RunEngine's own daemon thread, started
        at `RunEngine.__init__` time — already alive here, well before
        `start_run_thread()`). So an async factory must be awaited *on that
        loop*, not a throwaway one, or the devices it builds end up bound to a
        loop the run itself never touches.
        `asyncio.run_coroutine_threadsafe` schedules the await onto `self.RE.loop`
        from whichever thread calls `reinitialize()` (the launch HTTP request
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
        never reaches `start_run_thread()` at all.
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

    def start_run_thread(self) -> None:
        """Run the plan built by `reinitialize()` in a daemon thread. Non-blocking.

        Deliberately minimal: all plan resolution/validation already happened
        in `reinitialize()`, so the only way this raises is thread creation
        itself failing — not a partially-built run.
        """
        if self._plan_gen is None:
            raise RuntimeError("start_run_thread() called before a successful reinitialize()")

        self.current_state = "running"
        self._thread = threading.Thread(target=self._run, name="bluesky-runner", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """The daemon thread body: drive the RunEngine, classify how it ended.

        `RE.abort()` called from another thread (see `stop_run_thread`)
        interrupts `RE(...)` here — depending on bluesky version/timing this
        surfaces either as `RE(...)` returning normally (an "abort" exit
        status recorded in the stop doc) or as an exception out of `RE(...)`
        (e.g. `RunEngineInterrupted`, confirmed against a real RunEngine).
        Either way, `_stop_requested` (set before `RE.abort()` is called) is
        what distinguishes an intentional stop from a genuine plan failure —
        not the exception's presence.

        `osprey_run_id` (set by `do_launch`, `runs.py`, after `reinitialize`
        and before `start_run_thread`) is not part of the `PlanRunner` Protocol,
        so it is read defensively via `getattr` — absent for any caller that
        doesn't set it (contract tests constructing a `BlueskyPlanRunner`
        directly). When present it is passed as a `RunEngine.__call__`
        metadata kwarg — `RE(plan, osprey_run_id=...)` — which bluesky records
        onto the start doc. A literal `md={...}` would nest it under an `md`
        key instead, which is not what the durable Tiled lookup expects.
        """
        osprey_run_id = getattr(self, "osprey_run_id", None)
        metadata_kw = {"osprey_run_id": osprey_run_id} if osprey_run_id is not None else {}
        try:
            self.RE(self._plan_gen, **metadata_kw)
        except Exception as exc:
            if self._stop_requested:
                self.current_state = "stopped"
                self._completion = 1.0
            else:
                logger.warning("BlueskyPlanRunner: RunEngine plan failed", exc_info=True)
                self.error_message = str(exc)
                self.current_state = "error"
        else:
            self.current_state = "stopped" if self._stop_requested else "completed"
            self._completion = 1.0

    def stop_run_thread(self) -> None:
        """Abort the running plan. Safe to call even if not active.

        `RunEngine.abort()`/`.pause()`/`.resume()` are bluesky's documented
        cross-thread control primitives — calling `abort()` from a thread
        other than the one running `RE(...)` is the supported way to stop a
        RunEngine driven in a background thread (as this one is).
        """
        self._stop_requested = True
        try:
            self.RE.abort(reason="stop_run_thread called")
        except Exception:
            logger.debug("BlueskyPlanRunner: RE.abort() raised (may not be running)", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def is_run_active(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def estimate_current_completion(self) -> float:
        return self._completion
