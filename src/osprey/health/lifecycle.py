"""Runtime-lifecycle unit for long-lived health surfaces.

A health surface may run inside a long-lived, possibly daemon-thread-hosted
process — the web terminal's sidecar app or the stdio health MCP server.
:class:`HealthRuntimeLifecycle` owns the single
:class:`~osprey.health.runtime.HealthRuntime` for that process and the rules
that keep its control-system connector safe there:

* **Lazy, snapshot-driven construction.** The runtime is created from the first
  refresh's ``control_system`` config snapshot — the CLI's own guard expression
  ``(expanded or {}).get("control_system", {}) or {}`` — via :meth:`reconcile`,
  called once per refresh cycle.
* **Re-snapshot before first connect; restart-notice after.** While the runtime
  has never constructed a connector (``ever_constructed`` is False), a changed
  ``control_system`` mapping simply replaces the runtime — race-free and with
  zero Channel Access risk, so a broken-config first refresh cannot latch an
  empty mapping. Once a connector *has* been constructed, a changed mapping is
  never swapped in-process (an in-flight CA connector swap is a libca crash
  class in a shared host process); instead it is surfaced explicitly as a
  one-time warning log plus an informational result row on every subsequent
  report until the process restarts.
* **Loop-affine, refresh-serialized teardown.** The connector must be
  disconnected on the event loop that constructed it. Both the lifespan shutdown
  path (:meth:`shutdown`) and the ``atexit`` hook first cancel and await any
  in-flight refresh task, then run :meth:`HealthRuntime.shutdown` — whose
  ``closed`` flag prevents any post-teardown resurrection. The ``atexit`` hook
  no-ops unless a connector was constructed, submits to the owning loop via
  ``run_coroutine_threadsafe``, and wraps its bounded wait so a wedged teardown
  logs a single warning rather than dumping a traceback to stderr at exit.

The unit is deliberately free of FastAPI and cache/scheduler concerns: the app
factory drives loop binding and lifespan teardown, and the refresh engine calls
:meth:`reconcile` and reads :attr:`runtime`. The engine supplies its in-flight
refresh task through ``inflight_task_provider`` so teardown can serialize
against it.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
from collections.abc import Callable
from typing import Any

from osprey.health.models import CheckResult, Status
from osprey.health.runtime import HealthRuntime

logger = logging.getLogger("osprey.health.lifecycle")

#: Default surface-neutral restart hint. Surfaces pass their own via
#: ``HealthRuntimeLifecycle(restart_hint=...)`` so the notice names the process
#: the operator (or agent) can actually restart.
DEFAULT_RESTART_HINT = "restart the health service"


def restart_notice_message(restart_hint: str = DEFAULT_RESTART_HINT) -> str:
    """Compose the message shown (log + result row) when ``control_system``
    config changes after a connector has been constructed and cannot be swapped
    in-process."""
    return f"control_system config changed; {restart_hint} to apply"


#: The default-composed message, kept for callers/tests that don't customize the hint.
RESTART_NOTICE_MESSAGE = restart_notice_message()

#: Bounded wait for the ``atexit`` teardown to complete on the owning loop. Kept
#: as a module constant so a process exiting while the loop is wedged never hangs
#: longer than this; tests monkeypatch it to stay fast.
ATEXIT_SHUTDOWN_TIMEOUT_S = 5.0

# Type of the callable the engine supplies to expose its in-flight refresh task.
InflightTaskProvider = Callable[[], "asyncio.Task[Any] | None"]


def control_system_snapshot(expanded: dict[str, Any] | None) -> dict[str, Any]:
    """Return the ``control_system`` section from an expanded config mapping.

    Mirrors the CLI guard expression ``(expanded or {}).get("control_system",
    {}) or {}`` exactly: a missing config, a missing section, or an explicit
    ``control_system: null`` all normalize to an empty mapping.
    """
    return (expanded or {}).get("control_system", {}) or {}


def _restart_notice_row(message: str) -> CheckResult:
    """Build the informational row surfaced after an unapplied config change."""
    return CheckResult(
        name="control_system",
        category="configuration",
        status=Status.WARNING,
        message=message,
    )


class HealthRuntimeLifecycle:
    """Owns the health sidecar's single connector runtime and its teardown.

    Args:
        inflight_task_provider: Optional callable returning the refresh engine's
            current in-flight suite task (or ``None`` when idle). Teardown
            cancels and awaits it before disconnecting the connector so the two
            never race. May instead be supplied later via
            :meth:`set_inflight_task_provider`.
        restart_hint: Surface-specific phrase naming the process to restart when
            a ``control_system`` change cannot be applied in-process (e.g.
            ``"restart the web terminal"``). Composed into the notice row and
            warning log via :func:`restart_notice_message`.
    """

    def __init__(
        self,
        inflight_task_provider: InflightTaskProvider | None = None,
        *,
        restart_hint: str = DEFAULT_RESTART_HINT,
    ) -> None:
        self.restart_notice_message = restart_notice_message(restart_hint)
        self._runtime: HealthRuntime | None = None
        # The ``control_system`` mapping the current runtime was built from.
        self._active_control_system: dict[str, Any] | None = None
        # The mapping we last emitted a restart-notice *log* for (dedupes the
        # warning while the result row still renders on every report).
        self._noticed_control_system: dict[str, Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._atexit_registered = False
        self._inflight_task_provider = inflight_task_provider

    @property
    def runtime(self) -> HealthRuntime | None:
        """The current runtime, or ``None`` before the first :meth:`reconcile`."""
        return self._runtime

    @property
    def atexit_registered(self) -> bool:
        """Whether the process-exit teardown hook is currently registered."""
        return self._atexit_registered

    def set_inflight_task_provider(self, provider: InflightTaskProvider | None) -> None:
        """Set (or clear) the callable exposing the engine's in-flight task."""
        self._inflight_task_provider = provider

    # -- snapshot / re-snapshot -------------------------------------------------

    def reconcile(self, expanded: dict[str, Any] | None) -> list[CheckResult]:
        """Reconcile the runtime with the latest config snapshot.

        Called once per refresh cycle with the just-loaded expanded config. On
        the first call it constructs the runtime from the ``control_system``
        snapshot. On later calls:

        * an unchanged snapshot is a no-op;
        * a changed snapshot while no connector was ever constructed silently
          replaces the runtime with one built from the new snapshot;
        * a changed snapshot after a connector was constructed logs a one-time
          warning and returns the restart-notice row, leaving the live runtime
          untouched.

        Returns the extra result rows the engine should append to the report —
        empty except for the post-construction restart-notice case, where the
        single notice row is returned on every subsequent divergent refresh.
        """
        snapshot = control_system_snapshot(expanded)

        if self._runtime is None:
            self._runtime = HealthRuntime(snapshot)
            self._active_control_system = snapshot
            return []

        if snapshot == self._active_control_system:
            # Back in sync with the live runtime (or never diverged): clear the
            # log-dedupe latch so a future change warns again.
            self._noticed_control_system = None
            return []

        if not self._runtime.ever_constructed:
            # No Channel Access client was ever created — dropping the old
            # runtime and re-snapshotting is race-free and CA-risk-free.
            self._runtime = HealthRuntime(snapshot)
            self._active_control_system = snapshot
            self._noticed_control_system = None
            return []

        # A connector is live; the config changed. Never swap in-process — warn
        # once and surface the notice row until the process restarts.
        if snapshot != self._noticed_control_system:
            logger.warning("HealthRuntime: %s", self.restart_notice_message)
            self._noticed_control_system = snapshot
        return [_restart_notice_row(self.restart_notice_message)]

    # -- teardown ---------------------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Capture the owning event loop (call from lifespan startup on the loop).

        With no argument the currently running loop is captured; an explicit loop
        may be passed (used by tests). The captured loop is the one the ``atexit``
        hook submits teardown to, guaranteeing connector disconnect runs on the
        thread that constructed it.
        """
        self._loop = loop or asyncio.get_running_loop()

    def register_atexit(self) -> None:
        """Register the process-exit teardown hook (idempotent)."""
        if self._atexit_registered:
            return
        atexit.register(self._atexit_shutdown)
        self._atexit_registered = True

    def unregister_atexit(self) -> None:
        """Unregister the process-exit teardown hook (idempotent).

        Called on clean lifespan shutdown so short-lived ``TestClient`` apps do
        not stack hooks across the process lifetime.
        """
        if not self._atexit_registered:
            return
        atexit.unregister(self._atexit_shutdown)
        self._atexit_registered = False

    async def shutdown(self) -> None:
        """Lifespan-shutdown teardown, run on the owning loop's thread.

        Cancels and awaits any in-flight refresh task, disconnects the connector
        exactly once (a never-constructed runtime is a no-op), then unregisters
        the ``atexit`` hook. Idempotent and exception-safe: a runtime whose
        ``disconnect`` raises is swallowed by :meth:`HealthRuntime.shutdown`.
        """
        await self._cancel_inflight()
        if self._runtime is not None:
            await self._runtime.shutdown()
        self.unregister_atexit()

    async def _cancel_inflight(self) -> None:
        """Cancel and await the engine's in-flight refresh task, if any.

        Best-effort: a task that raises on cancellation (or otherwise) must not
        prevent teardown from proceeding to the connector disconnect.
        """
        if self._inflight_task_provider is None:
            return
        task = self._inflight_task_provider()
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug(
                "HealthRuntime: in-flight refresh raised during teardown (ignored)",
                exc_info=True,
            )

    def _atexit_shutdown(self) -> None:
        """Process-exit hook: disconnect the connector on the owning loop.

        No-ops unless a connector was actually constructed (nothing to
        disconnect otherwise, and no reason to touch the loop). Submits the
        teardown coroutine to the owning loop via ``run_coroutine_threadsafe``
        and waits at most :data:`ATEXIT_SHUTDOWN_TIMEOUT_S`. On timeout or any
        error it logs a single warning and returns — never a stderr traceback at
        interpreter exit.
        """
        runtime = self._runtime
        if runtime is None or not runtime.ever_constructed:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._teardown_connector(), loop)
            future.result(timeout=ATEXIT_SHUTDOWN_TIMEOUT_S)
        except Exception:
            # exc_info deliberately omitted: one clean line, no exit-time traceback.
            logger.warning("HealthRuntime: connector teardown at exit did not complete cleanly")

    async def _teardown_connector(self) -> None:
        """Cancel the in-flight refresh, then disconnect the connector once."""
        await self._cancel_inflight()
        if self._runtime is not None:
            await self._runtime.shutdown()
