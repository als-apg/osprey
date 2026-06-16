"""Trigger-source plugin contract for the event dispatcher.

A trigger source watches for external events (webhook POSTs, cron ticks, EPICS
PV changes, …) and, for each event, invokes a fire callback supplied by the
dispatcher. The dispatcher owns the pool, registry, and on_error policy; a source
only detects events and forwards (trigger, payload) pairs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from osprey.dispatch.trigger_config import TriggerConfig

# Invoked by a source for each detected event. Returns the dispatch_id of the
# queued run, or None when the trigger was skipped (e.g. disabled). Raises
# osprey.dispatch.pool.QueueFullError when the dispatch pool is saturated.
FireCallback = Callable[["TriggerConfig", dict[str, Any]], Awaitable[str | None]]


@runtime_checkable
class TriggerSource(Protocol):
    """A pluggable event source. Discovered via the 'osprey.trigger_sources' entry-point group.

    Lifecycle is two-phase because FastMCP routes must be registered before the
    ASGI app is built (factory time), while background tasks need a running event
    loop (lifespan time):

      1. register_routes(mcp_app)  -- factory time, synchronous, before app build
      2. await start(triggers, fire_callback)  -- lifespan startup, running loop
      3. await stop()  -- lifespan shutdown
    """

    source_type: ClassVar[str]

    def register_routes(self, mcp_app: FastMCP) -> None:
        """Register any HTTP routes this source needs (factory time, before app build).

        Route-less sources implement this as a no-op. Routes registered after the
        FastMCP app is built are not served, so all route registration happens here.
        """
        ...

    async def start(self, triggers: list[TriggerConfig], fire_callback: FireCallback) -> None:
        """Begin generating events (lifespan startup, with a running event loop).

        Args:
            triggers: Triggers pre-filtered to this source's type.
            fire_callback: Coroutine invoked once per detected event.
        """
        ...

    async def stop(self) -> None:
        """Stop generating events and release any resources."""
        ...
