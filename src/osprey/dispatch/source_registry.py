"""Discovers trigger sources from entry points and orchestrates their lifecycle."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from osprey.dispatch.sources.base import TriggerSource

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from osprey.dispatch.sources.base import FireCallback
    from osprey.dispatch.trigger_config import TriggerConfig

logger = logging.getLogger("osprey.dispatch.source_registry")

ENTRY_POINT_GROUP = "osprey.trigger_sources"


class SourceRegistry:
    """Loads trigger-source classes and manages started source instances."""

    def __init__(self) -> None:
        self._source_classes: dict[str, type[TriggerSource]] = {}
        self._active: list[tuple[TriggerSource, list[TriggerConfig]]] = []

    def discover(self) -> None:
        """Load source classes from the ``osprey.trigger_sources`` entry-point group.

        Idempotent: re-discovering the same entry point (same name → same class)
        is a no-op. A name mapped to a DIFFERENT class raises ValueError.
        """
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            cls: type[TriggerSource] = ep.load()
            # ``TriggerSource`` has a non-method member (``source_type``), so it
            # cannot be used with ``issubclass``; duck-type the lifecycle instead.
            if not (
                isinstance(cls, type)
                and callable(getattr(cls, "register_routes", None))
                and callable(getattr(cls, "start", None))
                and callable(getattr(cls, "stop", None))
            ):
                raise TypeError(
                    f"Entry point {ep.name!r} loaded {cls!r}, which is not a "
                    "TriggerSource (missing register_routes/start/stop)"
                )
            existing = self._source_classes.get(ep.name)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"Conflicting trigger source for {ep.name!r}: {existing!r} vs {cls!r}"
                )
            self._source_classes[ep.name] = cls

    def setup(self, triggers: list[TriggerConfig], mcp_app: FastMCP) -> None:
        """Factory-time: group triggers by source type, instantiate each needed
        source, register its routes, and remember it for start_all().

        Triggers whose source has no registered class are logged and skipped.
        """
        grouped: dict[str, list[TriggerConfig]] = {}
        for trigger in triggers:
            grouped.setdefault(trigger.source, []).append(trigger)

        for source_type, group_triggers in grouped.items():
            cls = self._source_classes.get(source_type)
            if cls is None:
                logger.warning(
                    "No registered trigger source for %r; skipping %d trigger(s): %s",
                    source_type,
                    len(group_triggers),
                    [t.name for t in group_triggers],
                )
                continue
            inst = cls()
            inst.register_routes(mcp_app)
            self._active.append((inst, group_triggers))

    async def start_all(self, fire_callback: FireCallback) -> None:
        """Lifespan-time: start every source instantiated by setup()."""
        for inst, group in self._active:
            await inst.start(group, fire_callback)

    async def stop_all(self) -> None:
        """Stop every active source instance.

        Best-effort: a failure stopping one source is logged but does not prevent
        the remaining sources from being stopped (so e.g. cron asyncio tasks are
        not leaked when an earlier source's ``stop`` raises).
        """
        for inst, _group in self._active:
            try:
                await inst.stop()
            except Exception:
                logger.exception("Error stopping trigger source %r", inst)
        self._active = []
