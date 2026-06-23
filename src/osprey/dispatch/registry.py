"""In-memory trigger registry with per-trigger event history."""

import asyncio
import json
from collections import deque
from datetime import UTC, datetime
from typing import Any

from osprey.dispatch.trigger_config import TriggerConfig

_HISTORY_MAX = 1000
_MAX_EVENT_DATA_BYTES = 64 * 1024
_VALID_STATUSES = {"active", "disabled"}


class TriggerRegistry:
    """Stores active triggers and their event history."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._triggers: dict[str, TriggerConfig] = {}
        self._status: dict[str, str] = {}
        self._last_fired: dict[str, str | None] = {}
        self._history: dict[str, deque] = {}

    async def register(self, trigger: TriggerConfig) -> None:
        """Register a trigger. Re-registration replaces the existing entry."""
        async with self._lock:
            self._triggers[trigger.name] = trigger
            self._status[trigger.name] = "active"
            self._last_fired[trigger.name] = None
            if trigger.name not in self._history:
                self._history[trigger.name] = deque(maxlen=_HISTORY_MAX)

    async def list_triggers(self) -> list[dict]:
        """Return summary dicts for all registered triggers."""
        async with self._lock:
            return [
                {
                    "name": name,
                    "source": cfg.source,
                    "status": self._status[name],
                    "last_fired": self._last_fired[name],
                }
                for name, cfg in self._triggers.items()
            ]

    async def get_status(self, name: str) -> dict:
        """Return status dict for a single trigger. Raises KeyError if not found."""
        async with self._lock:
            if name not in self._triggers:
                raise KeyError(f"Trigger '{name}' not registered")
            cfg = self._triggers[name]
            return {
                "name": name,
                "source": cfg.source,
                "status": self._status[name],
                "last_fired": self._last_fired[name],
            }

    async def set_status(self, name: str, status: str) -> str:
        """Update a trigger's status. Returns the new status."""
        if status not in _VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}; expected one of {_VALID_STATUSES}")
        async with self._lock:
            if name not in self._triggers:
                raise KeyError(f"Trigger '{name}' not registered")
            self._status[name] = status
            return status

    async def record_event(self, name: str, event_data: dict[str, Any], result: str) -> None:
        """Append an event to the trigger's history and update last_fired timestamp.

        Large payloads are truncated to _MAX_EVENT_DATA_BYTES to bound RAM usage
        given the per-trigger history cap of _HISTORY_MAX entries.
        """
        async with self._lock:
            if name not in self._triggers:
                raise KeyError(f"Trigger '{name}' not registered")
            ts = datetime.now(tz=UTC).isoformat()
            self._last_fired[name] = ts

            try:
                data_json = json.dumps(event_data, default=str)
            except (TypeError, ValueError):
                data_json = str(event_data)
            if len(data_json) > _MAX_EVENT_DATA_BYTES:
                event_data = {
                    "_truncated": True,
                    "_original_size": len(data_json),
                    "_preview": data_json[:4096],
                }

            self._history[name].append(
                {"timestamp": ts, "event_data": event_data, "result": result}
            )

    async def get_history(
        self,
        name: str,
        limit: int = 50,
        since_seconds: float | None = None,
    ) -> list[dict]:
        """Return the most recent `limit` events for a trigger (newest-last order).

        If `since_seconds` is given, entries older than that many seconds ago
        are filtered out before applying `limit`.
        """
        async with self._lock:
            if name not in self._triggers:
                raise KeyError(f"Trigger '{name}' not registered")
            entries = list(self._history[name])
            if since_seconds is not None:
                cutoff = datetime.now(tz=UTC).timestamp() - since_seconds
                filtered = []
                for entry in entries:
                    try:
                        entry_ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                    except (KeyError, ValueError):
                        continue
                    if entry_ts >= cutoff:
                        filtered.append(entry)
                entries = filtered
            return entries[-limit:] if limit > 0 else entries
