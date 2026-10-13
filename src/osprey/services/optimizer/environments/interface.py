"""The engine's sync ``InterfaceProtocol`` over OSPREY's connector-host pool.

Every channel access the optimizer makes goes through
:class:`osprey_connectors.ipc.pool.ConnectorHostPool`: one connector-host child
per ``(control target, execution mode)``, each bound to its own Channel Access
routing. So the API process can serve several targets, and a job worker serves
the one its job names. Every write is ``write_channel_checked(confirm=True)``,
which re-checks the target's write posture and the channel limits (``max_step``
included) per put, and never retries.

The engine is synchronous and the pool is async and bound to one event loop, so
each process runs one :class:`_PoolBridge`: a daemon thread with that loop and
the pool on it. Calls block on ``run_coroutine_threadsafe(...).result()``.

Ported from tuning_scripts' ``als_plugin/osprey_interface.py``. The error
mapping is unchanged:

- ``ChannelWriteBlockedError`` (refused, never sent) becomes ``WriteRefusedError``.
- ``ChannelWriteFailedError`` (sent, not confirmed) becomes ``EvaluationError``,
  or ``ConnectionError`` when the channel cannot be reached at all.
- A read that never connects becomes ``ConnectionError``, and a read with no
  value becomes ``TimeoutError``.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
from collections.abc import Coroutine
from typing import Any

from osprey.services.optimizer.models.exceptions import ConnectionError as OptimizerConnectionError
from osprey.services.optimizer.models.exceptions import (
    EvaluationError,
    InterfaceError,
    WriteRefusedError,
)
from osprey.services.optimizer.models.exceptions import TimeoutError as OptimizerTimeoutError
from osprey.services.optimizer.models.protocols import PVReading
from osprey.services.optimizer.models.types import PVValues, VariableInputs

logger = logging.getLogger(__name__)

# Backstop on one bridged call (read, or write plus its confirming re-read).
# Longer than the connector's own timeouts so their typed errors arrive first.
OP_TIMEOUT: float = 30.0

CONTROL_TARGETS = ("va", "standin", "live")


def _to_native(value: Any) -> Any:
    """JSON-safe native value: numpy scalars and arrays to Python, NaN to None."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def _element_count(value: Any) -> int | None:
    return len(value) if isinstance(value, list) else 1 if value is not None else None


def _reading(cv: Any) -> PVReading:
    """Map an OSPREY ``ChannelValue`` onto a ``PVReading`` (metadata best-effort)."""
    md = cv.metadata
    raw = getattr(md, "raw_metadata", None) or {}
    ts = getattr(md, "timestamp", None) or cv.timestamp
    value = _to_native(cv.value)
    return PVReading(
        value=value,
        connected=True,
        units=getattr(md, "units", None) or None,
        precision=getattr(md, "precision", None),
        severity=raw.get("severity"),
        status=raw.get("status"),
        timestamp=ts.timestamp() if ts is not None else None,
        enum_strings=getattr(md, "enum_labels", None),
        count=_element_count(value),
    )


def pool_section(control_system: dict[str, Any]) -> dict[str, Any]:
    """The ``control_system`` section the optimizer's pool is built from.

    The project's own section, with ``fresh_reads`` on in every connector block:
    the engine reads a readback right after writing its setpoint, and the
    connector's default monitor-cache read can still hold the value from before
    the write.
    """
    section = copy.deepcopy(control_system)
    for block in (section.get("connector") or {}).values():
        if isinstance(block, dict):
            block["fresh_reads"] = True
    return section


class _PoolBridge:
    """One event-loop thread and one ``ConnectorHostPool`` for this process."""

    def __init__(self) -> None:
        from osprey_connectors.config import default_config_path, get_config_value
        from osprey_connectors.ipc.pool import ConnectorHostPool

        section = get_config_value("control_system", None)
        if not section:
            raise InterfaceError(
                "The project config has no control_system section, so the optimizer has no "
                "control target to reach. Set CONFIG_FILE to the deployment's config.yml."
            )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="optimizer-connector-pool", daemon=True
        )
        self._thread.start()
        self._pool = ConnectorHostPool(pool_section(section), config_file=default_config_path())
        self._connectors: dict[tuple[str, str | None], Any] = {}

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float = OP_TIMEOUT) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            # The operation's own TimeoutError (future done) keeps its type for the caller's mapping.
            if future.done():
                raise
            future.cancel()
            raise OptimizerTimeoutError(
                f"Control-system call did not complete within {timeout}s"
            ) from None

    def connector(self, target: str, execution_mode: str | None) -> Any:
        key = (target, execution_mode)
        if key not in self._connectors:
            self._connectors[key] = self.run(
                self._pool.connector(target, execution_mode=execution_mode)
            )
        return self._connectors[key]

    def close(self) -> None:
        try:
            self.run(self._pool.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()


_BRIDGE: _PoolBridge | None = None
_BRIDGE_LOCK = threading.Lock()


def get_bridge() -> _PoolBridge:
    """This process's pool bridge, built on first use."""
    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            _BRIDGE = _PoolBridge()
        return _BRIDGE


def reset_bridge() -> None:
    """Close and drop this process's bridge. For tests and worker shutdown."""
    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is not None:
            _BRIDGE.close()
            _BRIDGE = None


class ConnectorInterface:
    """``InterfaceProtocol`` for one control target, over the process's pool bridge."""

    def __init__(
        self, target: str, execution_mode: str | None = None, *, bridge: _PoolBridge | None = None
    ) -> None:
        if target not in CONTROL_TARGETS:
            raise InterfaceError(
                f"Unknown control target {target!r}; expected one of {CONTROL_TARGETS}"
            )
        if execution_mode not in (None, "readonly"):
            raise InterfaceError(
                f"Unknown execution mode {execution_mode!r}; expected None or 'readonly'"
            )
        self.target = target
        self.execution_mode = execution_mode
        self._bridge = bridge or get_bridge()
        self._connector = self._bridge.connector(target, execution_mode)

    def read_one(self, channel: str) -> PVReading:
        try:
            cv = self._bridge.run(self._connector.read_channel(channel))
        except TimeoutError as exc:
            raise OptimizerTimeoutError(f"Read timed out for {channel}") from exc
        except (ConnectionError, ValueError) as exc:
            raise OptimizerConnectionError(f"Read failed for {channel}: {exc}") from exc
        if cv.value is None:
            raise OptimizerTimeoutError(f"Read returned no value for {channel}")
        return _reading(cv)

    def read_many(self, channel_names: list[str]) -> dict[str, PVReading]:
        """One reading per requested name; a channel that failed is ``connected=False``."""
        if not channel_names:
            return {}
        try:
            results = self._bridge.run(self._connector.read_multiple_channels(channel_names))
        except TimeoutError as exc:
            raise OptimizerTimeoutError(
                f"Batch read timed out for {len(channel_names)} channels"
            ) from exc
        except (ConnectionError, ValueError) as exc:
            raise OptimizerConnectionError(f"Batch read failed: {exc}") from exc
        readings: dict[str, PVReading] = {}
        for name in channel_names:
            cv = results.get(name)
            if cv is None or cv.value is None:
                logger.warning("%s missing from batch read; marking disconnected", name)
                readings[name] = PVReading(value=None, connected=False)
            else:
                readings[name] = _reading(cv)
        return readings

    def set_values(self, channel_inputs: VariableInputs) -> PVValues:
        """Write each channel through the reference monitor; raise on the first failure."""
        from osprey_connectors.errors import ChannelWriteBlockedError, ChannelWriteFailedError

        outputs: PVValues = {}
        for channel, value in channel_inputs.items():
            try:
                self._bridge.run(
                    self._connector.write_channel_checked(channel, value, confirm=True)
                )
            except ChannelWriteBlockedError as exc:
                raise WriteRefusedError(
                    f"Write to {channel} refused [{exc.reason}]: {exc}", reason=exc.reason
                ) from exc
            except ChannelWriteFailedError as exc:
                if exc.reason == "FAILED" and not self.check_connection(channel):
                    raise OptimizerConnectionError(
                        f"Write connect failed for {channel}: {exc}"
                    ) from exc
                raise EvaluationError(f"Write to {channel} failed [{exc.reason}]: {exc}") from exc
            except TimeoutError as exc:
                raise OptimizerTimeoutError(f"Write timed out for {channel}") from exc
            except ConnectionError as exc:
                raise OptimizerConnectionError(
                    f"Write connect failed for {channel}: {exc}"
                ) from exc
            outputs[channel] = value
        return outputs

    def check_connection(self, pv_name: str | None = None) -> bool | None:
        if not pv_name:
            return None
        try:
            return bool(self._bridge.run(self._connector.validate_channel(pv_name)))
        except (ConnectionError, TimeoutError):
            return False
