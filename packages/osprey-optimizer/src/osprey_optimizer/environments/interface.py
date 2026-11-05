"""The engine's sync ``InterfaceProtocol`` over OSPREY's connector-host pool.

Every channel access an environment file makes goes through
:class:`osprey_connectors.ipc.ConnectorHostPool`: one connector-host child
per ``(control target, execution mode)``, each bound to its own Channel Access
routing. So the API process can serve several targets, and a job worker serves
the one its job names. Every write is ``write_channel_checked``, which
re-checks the target's write posture and the channel limits (``max_step``
included) per put, confirms it as the limits database says, and never retries.
A plugin (opt-in through ``$OPTIMIZER_PLUGINS``) must reach the machine through
this interface too, but nothing yet enforces that for plugin code.

**One write call per set.** :meth:`ConnectorInterface.set_values` sends its
whole set of channels as one ``write_multiple_channels`` call, so one
evaluation is one call, and under a program's write authorization one step
(:mod:`osprey_connectors.write_authorization`). The connector child is the one
for the owner and the authorization the current :class:`~osprey_optimizer.environments.write_authority.WriteAuthority`
names; reads go to the owner's child that names none. A refusal whose verdict
reason is an ``auth_*`` word ends the authority and raises
:class:`~osprey_optimizer.models.exceptions.AuthorizationEndedError`, and an
authority that requires an authorization it does not hold refuses the write
before any connector is asked.

The engine is synchronous and the pool is async and bound to one event loop, so
each process runs one :class:`_PoolBridge`: a daemon thread with that loop and
the pool on it. Calls block on ``run_coroutine_threadsafe(...).result()``.

Ported from tuning_scripts' facility plugin interface. The error
mapping is unchanged:

- ``ChannelWriteBlockedError`` (refused, never sent) becomes ``WriteRefusedError``,
  or ``AuthorizationEndedError`` for an ``auth_*`` verdict reason.
- ``ChannelWriteFailedError`` (sent, not confirmed) becomes ``EvaluationError``,
  or ``ConnectionError`` when the channel cannot be reached at all.
- A read that never connects becomes ``ConnectionError``, and a read with no
  value becomes ``TimeoutError``.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import threading
from collections.abc import Coroutine
from typing import Any

from osprey_optimizer.environments.write_authority import (
    NO_AUTHORIZATION,
    WriteAuthority,
    current_write_authority,
)
from osprey_optimizer.models.exceptions import (
    AuthorizationEndedError,
    EvaluationError,
    InterfaceError,
    WriteRefusedError,
)
from osprey_optimizer.models.exceptions import ConnectionError as OptimizerConnectionError
from osprey_optimizer.models.exceptions import TimeoutError as OptimizerTimeoutError
from osprey_optimizer.models.protocols import PVReading
from osprey_optimizer.models.types import PVValues, VariableInputs

logger = logging.getLogger(__name__)

# Backstop on one bridged call (read, or write plus its confirming re-read).
# Longer than the pool's own deadline for a call with no timeout
# (osprey_connectors.ipc.DEFAULT_CALL_DEADLINE_S, 60 s) plus its ping and the
# kill that follows, so the pool's typed errors arrive first and its kill of a
# child whose write timed out is never cut short by this one.
OP_TIMEOUT: float = 75.0

CONTROL_TARGETS = ("va", "standin", "live")


def _to_native(value: Any) -> Any:
    """JSON-safe native value: numpy scalars and arrays to Python, NaN to None."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and math.isnan(value):
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


#: A bridge's connector key: ``(target, execution_mode, owner, authorization)``.
_ConnectorKey = tuple[str, str | None, str | None, str | None]


class _PoolBridge:
    """One event-loop thread and one ``ConnectorHostPool`` for this process."""

    def __init__(self) -> None:
        from osprey_connectors.config import default_config_path, get_config_value
        from osprey_connectors.ipc import ConnectorHostPool

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
        # The engine re-writes the same setpoints every step and restores them
        # after a run, so a write that timed out must not land after a newer one.
        self._pool = ConnectorHostPool(
            pool_section(section), config_file=default_config_path(), kill_on_write_timeout=True
        )
        self._connectors: dict[_ConnectorKey, Any] = {}

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

    def connector(
        self,
        target: str,
        execution_mode: str | None,
        *,
        owner: str | None = None,
        authorization: str | None = None,
    ) -> Any:
        """The pool's connector for this key, its child brought up on first use."""
        key = (target, execution_mode, owner, authorization)
        if key not in self._connectors:
            self._connectors[key] = self.run(
                self._pool.connector(
                    target, execution_mode=execution_mode, owner=owner, authorization=authorization
                )
            )
        return self._connectors[key]

    def retire(
        self,
        target: str,
        execution_mode: str | None,
        *,
        owner: str | None = None,
        authorization: str | None = None,
    ) -> None:
        """Stop this key's child, if one was brought up, and forget its connector.

        For an authorization that has ended: no process keeps a child that
        names it once the run or the route that held it is done.
        """
        connector = self._connectors.pop((target, execution_mode, owner, authorization), None)
        if connector is not None:
            self.run(connector.disconnect())

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


def existing_bridge() -> _PoolBridge | None:
    """This process's pool bridge if one was built, without building one."""
    with _BRIDGE_LOCK:
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
        # Brought up now, so a target the pool cannot serve fails at construction.
        self._reader()

    def _reader(self) -> Any:
        """The connector reads go to: the current owner's, naming no authorization."""
        authority = current_write_authority()
        owner = authority.owner if authority is not None else None
        return self._bridge.connector(self.target, self.execution_mode, owner=owner)

    def _writer(self) -> tuple[Any, WriteAuthority | None]:
        """The connector this write goes to, and the authority it goes out under.

        Raises:
            AuthorizationEndedError: The authority has ended.
            WriteRefusedError: The authority requires an authorization and
                names none (``no_authorization``).
        """
        authority = current_write_authority()
        if authority is None:
            return self._bridge.connector(self.target, self.execution_mode), None
        if self.execution_mode is not None:
            # A readonly call writes nothing whatever it holds: the child refuses it.
            return self._bridge.connector(
                self.target, self.execution_mode, owner=authority.owner
            ), authority
        if authority.ended is not None:
            raise AuthorizationEndedError(authority.ended)
        if authority.required and authority.authorization is None:
            raise WriteRefusedError(
                f"Write to {self.target} refused [{NO_AUTHORIZATION}]: this process writes "
                "only under a write authorization, and holds none.",
                reason=NO_AUTHORIZATION,
            )
        connector = self._bridge.connector(
            self.target,
            self.execution_mode,
            owner=authority.owner,
            authorization=authority.authorization,
        )
        return connector, authority

    def read_one(self, channel: str) -> PVReading:
        from osprey_connectors.errors import ChannelError

        try:
            cv = self._bridge.run(self._reader().read_channel(channel))
        except TimeoutError as exc:
            raise OptimizerTimeoutError(f"Read timed out for {channel}") from exc
        except (ConnectionError, ValueError, ChannelError) as exc:
            # ChannelError: this one channel is unreachable or not a channel; it
            # has no builtin base, so the ConnectionError arm alone misses it.
            raise OptimizerConnectionError(f"Read failed for {channel}: {exc}") from exc
        if cv.value is None:
            raise OptimizerTimeoutError(f"Read returned no value for {channel}")
        return _reading(cv)

    def read_many(self, channel_names: list[str]) -> dict[str, PVReading]:
        """One reading per requested name; a channel that failed is ``connected=False``."""
        if not channel_names:
            return {}
        try:
            results = self._bridge.run(self._reader().read_multiple_channels(channel_names))
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
        """Write every channel in one call through the reference monitor; raise on the first failure.

        One ``write_multiple_channels`` call, so the set is one write call
        and, under a write authorization, one step: judged whole before any
        put, so an unlisted channel or a value out of range writes nothing.
        Each result is then read in order and the first that is not
        confirmed (or unrequested) raises.
        """
        from osprey_connectors.control_system import raise_for_write_result
        from osprey_connectors.errors import (
            ChannelError,
            ChannelLimitsViolationError,
            ChannelWriteBlockedError,
            ChannelWriteFailedError,
        )

        if not channel_inputs:
            return {}
        connector, authority = self._writer()
        operations = list(channel_inputs.items())
        names = ", ".join(channel for channel, _ in operations)
        try:
            results = self._bridge.run(connector.write_multiple_channels(operations))
            for result in results:
                raise_for_write_result(result)
        except ChannelLimitsViolationError as exc:
            # The base batch loop lets a limits refusal out of the channel it
            # concerns; it is a refusal like any other.
            raise WriteRefusedError(
                f"Write to {exc.channel_address} refused [LIMITS]: {exc}", reason="LIMITS"
            ) from exc
        except ChannelWriteBlockedError as exc:
            verdict = getattr(exc, "verdict_reason", None)
            if isinstance(verdict, str) and verdict.startswith("auth_"):
                if authority is not None:
                    authority.end(verdict, str(exc))
                raise AuthorizationEndedError(verdict, str(exc)) from exc
            raise WriteRefusedError(
                f"Write to {exc.channel_address} refused [{exc.reason}]: {exc}",
                reason=exc.reason,
                verdict_reason=verdict,
            ) from exc
        except ChannelWriteFailedError as exc:
            if exc.reason == "FAILED" and not self.check_connection(exc.channel_address):
                raise OptimizerConnectionError(
                    f"Write connect failed for {exc.channel_address}: {exc}"
                ) from exc
            raise EvaluationError(
                f"Write to {exc.channel_address} failed [{exc.reason}]: {exc}"
            ) from exc
        except TimeoutError as exc:
            # Before ConnectionError: a ConnectorHostWriteTimeoutError is both,
            # and a write that may or may not have landed is a timeout.
            raise OptimizerTimeoutError(f"Write timed out for {names}") from exc
        except (ConnectionError, ChannelError) as exc:
            # ChannelError last: the refusal and failure arms above are its
            # subclasses; what is left is a channel being unreachable.
            raise OptimizerConnectionError(f"Write connect failed for {names}: {exc}") from exc
        return dict(channel_inputs)

    def check_connection(self, pv_name: str | None = None) -> bool | None:
        if not pv_name:
            return None
        try:
            return bool(self._bridge.run(self._reader().validate_channel(pv_name)))
        except (ConnectionError, TimeoutError):
            return False
