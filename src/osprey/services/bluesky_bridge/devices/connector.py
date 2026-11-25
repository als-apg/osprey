"""ophyd-async device factory mediated entirely by the OSPREY connector.

Design reversal (R8): the sibling ``epics.py`` factory was built on an
explicit design ruling that ophyd-async already speaks Channel Access
directly, so no OSPREY connector was needed for the plan device layer.
Phase 4's complete-mediation mandate OVERRIDES that ruling: direct CA from
``epics.py`` is exactly the unmediated second read/write path that
mediation is closing. This module is the replacement device layer — every
plan read and every plan write, for every device built here, goes through
the OSPREY connector
(:class:`osprey_connectors.control_system.ControlSystemConnector`):
reads via ``connector.read_channel``, writes via
``connector.write_channel_checked``, which raises on any refused, failed,
mismatched or unconfirmed write so a bad write aborts the RunEngine rather
than silently continuing a plan. There is no raw Channel Access client library,
no low-level EPICS signal backend, and no direct PV access anywhere in
this module.

Device-level delegation over the stable ophyd-async public API (design
decision D1): ``ConnectorSettable``/``ConnectorReadable``/
``ConnectorTriggerable`` are plain ``StandardReadable`` subclasses that call
the connector from ``set()``/``trigger()``/``read()``/``describe()``. A
triggerable fires its command through ``osprey_connectors.command.fire``,
which writes it once through ``connector.write_channel_checked`` and waits for
its status to report the act ended; a ``ConnectorPositioner`` fires the
command that starts its move the same way, after writing its target. This is deliberately NOT a custom
``SignalBackend`` — the ophyd-async ``Signal``/backend protocol is an
internal extension point, while ``StandardReadable``'s ``set``/``read``/
``describe``/``connect`` are the stable public device contract that plans
and the RunEngine actually consume.

Imports ophyd-async (a core dependency), so this module (like the rest of
``devices/``) is kept out of the bridge lifecycle core's import path
(``app.py``, ``runs.py``, ``security.py``), which stays
import-clean of ophyd.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence
from typing import Any

from bluesky.protocols import DataKey, Hints, Location, Reading
from ophyd_async.core import AsyncStatus, StandardReadable

from osprey.services.bluesky_bridge.plan_fields import SETPOINT_KEY_SUFFIX
from osprey.services.bluesky_bridge.plan_writes import ChannelWrite
from osprey_connectors.command import (
    CommandAct,
    CommandBusyError,
    CommandNoEffectError,
    CommandStatusError,
    fire,
)
from osprey_connectors.control_system import values_match
from osprey_connectors.errors import ChannelError
from osprey_connectors.settle import Landing, SettleTimeoutError, settle, settle_bound

from ._connect import connect_all
from .specs import ReadableSpec, SettableSpec, TriggerableSpec

#: The value written to a command to fire it.
PRESS_VALUE = 1

#: Env var carrying the floor on how close a readback must come to the demand
#: before ``ConnectorSettable.set()`` calls a move settled. Authored per
#: facility in the build profile and rendered into the compose file; the
#: container cannot read `config.yml`, so the env var is the whole channel. The
#: floor is never a ceiling: a device's declared tolerance applies on every
#: lane when it is wider, and its readback's motion band does on a simulated
#: lane.
SETTLE_TOLERANCE_ENV = "BLUESKY_SETTLE_TOLERANCE"

#: Used when the variable is unset. A float-noise bound: the floor for a
#: setpoint/readback pair kept in exact software sync. A device that settles
#: wider carries its tolerance, or its readback's motion band, in the device
#: file.
DEFAULT_SETTLE_TOLERANCE = 1e-9

#: Env var bounding how long ``ConnectorSettable.set()`` polls the readback
#: channel for settlement after the write verifies.
SETTLE_TIMEOUT_ENV = "BLUESKY_SETTLE_TIMEOUT_S"

#: Used when the variable is unset.
DEFAULT_SETTLE_TIMEOUT_S = 5.0


def settle_tolerance() -> float:
    """Max ``abs(readback - demand)`` for a move to count as settled.

    How far a readback may sit from its demand before a plan calls the move
    done is a property of the DEVICES a facility drives, not of OSPREY: an
    aliased software setpoint settles to float noise, a magnet or a gap does
    not. Read from `os.environ` on every call — never cached at import — so a
    test (and a re-exec'd process) sees the value its environment declares.

    Raises:
        ValueError: if the variable is set to something that is not a number,
            or to a value below zero. A negative tolerance can never be
            satisfied, so it would turn every move into a timeout.
    """
    return _positive_float(SETTLE_TOLERANCE_ENV, DEFAULT_SETTLE_TOLERANCE, allow_zero=True)


def settle_timeout_s() -> float:
    """How long ``set()`` polls the readback before raising ``TimeoutError``.

    Fail-closed: running out of budget raises out of the ``AsyncStatus`` and
    aborts the RunEngine. Raising this key gives a slow device more room; it
    never turns an unsettled move into a successful one.

    Raises:
        ValueError: if the variable is set to something that is not a number,
            or to a value at or below zero — a zero budget would abort every
            move that needs even one poll.
    """
    return _positive_float(SETTLE_TIMEOUT_ENV, DEFAULT_SETTLE_TIMEOUT_S, allow_zero=False)


def _positive_float(env_var: str, default: float, *, allow_zero: bool) -> float:
    """Read *env_var* as a float, falling back to *default* when unset."""
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"{env_var}={raw!r} is not a number; set it to a decimal value (default {default})"
        ) from None
    if value < 0 or (value == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{env_var}={raw!r} must be {bound}")
    return value


def _data_key(address: str, unit: str | None, shape: tuple[int] | None = None) -> DataKey:
    """The data key for ``address``: a scalar number, or with ``shape`` an array of
    that length, carrying ``units`` when ``unit`` is known."""
    key: DataKey = {"source": f"connector:{address}", "dtype": "number", "shape": []}
    if shape is not None:
        key["dtype"] = "array"
        key["shape"] = list(shape)
    if unit is not None:
        key["units"] = unit
    return key


class ConnectorSettable(StandardReadable):
    """A settable/readable PV pair mediated entirely by the OSPREY connector.

    Declares no ophyd-async EPICS signals at all — this package ships no
    direct Channel Access device class, deliberately, so that no read or
    write can bypass the connector's reference monitor. ``set()`` writes the
    setpoint through ``connector.write_channel_checked`` — which raises on
    any refusal, failure, mismatch, or unconfirmed write, aborting the
    RunEngine — then waits in the shared settle step
    (:func:`osprey_connectors.settle.settle`) for the (possibly separate)
    readback to come within :func:`osprey_connectors.settle.settle_bound` of
    the demand, with :func:`settle_tolerance` as the floor and
    :func:`settle_timeout_s` as the budget.
    ``read()``/``describe()`` are overridden
    to return the *live* readback via the connector on every call — never a
    cached/soft value — so a plan's ``trigger_and_read`` document always
    reflects the current mediated state.

    When ``readback_pv`` is omitted, ``readback`` aliases ``setpoint_pv``:
    there is no independent readback to settle against, and a ``confirmed``
    write already means the connector re-read that exact channel and found
    the value sent, so the settle poll is skipped entirely. The poll loop
    runs when the write was ``unrequested`` (the channel is configured
    ``confirm: false``, so nothing was checked). With a separate
    ``readback_pv`` the write's outcome is about the *setpoint*, not the
    readback channel, so it never stands in for a settle sample there.

    A device with a separate ``readback_pv`` reports BOTH channels: the
    readback under ``<name>`` (the hinted field, what ``bps.rd`` returns) and
    the demand under ``<name>_setpoint`` (:data:`SETPOINT_KEY_SUFFIX`), each
    read live through the connector. Where the device is and where it was
    told to go are two different facts about a device that takes time to
    move, and a plan that decides when to advance on the difference between
    them needs both off the one device it drives -- the alternative is a
    second device aliasing the setpoint channel, which puts the same address
    in the worker's namespace twice. An aliased device reports one key: its
    readback IS its demand.

    The device is Bluesky-``Locatable``, so ``bps.loc`` answers where it
    was told to go and where it is, both read live through the connector.

    :meth:`set_writes` states the one write ``set()`` makes: the demand to the
    setpoint channel, never to the readback.

    ``unit`` is the unit of the value reported under ``<name>`` and
    ``setpoint_unit`` the unit of the demand under ``<name>_setpoint``; each is
    stated as the data key's ``units`` in :meth:`describe`, and a key whose
    unit is ``None`` carries no ``units``.

    The OSPREY connector instance is stored as ``self._osprey_connector``,
    not ``self._connector``: ophyd-async's own ``Device.__init__`` already
    owns the ``self._connector`` attribute (its internal
    ``DeviceConnector``, used by ``Device.connect()``); reusing that name
    for the OSPREY connector would silently clobber ophyd-async's connect
    machinery.
    """

    def __init__(
        self,
        connector: Any,
        setpoint_pv: str,
        readback_pv: str | None = None,
        name: str = "",
        settle_tolerance: float | None = None,
        settle_relative: float | None = None,
        motion_band: float | None = None,
        simulated: bool = False,
        unit: str | None = None,
        setpoint_unit: str | None = None,
    ) -> None:
        self._osprey_connector = connector
        self._setpoint_pv = setpoint_pv
        self._readback_pv = readback_pv or setpoint_pv
        self._settle_tolerance = settle_tolerance
        self._settle_relative = settle_relative
        self._motion_band = motion_band
        self._simulated = simulated
        self._unit = unit
        self._setpoint_unit = setpoint_unit
        super().__init__(name=name)

    def set_writes(self, value: float) -> tuple[ChannelWrite, ...]:
        """The channel writes ``set(value)`` makes: ``value`` to the setpoint. No I/O."""
        return (ChannelWrite(self._setpoint_pv, value),)

    @AsyncStatus.wrap
    async def set(self, value: float) -> None:
        """Write ``value`` through the connector, then wait for its readback to settle.

        Raises:
            ChannelWriteBlockedError: The write was refused and no value was
                written — either by the reference monitor (writes disabled,
                limits, validation), in which case it was never attempted, or
                by the control system itself (CONTROL_SYSTEM_REFUSED).
            ChannelWriteFailedError: The write was sent but the channel does
                not verifiably hold the value sent — ``FAILED`` (the control
                system did not take it), ``MISMATCH`` (the channel holds a
                different value, e.g. a clamped setpoint) or ``UNCONFIRMED``
                (sent, but not acknowledged in time or the confirming re-read
                itself failed). All three abort the
                plan: an unconfirmed or mismatched write is a false premise
                for every step that follows it.
            ChannelUnreachableError: Propagated unchanged from the
                connector's write, for a channel that could not be reached.
            ConnectorUnavailableError: Propagated unchanged: the connector
                itself was lost, on the write or on the readback.
            TimeoutError: Either propagated unchanged from the connector's
                write, or a ``SettleTimeoutError`` when ``readback`` does not
                settle within its bound of ``value`` within
                :func:`settle_timeout_s` seconds.
            SettleReadError: A ``ChannelReadFailedError``: the readback cannot
                be reached, is not a channel, or times out while settling, or
                it answered no value; names the readback. A readback that
                answers no value aborts the plan at once rather than after the
                settle timeout.

            Every one of these propagates uncaught through the
            ``AsyncStatus`` this method is wrapped in, aborting the
            RunEngine — this is the entire safety point of routing writes
            through ``write_channel_checked`` instead of a bare
            ``write_channel``.
        """
        # No ``confirm`` kwarg: whether a channel is confirmed is the channel's
        # own resolved policy, and a device layer has no opinion to add.
        result = await self._osprey_connector.write_channel_checked(self._setpoint_pv, value)

        if self._readback_pv == self._setpoint_pv and result.outcome == "confirmed":
            # ``confirmed`` means the connector re-read this very channel and
            # found the value sent — that IS the settle, so return on the word,
            # never on arithmetic over ``observed_value``: the settle tolerance
            # is stricter, by default, than the connector's comparison rule
            # (``isclose(rel_tol=1e-6)``), so re-checking a confirmed write here
            # would poll the setpoint just written for the full settle timeout
            # and then raise on a write that succeeded. (``WriteOutcome`` is a
            # ``StrEnum``, so the word compares equal without importing the
            # connector package into this deliberately duck-typed device layer.)
            return

        # Both budgets are read HERE, once per call, rather than bound as
        # defaults: the facility authors them in its profile and they reach the
        # container as env vars, so the value in force is whatever the process
        # environment says at the moment the move starts.
        await settle(self._osprey_connector, [self._landing(value)], timeout_s=settle_timeout_s())

    def _landing(self, value: float) -> Landing:
        """The readback's landing for a move to ``value``, its bound read from the environment now."""
        bound = settle_bound(
            self._tolerance_record(),
            value,
            motion=self._motion_band or 0.0,
            simulated=self._simulated,
            floor=settle_tolerance(),
        )
        # The floor is always a number, so the bound always is.
        assert bound is not None
        return Landing(self._setpoint_pv, self._readback_pv, value, bound)

    def _tolerance_record(self) -> dict[str, float] | None:
        """The device's declared tolerance as the facility states one, or ``None``."""
        if self._settle_tolerance is not None:
            return {"absolute": self._settle_tolerance}
        if self._settle_relative is not None:
            return {"relative": self._settle_relative}
        return None

    def channel_addresses(self) -> tuple[str, ...]:
        """The addresses a run of this device touches: the setpoint and the readback."""
        return (self._setpoint_pv, self._readback_pv)

    @property
    def has_distinct_readback(self) -> bool:
        """Whether this device reads back a channel other than the one it writes."""
        return self._readback_pv != self._setpoint_pv

    @property
    def setpoint_key(self) -> str:
        """The data key the demand is reported under, ``<name>_setpoint``.

        Only present in ``read()``/``describe()`` when
        :attr:`has_distinct_readback` -- an aliased device's demand IS its
        readback, and the key would restate ``<name>``.
        """
        return f"{self.name}{SETPOINT_KEY_SUFFIX}"

    async def locate(self) -> Location[Any]:
        """Return where the device was told to go and where it is, read live.

        A device with a distinct readback reads its readback channel, then its
        setpoint channel, in :meth:`read`'s order. An aliased device makes one
        read and reports that value for both: its readback IS its demand.
        Values are returned as the connector gives them.
        """
        reading = await self._osprey_connector.read_channel(self._readback_pv)
        if not self.has_distinct_readback:
            return Location(setpoint=reading.value, readback=reading.value)
        demand = await self._osprey_connector.read_channel(self._setpoint_pv)
        return Location(setpoint=demand.value, readback=reading.value)

    async def read(self) -> dict[str, Reading[Any]]:
        """Return the live readback value, read fresh through the connector.

        Never returns a cached/soft value: every call issues a new
        ``connector.read_channel`` so the document a plan records reflects
        the current mediated state of the readback channel. A device with a
        distinct readback also reads its setpoint channel and reports the
        demand under :attr:`setpoint_key`.
        """
        reading = await self._osprey_connector.read_channel(self._readback_pv)
        document: dict[str, Reading[Any]] = {
            self.name: {"value": reading.value, "timestamp": time.time()}
        }
        if self.has_distinct_readback:
            demand = await self._osprey_connector.read_channel(self._setpoint_pv)
            document[self.setpoint_key] = {"value": demand.value, "timestamp": time.time()}
        return document

    async def describe(self) -> dict[str, DataKey]:
        """Describe the live readback channel as a scalar numeric data key,
        and the setpoint channel beside it when the two differ, each with the
        ``units`` the device was given. Reads nothing through the connector."""
        described: dict[str, DataKey] = {
            self.name: _data_key(self._readback_pv, self._unit),
        }
        if self.has_distinct_readback:
            described[self.setpoint_key] = _data_key(self._setpoint_pv, self._setpoint_unit)
        return described

    @property
    def hints(self) -> Hints:
        """Declare this movable's readback field as the one hinted field.

        ``StandardReadable`` builds its hints by aggregating over the
        ophyd-async signals declared on the device; this class declares
        none (every read goes through the connector instead), so the
        inherited property would report no fields at all and consumers of
        the run — live table, plot axes, ``PeakStats`` — would have nothing
        to key on. The readback key, named for the device, is the field
        named here; the ``<name>_setpoint`` key a distinct-readback device
        also emits is deliberately NOT hinted, so ``bps.rd`` keeps returning
        where the device is and a live table plots the machine, not the
        demand.

        Overriding as a property is required, not stylistic: the base class
        declares ``hints`` read-only, so assigning an instance attribute in
        ``__init__`` raises ``AttributeError``.
        """
        return {"fields": [self.name]}


class ConnectorPositioner(ConnectorSettable):
    """A setpoint whose move starts only when a command is written: a positioner.

    ``set(value)`` makes its writes in the order :meth:`set_writes` states them:

    1. it reads the start command's status once and refuses with
       :class:`~osprey_connectors.command.CommandBusyError`, writing nothing,
       when the status is not idle, so nothing is written while a previous act
       runs;
    2. it writes ``value`` to the setpoint through ``write_channel_checked``,
       as :class:`ConnectorSettable` does;
    3. it fires the start command with :data:`PRESS_VALUE` through
       ``osprey_connectors.command.fire``, which waits for the status to leave
       idle and return within the act's bound. A failure of ``fire`` fails the
       move with a note that the setpoint now holds the target;
    4. when the status never leaves idle, the readback is read once: within
       its settle bound of the target the move is done, the stage was already
       there; elsewhere, or with no separate readback, the move fails, the
       start had no effect;
    5. after a completed act, a separate readback settles as a settable's
       does, its budget counted from the act's end; with none, the act's end
       is the move's confirmation and nothing is polled.

    The RunEngine calls :meth:`stop` on a pause, a suspend, an abort, a
    failure and at the run's end. A move still unfinished then is cancelled:
    nothing more is written to the device, and a stage already moving moves
    on by itself. A finished move's ``stop()`` does nothing, and ``stop()``
    writes nothing: :meth:`stop_writes` states no write.

    The device is named, read and recorded as the settable it is: the start
    command and its status are probed before a run (:meth:`channel_addresses`)
    and never recorded.
    """

    def __init__(
        self,
        connector: Any,
        setpoint_pv: str,
        readback_pv: str | None = None,
        *,
        started_by: CommandAct,
        **settable: Any,
    ) -> None:
        super().__init__(connector, setpoint_pv, readback_pv, **settable)
        self._started_by = started_by
        self._move: asyncio.Task[Any] | None = None

    def set_writes(self, value: float) -> tuple[ChannelWrite, ...]:
        """The channel writes ``set(value)`` makes, in order: ``value`` to the
        setpoint, then :data:`PRESS_VALUE` to the start command. No I/O."""
        return (
            ChannelWrite(self._setpoint_pv, value),
            ChannelWrite(self._started_by.command, PRESS_VALUE),
        )

    def stop_writes(self) -> tuple[ChannelWrite, ...]:
        """The channel writes ``stop()`` makes: none. No I/O."""
        return ()

    def channel_addresses(self) -> tuple[str, ...]:
        """The addresses a run of this device touches: the setpoint, the
        readback, the start command and its status."""
        return (*super().channel_addresses(), self._started_by.command, self._started_by.status)

    @AsyncStatus.wrap
    async def set(self, value: float) -> None:
        """Write ``value``, fire the start command and wait for the move to end.

        Raises:
            CommandBusyError: The status was not idle; nothing was written.
            CommandNoEffectError: The status never left idle and the readback
                is not at the target, or there is no separate readback.
            CommandTimeoutError: The status left idle and did not return
                within the act's bound.
            CommandStatusError: The status cannot be read or watched.
            SettleTimeoutError: The readback did not settle after the act.

            Every error :meth:`ConnectorSettable.set` raises on the setpoint
            write or the settle, unchanged.
        """
        self._move = asyncio.current_task()
        target, start = self.set_writes(value)
        await self._refuse_while_busy()
        # No ``confirm`` kwarg, as on a settable: the channel's own policy.
        await self._osprey_connector.write_channel_checked(target.address, target.value)
        landing = self._landing(value)
        try:
            await fire(self._osprey_connector, self._started_by, start.value)
        except CommandNoEffectError as no_effect:
            await self._already_there(no_effect, landing)
        except Exception as failed:
            failed.add_note(f"{target.address} holds the target {value}")
            raise
        else:
            if self.has_distinct_readback:
                await settle(self._osprey_connector, [landing], timeout_s=settle_timeout_s())
        self._move = None

    async def _refuse_while_busy(self) -> None:
        """Read the status once; refuse when it is not the act's idle value.

        Compared as ``fire`` compares, with ``values_match`` and the reading's
        label.

        Raises:
            CommandBusyError: The status is not idle.
            CommandStatusError: The read failed, answered no value, or ``idle``
                is a label and the reading carries none.
        """
        act = self._started_by
        try:
            reading = await self._osprey_connector.read_channel(act.status)
        except ChannelError as exc:
            raise CommandStatusError(act, exc) from exc
        value = getattr(reading, "value", None)
        if value is None:
            raise CommandStatusError(act, reason="no value")
        label = getattr(getattr(reading, "metadata", None), "enum_label", None)
        if isinstance(act.idle, str) and label is None:
            raise CommandStatusError(
                act,
                reason=f"idle is the label {act.idle!r} and the reading {value} carries no label",
            )
        if not values_match(act.idle, value, enum_label=label):
            raise CommandBusyError(act, value)

    async def _already_there(self, no_effect: CommandNoEffectError, landing: Landing) -> None:
        """Return when the readback already sits within its bound of the target.

        Raises:
            CommandNoEffectError: ``no_effect``, noted with the readback and the
                target, when the readback is elsewhere or there is no separate
                readback.
            SettleReadError: The readback cannot be read, noted with the
                target the setpoint holds.
        """
        if not self.has_distinct_readback:
            no_effect.add_note(
                f"start had no effect: {self._setpoint_pv} has no separate readback, "
                f"target {landing.demand}"
            )
            raise no_effect
        try:
            await settle(self._osprey_connector, [landing], timeout_s=0)
        except SettleTimeoutError as elsewhere:
            no_effect.add_note(
                f"start had no effect: {landing.readback} reads {elsewhere.last}, "
                f"target {landing.demand}"
            )
            raise no_effect from None
        except Exception as failed:
            failed.add_note(f"{self._setpoint_pv} holds the target {landing.demand}")
            raise

    async def stop(self, *, success: bool = True) -> None:  # noqa: ARG002 - the bluesky verb signature
        """Cancel an unfinished move and wait for it to end; write nothing."""
        move = self._move
        if move is not None and not move.done():
            move.cancel()
            await asyncio.wait([move])


class ConnectorReadable(StandardReadable):
    """A single read-only channel mediated entirely by the OSPREY connector.

    Trigger-less (no ``trigger()`` method), and every ``read()`` performs a
    fresh ``connector.read_channel``
    call rather than returning a cached/soft value — a soft signal would
    return a stale value, defeating the point of live mediation.

    The OSPREY connector instance is stored as ``self._osprey_connector``,
    for the same reason as :class:`ConnectorSettable`: ``self._connector``
    is already owned by ophyd-async's ``Device.__init__``.
    """

    def __init__(
        self,
        connector: Any,
        read_pv: str,
        name: str = "",
        unit: str | None = None,
        shape: tuple[int] | None = None,
    ) -> None:
        self._osprey_connector = connector
        self._read_pv = read_pv
        self._unit = unit
        self._shape = shape
        super().__init__(name=name)

    def channel_addresses(self) -> tuple[str, ...]:
        """The address a run of this device touches: its one channel."""
        return (self._read_pv,)

    async def read(self) -> dict[str, Reading[Any]]:
        """Return the live value, read fresh through the connector."""
        reading = await self._osprey_connector.read_channel(self._read_pv)
        return {self.name: {"value": reading.value, "timestamp": time.time()}}

    async def describe(self) -> dict[str, DataKey]:
        """Describe the live channel as a scalar number, or an array of the length
        the facility states, with its ``units`` when the device was given one."""
        return {self.name: _data_key(self._read_pv, self._unit, self._shape)}

    @property
    def hints(self) -> Hints:
        """Declare this readable's single field as the hinted one.

        Same reasoning as :attr:`ConnectorSettable.hints`: no ophyd-async
        signals are declared here for ``StandardReadable`` to aggregate, so
        the inherited property would report no fields, and the one data key
        ``read()``/``describe()`` emit is named for the device. Must be a
        property override — the base class declares ``hints`` read-only.
        """
        return {"fields": [self.name]}


class _Firing:
    """One command fired through a connector, and the trigger still running, if any."""

    def __init__(self, connector: Any, act: CommandAct) -> None:
        self.connector = connector
        self.act = act
        self.pending: asyncio.Task[Any] | None = None

    @property
    def running(self) -> bool:
        """Whether a trigger has started and not finished."""
        return self.pending is not None and not self.pending.done()


async def _refuse(error: Exception) -> None:
    raise error


class ConnectorTriggerable(StandardReadable):
    """A command a plan triggers, fired and waited for through the OSPREY connector.

    ``trigger()`` writes :data:`PRESS_VALUE` to the command once, through
    ``osprey_connectors.command.fire``: the status is read and must be idle,
    the command is written through ``write_channel_checked`` unconfirmed (the
    wait confirms it), and the trigger completes when the status has left idle
    and come back within the act's bound. Every failure of ``fire`` fails the
    trigger's status. While one trigger is unfinished, another is refused and
    writes nothing.

    No ``set()`` and no ``stop()``: a plan triggers this device and never sets
    it. :meth:`cancel_trigger` cancels an unfinished trigger; a trigger
    cancelled before its write writes nothing, and a command already written
    runs its act to the end in the hardware while the run stops waiting for
    it. The RunEngine calls :meth:`pause` on every pause and suspend, and it
    cancels; :meth:`resume` does nothing, so a run resumed after a pause fails
    at the trigger the pause cut.

    ``read()``/``describe()`` report the status, read fresh through the
    connector, under the device's name: once per trigger, the run records what
    the act left the status at.

    :meth:`trigger_writes` states the one write ``trigger()`` makes.

    The OSPREY connector instance is stored as ``self._osprey_connector`` for
    the same reason as :class:`ConnectorSettable`.
    """

    def __init__(
        self,
        connector: Any,
        command: str,
        status: str,
        idle: str | int | float | bool,
        timeout_s: float,
        name: str = "",
    ) -> None:
        self._osprey_connector = connector
        self._firing = _Firing(connector, CommandAct(command, status, idle, timeout_s))
        super().__init__(name=name)

    def trigger_writes(self) -> tuple[ChannelWrite, ...]:
        """The channel writes ``trigger()`` makes: the press value to the command. No I/O."""
        return (ChannelWrite(self._firing.act.command, PRESS_VALUE),)

    def trigger(self) -> AsyncStatus:
        """Fire the command and wait for its act to end.

        Returns a failed status, having written nothing, while the previous
        trigger is unfinished.
        """
        if self._firing.running:
            return AsyncStatus(
                _refuse(RuntimeError(f"{self.name}: the previous trigger is still running"))
            )
        status = AsyncStatus(fire(self._firing.connector, self._firing.act, PRESS_VALUE))
        self._firing.pending = status.task
        return status

    def cancel_trigger(self) -> None:
        """Cancel an unfinished trigger; a finished one, or none, is left alone."""
        pending = self._firing.pending
        if pending is not None and not pending.done():
            pending.cancel()

    def pause(self) -> None:
        """Cancel an unfinished trigger: nothing is written after a pause."""
        self.cancel_trigger()

    def resume(self) -> None:
        """Do nothing: a trigger cut by a pause is not fired again."""

    def channel_addresses(self) -> tuple[str, ...]:
        """The addresses a run of this device touches: the command and the status."""
        return (self._firing.act.command, self._firing.act.status)

    async def read(self) -> dict[str, Reading[Any]]:
        """Return the status, read fresh through the connector, under the device's name."""
        reading = await self._osprey_connector.read_channel(self._firing.act.status)
        return {self.name: {"value": reading.value, "timestamp": time.time()}}

    async def describe(self) -> dict[str, DataKey]:
        """Describe the status under the device's name. Reads nothing."""
        return {self.name: _data_key(self._firing.act.status, None)}

    @property
    def hints(self) -> Hints:
        """Declare the device's one field as the hinted one; see
        :attr:`ConnectorReadable.hints`."""
        return {"fields": [self.name]}


async def build_devices(
    settables: Sequence[SettableSpec] = (),
    readables: Sequence[ReadableSpec] = (),
    connector: Any = None,
    *,
    simulated: bool = False,
    triggerables: Sequence[TriggerableSpec] = (),
) -> dict[str, Any]:
    """Build and connect connector-mediated devices, keyed by name.

    Matches the ``get_devices() -> dict[str, Any]`` shape ``plans.py``'s
    built-in plans (and any facility-injected plan, per ``plan_loader.py``)
    resolve device names against — the same factory contract
    ``mock.build_devices`` provides. Connection (and
    why it's an explicit ``connect()`` rather than ``init_devices()``) is
    handled by :func:`._connect.connect_all`; a ``ConnectorSettable``/
    ``ConnectorReadable`` declares no ophyd-async signals, so this connects
    as a no-op per device, same as the other factories in this package.

    Args:
        settables: Specs for the ``ConnectorSettable`` instances to build; a
            spec that states ``started_by`` builds a ``ConnectorPositioner``.
        readables: Specs for the ``ConnectorReadable`` instances to build.
        connector: The OSPREY control-system connector every built device
            delegates its reads/writes to. Every device shares this same
            connector instance.
        simulated: Whether the connector serves the simulator, so a device
            without a declared tolerance settles within its readback's
            motion band. False, the fail-safe toward hardware, leaves such a
            device on the profile's floor.
        triggerables: Specs for the ``ConnectorTriggerable`` instances to build.

    Returns:
        Mapping of device name to connected device instance.

    Raises:
        ValueError: If ``connector`` is None — failing here, at the
            misconfiguration site, instead of as an ``AttributeError`` deep
            inside a device's ``set()``/``read()`` while a plan runs.
    """
    if connector is None:
        raise ValueError(
            "build_devices requires a connector — every built device delegates "
            "its reads and writes to it"
        )
    devices: dict[str, Any] = {}
    for settable_spec in settables:
        settable: dict[str, Any] = {
            "name": settable_spec.name,
            "settle_tolerance": settable_spec.settle_tolerance,
            "settle_relative": settable_spec.settle_relative,
            "motion_band": settable_spec.motion_band,
            "simulated": simulated,
            "unit": settable_spec.unit,
            "setpoint_unit": settable_spec.setpoint_unit,
        }
        if settable_spec.started_by is not None:
            devices[settable_spec.name] = ConnectorPositioner(
                connector,
                settable_spec.setpoint_pv,
                settable_spec.readback_pv,
                started_by=settable_spec.started_by,
                **settable,
            )
        else:
            devices[settable_spec.name] = ConnectorSettable(
                connector, settable_spec.setpoint_pv, settable_spec.readback_pv, **settable
            )
    for readable_spec in readables:
        devices[readable_spec.name] = ConnectorReadable(
            connector,
            readable_spec.read_pv,
            name=readable_spec.name,
            unit=readable_spec.unit,
            shape=readable_spec.shape,
        )
    for triggerable_spec in triggerables:
        devices[triggerable_spec.name] = ConnectorTriggerable(
            connector,
            triggerable_spec.command,
            triggerable_spec.status,
            triggerable_spec.idle,
            triggerable_spec.timeout_s,
            name=triggerable_spec.name,
        )
    return await connect_all(devices)
