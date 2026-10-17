"""The facility's sequence module: what fires the injection, and what it may write.

A sequence module is one ``.py`` file, :data:`SEQUENCE_MODULE_FILE`, in the
directory :func:`~.startup.resolve_sequence_dir` resolves. It is loaded once,
at service start, through :func:`~osprey.utils.module_loading.load_by_path`,
and validated before the service serves anything:

* **Entry points** are coroutine functions taking the sequence handle:
  ``inject`` (required) fires one injection; ``bunch_clean`` and ``bpm_mode``
  are optional. Any of the three that is defined but not a coroutine function
  is refused.
* **``SEQUENCE_METADATA``** is a mapping with ``max_pause_s`` (required) and
  optionally ``slots`` and ``writes``:

  - ``slots`` maps a slot name to one address (a string) or to one address per
    BPM, named by the facility signal those channels carry (``{"tag":
    "<signal>"}``). Module slots are settings: a slot naming an arm command or
    the injection request, or tagging the arm signal, is refused, because the
    service alone arms and only ``inject`` requests.
  - ``writes`` maps an entry point to the slot names it may write. Every name
    must be a declared slot and every entry point one the module defines;
    ``inject`` has no entry, since it writes the injection request and nothing
    else.
  - ``max_pause_s`` is the longest the module ever waits between two calls, a
    positive number no larger than :data:`~.constants.MAX_PAUSE_CEILING_S`.

A directory that is unset, missing, or holds no :data:`SEQUENCE_MODULE_FILE`
leaves the service without a module: it starts, and its shots and setup
refuse ``no_sequence_module``. A module that does not load or does not
validate refuses the service's start, naming the file and what is wrong.

An entry point runs with a :class:`SequenceHandle`, built for that one entry
point inside an action that holds the run lock:

* ``write("injection_request", value)`` fires the injection request through
  :func:`~osprey.runtime.guarded_run.command_write`, from ``inject`` only.
* ``write(<slot>, value)`` writes a declared module slot, from an entry point
  whose ``writes`` lists it, through the connector's checked setting path
  with the confirm each channel's limits record asks for. Before each address
  is written its current value is read, and every write that was not refused
  is listed as a :class:`~.state.Moved`, so it can be put back.
* Any other write (an arm command, an address, an undeclared slot, a slot
  this entry point does not declare) is refused before anything is read or
  written; no entry point ever arms.
* ``read``, ``wait_for`` and ``sleep`` read channels, poll one until it
  satisfies a condition, and pause no longer than ``max_pause_s``. A read of
  ``"injection_request"`` reads the request the build names, so a module that
  counts requests up never names its address.

Every connector call runs through :func:`~.calls.guarded_call`, and every
handle call is a stop checkpoint: once the action's stop event is set, the
next checkpoint raises :class:`SequenceStopped`, and a wait or a pause in
progress ends within one poll.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any

from osprey.services.trajectory.calls import guarded_call, read_value
from osprey.services.trajectory.constants import CHECKPOINT_POLL_S, MAX_PAUSE_CEILING_S
from osprey.services.trajectory.refusal import Refused
from osprey.services.trajectory.state import Moved

if TYPE_CHECKING:
    from osprey.services.trajectory.startup import StartupRefused, TrajectoryCommands

__all__ = [
    "ARM_SIGNAL",
    "ENTRY_POINTS",
    "INJECTION_REQUEST",
    "METADATA_NAME",
    "REQUIRED_ENTRY_POINT",
    "SEQUENCE_MODULE_FILE",
    "SEQUENCE_MODULE_PREFIX",
    "SequenceHandle",
    "SequenceMetadata",
    "SequenceModule",
    "SequenceRefused",
    "SequenceSlot",
    "SequenceStopped",
    "load_sequence",
    "validate_metadata",
]

logger = logging.getLogger(__name__)

#: The file in the sequence directory that holds the module.
SEQUENCE_MODULE_FILE = "sequence.py"
#: The ``sys.modules`` prefix a loaded sequence module is registered under.
SEQUENCE_MODULE_PREFIX = "osprey_trajectory_sequence_"
#: The module-level mapping that declares slots, writes and the longest pause.
METADATA_NAME = "SEQUENCE_METADATA"
#: The entry point every module defines.
REQUIRED_ENTRY_POINT = "inject"
#: Every entry point a module may define, the required one first.
ENTRY_POINTS: tuple[str, ...] = (REQUIRED_ENTRY_POINT, "bunch_clean", "bpm_mode")
#: The facility signal of the monitors' arm commands, which no module slot may tag.
ARM_SIGNAL = "bpm_arm_command"
#: The name a module writes the injection request by.
INJECTION_REQUEST = "injection_request"

_METADATA_KEYS = frozenset({"slots", "writes", "max_pause_s"})
_TAG_KEY = "tag"


@dataclass(frozen=True)
class SequenceSlot:
    """One module slot: one address, or one address per BPM by signal tag.

    Exactly one of ``address`` and ``tag`` is set.

    Attributes:
        name: The slot's name in ``SEQUENCE_METADATA["slots"]``.
        address: The one address the slot writes, or ``None``.
        tag: The facility signal whose channel on each BPM the slot writes,
            or ``None``.
    """

    name: str
    address: str | None = None
    tag: str | None = None

    @property
    def per_bpm(self) -> bool:
        """Whether the slot is one address per BPM."""
        return self.tag is not None


@dataclass(frozen=True)
class SequenceMetadata:
    """A module's validated ``SEQUENCE_METADATA``.

    Attributes:
        slots: Slot name to slot, in declaration order.
        writes: Entry point to the slot names it may write; an entry point
            absent here writes no module slot.
        max_pause_s: The longest the module waits between two calls.
    """

    slots: Mapping[str, SequenceSlot]
    writes: Mapping[str, tuple[str, ...]]
    max_pause_s: float


@dataclass(frozen=True)
class SequenceModule:
    """A loaded, validated sequence module.

    Attributes:
        path: The file it was loaded from.
        module: The loaded module.
        metadata: Its validated ``SEQUENCE_METADATA``.
        entry_points: Entry-point name to coroutine function, for the entry
            points the module defines; ``inject`` is always present.
    """

    path: Path
    module: ModuleType
    metadata: SequenceMetadata
    entry_points: Mapping[str, Callable[..., Any]]

    @property
    def inject(self) -> Callable[..., Any]:
        """The required entry point that fires one injection."""
        return self.entry_points[REQUIRED_ENTRY_POINT]

    def entry_point(self, name: str) -> Callable[..., Any] | None:
        """The coroutine function for *name*, or ``None`` when the module has none."""
        return self.entry_points.get(name)


def _refused(path: Path, problem: str) -> StartupRefused:
    from osprey.services.trajectory.startup import StartupRefused

    return StartupRefused(f"sequence module {path}: {problem}")


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _slot(path: Path, name: Any, value: Any, commands: TrajectoryCommands | None) -> SequenceSlot:
    if not _name(name):
        raise _refused(path, f"slot name {name!r} is not a non-empty string")
    if _name(value):
        if commands is not None and value in commands.commands:
            raise _refused(
                path,
                f"slot {name!r} names {value!r}, a command the service alone writes "
                "(an arm command or the injection request)",
            )
        return SequenceSlot(name=name, address=value)
    if isinstance(value, Mapping) and set(value) == {_TAG_KEY} and _name(value[_TAG_KEY]):
        tag = value[_TAG_KEY]
        if tag == ARM_SIGNAL:
            raise _refused(path, f"slot {name!r} tags {ARM_SIGNAL!r}: the service alone arms")
        return SequenceSlot(name=name, tag=tag)
    raise _refused(
        path,
        f"slot {name!r} is {value!r}: a slot is one address (a string) or "
        f'one address per BPM ({{"{_TAG_KEY}": "<signal>"}})',
    )


def _max_pause(path: Path, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise _refused(path, f"max_pause_s is {value!r}, not a number of seconds")
    if value <= 0:
        raise _refused(path, f"max_pause_s is {value!r}; it must be positive")
    if value > MAX_PAUSE_CEILING_S:
        raise _refused(
            path, f"max_pause_s is {value!r}, above the ceiling of {MAX_PAUSE_CEILING_S} s"
        )
    return float(value)


def validate_metadata(
    path: Path,
    metadata: Any,
    *,
    defined: frozenset[str],
    commands: TrajectoryCommands | None,
) -> SequenceMetadata:
    """Validate a module's ``SEQUENCE_METADATA``.

    Args:
        path: The module's file, named in every refusal.
        metadata: The module's ``SEQUENCE_METADATA`` as it found it.
        defined: The entry points the module defines.
        commands: The service's command set, whose addresses no slot may
            name; ``None`` when the build has none.

    Returns:
        The validated metadata.

    Raises:
        StartupRefused: The metadata is malformed or declares a forbidden slot
            or write.
    """
    if not isinstance(metadata, Mapping):
        raise _refused(path, f"{METADATA_NAME} is {type(metadata).__name__}, not a mapping")
    unknown = sorted(str(key) for key in set(metadata) - _METADATA_KEYS)
    if unknown:
        raise _refused(
            path,
            f"{METADATA_NAME} has unknown keys {unknown}; it holds only {sorted(_METADATA_KEYS)}",
        )
    if "max_pause_s" not in metadata:
        raise _refused(path, f"{METADATA_NAME} has no max_pause_s")
    max_pause_s = _max_pause(path, metadata["max_pause_s"])

    raw_slots = metadata.get("slots", {})
    if not isinstance(raw_slots, Mapping):
        raise _refused(path, f"slots is {type(raw_slots).__name__}, not a mapping")
    slots = {name: _slot(path, name, value, commands) for name, value in raw_slots.items()}

    raw_writes = metadata.get("writes", {})
    if not isinstance(raw_writes, Mapping):
        raise _refused(path, f"writes is {type(raw_writes).__name__}, not a mapping")
    writes: dict[str, tuple[str, ...]] = {}
    for entry, names in raw_writes.items():
        if entry == REQUIRED_ENTRY_POINT:
            raise _refused(
                path,
                f"writes names {REQUIRED_ENTRY_POINT!r}, which writes the injection "
                "request and nothing else",
            )
        if entry not in ENTRY_POINTS:
            raise _refused(
                path, f"writes names {entry!r}, not an entry point ({list(ENTRY_POINTS)})"
            )
        if entry not in defined:
            raise _refused(path, f"writes names {entry!r}, which the module does not define")
        if isinstance(names, str) or not isinstance(names, list | tuple):
            raise _refused(path, f"writes[{entry!r}] is {names!r}, not a list of slot names")
        undeclared = [name for name in names if name not in slots]
        if undeclared:
            raise _refused(path, f"writes[{entry!r}] names undeclared slots {undeclared}")
        if len(set(names)) != len(names):
            raise _refused(path, f"writes[{entry!r}] names a slot twice")
        writes[entry] = tuple(names)

    return SequenceMetadata(
        slots=MappingProxyType(slots),
        writes=MappingProxyType(writes),
        max_pause_s=max_pause_s,
    )


def _entry_points(path: Path, module: ModuleType) -> dict[str, Callable[..., Any]]:
    found: dict[str, Callable[..., Any]] = {}
    for name in ENTRY_POINTS:
        if not hasattr(module, name):
            continue
        function = getattr(module, name)
        if not inspect.iscoroutinefunction(function):
            raise _refused(path, f"entry point {name!r} is not a coroutine function (async def)")
        found[name] = function
    if REQUIRED_ENTRY_POINT not in found:
        raise _refused(path, f"the module defines no {REQUIRED_ENTRY_POINT!r} entry point")
    return found


def load_sequence(
    directory: Path | None, commands: TrajectoryCommands | None
) -> SequenceModule | None:
    """Load and validate the sequence module in *directory*, once.

    Args:
        directory: The resolved sequence directory, or ``None`` when the
            deployment names none.
        commands: The service's command set, or ``None`` when the build has
            no trajectory model.

    Returns:
        The validated module, or ``None`` when *directory* is unset, missing,
        or holds no :data:`SEQUENCE_MODULE_FILE`.

    Raises:
        StartupRefused: The module fails to load or to validate.
    """
    if directory is None:
        logger.info("trajectory service: no sequence directory; shots and setup are refused")
        return None
    path = Path(directory) / SEQUENCE_MODULE_FILE
    if not path.is_file():
        logger.warning(
            "trajectory service: no sequence module at %s; shots and setup are refused", path
        )
        return None
    from osprey.utils.module_loading import load_by_path

    try:
        module = load_by_path(path, SEQUENCE_MODULE_PREFIX)
    except Exception as exc:
        raise _refused(path, f"failed to load: {type(exc).__name__}: {exc}") from exc
    entry_points = _entry_points(path, module)
    if not hasattr(module, METADATA_NAME):
        raise _refused(path, f"the module defines no {METADATA_NAME}")
    metadata = validate_metadata(
        path,
        getattr(module, METADATA_NAME),
        defined=frozenset(entry_points),
        commands=commands,
    )
    logger.info(
        "trajectory service: sequence module %s (entry points %s, slots %s)",
        path,
        ", ".join(entry_points),
        ", ".join(metadata.slots) or "none",
    )
    return SequenceModule(
        path=path,
        module=module,
        metadata=metadata,
        entry_points=MappingProxyType(entry_points),
    )


# -- the handle an entry point runs with ------------------------------------------


class SequenceStopped(Exception):
    """The action's stop event was set; the entry point ends at this checkpoint."""


class SequenceRefused(Refused):
    """A handle call the sequence contract forbids; nothing was read or written for it."""

    statuses = {"sequence_refused": 409}

    def __init__(self, detail: str) -> None:
        super().__init__("sequence_refused", detail)


class SequenceHandle:
    """What one entry point of the sequence module reads, writes and waits through.

    Built for one entry point and one action: the action holds the run lock
    and binds the owner the connector writes for, so the handle neither takes
    the lock nor names an owner.

    Attributes:
        entry_point: The entry point this handle runs.
        moved: Every module-slot write that was not refused, in order, in
            put-back shape.
        commands_fired: Every command address that went out, in order.
    """

    def __init__(
        self,
        connector: Any,
        *,
        commands: TrajectoryCommands | None,
        metadata: SequenceMetadata,
        entry_point: str,
        stop: asyncio.Event,
        tag_addresses: Mapping[str, Sequence[str]] | None = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = CHECKPOINT_POLL_S,
    ) -> None:
        """Build the handle.

        Args:
            connector: The action's connector.
            commands: The service's command set; ``None`` when the build has
                none, and then nothing may request an injection.
            metadata: The module's validated metadata.
            entry_point: The entry point about to run, one of
                :data:`ENTRY_POINTS`.
            stop: The action's stop event.
            tag_addresses: Facility signal to its addresses, one per BPM in
                BPM order, for the per-BPM slots the module declares.
            sleep: Awaited to pause; :func:`asyncio.sleep` unless a test
                injects its own clock.
            clock: Seconds, monotonic; bounds :meth:`wait_for`.
            poll_s: Seconds between two reads of a :meth:`wait_for`.

        Raises:
            ValueError: *entry_point* is not an entry point.
        """
        if entry_point not in ENTRY_POINTS:
            raise ValueError(f"{entry_point!r} is not an entry point ({list(ENTRY_POINTS)})")
        self._connector = connector
        self._commands = commands
        self._metadata = metadata
        self.entry_point = entry_point
        self._stop = stop
        self._tag_addresses = dict(tag_addresses or {})
        self._sleep = sleep
        self._clock = clock
        self._poll_s = poll_s
        self.moved: list[Moved] = []
        self.commands_fired: list[str] = []

    # -- checkpoints -----------------------------------------------------------

    def _checkpoint(self) -> None:
        if self._stop.is_set():
            raise SequenceStopped(f"stopped in {self.entry_point}")

    async def _pause(self, seconds: float) -> None:
        """Pause *seconds*, ending at once when the stop event is set."""
        self._checkpoint()
        sleeping = asyncio.ensure_future(self._sleep(seconds))
        stopping = asyncio.ensure_future(self._stop.wait())
        try:
            await asyncio.wait({sleeping, stopping}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (sleeping, stopping):
                task.cancel()
            await asyncio.gather(sleeping, stopping, return_exceptions=True)
        self._checkpoint()
        if sleeping.done() and not sleeping.cancelled() and sleeping.exception() is not None:
            raise sleeping.exception()  # type: ignore[misc]

    # -- addresses -------------------------------------------------------------

    def _slot_addresses(self, slot: SequenceSlot) -> list[str]:
        if slot.address is not None:
            return [slot.address]
        addresses = list(self._tag_addresses.get(slot.tag or "", ()))
        if not addresses:
            raise SequenceRefused(
                f"slot {slot.name!r} tags {slot.tag!r}, and no BPM carries that signal"
            )
        return addresses

    def _readable(self, name: str) -> list[str] | str:
        """The address(es) *name* reads: the request's, a declared slot's, or *name* itself."""
        if not isinstance(name, str) or not name:
            raise SequenceRefused(f"{name!r} is not a channel name")
        if name == INJECTION_REQUEST and self._commands is not None:
            return self._commands.injection_request
        slot = self._metadata.slots.get(name)
        if slot is None:
            return name
        return self._slot_addresses(slot) if slot.per_bpm else str(slot.address)

    # -- reads -----------------------------------------------------------------

    async def _read_one(self, address: str) -> Any:
        self._checkpoint()
        value = await read_value(self._connector, address)
        self._checkpoint()
        return value

    async def read(self, name: str) -> Any:
        """Read a channel: the injection request or a declared slot by its name, or any address.

        Args:
            name: :data:`INJECTION_REQUEST`, a declared slot, or a channel address.

        Returns:
            The value; for a per-BPM slot, a list in BPM order.

        Raises:
            SequenceStopped: The stop event is set.
            SequenceRefused: *name* is not a channel name.
        """
        target = self._readable(name)
        if isinstance(target, list):
            return [await self._read_one(address) for address in target]
        return await self._read_one(target)

    async def wait_for(self, name: str, condition: Any, *, timeout: float) -> Any:
        """Read *name* until it satisfies *condition*, polling the stop event.

        Args:
            name: A one-address slot, or a channel address.
            condition: A predicate on the value, or the value to equal.
            timeout: Seconds to wait at most, positive.

        Returns:
            The first value that satisfied *condition*.

        Raises:
            SequenceStopped: The stop event was set before the value came.
            SequenceRefused: *name* is a per-BPM slot or *timeout* is not positive.
            TimeoutError: The value did not come within *timeout*.
        """
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not (math.isfinite(timeout) and timeout > 0)
        ):
            raise SequenceRefused(f"wait_for timeout {timeout!r} is not a positive number")
        target = self._readable(name)
        if isinstance(target, list):
            raise SequenceRefused(f"wait_for reads one channel; slot {name!r} is one per BPM")
        matches = condition if callable(condition) else (lambda value: value == condition)
        deadline = self._clock() + timeout
        while True:
            value = await self._read_one(target)
            if matches(value):
                return value
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError(f"{name} did not reach the awaited value within {timeout} s")
            await self._pause(min(self._poll_s, remaining))

    async def sleep(self, seconds: float) -> None:
        """Pause, no longer than the module's ``max_pause_s``, ending early on stop.

        Raises:
            SequenceStopped: The stop event is or becomes set.
            SequenceRefused: *seconds* is negative or above ``max_pause_s``.
        """
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, int | float)
            or not (math.isfinite(seconds) and 0 <= seconds <= self._metadata.max_pause_s)
        ):
            raise SequenceRefused(
                f"sleep {seconds!r} s is outside 0 to max_pause_s ({self._metadata.max_pause_s} s)"
            )
        await self._pause(float(seconds))

    # -- writes ----------------------------------------------------------------

    async def write(self, name: str, value: Any) -> None:
        """Write the injection request, or a module slot this entry point declares.

        Args:
            name: :data:`INJECTION_REQUEST`, or a slot listed in ``writes``
                for this entry point.
            value: The value to write; a per-BPM slot gets it at every BPM.

        Raises:
            SequenceStopped: The stop event is set.
            SequenceRefused: The write is not one this entry point may make;
                nothing was read or written.
            Exception: Whatever the command write or the checked setting
                write raised, after the write was recorded.
        """
        self._checkpoint()
        if name == INJECTION_REQUEST:
            await self._request(value)
            return
        slot = self._metadata.slots.get(name) if isinstance(name, str) else None
        if slot is None:
            raise SequenceRefused(
                f"{self.entry_point} may not write {name!r}: a module writes the injection "
                "request (from inject) and its declared slots, never an arm command or "
                "an address"
            )
        if name not in self._metadata.writes.get(self.entry_point, ()):
            raise SequenceRefused(
                f"{self.entry_point} may not write slot {name!r}; its writes are "
                f"{list(self._metadata.writes.get(self.entry_point, ()))}"
            )
        for address in self._slot_addresses(slot):
            await self._set(address, value)

    async def _request(self, value: Any) -> None:
        if self.entry_point != REQUIRED_ENTRY_POINT:
            raise SequenceRefused(
                f"{self.entry_point} may not write the injection request; only "
                f"{REQUIRED_ENTRY_POINT} requests an injection"
            )
        commands = self._commands
        if commands is None:
            raise SequenceRefused("the build has no injection request to write")
        from osprey.runtime.guarded_run import command_write

        def record(outcome: Any) -> None:
            if isinstance(outcome, list):
                self.commands_fired.extend(outcome)
            else:
                self.commands_fired.extend(getattr(outcome, "commands_fired", None) or ())

        await guarded_call(
            lambda: command_write(
                self._connector,
                [commands.injection_request],
                value,
                commands=commands.commands,
            ),
            record=record,
        )
        self._checkpoint()

    async def _set(self, address: str, value: Any) -> None:
        from osprey_connectors.errors import ChannelWriteBlockedError

        before = await self._read_one(address)

        def record(outcome: Any) -> None:
            if isinstance(outcome, ChannelWriteBlockedError):
                return
            if isinstance(outcome, BaseException):
                # The write may have landed; it is listed so it can be put back.
                word = getattr(outcome, "outcome", None) or "failed"
            else:
                word = outcome.outcome
            self.moved.append(Moved(name=address, before=before, outcome=str(word)))

        # The confirm is left to the connector, which takes each channel's
        # from its limits record.
        await guarded_call(
            lambda: self._connector.write_channel_checked(address, value), record=record
        )
        self._checkpoint()
