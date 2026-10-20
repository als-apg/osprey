"""The facility's sequence module: what fires the injection.

A sequence module is one ``.py`` file, :data:`SEQUENCE_MODULE_FILE`, in the
directory :func:`~.startup.resolve_sequence_dir` resolves. It is loaded once,
at service start, through :func:`~osprey.utils.module_loading.load_by_path`,
and validated before the service serves anything:

* **``inject``** is its one entry point, a coroutine function taking the
  sequence handle, which fires one injection. A module without it, or whose
  ``inject`` is not a coroutine function, is refused. Everything else the
  service writes around an injection -- arming, settings, cleaning -- is
  declared in the measurement file's ``trajectory`` block, never here.
* **``SEQUENCE_METADATA``** is a mapping holding exactly ``max_pause_s``: the
  longest the module ever waits between two calls, a positive number no
  larger than :data:`~.constants.MAX_PAUSE_CEILING_S`.

A directory that is unset, missing, or holds no :data:`SEQUENCE_MODULE_FILE`
leaves the service without a module: it starts, and its shots refuse
``no_sequence_module``. A module that does not load or does not validate
refuses the service's start, naming the file and what is wrong.

``inject`` runs with a :class:`SequenceHandle`, built for one injection inside
an action that holds the run lock:

* ``write("injection_request", value)`` fires the injection request through
  :func:`~osprey.runtime.guarded_run.command_write`. Any other write is
  refused before anything is read or written: the module never arms and
  never moves a setting.
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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from osprey.services.trajectory.calls import guarded_call, read_value
from osprey.services.trajectory.constants import CHECKPOINT_POLL_S, MAX_PAUSE_CEILING_S
from osprey.services.trajectory.refusal import Refused

if TYPE_CHECKING:
    from osprey.services.trajectory.startup import StartupRefused, TrajectoryCommands

__all__ = [
    "INJECTION_REQUEST",
    "METADATA_NAME",
    "REQUIRED_ENTRY_POINT",
    "SEQUENCE_MODULE_FILE",
    "SEQUENCE_MODULE_PREFIX",
    "SequenceHandle",
    "SequenceMetadata",
    "SequenceModule",
    "SequenceRefused",
    "SequenceStopped",
    "load_sequence",
    "validate_metadata",
]

logger = logging.getLogger(__name__)

#: The file in the sequence directory that holds the module.
SEQUENCE_MODULE_FILE = "sequence.py"
#: The ``sys.modules`` prefix a loaded sequence module is registered under.
SEQUENCE_MODULE_PREFIX = "osprey_trajectory_sequence_"
#: The module-level mapping that declares the longest pause.
METADATA_NAME = "SEQUENCE_METADATA"
#: The module's one entry point.
REQUIRED_ENTRY_POINT = "inject"
#: The name a module writes the injection request by.
INJECTION_REQUEST = "injection_request"

_METADATA_KEYS = frozenset({"max_pause_s"})


@dataclass(frozen=True)
class SequenceMetadata:
    """A module's validated ``SEQUENCE_METADATA``.

    Attributes:
        max_pause_s: The longest the module waits between two calls.
    """

    max_pause_s: float


@dataclass(frozen=True)
class SequenceModule:
    """A loaded, validated sequence module.

    Attributes:
        path: The file it was loaded from.
        module: The loaded module.
        metadata: Its validated ``SEQUENCE_METADATA``.
        inject: The coroutine function that fires one injection.
    """

    path: Path
    module: ModuleType
    metadata: SequenceMetadata
    inject: Callable[..., Any]


def _refused(path: Path, problem: str) -> StartupRefused:
    from osprey.services.trajectory.startup import StartupRefused

    return StartupRefused(f"sequence module {path}: {problem}")


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


def validate_metadata(path: Path, metadata: Any) -> SequenceMetadata:
    """Validate a module's ``SEQUENCE_METADATA``.

    Args:
        path: The module's file, named in every refusal.
        metadata: The module's ``SEQUENCE_METADATA`` as it found it.

    Returns:
        The validated metadata.

    Raises:
        StartupRefused: The metadata is not a mapping holding exactly a
            valid ``max_pause_s``.
    """
    if not isinstance(metadata, Mapping):
        raise _refused(path, f"{METADATA_NAME} is {type(metadata).__name__}, not a mapping")
    unknown = sorted(str(key) for key in set(metadata) - _METADATA_KEYS)
    if unknown:
        raise _refused(
            path,
            f"{METADATA_NAME} has unknown keys {unknown}; it holds only "
            f"{sorted(_METADATA_KEYS)} (settings and command steps are declared in the "
            "measurement file's trajectory block)",
        )
    if "max_pause_s" not in metadata:
        raise _refused(path, f"{METADATA_NAME} has no max_pause_s")
    return SequenceMetadata(max_pause_s=_max_pause(path, metadata["max_pause_s"]))


def _inject(path: Path, module: ModuleType) -> Callable[..., Any]:
    if not hasattr(module, REQUIRED_ENTRY_POINT):
        raise _refused(path, f"the module defines no {REQUIRED_ENTRY_POINT!r} entry point")
    function: Callable[..., Any] = getattr(module, REQUIRED_ENTRY_POINT)
    if not inspect.iscoroutinefunction(function):
        raise _refused(
            path, f"entry point {REQUIRED_ENTRY_POINT!r} is not a coroutine function (async def)"
        )
    return function


def load_sequence(directory: Path | None) -> SequenceModule | None:
    """Load and validate the sequence module in *directory*, once.

    Args:
        directory: The resolved sequence directory, or ``None`` when the
            deployment names none.

    Returns:
        The validated module, or ``None`` when *directory* is unset, missing,
        or holds no :data:`SEQUENCE_MODULE_FILE`.

    Raises:
        StartupRefused: The module fails to load or to validate.
    """
    if directory is None:
        logger.info("trajectory service: no sequence directory; shots are refused")
        return None
    path = Path(directory) / SEQUENCE_MODULE_FILE
    if not path.is_file():
        logger.warning("trajectory service: no sequence module at %s; shots are refused", path)
        return None
    from osprey.utils.module_loading import load_by_path

    try:
        module = load_by_path(path, SEQUENCE_MODULE_PREFIX)
    except Exception as exc:
        raise _refused(path, f"failed to load: {type(exc).__name__}: {exc}") from exc
    inject = _inject(path, module)
    if not hasattr(module, METADATA_NAME):
        raise _refused(path, f"the module defines no {METADATA_NAME}")
    metadata = validate_metadata(path, getattr(module, METADATA_NAME))
    logger.info(
        "trajectory service: sequence module %s (max_pause_s %g)", path, metadata.max_pause_s
    )
    return SequenceModule(path=path, module=module, metadata=metadata, inject=inject)


# -- the handle inject runs with ----------------------------------------------------


class SequenceStopped(Exception):
    """The action's stop event was set; ``inject`` ends at this checkpoint."""


class SequenceRefused(Refused):
    """A handle call the sequence contract forbids; nothing was read or written for it."""

    statuses = {"sequence_refused": 409}

    def __init__(self, detail: str) -> None:
        super().__init__("sequence_refused", detail)


class SequenceHandle:
    """What ``inject`` reads, writes and waits through, for one injection.

    Built inside an action that holds the run lock and binds the owner the
    connector writes for, so the handle neither takes the lock nor names an
    owner.

    Attributes:
        commands_fired: Every command address that went out, in order.
    """

    def __init__(
        self,
        connector: Any,
        *,
        commands: TrajectoryCommands | None,
        metadata: SequenceMetadata,
        stop: asyncio.Event,
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
            stop: The action's stop event.
            sleep: Awaited to pause; :func:`asyncio.sleep` unless a test
                injects its own clock.
            clock: Seconds, monotonic; bounds :meth:`wait_for`.
            poll_s: Seconds between two reads of a :meth:`wait_for`.
        """
        self._connector = connector
        self._commands = commands
        self._metadata = metadata
        self._stop = stop
        self._sleep = sleep
        self._clock = clock
        self._poll_s = poll_s
        self.commands_fired: list[str] = []

    # -- checkpoints -----------------------------------------------------------

    def _checkpoint(self) -> None:
        if self._stop.is_set():
            raise SequenceStopped(f"stopped in {REQUIRED_ENTRY_POINT}")

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

    # -- reads -----------------------------------------------------------------

    def _readable(self, name: str) -> str:
        """The address *name* reads: the request's, or *name* itself."""
        if not isinstance(name, str) or not name:
            raise SequenceRefused(f"{name!r} is not a channel name")
        if name == INJECTION_REQUEST and self._commands is not None:
            return self._commands.injection_request
        return name

    async def _read_one(self, address: str) -> Any:
        self._checkpoint()
        value = await read_value(self._connector, address)
        self._checkpoint()
        return value

    async def read(self, name: str) -> Any:
        """Read a channel: the injection request by its name, or any address.

        Args:
            name: :data:`INJECTION_REQUEST`, or a channel address.

        Returns:
            The value.

        Raises:
            SequenceStopped: The stop event is set.
            SequenceRefused: *name* is not a channel name.
        """
        return await self._read_one(self._readable(name))

    async def wait_for(self, name: str, condition: Any, *, timeout: float) -> Any:
        """Read *name* until it satisfies *condition*, polling the stop event.

        Args:
            name: :data:`INJECTION_REQUEST`, or a channel address.
            condition: A predicate on the value, or the value to equal.
            timeout: Seconds to wait at most, positive.

        Returns:
            The first value that satisfied *condition*.

        Raises:
            SequenceStopped: The stop event was set before the value came.
            SequenceRefused: *timeout* is not positive.
            TimeoutError: The value did not come within *timeout*.
        """
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not (math.isfinite(timeout) and timeout > 0)
        ):
            raise SequenceRefused(f"wait_for timeout {timeout!r} is not a positive number")
        target = self._readable(name)
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

    # -- the request -----------------------------------------------------------

    async def write(self, name: str, value: Any) -> None:
        """Write the injection request.

        Args:
            name: :data:`INJECTION_REQUEST`; any other name is refused.
            value: The value to write.

        Raises:
            SequenceStopped: The stop event is set.
            SequenceRefused: *name* is not the injection request, or the build
                has none; nothing was read or written.
            Exception: Whatever the command write raised, after the commands
                that went out were recorded.
        """
        self._checkpoint()
        if name != INJECTION_REQUEST:
            raise SequenceRefused(
                f"{REQUIRED_ENTRY_POINT} may not write {name!r}: a sequence module writes "
                "the injection request and nothing else"
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
