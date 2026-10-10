"""The simulator, served in this process: the built facility's simulator, with no network.

The connector serves exactly the channel and status addresses of the simulator view
``osprey build`` writes at ``data/simulator/`` beside the rendered config,
through the composite (:class:`~osprey_connectors.simulation.composite.Composite`):
the texture and one physics child per served model whose engine is not
``texture``. An address outside the view is refused with
``<address> is not in build/facility.json``.

* A read carries the composite's value, motion included; a ``bool`` or
  ``enum`` channel reads as its option index, a waveform as an array of its
  shape. A channel of a failed model reads with alarm severity 3 (``UDF``).
* A confirming read carries the held value, without motion.
* Metadata comes from the channel's record in ``variables.json``: its
  description, and the unit of its variable.
* A write to a channel that is not a writable setpoint is refused before
  anything is put; a write the composite refuses is refused with the
  composite's text.
* One tick task runs at ``simulation.tick_s``; after every tick and every
  write a subscription fires for its channel when the held value changed or
  the channel declares motion.

**Session writes.** Only :meth:`VAInProcessConnector.write_channel` journals: after
the limits validator and a successful put it appends ``[seq, address, value]``
to ``<simulation state dir>/inprocess/writes.json``, a document
``{active_set_sha256, seq, writes}`` updated under ``flock`` on
``writes.json.lock`` and replaced atomically. Every operation reads the active
scenario set first, then the journal: when ``seq`` moved, the composite is
reset and the journal applied as one write, provided every entry is a setpoint
of the view, writable, inside its band, and was written under the current
active set; otherwise nothing from it is applied, one line
``writes journal rejected: <address>: <reason>`` is logged, a
``journal-rejected`` record is appended to every physics model's log, and the
journal is emptied, so the writes that follow are journalled afresh. A reset or a
change of the active set empties the journal. The journal is this connector's alone:
no hardware path reads it.

``lume`` and the simulation package are imported when used, never at module import.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import inspect
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey_connectors.config import get_facility_timezone
from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
    values_match,
)
from osprey_connectors.logger import get_logger
from osprey_connectors.simulation.view import (
    VARIABLES_FILE,
    VIEW_RELPATH,
    NoSimulatorView,
    SimulatorView,
    ViewSchemaError,
)

if TYPE_CHECKING:
    from osprey_connectors.simulation.composite import Composite

logger = get_logger("va_in_process_connector")

__all__ = [
    "JOURNAL_DIR",
    "JOURNAL_FILE",
    "JOURNAL_NOT_UPDATED",
    "JOURNAL_REJECTED",
    "JOURNAL_REJECTED_EVENT",
    "NO_VIEW_MESSAGE",
    "SIMULATOR_VIEW_SETTING",
    "VAInProcessConnector",
    "active_set_sha256",
    "not_in_facility",
    "simulation_state_dir",
    "simulator_view_dir",
]

#: The ``connect()`` setting naming the simulator view directory to serve.
SIMULATOR_VIEW_SETTING = "simulator_view"

#: Why the in-process simulator refuses to connect without a built simulator view.
NO_VIEW_MESSAGE = "the in-process simulator needs a built simulator view: run osprey build"

#: Why a write to a channel that is not a writable setpoint is refused.
NOT_WRITABLE = "not a writable setpoint"

#: The alarm a read of a failed model's channel carries.
UDF_SEVERITY = 3
UDF_STATUS = "UDF"

_RENDERED_CONFIG = "config.yml"
_LABELLED = ("bool", "enum")
_REFUSED_BY_SIMULATOR = "CONTROL_SYSTEM_REFUSED"

#: The session-writes journal, under the simulation state directory.
JOURNAL_DIR = "inprocess"
JOURNAL_FILE = "writes.json"
_JOURNAL_LOCK = f"{JOURNAL_FILE}.lock"

#: The line a journal that is not replayed logs, before ``<address>: <reason>``.
JOURNAL_REJECTED = "writes journal rejected"

#: The event a journal that is not replayed appends to every physics model's log.
JOURNAL_REJECTED_EVENT = "journal-rejected"

#: The line a write the journal could not record logs, before the error.
JOURNAL_NOT_UPDATED = "writes journal not updated"

#: Why an operation on a connector that is not connected is refused.
NOT_CONNECTED = "the in-process simulator is not connected"

#: Why an address the built facility file does not hold is refused.
_NOT_IN_FACILITY = "not in build/facility.json"


def not_in_facility(address: str) -> str:
    """The refusal for an address the built facility file does not hold."""
    return f"{address} is {_NOT_IN_FACILITY}"


def _loaded_config_path() -> str | None:
    """The config this process's unqualified lookups read, loading it when none is."""
    from osprey_connectors.config import default_config_path, get_config_builder

    path = default_config_path()
    if path is not None:
        return path
    try:
        get_config_builder()
    except (FileNotFoundError, IsADirectoryError, KeyError, RuntimeError, ValueError):
        return None
    return default_config_path()


def simulator_view_dir(setting: str | Path | None = None) -> SimulatorView:
    """The simulator view the in-process simulator serves, opened.

    Args:
        setting: The view directory a ``connect()`` call names; ``None`` reads
            ``data/simulator/`` beside the config this process loaded.

    Returns:
        The view.

    Raises:
        RuntimeError: There is no view there, the message being
            :data:`NO_VIEW_MESSAGE`; or the view is from an older build, the
            message saying to rebuild.
    """
    if setting:
        path = Path(setting).expanduser()
    else:
        config_path = _loaded_config_path()
        if config_path is None:
            raise RuntimeError(NO_VIEW_MESSAGE)
        path = Path(config_path).parent / VIEW_RELPATH
    try:
        return SimulatorView.open(path)
    except NoSimulatorView:
        raise RuntimeError(NO_VIEW_MESSAGE) from None
    except ViewSchemaError as error:
        raise RuntimeError(str(error)) from None


def _rendered_config(view: Path) -> tuple[Path, dict[str, Any]]:
    """The rendered config beside a view, and its contents; empty when there is none."""
    from osprey_connectors.config import load_project_config

    path = view.parent.parent / _RENDERED_CONFIG
    if not path.is_file():
        return path, {}
    try:
        return path, load_project_config(path)
    except Exception as exc:
        logger.warning(f"cannot read {path} beside the simulator view: {exc}")
        return path, {}


def simulation_state_dir(view: Path) -> Path:
    """The simulation state directory of the render a view belongs to.

    The directory holding the ``active_scenarios`` file, resolved from the
    rendered config beside the view as every simulation reader resolves it.
    """
    from osprey_connectors.workspace import repo_root_for_config, resolve_simulation_state_dir

    config_path, config = _rendered_config(view)
    root = Path(str(config.get("project_root") or repo_root_for_config(config_path)))
    return resolve_simulation_state_dir(config, root)


def active_set_sha256(active: Sequence[str]) -> str:
    """The digest a journal records for the active scenario set it was written under."""
    return hashlib.sha256("\n".join(active).encode("utf-8")).hexdigest()


class _Journal:
    """The session-writes journal: read whole, replaced atomically under its lock."""

    def __init__(self, state_dir: Path) -> None:
        self.path = state_dir / JOURNAL_DIR / JOURNAL_FILE
        self._lock = self.path.with_name(_JOURNAL_LOCK)

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock, "a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self) -> tuple[Any, str]:
        """The parsed document and its text; ``(None, "")`` when there is none."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None, ""
        try:
            return json.loads(text), text
        except ValueError:
            return None, text

    def write(self, document: Mapping[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{JOURNAL_FILE}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(document, handle, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise


def _seq(document: Any) -> int | None:
    """A journal document's ``seq``, or ``None`` when it is not a journal."""
    if not isinstance(document, dict) or not isinstance(document.get("writes"), list):
        return None
    seq = document.get("seq")
    return seq if isinstance(seq, int) and not isinstance(seq, bool) else None


def _mark(document: Any, text: str) -> int | str:
    """What one look at the journal saw: its ``seq``, else its text, ``0`` when empty."""
    seq = _seq(document)
    return (0 if not text else text) if seq is None else seq


class VAInProcessConnector(ControlSystemConnector):
    """Serve the built facility's simulator view in process.

    Example:
        >>> connector = VAInProcessConnector()
        >>> await connector.connect({"response_delay_ms": 10})
        >>> value = await connector.read_channel("SR:DIAG:BPM:01:POSITION:X")
    """

    def __init__(self) -> None:
        self._connected = False
        self._composite: Composite | None = None
        self._records: dict[str, Mapping[str, Any]] = {}
        self._served: frozenset[str] = frozenset()
        self._moving: frozenset[str] = frozenset()
        self._subscriptions: dict[str, tuple[str, Callable[[ChannelValue], Any]]] = {}
        self._last_held: dict[str, Any] = {}
        self._tick_task: asyncio.Task[None] | None = None
        self._tick_s = 1.0
        self._channels: frozenset[str] = frozenset()
        self._journal: _Journal | None = None
        self._active_sha: str | None = None
        self._journal_mark: int | str | None = None
        self._state_file: Path | None = None
        self._state_mark: int | None = None
        self._holds_writes = False

    async def connect(self, config: dict[str, Any]) -> None:
        """Build the composite of the simulator view and start the tick.

        Args:
            config: Connector settings:
                - response_delay_ms: Simulated response delay (default: 10)
                - simulator_view: The view directory to serve; absent serves
                  ``data/simulator/`` beside the loaded config.

        Raises:
            RuntimeError: There is no built simulator view, or it is from an
                older build.
            ValueError: ``simulation.tick_s`` in the rendered config is not a
                number greater than zero.
        """
        from osprey_connectors.simulation import resolve_tick_s
        from osprey_connectors.simulation.composite import Composite

        self._response_delay = config.get("response_delay_ms", 10) / 1000.0

        from osprey_connectors.control_system.limits_validator import LimitsValidator

        self._limits_validator = LimitsValidator.from_config(connector_type=self._connector_type)
        if self._limits_validator:
            logger.debug("In-process simulator: limits validator initialized")

        view = simulator_view_dir(config.get(SIMULATOR_VIEW_SETTING))
        _config_path, rendered = _rendered_config(view.path)
        self._tick_s = resolve_tick_s(rendered)
        state_dir = simulation_state_dir(view.path)
        try:
            self._records = {
                str(entry["address"]): entry for entry in view.document(VARIABLES_FILE)["channels"]
            }
            self._moving = frozenset(view.moving())
            self._channels = frozenset(view.channels())
            self._served = self._channels | frozenset(view.status_addresses().values())
            self._composite = Composite(view, state_dir=state_dir, instance="inprocess")
        except ViewSchemaError as error:
            raise RuntimeError(str(error)) from None
        self._journal = _Journal(state_dir)
        from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME

        self._state_file = state_dir / ACTIVE_SCENARIOS_FILENAME
        self._state_mark = None
        self._active_sha = None
        self._journal_mark = None
        self._holds_writes = False
        self._sync()
        self._connected = True
        self._tick_task = asyncio.create_task(self._tick())
        logger.debug(f"In-process simulator serving {view.path}")

    async def disconnect(self) -> None:
        """Stop the tick and drop the composite."""
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except (asyncio.CancelledError, Exception):
                pass
            self._tick_task = None
        self._subscriptions.clear()
        self._last_held.clear()
        self._composite = None
        self._records = {}
        self._served = frozenset()
        self._channels = frozenset()
        self._moving = frozenset()
        self._connected = False
        logger.debug("In-process simulator disconnected")

    # -- the journal -----------------------------------------------------------

    def _sync(self) -> None:
        """Read the active set, then the journal; replay the journal when it moved."""
        if self._composite is None or self._journal is None:
            return
        # The state file is looked at before the composite looks at it, so a
        # rewrite between the two looks shows here on the next operation.
        state_mark = self._state_stamp()
        sha = active_set_sha256(self._composite.active)
        rebuilt = state_mark != self._state_mark and self._active_sha is not None
        self._state_mark = state_mark
        if sha != self._active_sha:
            changed = self._active_sha is not None
            self._active_sha = sha
            # The first connector to see a new active set empties the journal; one
            # that sees it later replays what was written under that set.
            if changed:
                self._holds_writes = False
                self._journal_mark = None
                if self._truncate(unless_current=True):
                    return
        if rebuilt:
            # The composite rebuilt from a rewrite of the same set and dropped
            # the session writes; the journal still holds them.
            self._holds_writes = False
            self._journal_mark = None
        document, text = self._journal.read()
        mark = _mark(document, text)
        if mark == self._journal_mark:
            return
        self._journal_mark = mark
        self._replay(document if _seq(document) is not None else None, text)

    def _state_stamp(self) -> int | None:
        """The ``active_scenarios`` file's mtime, ``None`` when there is none."""
        if self._state_file is None:
            return None
        try:
            return self._state_file.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _replay(self, document: Mapping[str, Any] | None, text: str) -> None:
        """Reset the composite and apply the journal as one write, when it is well formed."""
        assert self._composite is not None
        if self._holds_writes:
            self._composite.reset()
            self._holds_writes = False
        if document is None:
            if text:
                self._reject(JOURNAL_FILE, "not a journal")
            return
        writes = document["writes"]
        if not writes:
            return
        rejection = self._rejection(document)
        if rejection is not None:
            self._reject(*rejection)
            return
        replayed: dict[str, Any] = {}
        for _seq_no, address, value in sorted(writes, key=lambda entry: entry[0]):
            replayed[address] = value
        try:
            self._composite.set(replayed)
        except Exception as exc:
            self._composite.reset()
            self._reject(next(iter(replayed)), str(exc))
            return
        self._holds_writes = True

    def _reject(self, address: str, reason: str) -> None:
        """Log a journal that is not replayed, in the process log and every physics
        model's, then empty it unless another connector moved it since."""
        logger.warning(f"{JOURNAL_REJECTED}: {address}: {reason}")
        if self._composite is not None:
            self._composite.log_event(JOURNAL_REJECTED_EVENT, address=address, reason=reason)
        self._truncate(unless_moved=True)

    def _rejection(self, document: Mapping[str, Any]) -> tuple[str, str] | None:
        """The first address and reason that keep a journal from being replayed."""
        writes = document["writes"]
        for entry in writes:
            if (
                not isinstance(entry, list)
                or len(entry) != 3
                or not isinstance(entry[0], int)
                or isinstance(entry[0], bool)
                or not isinstance(entry[1], str)
            ):
                return JOURNAL_FILE, f"malformed entry {entry!r}"
        if document.get("active_set_sha256") != self._active_sha:
            return writes[0][1], "written under another active scenario set"
        for _seq_no, address, value in writes:
            record = self._records.get(address)
            if record is None or address not in self._channels:
                return address, _NOT_IN_FACILITY
            if record.get("role") != "setpoint":
                return address, "not a setpoint"
            if record.get("writable") is not True:
                return address, NOT_WRITABLE
            band = record.get("value_range")
            if band:
                low, high = band
                number = value if isinstance(value, int | float) else None
                if (
                    number is None
                    or isinstance(number, bool)
                    or math.isnan(number)
                    or (low is not None and number < low)
                    or (high is not None and number > high)
                ):
                    return address, f"{value!r} is outside [{low}, {high}]"
        return None

    def _append(self, address: str, value: Any) -> None:
        """Append one write the composite took, under the journal's lock."""
        assert self._journal is not None
        with self._journal.locked():
            document, _text = self._journal.read()
            seq = _seq(document)
            synced = seq == self._journal_mark or (seq is None and self._journal_mark == 0)
            writes: list[Any] = []
            if seq is not None and document.get("active_set_sha256") == self._active_sha:
                writes = list(document["writes"])
            seq = (seq or 0) + 1
            writes.append([seq, address, value])
            self._journal.write(
                {"active_set_sha256": self._active_sha, "seq": seq, "writes": writes}
            )
        self._holds_writes = True
        # A journal another process moved since the last look is replayed on
        # the next operation, this write with it.
        self._journal_mark = seq if synced else None

    def _truncate(self, *, unless_current: bool = False, unless_moved: bool = False) -> bool:
        """Empty the journal under its lock, recording the current active set; with
        ``unless_current`` a journal already written under that set is kept, with
        ``unless_moved`` a journal that changed since the last look is kept.
        Returns whether the journal was emptied."""
        assert self._journal is not None
        with self._journal.locked():
            document, text = self._journal.read()
            if (
                unless_current
                and _seq(document) is not None
                and document.get("active_set_sha256") == self._active_sha
            ):
                return False
            if unless_moved and _mark(document, text) != self._journal_mark:
                return False
            seq = (_seq(document) or 0) + 1
            self._journal.write({"active_set_sha256": self._active_sha, "seq": seq, "writes": []})
        self._journal_mark = seq
        return True

    async def reset(self) -> None:
        """Return the simulator to its start state and empty the journal.

        Raises:
            RuntimeError: The connector is not connected.
        """
        if self._composite is None:
            raise RuntimeError(NOT_CONNECTED)
        self._sync()
        self._composite.reset()
        self._holds_writes = False
        self._truncate()

    # -- the view ------------------------------------------------------------

    def _require(self, address: str) -> Composite:
        """The composite, once ``address`` is one it serves.

        Raises:
            ValueError: ``address`` is not in the view.
            RuntimeError: The connector is not connected.
        """
        if self._composite is None:
            raise RuntimeError(NOT_CONNECTED)
        if address not in self._served:
            raise ValueError(not_in_facility(address))
        return self._composite

    def _wire(self, address: str, value: Any) -> Any:
        """A value as the wire carries it: a label as its index, a waveform as an array."""
        record = self._records.get(address)
        if record is None:
            return value
        value_type = record.get("value_type")
        if value_type in _LABELLED:
            from osprey_connectors.simulation import values

            options = values.channel_labels(record)
            return options.index(value) if value in options else value
        if value_type == "waveform":
            import numpy as np

            shape = tuple(int(size) for size in record.get("shape") or ())
            return np.asarray(value, dtype=np.float64).reshape(shape or (-1,))
        return value

    def _metadata(self, address: str, composite: Composite) -> ChannelMetadata:
        """Metadata from the channel record, alarm from the composite's severity."""
        record = self._records.get(address, {})
        variable = composite.supported_variables[address]
        unit = getattr(variable, "unit", None) or record.get("unit") or ""
        severity = composite.output_severity([address])
        labels: list[str] | None = None
        if record.get("value_type") in _LABELLED:
            from osprey_connectors.simulation import values

            labels = values.channel_labels(record)
        return ChannelMetadata(
            units=str(unit),
            timestamp=datetime.now(get_facility_timezone()),
            description=record.get("description"),
            alarm_severity=UDF_SEVERITY if address in severity else None,
            alarm_status=UDF_STATUS if address in severity else None,
            enum_labels=labels,
        )

    def _reading(self, address: str, *, held: bool) -> ChannelValue:
        composite = self._require(address)
        stored = composite.held([address])[address] if held else composite.get(address)
        metadata = self._metadata(address, composite)
        if metadata.enum_labels is not None and stored in metadata.enum_labels:
            metadata.enum_label = str(stored)
        return ChannelValue(
            value=self._wire(address, stored),
            timestamp=metadata.timestamp or datetime.now(get_facility_timezone()),
            metadata=metadata,
        )

    def _held_value(self, address: str) -> Any:
        composite = self._require(address)
        return self._wire(address, composite.held([address])[address])

    # -- reads -----------------------------------------------------------------

    async def read_channel(
        self,
        channel_address: str,
        timeout: float | None = None,  # noqa: ARG002 - ControlSystemConnector.read_channel signature; an in-process read never blocks
    ) -> ChannelValue:
        """Read a channel as the simulator serves it, motion included.

        Raises:
            ValueError: The address is not in the built facility file.
        """
        await asyncio.sleep(self._response_delay)
        self._sync()
        return self._reading(channel_address, held=False)

    async def _confirming_read(self, channel_address: str) -> ChannelValue:
        """Read a channel back to confirm a write: the held value, without motion.

        Confirmation reports what the simulated control system holds; the
        motion ``read_channel`` adds models measuring a live signal, so adding
        it here would manufacture a mismatch on every write.
        """
        await asyncio.sleep(self._response_delay)
        self._sync()
        return self._reading(channel_address, held=True)

    def _current_value_reader(self) -> Callable[[str], Any] | None:
        """What the simulated control system holds, read without motion."""
        return self._held_value

    async def read_multiple_channels(
        self,
        channel_addresses: list[str],
        timeout: float | None = None,
    ) -> dict[str, ChannelValue]:
        """Read multiple channels concurrently."""
        return await self._read_concurrently(channel_addresses, timeout)

    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        """The channel record's description and its variable's unit.

        Raises:
            ValueError: The address is not in the built facility file.
        """
        composite = self._require(channel_address)
        self._sync()
        return self._metadata(channel_address, composite)

    async def validate_channel(self, channel_address: str) -> bool:
        """Whether the built facility file holds the address."""
        return channel_address in self._served

    # -- writes ----------------------------------------------------------------

    def _refused(self, address: str, value: Any, message: str) -> ChannelWriteResult:
        logger.warning(f"In-process write refused: {address}: {message}")
        return ChannelWriteResult(
            channel_address=address,
            value_written=value,
            outcome=WriteOutcome.REFUSED,
            refusal_reason=_REFUSED_BY_SIMULATOR,
            error_message=message,
        )

    def _coerce(self, address: str, value: Any) -> Any:
        """``value`` in the stored representation of the channel's ``value_type``.

        Raises:
            ValueError: The channel cannot hold the value.
        """
        from osprey_connectors.simulation import values

        record = self._records[address]
        if hasattr(value, "tolist"):
            value = value.tolist()
        return values.coerce(
            value, record.get("value_type"), record.get("options"), record.get("shape")
        )

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,  # noqa: ARG002 - ControlSystemConnector.write_channel signature; an in-process write never blocks
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """Write a value to a setpoint, confirming it unless asked not to.

        In order: an address outside the view, or a channel that is not a
        writable setpoint, is refused; the limits are validated; the value is
        put into the composite, whose refusal is a refusal with its text; the
        subscriptions fire; the channel is re-read and compared.

        Args:
            channel_address: A writable setpoint of the view.
            value: Value to write.
            timeout: Ignored in process.
            confirm: Whether to re-read the channel and compare, or ``None`` to
                resolve the policy for this channel from the limits database.

        Returns:
            ChannelWriteResult carrying the outcome and what the channel was
            seen to hold.

        Raises:
            ChannelLimitsViolationError: If limits validation fails (when enabled)
            RuntimeError: The connector is not connected.
        """
        if self._composite is None:
            raise RuntimeError(NOT_CONNECTED)
        if channel_address not in self._served:
            return self._refused(channel_address, value, not_in_facility(channel_address))
        self._sync()
        if self._composite.supported_variables[channel_address].read_only:
            return self._refused(
                channel_address, value, f"Write to '{channel_address}' refused: {NOT_WRITABLE}"
            )

        # Validate limits (FAIL CLOSED). A limits violation propagates
        # unchanged; any other error means the check could not be made, and an
        # unmade check is not permission to write.
        if self._limits_validator:
            from osprey_connectors.errors import ChannelLimitsViolationError

            try:
                self._limits_validator.validate(
                    channel_address, value, read_current=self._current_value_reader()
                )
                logger.debug(f"✓ Limits validation passed: {channel_address}={value}")
            except ChannelLimitsViolationError:
                raise
            except Exception as e:
                return self._validation_refusal(channel_address, value, e)

        if confirm is None:
            confirm = self._resolve_confirm(channel_address)

        await asyncio.sleep(self._response_delay)

        try:
            stored = self._coerce(channel_address, value)
        except Exception as e:
            logger.warning(f"In-process write failed for {channel_address}: {e}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.FAILED,
                error_message=f"In-process write failed: {e}",
            )

        before = self._subscribed_held()
        try:
            self._put(channel_address, stored)
        except Exception as e:
            from lume.exceptions import ReadOnlyError

            if isinstance(e, ValueError | ReadOnlyError):
                return self._refused(channel_address, value, str(e))
            logger.warning(f"In-process write failed for {channel_address}: {e}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.FAILED,
                error_message=f"In-process write failed: {e}",
            )
        try:
            self._append(channel_address, stored)
        except OSError as exc:
            # The composite holds the write; only its survival across
            # connectors is lost.
            self._holds_writes = True
            logger.warning(f"{JOURNAL_NOT_UPDATED}: {exc}")
        await self._notify(before)

        if not confirm:
            logger.debug(f"In-process write (unconfirmed by policy): {channel_address} = {value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNREQUESTED,
                notes="Confirmation not requested (in process)",
            )

        try:
            observed = await self._confirming_read(channel_address)
        except Exception as e:
            logger.warning(f"In-process confirming read failed for {channel_address}: {e}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNCONFIRMED,
                error_message=f"In-process confirming read failed: {e}",
                notes=f"Confirming read raised: {e} (in process)",
            )

        outcome = (
            WriteOutcome.CONFIRMED
            if values_match(value, observed.value, enum_label=observed.metadata.enum_label)
            else WriteOutcome.MISMATCH
        )
        if outcome is WriteOutcome.MISMATCH:
            logger.warning(
                f"In-process write mismatch: {channel_address} sent {value}, observed {observed.value}"
            )
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=outcome,
            observed_value=observed.value,
            alarm_status=observed.metadata.alarm_status,
            alarm_severity=observed.metadata.alarm_severity,
            notes=f"Observed {observed.value}, sent {value} (in process)",
        )

    def _put(self, channel_address: str, value: Any) -> None:
        """Set ``value`` in the composite; raises whatever the composite raises."""
        self._require(channel_address).set({channel_address: value})

    # -- subscriptions and the tick ------------------------------------------

    async def subscribe(
        self, channel_address: str, callback: Callable[[ChannelValue], None]
    ) -> str:
        """Subscribe to a channel; the callback fires after a tick or a write.

        Raises:
            ValueError: The address is not in the built facility file.
        """
        self._require(channel_address)
        self._sync()
        sub_id = f"inprocess_{channel_address}_{id(callback)}"
        self._subscriptions[sub_id] = (channel_address, callback)
        self._last_held[channel_address] = self._held_value(channel_address)
        logger.debug(f"In-process subscription created: {sub_id}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from channel changes."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.debug(f"In-process subscription removed: {subscription_id}")

    def _subscribed_held(self) -> dict[str, Any]:
        addresses = sorted({address for address, _ in self._subscriptions.values()})
        if not addresses or self._composite is None:
            return {}
        return {address: self._held_value(address) for address in addresses}

    async def _notify(self, before: Mapping[str, Any] | None = None) -> None:
        """Fire each subscription whose channel's held value changed or that moves."""
        if not self._subscriptions or self._composite is None:
            return
        previous = self._last_held if before is None else before
        now = self._subscribed_held()
        due = {
            address
            for address, value in now.items()
            if address in self._moving or not values_match(previous.get(address), value)
        }
        self._last_held.update(now)
        for address, callback in list(self._subscriptions.values()):
            if address not in due:
                continue
            try:
                result = callback(self._reading(address, held=False))
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                logger.warning(f"In-process subscription callback for {address} raised: {exc}")

    async def _tick(self) -> None:
        """Every ``tick_s``: fire the subscriptions that are due."""
        while True:
            await asyncio.sleep(self._tick_s)
            try:
                self._sync()
                await self._notify()
            except Exception as exc:
                logger.warning(f"In-process tick failed: {exc}")
