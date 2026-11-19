"""The simulator, served in this process: the built facility's simulator, with no network.

The connector serves exactly the channel and status addresses of the simulator view
``osprey build`` writes at ``data/simulator/`` beside the rendered config,
through the composite (:class:`~osprey_connectors.simulation.composite.Composite`):
the texture and one physics child per served model whose engine is not
``texture``. An address outside the view is refused with
``<address> is not a channel of this facility``.

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
  the channel moves under the active scenarios.

**Session writes.** Only a write's send (:meth:`VAInProcessConnector._send`) journals:
after the limits validator and a successful put it appends ``[seq, address, value]``
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
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeAlias

from pydantic import ConfigDict, field_validator
from typing_extensions import override

from osprey_connectors._contract_types import ContractAny
from osprey_connectors.control_system._base import (
    Alarm,
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    ControlSystemSettings,
    SendReport,
    _settings_as,
    values_match,
)
from osprey_connectors.errors import NOT_A_CHANNEL, ChannelNotFoundError, ConnectorUnavailableError
from osprey_connectors.logger import get_logger
from osprey_connectors.model import FrozenModel
from osprey_connectors.refusals import RefusalReason
from osprey_connectors.safety_file import atomic_replace
from osprey_connectors.simulation.records import VariablesDocument
from osprey_connectors.simulation.values import LABELLED_TYPES
from osprey_connectors.simulation.view import (
    SETPOINT,
    VIEW_RELPATH,
    NoSimulatorViewError,
    SimulatorView,
    ViewSchemaError,
)

if TYPE_CHECKING:
    from osprey_connectors.control_system._limits_validator import CurrentValueReader
    from osprey_connectors.simulation.composite import Composite
    from osprey_connectors.simulation.records import ChannelRecord

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

#: Why an operation on a connector that is not connected is refused: the one
#: case where the in-process simulator's connector itself is gone.
NOT_CONNECTED = "the in-process simulator is not connected"


def _loaded_config_path() -> str | None:
    """The config this process's unqualified lookups read, loading it when none is."""
    from osprey_connectors.config import active_config, current_config
    from osprey_connectors.errors import ConfigLoadError, ConfigurationError

    loaded = active_config()
    if loaded is not None:
        return str(loaded.path)
    try:
        return str(current_config().path)
    except (ConfigLoadError, ConfigurationError):
        return None


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
    except NoSimulatorViewError:
        raise RuntimeError(NO_VIEW_MESSAGE) from None
    except ViewSchemaError as error:
        raise RuntimeError(str(error)) from None


def _rendered_config(view: Path) -> tuple[Path, Mapping[str, object]]:
    """The rendered config beside a view, and its contents; empty when there is none."""
    from osprey_connectors.config import load_config

    path = view.parent.parent / _RENDERED_CONFIG
    if not path.is_file():
        return path, {}
    try:
        return path, load_config(path).raw_tree()
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


#: Read back only by the in-process connector that wrote it, in the same process.
_JOURNAL_FILE_MODE = 0o600


class _JournalDocument(FrozenModel):
    """A session-writes journal, as the file holds it.

    A document with an integer ``seq`` and a list of ``writes`` is a journal,
    whatever else it carries: a key the reader does not use is ignored, as it
    always has been. Each write is checked on its own when the journal is
    replayed (:func:`_journal_entry`), so one malformed write rejects the
    journal rather than making it no journal at all. An absent
    ``active_set_sha256`` matches no active set, as ``None`` does.
    """

    model_config = ConfigDict(extra="ignore")

    active_set_sha256: ContractAny = None
    seq: int
    writes: list[ContractAny]


#: One journalled write, ``[seq, address, value]``.
_JournalEntry: TypeAlias = tuple[int, str, object]


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

    def read(self) -> tuple[_JournalDocument | None, str]:
        """The journal and its text; ``None`` for a file that is not a journal.

        ``(None, "")`` when there is no file.
        """
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None, ""
        try:
            return _JournalDocument.model_validate(json.loads(text)), text
        except ValueError:
            # Not JSON (an over-long integer included), or JSON that is not a
            # journal: a ValidationError is a ValueError.
            return None, text

    def stamp(self) -> tuple[int, int, int] | None:
        """The file's (inode, mtime_ns, size), ``None`` when there is none."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_ino, stat.st_mtime_ns, stat.st_size

    def write(self, document: Mapping[str, object]) -> None:
        atomic_replace(
            self.path,
            json.dumps(document, sort_keys=True).encode("utf-8"),
            mode=_JOURNAL_FILE_MODE,
            sync="best_effort",
        )


def _journal_entry(entry: object) -> _JournalEntry | None:
    """One journalled ``[seq, address, value]`` write, or ``None`` when it is malformed."""
    if not isinstance(entry, list) or len(entry) != 3:
        return None
    seq_no, address, value = entry
    if not isinstance(seq_no, int) or isinstance(seq_no, bool) or not isinstance(address, str):
        return None
    return seq_no, address, value


def _mark(document: _JournalDocument | None, text: str) -> int | str:
    """What one look at the journal saw: its ``seq``, else its text, ``0`` when empty."""
    return (text or 0) if document is None else document.seq


class VAInProcessSettings(ControlSystemSettings):
    """The ``virtual_accelerator`` block, as the simulator in process reads it."""

    #: Simulated response delay, in milliseconds.
    response_delay_ms: float = 10
    #: The view directory to serve; ``None`` serves ``data/simulator/`` beside
    #: the loaded config.
    simulator_view: Path | None = None

    @field_validator(SIMULATOR_VIEW_SETTING, mode="before")
    @classmethod
    def _view_path(cls, value: object) -> object:
        if not value:
            return None
        return Path(value) if isinstance(value, str | PathLike) else value


class VAInProcessConnector(ControlSystemConnector):
    """Serve the built facility's simulator view in process.

    Example:
        >>> connector = VAInProcessConnector()
        >>> await connector.connect(ConnectorContext(settings={"response_delay_ms": 10}))
        >>> value = await connector.read_channel("SR:DIAG:BPM:01:POSITION:X")
    """

    settings_model: ClassVar[type[ControlSystemSettings]] = VAInProcessSettings

    def __init__(self) -> None:
        self._connected = False
        self._composite: Composite | None = None
        self._records: dict[str, ChannelRecord] = {}
        self._served: frozenset[str] = frozenset()
        self._subscriptions: dict[str, tuple[str, Callable[[ChannelValue], object]]] = {}
        self._last_held: dict[str, object] = {}
        self._tick_task: asyncio.Task[None] | None = None
        self._tick_s = 1.0
        self._channels: frozenset[str] = frozenset()
        self._journal: _Journal | None = None
        self._active_sha: str | None = None
        self._journal_mark: int | str | None = None
        # The stamp of the journal _journal_mark was read from; None when no read gave the mark.
        self._journal_seen: tuple[int, int, int] | None = None
        self._state_file: Path | None = None
        self._state_mark: int | None = None
        self._holds_writes = False

    @override
    async def _open(self, settings: ControlSystemSettings) -> None:
        """Build the composite of the simulator view and start the tick.

        Args:
            settings: Connector settings:
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

        read = _settings_as(settings, VAInProcessSettings)
        self._response_delay = read.response_delay_ms / 1000.0

        view = simulator_view_dir(read.simulator_view)
        _config_path, rendered = _rendered_config(view.path)
        self._tick_s = resolve_tick_s(rendered)
        state_dir = simulation_state_dir(view.path)
        try:
            self._records = {
                entry.address: entry for entry in view.document(VariablesDocument).channels
            }
            self._channels = frozenset(view.channels())
            self._served = (
                self._channels
                | frozenset(view.status_addresses().values())
                | {
                    address
                    for signals in view.ioc_health_by_model().values()
                    for address in signals.values()
                }
            )
            self._composite = Composite(view, state_dir=state_dir, instance="inprocess")
        except ViewSchemaError as error:
            raise RuntimeError(str(error)) from None
        self._journal = _Journal(state_dir)
        from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME

        self._state_file = state_dir / ACTIVE_SCENARIOS_FILENAME
        self._state_mark = None
        self._active_sha = None
        self._journal_mark = None
        self._journal_seen = None
        self._holds_writes = False
        self._sync()
        self._connected = True
        self._tick_task = asyncio.create_task(self._tick())
        logger.debug(f"In-process simulator serving {view.path}")

    @override
    async def disconnect(self) -> None:
        """Stop the tick and drop the composite."""
        if self._tick_task is not None:
            self._tick_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._tick_task
            self._tick_task = None
        self._subscriptions.clear()
        self._last_held.clear()
        self._composite = None
        self._records = {}
        self._served = frozenset()
        self._channels = frozenset()
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
                self._journal_seen = None
                if self._truncate(unless_current=True):
                    return
        if rebuilt:
            # The composite rebuilt from a rewrite of the same set and dropped
            # the session writes; the journal still holds them.
            self._holds_writes = False
            self._journal_mark = None
            self._journal_seen = None
        stamp = self._journal.stamp()
        # Every write replaces the journal file, so an unchanged stamp is an
        # unchanged journal.
        if stamp is not None and self._journal_mark is not None and stamp == self._journal_seen:
            return
        document, text = self._journal.read()
        mark = _mark(document, text)
        self._journal_seen = stamp
        if mark == self._journal_mark:
            return
        self._journal_mark = mark
        self._replay(document, text)

    def _state_stamp(self) -> int | None:
        """The ``active_scenarios`` file's mtime, ``None`` when there is none."""
        if self._state_file is None:
            return None
        try:
            return self._state_file.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _replay(self, document: _JournalDocument | None, text: str) -> None:
        """Reset the composite and apply the journal as one write, when it is well formed."""
        assert self._composite is not None
        if self._holds_writes:
            self._composite.reset()
            self._holds_writes = False
        if document is None:
            if text:
                self._reject(JOURNAL_FILE, "not a journal")
            return
        writes = document.writes
        if not writes:
            return
        rejection = self._rejection(document)
        if rejection is not None:
            self._reject(*rejection)
            return
        # Every entry is well formed: _rejection found none that is not.
        entries = [entry for entry in map(_journal_entry, writes) if entry is not None]
        replayed: dict[str, object] = {
            address: value
            for _seq_no, address, value in sorted(entries, key=lambda entry: entry[0])
        }
        try:
            self._composite.set(replayed)
        except Exception as exc:
            self._composite.reset()
            self._reject(getattr(exc, "address", None) or JOURNAL_FILE, str(exc))
            return
        self._holds_writes = True

    def _reject(self, address: str, reason: str) -> None:
        """Log a journal that is not replayed, in the process log and every physics
        model's, then empty it unless another connector moved it since."""
        logger.warning(f"{JOURNAL_REJECTED}: {address}: {reason}")
        if self._composite is not None:
            self._composite.log_event(JOURNAL_REJECTED_EVENT, address=address, reason=reason)
        self._truncate(unless_moved=True)

    def _rejection(self, document: _JournalDocument) -> tuple[str, str] | None:
        """The first address and reason that keep a journal from being replayed."""
        entries: list[_JournalEntry] = []
        for entry in document.writes:
            parsed = _journal_entry(entry)
            if parsed is None:
                return JOURNAL_FILE, f"malformed entry {entry!r}"
            entries.append(parsed)
        if document.active_set_sha256 != self._active_sha:
            return entries[0][1], "written under another active scenario set"
        for _seq_no, address, value in entries:
            record = self._records.get(address)
            if record is None or address not in self._channels:
                return address, NOT_A_CHANNEL
            if record.role != SETPOINT:
                return address, "not a setpoint"
            if not record.writable:
                return address, NOT_WRITABLE
            band = record.value_range
            if band:
                low, high = band
                number = value if isinstance(value, int | float) else None
                if (
                    number is None
                    or isinstance(number, bool)
                    or math.isnan(number)
                    or number < low
                    or number > high
                ):
                    return address, f"{value!r} is outside [{low}, {high}]"
        return None

    def _append(self, address: str, value: object) -> None:
        """Append one write the composite took, under the journal's lock."""
        assert self._journal is not None
        with self._journal.locked():
            document, _text = self._journal.read()
            seq = None if document is None else document.seq
            synced = seq == self._journal_mark or (seq is None and self._journal_mark == 0)
            writes: list[object] = []
            if document is not None and document.active_set_sha256 == self._active_sha:
                writes = list(document.writes)
            seq = (seq or 0) + 1
            writes.append([seq, address, value])
            self._journal.write(
                {"active_set_sha256": self._active_sha, "seq": seq, "writes": writes}
            )
        self._holds_writes = True
        # A journal another process moved since the last look is replayed on
        # the next operation, this write with it.
        self._journal_mark = seq if synced else None
        self._journal_seen = None

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
                and document is not None
                and document.active_set_sha256 == self._active_sha
            ):
                return False
            if unless_moved and _mark(document, text) != self._journal_mark:
                return False
            seq = (0 if document is None else document.seq) + 1
            self._journal.write({"active_set_sha256": self._active_sha, "seq": seq, "writes": []})
        self._journal_mark = seq
        self._journal_seen = None
        return True

    async def reset(self) -> None:
        """Return the simulator to its start state and empty the journal.

        Raises:
            ConnectorUnavailableError: The connector is not connected.
        """
        if self._composite is None:
            raise ConnectorUnavailableError(NOT_CONNECTED)
        self._sync()
        self._composite.reset()
        self._holds_writes = False
        self._truncate()

    # -- the view ------------------------------------------------------------

    def _require(self, address: str) -> Composite:
        """The composite, once ``address`` is one it serves.

        Raises:
            ChannelNotFoundError: ``address`` is not a channel of the facility.
            ConnectorUnavailableError: The connector is not connected.
        """
        if self._composite is None:
            raise ConnectorUnavailableError(NOT_CONNECTED)
        if address not in self._served:
            raise ChannelNotFoundError(address)
        return self._composite

    def _wire(self, address: str, value: object) -> object:
        """A value as the wire carries it: a label as its index, a waveform as an array."""
        record = self._records.get(address)
        if record is None:
            return value
        value_type = record.value_type
        if value_type in LABELLED_TYPES:
            options = record.labels
            return options.index(value) if isinstance(value, str) and value in options else value
        if value_type == "waveform":
            import numpy as np

            shape = record.shape or ()
            return np.asarray(value, dtype=np.float64).reshape(shape or (-1,))
        return value

    def _metadata(self, address: str, composite: Composite) -> ChannelMetadata:
        """Metadata from the channel record."""
        record = self._records.get(address)
        variable = composite.supported_variables[address]
        unit = getattr(variable, "unit", None) or (record and record.unit) or ""
        labels: tuple[str, ...] | None = None
        if record is not None and record.value_type in LABELLED_TYPES:
            labels = record.labels
        return ChannelMetadata(
            units=str(unit),
            description=None if record is None else record.description,
            enum_labels=labels,
        )

    def _reading(self, address: str, *, held: bool) -> ChannelValue:
        """One reading, its alarm from the composite's severity: a failed model reads ``UDF``."""
        composite = self._require(address)
        stored: object = composite.held([address])[address] if held else composite.get(address)
        metadata = self._metadata(address, composite)
        failed = address in composite.output_severity([address])
        labelled = metadata.enum_labels is not None and stored in metadata.enum_labels
        return ChannelValue(
            address=address,
            value=self._wire(address, stored),
            sampled_at=datetime.now(UTC),
            alarm=Alarm(status=UDF_STATUS, severity=UDF_SEVERITY) if failed else None,
            enum_label=str(stored) if labelled else None,
            metadata=metadata,
        )

    def _held_value(self, address: str) -> object:
        composite = self._require(address)
        return self._wire(address, composite.held([address])[address])

    # -- reads -----------------------------------------------------------------

    @override
    async def read_channel(
        self,
        address: str,
        *,
        timeout_s: float
        | None = None,  # ControlSystemConnector.read_channel signature; an in-process read never blocks
    ) -> ChannelValue:
        """Read a channel as the simulator serves it, motion included.

        Raises:
            ChannelNotFoundError: The address is not a channel of the facility.
        """
        await asyncio.sleep(self._response_delay)
        self._sync()
        return self._reading(address, held=False)

    @override
    async def _confirming_read(
        self,
        address: str,
        sent: SendReport,
        *,
        timeout_s: float,
    ) -> ChannelValue:
        """Read a channel back to confirm a write: the held value, without motion.

        Confirmation reports what the simulated control system holds; the
        motion ``read_channel`` adds models measuring a live signal, so adding
        it here would manufacture a mismatch on every write.
        """
        await asyncio.sleep(self._response_delay)
        self._sync()
        return self._reading(address, held=True)

    @override
    def current_value_reader(self) -> CurrentValueReader:
        """What the simulated control system holds, read without motion."""
        return self._held_value

    @override
    async def get_metadata(self, address: str) -> ChannelMetadata:
        """The channel record's description and its variable's unit.

        Raises:
            ChannelNotFoundError: The address is not a channel of the facility.
        """
        composite = self._require(address)
        self._sync()
        return self._metadata(address, composite)

    @override
    async def validate_channel(self, address: str) -> bool:
        """Whether the built facility file holds the address."""
        return address in self._served

    # -- writes ----------------------------------------------------------------

    def _refused(self, address: str, value: object, message: str) -> ChannelWriteResult:
        logger.warning(f"In-process write refused: {address}: {message}")
        return ChannelWriteResult.refused(
            address, value, RefusalReason.CONTROL_SYSTEM_REFUSED, message
        )

    def _coerce(self, address: str, value: object) -> object:
        """``value`` in the stored representation of the channel's ``value_type``.

        Raises:
            ValueError: The channel cannot hold the value.
        """
        from osprey_connectors.simulation import values

        record = self._records[address]
        if hasattr(value, "tolist"):
            value = value.tolist()
        return values.coerce(value, record.value_type, record.options, record.shape)

    @override
    def _precheck(self, address: str, value: object) -> tuple[str, ChannelWriteResult | None]:
        """Refuse an address outside the view and a channel that is no writable setpoint."""
        if self._composite is None:
            raise ConnectorUnavailableError(NOT_CONNECTED)
        if address not in self._served:
            unknown = ChannelNotFoundError(address)
            logger.warning(f"In-process write refused: {address}: {unknown}")
            return address, ChannelWriteResult.refused(
                address, value, RefusalReason.CHANNEL_NOT_FOUND, str(unknown)
            )
        self._sync()
        if self._composite.supported_variables[address].read_only:
            return address, self._refused(
                address, value, f"Write to '{address}' refused: {NOT_WRITABLE}"
            )
        return address, None

    @override
    async def _current_values(self, addresses: list[str]) -> dict[str, object] | None:
        """What the simulated control system holds: read in process, no thread."""
        held: dict[str, object] = {}
        for address in addresses:
            try:
                held[address] = self._held_value(address)
            except Exception as exc:
                held[address] = exc
        return held

    @override
    async def _run_check(self, check: Callable[[], None], *, timeout_s: float) -> None:
        """Run a write's limits check inline: an in-process read never blocks."""
        check()

    @override
    async def _send(
        self,
        address: str,
        value: ContractAny,
        *,
        timeout_s: float,
        wait: bool,
    ) -> SendReport:
        """Put the value into the composite, journal it and fire the subscriptions.

        A value the channel's ``value_type`` cannot hold, or a put the
        composite raises on, is a write the simulator did not take; a put the
        composite refuses (``ValueError``, ``ReadOnlyError``) is refused with
        the composite's text.
        """
        await asyncio.sleep(self._response_delay)

        try:
            stored = self._coerce(address, value)
        except Exception as e:
            logger.warning(f"In-process write failed for {address}: {e}")
            return SendReport(fault="not_taken", detail=f"the simulator did not take it: {e}")

        before = self._subscribed_held()
        try:
            self._put(address, stored)
        except Exception as e:
            from lume.exceptions import ReadOnlyError

            if isinstance(e, ValueError | ReadOnlyError):
                return SendReport(fault="cs_refused", detail=str(e) or "the simulator refused it")
            logger.warning(f"In-process write failed for {address}: {e}")
            return SendReport(fault="not_taken", detail=f"the simulator did not take it: {e}")
        try:
            self._append(address, stored)
        except OSError as exc:
            # The composite holds the write; only its survival across
            # connectors is lost.
            self._holds_writes = True
            logger.warning(f"{JOURNAL_NOT_UPDATED}: {exc}")
        await self._notify(before)
        return SendReport()

    def _put(self, address: str, value: object) -> None:
        """Set ``value`` in the composite; raises whatever the composite raises."""
        self._require(address).set({address: value})

    # -- subscriptions and the tick ------------------------------------------

    @override
    async def subscribe(
        self,
        address: str,
        callback: Callable[[ChannelValue], None],
        *,
        on_connection: Callable[[bool], None] | None = None,
    ) -> str:
        """Subscribe to a channel; the callback fires after a tick or a write.

        A channel of the facility is served by this process and never drops, so
        ``on_connection(True)`` comes once, before this returns, and ``False``
        never does.

        Raises:
            ChannelNotFoundError: The address is not a channel of the facility.
        """
        self._require(address)
        self._sync()
        sub_id = f"inprocess_{address}_{id(callback)}"
        self._subscriptions[sub_id] = (address, callback)
        self._last_held[address] = self._held_value(address)
        logger.debug(f"In-process subscription created: {sub_id}")
        if on_connection is not None:
            try:
                on_connection(True)
            except Exception as exc:
                logger.warning(f"In-process connection callback for {address} raised: {exc}")
        return sub_id

    @override
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from channel changes."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.debug(f"In-process subscription removed: {subscription_id}")

    def _subscribed_held(self) -> dict[str, object]:
        addresses = sorted({address for address, _ in self._subscriptions.values()})
        if not addresses or self._composite is None:
            return {}
        return {address: self._held_value(address) for address in addresses}

    async def _notify(self, before: Mapping[str, object] | None = None) -> None:
        """Fire each subscription whose held value changed or whose channel moves under the active scenarios."""
        if not self._subscriptions or self._composite is None:
            return
        previous = self._last_held if before is None else before
        now = self._subscribed_held()
        moving = self._composite.moving()
        due = {
            address
            for address, value in now.items()
            if address in moving or not values_match(previous.get(address), value)
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
