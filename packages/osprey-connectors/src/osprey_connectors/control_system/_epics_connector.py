"""
EPICS control system connector: Channel Access through aioca, PVAccess through p4p.

Every Channel Access call runs on the process's one CA event loop
(:mod:`osprey_connectors.control_system._ca_loop`), whichever loop or thread
the connector was called from.
"""

import asyncio
import concurrent.futures
import contextlib
import contextvars
import fnmatch
import operator
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from osprey_connectors.call_timeout import call_timeout_s
from osprey_connectors.config import config_flag, get_facility_timezone
from osprey_connectors.control_system._base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
    _verdict_refusal,
    values_match,
)
from osprey_connectors.control_system._ca_loop import CaLoop, current_ca_env, get_ca_loop
from osprey_connectors.control_system._epics_bulk import CABulk, normalise_address
from osprey_connectors.control_system._limits_validator import (
    DEFAULT_STEP_READ_TIMEOUT_SECONDS,
    c_numeral,
    step_read_timeout_seconds,
)
from osprey_connectors.errors import (
    ChannelUnreachableError,
    ConnectorInternalError,
    ConnectorUnavailableError,
    RefusalReason,
)
from osprey_connectors.logger import get_logger
from osprey_connectors.posture_env import is_readonly_run
from osprey_connectors.types import WRITES_KEY, machine_kind

logger = get_logger("epics_connector")

#: Seconds between two stale-transport probes of :meth:`EPICSConnector.wait_stale`.
STALE_PROBE_INTERVAL_S = 1.0
#: A subscribed channel counts as held down once it has been down this long.
STALE_HOLD_DOWN_S = 3.0
#: A one-shot read that failed to connect keeps its channel held down this long.
STALE_ONE_SHOT_WINDOW_S = 10.0
#: A held-down name still down this long after a channel connected since it went down is stale.
STALE_CONFIRM_S = 0.5
#: A held-down name down this long with no channel connecting since is stale too.
STALE_SILENT_S = 30.0
#: Name prefix of the threads that run a write's limits validation (see ``_WriteExecutor``).
WRITE_THREAD_PREFIX = "epics-write"
#: Most write-validation threads alive at once (see ``_WriteExecutor``).
WRITE_THREAD_CAP = 64


class _WriteExecutor(concurrent.futures.Executor):
    """An executor that starts one thread per submitted call, so nothing ever queues.

    A write's put and confirming read run on the CA loop and hold no thread.
    Its limits validation does: the validator reads the channel's current value
    for a ``max_step`` check synchronously, within S. That bound only holds from
    the moment a thread starts the validation, and on the loop's default
    executor it would queue behind PVA reads, which share that pool and can fill
    it. Starting a thread per call removes the queue: every thread ends within
    S, so the number alive is at most the number of writes in flight, which the
    session already limits to one per address.

    ``max_threads`` caps that number on its own: a call submitted while that
    many threads are alive raises
    :class:`~osprey_connectors.errors.ConnectorInternalError` and does not run,
    so a write refused this way sends nothing. A full executor is this
    connector's own machinery saturated, not a reachability answer, so no
    supervisor replaces the connector for it.
    """

    def __init__(
        self, thread_name_prefix: str = WRITE_THREAD_PREFIX, max_threads: int = WRITE_THREAD_CAP
    ) -> None:
        self._prefix = thread_name_prefix
        self._max_threads = max_threads
        self._lock = threading.Lock()
        self._threads: set[threading.Thread] = set()
        self._serial = 0

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        future: concurrent.futures.Future = concurrent.futures.Future()

        def run() -> None:
            try:
                if not future.set_running_or_notify_cancel():
                    return
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:  # handed to the awaiting caller
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            finally:
                with self._lock:
                    self._threads.discard(threading.current_thread())

        with self._lock:
            if len(self._threads) >= self._max_threads:
                raise ConnectorInternalError(
                    f"{len(self._threads)} write threads are already running "
                    f"(the cap is {self._max_threads}); this write was not sent"
                )
            # Nothing queues here, so a call that races shutdown still runs.
            self._serial += 1
            thread = threading.Thread(target=run, name=f"{self._prefix}_{self._serial}")
            self._threads.add(thread)
        thread.start()
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        del cancel_futures  # a started phase is never cancelled; nothing queues
        with self._lock:
            threads = list(self._threads)
        if wait:
            for thread in threads:
                thread.join()


# Channel Access status codes the connector branches on (EPICS base caerr.h).
# epicscorelibs' cadef names only some of them, so they are spelled out here.
_ECA_TIMEOUT = 80  # "User specified timeout on IO operation expired"
_ECA_NOWTACCESS = 376  # "Write access denied"
_ECA_PUTFAIL = 160  # "Channel write request failed": the IOC did not take the put

# EPICS alarm status names, indexed by the code Channel Access reports
# (EPICS base alarm.h; the same table pyepics' AlarmStatus carried).
_ALARM_NAMES = (
    "NO_ALARM",
    "READ",
    "WRITE",
    "HIHI",
    "HIGH",
    "LOLO",
    "LOW",
    "STATE",
    "COS",
    "COMM",
    "TIMEOUT",
    "HW_LIMIT",
    "CALC",
    "SCAN",
    "LINK",
    "SOFT",
    "BAD_SUB",
    "UDF",
    "DISABLE",
    "SIMM",
    "READ_ACCESS",
    "WRITE_ACCESS",
)

# Channel Access field types, indexed by DBF code (db_access.h).
_DBF_NAMES = ("string", "short", "float", "enum", "char", "long", "double")
_DBF_SHORT, _DBF_FLOAT, _DBF_ENUM, _DBF_CHAR, _DBF_LONG, _DBF_DOUBLE = 1, 2, 3, 4, 5, 6
_DBF_INTEGER = (_DBF_SHORT, _DBF_CHAR, _DBF_LONG)
_DBF_REAL = (_DBF_FLOAT, _DBF_DOUBLE)

#: The range each whole-number field type's wire form holds: DBR_SHORT is
#: int16, DBR_ENUM uint16, DBR_CHAR uint8 and DBR_LONG int32. A number sent to
#: one of these fields is converted by the IOC without a check, so 1.5 reaches
#: a ``longout`` as 1 and 2**31 as -2147483648.
_DBF_WHOLE_RANGES = {
    _DBF_SHORT: (-(2**15), 2**15 - 1),
    _DBF_ENUM: (0, 2**16 - 1),
    _DBF_CHAR: (0, 2**8 - 1),
    _DBF_LONG: (-(2**31), 2**31 - 1),
}


def _alarm_name(code: Any) -> str:
    """Map a Channel Access alarm status code onto its EPICS name.

    Channel Access reports the alarm status as a small integer; the rest of
    OSPREY (and ``ChannelMetadata.alarm_status``) speaks names, which is what
    PVAccess already emits. This is the one place codes become names, so the
    shared connector core stays free of EPICS constants.

    Anything the table cannot map — an out-of-range code, ``None`` from a
    channel that has not reported yet — becomes ``"UNKNOWN"``.
    """
    if isinstance(code, bool):
        return "UNKNOWN"
    try:
        index = int(code)
    except (TypeError, ValueError):
        return "UNKNOWN"
    return _ALARM_NAMES[index] if 0 <= index < len(_ALARM_NAMES) else "UNKNOWN"


def _ca_type_name(datatype: Any) -> str | None:
    """The field type of a reading, spelled ``time_<type>`` as pyepics spelled it.

    Readings are requested in the ``TIME`` form, and ``raw_metadata["type"]``
    has always carried that spelling, so it is kept.
    """
    if isinstance(datatype, int) and 0 <= datatype < len(_DBF_NAMES):
        return f"time_{_DBF_NAMES[datatype]}"
    return None


def _is_enum_type(pv_type: Any) -> bool:
    """True when a Channel Access field type names an enumeration.

    The type of an ``mbbi``/``bi``/``bo`` is spelled ``"enum"`` or, in the
    ``time`` form readings are requested in, ``"time_enum"`` — hence the
    substring test rather than an equality check against one spelling.
    """
    return "enum" in str(pv_type)


def _ca_plain(value: Any) -> Any:
    """An aioca reading as a plain Python value.

    aioca hands back subclasses of ``str``/``int``/``float`` and of
    ``numpy.ndarray`` that carry the reading's metadata as attributes. Those
    attributes are copied into :class:`ChannelMetadata`; the value itself
    leaves the connector as the plain type, so nothing downstream sees or
    serializes aioca's classes.
    """
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "__array__"):
        import numpy

        return numpy.asarray(value)
    return value


class _Unwritable(ValueError):
    """A value the channel's field type cannot hold exactly; nothing was sent."""


def _whole_number(element: Any, field_type: int) -> int:
    """``element`` as the integer a whole-number field holds, exactly.

    Text is read by the C rule an integer field reads it with
    (:func:`c_numeral`); a number must already be whole. Either way the result
    must fit the field's wire type. Raises :class:`_Unwritable` otherwise.
    """
    if isinstance(element, str):
        number = c_numeral(element, integer_field=True)
        if number is None:
            raise _Unwritable(f"{element!r} is not a numeral an integer field reads")
    elif isinstance(element, bytes | bytearray):
        raise _Unwritable(f"{element!r} is not a whole number, and the channel holds integers")
    else:
        try:
            number = operator.index(element)
        except TypeError:
            try:
                real = float(element)
            except (TypeError, ValueError):
                real = None
            if real is None or not real.is_integer():
                raise _Unwritable(
                    f"{element!r} is not a whole number, and the channel holds integers"
                ) from None
            number = int(real)
    low, high = _DBF_WHOLE_RANGES[field_type]
    if not low <= number <= high:
        raise _Unwritable(f"{element!r} is outside the channel's integer range [{low}, {high}]")
    return int(number)


def _whole_numbers(value: Any, field_type: int) -> Any:
    """``value`` as the whole number, or list of them, a whole-number field holds."""
    if isinstance(value, list | tuple):
        return [_whole_number(element, field_type) for element in value]
    if hasattr(value, "__array__"):
        import numpy

        elements = numpy.asarray(value).tolist()
        if isinstance(elements, list):
            return [_whole_number(element, field_type) for element in elements]
        return _whole_number(elements, field_type)
    return _whole_number(value, field_type)


def _ca_put_form(value: Any, field_type: Any, count: Any) -> tuple[Any, bool]:
    """``value`` as it is put to a field of ``field_type``, and whether it goes as text.

    Left to itself, aioca sends any ``str`` as ``DBR_STRING`` and the IOC
    converts it, stopping at the first character it cannot read: ``"1e3"``
    lands in an integer field as 1 and ``"12.7"`` as 12. So text bound for a
    numeric field is read here, by the rule for that field (:func:`c_numeral`),
    and sent as the number it spells; text the rule cannot read is refused.
    Text bound for a char array is sent as its characters with a terminating
    NUL, which is how a char waveform holds a string. String and enum fields
    take the text as it is: the IOC reads an enum's label itself.

    A number bound for a short, char, long or enum field, or each element of
    an array bound for one, is sent only when the field holds it exactly: a
    whole number within the field's wire range. A fraction, a NaN or a number
    out of range is refused rather than cut down on its way in.

    Raises :class:`_Unwritable` when the value cannot be written exactly.
    """
    if isinstance(value, str):
        if field_type == _DBF_CHAR and isinstance(count, int) and count > 1:
            encoded = value.encode()
            if len(encoded) >= count:
                raise _Unwritable(
                    f"text of {len(encoded)} bytes does not fit a {count}-element char array "
                    "with its terminating NUL"
                )
            return value + "\0", True
        if field_type in _DBF_INTEGER:
            return _whole_number(value, field_type), False
        if field_type in _DBF_REAL:
            number = c_numeral(value, integer_field=False)
            if number is None:
                raise _Unwritable(f"{value!r} is not a numeral a floating-point field reads")
            return number, False
        return value, False
    if field_type in _DBF_WHOLE_RANGES:
        return _whole_numbers(value, field_type), False
    return value, False


def _char_text(value: Any) -> Any:
    """A char array read back as the text it holds, up to its first NUL."""
    if hasattr(value, "__array__") and not isinstance(value, str):
        import numpy

        raw = numpy.asarray(value).astype(numpy.uint8).tobytes()
        return raw.split(b"\0", 1)[0].decode(errors="replace")
    return value


def _ca_failure(channel_address: str, failure: Any, action: str) -> Exception:
    """The exception a failed aioca call is raised as: the channel did not answer.

    A Channel Access failure is about the one channel: the CA context belongs
    to the process and keeps serving every other channel. A channel that timed
    out and one that was lost are both :class:`ChannelUnreachableError`, and the
    text says which.
    """
    if getattr(failure, "errorcode", None) == _ECA_TIMEOUT:
        return ChannelUnreachableError(
            channel_address, f"{action} '{channel_address}' timed out: {failure}"
        )
    return ChannelUnreachableError(
        channel_address, f"{action} '{channel_address}' failed: {failure}"
    )


def _enum_label_fields(labels: Any, index: Any) -> tuple[list[str] | None, str | None]:
    """Normalize a raw label list and resolve the label for ``index``.

    Every branch fails soft: a control system that reports no labels, reports
    them as something other than a sequence, or answers with an index outside
    the list yields ``None`` rather than raising. A reading must never be lost
    because its labels could not be resolved — the index alone is still a
    correct, complete answer.
    """
    if labels is None or isinstance(labels, (str, bytes)):
        return None, None
    try:
        normalized = [str(label) for label in labels]
    except TypeError:  # not iterable — nothing usable was reported
        return None, None
    if not normalized:
        return None, None

    label: str | None = None
    if isinstance(index, bool):
        # bool is an int subclass, and a bi record's 0/1 is meaningful; keep it.
        index = int(index)
    if isinstance(index, int) and 0 <= index < len(normalized):
        label = normalized[index]
    return normalized, label


def _severity_int(severity: Any) -> int | None:
    """Coerce a reported alarm severity to ``int``, or ``None`` if unreported.

    ``None`` means "the protocol carried no severity" and is deliberately
    distinct from a reported healthy ``0``, so a consumer can tell "no alarm"
    from "no information".
    """
    if severity is None:
        return None
    try:
        return int(severity)
    except (TypeError, ValueError):
        return None


def _readback_alarm_fields(readback: Any) -> tuple[str | None, int | None]:
    """Alarm name and severity of a readback, or ``(None, None)`` if unreported.

    ``None`` means "the readback carried no alarm metadata" and is deliberately
    distinct from a reported healthy readback (severity ``0``), so a consumer
    can tell "no alarm" from "no information".
    """
    metadata = getattr(readback, "metadata", None)
    if metadata is None:
        return None, None

    status = getattr(metadata, "alarm_status", None)
    raw = getattr(metadata, "raw_metadata", None) or {}
    severity = _severity_int(raw.get("severity") if isinstance(raw, dict) else None)

    return (str(status) if status is not None else None, severity)


#: ``ChannelMetadata`` limit field -> attribute of a CTRL-form aioca reading carrying it.
_CA_CTRL_LIMITS: tuple[tuple[str, str], ...] = (
    ("display_low", "lower_disp_limit"),
    ("display_high", "upper_disp_limit"),
    ("control_low", "lower_ctrl_limit"),
    ("control_high", "upper_ctrl_limit"),
    ("warning_low", "lower_warning_limit"),
    ("warning_high", "upper_warning_limit"),
    ("alarm_low", "lower_alarm_limit"),
    ("alarm_high", "upper_alarm_limit"),
)


def _ca_ctrl_limits(ctrl: Any, writable: bool | None) -> dict[str, Any]:
    """The CTRL limits of a channel and this client's write access, as metadata kwargs.

    Read off the channel's CTRL-form reading; an absent reading (its fetch
    failed) yields no limits. The values are handed over raw --
    :class:`ChannelMetadata` normalises them, which is what turns an unset
    ``0``/``0`` range into ``None``.
    """
    found: dict[str, Any] = {field: getattr(ctrl, attr, None) for field, attr in _CA_CTRL_LIMITS}
    found["writable"] = writable
    return found


# Normative-type identifiers the PVA read path maps specially. Compared with a
# prefix match so a minor revision ("epics:nt/NTEnum:1.1") keeps its mapping
# instead of silently degrading to the generic scalar path.
_NT_ENUM_ID = "epics:nt/NTEnum:"
_NT_NDARRAY_ID = "epics:nt/NTNDArray:"

# pvRequest used whenever only a channel's metadata is wanted. Asking for the
# whole structure would pull the payload too — on an NTNDArray that is a full
# camera frame per call, and a frame in a codec the connector cannot decode
# would make a metadata lookup fail on a channel that is perfectly reachable.
_PVA_METADATA_REQUEST = "field(alarm,timeStamp,display,control,valueAlarm)"


def _pva_type_id(value: Any) -> str:
    """Return the normative-type id of a p4p Value, or "" when it has none."""
    get_id = getattr(value, "getID", None)
    if not callable(get_id):
        return ""
    try:
        return str(get_id())
    except Exception:
        return ""


def _pva_field(container: Any, name: str, default: Any = None) -> Any:
    """Read one field from a p4p Value (or nested sub-Value), tolerating absence.

    Optional NT fields (``display``, ``precision``, ``attribute``) are present on
    some servers and missing on others, so every access on the read path goes
    through here rather than assuming a full structure.
    """
    if container is None:
        return default
    getter = getattr(container, "get", None)
    if callable(getter):
        try:
            found = getter(name, None)
        except Exception:
            found = None
        return default if found is None else found
    try:
        found = container[name]
    except Exception:
        return default
    return default if found is None else found


def _pva_color_mode(value: Any) -> Any:
    """Extract the areaDetector ColorMode NTAttribute, or None when absent."""
    for entry in _pva_field(value, "attribute", []) or []:
        if _pva_field(entry, "name", "") == "ColorMode":
            return _pva_field(entry, "value")
    return None


class _ChannelSubscription:
    """One active subscription plus the teardown its transport needs.

    Both transports store one of these in ``EPICSConnector._subscriptions``:
    Channel Access keeps a handle onto its address's record (``_epics_bulk``),
    PVAccess the p4p ``Subscription``; both are released with ``close()``.
    The record belongs to the CA loop, so the release is handed to that loop
    (``ca_loop``) rather than run on whichever thread unsubscribes. Dispatching
    here is what lets ``unsubscribe`` and ``disconnect`` walk subscription ids
    without knowing which protocol opened them.

    ``close()`` is idempotent and best effort: closing twice is a no-op, and a
    handle that fails to release is logged rather than raised, so tearing down
    a whole connector cannot be derailed by one dead channel.
    """

    __slots__ = ("_closed", "ca_loop", "handle", "kind")

    def __init__(self, kind: str, handle: Any, ca_loop: Any = None) -> None:
        self.kind = kind
        self.handle = handle
        self.ca_loop = ca_loop
        self._closed = False

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has run."""
        return self._closed

    def close(self) -> None:
        """Release the underlying handle. Safe to call any number of times."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.ca_loop is not None:
                self.ca_loop.call_soon(self._release)
            else:
                self.handle.close()
        except Exception as exc:  # teardown is best effort, never fatal
            logger.debug(f"Subscription teardown failed for {self.kind} handle: {exc}")

    def _release(self) -> None:
        """Close the handle; on the CA loop."""
        try:
            if self.handle is not None:
                self.handle.close()
        except Exception as exc:  # teardown is best effort, never fatal
            logger.debug(f"Subscription teardown failed for {self.kind} handle: {exc}")


class _BulkHandle:
    """A CA subscription's handle: closing it leaves its address's record (CA loop only)."""

    __slots__ = ("bulk", "sub_id")

    def __init__(self, bulk: CABulk, sub_id: str) -> None:
        self.bulk = bulk
        self.sub_id = sub_id

    def close(self) -> None:
        self.bulk.unsubscribe(self.sub_id)


class _PvaMetadataSlot:
    """The metadata one PVA subscription has learned from its monitor.

    The first value update after each (re)connection is the subscription's
    metadata source: it carries the same display / control / valueAlarm fields
    a metadata-only get asks for, so ``get_metadata`` of a subscribed address
    answers from it instead of opening a second request. A connection-down
    update empties the slot, so a lookup during an outage waits for the
    channel to come back rather than answering from a server that is gone.

    ``metadata`` is written on p4p's worker thread and read on the event loop,
    so it lives under ``lock``; ``waiters`` are futures of ``get_metadata``
    calls parked until the first update lands. They are settled on their own
    loop (``call_soon_threadsafe``), with the metadata, or with ``None`` when
    the subscription is closed underneath them.
    """

    __slots__ = ("address", "lock", "metadata", "waiters")

    def __init__(self, address: str) -> None:
        self.address = address
        self.lock = threading.Lock()
        self.metadata: ChannelMetadata | None = None
        self.waiters: list[asyncio.Future] = []

    def settle(self, waiters: list[asyncio.Future], result: ChannelMetadata | None) -> None:
        """Hand ``result`` to each waiter on the loop that owns it."""
        for waiter in waiters:
            # A waiter whose loop is closed has nobody waiting on it.
            with contextlib.suppress(RuntimeError):
                waiter.get_loop().call_soon_threadsafe(_settle_waiter, waiter, result)


def _settle_waiter(waiter: asyncio.Future, result: ChannelMetadata | None) -> None:
    """Resolve a parked ``get_metadata`` future unless it already finished."""
    if not waiter.done():
        waiter.set_result(result)


class EPICSConnector(ControlSystemConnector):
    """
    EPICS control system connector: Channel Access through aioca.

    Provides read/write access to EPICS Process Variables through the
    Channel Access protocol (and PVAccess for configured ``pva_channels``).
    Supports gateway configuration for remote access and
    read-only/write-access gateways.

    Example:
        Direct gateway connection:
        >>> config = {
        >>>     'timeout_s': 5.0,
        >>>     'gateways': {
        >>>         'read_only': {
        >>>             'address': 'gw.example.org',
        >>>             'port': 5064
        >>>         }
        >>>     }
        >>> }
        >>> connector = EPICSConnector()
        >>> await connector.connect(config)
        >>> value = await connector.read_channel('BEAM:CURRENT')
        >>> print(f"Beam current: {value.value} {value.metadata.units}")

        SSH tunnel connection:
        >>> config = {
        >>>     'timeout_s': 5.0,
        >>>     'gateways': {
        >>>         'read_only': {
        >>>             'address': 'localhost',
        >>>             'port': 5064,  # local end of the SSH tunnel
        >>>             'use_name_server': True
        >>>         }
        >>>     }
        >>> }
        >>> connector = EPICSConnector()
        >>> await connector.connect(config)
        >>> value = await connector.read_channel('BEAM:CURRENT')
        >>> print(f"Beam current: {value.value} {value.metadata.units}")
    """

    def __init__(self):
        self._connected = False
        # The `max_step` fresh-read budget, replaced from the connector's own
        # config block on connect. Set here so the reader is usable on a
        # connector that has not connected — the fallback is the same number
        # the block's default resolves to.
        self._step_read_timeout = DEFAULT_STEP_READ_TIMEOUT_SECONDS
        # The `fresh_reads` key, kept readable; every CA read goes to the wire.
        self._fresh_reads = False
        self._subscriptions: dict[str, Any] = {}
        # aioca, and the CA loop every call to it runs on; both set by connect().
        self._aioca: Any = None
        self._ca: CaLoop = None  # type: ignore[assignment]
        # Per channel, the CTRL-form reading that carries units, precision,
        # enum labels and limits for one-shot reads: fetched on a channel's
        # first read, then reused, since they describe the record rather than
        # the reading. Subscriptions keep their own, fresh per connection (see
        # _epics_bulk); the two can differ after a record's limits change: a
        # one-shot read describes the record as first seen, a subscription as
        # it is now. Touched only on the CA loop.
        self._ca_ctrl: dict[str, Any] = {}
        self._epics_configured = False
        # The gateway role connect() pointed Channel Access at, and whether the
        # block configures a write gateway at all. A write the verdict allows
        # is refused ``switching`` while a configured write gateway was not
        # selected (see :meth:`_role_refusal`); a connector that never
        # connected selected nothing and has no write gateway to lag behind.
        self._selected_role: str | None = None
        self._write_gateway_configured = False
        # PVA routing state. Empty globs => this connector is pure Channel
        # Access: p4p is never imported and no PVA environment variable is set.
        self._pva_channel_globs: list[str] = []
        self._p4p: Any = None
        self._pva_context: Any = None
        # Per PVA subscription id: the metadata its monitor delivered.
        self._pva_metadata_slots: dict[str, _PvaMetadataSlot] = {}
        # The Channel Access subscription path (_epics_bulk), built on first need.
        self._bulk: CABulk | None = None
        self._bulk_lock = threading.Lock()
        # Stale-transport bookkeeping (wait_stale). The name server this
        # connector was configured with, as ``host:port``; None for a
        # UDP-search connector, which never reports stale.
        self._name_server: str | None = None
        # One-shot connects, written on the CA loop: the monotonic time each
        # channel last failed to connect (cleared when it connects again), the
        # channels that have connected, and when a one-shot channel last
        # connected newly (its first connect, or its first after a failure).
        self._connect_failures: dict[str, float] = {}
        self._one_shot_connected: set[str] = set()
        self._one_shot_connect_at: float | None = None
        # The executor a write's limits validation runs on, built on first
        # write and shut down on disconnect; see _write_pool().
        self._write_executor: _WriteExecutor | None = None
        self._write_executor_lock = threading.Lock()

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Configure EPICS environment and test connection.

        Args:
            config: Configuration with keys:
                - timeout_s: Default timeout in seconds (default: 5.0)
                - fresh_reads: (optional, no effect) Kept so existing configs
                  load. Every Channel Access read now asks the IOC.
                - gateways: Gateway configuration dict with:
                    - read_only: {address, port, use_name_server} for read operations
                    - write_access: {address, port, use_name_server} for write operations

                Gateway sub-keys:
                    - address: Gateway hostname or IP
                    - port: Gateway port number
                    - use_name_server: (optional) Use EPICS_CA_NAME_SERVERS instead of
                      EPICS_CA_ADDR_LIST. Required for SSH tunnels. Default: False
                - pva_channels: (optional) List of glob patterns. Channel addresses
                  matching any pattern are served over PVAccess (p4p) instead of
                  Channel Access. Absent or empty => pure Channel Access, exactly
                  as before: p4p is not imported and no PVA env var is touched.
                - pva_gateway: (optional) {address, port, use_name_server} naming
                  where PVA searches go. Only read when pva_channels is non-empty,
                  and the ONLY route: the connector host scrubs every inherited
                  EPICS_PVA_* variable before connect() runs, so a value set in
                  the deployment's environment never reaches p4p.
                    - address: one host, or a space-separated list of hosts (what
                      a facility with many PVA servers and no gateway needs).
                    - port: (optional) With use_name_server false the entries go
                      to EPICS_PVA_ADDR_LIST as UDP search targets, and no port
                      is appended unless one is set here (p4p's default, 5076,
                      applies). Set, it is appended to every listed host. With
                      use_name_server true the entry is a TCP name server in
                      EPICS_PVA_NAME_SERVERS, defaulting to 5075.
                    - use_name_server: (optional) TCP instead of UDP search.
                      Required for SSH tunnels. Default: False

        Raises:
            ImportError: If aioca is not installed, or if PVA channels are
                configured and p4p is not installed
            RuntimeError: If this process's Channel Access context already
                serves different ``EPICS_CA_*`` settings
            ValueError: If the block still carries ``timeout``, or if
                ``timeout_s`` is not a positive, finite number
            ConnectionError: If this is a readonly run and the block configures
                a ``write_access`` gateway but no ``read_only`` one
        """
        # Import aioca here to give a clear error if it is not installed. It
        # loads libca from epicscorelibs, which ships one per architecture.
        try:
            import aioca

            self._aioca = aioca
        except ImportError:
            raise ImportError(
                "aioca is required for EPICS Channel Access. Install with: pip install aioca"
            ) from None

        # Refused before the gateway selection below rewrites the process-wide
        # EPICS_CA_* environment, so an unusable bound never repoints CA.
        self._timeout = call_timeout_s(config, self._connector_type)

        # Select the CA gateway. EPICS uses one process-wide context, so the
        # connector points at a single gateway. A read-only gateway rejects
        # writes, so a connector whose writes the verdict allows routes through
        # the write-capable gateway; otherwise it stays on read_only, and a
        # write it should not make is also refused by the gateway. A machine
        # reached only through a write gateway reads through it while its
        # writes are off, refused in software. A readonly run never routes
        # through the write gateway, so a raw caput issued after read_channel()
        # in the same process is refused by the gateway rather than trusted.
        # The rule is the one _gateway_role._selected_role states for every
        # caller; it is applied here, not restated.
        from osprey_connectors.control_system._gateway_role import (
            _selected_role,
            readonly_gateway_refusal,
        )

        gateways = config.get("gateways", {}) or {}
        # The whole write verdict, the same one the reference monitor takes.
        try:
            writes_enabled = self._writes_enabled
        except (FileNotFoundError, KeyError, RuntimeError):
            writes_enabled = False  # Can't tell -> assume the safe (read-only) path

        readonly_run = is_readonly_run()
        write_gateway = gateways.get("write_access") or {}
        gateway_role = _selected_role(
            gateways, writes_enabled=bool(writes_enabled), readonly_run=readonly_run
        )
        gateway_config = gateways.get(gateway_role) or {}
        if write_gateway and not gateway_config:
            # Only a readonly run selects no gateway while a write gateway is
            # configured. Connecting with none would leave Channel Access to
            # search the ambient network, so the run refuses instead.
            # The kind comes from the configured section, as the pool's refusal
            # takes it: a readonly run's verdict reads no config, so it would
            # leave the real machine unnamed.
            try:
                from osprey_connectors.config import get_config_value

                section = get_config_value("control_system", {})
            except (FileNotFoundError, RuntimeError):
                section = {}
            raise ConnectionError(
                readonly_gateway_refusal(machine_kind(section, self._control_target))
            )
        if gateway_role == "write_access" and not writes_enabled:
            logger.debug(
                "EPICS connector: no read_only gateway; reading through write_access "
                "with writes refused"
            )
        elif gateway_role == "write_access":
            logger.debug("EPICS connector: routing through write_access gateway (writes enabled)")
        elif readonly_run and write_gateway:
            logger.debug("EPICS connector: readonly run, staying on read_only gateway")
        elif writes_enabled and not write_gateway:
            logger.warning(
                f"Writes are allowed ({WRITES_KEY}: allowed) but no gateways.write_access "
                "is configured; routing through the read_only gateway, which may "
                "reject writes. Configure gateways.write_access to enable hardware writes."
            )

        # The EPICS_CA_* settings this connector needs, planned against the
        # environment as it stands.
        ca_env = current_ca_env()
        self._name_server = None
        if gateway_config:
            address = gateway_config.get("address", "")
            port = gateway_config.get("port", 5064)
            use_name_server = config_flag(
                gateway_config.get("use_name_server"),
                key=f"control_system.connector.{self._connector_type}.gateways."
                f"{gateway_role}.use_name_server",
            )

            # Clear conflicting variables first — having both CA_ADDR_LIST and
            # CA_NAME_SERVERS set causes TCP connection attempts that block
            # asyncio.to_thread() worker threads.
            if use_name_server:
                # Use CA_NAME_SERVERS (required for SSH tunnels and some gateway configurations)
                ca_env["EPICS_CA_NAME_SERVERS"] = f"{address}:{port}"
                self._name_server = f"{address}:{port}"
                ca_env["EPICS_CA_ADDR_LIST"] = None
                ca_env["EPICS_CA_SERVER_PORT"] = None
                logger.debug(f"Using EPICS_CA_NAME_SERVERS: {address}:{port}")
            else:
                # Use CA_ADDR_LIST (standard gateway configuration)
                ca_env["EPICS_CA_ADDR_LIST"] = address
                ca_env["EPICS_CA_SERVER_PORT"] = str(port)
                ca_env["EPICS_CA_NAME_SERVERS"] = None
                logger.debug(f"Using EPICS_CA_ADDR_LIST: {address}, CA_SERVER_PORT: {port}")

            ca_env["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

        # Channel Access reads the EPICS_CA_* environment once, when the
        # process's CA context is created. A context already serving other
        # settings is dropped for a fresh one rather than left on the old
        # gateway: a read-only run must never ride an earlier connector's write
        # gateway, and a notebook kernel that follows a target switch must
        # reach the new machine.
        self._ca = get_ca_loop()
        await self._ca.point_at(ca_env)
        self._ca_bulk()
        self._selected_role = gateway_role if gateway_config else None
        self._write_gateway_configured = bool(write_gateway)
        if gateway_config:
            logger.debug(f"Configured EPICS gateway: {address}:{port}")
            self._epics_configured = True

        # The ceiling on the fresh read a `max_step` check makes before a
        # write. A facility-network fact like `timeout_s`, and read from
        # the same block: a gateway two hops away answers slower than a soft
        # IOC on this host. Running out of budget answers None, which refuses
        # the write — raising it buys a slow channel more room, never a
        # weaker check.
        self._step_read_timeout = step_read_timeout_seconds(config, self._connector_type)
        # `fresh_reads` once made every read bypass pyepics' monitor cache.
        # Every Channel Access read now asks the IOC, so the key changes
        # nothing; it is still read, so a deployment that sets it keeps
        # loading, and a misspelt value still refuses to connect.
        self._fresh_reads = config_flag(
            config.get("fresh_reads"),
            key=f"control_system.connector.{self._connector_type}.fresh_reads",
        )

        # Configure PVAccess routing. Addresses matching one of these globs are
        # served by the p4p client; every other address keeps using Channel
        # Access. An absent or empty list is a hard no-op: no p4p import, no PVA
        # environment variable, no client context.
        pva_channels = config.get("pva_channels") or []
        if isinstance(pva_channels, str):
            pva_channels = [pva_channels]
        self._pva_channel_globs = [str(p).strip() for p in pva_channels if str(p).strip()]

        if self._pva_channel_globs:
            # Import p4p here (never at module scope) to give a clear error if
            # the PVA client is not installed, mirroring the aioca guard above.
            try:
                import p4p
                import p4p.client.thread  # loads the client API

                self._p4p = p4p
            except ImportError:
                raise ImportError(
                    "p4p is required for PVA channels (control_system.connector."
                    "epics.pva_channels). Install with: pip install p4p"
                ) from None

            pva_gateway = config.get("pva_gateway") or {}
            if pva_gateway:
                pva_address = str(pva_gateway.get("address", ""))
                pva_port = pva_gateway.get("port")
                pva_use_name_server = config_flag(
                    pva_gateway.get("use_name_server"),
                    key=f"control_system.connector.{self._connector_type}.pva_gateway."
                    "use_name_server",
                )
                # PVA carries the port inside the address entry itself — there is
                # no client-side "server port" variable to set, unlike CA.
                if pva_use_name_server:
                    # Name servers make the client connect by TCP instead of
                    # UDP-searching — required for SSH tunnels. 5075 is the PVA
                    # TCP port.
                    pva_endpoint = f"{pva_address}:{pva_port or 5075}"
                    os.environ["EPICS_PVA_NAME_SERVERS"] = pva_endpoint
                    os.environ.pop("EPICS_PVA_ADDR_LIST", None)
                    logger.debug(f"Using EPICS_PVA_NAME_SERVERS: {pva_endpoint}")
                else:
                    # Address-list entries are UDP search targets, whose default
                    # port is 5076, not the TCP 5075 — an appended 5075 makes
                    # every search miss. So no port is appended unless one is
                    # set explicitly, and then to every host of the list.
                    hosts = pva_address.split()
                    if pva_port:
                        hosts = [f"{host}:{pva_port}" for host in hosts]
                    pva_endpoint = " ".join(hosts)
                    os.environ["EPICS_PVA_ADDR_LIST"] = pva_endpoint
                    os.environ.pop("EPICS_PVA_NAME_SERVERS", None)
                    logger.debug(f"Using EPICS_PVA_ADDR_LIST: {pva_endpoint}")

                # Containment, mirroring EPICS_CA_AUTO_ADDR_LIST above: a
                # deployment deliberately pinned to a gateway must not also
                # broadcast-discover servers on the local subnet.
                os.environ["EPICS_PVA_AUTO_ADDR_LIST"] = "NO"
                logger.debug(f"Configured PVA gateway: {pva_endpoint}")

            # Create the PVA client context EAGERLY, here, rather than on first
            # read: concurrent reads (read_multiple_channels gathers several
            # asyncio.to_thread calls) would otherwise race to build it.
            try:
                self._pva_context = self._p4p.client.thread.Context("pva")
            except (RuntimeError, OSError) as exc:
                raise ConnectorUnavailableError(
                    f"PVA client context could not be built: {exc}"
                ) from exc
            logger.debug(f"PVA routing enabled for {len(self._pva_channel_globs)} glob pattern(s)")

        # Initialize limits validator for automatic validation and confirm policy
        from osprey_connectors.control_system._limits_validator import LimitsValidator

        self._limits_validator = LimitsValidator.from_config(connector_type=self._connector_type)
        if self._limits_validator:
            logger.debug("EPICS connector: limits validator initialized")

        self._connected = True
        logger.debug("EPICS connector initialized")

    def _is_pva_channel(self, channel_address: str) -> bool:
        """Return True when this address is routed over PVAccess instead of CA.

        Matching uses :func:`fnmatch.fnmatchcase`, which is case-sensitive and
        platform-independent — plain ``fnmatch.fnmatch`` applies
        ``os.path.normcase`` and would make routing depend on the host OS.
        Addresses stay raw: no ``pva://`` scheme is ever added or stripped, so
        the limits DB, channel DBs and audit records keep matching on the same
        strings they always did.

        A connector with no configured globs always returns False.
        """
        return any(
            fnmatch.fnmatchcase(channel_address, pattern) for pattern in self._pva_channel_globs
        )

    async def disconnect(self) -> None:
        """Cleanup EPICS connections on both transports.

        Order matters: subscriptions first, then the PVA client context, then
        Channel Access. A p4p ``Subscription`` belongs to the ``Context`` that
        opened it, so the context outlives its monitors here rather than being
        torn down underneath them.

        Closing the context is not optional bookkeeping: it owns worker
        threads, and connector unavailability invalidates and rebuilds the
        connector in production — a context left open on every invalidation
        leaks those threads for the life of the process. The Channel Access
        context is not this connector's: it is one per process, on the CA loop,
        so dropping and rebuilding a connector in process gives it no fresh CA
        transport; for Channel Access the recovery is a fresh connector-host
        child. The close is
        idempotent (a connector that never built a context, or that is
        disconnected twice, does nothing) and best effort, like the CA teardown
        below.
        """
        # Unsubscribe from all active subscriptions
        for sub_id in list(self._subscriptions.keys()):
            await self.unsubscribe(sub_id)

        # Close the PVA client context, once, after the monitors it owns
        pva_context, self._pva_context = self._pva_context, None
        if pva_context is not None:
            try:
                pva_context.close()
            except Exception as exc:  # teardown is best effort, never fatal
                logger.debug(f"PVA client context close failed: {exc}")
            else:
                logger.debug("PVA client context closed")

        # Channel Access channels belong to the process's CA loop and outlive
        # any one connector; only this connector's cached control fields go.
        self._ca_ctrl.clear()

        # Shut the write executor down without waiting: a validation still
        # running finishes within its own bound, and a later connect() builds
        # a fresh executor on its first write.
        with self._write_executor_lock:
            executor, self._write_executor = self._write_executor, None
        if executor is not None:
            executor.shutdown(wait=False)

        self._connected = False
        logger.info("EPICS connector disconnected")

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        """
        Read current value from EPICS channel.

        Addresses matching one of the configured ``pva_channels`` globs are read
        over PVAccess, a blocking client call run in a worker thread. Every
        other address is read over Channel Access, on the process's CA loop.

        Args:
            channel_address: EPICS channel address (e.g., 'BEAM:CURRENT')
            timeout: Timeout in seconds (uses default if None)

        Returns:
            ChannelValue with current value, timestamp, and metadata

        Raises:
            ChannelUnreachableError: The channel did not connect or did not
                answer in time
            ConnectorUnavailableError: The PVA client context is gone
            ValueError: If a PVA channel serves a compressed NTNDArray
        """
        timeout = timeout or self._timeout

        if self._is_pva_channel(channel_address):
            # p4p's thread client blocks, so it runs in a worker thread.
            return await asyncio.to_thread(self._read_channel_pva, channel_address, timeout)

        name = normalise_address(channel_address)
        deadline = (time.time() + timeout,)
        return await self._ca.run(self._read_channel_ca(name, deadline, timeout))

    # ------------------------------------------------------------------
    # Channel Access (aioca). Every coroutine here runs on the CA loop.
    # Each phase is bounded by one absolute deadline, ``(time.time() + T,)``,
    # handed to every aioca call in it: aioca bounds connect and call
    # together under one timeout, and a tuple is a deadline rather than a
    # fresh allowance per call.
    # ------------------------------------------------------------------

    async def _ca_connect(self, name: str, deadline: tuple[float]) -> Any:
        """Connect ``name`` within ``deadline``; record the outcome for :meth:`wait_stale`."""
        connected = await self._aioca.connect(name, timeout=deadline, throw=False)
        if connected:
            # A new connect: the channel's first, or its first since it failed,
            # and not one a subscription record already holds connected. A
            # name's first connect after a repoint is not counted: the repoint
            # purged aioca's cache, not this bookkeeping, so it is a false
            # negative only, which the silent rule backs.
            new = name not in self._one_shot_connected or name in self._connect_failures
            if new and not (self._bulk is not None and self._bulk.is_down(name) is False):
                self._one_shot_connect_at = time.monotonic()
            self._one_shot_connected.add(name)
            self._connect_failures.pop(name, None)
        else:
            self._connect_failures[name] = time.monotonic()
        return connected

    async def _read_channel_ca(
        self, name: str, deadline: tuple[float], timeout: float, *, ctrl: str = "always"
    ) -> ChannelValue:
        """Read a channel's value, alarm state and timestamp from the IOC, within ``deadline``.

        Connecting is a step of its own, so a channel nobody answers for is
        reported as the connection failure it is, apart from a get that timed
        out on a channel that did connect. Every read asks the IOC: there is
        no monitor cache to answer from, so no reading can be stale.

        ``ctrl="always"`` enriches the reading with the channel's control
        fields (units, precision, labels, limits) and write access;
        ``ctrl="enum"`` fetches them only for an enumeration, whose label a
        write confirmation compares -- the one thing it needs them for.
        ``timeout`` is only quoted in the error.
        """
        aioca = self._aioca
        connected = await self._ca_connect(name, deadline)
        if not connected:
            raise ChannelUnreachableError(
                name, f"Failed to connect to PV '{name}' (timeout after {timeout}s)"
            )
        reading = await aioca.caget(name, format=aioca.FORMAT_TIME, timeout=deadline, throw=False)
        if isinstance(reading, aioca.CANothing):
            raise _ca_failure(name, reading, "Read of")
        want_ctrl = ctrl == "always" or _is_enum_type(
            _ca_type_name(getattr(reading, "datatype", None))
        )
        fields = await self._ca_ctrl_fields(name, deadline) if want_ctrl else None
        writable = None
        if ctrl == "always":
            info = await aioca.cainfo(name, timeout=deadline, throw=False)
            if not isinstance(info, aioca.CANothing):
                writable = getattr(info, "write", None)
        return self._ca_channel_value(reading, fields, writable)

    async def _ca_ctrl_fields(self, name: str, deadline: tuple[float]) -> Any:
        """The channel's CTRL-form reading (units, precision, labels, limits), or ``None``.

        Fetched once per channel and reused: these fields describe the record,
        not the reading. One element is asked for, so a waveform's payload is
        not shipped with its metadata. A fetch that fails (or runs out of the
        deadline) is logged and not cached, so the next read tries again; the
        reading it was meant to enrich still stands.
        """
        cached = self._ca_ctrl.get(name)
        if cached is not None:
            return cached
        aioca = self._aioca
        fields = await aioca.caget(
            name, format=aioca.FORMAT_CTRL, count=1, timeout=deadline, throw=False
        )
        if isinstance(fields, aioca.CANothing):
            logger.debug("Could not fetch control fields for '%s': %s", name, fields)
            return None
        self._ca_ctrl[name] = fields
        return fields

    def _ca_channel_value(
        self, reading: Any, ctrl: Any, writable: bool | None = None
    ) -> ChannelValue:
        """Build the connector's ChannelValue from a TIME-form reading and its CTRL fields.

        The alarm status is reported by name; the raw CA code stays in
        raw_metadata next to the severity so nothing is lost. The read path and
        the monitor path both come through here, so they cannot drift apart.
        """
        tz = get_facility_timezone()
        value = _ca_plain(reading)
        stamp = getattr(reading, "timestamp", None)
        timestamp = datetime.fromtimestamp(stamp, tz) if stamp else datetime.now(tz)
        alarm_code = getattr(reading, "status", None)
        severity = getattr(reading, "severity", None)
        pv_type = _ca_type_name(getattr(reading, "datatype", None))
        labels, label = (
            _enum_label_fields(getattr(ctrl, "enums", None), value)
            if _is_enum_type(pv_type)
            else (None, None)
        )
        metadata = ChannelMetadata(
            units=getattr(ctrl, "units", "") or "",
            precision=getattr(ctrl, "precision", None),
            alarm_status=_alarm_name(alarm_code),
            alarm_severity=_severity_int(severity),
            timestamp=timestamp,
            enum_labels=labels,
            enum_label=label,
            raw_metadata={
                "status": alarm_code,
                "severity": severity,
                "type": pv_type,
                "count": getattr(reading, "element_count", None),
            },
            **_ca_ctrl_limits(ctrl, writable),
        )
        return ChannelValue(value=value, timestamp=timestamp, metadata=metadata)

    @staticmethod
    def _ca_ctrl_metadata(ctrl: Any, writable: bool | None) -> ChannelMetadata:
        """Metadata of a subscribed channel that has delivered control data but no value yet."""
        return ChannelMetadata(
            units=getattr(ctrl, "units", "") or "",
            precision=getattr(ctrl, "precision", None),
            enum_labels=_enum_label_fields(getattr(ctrl, "enums", None), None)[0],
            **_ca_ctrl_limits(ctrl, writable),
        )

    async def _put_ca(
        self, channel_address: str, value: Any, wait: bool, deadline: tuple[float]
    ) -> tuple[str, Any]:
        """Put ``value`` within ``deadline``; say whether it was sent, acknowledged, or refused.

        Connect, field-type lookup and put share the one deadline T the
        caller set before handing this to the CA loop, so time spent queued
        for the loop is charged to it too.

        Answers one of:

        - ``("not_sent", failure)``: the channel never connected, or the
          deadline ran out before the put was issued, so nothing left this
          process.
        - ``("unacknowledged", failure)``: the put went out and its callback did
          not arrive in time. aioca reports this and a connect timeout with the
          same code, which is why connecting is a step of its own here.
        - ``("unwritable", reason)``: the value is one the channel's field
          type cannot hold exactly (see :func:`_ca_put_form`), so nothing was
          sent.
        - ``("cs_refused", failure)``: the IOC's access security denied it.
        - ``("put_failed", failure)``: the IOC answered the put and did not
          take it (a disabled record, a value it could not convert).
        - ``("sent", (sent, as_text))``: sent, and, when ``wait``,
          acknowledged. ``sent`` is the value as it went out (text read into a
          number for a numeric field, a whole float as the integer it is), the
          one a confirming read is compared with; ``as_text`` says it went to
          a char array as text.

        Any other failure is raised: it leaves the outcome genuinely unknown,
        and every answer above claims to know it.
        """
        aioca = self._aioca
        connected = await self._ca_connect(channel_address, deadline)
        if not connected:
            return "not_sent", f"did not connect: {connected}"
        # The channel's own field type decides what goes out. cainfo reads it
        # from the channel just connected, without a round trip to the IOC.
        info = await aioca.cainfo(channel_address, timeout=deadline, throw=False)
        if isinstance(info, aioca.CANothing):
            return "not_sent", f"field type unreadable: {info}"
        try:
            put_value, as_text = _ca_put_form(value, info.datatype, info.count)
        except _Unwritable as reason:
            return "unwritable", reason
        datatype = aioca.DBR_CHAR_STR if as_text else None
        sent = value if as_text else put_value
        if time.time() >= deadline[0]:
            # aioca would time an exhausted deadline out before sending,
            # and that would read as a put that went out unacknowledged.
            return "not_sent", TimeoutError("the put deadline ran out before the put")
        done = await aioca.caput(
            channel_address, put_value, datatype=datatype, wait=wait, timeout=deadline, throw=False
        )
        if done:
            return "sent", (sent, as_text)
        code = getattr(done, "errorcode", None)
        if code == _ECA_TIMEOUT:
            return "unacknowledged", done
        if code == _ECA_NOWTACCESS:
            return "cs_refused", done
        if code == _ECA_PUTFAIL:
            return "put_failed", done
        raise ChannelUnreachableError(
            channel_address, f"Write to '{channel_address}' failed: {done}"
        )

    # ------------------------------------------------------------------
    # PVAccess read path
    # ------------------------------------------------------------------

    def _pva_connection_error_types(self) -> tuple[type[BaseException], ...]:
        """p4p exceptions that mean "the channel is not reachable".

        Looked up through ``self._p4p`` (never a module-scope import) and
        filtered to real exception classes, so a p4p release that drops or
        renames one of them degrades to "not caught here" instead of crashing
        the read with a TypeError inside the except clause. Read only by
        :meth:`_pva_failure`.
        """
        thread_module = getattr(getattr(self._p4p, "client", None), "thread", None)
        found: list[type[BaseException]] = []
        for name in ("Disconnected", "RemoteError", "Cancelled"):
            candidate = getattr(thread_module, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                found.append(candidate)
        return tuple(found)

    def _pva_failure(self, channel_address: str, failure: Exception, action: str) -> Exception:
        """The exception a failed p4p call is raised as, the PVA twin of :func:`_ca_failure`.

        A get that timed out and a channel p4p reports disconnected, remote-failed
        or cancelled are both :class:`ChannelUnreachableError`, naming the
        channel: the PVA client context keeps serving every other channel.
        Anything else is returned unchanged, for the caller to re-raise as it is.
        """
        if isinstance(failure, TimeoutError):
            return ChannelUnreachableError(
                channel_address, f"{action} PVA channel '{channel_address}' timed out: {failure}"
            )
        if isinstance(failure, self._pva_connection_error_types()):
            return ChannelUnreachableError(
                channel_address, f"Failed to connect to PVA channel '{channel_address}': {failure}"
            )
        return failure

    def _require_pva_context(self, channel_address: str, action: str) -> Any:
        """The PVA client context, or the connector's own unavailability.

        Raises:
            ConnectorUnavailableError: No PVA client context is open, so no PVA
                channel can be served until the connector is rebuilt.
        """
        context = self._pva_context
        if context is None:
            raise ConnectorUnavailableError(
                f"Cannot {action} PVA channel '{channel_address}': no PVA client context. "
                "connect() builds it from control_system.connector.epics.pva_channels."
            )
        return context

    def _pva_get(self, channel_address: str, timeout: float, request: str | None = None) -> Any:
        """Blocking p4p get, returning the raw p4p Value (runs in a thread pool).

        Failures are raised through :meth:`_pva_failure`, the one funnel that
        classifies a p4p failure, so a PVA channel that does not answer is the
        same :class:`ChannelUnreachableError` a CA one is; a missing client
        context is the connector's own :class:`ConnectorUnavailableError`.

        A default p4p Context unwraps normative types into augmented python
        objects (ntfloat, ntenum, ntndarray) that keep the underlying Value on
        ``.raw``. Returning that raw structure — rather than the augmented
        object — is what lets the codec guard run before any reshape, and keeps
        the mapping identical for an unwrap-disabled Context.

        Args:
            channel_address: PVA channel to read.
            timeout: Timeout in seconds.
            request: Optional pvRequest string (e.g. :data:`_PVA_METADATA_REQUEST`).
                When omitted, the server's whole structure is fetched.
        """
        context = self._require_pva_context(channel_address, "read")

        try:
            if request is None:
                result = context.get(channel_address, timeout=timeout)
            else:
                result = context.get(channel_address, request=request, timeout=timeout)
        except Exception as exc:
            failure = self._pva_failure(channel_address, exc, f"Read (timeout {timeout}s) of")
            if failure is exc:
                raise
            raise failure from exc

        return getattr(result, "raw", result)

    def _read_channel_pva(self, channel_address: str, timeout: float) -> ChannelValue:
        """Synchronous PVAccess read (runs in a thread pool, like the CA read)."""
        value = self._pva_get(channel_address, timeout)
        return self._pva_channel_value(channel_address, value)

    def _pva_channel_value(self, channel_address: str, value: Any) -> ChannelValue:
        """Map a p4p Value onto :class:`ChannelValue` by normative type."""
        type_id = _pva_type_id(value)
        timestamp = self._pva_timestamp(value)
        raw_metadata: dict[str, Any] = {"nt_id": type_id}
        raw_metadata.update(self._pva_alarm_metadata(value))

        if type_id.startswith(_NT_NDARRAY_ID):
            payload = self._pva_ndarray_value(channel_address, value)
        elif type_id.startswith(_NT_ENUM_ID):
            payload = self._pva_enum_value(value)
        else:
            payload = self._pva_scalar_value(value)

        raw_metadata.update(payload.pop("raw_metadata", {}))

        metadata = self._pva_metadata(value, timestamp, raw_metadata)
        # Only the enum branch contributes these; every other payload leaves the
        # fields at their None default, which is what marks a channel as not
        # enum-typed.
        metadata.enum_labels = payload.get("enum_labels")
        metadata.enum_label = payload.get("enum_label")

        return ChannelValue(value=payload["value"], timestamp=timestamp, metadata=metadata)

    def _pva_metadata(
        self, value: Any, timestamp: datetime, raw_metadata: dict[str, Any]
    ) -> ChannelMetadata:
        """Map the NT ``display``, ``control``, ``valueAlarm`` and ``alarm`` fields.

        Reads nothing but metadata fields, so it maps a full read and a
        metadata-only get (:data:`_PVA_METADATA_REQUEST`) identically — the
        payload branches contribute only what they put in ``raw_metadata``.
        ``control`` and ``valueAlarm`` are optional NT fields: where a server
        omits them, or reports something non-numeric, the limits are ``None``
        (:class:`ChannelMetadata` normalises whatever is handed over).
        """
        display = _pva_field(value, "display")
        control = _pva_field(value, "control")
        value_alarm = _pva_field(value, "valueAlarm")
        units = _pva_field(display, "units", "") or ""
        precision = _pva_field(display, "precision")
        description = _pva_field(display, "description") or None

        return ChannelMetadata(
            units=str(units),
            precision=int(precision) if isinstance(precision, int | float) else None,
            alarm_status=raw_metadata.get("alarm_message") or None,
            alarm_severity=_severity_int(raw_metadata.get("severity")),
            timestamp=timestamp,
            description=str(description) if description else None,
            display_low=_pva_field(display, "limitLow"),
            display_high=_pva_field(display, "limitHigh"),
            control_low=_pva_field(control, "limitLow"),
            control_high=_pva_field(control, "limitHigh"),
            warning_low=_pva_field(value_alarm, "lowWarningLimit"),
            warning_high=_pva_field(value_alarm, "highWarningLimit"),
            alarm_low=_pva_field(value_alarm, "lowAlarmLimit"),
            alarm_high=_pva_field(value_alarm, "highAlarmLimit"),
            raw_metadata=raw_metadata,
        )

    def _pva_timestamp(self, value: Any) -> datetime:
        """Channel timestamp from the NT timeStamp field, else now()."""
        stamp = _pva_field(value, "timeStamp")
        seconds = _pva_field(stamp, "secondsPastEpoch", 0) or 0
        nanoseconds = _pva_field(stamp, "nanoseconds", 0) or 0
        if isinstance(seconds, int | float) and seconds:
            nanos = nanoseconds if isinstance(nanoseconds, int | float) else 0
            return datetime.fromtimestamp(seconds + nanos * 1e-9, get_facility_timezone())
        return datetime.now(get_facility_timezone())

    def _pva_alarm_metadata(self, value: Any) -> dict[str, Any]:
        """Alarm fields, recorded raw so nothing about the alarm state is lost."""
        alarm = _pva_field(value, "alarm")
        if alarm is None:
            return {}
        message = _pva_field(alarm, "message", "") or ""
        return {
            "severity": _pva_field(alarm, "severity"),
            "status": _pva_field(alarm, "status"),
            "alarm_message": str(message),
        }

    def _pva_scalar_value(self, value: Any) -> dict[str, Any]:
        """NTScalar / NTScalarArray / unrecognized structure: the value field as-is."""
        scalar = _pva_field(value, "value")
        dtype = getattr(scalar, "dtype", None)
        return {
            "value": scalar,
            "raw_metadata": {"dtype": str(dtype) if dtype is not None else type(scalar).__name__},
        }

    def _pva_enum_value(self, value: Any) -> dict[str, Any]:
        """NTEnum: the INDEX is the value; the choices become enum metadata.

        Both halves of a state reading reach the operator, and each in the place
        it belongs: ``value`` is the index, so a reading has one machine-readable
        type whichever protocol served it, and the choice string arrives as
        ``ChannelMetadata.enum_label`` — surfaced in the tool envelope, so
        nobody has to know that "2" means ACQUIRING. Channel Access maps enums
        the same way, which is what makes a channel's reading independent of
        whether it was routed over CA or PVA.

        The index and the choices list stay in ``raw_metadata`` as well: they
        predate the typed fields and are what a PVA-specific consumer already
        reads.
        """
        enum_field = _pva_field(value, "value")
        index = _pva_field(enum_field, "index")
        choices = list(_pva_field(enum_field, "choices", []) or [])
        labels, label = _enum_label_fields(choices, index)

        return {
            "value": index,
            "enum_labels": labels,
            "enum_label": label,
            "raw_metadata": {
                "dtype": "enum",
                "enum_index": index,
                "enum_choices": choices,
            },
        }

    def _pva_ndarray_value(self, channel_address: str, value: Any) -> dict[str, Any]:
        """NTNDArray: the carried array, reshaped per its dimension list.

        The array is taken from the selected union member exactly as p4p decoded
        it — never re-cast — so an unsigned frame stays unsigned instead of
        reading as negative pixels.
        """
        codec = _pva_field(value, "codec")
        codec_name = str(_pva_field(codec, "name", "") or "")
        dimensions = [_pva_field(dim, "size") for dim in _pva_field(value, "dimension", []) or []]

        if codec_name:
            # Never reshape a compressed payload: the union carries the
            # compressed byte blob, and reshaping it would produce a plausible
            # image full of meaningless pixel statistics.
            raise ValueError(
                f"Cannot read '{channel_address}': compressed NTNDArray unsupported; "
                f"disable ADPva compression for this channel "
                f"(codec={codec_name!r}, dimensions={dimensions})"
            )

        array = _pva_field(value, "value")
        color_mode = _pva_color_mode(value)
        raw_metadata: dict[str, Any] = {
            "dtype": str(getattr(array, "dtype", type(array).__name__)),
            "dimensions": dimensions,
            "color_mode": color_mode,
            "codec": codec_name,
            "unique_id": _pva_field(value, "uniqueId"),
        }

        # NT dimensions run innermost-first (width, height, ...); numpy shape is
        # the reverse. This holds for the RGB color modes too, whose dimension
        # list already carries the 3 in its own place.
        shape = tuple(int(size) for size in reversed(dimensions) if isinstance(size, int))
        if shape and hasattr(array, "reshape"):
            try:
                array = array.reshape(shape)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot read '{channel_address}': NTNDArray dimensions {dimensions} "
                    f"do not describe its {getattr(array, 'size', '?')}-element payload ({exc})"
                ) from None

        raw_metadata["shape"] = list(getattr(array, "shape", ()))
        return {"value": array, "raw_metadata": raw_metadata}

    def current_value_reader(self) -> Callable[[str], Any] | None:
        """The channel's present value, read with the client this connector connected with.

        The limits validator calls this synchronously from a worker thread, so
        the read is run on the CA loop and waited for here. Any failure — no
        connection, no answer within ``step_read_timeout`` — answers ``None``,
        which fails the step check closed.

        A PVA-routed address answers ``None`` too. ``write_channel`` refuses
        those before validation ever runs, so this is the belt to that braces:
        the CA client must not be pointed at an address routed over PVAccess.
        """

        def read_current(channel_address: str) -> Any:
            if self._is_pva_channel(channel_address):
                return None
            aioca = self._aioca
            budget = self._step_read_timeout
            try:
                # One budget S for connection and get together (aioca bounds
                # both under one timeout); the extra second only backs up a
                # CA loop that does not answer at all.
                reading = self._ca.run_sync(
                    aioca.caget(normalise_address(channel_address), timeout=budget, throw=False),
                    timeout=budget + 1.0,
                )
            except Exception as exc:  # the CA loop did not answer in time
                logger.debug(f"max_step read of '{channel_address}' failed: {exc}")
                return None
            if isinstance(reading, aioca.CANothing):
                return None
            return _ca_plain(reading)

        return read_current

    def _write_pool(self) -> _WriteExecutor:
        """The executor a write's limits validation runs on, built on first use."""
        with self._write_executor_lock:
            if self._write_executor is None:
                self._write_executor = _WriteExecutor()
            return self._write_executor

    async def _run_write_phase(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run one blocking write phase on the write executor, inside the caller's context.

        ``loop.run_in_executor`` alone starts the worker from a fresh context;
        running ``fn`` through ``contextvars.copy_context().run`` carries every
        context variable of the write into it, as ``asyncio.to_thread`` does.
        """
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        return await loop.run_in_executor(self._write_pool(), context.run, fn, *args)

    def _role_refusal(self, channel_address: str, value: Any) -> ChannelWriteResult | None:
        """The ``switching`` refusal for a write the selected gateway cannot carry, or ``None``.

        Called only for a write the verdict allowed. The block configures a
        ``write_access`` gateway and ``connect()`` did not select it: the switch
        turned on after this connector chose its gateway, and the write would go
        out on the read gateway. The connector re-selects on its next
        ``connect()`` (a pooled child is replaced by a fresh one); until then the
        write is refused, worded as the ``switching`` reason. With no write
        gateway configured the connector writes through what it has.
        """
        if not self._write_gateway_configured or self._selected_role == "write_access":
            return None
        # Imported here, as ``_base`` does: ``posture_store`` imports ``_base``.
        from osprey_connectors.posture_store import WriteVerdict

        allowed = self._last_verdict
        verdict = WriteVerdict(
            allowed=False,
            reason="switching",
            target=allowed.target if allowed is not None else self._control_target,
            kind=allowed.kind if allowed is not None else None,
        )
        return _verdict_refusal(channel_address, value, verdict)

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """
        Write a value to an EPICS channel and confirm the channel took it.

        The connector automatically:
        1. Validates limits (min/max/step/writable) if limits checking is enabled
        2. Puts the value, waiting for the IOC's put-callback when confirming
        3. Re-reads the channel once and compares what it now holds with what
           was sent, using the shared :func:`values_match` rule

        Args:
            channel_address: EPICS channel address
            value: Value to write
            timeout: Timeout in seconds
            confirm: Whether to confirm the write by re-reading the channel.
                ``None`` means "no opinion" and resolves this channel's own
                policy from the limits database.

        Returns:
            ChannelWriteResult carrying the one outcome word, and — when the
            write was confirmed — what the channel was seen to hold

        Raises:
            ChannelUnreachableError: If the put fails in a way that leaves its
                outcome unknown
            ChannelLimitsViolationError: If limits validation fails (when enabled)
        """
        # Step 0: PVA-routed addresses are read-only. This check MUST stay ahead
        # of limits validation: validating max_step reads the channel's current
        # value with a blocking `caget`, so a refusal placed after it would put
        # the CA client on an address routed over PVAccess — the very thing this
        # refusal exists to prevent — and would do it for the arrays PVA channels
        # carry, which have no scalar limit to be compared against.
        if self._is_pva_channel(channel_address):
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.REFUSED,
                refusal_reason=RefusalReason.VALIDATION_ERROR,
                error_message=(
                    f"Write to '{channel_address}' refused: PVAccess writes are not supported. "
                    "This address matches a configured 'pva_channels' pattern, so it is routed "
                    "over PVAccess and is read-only in this deployment. No write was attempted."
                ),
            )

        # Step 0b: the verdict allowed this write (the guard asked it first), but
        # the gateway this connector selected at connect() may predate the
        # switch. Refused before limits validation reads anything.
        role_refusal = self._role_refusal(channel_address, value)
        if role_refusal is not None:
            return role_refusal

        timeout = timeout or self._timeout

        # Step 1: Resolve the channel's confirmation policy when the caller has
        # no opinion (cheap, on-loop). An explicit False is an answer and is
        # never re-resolved.
        if confirm is None:
            confirm = self._resolve_confirm(channel_address)

        # Import here to avoid circular dependency
        from osprey_connectors.errors import ChannelLimitsViolationError

        # Step 2: Validate limits (FAIL CLOSED) in a thread of the write
        # executor, so a caller on the event loop is never stalled by the
        # blocking read that max_step validation waits for, and the validation
        # never queues behind PVA reads on the default executor.
        def _validate():
            if self._limits_validator:
                try:
                    self._limits_validator.validate(
                        channel_address, value, read_current=self.current_value_reader()
                    )
                    logger.debug(f"✓ Limits validation passed: {channel_address}={value}")
                except ChannelLimitsViolationError:
                    raise  # limits refusal propagates unchanged (carries LIMITS semantics)
                except Exception as e:
                    # FAIL CLOSED: any other validation error refuses the write — no caput
                    # issued. The refusal itself is built on the loop, by the base class's
                    # one helper, so all four connectors word it identically.
                    return e
            return None

        validation_error = await self._run_write_phase(_validate)
        if validation_error is not None:
            return self._validation_refusal(channel_address, value, validation_error)

        # Step 3: The put. The put-callback is Channel Access's acknowledgement
        # that the IOC processed the put, so a confirming write waits for it:
        # the read that follows would otherwise race the record's own
        # processing. A write nobody asked to confirm does not wait. The put
        # phase has one deadline T, set here so the wait for the CA loop is
        # charged to it as well.
        put_deadline = (time.time() + timeout,)
        put_result, payload = await self._ca.run(
            self._put_ca(
                normalise_address(channel_address), value, wait=bool(confirm), deadline=put_deadline
            )
        )

        if put_result == "cs_refused":
            # The control system was asked and said no: access security, not
            # a client bug, and nothing was written.
            logger.warning(f"Control system refused write to {channel_address}: {payload}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.REFUSED,
                refusal_reason=RefusalReason.CONTROL_SYSTEM_REFUSED,
                error_message=(
                    f"Write to '{channel_address}' refused by the control system "
                    f"(access security); no value was written: {payload}"
                ),
            )

        # The value was sent and its callback never came: nothing has said the
        # IOC took it, so the outcome is unknown and no re-read is made.
        if put_result == "unacknowledged":
            logger.warning(
                f"EPICS put not acknowledged within {timeout}s: {channel_address} = {value}"
            )
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNCONFIRMED,
                error_message=(
                    f"Write to '{channel_address}' could not be confirmed — the control "
                    f"system did not acknowledge the put within {timeout}s"
                ),
            )

        if put_result == "unwritable":
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.REFUSED,
                refusal_reason=RefusalReason.VALIDATION_ERROR,
                error_message=(
                    f"Write to '{channel_address}' refused: {payload}. No value was sent."
                ),
            )

        if put_result == "put_failed":
            logger.warning(f"IOC did not take the put to {channel_address}: {payload}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.FAILED,
                error_message=(
                    f"Write to '{channel_address}' failed: the IOC answered the put and "
                    f"did not take it: {payload}"
                ),
            )

        if put_result == "not_sent":
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.FAILED,
                error_message=(
                    f"Failed to write to channel '{channel_address}': no value was sent "
                    f"within {timeout}s ({payload})"
                ),
            )

        if not confirm:
            # Nothing was checked, and nothing may be claimed: the result stays
            # value-less by design.
            logger.debug(f"EPICS write (unconfirmed by request): {channel_address} = {value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNREQUESTED,
            )

        # Step 4: One fresh read of the channel that was just written.
        try:
            observed = await self._confirming_read(channel_address, timeout)
        except Exception as e:
            logger.warning(f"EPICS confirming read failed for {channel_address}: {e}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNCONFIRMED,
                error_message=(
                    f"Write to '{channel_address}' could not be confirmed — "
                    f"the read that followed it failed: {e}"
                ),
            )

        sent, as_text = payload
        if as_text:
            # Written to a char array as text, so it is read back as text.
            observed.value = _char_text(observed.value)

        alarm_status, alarm_severity = _readback_alarm_fields(observed)
        metadata = getattr(observed, "metadata", None)
        enum_label = getattr(metadata, "enum_label", None) if metadata is not None else None

        if values_match(sent, observed.value, enum_label=enum_label):
            logger.debug(f"EPICS write confirmed: {channel_address} = {observed.value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.CONFIRMED,
                observed_value=observed.value,
                alarm_status=alarm_status,
                alarm_severity=alarm_severity,
            )

        logger.warning(
            f"EPICS write mismatch on {channel_address}: sent {value}, "
            f"channel holds {observed.value}"
        )
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.MISMATCH,
            # No error_message: both values are on the result, and the raising
            # path composes "sent X, channel holds Y" from them — a message set
            # here would take that wording's place.
            observed_value=observed.value,
            alarm_status=alarm_status,
            alarm_severity=alarm_severity,
            notes=f"Channel holds {observed.value}, sent {value}",
        )

    async def _confirming_read(self, channel_address: str, timeout: float) -> ChannelValue:
        """Read the channel a write just touched, straight off the wire.

        The ordinary read path already asks the IOC every time, so a reading
        taken after the put-callback cannot be a stale cached one. A read that
        fails raises, and the caller reports the write as unconfirmed: what
        the channel holds is unknown, not different.

        Connect and get share one deadline T; the control fields are fetched
        only for an enumeration, whose label the comparison uses, and under
        the same deadline.
        """
        deadline = (time.time() + timeout,)
        return await self._ca.run(
            self._read_channel_ca(
                normalise_address(channel_address), deadline, timeout, ctrl="enum"
            )
        )

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        """Read multiple channels concurrently."""
        return await self._read_concurrently(channel_addresses, timeout)

    async def subscribe(
        self,
        channel_address: str,
        callback: Callable[[ChannelValue], None],
        *,
        on_connection: Callable[[bool], None] | None = None,
    ) -> str:
        """
        Subscribe to channel value changes.

        Args:
            channel_address: EPICS channel address
            callback: Function to call when value changes
            on_connection: Optional function told ``True`` when the channel
                connects and ``False`` when it drops (see the base class)

        Returns:
            Subscription ID for later unsubscription

        Routing mirrors the read path: a PVA-routed address opens a p4p
        monitor, every other address joins the address's Channel Access record
        (:mod:`~osprey_connectors.control_system._epics_bulk`): one pair of
        monitors per address, every value carrying the control data of the
        connection it arrived on, connection state reported ahead of the
        connection's first value. Either way the subscriber callbacks run on
        the event loop, never on the transport's own thread.

        A Channel Access subscription outlives a repoint: when a connect
        elsewhere in the process moves CA to another gateway, the record
        reports the channel down and opens it again on the new context.
        Cancelling the call leaves no subscriber behind it.
        """
        loop = asyncio.get_running_loop()

        if self._is_pva_channel(channel_address):
            return self._subscribe_pva(channel_address, callback, loop, on_connection)

        bulk = self._ca_bulk()
        subscription = _ChannelSubscription("ca", None, ca_loop=self._ca)
        sub_id = f"{channel_address}_{id(subscription)}"
        # Closing the subscription leaves the record, on the CA loop.
        subscription.handle = _BulkHandle(bulk, sub_id)
        try:
            await self._ca.run(
                bulk.subscribe(
                    channel_address,
                    sub_id,
                    callback,
                    loop,
                    on_connection,
                    released=lambda: subscription.closed,
                )
            )
        except asyncio.CancelledError:
            # The join may already be queued on the CA loop, or done with its
            # result in flight; the release is queued behind it.
            subscription.close()
            raise

        self._subscriptions[sub_id] = subscription
        logger.debug(f"EPICS subscription created: {sub_id}")
        return sub_id

    def _ca_bulk(self) -> CABulk:
        """This connector's Channel Access subscription path, built once.

        ``connect()`` builds it; the lock keeps two first subscribers on
        different loops from building two.
        """
        with self._bulk_lock:
            if self._bulk is None:
                self._bulk = CABulk(
                    self._aioca,
                    self._ca,
                    value_factory=self._ca_channel_value,
                    metadata_factory=self._ca_ctrl_metadata,
                    held_warning_s=self._timeout,
                )
            return self._bulk

    async def warm(self) -> None:
        """Create the Channel Access context the subscription path runs under.

        The session calls this (bounded) when it opens, so the first
        subscriber does not pay for creating the context. No channel is opened.
        """
        if self._aioca is None or self._ca is None:
            return
        aioca = self._aioca

        async def create_context() -> None:
            aioca.get_channel_infos()

        await self._ca.run(create_context())

    async def wait_stale(self) -> None:
        """Raise :class:`ConnectorUnavailableError` once this connector's Channel Access transport has gone stale.

        After an outage of a name server, libca may search for a channel it
        held again only after a long back-off, and its own connection to the
        name server can stay broken for good, so the channel stays down after
        the server is back. This watch reads aioca's own channel state on the
        CA loop, once per :data:`STALE_PROBE_INTERVAL_S`; it makes no CA call
        and opens no connection for it, so it cannot ask the server whether it
        serves a name.

        A name is held down when it connected in this process and is down now:
        its subscription record has been down longer than
        :data:`STALE_HOLD_DOWN_S`, or it is a one-shot channel that failed to
        connect within :data:`STALE_ONE_SHOT_WINDOW_S` and is still not
        connected. A held-down name is stale when

        * some channel of this connector connected newly after it went down --
          a subscription record's connection came up, or a one-shot channel
          connected for the first time or for the first time since it failed --
          and it is still down :data:`STALE_CONFIRM_S` later: the transport
          carries again, and libca has not caught up; or
        * it is a subscription record (a one-shot failure is held down only
          for :data:`STALE_ONE_SHOT_WINDOW_S`, so only the first rule confirms
          it) and has been down :data:`STALE_SILENT_S` with no such connect: nothing
          here can tell a server that is gone from a connection libca will not
          rebuild, so a long silence counts as stale. A server that really is
          gone costs one fresh connector, not a loop: see below.

        A name that never connected in this process never counts: nothing here
        can tell a name nobody serves from one the transport fails to reach,
        and a successor of a stale process only counts names it connected
        itself. So a process in which nothing has connected yet -- started
        during an outage, or the successor of a stale exit while the server
        is still gone -- never reports stale; it recovers only if libca
        reconnects to the name server by itself. Without a name server (UDP search) the server's
        beacons make libca search again on a restart, and this waits for ever,
        like the base class. A caller that is shutting down cancels it.

        A stale transport is a lost connector: libca will not rebuild it, so
        every channel it held fails alike until a fresh connector replaces this
        one. It is raised as :class:`ConnectorUnavailableError`, whose text
        names the stale channels and why they count, and the supervisor that
        owns this connector hands that to the recovery policy
        (:mod:`osprey_connectors.recovery`) like any other lost connector.
        """
        if self._name_server is None:
            await asyncio.get_running_loop().create_future()
        while True:
            stale = await self._ca.run(self._stale_now())
            if stale:
                names = sorted(stale)
                reasons = ", ".join(sorted(set(stale.values())))
                logger.warning("EPICS transport stale: %s still down here (%s)", names, reasons)
                raise ConnectorUnavailableError(
                    f"EPICS Channel Access transport is stale: {', '.join(names)} still down "
                    f"here ({reasons}); only a fresh connector in a fresh process reconnects"
                )
            await asyncio.sleep(STALE_PROBE_INTERVAL_S)

    def channel_key(self, address: str) -> str:
        """``ca://X``, ``X.VAL`` and ``X`` are one record's value field: key ``X``."""
        return normalise_address(address.strip()).removesuffix(".VAL")

    def _one_shot_states(self) -> dict[str, bool]:
        """Connected state of every channel in aioca's cache, by name. CA loop.

        A channel absent from the cache (never opened, or purged by a repoint)
        is absent here, and so is not held down.
        """
        try:
            infos = self._aioca.get_channel_infos()
        except Exception:
            return {}
        return {info.name: bool(info.connected) for info in infos}

    async def _held_down(self) -> dict[str, float]:
        """Each name held down right now, with when it went down (see :meth:`wait_stale`). CA loop."""
        now = time.monotonic()
        held: dict[str, float] = {}
        if self._bulk is not None:
            for name, connected, down_since, ever_connected in self._bulk.snapshot():
                if connected or down_since is None or not ever_connected:
                    continue
                if now - down_since <= STALE_HOLD_DOWN_S:
                    continue
                held[name] = down_since
        states = self._one_shot_states()
        for name, failed_at in self._connect_failures.items():
            if now - failed_at > STALE_ONE_SHOT_WINDOW_S:
                continue
            if states.get(name, True):
                continue
            if name not in self._one_shot_connected:
                continue
            held.setdefault(name, failed_at)
        return held

    def _last_connect_at(self) -> float | None:
        """When a channel of this connector last connected newly; None if none has. CA loop."""
        times = [self._one_shot_connect_at]
        if self._bulk is not None:
            times.append(self._bulk.last_connect_at)
        known = [t for t in times if t is not None]
        return max(known) if known else None

    async def _stale_now(self) -> dict[str, str]:
        """Each held-down name that is stale now, with why (see :meth:`wait_stale`). CA loop."""
        now = time.monotonic()
        last = self._last_connect_at()
        stale: dict[str, str] = {}
        for name, down_since in (await self._held_down()).items():
            if last is not None and last > down_since:
                if now - last >= STALE_CONFIRM_S:
                    stale[name] = "a channel connected since it went down"
            elif now - down_since >= STALE_SILENT_S:
                stale[name] = f"down {STALE_SILENT_S:g}s with no channel connecting"
        return stale

    def _subscribe_pva(
        self,
        channel_address: str,
        callback: Callable[[ChannelValue], None],
        loop: asyncio.AbstractEventLoop,
        on_connection: Callable[[bool], None] | None = None,
    ) -> str:
        """Open a p4p monitor and adapt its updates to the connector callback.

        p4p delivers monitor updates on its own worker thread, and hands the
        callback whatever the subscription queue holds — including exception
        instances (``Disconnected``, ``RemoteError``, ``Finished``) when the
        event is connection state rather than data. Those are filtered exactly
        the way p4p's own dispatcher classifies them: a connection event is not
        a channel value and must never reach the subscriber as one.

        A mapping failure (an NTNDArray in a codec the connector cannot decode,
        say) drops that one update with a log line. Raising instead would
        propagate into p4p's dispatcher, which responds by closing the
        subscription outright — one bad frame would silently end the monitor.

        The monitor is opened with ``notify_disconnect=True``, so p4p reports
        every loss of the channel as a ``Disconnected`` update instead of
        going silent. Any exception update means the channel is down:
        ``on_connection`` is told ``False`` (once, and only after the channel
        had connected — a channel that never connects reports nothing) and
        the update never becomes a value. The monitor is never closed or
        re-created on a drop; p4p re-establishes it itself, and the first value
        after that is preceded by ``on_connection(True)``. Every report is
        posted to the loop ahead of anything that comes after it.

        The first value update after each (re)connection is also the
        subscription's metadata source (:class:`_PvaMetadataSlot`): it settles
        any :meth:`get_metadata` call waiting on this address. A drop empties
        the slot again.
        """
        context = self._require_pva_context(channel_address, "subscribe to")

        connected = False
        slot = _PvaMetadataSlot(channel_address)

        def pva_callback(update: Any) -> None:
            """Wrapper to convert a p4p monitor update to our format."""
            nonlocal connected
            if isinstance(update, Exception):
                # Connection state, never data: the channel is down until a
                # value arrives again, and so is the metadata it carried.
                logger.debug(f"PVA monitor event for '{channel_address}': {update!r}")
                with slot.lock:
                    slot.metadata = None
                if connected and on_connection is not None:
                    loop.call_soon_threadsafe(on_connection, False)
                connected = False
                return
            try:
                channel_value = self._pva_channel_value(
                    channel_address, getattr(update, "raw", update)
                )
            except Exception as exc:
                logger.warning(f"Dropping PVA update for '{channel_address}': {exc}")
                return
            if not connected and on_connection is not None:
                loop.call_soon_threadsafe(on_connection, True)
            connected = True
            waiters: list[asyncio.Future] = []
            with slot.lock:
                if slot.metadata is None:
                    slot.metadata = channel_value.metadata
                    waiters, slot.waiters = slot.waiters, []
            slot.settle(waiters, channel_value.metadata)
            # Schedule callback in event loop
            loop.call_soon_threadsafe(callback, channel_value)

        subscription = context.monitor(channel_address, pva_callback, notify_disconnect=True)

        sub_id = f"{channel_address}_{id(subscription)}"
        self._subscriptions[sub_id] = _ChannelSubscription("pva", subscription)
        self._pva_metadata_slots[sub_id] = slot

        logger.debug(f"PVA subscription created: {sub_id}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from channel changes, whichever transport opened them."""
        subscription = self._subscriptions.pop(subscription_id, None)
        if subscription is None:
            return
        subscription.close()
        slot = self._pva_metadata_slots.pop(subscription_id, None)
        if slot is not None:
            # A get_metadata parked on this subscription falls back to a get.
            with slot.lock:
                waiters, slot.waiters = slot.waiters, []
            slot.settle(waiters, None)
        logger.debug(f"EPICS subscription removed: {subscription_id}")

    def _read_metadata_pva(self, channel_address: str, timeout: float) -> ChannelMetadata:
        """Metadata-only PVAccess get (runs in a thread pool, like the reads).

        Asks the server for :data:`_PVA_METADATA_REQUEST` only, so a metadata
        lookup on a camera channel costs a few fields instead of a whole frame,
        and maps the reply with the same helpers the full read uses — the
        payload branches are skipped entirely, because the reply carries no
        payload to branch on. Failures translate the same way as
        :meth:`_read_channel_pva` — both go through :meth:`_pva_get`.

        One consequence: the returned metadata carries no ``enum_labels``. The
        NTEnum choices live under ``value``, which this request deliberately
        does not ask for, and the label for "the current value" is meaningless
        without a value anyway. Enum labels come from a read.
        """
        value = self._pva_get(channel_address, timeout, request=_PVA_METADATA_REQUEST)
        raw_metadata: dict[str, Any] = {"nt_id": _pva_type_id(value)}
        raw_metadata.update(self._pva_alarm_metadata(value))
        return self._pva_metadata(value, self._pva_timestamp(value), raw_metadata)

    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        """Get metadata for a channel, over whichever transport serves it.

        A PVA address this connector is subscribed to answers from the
        subscription: the first update of its monitor is the metadata source,
        and the call awaits that update for up to ``timeout_s``, then raises
        :class:`TimeoutError` — it never opens a second request for a channel
        it is already watching. An unsubscribed PVA address asks for the
        metadata fields alone rather than reading the channel and discarding
        its value.

        A subscribed CA address answers from its subscription record's CTRL
        event (units, precision, enum labels, the limits, write access), awaited
        the same way; an unsubscribed CA address is read once, because a read
        carries units, precision and limits with the value.

        Raises:
            ChannelUnreachableError: A subscribed address delivered no update
                (PVA) or no control data (CA) within ``timeout_s``: it is
                down, or was never reached.
        """
        if self._is_pva_channel(channel_address):
            slots = [
                slot
                for slot in self._pva_metadata_slots.values()
                if slot.address == channel_address
            ]
            if slots:
                metadata = await self._await_subscribed_pva_metadata(channel_address, slots)
                if metadata is not None:
                    return metadata
                # Every subscription closed while we waited: no monitor to ask.
            return await asyncio.to_thread(self._read_metadata_pva, channel_address, self._timeout)

        if self._bulk is not None:
            bulk_metadata: ChannelMetadata | None = await self._ca.run(
                self._bulk.metadata(channel_address, self._timeout)
            )
            if bulk_metadata is not None:
                return bulk_metadata
        channel_value = await self.read_channel(channel_address)
        return channel_value.metadata

    async def _await_subscribed_pva_metadata(
        self, channel_address: str, slots: list[_PvaMetadataSlot]
    ) -> ChannelMetadata | None:
        """Metadata of a subscribed PVA address, awaiting its first update.

        Returns ``None`` when the subscriptions were closed while waiting (the
        caller then falls back to a metadata-only get); raises
        :class:`ChannelUnreachableError` when no update arrived within
        ``timeout_s``.
        """
        waiter: asyncio.Future = asyncio.get_running_loop().create_future()
        parked: list[_PvaMetadataSlot] = []
        try:
            for slot in slots:
                with slot.lock:
                    if slot.metadata is not None:
                        return slot.metadata
                    slot.waiters.append(waiter)
                parked.append(slot)
            try:
                return await asyncio.wait_for(waiter, self._timeout)
            except TimeoutError:
                raise ChannelUnreachableError(
                    channel_address,
                    f"PVA channel '{channel_address}' is subscribed but delivered no "
                    f"update within {self._timeout}s",
                ) from None
        finally:
            for slot in parked:
                with slot.lock:
                    if waiter in slot.waiters:
                        slot.waiters.remove(waiter)

    async def validate_channel(self, channel_address: str) -> bool:
        """
        Check if channel exists and is accessible.

        A PVA address is probed with the same metadata-only get
        :meth:`get_metadata` uses: reachability is a property of the channel,
        not of its payload, so a camera serving frames in a codec the connector
        cannot decode still validates as reachable.

        Args:
            channel_address: EPICS channel address

        Returns:
            True if channel can be accessed
        """
        timeout = self._timeout
        try:
            if self._is_pva_channel(channel_address):
                await asyncio.to_thread(self._read_metadata_pva, channel_address, timeout)
            else:
                await self.read_channel(channel_address, timeout=timeout)
            return True
        except Exception as e:
            logger.debug(f"Channel validation failed for {channel_address}: {e}")
            return False
