"""
EPICS control system connector: Channel Access through aioca, PVAccess through p4p.

Every Channel Access call runs on the process's one CA event loop
(:mod:`osprey_connectors.control_system.ca_loop`), whichever loop or thread
the connector was called from.
"""

import asyncio
import fnmatch
import operator
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

from osprey_connectors.config import config_flag, get_facility_timezone
from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
    is_readonly_run,
    values_match,
)
from osprey_connectors.control_system.ca_loop import CaLoop, current_ca_env, get_ca_loop
from osprey_connectors.control_system.call_timeout import call_timeout_s
from osprey_connectors.control_system.limits_validator import (
    DEFAULT_STEP_READ_TIMEOUT_SECONDS,
    c_numeral,
    step_read_timeout_seconds,
)
from osprey_connectors.logger import get_logger
from osprey_connectors.types import writes_enabled_key

logger = get_logger("epics_connector")


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
    """The exception a failed aioca call is raised as: a timeout, or a lost channel."""
    if getattr(failure, "errorcode", None) == _ECA_TIMEOUT:
        return TimeoutError(f"{action} '{channel_address}' timed out: {failure}")
    return ConnectionError(f"{action} '{channel_address}' failed: {failure}")


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


# Normative-type identifiers the PVA read path maps specially. Compared with a
# prefix match so a minor revision ("epics:nt/NTEnum:1.1") keeps its mapping
# instead of silently degrading to the generic scalar path.
_NT_ENUM_ID = "epics:nt/NTEnum:"
_NT_NDARRAY_ID = "epics:nt/NTNDArray:"

# pvRequest used whenever only a channel's metadata is wanted. Asking for the
# whole structure would pull the payload too — on an NTNDArray that is a full
# camera frame per call, and a frame in a codec the connector cannot decode
# would make a metadata lookup fail on a channel that is perfectly reachable.
_PVA_METADATA_REQUEST = "field(alarm,timeStamp,display)"


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
    Channel Access keeps the aioca ``Subscription``, PVAccess the p4p
    ``Subscription``; both are released with ``close()``. The aioca one
    belongs to the CA loop, so its release is handed to that loop
    (``ca_loop``) rather than run on whichever thread unsubscribes. Dispatching
    here is what lets ``unsubscribe`` and ``disconnect`` walk subscription ids
    without knowing which protocol opened them.

    A Channel Access subscription can also carry ``reopen``, registered with
    the CA loop so a repoint opens the monitor again on the new context and
    swaps it in as ``handle``. The release forgets it and closes whichever
    handle is current when it runs: both run on the loop thread, so a
    subscription closed before, during or after a repoint stays closed.

    ``close()`` is idempotent and best effort: closing twice is a no-op, and a
    handle that fails to release is logged rather than raised, so tearing down
    a whole connector cannot be derailed by one dead channel.
    """

    __slots__ = ("_closed", "ca_loop", "handle", "kind", "reopen")

    def __init__(
        self,
        kind: str,
        handle: Any,
        ca_loop: Any = None,
        reopen: Callable[[], None] | None = None,
    ) -> None:
        self.kind = kind
        self.handle = handle
        self.ca_loop = ca_loop
        self.reopen = reopen
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
        """Forget the re-opener and close the current handle; on the CA loop."""
        try:
            if self.reopen is not None:
                self.ca_loop.discard_reopener(self.reopen)
            if self.handle is not None:
                self.handle.close()
        except Exception as exc:  # teardown is best effort, never fatal
            logger.debug(f"Subscription teardown failed for {self.kind} handle: {exc}")


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
        # Per channel, the CTRL-form reading that carries units, precision and
        # enum labels: fetched on a channel's first read, then reused, since
        # they describe the record rather than the reading. Touched only on
        # the CA loop.
        self._ca_ctrl: dict[str, Any] = {}
        self._ca_ctrl_fetches: set[Any] = set()
        self._epics_configured = False
        # PVA routing state. Empty globs => this connector is pure Channel
        # Access: p4p is never imported and no PVA environment variable is set.
        self._pva_channel_globs: list[str] = []
        self._p4p: Any = None
        self._pva_context: Any = None

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
        # writes, so a deployment that arms writes for this connector's type
        # must route through the write-capable gateway. Defense-in-depth: only
        # use write_access when this type is armed, so a type left unarmed also
        # has its writes rejected at the network layer (reinforcing the posture
        # this connector runs under).
        gateways = config.get("gateways", {})
        # One posture rule, shared with the reference monitor: the per-type
        # block when the factory stamped a type, the deployment-wide key when
        # it did not, and never armed during a readonly run.
        try:
            writes_enabled = self._writes_enabled
        except (FileNotFoundError, KeyError, RuntimeError):
            writes_enabled = False  # Can't tell -> assume the safe (read-only) path

        # A readonly sandbox run stays on the read_only gateway even where the
        # type is armed: the per-run claim is enforced at the network layer, so
        # a raw caput issued after read_channel() in the same process is
        # rejected by the gateway rather than trusted.
        readonly_run = is_readonly_run()
        write_gateway = gateways.get("write_access") or {}
        gateway_role = "read_only"
        if writes_enabled and write_gateway:
            gateway_role = "write_access"
            gateway_config = write_gateway
            logger.debug("EPICS connector: routing through write_access gateway (writes enabled)")
        else:
            gateway_config = gateways.get("read_only", {})
            if readonly_run and write_gateway:
                logger.debug("EPICS connector: readonly run, staying on read_only gateway")
            elif writes_enabled and not write_gateway:
                posture_key = writes_enabled_key(self._connector_type)
                logger.warning(
                    f"{posture_key} is true but no gateways.write_access "
                    "is configured; routing through the read_only gateway, which may "
                    "reject writes. Configure gateways.write_access to enable hardware writes."
                )

        # The EPICS_CA_* settings this connector needs, planned against the
        # environment as it stands.
        ca_env = current_ca_env()
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
            self._pva_context = self._p4p.client.thread.Context("pva")
            logger.debug(f"PVA routing enabled for {len(self._pva_channel_globs)} glob pattern(s)")

        # Initialize limits validator for automatic validation and confirm policy
        from osprey_connectors.control_system.limits_validator import LimitsValidator

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
        threads, and a ``ConnectionError`` invalidates and rebuilds the
        connector in production — a context left open on every invalidation
        leaks those threads for the life of the process. The close is
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
            ConnectionError: If channel cannot be connected
            TimeoutError: If operation times out
            ValueError: If a PVA channel serves a compressed NTNDArray
        """
        timeout = timeout or self._timeout

        if self._is_pva_channel(channel_address):
            # p4p's thread client blocks, so it runs in a worker thread.
            return await asyncio.to_thread(self._read_channel_pva, channel_address, timeout)

        return await self._ca.run(self._read_channel_ca(channel_address, timeout))

    # ------------------------------------------------------------------
    # Channel Access (aioca). Every coroutine here runs on the CA loop.
    # ------------------------------------------------------------------

    async def _read_channel_ca(self, channel_address: str, timeout: float) -> ChannelValue:
        """Read a channel's value, alarm state and timestamp from the IOC.

        Connecting is a step of its own, so a channel nobody answers for is
        reported as the connection failure it is, apart from a get that timed
        out on a channel that did connect. Every read asks the IOC: there is
        no monitor cache to answer from, so no reading can be stale.
        """
        aioca = self._aioca
        connected = await aioca.connect(channel_address, timeout=timeout, throw=False)
        if not connected:
            raise ConnectionError(
                f"Failed to connect to PV '{channel_address}' (timeout after {timeout}s)"
            )
        reading = await aioca.caget(
            channel_address, format=aioca.FORMAT_TIME, timeout=timeout, throw=False
        )
        if isinstance(reading, aioca.CANothing):
            raise _ca_failure(channel_address, reading, "Read of")
        ctrl = await self._ca_ctrl_fields(channel_address, timeout)
        return self._ca_channel_value(reading, ctrl)

    async def _ca_ctrl_fields(self, channel_address: str, timeout: float) -> Any:
        """The channel's CTRL-form reading (units, precision, enum labels), or ``None``.

        Fetched once per channel and reused: these fields describe the record,
        not the reading. A fetch that fails is logged and not cached, so the
        next read tries again; the reading it was meant to enrich still stands.
        """
        cached = self._ca_ctrl.get(channel_address)
        if cached is not None:
            return cached
        aioca = self._aioca
        ctrl = await aioca.caget(
            channel_address, format=aioca.FORMAT_CTRL, timeout=timeout, throw=False
        )
        if isinstance(ctrl, aioca.CANothing):
            logger.debug("Could not fetch control fields for '%s': %s", channel_address, ctrl)
            return None
        self._ca_ctrl[channel_address] = ctrl
        return ctrl

    def _ca_channel_value(self, reading: Any, ctrl: Any) -> ChannelValue:
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
        )
        return ChannelValue(value=value, timestamp=timestamp, metadata=metadata)

    async def _put_ca(
        self, channel_address: str, value: Any, wait: bool, timeout: float
    ) -> tuple[str, Any]:
        """Put ``value``; say whether it was sent, acknowledged, or refused.

        Answers one of:

        - ``("not_sent", failure)``: the channel never connected, so nothing
          left this process.
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
        connected = await aioca.connect(channel_address, timeout=timeout, throw=False)
        if not connected:
            return "not_sent", connected
        # The channel's own field type decides what goes out. cainfo reads it
        # from the channel just connected, without a round trip to the IOC.
        info = await aioca.cainfo(channel_address, timeout=timeout, throw=False)
        if isinstance(info, aioca.CANothing):
            return "not_sent", info
        try:
            put_value, as_text = _ca_put_form(value, info.datatype, info.count)
        except _Unwritable as reason:
            return "unwritable", reason
        datatype = aioca.DBR_CHAR_STR if as_text else None
        sent = value if as_text else put_value
        done = await aioca.caput(
            channel_address, put_value, datatype=datatype, wait=wait, timeout=timeout, throw=False
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
        raise ConnectionError(f"Write to '{channel_address}' failed: {done}")

    # ------------------------------------------------------------------
    # PVAccess read path
    # ------------------------------------------------------------------

    def _pva_connection_error_types(self) -> tuple[type[BaseException], ...]:
        """p4p exceptions that mean "the channel is not reachable".

        Looked up through ``self._p4p`` (never a module-scope import) and
        filtered to real exception classes, so a p4p release that drops or
        renames one of them degrades to "not caught here" instead of crashing
        the read with a TypeError inside the except clause.
        """
        thread_module = getattr(getattr(self._p4p, "client", None), "thread", None)
        found: list[type[BaseException]] = []
        for name in ("Disconnected", "RemoteError", "Cancelled"):
            candidate = getattr(thread_module, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                found.append(candidate)
        return tuple(found)

    def _pva_get(self, channel_address: str, timeout: float, request: str | None = None) -> Any:
        """Blocking p4p get, returning the raw p4p Value (runs in a thread pool).

        Failures are re-raised as stdlib ``ConnectionError`` / ``TimeoutError``
        so the MCP error envelope and the connector-invalidation logic treat a
        PVA outage exactly like a CA one. p4p's own ``TimeoutError`` already IS
        the builtin; ``Disconnected`` (a RuntimeError) is not, and would
        otherwise escape as an unclassified error.

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
        context = self._pva_context
        if context is None:
            raise ConnectionError(
                f"Cannot read PVA channel '{channel_address}': no PVA client context. "
                "connect() builds it from control_system.connector.epics.pva_channels."
            )

        try:
            if request is None:
                result = context.get(channel_address, timeout=timeout)
            else:
                result = context.get(channel_address, request=request, timeout=timeout)
        except TimeoutError:
            raise TimeoutError(
                f"Failed to read PVA channel '{channel_address}' (timeout after {timeout}s)"
            ) from None
        except self._pva_connection_error_types() as exc:
            raise ConnectionError(
                f"Failed to connect to PVA channel '{channel_address}': {exc}"
            ) from exc

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
        """Map the NT ``display`` and ``alarm`` fields onto :class:`ChannelMetadata`.

        Reads nothing but metadata fields, so it maps a full read and a
        metadata-only get (:data:`_PVA_METADATA_REQUEST`) identically — the
        payload branches contribute only what they put in ``raw_metadata``.
        """
        display = _pva_field(value, "display")
        units = _pva_field(display, "units", "") or ""
        precision = _pva_field(display, "precision")
        description = _pva_field(display, "description") or None
        display_low = _pva_field(display, "limitLow")
        display_high = _pva_field(display, "limitHigh")

        return ChannelMetadata(
            units=str(units),
            precision=int(precision) if isinstance(precision, int | float) else None,
            alarm_status=raw_metadata.get("alarm_message") or None,
            alarm_severity=_severity_int(raw_metadata.get("severity")),
            timestamp=timestamp,
            description=str(description) if description else None,
            display_low=float(display_low) if isinstance(display_low, int | float) else None,
            display_high=float(display_high) if isinstance(display_high, int | float) else None,
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

    def _current_value_reader(self) -> Callable[[str], Any] | None:
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
                reading = self._ca.run_sync(
                    aioca.caget(channel_address, timeout=budget, throw=False),
                    timeout=budget + 1.0,
                )
            except Exception as exc:  # the CA loop did not answer in time
                logger.debug(f"max_step read of '{channel_address}' failed: {exc}")
                return None
            if isinstance(reading, aioca.CANothing):
                return None
            return _ca_plain(reading)

        return read_current

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
            ConnectionError: If channel cannot be connected
            TimeoutError: If operation times out
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
                refusal_reason="VALIDATION_ERROR",
                error_message=(
                    f"Write to '{channel_address}' refused: PVAccess writes are not supported. "
                    "This address matches a configured 'pva_channels' pattern, so it is routed "
                    "over PVAccess and is read-only in this deployment. No write was attempted."
                ),
            )

        timeout = timeout or self._timeout

        # Step 1: Resolve the channel's confirmation policy when the caller has
        # no opinion (cheap, on-loop). An explicit False is an answer and is
        # never re-resolved.
        if confirm is None:
            confirm = self._resolve_confirm(channel_address)

        # Import here to avoid circular dependency
        from osprey_connectors.errors import ChannelLimitsViolationError

        # Step 2: Validate limits (FAIL CLOSED) in a worker thread, so a caller
        # on the event loop is never stalled by the blocking read that max_step
        # validation waits for.
        def _validate():
            if self._limits_validator:
                try:
                    self._limits_validator.validate(
                        channel_address, value, read_current=self._current_value_reader()
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

        validation_error = await asyncio.to_thread(_validate)
        if validation_error is not None:
            return self._validation_refusal(channel_address, value, validation_error)

        # Step 3: The put. The put-callback is Channel Access's acknowledgement
        # that the IOC processed the put, so a confirming write waits for it:
        # the read that follows would otherwise race the record's own
        # processing. A write nobody asked to confirm does not wait.
        put_result, payload = await self._ca.run(
            self._put_ca(channel_address, value, wait=bool(confirm), timeout=timeout)
        )

        if put_result == "cs_refused":
            # The control system was asked and said no: access security, not
            # a client bug, and nothing was written.
            logger.warning(f"Control system refused write to {channel_address}: {payload}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.REFUSED,
                refusal_reason="CONTROL_SYSTEM_REFUSED",
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
                refusal_reason="VALIDATION_ERROR",
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
                    f"Failed to write to channel '{channel_address}': it did not connect "
                    f"within {timeout}s, so no value was sent"
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
        """
        return await self._ca.run(self._read_channel_ca(channel_address, timeout))

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        """Read multiple channels concurrently."""
        return await self._read_concurrently(channel_addresses, timeout)

    async def subscribe(
        self, channel_address: str, callback: Callable[[ChannelValue], None]
    ) -> str:
        """
        Subscribe to channel value changes.

        Args:
            channel_address: EPICS channel address
            callback: Function to call when value changes

        Returns:
            Subscription ID for later unsubscription

        Routing mirrors the read path: a PVA-routed address opens a p4p
        monitor, every other address a Channel Access monitor. Either way the
        subscriber callback runs on the event loop, never on the transport's
        own worker thread.

        A Channel Access monitor outlives a repoint: when a connect elsewhere in
        the process moves CA to another gateway, the monitor is opened again on
        the new context, and its first update there reaches the callback like
        any other. Cancelling the call leaves no monitor open behind it.
        """
        loop = asyncio.get_running_loop()

        if self._is_pva_channel(channel_address):
            return self._subscribe_pva(channel_address, callback, loop)

        aioca = self._aioca
        subscription = _ChannelSubscription("ca", None, ca_loop=self._ca)

        def ca_callback(reading: Any) -> None:
            """Adapt one aioca monitor update; runs on the CA loop.

            A disconnect arrives as a ``CANothing`` rather than a reading: it is
            connection state, not a channel value, so it never reaches the
            subscriber. Units and enum labels come from the channel's CTRL
            fields, fetched in the background when the monitor opens: an update
            that lands first carries the index alone rather than waiting.
            """
            if isinstance(reading, aioca.CANothing):
                logger.debug(f"CA monitor event for '{channel_address}': {reading}")
                return
            try:
                channel_value = self._ca_channel_value(reading, self._ca_ctrl.get(channel_address))
            except Exception as exc:
                logger.warning(f"Dropping CA update for '{channel_address}': {exc}")
                return
            try:
                loop.call_soon_threadsafe(callback, channel_value)
            except RuntimeError:
                # The subscriber's loop has closed (osprey.runtime runs one per
                # call). Raised here, aioca would print it to stdout -- a stdio
                # MCP server's protocol stream -- and drop the monitor; close
                # the subscription here instead, quietly, so a repoint does not
                # re-open it either.
                logger.debug(f"Closing CA monitor on '{channel_address}': its loop has closed")
                subscription.close()

        def open_monitor() -> None:
            # On the CA loop: aioca binds a monitor to the loop it is opened on.
            subscription.handle = aioca.camonitor(
                channel_address, ca_callback, format=aioca.FORMAT_TIME, notify_disconnect=True
            )

        async def open_first() -> None:
            # Closed already: the caller gave up waiting before the loop got
            # here, and its release has run or is queued behind this.
            if subscription.closed:
                return
            # The task is held until it finishes, so it is not collected mid-fetch.
            fetch = asyncio.ensure_future(self._ca_ctrl_fields(channel_address, self._timeout))
            self._ca_ctrl_fetches.add(fetch)
            fetch.add_done_callback(self._ca_ctrl_fetches.discard)
            open_monitor()
            # A repoint purges every aioca channel without telling the monitor's
            # callback; the CA loop re-opens this one on the new context.
            subscription.reopen = open_monitor
            self._ca.add_reopener(open_monitor)

        try:
            await self._ca.run(open_first())
        except asyncio.CancelledError:
            # A caller that bounds the open (the dispatch trigger source does)
            # cancels it here, but the open may already be queued on the CA
            # loop, or done with its result still in flight. Closing hands the
            # release to that loop, behind the open: no monitor is left firing
            # with nobody to unsubscribe it.
            subscription.close()
            raise

        sub_id = f"{channel_address}_{id(subscription)}"
        self._subscriptions[sub_id] = subscription

        logger.debug(f"EPICS subscription created: {sub_id}")
        return sub_id

    def _subscribe_pva(
        self,
        channel_address: str,
        callback: Callable[[ChannelValue], None],
        loop: asyncio.AbstractEventLoop,
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
        """
        context = self._pva_context
        if context is None:
            raise ConnectionError(
                f"Cannot subscribe to PVA channel '{channel_address}': no PVA client context. "
                "connect() builds it from control_system.connector.epics.pva_channels."
            )

        def pva_callback(update: Any) -> None:
            """Wrapper to convert a p4p monitor update to our format."""
            if isinstance(update, Exception):
                logger.debug(f"PVA monitor event for '{channel_address}': {update!r}")
                return
            try:
                channel_value = self._pva_channel_value(
                    channel_address, getattr(update, "raw", update)
                )
            except Exception as exc:
                logger.warning(f"Dropping PVA update for '{channel_address}': {exc}")
                return
            # Schedule callback in event loop
            loop.call_soon_threadsafe(callback, channel_value)

        subscription = context.monitor(channel_address, pva_callback)

        sub_id = f"{channel_address}_{id(subscription)}"
        self._subscriptions[sub_id] = _ChannelSubscription("pva", subscription)

        logger.debug(f"PVA subscription created: {sub_id}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from channel changes, whichever transport opened them."""
        subscription = self._subscriptions.pop(subscription_id, None)
        if subscription is None:
            return
        subscription.close()
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

        The PVA path asks for the metadata fields alone rather than reading the
        channel and discarding its value; the CA path keeps reading the channel,
        because a read carries units and precision with the value.
        """
        if self._is_pva_channel(channel_address):
            return await asyncio.to_thread(self._read_metadata_pva, channel_address, self._timeout)

        channel_value = await self.read_channel(channel_address)
        return channel_value.metadata

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
