"""
EPICS control system connector using pvapy.

Provides interface to both EPICS transports — Channel Access (CA) and PVAccess
(PVA) — through ONE client library: pvapy (``import pvaccess``). Every channel
is a ``pvaccess.Channel(address, provider)`` whose provider is ``pvaccess.CA``
or ``pvaccess.PVA``; which one an address gets is decided by the configured
``pva_channels`` globs, exactly as before.

One client for both transports is what lets one mapping serve both: pvapy
answers a Channel Access get with the same normative-type shape a PVAccess
server sends (``value``, ``alarm``, ``timeStamp``, ``display``; an enum as
``value: {index, choices}``), so a reading means the same thing whichever
protocol served it.
"""

import asyncio
import concurrent.futures
import fnmatch
import os
import queue
import re
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from osprey_connectors.config import get_facility_timezone
from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
    is_readonly_run,
    values_match,
)
from osprey_connectors.control_system.limits_validator import (
    DEFAULT_STEP_READ_TIMEOUT_SECONDS,
    step_read_timeout_seconds,
)
from osprey_connectors.errors import ClientEndpointConflictError
from osprey_connectors.logger import get_logger
from osprey_connectors.types import writes_enabled_key

logger = get_logger("epics_connector")

_T = TypeVar("_T")


# ----------------------------------------------------------------------
# pvRequests
# ----------------------------------------------------------------------

# A Channel Access read. pvapy's CA provider cannot return ``timeStamp``
# together with ``display``/``control``: asking for all of them silently drops
# the timestamp (and so does a bare ``field()`` on an ao). The value, its alarm
# and its timestamp are therefore one request — an enum's choices come with
# ``value`` — and the display metadata is a second, cached one
# (:data:`_CA_DISPLAY_REQUEST`).
_CA_READ_REQUEST = "field(value,alarm,timeStamp)"

# The display metadata (units, format, description, display limits) of a CA
# channel. Static per record in practice, so it is fetched once per channel and
# cached; see :meth:`EPICSConnector._ca_display`.
_CA_DISPLAY_REQUEST = "field(display)"

# A PVAccess read: the server's whole structure, which for a normative type
# carries value, alarm, timeStamp and display together.
_PVA_READ_REQUEST = "field()"

# pvRequest used whenever only a PVA channel's metadata is wanted. Asking for
# the whole structure would pull the payload too — on an NTNDArray that is a
# full camera frame per call, and a frame in a codec the connector cannot
# decode would make a metadata lookup fail on a channel that is perfectly
# reachable.
_PVA_METADATA_REQUEST = "field(alarm,timeStamp,display)"

# The just-the-value read the ``max_step`` check makes before a write.
_VALUE_REQUEST = "field(value)"

# A put that returns only once the IOC's put-callback has fired — Channel
# Access's acknowledgement that the record processed the put. pvapy's plain
# ``put`` does not wait for it, and ``record[block=true]`` without a
# ``field()`` clause is rejected, hence the whole string. Note that
# ``Channel.setTimeout`` does NOT bound this put (a put-callback that takes
# 1.5 s completes under a 0.5 s channel timeout), so the connector enforces
# its own deadline around it; see :meth:`EPICSConnector._put`.
_CONFIRMING_PUT_REQUEST = "record[block=true]field(value)"

#: How long a monitor waits before retrying a display fetch that failed.
_DISPLAY_RETRY_S = 5.0

# What a monitor asks for, per transport, for the same reason as the reads.
_CA_MONITOR_REQUEST = _CA_READ_REQUEST
_PVA_MONITOR_REQUEST = _PVA_READ_REQUEST


# ----------------------------------------------------------------------
# pvapy error classification
# ----------------------------------------------------------------------

# pvapy raises ONE exception type, ``pvaccess.PvaException``, for everything —
# an unreachable channel, an access-security denial, a malformed value — so the
# only thing that tells them apart is the message text. These are the two
# texts the connector acts on, pinned against a real IOC by
# tests/connectors/test_epics_soft_ioc.py so a pvapy upgrade that rewords them
# fails there rather than silently changing what an outcome means:
#
#   "Channel SPK:NOPE timed out."                           (connect or get)
#   "channel SPK:SP PvaClientPut::put Write access denied"  (access security)
_TIMED_OUT_PATTERN = re.compile(r"^Channel \S+ timed out\.$")
_ACCESS_DENIED_TEXT = "Write access denied"

#: :func:`_classify_client_error` verdicts.
_UNREACHABLE = "unreachable"
_ACCESS_DENIED = "access_denied"


def _classify_client_error(exc: BaseException, error_type: Any) -> str | None:
    """Name what a pvapy failure means, or ``None`` when it is not recognized.

    ``error_type`` is ``pvaccess.PvaException``, passed in by the connector
    that imported pvapy (this module never imports it at module scope). A
    failure of any other type — a Boost ``ArgumentError`` for a value pvapy
    cannot convert, say — is never classified, whatever its text.

    Returns:
        :data:`_UNREACHABLE` for a channel that did not connect or answer in
        time, :data:`_ACCESS_DENIED` for a put the control system's access
        security refused, and ``None`` for everything else. ``None`` is the
        safe answer: the caller re-raises, so an unrecognized failure stays an
        error whose outcome is unknown — it is never reported as a refusal,
        which would claim that nothing was written.
    """
    if not isinstance(error_type, type) or not isinstance(exc, error_type):
        return None
    message = str(exc).strip()
    if _TIMED_OUT_PATTERN.match(message):
        return _UNREACHABLE
    if _ACCESS_DENIED_TEXT in message:
        return _ACCESS_DENIED
    return None


# ----------------------------------------------------------------------
# Mapping helpers (shared by both transports)
# ----------------------------------------------------------------------

# Normative-type kinds the read path maps specially, recorded as
# ``raw_metadata["nt_type"]``. pvapy exposes no normative-type id without
# rendering the whole value to text (a full camera frame, for an NTNDArray), so
# the kind is recognized from the structure's shape instead: see
# :func:`_nt_kind`.
_NT_ENUM = "NTEnum"
_NT_NDARRAY = "NTNDArray"


def _nt_kind(structure: Any) -> str | None:
    """Recognize an NTEnum or an NTNDArray from its introspection dict.

    ``structure`` is ``PvObject.getStructureDict()`` — field types only, never
    data, so this costs nothing on a camera channel. pvapy spells a union as a
    tuple and a sub-structure as a dict, which is enough to tell the two
    normative types that need special mapping apart from everything else:

    * NTNDArray — ``value`` is a union and ``codec`` and ``dimension`` are
      present (the fields the NTNDArray mapping reads).
    * NTEnum — ``value`` is a structure carrying ``index`` and ``choices``.
      Channel Access enums (mbbo/bo) arrive in exactly this shape too.

    Anything else is mapped as a scalar or scalar array: ``None``.
    """
    if not isinstance(structure, dict):
        return None
    value = structure.get("value")
    if isinstance(value, tuple) and "codec" in structure and "dimension" in structure:
        return _NT_NDARRAY
    if isinstance(value, dict) and "index" in value and "choices" in value:
        return _NT_ENUM
    return None


def _top_fields(obj: Any) -> dict[str, Any]:
    """The top-level fields of a pvapy ``PvObject`` as a plain dict.

    Each field is converted once, by pvapy's own ``obj[name]``: a sub-structure
    becomes a dict, a union a ``(value_dict, type_dict)`` tuple, an array a
    numpy array. Everything downstream works on plain Python values, so an
    optional field that a server does not send is simply a missing key. Going
    field by field rather than through ``toDict()`` keeps one field pvapy
    cannot convert from costing the reading all the others.
    """
    fields: dict[str, Any] = {}
    for name in obj.keys():
        try:
            fields[name] = obj[name]
        except Exception as exc:  # one unconvertible field must not lose the read
            logger.debug(f"Could not convert field '{name}': {exc}")
    return fields


def _sub(container: Any, name: str) -> dict[str, Any]:
    """A sub-structure as a dict, or ``{}`` when absent or not a structure."""
    if not isinstance(container, dict):
        return {}
    found = container.get(name)
    return found if isinstance(found, dict) else {}


def _union_value(union: Any) -> Any:
    """The selected member of a pvapy union, or ``None`` when nothing is selected.

    pvapy hands a union over as ``(value_dict, type_dict)``, where
    ``value_dict`` holds only the selected member (``{'ushortValue': array}``)
    — or nothing, for an empty union.
    """
    if isinstance(union, tuple) and union and isinstance(union[0], dict):
        return next(iter(union[0].values()), None)
    return None


_FORMAT_PRECISION = re.compile(r"\.(\d+)")


def _precision(display: dict[str, Any]) -> int | None:
    """Display precision, from a ``precision`` field or a ``format`` string.

    A PVA server may carry precision as a number, but pvapy's Channel Access
    provider (and pvapy's own NT servers) carry it only inside ``format``, as
    a FORTRAN-style edit descriptor: ``"F9.3"`` is 3 decimals, ``"F8.2"`` is 2,
    and ``"I12"`` — an integer format with no fractional part — reports none.
    """
    number = display.get("precision")
    if isinstance(number, int) and not isinstance(number, bool):
        return number
    fmt = display.get("format")
    if isinstance(fmt, str):
        match = _FORMAT_PRECISION.search(fmt)
        if match:
            return int(match.group(1))
    return None


def _finite_float(value: Any) -> float | None:
    """``float(value)`` for a real number, else ``None``."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


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


def _timestamp(fields: dict[str, Any]) -> datetime:
    """Channel timestamp from the NT ``timeStamp`` field, else now().

    Both providers report POSIX-epoch seconds (pvapy converts Channel Access's
    1990 EPICS epoch). A record that has never processed reports 0, which is
    "no timestamp", not 1970.
    """
    stamp = _sub(fields, "timeStamp")
    seconds = stamp.get("secondsPastEpoch") or 0
    nanoseconds = stamp.get("nanoseconds") or 0
    if isinstance(seconds, int | float) and seconds:
        nanos = nanoseconds if isinstance(nanoseconds, int | float) else 0
        return datetime.fromtimestamp(seconds + nanos * 1e-9, get_facility_timezone())
    return datetime.now(get_facility_timezone())


def _alarm_metadata(fields: dict[str, Any], provider: str) -> dict[str, Any]:
    """Alarm fields, recorded raw so nothing about the alarm state is lost.

    ``alarm.message`` is the alarm status by NAME: pvapy's Channel Access
    provider spells the CA status there (``'UDF'``, ``'HIHI'``), so no CA
    status-code table is needed. ``alarm.status`` is kept alongside it, but on
    CA it is the normative-type status (DEVICE, UNDEFINED, ...), not the CA
    code.

    One gap is filled: for a healthy CA record the provider sends an EMPTY
    message rather than ``'NO_ALARM'``. A CA reading with severity 0 and no
    message is therefore named ``'NO_ALARM'`` — the name the connector has
    always reported for it. PVA keeps an empty message as "not reported":
    a PVA server's message is free text, not a status name.
    """
    if "alarm" not in fields:
        return {}
    alarm = _sub(fields, "alarm")
    message = str(alarm.get("message") or "")
    if not message and provider == "ca" and alarm.get("severity") == 0:
        message = "NO_ALARM"
    return {
        "severity": alarm.get("severity"),
        "status": alarm.get("status"),
        "alarm_message": message,
    }


def _metadata(
    fields: dict[str, Any], timestamp: datetime, raw_metadata: dict[str, Any]
) -> ChannelMetadata:
    """Map the NT ``display`` and ``alarm`` fields onto :class:`ChannelMetadata`.

    Reads nothing but metadata fields, so it maps a full read and a
    metadata-only get (:data:`_PVA_METADATA_REQUEST`) identically — the
    payload branches contribute only what they put in ``raw_metadata``.
    """
    display = _sub(fields, "display")
    description = display.get("description") or None
    return ChannelMetadata(
        units=str(display.get("units") or ""),
        precision=_precision(display),
        alarm_status=raw_metadata.get("alarm_message") or None,
        alarm_severity=_severity_int(raw_metadata.get("severity")),
        timestamp=timestamp,
        description=str(description) if description else None,
        display_low=_finite_float(display.get("limitLow")),
        display_high=_finite_float(display.get("limitHigh")),
        raw_metadata=raw_metadata,
    )


def _scalar_value(fields: dict[str, Any]) -> dict[str, Any]:
    """Scalar / scalar array / unrecognized structure: the value field as-is."""
    scalar = fields.get("value")
    dtype = getattr(scalar, "dtype", None)
    return {
        "value": scalar,
        "raw_metadata": {"dtype": str(dtype) if dtype is not None else type(scalar).__name__},
    }


def _enum_value(fields: dict[str, Any]) -> dict[str, Any]:
    """An enum (NTEnum, or a CA mbbo/bo): the INDEX is the value.

    Both halves of a state reading reach the operator, and each in the place
    it belongs: ``value`` is the index, so a reading has one machine-readable
    type whichever protocol served it, and the choice string arrives as
    ``ChannelMetadata.enum_label`` — surfaced in the tool envelope, so nobody
    has to know that "2" means ACQUIRING.

    The index and the choices list stay in ``raw_metadata`` as well: they
    predate the typed fields and are what a PVA-specific consumer already
    reads.
    """
    enum_field = _sub(fields, "value")
    index = enum_field.get("index")
    choices = list(enum_field.get("choices") or [])
    labels, label = _enum_label_fields(choices, index)
    return {
        "value": index,
        "enum_labels": labels,
        "enum_label": label,
        "raw_metadata": {"dtype": "enum", "enum_index": index, "enum_choices": choices},
    }


def _color_mode(fields: dict[str, Any]) -> Any:
    """The areaDetector ColorMode NTAttribute, or None when absent.

    An attribute's ``value`` is a variant union, which pvapy spells
    ``({'value': 0}, {...})``.
    """
    for entry in fields.get("attribute") or []:
        if isinstance(entry, dict) and entry.get("name") == "ColorMode":
            return _union_value(entry.get("value"))
    return None


def _ndarray_value(channel_address: str, fields: dict[str, Any]) -> dict[str, Any]:
    """NTNDArray: the carried array, reshaped per its dimension list.

    The array is taken from the selected union member exactly as pvapy decoded
    it — never re-cast — so an unsigned frame stays unsigned instead of
    reading as negative pixels.
    """
    codec_name = str(_sub(fields, "codec").get("name") or "")
    dimensions = [dim.get("size") for dim in fields.get("dimension") or [] if isinstance(dim, dict)]

    if codec_name:
        # Never reshape a compressed payload: the union carries the compressed
        # byte blob, and reshaping it would produce a plausible image full of
        # meaningless pixel statistics.
        raise ValueError(
            f"Cannot read '{channel_address}': compressed NTNDArray unsupported; "
            f"disable ADPva compression for this channel "
            f"(codec={codec_name!r}, dimensions={dimensions})"
        )

    array = _union_value(fields.get("value"))
    raw_metadata: dict[str, Any] = {
        "dtype": str(getattr(array, "dtype", type(array).__name__)),
        "dimensions": dimensions,
        "color_mode": _color_mode(fields),
        "codec": codec_name,
        "unique_id": fields.get("uniqueId"),
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


def _channel_value(
    channel_address: str,
    obj: Any,
    provider: str,
    display: dict[str, Any] | None = None,
) -> ChannelValue:
    """Map one pvapy ``PvObject`` onto :class:`ChannelValue`, for either transport.

    Args:
        channel_address: The channel the object was read from (for messages).
        obj: What ``Channel.get`` or a monitor update delivered.
        provider: ``"ca"`` or ``"pva"``, recorded in ``raw_metadata``.
        display: Display metadata fetched separately — the Channel Access
            path, whose read cannot carry ``display`` alongside ``timeStamp``.
            Used only when the object itself carries no ``display``.
    """
    try:
        structure = obj.getStructureDict()
    except Exception:  # introspection is an optimization, never a precondition
        structure = None
    kind = _nt_kind(structure)
    fields = _top_fields(obj)
    if display and "display" not in fields:
        fields["display"] = display
    if provider == "ca" and kind is None and "value" in fields:
        value_type = structure.get("value") if isinstance(structure, dict) else None
        fields["value"] = _unsigned_chars(fields["value"], value_type)

    timestamp = _timestamp(fields)
    raw_metadata: dict[str, Any] = {"provider": provider, "nt_type": kind}
    raw_metadata.update(_alarm_metadata(fields, provider))

    if kind == _NT_NDARRAY:
        payload = _ndarray_value(channel_address, fields)
    elif kind == _NT_ENUM:
        payload = _enum_value(fields)
    else:
        payload = _scalar_value(fields)

    raw_metadata.update(payload.pop("raw_metadata", {}))
    metadata = _metadata(fields, timestamp, raw_metadata)
    # Only the enum branch contributes these; every other payload leaves the
    # fields at their None default, which is what marks a channel as not
    # enum-typed.
    metadata.enum_labels = payload.get("enum_labels")
    metadata.enum_label = payload.get("enum_label")
    return ChannelValue(value=payload["value"], timestamp=timestamp, metadata=metadata)


def _put_value(value: Any) -> Any:
    """Reduce a value to a type pvapy's ``Channel.put`` overloads accept.

    ``put`` is a set of C++ overloads over Python scalars, ``str`` and
    ``list``; a numpy array or a tuple matches none of them and raises a Boost
    argument error. numpy scalars and arrays both answer ``tolist()`` with the
    plain Python equivalent; a tuple becomes a list.
    """
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (str, bytes)):
        return tolist()
    return value


# pvapy's scalar types by name. ``str()`` of a ``pvaccess.ScalarType`` is the
# bare name (``"DOUBLE"``), which is what these sets hold.
_FLOAT_TYPES = frozenset({"DOUBLE", "FLOAT"})
_INTEGER_TYPES = frozenset({"BYTE", "UBYTE", "SHORT", "USHORT", "INT", "UINT", "LONG", "ULONG"})
# Channel Access has one 8-bit type, DBR_CHAR, which pvapy's CA provider hands
# over as a signed BYTE whatever the record's FTVL says.
_CHAR_TYPES = frozenset({"BYTE", "UBYTE"})


def _type_name(value_type: Any) -> str | None:
    """The scalar type name of an introspection entry, or ``None`` for a container."""
    if value_type is None or isinstance(value_type, dict | list | tuple):
        return None
    return str(value_type).rsplit(".", 1)[-1]


def _is_char_array(value_type: Any) -> bool:
    """True for a Channel Access char array (a ``waveform`` of FTVL CHAR or UCHAR)."""
    return (
        isinstance(value_type, list)
        and len(value_type) == 1
        and _type_name(value_type[0]) in _CHAR_TYPES
    )


class _UnwritableValue(ValueError):
    """The value cannot be represented in the channel's type; nothing was sent."""


class _NothingSent(ConnectionError):
    """A write that is KNOWN not to have left the connector.

    Raised by :meth:`EPICSConnector._put` when the channel never connected,
    when the write's deadline passed before the put was issued, or when an
    earlier put to the same channel was still waiting for the IOC. The caller
    reports it as ``FAILED`` — never ``UNCONFIRMED``, which would claim a
    value was sent.
    """


def _char_array_payload(payload: Any, nelm: int | None) -> list[int]:
    """What a Channel Access char array is written with: signed bytes.

    A ``str`` is encoded as UTF-8 and NUL-terminated, then cut to the record's
    ``NELM`` when that is known — what pyepics did for a string put to a char
    waveform. ``bytes`` and integer sequences are written as they are. Every
    element must fit a byte (-128..255); unsigned values above 127 go over the
    wire as their signed equivalent, since DBR_CHAR carries 8 bits either way
    and pvapy's CA provider only accepts the signed range.
    """
    if isinstance(payload, str):
        data: list[Any] = [*payload.encode("utf-8"), 0]
        if nelm:
            data = data[:nelm]
    elif isinstance(payload, bytes | bytearray):
        data = list(payload)
    elif isinstance(payload, list):
        data = payload
    else:
        data = [payload]
    signed: list[int] = []
    for element in data:
        if isinstance(element, float) and element.is_integer():
            element = int(element)
        if isinstance(element, bool) or not isinstance(element, int) or not -128 <= element <= 255:
            raise _UnwritableValue(f"{element!r} does not fit a char array element (-128..255)")
        signed.append(element - 256 if element > 127 else element)
    return signed


def _typed_payload(value: Any, value_type: Any, pvaccess: Any, nelm: int | None) -> Any:
    """What to hand ``Channel.put`` so the IOC receives exactly ``value``.

    ``value_type`` is the channel's ``value`` entry from
    ``Channel.getIntrospectionDict()`` (``None`` when unknown).

    pvapy's scalar put overloads convert a Python number to text with six
    significant digits before parsing it into the channel's type, so
    ``put(1.2345678901234567)`` stores 1.23457 and ``put(499654321.5)``
    stores 499654000. A PvObject is written exactly — but only when its type
    is the channel's own: a DOUBLE PvObject put to a ``longout`` succeeds and
    changes nothing. So a number goes out as:

    * floating-point channel — a PvObject of the channel's own type (exact);
    * integer or enum channel — a Python ``int`` (its text is exact), and a
      float with a fractional part is refused rather than truncated;
    * string channel — ``str(number)``, the shortest exact spelling;
    * char array — see :func:`_char_array_payload`.

    Everything else (text, bools, lists, an unknown type) goes out as before.

    Raises:
        _UnwritableValue: The value cannot be represented in the channel's type.
    """
    payload = _put_value(value)
    if _is_char_array(value_type):
        return _char_array_payload(payload, nelm)
    if isinstance(payload, bool) or not isinstance(payload, int | float):
        return payload
    name = _type_name(value_type)
    if name in _FLOAT_TYPES:
        return pvaccess.PvObject({"value": value_type}, {"value": float(payload)})
    if name == "STRING":
        return str(payload)
    if name in _INTEGER_TYPES or isinstance(value_type, dict):
        if isinstance(payload, float):
            if not payload.is_integer():
                raise _UnwritableValue(
                    f"{payload!r} is not a whole number, and the channel holds integers"
                )
            return int(payload)
    return payload


def _unsigned_chars(value: Any, value_type: Any) -> Any:
    """A Channel Access char value as the unsigned bytes pyepics reported.

    DBR_CHAR is ``epicsUInt8`` on the wire; pvapy's CA provider hands it over
    as a signed BYTE, so ``[200, 65, 255]`` would read as ``[-56, 65, -1]``.
    pyepics read FTVL CHAR and UCHAR alike as unsigned (``c_ubyte``), and so
    does this.
    """
    dtype = getattr(value, "dtype", None)
    if dtype is not None and str(dtype) == "int8" and hasattr(value, "view"):
        return value.view("uint8")
    if (
        _type_name(value_type) in _CHAR_TYPES
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return value & 0xFF
    return value


def _char_array_text(value: Any) -> str | None:
    """A char array's bytes as text, up to the first NUL; ``None`` if not a char array."""
    dtype = getattr(value, "dtype", None)
    if dtype is None or str(dtype) not in ("uint8", "int8") or not hasattr(value, "tobytes"):
        return None
    raw = bytes(value.view("uint8").tobytes()) if str(dtype) == "int8" else bytes(value.tobytes())
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


class _EpicsWorkers:
    """Daemon worker threads for every blocking pvapy call — threads that never exit.

    Why not ``asyncio.to_thread``: EPICS libCom registers per-thread state for
    any foreign (non-EPICS) thread that calls into it, and on macOS its
    thread-exit destructor fails (``pthread_attr_destroy ERROR Invalid
    argument``) and then *suspends the exiting thread forever*
    (``cantProceed``). Anything that joins such a thread hangs — a
    ``ThreadPoolExecutor`` shutdown, and so ``asyncio.run``'s
    ``shutdown_default_executor``, which is how a connector-host child or a
    sandbox script ends. Measured against pvapy 5.6.0 with a bare
    ``threading.Thread`` doing one CA get.

    This is a macOS defect. The same probe was verified clean on glibc
    Linux (amd64 and aarch64: the thread joins, the process exits 0), which is
    where production runs. The workers stay anyway, and must not be
    "optimized" back to ``asyncio.to_thread``: developers and a CI lane run
    on macOS, and on Linux they cost nothing. tests/connectors/
    test_epics_soft_ioc.py fails if a scenario's process does not exit
    promptly after disconnect().

    So pvapy is only ever called from these threads: daemons (the interpreter
    never joins them at exit) that loop forever (so the broken destructor
    never runs). One pool serves every connector in the process — threads
    that never exit are worth sharing — and grows on demand up to
    ``max_workers``, the same ceiling the default executor uses. A confirming
    put abandoned at its deadline keeps its worker until the IOC answers.
    """

    def __init__(self, max_workers: int) -> None:
        self._max_workers = max_workers
        self._queue: queue.SimpleQueue[
            tuple[concurrent.futures.Future[Any], Callable[..., Any], tuple[Any, ...]]
        ] = queue.SimpleQueue()
        self._idle = threading.Semaphore(0)
        self._lock = threading.Lock()
        self._threads = 0

    def submit(self, fn: Callable[..., Any], *args: Any) -> "concurrent.futures.Future[Any]":
        """Run ``fn(*args)`` on a worker; the returned future carries its outcome."""
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._queue.put((future, fn, args))
        if not self._idle.acquire(blocking=False):
            with self._lock:
                if self._threads < self._max_workers:
                    self._threads += 1
                    threading.Thread(
                        target=self._work, name=f"epics-worker-{self._threads}", daemon=True
                    ).start()
        return future

    def _work(self) -> None:
        _worker_local.active = True
        while True:
            future, fn, args = self._queue.get()
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn(*args))
                except BaseException as exc:  # handed to the awaiting caller
                    future.set_exception(exc)
            del future, fn, args  # drop references while idle
            self._idle.release()


_workers = _EpicsWorkers(max_workers=min(32, (os.cpu_count() or 1) + 4))


#: Marks :data:`_workers`' own threads (``active`` is set on each).
_worker_local = threading.local()


def _on_worker() -> bool:
    """True on one of :data:`_workers`' threads."""
    return bool(getattr(_worker_local, "active", False))


# ----------------------------------------------------------------------
# The endpoint each provider is bound to, per process
# ----------------------------------------------------------------------

#: The environment variables that name a provider's endpoint — the ones
#: ``connect()`` sets — keyed by provider (``"ca"`` / ``"pva"``).
_ENDPOINT_VARS: dict[str, tuple[str, ...]] = {
    "ca": (
        "EPICS_CA_ADDR_LIST",
        "EPICS_CA_SERVER_PORT",
        "EPICS_CA_NAME_SERVERS",
        "EPICS_CA_AUTO_ADDR_LIST",
    ),
    "pva": ("EPICS_PVA_ADDR_LIST", "EPICS_PVA_NAME_SERVERS", "EPICS_PVA_AUTO_ADDR_LIST"),
}

#: An endpoint: each of a provider's :data:`_ENDPOINT_VARS` and its value
#: (``None`` when unset).
_Endpoint = dict[str, str | None]

#: What each pvaccess module was bound to: the endpoint environment in force
#: when this module created that client's FIRST channel of each provider.
#: pvapy reads ``EPICS_CA_*`` / ``EPICS_PVA_*`` at that moment and never again,
#: so this is the only endpoint the provider can reach for the rest of the
#: process. Keyed by the module object rather than held as one global so a
#: test's stand-in client carries its own record.
_bound_endpoints: "weakref.WeakKeyDictionary[Any, dict[str, _Endpoint]]" = (
    weakref.WeakKeyDictionary()
)
#: Serializes the record above against ``connect()``'s check-then-set of the
#: environment, so a channel is never created between the two.
_endpoint_lock = threading.Lock()


def _provider(pva: bool) -> str:
    return "pva" if pva else "ca"


def _current_endpoint(provider: str) -> _Endpoint:
    """The endpoint the process environment names for ``provider`` right now."""
    return {var: os.environ.get(var) for var in _ENDPOINT_VARS[provider]}


def _describe_endpoint(endpoint: _Endpoint, provider: str) -> str:
    """``endpoint`` as ``VAR=value`` pairs, for an operator to compare."""
    named = [f"{var}={value}" for var, value in endpoint.items() if value is not None]
    return ", ".join(named) or f"no EPICS_{provider.upper()}_* variable set"


def _endpoint_conflict(
    pvaccess: Any, provider: str, requested: _Endpoint
) -> ClientEndpointConflictError | None:
    """The refusal for ``requested``, or ``None`` when the client can still reach it.

    The caller holds :data:`_endpoint_lock`. A provider this client has not
    bound yet can reach anything; one it has bound reaches only that.
    """
    bound = _bound_endpoints.get(pvaccess, {}).get(provider)
    if bound is None or bound == requested:
        return None
    protocol = "PVAccess" if provider == "pva" else "Channel Access"
    return ClientEndpointConflictError(
        provider,
        dict(bound),
        dict(requested),
        f"Refusing to connect the EPICS connector to the {protocol} endpoint "
        f"[{_describe_endpoint(requested, provider)}]: this process's pvapy client is "
        f"already bound to [{_describe_endpoint(bound, provider)}]. pvapy reads the "
        f"EPICS_{provider.upper()}_* environment once per process, when its first "
        f"{protocol} channel is created, so a connector built here would still talk to "
        "the old endpoint. Nothing was read or written. Start a fresh process to reach "
        "the new endpoint (restart the notebook kernel, or run the code in a new sandbox).",
    )


def _new_channel(
    pvaccess: Any, channel_address: str, pva: bool, expected: dict[str, _Endpoint] | None
) -> Any:
    """Create a pvapy Channel, recording the endpoint its provider binds to.

    The first channel of a provider binds it to the environment in force now;
    that is recorded so a later ``connect()`` asking for another endpoint is
    refused instead of silently reaching this one. ``expected`` is the
    endpoint the creating connector configured (``None`` for one wired
    without ``connect()``); a channel whose provider is bound elsewhere is
    refused here too, for the connector that configured its endpoint and was
    overtaken by another before its first channel.

    Raises:
        ClientEndpointConflictError: The provider is bound to an endpoint other
            than ``expected``.
    """
    provider = _provider(pva)
    with _endpoint_lock:
        record = _bound_endpoints.setdefault(pvaccess, {})
        if provider not in record:
            record[provider] = _current_endpoint(provider)
        if expected is not None and provider in expected:
            conflict = _endpoint_conflict(pvaccess, provider, expected[provider])
            if conflict is not None:
                raise conflict
        return pvaccess.Channel(channel_address, pvaccess.PVA if pva else pvaccess.CA)


def _backstop(timeout: float) -> float:
    """The outer deadline on an offloaded pvapy call bounded by ``timeout``.

    Each pvapy call a connector makes is bounded by the channel timeout, but
    one operation may make up to three of them in a row (waiting out another
    caller's connect, the get, a display get) and may queue for a worker
    first. This is the backstop that guarantees the loop never awaits a
    pvapy call forever — not the budget the call is expected to use.
    """
    return 3.0 * float(timeout) + 1.0


def _retrieve(future: "asyncio.Future[Any]") -> None:
    """Done-callback that marks a late outcome as seen, so it is never logged."""
    if not future.cancelled():
        future.exception()


class _ChannelEntry:
    """One cached pvapy Channel and what the connector learned about it.

    ``connect_lock`` serializes the FIRST use of the channel (and every use
    while it is disconnected): pvapy does not survive concurrent first use —
    measured against pvapy 5.6.0, of 14 threads calling ``get`` on one fresh
    Channel, one returned and the rest never did. Once the channel has
    answered and is connected, calls go straight through.

    ``value_type`` is the ``value`` entry of the channel's introspection —
    what a write must be converted to — and ``nelm`` a char array's element
    count; both are learned on the first write.
    """

    __slots__ = ("channel", "connect_lock", "nelm", "typed", "used", "value_type")

    def __init__(self, channel: Any) -> None:
        self.channel = channel
        self.connect_lock = threading.Lock()
        self.used = False
        self.typed = False
        self.value_type: Any = None
        self.nelm: int | None = None


class _ChannelSubscription:
    """One active monitor plus the teardown it needs.

    Both transports are pvapy monitors now: a dedicated ``Channel`` (never the
    connector's shared read channel, since a Channel runs one monitor) with
    one named subscriber. ``close()`` stops the monitor — immediate in pvapy —
    and removes the subscriber.

    ``close()`` is idempotent and best effort: closing twice is a no-op, and a
    handle that fails to release is logged rather than raised, so tearing down
    a whole connector cannot be derailed by one dead channel.
    """

    __slots__ = ("_closed", "channel", "name")

    def __init__(self, channel: Any, name: str) -> None:
        self.channel = channel
        self.name = name
        self._closed = False

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has run."""
        return self._closed

    def close(self) -> None:
        """Release the monitor. Safe to call any number of times."""
        if self._closed:
            return
        self._closed = True
        for step in (self.channel.stopMonitor, lambda: self.channel.unsubscribe(self.name)):
            try:
                step()
            except Exception as exc:  # teardown is best effort, never fatal
                logger.debug(f"Subscription teardown step failed for '{self.name}': {exc}")


class EPICSConnector(ControlSystemConnector):
    """
    EPICS control system connector using pvapy.

    Provides read/write access to EPICS Process Variables over Channel Access,
    and read access over PVAccess for addresses matching ``pva_channels``.
    Supports gateway configuration for remote access and
    read-only/write-access gateways.

    Example:
        Direct gateway connection:
        >>> config = {
        >>>     'timeout': 5.0,
        >>>     'gateways': {
        >>>         'read_only': {
        >>>             'address': 'cagw-alsdmz.als.lbl.gov',
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
        >>>     'timeout': 5.0,
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
        self._timeout = 5.0
        # The `max_step` fresh-read budget, replaced from the connector's own
        # config block on connect. Set here so the reader is usable on a
        # connector that has not connected — the fallback is the same number
        # the block's default resolves to.
        self._step_read_timeout = DEFAULT_STEP_READ_TIMEOUT_SECONDS
        self._subscriptions: dict[str, _ChannelSubscription] = {}
        # pvapy Channels, one per (address, provider), reused by every read and
        # put: a first connect costs ~0.1 s, and one connected Channel serves
        # concurrent gets from many threads. Guarded by a lock so two threads
        # reading a new address at once build one Channel, not two; each
        # entry's own lock then serializes the channel's first use.
        self._channels: dict[tuple[str, bool], _ChannelEntry] = {}
        # The confirming put in flight per CA address, if any; see _put().
        self._inflight_puts: dict[str, concurrent.futures.Future[Any]] = {}
        # Channel Access display metadata per address; see _ca_display().
        self._ca_displays: dict[str, dict[str, Any]] = {}
        self._channels_lock = threading.Lock()
        self._epics_configured = False
        # The pvaccess module, imported by connect() (never at module scope:
        # this file is held to import isolation).
        self._pvaccess: Any = None
        # The endpoint environment connect() configured, per provider; checked
        # again when a channel is created. None on a connector wired without
        # connect().
        self._endpoints: dict[str, _Endpoint] | None = None
        # PVA routing state. Empty globs => every address is Channel Access and
        # no PVA environment variable is set.
        self._pva_channel_globs: list[str] = []

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Configure EPICS environment and load the pvapy client.

        Every read goes to the IOC: pvapy keeps no monitor cache, so there is
        no stale cached value to bypass (the former ``fresh_reads`` option is
        gone — every read is already fresh).

        Args:
            config: Configuration with keys:
                - timeout: Default timeout in seconds (default: 5.0)
                - gateways: Gateway configuration dict with:
                    - read_only: {address, port, use_name_server} for read operations
                    - write_access: {address, port, use_name_server} for write operations

                Gateway sub-keys:
                    - address: Gateway hostname or IP
                    - port: Gateway port number
                    - use_name_server: (optional) Use EPICS_CA_NAME_SERVERS instead of
                      EPICS_CA_ADDR_LIST. Required for SSH tunnels. Default: False
                - pva_channels: (optional) List of glob patterns. Channel addresses
                  matching any pattern are served over PVAccess instead of
                  Channel Access. Absent or empty => pure Channel Access: no PVA
                  env var is touched.
                - pva_gateway: (optional) {address, port, use_name_server} naming
                  where PVA searches go. Only read when pva_channels is non-empty,
                  and the ONLY route: the connector host scrubs every inherited
                  EPICS_PVA_* variable before connect() runs, so a value set in
                  the deployment's environment never reaches pvapy.
                    - address: one host, or a space-separated list of hosts (what
                      a facility with many PVA servers and no gateway needs).
                    - port: (optional) With use_name_server false the entries go
                      to EPICS_PVA_ADDR_LIST as UDP search targets, and no port
                      is appended unless one is set here (the PVA default, 5076,
                      applies). Set, it is appended to every listed host. With
                      use_name_server true the entry is a TCP name server in
                      EPICS_PVA_NAME_SERVERS, defaulting to 5075.
                    - use_name_server: (optional) TCP instead of UDP search.
                      Required for SSH tunnels. Default: False

        The environment is read by pvapy when the FIRST channel of a provider
        is created, and is fixed for the process from then on. connect()
        creates no channel, so the variables it sets here are the ones that
        count — provided nothing else in this process opened a channel first.
        That is why a target switch is a new connector-host process, never a
        second connect() in the same one — and why a second connect() in the
        same process that needs a different endpoint than the one pvapy bound
        to is refused rather than allowed to reach the old one silently. The
        same endpoint reconnects freely.

        Raises:
            ImportError: If pvapy is not installed
            ClientEndpointConflictError: If this process's pvapy client is
                already bound to a different Channel Access (or, with
                ``pva_channels``, PVAccess) endpoint. The environment is left
                untouched and the connector stays unconnected.
        """
        # Import pvapy here (never at module scope) to give a clear error if it
        # is not installed, and to keep this module importable without it.
        try:
            import pvaccess
        except ImportError:
            raise ImportError(
                "pvapy is required for the EPICS connector. Install with: pip install pvapy"
            ) from None

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
        if writes_enabled and write_gateway:
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

        # What this connector sets in the environment, per variable (None
        # unsets it). Worked out first and applied only once the client is
        # known to still be able to reach it — see the check below.
        ca_updates: dict[str, str | None] = {}
        if gateway_config:
            address = gateway_config.get("address", "")
            port = gateway_config.get("port", 5064)
            # Explicit configuration for connection method
            # Config system automatically converts "true"/"false" strings to booleans
            use_name_server = gateway_config.get("use_name_server", False)

            # Configure EPICS environment variables
            # Clear conflicting variables first — having both CA_ADDR_LIST and
            # CA_NAME_SERVERS set causes TCP connection attempts that block
            # the connector's worker threads.
            if use_name_server:
                # Use CA_NAME_SERVERS (required for SSH tunnels and some gateway configurations)
                ca_updates["EPICS_CA_NAME_SERVERS"] = f"{address}:{port}"
                ca_updates["EPICS_CA_ADDR_LIST"] = None
                ca_updates["EPICS_CA_SERVER_PORT"] = None
                logger.debug(f"Using EPICS_CA_NAME_SERVERS: {address}:{port}")
            else:
                # Use CA_ADDR_LIST (standard gateway configuration)
                ca_updates["EPICS_CA_ADDR_LIST"] = address
                ca_updates["EPICS_CA_SERVER_PORT"] = str(port)
                ca_updates["EPICS_CA_NAME_SERVERS"] = None
                logger.debug(f"Using EPICS_CA_ADDR_LIST: {address}, CA_SERVER_PORT: {port}")

            ca_updates["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
            logger.debug(f"Configured EPICS gateway: {address}:{port}")

        # Configure PVAccess routing. Addresses matching one of these globs are
        # served by pvapy's PVA provider; every other address by its CA
        # provider. An absent or empty list sets no PVA environment variable.
        pva_channels = config.get("pva_channels") or []
        if isinstance(pva_channels, str):
            pva_channels = [pva_channels]
        pva_globs = [str(p).strip() for p in pva_channels if str(p).strip()]

        pva_updates: dict[str, str | None] = {}
        if pva_globs:
            pva_gateway = config.get("pva_gateway") or {}
            if pva_gateway:
                pva_address = str(pva_gateway.get("address", ""))
                pva_port = pva_gateway.get("port")
                # Config system automatically converts "true"/"false" to booleans
                pva_use_name_server = pva_gateway.get("use_name_server", False)
                # PVA carries the port inside the address entry itself — there is
                # no client-side "server port" variable to set, unlike CA.
                if pva_use_name_server:
                    # Name servers make the client connect by TCP instead of
                    # UDP-searching — required for SSH tunnels. 5075 is the PVA
                    # TCP port.
                    pva_endpoint = f"{pva_address}:{pva_port or 5075}"
                    pva_updates["EPICS_PVA_NAME_SERVERS"] = pva_endpoint
                    pva_updates["EPICS_PVA_ADDR_LIST"] = None
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
                    pva_updates["EPICS_PVA_ADDR_LIST"] = pva_endpoint
                    pva_updates["EPICS_PVA_NAME_SERVERS"] = None
                    logger.debug(f"Using EPICS_PVA_ADDR_LIST: {pva_endpoint}")

                # Containment, mirroring EPICS_CA_AUTO_ADDR_LIST above: a
                # deployment deliberately pinned to a gateway must not also
                # broadcast-discover servers on the local subnet.
                pva_updates["EPICS_PVA_AUTO_ADDR_LIST"] = "NO"
                logger.debug(f"Configured PVA gateway: {pva_endpoint}")

            logger.debug(f"PVA routing enabled for {len(pva_globs)} glob pattern(s)")

        # pvapy reads a provider's environment once per process, at its first
        # channel. If this process's client is already bound to a different
        # endpoint than the one worked out above, a connector built now would
        # talk to the old one under this configuration's name — so it is
        # refused, before the environment is touched. Every non-PVA address
        # is Channel Access, so CA is always checked; PVA only when routed.
        updates = {"ca": ca_updates}
        if pva_globs:
            updates["pva"] = pva_updates
        with _endpoint_lock:
            endpoints: dict[str, _Endpoint] = {}
            for provider, changes in updates.items():
                endpoint = _current_endpoint(provider)
                endpoint.update(changes)
                conflict = _endpoint_conflict(pvaccess, provider, endpoint)
                if conflict is not None:
                    raise conflict
                endpoints[provider] = endpoint
            for changes in updates.values():
                for var, value in changes.items():
                    if value is None:
                        os.environ.pop(var, None)
                    else:
                        os.environ[var] = value

        self._pvaccess = pvaccess
        self._endpoints = endpoints
        self._pva_channel_globs = pva_globs
        if ca_updates:
            self._epics_configured = True

        self._timeout = config.get("timeout", 5.0)
        # The ceiling on the fresh read a `max_step` check makes before a
        # write. A facility-network fact like `timeout` above, and read from
        # the same block: a gateway two hops away answers slower than a soft
        # IOC on this host. Running out of budget answers None, which refuses
        # the write — raising it buys a slow channel more room, never a
        # weaker check.
        self._step_read_timeout = step_read_timeout_seconds(config, self._connector_type)

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
        """Stop every monitor and drop every cached channel.

        Monitors go first: each runs on a Channel of its own, and stopping it is
        immediate in pvapy. pvapy Channels have no close of their own — the
        client releases a channel when the last reference to it goes — so
        forgetting the cached ones is the teardown. A connector disconnected
        twice does nothing the second time.
        """
        for sub_id in list(self._subscriptions.keys()):
            await self.unsubscribe(sub_id)

        with self._channels_lock:
            self._channels.clear()
            self._ca_displays.clear()

        self._connected = False
        logger.info("EPICS connector disconnected")

    # ------------------------------------------------------------------
    # pvapy plumbing
    # ------------------------------------------------------------------

    def _require_client(self, channel_address: str) -> Any:
        """The pvaccess module, or a ConnectionError on a connector never connected."""
        if self._pvaccess is None:
            raise ConnectionError(
                f"Cannot reach channel '{channel_address}': the EPICS connector is not "
                "connected (connect() loads the pvapy client)."
            )
        return self._pvaccess

    @staticmethod
    async def _offload(
        fn: Callable[..., _T], *args: Any, deadline: float | None = None, what: str = ""
    ) -> _T:
        """Await ``fn(*args)`` run on a pvapy worker (see :class:`_EpicsWorkers`).

        With a ``deadline`` (seconds), a call that has not finished by then
        raises ConnectionError naming ``what``; the worker is left to finish
        and its late outcome is discarded. Every read-side call passes one
        (:func:`_backstop`), so no caller awaits pvapy forever.
        """
        future: concurrent.futures.Future[_T] = _workers.submit(fn, *args)
        wrapped = asyncio.wrap_future(future)
        if deadline is None:
            return await wrapped
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), deadline)
        except TimeoutError:
            if wrapped.done():  # the call's own TimeoutError, not the deadline
                return wrapped.result()
            wrapped.add_done_callback(_retrieve)
            raise ConnectionError(
                f"{what or 'EPICS call'} did not complete within {deadline:.1f}s"
            ) from None

    def _entry(self, channel_address: str, pva: bool) -> _ChannelEntry:
        """The cached channel entry for this address and provider (thread-safe)."""
        pvaccess = self._require_client(channel_address)
        key = (channel_address, pva)
        with self._channels_lock:
            entry = self._channels.get(key)
            if entry is None:
                channel = _new_channel(pvaccess, channel_address, pva, self._endpoints)
                entry = _ChannelEntry(channel)
                self._channels[key] = entry
        return entry

    def _channel(self, channel_address: str, pva: bool) -> Any:
        """The cached pvapy Channel for this address and provider (thread-safe)."""
        return self._entry(channel_address, pva).channel

    def _use(
        self, channel_address: str, pva: bool, timeout: float, call: Callable[[Any], _T]
    ) -> _T:
        """Run ``call(channel)`` on the cached channel, serializing its first use.

        A channel that has answered before and is connected is used directly.
        Otherwise the caller takes the entry's connect lock — waiting at most
        ``timeout`` for another caller's connect to finish — and makes its
        call under it, so exactly one thread drives a connect at a time (see
        :class:`_ChannelEntry`). Callers still waiting when the connect
        succeeds then go straight through.

        Raises:
            ConnectionError: Another caller's connect outlasted ``timeout``.
        """
        entry = self._entry(channel_address, pva)
        channel = entry.channel
        if entry.used and channel.isConnected():
            return call(channel)
        if not entry.connect_lock.acquire(timeout=max(float(timeout), 0.0)):
            protocol = "PVA" if pva else "CA"
            raise ConnectionError(
                f"Failed to connect to {protocol} channel '{channel_address}' "
                f"(timeout after {timeout}s): its connect did not complete"
            )
        try:
            result = call(channel)
            entry.used = True
            return result
        finally:
            entry.connect_lock.release()

    def _classify(self, exc: BaseException) -> str | None:
        """:func:`_classify_client_error` against this connector's pvaccess module."""
        error_type = getattr(self._pvaccess, "PvaException", None)
        return _classify_client_error(exc, error_type)

    def _get(self, channel_address: str, pva: bool, request: str, timeout: float) -> Any:
        """Blocking pvapy get, returning the raw ``PvObject`` (runs on a pvapy worker).

        ``Channel.setTimeout`` bounds connect and get together. The channel is
        shared, so two concurrent calls with different timeouts race to set it
        — the loser's get runs under the other's budget. That is accepted: every
        caller passes the connector's configured timeouts, and the only
        consequence is a get that gives up a little early or late.

        A channel that did not connect or answer in time is re-raised as the
        stdlib ``ConnectionError``, so the MCP error envelope and the
        connector-invalidation logic treat an outage the same on both
        transports. Every other failure propagates unchanged.
        """

        def get(channel: Any) -> Any:
            channel.setTimeout(float(timeout))
            return channel.get(request)

        try:
            return self._use(channel_address, pva, timeout, get)
        except Exception as exc:
            if self._classify(exc) == _UNREACHABLE:
                protocol = "PVA" if pva else "CA"
                raise ConnectionError(
                    f"Failed to connect to {protocol} channel '{channel_address}' "
                    f"(timeout after {timeout}s): {exc}"
                ) from exc
            raise

    def _ca_display(self, channel_address: str, timeout: float) -> dict[str, Any]:
        """The display metadata of a CA channel, fetched once and cached.

        Channel Access cannot deliver ``display`` in the same get as
        ``timeStamp`` (see :data:`_CA_READ_REQUEST`), so it takes a second
        round trip — paid on the first read of a channel only, the way pyepics
        used to cache a PV's control variables. Units, precision, description
        and display limits are record configuration, not live state.

        Never fatal: a failed fetch is logged and answers ``{}``, and is retried
        on the next read. The read it enriches already has its value; losing
        that value for want of its units would be the wrong trade. An enum
        record answers with no ``display`` at all, which is cached as ``{}``.
        """
        with self._channels_lock:
            cached = self._ca_displays.get(channel_address)
        if cached is not None:
            return cached
        try:
            result = self._get(channel_address, False, _CA_DISPLAY_REQUEST, timeout)
            display = _sub(_top_fields(result), "display")
        except Exception as exc:  # metadata is an enrichment, never a precondition
            logger.debug(f"Could not fetch display metadata for '{channel_address}': {exc}")
            return {}
        with self._channels_lock:
            self._ca_displays[channel_address] = display
        return display

    def _has_ca_display(self, channel_address: str) -> bool:
        """True once this channel's display metadata has been fetched and cached."""
        with self._channels_lock:
            return channel_address in self._ca_displays

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        """
        Read current value from EPICS channel.

        Addresses matching one of the configured ``pva_channels`` globs are read
        over PVAccess; every other address is read over Channel Access. Both
        are blocking pvapy calls, so both run in a worker thread, and both
        always ask the IOC (pvapy has no monitor cache).

        Args:
            channel_address: EPICS channel address (e.g., 'BEAM:CURRENT')
            timeout: Timeout in seconds (uses default if None)

        Returns:
            ChannelValue with current value, timestamp, and metadata

        Raises:
            ConnectionError: If channel cannot be connected
            ValueError: If a PVA channel serves a compressed NTNDArray
        """
        timeout = timeout or self._timeout
        what = f"Read of channel '{channel_address}'"

        if self._is_pva_channel(channel_address):
            reader = self._read_channel_pva
        else:
            reader = self._read_channel_ca
        return await self._offload(
            reader, channel_address, timeout, deadline=_backstop(timeout), what=what
        )

    def _read_channel_ca(self, channel_address: str, timeout: float) -> ChannelValue:
        """Synchronous Channel Access read (runs on a pvapy worker)."""
        result = self._get(channel_address, False, _CA_READ_REQUEST, timeout)
        display = self._ca_display(channel_address, timeout)
        return _channel_value(channel_address, result, "ca", display=display)

    def _read_channel_pva(self, channel_address: str, timeout: float) -> ChannelValue:
        """Synchronous PVAccess read (runs on a pvapy worker, like the CA read)."""
        result = self._get(channel_address, True, _PVA_READ_REQUEST, timeout)
        return _channel_value(channel_address, result, "pva")

    def _current_value_reader(self) -> Callable[[str], Any] | None:
        """The channel's present value, read with the client this connector connected with.

        Never import a client library here: the one this connector holds is
        configured for the deployment's gateway, and this file is held to
        import isolation besides.

        Any failure answers ``None``, which the limits validator treats as
        "cannot verify the step" and refuses the write. An enum answers its
        index, the number a step is measured in.

        A PVA-routed address answers ``None`` as well, which fails the step
        check closed. ``write_channel`` refuses those before validation ever
        runs, so this is the belt to that braces.

        The read itself always runs on a pvapy worker (:class:`_EpicsWorkers`),
        whichever thread calls the reader: the runtime calls it on its own
        thread — in a notebook, a ``ThreadPoolExecutor`` thread that is joined
        when the call returns, which on macOS hangs forever once it has touched
        pvapy. A caller off the workers waits for the read at most
        :func:`_backstop` of the step budget, and a read that has not answered
        by then answers ``None``. The connector's own validation already runs
        on a worker, so it reads in place rather than queueing behind itself.
        """

        def read(channel_address: str) -> Any:
            try:
                result = self._get(channel_address, False, _VALUE_REQUEST, self._step_read_timeout)
                value = result["value"]
            except Exception as exc:
                logger.debug(f"Step-check read of '{channel_address}' failed: {exc}")
                return None
            if isinstance(value, dict) and "index" in value:
                return value["index"]
            return value

        def read_current(channel_address: str) -> Any:
            if self._is_pva_channel(channel_address):
                return None
            if _on_worker():
                return read(channel_address)
            future = _workers.submit(read, channel_address)
            deadline = _backstop(self._step_read_timeout)
            try:
                return future.result(timeout=deadline)
            except concurrent.futures.TimeoutError:
                logger.debug(
                    f"Step-check read of '{channel_address}' did not answer within {deadline:.1f}s"
                )
                return None

        return read_current

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

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
            ChannelLimitsViolationError: If limits validation fails (when enabled)
        """
        # Step 0: PVA-routed addresses are read-only. This check MUST stay ahead
        # of limits validation: validating max_step reads the channel's current
        # value over Channel Access, so a refusal placed after it would point
        # the CA provider at an address routed over PVAccess — the very thing
        # this refusal exists to prevent — and would do it for the arrays PVA
        # channels carry, which have no scalar limit to be compared against.
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

        # Step 2: Validate limits (FAIL CLOSED), off the loop: max_step
        # validation makes a blocking read of the channel's current value.
        def _validate() -> Exception | None:
            if not self._limits_validator:
                return None
            try:
                self._limits_validator.validate(
                    channel_address, value, read_current=self._current_value_reader()
                )
                logger.debug(f"✓ Limits validation passed: {channel_address}={value}")
            except ChannelLimitsViolationError:
                raise  # limits refusal propagates unchanged (carries LIMITS semantics)
            except Exception as e:
                # FAIL CLOSED: any other validation error refuses the write — no
                # put issued. The refusal itself is built on the loop, by the
                # base class's one helper, so all four connectors word it
                # identically.
                return e
            return None

        # A validation that never answers has not said yes: past its backstop
        # the write is refused like any other check that could not be made.
        try:
            validation_error = await self._offload(
                _validate,
                deadline=_backstop(self._step_read_timeout),
                what=f"Limits validation of '{channel_address}'",
            )
        except ConnectionError as e:
            validation_error = e
        if validation_error is not None:
            return self._validation_refusal(channel_address, value, validation_error)

        # Step 3: The put.
        try:
            acknowledged = await self._put(channel_address, value, confirm=confirm, timeout=timeout)
        except _NothingSent as e:
            # Known: the value never left the connector. The channel is
            # unreachable (or still busy), which is a verdict on this row, not
            # a fault of the connector — a batch goes on to the next write.
            logger.warning(f"EPICS write not sent: {channel_address}: {e}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.FAILED,
                error_message=f"Write to '{channel_address}' failed; nothing was sent: {e}",
            )
        except _UnwritableValue as e:
            return self._validation_refusal(channel_address, value, e)
        except Exception as e:
            if self._classify(e) != _ACCESS_DENIED:
                # Every other put error stays a raised failure: it leaves the
                # outcome genuinely unknown, and a refusal claims the opposite.
                raise
            # The control system was asked and said no (IOC access security).
            logger.warning(f"Control system refused write to {channel_address}: {e}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.REFUSED,
                refusal_reason="CONTROL_SYSTEM_REFUSED",
                error_message=(
                    f"Write to '{channel_address}' refused by the control system "
                    f"(access security); no value was written: {e}"
                ),
            )

        if not acknowledged:
            # The value was sent; nothing has said the IOC took it, so the
            # outcome is unknown and no re-read is made.
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

        alarm_status, alarm_severity = _readback_alarm_fields(observed)
        metadata = getattr(observed, "metadata", None)
        enum_label = getattr(metadata, "enum_label", None) if metadata is not None else None
        observed_value = observed.value
        if isinstance(value, str):
            # Text written to a char array reads back as its bytes; compare
            # (and report) it as the text it spells.
            text = _char_array_text(observed_value)
            if text is not None:
                observed_value = text

        if values_match(value, observed_value, enum_label=enum_label):
            logger.debug(f"EPICS write confirmed: {channel_address} = {observed_value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.CONFIRMED,
                observed_value=observed_value,
                alarm_status=alarm_status,
                alarm_severity=alarm_severity,
            )

        logger.warning(
            f"EPICS write mismatch on {channel_address}: sent {value}, "
            f"channel holds {observed_value}"
        )
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.MISMATCH,
            # No error_message: both values are on the result, and the raising
            # path composes "sent X, channel holds Y" from them — a message set
            # here would take that wording's place.
            observed_value=observed_value,
            alarm_status=alarm_status,
            alarm_severity=alarm_severity,
            notes=f"Channel holds {observed_value}, sent {value}",
        )

    async def _put(
        self, channel_address: str, value: Any, *, confirm: bool, timeout: float
    ) -> bool:
        """Put ``value`` over Channel Access; True unless an awaited ack never came.

        Both kinds of put run on a pvapy worker under one deadline,
        ``timeout``, which covers connecting, the put and — for a confirming
        put — the IOC's put-callback (:data:`_CONFIRMING_PUT_REQUEST`), which
        pvapy's blocking put waits for with no deadline of its own. Past the
        deadline the answer is False: the value was sent and nothing
        acknowledged it, which the caller reports as UNCONFIRMED. The put
        itself cannot be interrupted (it is a blocking C call) and is left to
        finish on its worker; see :class:`_EpicsWorkers`.

        "Sent" is only claimed once it is true. The channel is connected, and
        its type learned (:meth:`_put_target`), BEFORE the put is issued, and
        "sent" and "abandoned" are decided under one lock, so either the put
        was issued (and the deadline reports it unacknowledged) or it never
        will be — and then :class:`_NothingSent` is raised.

        One confirming put per channel at a time. pvapy's blocking put holds
        the GIL while it waits behind another blocking put pending on the same
        Channel — measured: a retry against a record whose put-callback takes
        1.5 s froze every Python thread, the event loop included, for 1.3 s,
        and queued retries against a 30 s record hung the process. So a
        confirming put to a channel whose previous one is still awaiting the
        IOC waits for it on the loop, within its own deadline, and never
        enters pvapy while it does; if the deadline passes first, nothing was
        sent. A plain put does not wait: pvapy sends it without blocking even
        behind a pending put-callback.

        Pinned against a real IOC in tests/connectors/test_epics_soft_ioc.py.
        ``Channel.asyncPut`` was measured as the alternative and rejected:
        destroying a Channel shortly after an asyncPut completes aborts the
        process (``epicsEvent::invalidSemaphore``), and destroying one later
        holds the GIL for ~0.7 s — and channels are dropped on disconnect.

        Raises:
            _NothingSent: The value is known not to have been sent.
            _UnwritableValue: The value does not fit the channel's type.
            ConnectionError: The channel failed after the value was sent.
            Exception: Any other pvapy failure, unchanged — including the
                access-security denial, which the caller classifies.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout)
        if confirm:
            await self._wait_for_inflight_put(channel_address, deadline, timeout)

        gate = threading.Lock()
        state = {"sent": False, "abandoned": False}

        def _issue() -> None:
            entry = self._put_target(channel_address, timeout)
            payload = self._payload(entry, channel_address, value, timeout)
            channel = entry.channel
            channel.setTimeout(float(timeout))
            with gate:
                if state["abandoned"]:
                    return  # the caller already reported that nothing was sent
                state["sent"] = True
            if confirm:
                channel.put(payload, _CONFIRMING_PUT_REQUEST)
            else:
                channel.put(payload)

        issued = _workers.submit(self._translating(channel_address, timeout, _issue))
        if confirm:
            self._track_inflight_put(channel_address, issued)
        put_done = asyncio.wrap_future(issued)
        # asyncio.wait neither raises nor cancels the future the worker settles.
        # Only a put still running at the deadline is decided here; a settled
        # one is judged below like any other, whichever timer fired first — so
        # pvapy's own channel timeout landing on the deadline is still a
        # connect that sent nothing.
        await asyncio.wait({put_done}, timeout=max(deadline - loop.time(), 0.0))
        if not put_done.done():
            put_done.add_done_callback(_retrieve)
            with gate:
                was_sent = state["sent"]
                state["abandoned"] = not was_sent
            if not was_sent:
                raise _NothingSent(f"channel '{channel_address}' did not connect within {timeout}s")
            return False
        try:
            put_done.result()
        except ConnectionError as exc:
            # Unreachable before the put was issued (connect, introspection,
            # NELM lookup): known not sent.
            if not state["sent"]:
                raise _NothingSent(str(exc)) from exc
            raise
        return True

    async def _wait_for_inflight_put(
        self, channel_address: str, deadline: float, timeout: float
    ) -> None:
        """Wait, on the loop, until no confirming put to this channel is in flight.

        Raises:
            _NothingSent: The earlier put was still pending at ``deadline``.
        """
        loop = asyncio.get_running_loop()
        while True:
            with self._channels_lock:
                pending = self._inflight_puts.get(channel_address)
            if pending is None or pending.done():
                return
            waiter = asyncio.wrap_future(pending)
            try:
                await asyncio.wait_for(asyncio.shield(waiter), max(deadline - loop.time(), 0.0))
            except TimeoutError:
                if waiter.done():  # the earlier put's own failure: it settled
                    continue
                waiter.add_done_callback(_retrieve)
                raise _NothingSent(
                    f"an earlier write to '{channel_address}' was still waiting for the IOC's "
                    f"put-callback after {timeout}s"
                ) from None
            except Exception:  # the earlier put's failure belongs to its own caller
                pass

    def _track_inflight_put(
        self, channel_address: str, issued: "concurrent.futures.Future[Any]"
    ) -> None:
        """Record ``issued`` as this channel's confirming put until it settles."""
        with self._channels_lock:
            self._inflight_puts[channel_address] = issued

        def _clear(future: "concurrent.futures.Future[Any]") -> None:
            with self._channels_lock:
                if self._inflight_puts.get(channel_address) is future:
                    del self._inflight_puts[channel_address]

        issued.add_done_callback(_clear)

    def _put_target(self, channel_address: str, timeout: float) -> _ChannelEntry:
        """The connected CA channel entry a put goes to, with its value type known.

        Connecting by introspection costs one round trip on the first write
        of a channel (and after it disconnects) and none after; the type it
        returns is what :func:`_typed_payload` converts the value to.
        """
        entry = self._entry(channel_address, False)
        if entry.typed and entry.channel.isConnected():
            return entry

        def introspect(channel: Any) -> Any:
            channel.setTimeout(float(timeout))
            return channel.getIntrospectionDict()

        structure = self._use(channel_address, False, timeout, introspect)
        entry.value_type = structure.get("value") if isinstance(structure, dict) else None
        entry.typed = True
        return entry

    def _payload(
        self, entry: _ChannelEntry, channel_address: str, value: Any, timeout: float
    ) -> Any:
        """``value`` converted for the channel ``entry`` describes (on a pvapy worker)."""
        nelm = None
        if isinstance(value, str) and _is_char_array(entry.value_type):
            if entry.nelm is None:
                entry.nelm = self._char_array_nelm(channel_address, timeout)
            nelm = entry.nelm
        return _typed_payload(value, entry.value_type, self._pvaccess, nelm)

    def _char_array_nelm(self, channel_address: str, timeout: float) -> int | None:
        """A char array record's ``NELM``, or ``None`` when it cannot be read.

        An over-long put is rejected by the IOC ("Invalid element count
        requested"), so text is cut to fit before it is sent.
        """
        record = channel_address.split(".", 1)[0]
        try:
            nelm = self._get(f"{record}.NELM", False, _VALUE_REQUEST, timeout)["value"]
        except Exception as exc:  # an enrichment: without it the IOC decides
            logger.debug(f"Could not read NELM of '{record}': {exc}")
            return None
        # NELM is a DBF_ULONG, which Channel Access serves as a DOUBLE.
        if isinstance(nelm, int | float) and not isinstance(nelm, bool) and nelm >= 1:
            return int(nelm)
        return None

    def _translating(
        self, channel_address: str, timeout: float, call: Callable[[], None]
    ) -> Callable[[], None]:
        """Wrap a blocking put so an unreachable channel raises ConnectionError.

        Only pvapy's own "timed out" text is translated; a ConnectionError the
        connector raised itself (a connect that did not complete) passes as is.
        """

        def run() -> None:
            try:
                call()
            except Exception as exc:
                if self._classify(exc) == _UNREACHABLE:
                    raise ConnectionError(
                        f"Failed to connect to CA channel '{channel_address}' "
                        f"(timeout after {timeout}s): {exc}"
                    ) from exc
                raise

        return run

    async def _confirming_read(self, channel_address: str, timeout: float) -> ChannelValue:
        """Read the channel a write just touched.

        Every read goes to the IOC (pvapy keeps no monitor cache), so this is
        the ordinary read path — alarm state, enum labels and timestamp are
        built the same way.

        A reading that carries no value is not a reading of the setpoint: a
        ``None`` compared against it would report a mismatch, and an
        ``observed_value`` of ``None``, for an observation that was never made.
        Confirmation is the one caller that cannot tolerate that, so here (and
        only here) a missing value is raised as a timeout, and the write is
        reported as unconfirmed: what the channel holds is unknown, not
        different.
        """
        observed = await self._offload(
            self._read_channel_ca,
            channel_address,
            timeout,
            deadline=_backstop(timeout),
            what=f"Confirming read of '{channel_address}'",
        )
        if observed.value is None:
            raise TimeoutError(f"confirming read of '{channel_address}' timed out after {timeout}s")
        return observed

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        """Read multiple channels concurrently."""
        tasks = [self.read_channel(ch_addr, timeout) for ch_addr in channel_addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            ch_addr: result
            for ch_addr, result in zip(channel_addresses, results, strict=False)
            if not isinstance(result, Exception)
        }

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

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

        Routing mirrors the read path: a PVA-routed address opens a PVA
        monitor, every other address a Channel Access one. Either way the
        subscriber callback runs on the event loop, never on pvapy's own worker
        thread, and the first update is the channel's current value.
        """
        loop = asyncio.get_running_loop()
        pva = self._is_pva_channel(channel_address)
        started: asyncio.Future[str] = asyncio.wrap_future(
            _workers.submit(self._start_monitor, channel_address, callback, loop, pva)
        )
        deadline = _backstop(self._timeout)
        try:
            return await asyncio.wait_for(asyncio.shield(started), deadline)
        except TimeoutError:
            if started.done():  # the subscribe's own TimeoutError, not the deadline
                return started.result()

            # A monitor that starts after its caller gave up would run unowned.
            def _stop_late(future: "asyncio.Future[str]") -> None:
                if future.cancelled() or future.exception() is not None:
                    return
                late = self._subscriptions.pop(future.result(), None)
                if late is not None:
                    _workers.submit(late.close)

            started.add_done_callback(_stop_late)
            raise ConnectionError(
                f"Subscribing to '{channel_address}' did not complete within {deadline:.1f}s"
            ) from None

    def _start_monitor(
        self,
        channel_address: str,
        callback: Callable[[ChannelValue], None],
        loop: asyncio.AbstractEventLoop,
        pva: bool,
    ) -> str:
        """Open a pvapy monitor on a Channel of its own (runs on a pvapy worker).

        A Channel runs one monitor, so a subscription never uses the shared
        read channel. A Channel Access monitor cannot carry ``display`` either,
        so its units and precision come from the same cached display metadata
        the reads use, fetched here — never on pvapy's callback thread, where a
        round trip would delay every update behind it.

        A mapping failure (an NTNDArray in a codec the connector cannot decode,
        say) drops that one update with a log line. pvapy would log a raising
        callback and keep the monitor running, but a log line naming the
        channel is the one worth having.
        """
        pvaccess = self._require_client(channel_address)
        display = {} if pva else self._ca_display(channel_address, self._timeout)
        provider = "pva" if pva else "ca"
        # A display fetch that failed here is retried — off pvapy's callback
        # thread, at most one at a time and every _DISPLAY_RETRY_S — until it
        # succeeds, so a monitor opened while the IOC was slow does not report
        # empty units for the rest of its life.
        display_state: dict[str, Any] = {
            "display": display,
            "known": pva or self._has_ca_display(channel_address),
            "fetching": False,
            "retry_at": 0.0,
        }
        state_lock = threading.Lock()

        def refresh_display() -> None:
            fetched = self._ca_display(channel_address, self._timeout)
            with state_lock:
                if self._has_ca_display(channel_address):
                    display_state["display"] = fetched
                    display_state["known"] = True
                display_state["fetching"] = False
                display_state["retry_at"] = time.monotonic() + _DISPLAY_RETRY_S

        def current_display() -> dict[str, Any]:
            with state_lock:
                if (
                    not display_state["known"]
                    and not display_state["fetching"]
                    and time.monotonic() >= display_state["retry_at"]
                ):
                    display_state["fetching"] = True
                    _workers.submit(refresh_display)
                current: dict[str, Any] = display_state["display"]
                return current

        def monitor_callback(update: Any) -> None:
            """Convert a pvapy monitor update to our format, on pvapy's thread."""
            try:
                channel_value = _channel_value(
                    channel_address, update, provider, display=current_display()
                )
            except Exception as exc:
                logger.warning(f"Dropping {provider.upper()} update for '{channel_address}': {exc}")
                return
            try:
                loop.call_soon_threadsafe(callback, channel_value)
            except RuntimeError:  # the loop closed under a still-running monitor
                logger.debug(f"Dropping update for '{channel_address}': event loop closed")

        channel = _new_channel(pvaccess, channel_address, pva, self._endpoints)
        name = f"osprey-{uuid.uuid4().hex}"
        subscription = _ChannelSubscription(channel, name)
        try:
            channel.setTimeout(float(self._timeout))
            channel.subscribe(name, monitor_callback)
            channel.startMonitor(_PVA_MONITOR_REQUEST if pva else _CA_MONITOR_REQUEST)
        except Exception as exc:
            subscription.close()
            if self._classify(exc) == _UNREACHABLE:
                raise ConnectionError(
                    f"Failed to monitor channel '{channel_address}': {exc}"
                ) from exc
            raise

        sub_id = f"{channel_address}_{name}"
        self._subscriptions[sub_id] = subscription
        logger.debug(f"EPICS subscription created: {sub_id}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from channel changes, whichever transport opened them."""
        subscription = self._subscriptions.pop(subscription_id, None)
        if subscription is None:
            return
        # On a pvapy worker like every other pvapy call (see _EpicsWorkers).
        try:
            await self._offload(
                subscription.close,
                deadline=_backstop(self._timeout),
                what=f"Unsubscribing '{subscription_id}'",
            )
        except ConnectionError as exc:  # the handle is already forgotten
            logger.warning(str(exc))
            return
        logger.debug(f"EPICS subscription removed: {subscription_id}")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _read_metadata_pva(self, channel_address: str, timeout: float) -> ChannelMetadata:
        """Metadata-only PVAccess get (runs on a pvapy worker, like the reads).

        Asks the server for :data:`_PVA_METADATA_REQUEST` only, so a metadata
        lookup on a camera channel costs a few fields instead of a whole frame,
        and maps the reply with the same helpers the full read uses — the
        payload branches are skipped entirely, because the reply carries no
        payload to branch on. Failures translate the same way as
        :meth:`_read_channel_pva` — both go through :meth:`_get`.

        One consequence: the returned metadata carries no ``enum_labels``. The
        NTEnum choices live under ``value``, which this request deliberately
        does not ask for, and the label for "the current value" is meaningless
        without a value anyway. Enum labels come from a read.
        """
        result = self._get(channel_address, True, _PVA_METADATA_REQUEST, timeout)
        fields = _top_fields(result)
        raw_metadata: dict[str, Any] = {"provider": "pva", "nt_type": None}
        raw_metadata.update(_alarm_metadata(fields, "pva"))
        return _metadata(fields, _timestamp(fields), raw_metadata)

    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        """Get metadata for a channel, over whichever transport serves it.

        The PVA path asks for the metadata fields alone rather than reading the
        channel and discarding its value; the CA path reads the channel, which
        is what carries its alarm state and timestamp (its display metadata is
        cached after the first read anyway).
        """
        if self._is_pva_channel(channel_address):
            return await self._offload(
                self._read_metadata_pva,
                channel_address,
                self._timeout,
                deadline=_backstop(self._timeout),
                what=f"Metadata read of '{channel_address}'",
            )

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
                await self._offload(
                    self._read_metadata_pva,
                    channel_address,
                    timeout,
                    deadline=_backstop(timeout),
                    what=f"Validation of '{channel_address}'",
                )
            else:
                await self.read_channel(channel_address, timeout=timeout)
            return True
        except Exception as e:
            logger.debug(f"Channel validation failed for {channel_address}: {e}")
            return False
