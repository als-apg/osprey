"""
TANGO Controls control system connector using PyTango.

Provides read/write access to TANGO device attributes. A channel address is
``domain/family/member/attribute`` — the TANGO device name plus the attribute
served by it — optionally prefixed with an explicit database,
``tango://host:port/domain/family/member/attribute``. Without the prefix the
connector reaches the database named by ``TANGO_HOST``, exactly as every other
TANGO client does.

Only attributes are exposed. TANGO *commands* (``command_inout``) carry
arbitrary payloads the limits database cannot bound, so they have no seam in
the connector contract — the same reason the EPICS connector refuses PVA RPC.
"""

import asyncio
import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from typing import ClassVar, Protocol

from typing_extensions import override

from osprey_connectors._contract_types import ContractAny
from osprey_connectors.call_timeout import DEFAULT_TIMEOUT_S
from osprey_connectors.control_system._base import (
    Alarm,
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    ControlSystemSettings,
    SendReport,
    _settings_as,
)
from osprey_connectors.control_system._limits_validator import (
    CurrentValueReader,
)
from osprey_connectors.errors import ChannelUnreachableError, ConnectorUnavailableError
from osprey_connectors.logger import get_logger
from osprey_connectors.model import evolve

logger = get_logger("tango_connector")

_ADDRESS_FORM = "domain/family/member/attribute"


class _DeviceAttribute(Protocol):
    """A PyTango ``DeviceAttribute``: one reading of an attribute."""

    value: object


class _DeviceProxy(Protocol):
    """The surface of a PyTango ``DeviceProxy`` the connector calls; PyTango ships no types."""

    def set_timeout_millis(self, timeout: int) -> None: ...

    def read_attribute(self, attribute: str) -> _DeviceAttribute: ...

    def write_attribute(self, attribute: str, value: object) -> None: ...

    def get_attribute_config(self, attribute: str) -> object: ...

    def subscribe_event(
        self, attribute: str, event_type: object, callback: Callable[[object], None]
    ) -> int: ...

    def unsubscribe_event(self, event_id: int) -> None: ...


class _Database(Protocol):
    """A PyTango ``Database`` client."""

    def get_info(self) -> object: ...


class _EventType(Protocol):
    """PyTango's ``EventType`` enumeration."""

    CHANGE_EVENT: object


class _PyTango(Protocol):
    """The surface of :mod:`tango` the connector calls."""

    EventType: _EventType

    def Database(self, *host_port: str) -> _Database: ...

    def DeviceProxy(self, name: str) -> _DeviceProxy: ...


#: TANGO attribute quality mapped onto the protocol-agnostic alarm fields:
#: the name OSPREY reports and an integer severity (0 healthy, higher worse).
#: ``ATTR_CHANGING`` is a healthy reading whose value is in motion; it is
#: reported like any other alarm state and NEVER changes a write outcome —
#: the write contract is one put, one read, one word, with no settling
#: concept (a mid-ramp readback is a ``mismatch``, exactly as on EPICS).
_QUALITY_ALARM_FIELDS: dict[str, tuple[str, int]] = {
    "ATTR_VALID": ("NO_ALARM", 0),
    "ATTR_CHANGING": ("CHANGING", 0),
    "ATTR_WARNING": ("WARNING", 1),
    "ATTR_ALARM": ("ALARM", 2),
    "ATTR_INVALID": ("INVALID", 3),
}


def _split_address(address: str) -> tuple[str, str]:
    """Split a channel address into (device name, attribute name).

    The last path segment is the attribute; everything before it is the TANGO
    device name, with an optional ``tango://host:port/`` database prefix kept
    on the device so an explicitly-addressed proxy stays explicitly addressed.

    Raises:
        ValueError: If the address does not name a device and an attribute.
    """
    stripped = address.strip()
    prefix = ""
    body = stripped
    if stripped.lower().startswith("tango://"):
        rest = stripped[len("tango://") :]
        host, sep, body = rest.partition("/")
        if not sep or not host:
            raise ValueError(
                f"Invalid TANGO channel address '{address}': expected "
                f"'tango://host:port/{_ADDRESS_FORM}'"
            )
        prefix = f"tango://{host}/"
    parts = body.split("/")
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"Invalid TANGO channel address '{address}': expected '{_ADDRESS_FORM}'")
    return prefix + "/".join(parts[:3]), parts[3]


def _quality_alarm(quality: object) -> Alarm | None:
    """The alarm a TANGO attribute quality reports, or ``None`` when it reports none.

    ``None`` means "not reported" — deliberately distinct from a reported
    healthy reading (``NO_ALARM`` / severity ``0``), matching the contract on
    :class:`~osprey_connectors.control_system.Alarm`. A quality outside the
    table is ``UNKNOWN`` with no severity.
    """
    if quality is None:
        return None
    name = getattr(quality, "name", None) or str(quality)
    fields = _QUALITY_ALARM_FIELDS.get(str(name))
    if fields is None:
        return Alarm(status="UNKNOWN")
    status, severity = fields
    return Alarm(status=status, severity=severity)


def _attribute_sampled_at(attr: object) -> datetime:
    """The reading's own instant in UTC, or now when the device reported none."""
    time_val = getattr(attr, "time", None)
    tv_sec = getattr(time_val, "tv_sec", None)
    if tv_sec is None:
        return datetime.now(UTC)
    tv_usec = getattr(time_val, "tv_usec", 0) or 0
    return datetime.fromtimestamp(tv_sec + tv_usec / 1e6, UTC)


def _enum_label_for(labels: tuple[str, ...] | None, index: object) -> str | None:
    """The label this reading's index resolves to, or ``None``.

    Fails soft on anything unresolvable: the integer index alone is still a
    correct, complete answer, exactly as on the EPICS connector.
    """
    if not labels:
        return None
    if isinstance(index, bool):
        index = int(index)
    if isinstance(index, int) and 0 <= index < len(labels):
        return labels[index]
    return None


class TangoSettings(ControlSystemSettings):
    """The ``tango`` block."""

    #: Overrides the ``TANGO_HOST`` environment for this connector's proxies,
    #: spelled ``host:port``; empty or absent keeps the environment's.
    tango_host: str | None = None


class TangoConnector(ControlSystemConnector):
    """
    TANGO Controls control system connector using PyTango.

    Provides read/write access to TANGO device attributes through cached
    ``DeviceProxy`` instances. Every PyTango call is a blocking device round
    trip, so each one runs in a thread; a read is therefore always fresh —
    there is no client-side monitor cache for a confirming read to bypass.
    """

    settings_model: ClassVar[type[ControlSystemSettings]] = TangoSettings

    def __init__(self) -> None:
        self._connected: bool = False
        self._proxies: dict[str, _DeviceProxy] = {}
        self._subscriptions: dict[str, tuple[_DeviceProxy, int]] = {}
        self._enum_labels: dict[str, tuple[str, ...] | None] = {}
        self._timeout_s: float = DEFAULT_TIMEOUT_S
        # PyTango and the ``tango_host`` override, both set by connect().
        self._tango: _PyTango | None = None
        self._tango_host: str | None = None

    @override
    async def _open(self, settings: ControlSystemSettings) -> None:
        """
        Import PyTango and verify the TANGO database is reachable.

        Args:
            settings: Optional keys — ``tango_host`` (overrides the ``TANGO_HOST``
                environment for this connector's proxies, spelled ``host:port``)
                and ``timeout_s`` (seconds per device call; default 5.0).

        Raises:
            ImportError: If PyTango is not installed.
            ValueError: If the block still carries ``timeout``, or if
                ``timeout_s`` is not a positive, finite number.
            ConnectorUnavailableError: If the TANGO database does not answer.
        """
        # Import PyTango here and give a clear error if not installed. The
        # deferred import is what lets the type register unconditionally on
        # machines with no TANGO environment (same pattern as DOOCS/MongoDB).
        try:
            import tango

            self._tango = tango
            logger.debug(
                f"TANGO connector: PyTango {getattr(tango, '__version__', 'unknown')} loaded"
            )
        except ImportError:
            raise ImportError("PyTango (the 'pytango' package) is required.") from None

        tango_host = _settings_as(settings, TangoSettings).tango_host or None
        self._tango_host = tango_host
        self._timeout_s = self._call_timeout_s

        # Test the database connection now, so a bad TANGO_HOST fails the
        # deployment's startup instead of its first tool call.
        try:
            if self._tango_host:
                host, _, port = self._tango_host.partition(":")
                database = await asyncio.to_thread(self._pytango.Database, host, port or "10000")
            else:
                database = await asyncio.to_thread(self._pytango.Database)
            info = await asyncio.to_thread(database.get_info)
            logger.debug(f"TANGO connector: database reachable — {str(info).splitlines()[0]}")
        except Exception as e:
            raise ConnectorUnavailableError(
                f"TANGO connector failed to reach the TANGO database: {e}"
            ) from e

        self._connected = True
        logger.debug("TANGO connector initialized")

    @override
    async def disconnect(self) -> None:
        """Cleanup TANGO subscriptions and drop the proxy cache."""
        for sub_id in list(self._subscriptions.keys()):
            try:
                await self.unsubscribe(sub_id)
            except Exception as e:  # a dead device must not wedge shutdown
                logger.debug(f"TANGO unsubscribe failed during disconnect: {e}")
        self._proxies.clear()
        self._enum_labels.clear()
        self._connected = False
        logger.info("TANGO connector disconnected")

    @property
    def _pytango(self) -> _PyTango:
        """PyTango, as connect() imported it.

        Raises:
            ConnectorUnavailableError: connect() has not run, so there is no
                TANGO client to call.
        """
        tango = self._tango
        if tango is None:
            raise ConnectorUnavailableError(
                "No TANGO client: connect() imports PyTango for this connector."
            )
        return tango

    def _get_proxy(self, device_name: str) -> _DeviceProxy:
        """The cached ``DeviceProxy`` for a device, created on first use."""
        proxy = self._proxies.get(device_name)
        if proxy is None:
            name = device_name
            if self._tango_host and not device_name.lower().startswith("tango://"):
                name = f"tango://{self._tango_host}/{device_name}"
            proxy = self._pytango.DeviceProxy(name)
            try:
                proxy.set_timeout_millis(int(self._timeout_s * 1000))
            except Exception:  # a proxy that cannot take a timeout keeps its default
                logger.debug(f"TANGO proxy for '{device_name}' kept its default timeout")
            self._proxies[device_name] = proxy
        return proxy

    def _labels_for(
        self, proxy: _DeviceProxy, device_name: str, attribute: str
    ) -> tuple[str, ...] | None:
        """Enum labels for a ``DevEnum`` attribute, cached per attribute.

        Fails soft to ``None``: labels are an enrichment, never a precondition
        of the read.
        """
        key = f"{device_name}/{attribute}"
        if key in self._enum_labels:
            return self._enum_labels[key]
        labels: tuple[str, ...] | None
        try:
            info = proxy.get_attribute_config(attribute)
            raw = getattr(info, "enum_labels", None)
            labels = tuple(str(label) for label in raw) if raw else None
        except Exception as exc:
            logger.debug(f"Could not fetch enum labels for '{key}': {exc}")
            labels = None
        self._enum_labels[key] = labels
        return labels

    def _read_channel_sync(self, address: str) -> ChannelValue:
        """Synchronous TANGO read (runs in a thread).

        Building the proxy and reading the attribute are the two device round
        trips, and a failure in either is reported as
        :class:`~osprey_connectors.errors.ChannelUnreachableError` naming the
        attribute, with PyTango's own error kept as the cause: the class the
        base contract owns for a channel that cannot be reached, which leaves
        the connector in place.

        The address parse above the guard and the mapping below it are outside
        it on purpose: a malformed address is a ``ValueError`` and a reading
        that cannot be mapped is neither — only the transport is translated.
        """
        device_name, attribute = _split_address(address)

        try:
            proxy = self._get_proxy(device_name)
            attr = proxy.read_attribute(attribute)
        except TimeoutError:
            raise
        except Exception as exc:
            raise ChannelUnreachableError(
                address, f"Failed to read TANGO attribute '{address}': {exc}"
            ) from exc

        value = attr.value
        quality: object = getattr(attr, "quality", None)
        sampled_at = _attribute_sampled_at(attr)

        enum_labels: tuple[str, ...] | None = None
        enum_label: str | None = None
        attr_type: object = getattr(attr, "type", None)
        if attr_type is not None and "DevEnum" in str(attr_type):
            enum_labels = self._labels_for(proxy, device_name, attribute)
            enum_label = _enum_label_for(enum_labels, value)

        metadata = ChannelMetadata(
            units="",
            precision=None,
            enum_labels=enum_labels,
            raw_metadata={
                "quality": str(getattr(quality, "name", None) or quality)
                if quality is not None
                else None,
            },
        )
        return ChannelValue(
            address=address,
            value=value,
            sampled_at=sampled_at,
            alarm=_quality_alarm(quality),
            enum_label=enum_label,
            metadata=metadata,
        )

    @override
    async def read_channel(self, address: str, *, timeout_s: float | None = None) -> ChannelValue:
        """
        Read the current value of a TANGO device attribute.

        Args:
            address: ``domain/family/member/attribute``
            timeout_s: Seconds the read is given; the block's ``timeout_s`` when
                omitted, which the device proxy also applies beneath it

        Returns:
            ChannelValue with value, ``sampled_at``, alarm state and (on ``DevEnum``
            attributes) the state labels

        Raises:
            ChannelUnreachableError: If the device or its attribute cannot be
                reached, or the read does not answer within its ceiling
            ValueError: If the address does not name a device and an attribute
        """
        ceiling = timeout_s if timeout_s is not None else self._timeout_s
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._read_channel_sync, address), ceiling
            )
        except TimeoutError:
            raise ChannelUnreachableError(
                address, f"TANGO attribute '{address}' did not answer within {ceiling}s"
            ) from None

    @override
    def current_value_reader(self) -> CurrentValueReader:
        """The attribute's present value, read through this connector's own proxy."""

        def read_current(address: str) -> object:
            device_name, attribute = _split_address(address)
            return self._get_proxy(device_name).read_attribute(attribute).value

        return read_current

    @override
    def _precheck(self, address: str, value: object) -> tuple[str, ChannelWriteResult | None]:
        """An address that names no device and attribute is a refusal, single or batched."""
        try:
            _split_address(address)
        except ValueError as exc:
            return address, self._validation_refusal(address, value, exc)
        return address, None

    @override
    async def _send(
        self, address: str, value: ContractAny, *, timeout_s: float, wait: bool
    ) -> SendReport:
        """Write the attribute, off the event loop and within *timeout_s*.

        ``write_attribute`` either returns or raises, so *wait* changes
        nothing; every PyTango read is a device round trip, so the confirming
        read that follows is fresh by construction, and on a ``DevEnum``
        attribute a value written as text is compared against its resolved
        label. A write that raises is one TANGO did not take. One that
        outlives its bound is unacknowledged: the worker thread cannot be
        stopped, so the value may still arrive. Creating a ``DeviceProxy`` is
        itself a database round trip, so a cold cache is filled in the thread
        too.
        """
        device_name, attribute = _split_address(address)

        def write() -> Exception | None:
            try:
                self._get_proxy(device_name).write_attribute(attribute, value)
            except Exception as exc:
                return exc
            return None

        try:
            error = await asyncio.wait_for(asyncio.to_thread(write), timeout_s)
        except TimeoutError:
            logger.warning(f"TANGO write of {address} did not return within {timeout_s}s")
            return SendReport(
                fault="unacknowledged",
                detail=f"TANGO did not return from the write within {timeout_s}s",
            )
        if error is not None:
            return SendReport(fault="not_taken", detail=f"TANGO did not take the value: {error}")
        return SendReport()

    @override
    async def subscribe(
        self,
        address: str,
        callback: Callable[[ChannelValue], None],
        *,
        on_connection: Callable[[bool], None]
        | None = None,  # accepted and ignored: TANGO reports no connection state
    ) -> str:
        """
        Subscribe to attribute change events.

        Requires the device to have change events configured — TANGO refuses
        the subscription otherwise, and that refusal propagates.

        Args:
            address: ``domain/family/member/attribute``
            callback: Function to call when the value changes
            on_connection: Accepted for the base-class contract and ignored —
                TANGO reports no connection state to this connector.

        Returns:
            Subscription ID for later unsubscription
        """
        device_name, attribute = _split_address(address)
        proxy = await asyncio.to_thread(self._get_proxy, device_name)
        loop = asyncio.get_event_loop()

        def tango_callback(event: object) -> None:
            """Wrapper converting a TANGO change event to OSPREY format."""
            if getattr(event, "err", False):
                return
            attr: _DeviceAttribute | None = getattr(event, "attr_value", None)
            if attr is None:
                return
            channel_value = ChannelValue(
                address=address,
                value=attr.value,
                sampled_at=_attribute_sampled_at(attr),
                alarm=_quality_alarm(getattr(attr, "quality", None)),
            )
            loop.call_soon_threadsafe(callback, channel_value)

        event_id = await asyncio.to_thread(
            proxy.subscribe_event,
            attribute,
            self._pytango.EventType.CHANGE_EVENT,
            tango_callback,
        )

        sub_id = f"{address}_{secrets.token_hex(8)}"
        self._subscriptions[sub_id] = (proxy, event_id)
        logger.debug(f"TANGO subscription created: {sub_id}")
        return sub_id

    @override
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from TANGO attribute change events."""
        if subscription_id in self._subscriptions:
            proxy, event_id = self._subscriptions.pop(subscription_id)
            await asyncio.to_thread(proxy.unsubscribe_event, event_id)
            logger.debug(f"TANGO subscription removed: {subscription_id}")

    @override
    async def get_metadata(self, address: str) -> ChannelMetadata:
        """Get metadata for a channel, enriched from the attribute config.

        The attribute config carries what a reading does not: units and the
        description. TANGO's ``min_value``/``max_value`` are write limits, not
        a display range, so they ride in ``raw_metadata`` rather than being
        reported as ``display_low``/``display_high`` — OSPREY's enforced
        bounds live in the limits database, not here.
        """
        channel_value = await self.read_channel(address)
        metadata = channel_value.metadata

        device_name, attribute = _split_address(address)
        try:
            proxy = self._get_proxy(device_name)
            info = await asyncio.to_thread(proxy.get_attribute_config, attribute)
        except Exception as exc:  # the reading's own metadata still stands
            logger.debug(f"Could not fetch attribute config for '{address}': {exc}")
            return metadata

        unit = getattr(info, "unit", "") or ""
        if unit.lower() in ("no unit", "none"):  # TANGO's spellings of "unitless"
            unit = ""
        description = getattr(info, "description", None) or None
        if description == "No description":
            description = None
        return evolve(
            metadata,
            units=unit,
            description=description,
            raw_metadata={
                **(metadata.raw_metadata or {}),
                "min_value": getattr(info, "min_value", None),
                "max_value": getattr(info, "max_value", None),
                "format": getattr(info, "format", None),
                "writable": str(getattr(info, "writable", None)),
            },
        )
