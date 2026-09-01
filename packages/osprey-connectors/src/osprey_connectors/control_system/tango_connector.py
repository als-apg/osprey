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
from datetime import datetime
from typing import Any

from osprey_connectors.config import get_facility_timezone
from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
    values_match,
)
from osprey_connectors.control_system.limits_validator import LimitsValidator
from osprey_connectors.logger import get_logger

logger = get_logger("tango_connector")

_ADDRESS_FORM = "domain/family/member/attribute"

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


def _split_address(channel_address: str) -> tuple[str, str]:
    """Split a channel address into (device name, attribute name).

    The last path segment is the attribute; everything before it is the TANGO
    device name, with an optional ``tango://host:port/`` database prefix kept
    on the device so an explicitly-addressed proxy stays explicitly addressed.

    Raises:
        ValueError: If the address does not name a device and an attribute.
    """
    address = channel_address.strip()
    prefix = ""
    body = address
    if address.lower().startswith("tango://"):
        rest = address[len("tango://") :]
        host, sep, body = rest.partition("/")
        if not sep or not host:
            raise ValueError(
                f"Invalid TANGO channel address '{channel_address}': expected "
                f"'tango://host:port/{_ADDRESS_FORM}'"
            )
        prefix = f"tango://{host}/"
    parts = body.split("/")
    if len(parts) != 4 or not all(parts):
        raise ValueError(
            f"Invalid TANGO channel address '{channel_address}': expected '{_ADDRESS_FORM}'"
        )
    return prefix + "/".join(parts[:3]), parts[3]


def _quality_fields(quality: Any) -> tuple[str | None, int | None]:
    """Alarm name and severity for a TANGO attribute quality, or ``(None, None)``.

    ``None`` means "not reported" — deliberately distinct from a reported
    healthy reading (``NO_ALARM`` / severity ``0``), matching the contract on
    :class:`~osprey_connectors.control_system.base.ChannelWriteResult`.
    """
    if quality is None:
        return None, None
    name = getattr(quality, "name", None) or str(quality)
    fields = _QUALITY_ALARM_FIELDS.get(str(name))
    if fields is None:
        return "UNKNOWN", None
    return fields


def _attribute_timestamp(attr: Any, tz: Any) -> datetime:
    """The reading's own timestamp, or now when the device reported none."""
    time_val = getattr(attr, "time", None)
    tv_sec = getattr(time_val, "tv_sec", None)
    if tv_sec is None:
        return datetime.now(tz)
    tv_usec = getattr(time_val, "tv_usec", 0) or 0
    return datetime.fromtimestamp(tv_sec + tv_usec / 1e6, tz)


def _enum_label_for(labels: list[str] | None, index: Any) -> str | None:
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


class TangoConnector(ControlSystemConnector):
    """
    TANGO Controls control system connector using PyTango.

    Provides read/write access to TANGO device attributes through cached
    ``DeviceProxy`` instances. Every PyTango call is a blocking device round
    trip, so each one runs in a thread; a read is therefore always fresh —
    there is no client-side monitor cache for a confirming read to bypass.
    """

    def __init__(self):
        self._connected: bool = False
        self._proxies: dict[str, Any] = {}
        self._subscriptions: dict[str, tuple[Any, int]] = {}
        self._enum_labels: dict[str, list[str] | None] = {}
        self._timeout: float = 5.0

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Import PyTango and verify the TANGO database is reachable.

        Args:
            config: Optional keys — ``tango_host`` (overrides the ``TANGO_HOST``
                environment for this connector's proxies, spelled ``host:port``)
                and ``timeout`` (seconds, per proxy call; default 5.0).

        Raises:
            ImportError: If PyTango is not installed.
            ConnectionError: If the TANGO database does not answer.
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

        self._tango_host: str | None = config.get("tango_host") or None
        self._timeout = float(config.get("timeout", 5.0))

        # Initialize limits validator for automatic validation and confirm policy
        self._limits_validator = LimitsValidator.from_config(connector_type=self._connector_type)
        if self._limits_validator:
            logger.debug("TANGO connector: limits validator initialized")

        # Test the database connection now, so a bad TANGO_HOST fails the
        # deployment's startup instead of its first tool call.
        try:
            if self._tango_host:
                host, _, port = self._tango_host.partition(":")
                database = await asyncio.to_thread(self._tango.Database, host, port or "10000")
            else:
                database = await asyncio.to_thread(self._tango.Database)
            info = await asyncio.to_thread(database.get_info)
            logger.debug(f"TANGO connector: database reachable — {str(info).splitlines()[0]}")
        except Exception as e:
            raise ConnectionError(f"TANGO connector failed to reach the TANGO database: {e}") from e

        self._connected = True
        logger.debug("TANGO connector initialized")

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

    def _get_proxy(self, device_name: str) -> Any:
        """The cached ``DeviceProxy`` for a device, created on first use."""
        proxy = self._proxies.get(device_name)
        if proxy is None:
            name = device_name
            if self._tango_host and not device_name.lower().startswith("tango://"):
                name = f"tango://{self._tango_host}/{device_name}"
            proxy = self._tango.DeviceProxy(name)
            try:
                proxy.set_timeout_millis(int(self._timeout * 1000))
            except Exception:  # a proxy that cannot take a timeout keeps its default
                logger.debug(f"TANGO proxy for '{device_name}' kept its default timeout")
            self._proxies[device_name] = proxy
        return proxy

    def _labels_for(self, proxy: Any, device_name: str, attribute: str) -> list[str] | None:
        """Enum labels for a ``DevEnum`` attribute, cached per attribute.

        Fails soft to ``None``: labels are an enrichment, never a precondition
        of the read.
        """
        key = f"{device_name}/{attribute}"
        if key in self._enum_labels:
            return self._enum_labels[key]
        labels: list[str] | None
        try:
            info = proxy.get_attribute_config(attribute)
            raw = getattr(info, "enum_labels", None)
            labels = [str(label) for label in raw] if raw else None
        except Exception as exc:
            logger.debug(f"Could not fetch enum labels for '{key}': {exc}")
            labels = None
        self._enum_labels[key] = labels
        return labels

    def _read_channel_sync(self, channel_address: str) -> ChannelValue:
        """Synchronous TANGO read (runs in a thread)."""
        device_name, attribute = _split_address(channel_address)
        proxy = self._get_proxy(device_name)

        attr = proxy.read_attribute(attribute)
        value = attr.value
        quality = getattr(attr, "quality", None)
        alarm_status, severity = _quality_fields(quality)
        timestamp = _attribute_timestamp(attr, get_facility_timezone())

        enum_labels: list[str] | None = None
        enum_label: str | None = None
        attr_type = getattr(attr, "type", None)
        if attr_type is not None and "DevEnum" in str(attr_type):
            enum_labels = self._labels_for(proxy, device_name, attribute)
            enum_label = _enum_label_for(enum_labels, value)

        metadata = ChannelMetadata(
            units="",
            precision=None,
            alarm_status=alarm_status,
            timestamp=timestamp,
            enum_labels=enum_labels,
            enum_label=enum_label,
            raw_metadata={
                # Severity rides in raw_metadata exactly as on the EPICS read
                # path; the write path lifts it onto the result.
                "severity": severity,
                "quality": str(getattr(quality, "name", None) or quality)
                if quality is not None
                else None,
            },
        )
        # ChannelMetadata grows a first-class alarm_severity field
        # independently of this connector; populate it where it exists rather
        # than depending on it.
        if hasattr(metadata, "alarm_severity"):
            metadata.alarm_severity = severity
        return ChannelValue(value=value, timestamp=timestamp, metadata=metadata)

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        """
        Read the current value of a TANGO device attribute.

        Args:
            channel_address: ``domain/family/member/attribute``
            timeout: Optional per-call ceiling in seconds; the proxy's own
                timeout (``timeout`` in the connector config) applies beneath it

        Returns:
            ChannelValue with value, timestamp, alarm state and (on ``DevEnum``
            attributes) the state labels

        Raises:
            ValueError: If the address does not name a device and an attribute
            TimeoutError: If the per-call ceiling elapses
        """
        call = asyncio.to_thread(self._read_channel_sync, channel_address)
        if timeout is not None:
            return await asyncio.wait_for(call, timeout)
        return await call

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """
        Write a value to a TANGO attribute, confirming it unless asked not to.

        A confirmed write is the value sent followed by one fresh read of the
        same attribute — every PyTango read is a device round trip, so the
        confirming read is fresh by construction — compared with
        :func:`~osprey_connectors.control_system.base.values_match`. On a
        ``DevEnum`` attribute a value written as text is compared against the
        readback's resolved label, exactly as on EPICS.

        The readback's attribute quality is reported on the result
        (``alarm_status`` / ``alarm_severity``), never raised on — and
        ``ATTR_CHANGING`` does not soften the verdict: a readback that does not
        yet hold the value sent is a ``mismatch``, whatever its quality says.

        Args:
            channel_address: ``domain/family/member/attribute``
            value: Value to write
            timeout: Optional per-call ceiling for the confirming read
            confirm: Whether to re-read and compare, or ``None`` to resolve the
                policy for this channel from the limits database

        Returns:
            ChannelWriteResult carrying the outcome and what the attribute was
            seen to hold

        Raises:
            ChannelLimitsViolationError: If limits validation fails (when enabled)
            ValueError: If the address does not name a device and an attribute
        """
        # The address must parse before anything is validated or sent.
        device_name, attribute = _split_address(channel_address)

        # Step 1: Validate limits (if enabled)
        if self._limits_validator:
            try:
                self._limits_validator.validate(channel_address, value)
                logger.debug(f"✓ Limits validation passed: {channel_address}={value}")
            except Exception as e:
                # Import here to avoid circular dependency
                from osprey_connectors.errors import ChannelLimitsViolationError

                # Re-raise limits violations
                if isinstance(e, ChannelLimitsViolationError):
                    raise

                # Log unexpected errors but don't block (fail-open for non-limit errors)
                logger.warning(f"Limits validation error (non-blocking): {e}")

        # Step 2: Resolve the confirmation policy. An explicit confirm — False
        # every bit as much as True — is an answer and is taken as given; only
        # an omitted one is resolved from the limits database.
        if confirm is None:
            confirm = self._resolve_confirm(channel_address)

        # Step 3: Send the value. Creating a DeviceProxy is itself a database
        # round trip, so a cold cache is filled in the thread too.
        try:
            proxy = await asyncio.to_thread(self._get_proxy, device_name)
            await asyncio.to_thread(proxy.write_attribute, attribute, value)
        except Exception as e:
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.FAILED,
                error_message=f"Failed to write to '{channel_address}': {e}",
                notes="TANGO did not take the value",
            )

        if not confirm:
            logger.debug(f"TANGO write (unconfirmed by request): {channel_address} = {value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNREQUESTED,
                notes="No confirmation requested",
            )

        # Step 4: Confirm with one fresh read
        try:
            readback = await self.read_channel(channel_address, timeout=timeout)
        except Exception as e:
            logger.warning(f"TANGO confirming read failed for {channel_address}: {e}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNCONFIRMED,
                error_message=f"Confirming read of '{channel_address}' failed: {e}",
                notes="The value was sent; what the attribute holds is unknown",
            )

        observed = readback.value
        confirmed = values_match(value, observed, enum_label=readback.metadata.enum_label)
        raw = readback.metadata.raw_metadata or {}

        logger.debug(
            f"TANGO write ({'confirmed' if confirmed else 'mismatch'}): "
            f"{channel_address} = {value!r}, observed {observed!r}"
        )

        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED if confirmed else WriteOutcome.MISMATCH,
            observed_value=observed,
            alarm_status=readback.metadata.alarm_status,
            alarm_severity=raw.get("severity"),
            notes=(
                f"Observed {observed!r}" if confirmed else f"Observed {observed!r}, sent {value!r}"
            ),
        )

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

    async def subscribe(
        self, channel_address: str, callback: Callable[[ChannelValue], None]
    ) -> str:
        """
        Subscribe to attribute change events.

        Requires the device to have change events configured — TANGO refuses
        the subscription otherwise, and that refusal propagates.

        Args:
            channel_address: ``domain/family/member/attribute``
            callback: Function to call when the value changes

        Returns:
            Subscription ID for later unsubscription
        """
        device_name, attribute = _split_address(channel_address)
        proxy = await asyncio.to_thread(self._get_proxy, device_name)
        loop = asyncio.get_event_loop()
        tz = get_facility_timezone()

        def tango_callback(event: Any) -> None:
            """Wrapper converting a TANGO change event to OSPREY format."""
            if getattr(event, "err", False):
                return
            attr = getattr(event, "attr_value", None)
            if attr is None:
                return
            alarm_status, severity = _quality_fields(getattr(attr, "quality", None))
            timestamp = _attribute_timestamp(attr, tz)
            metadata = ChannelMetadata(
                alarm_status=alarm_status,
                timestamp=timestamp,
                raw_metadata={"severity": severity},
            )
            if hasattr(metadata, "alarm_severity"):
                metadata.alarm_severity = severity
            channel_value = ChannelValue(value=attr.value, timestamp=timestamp, metadata=metadata)
            loop.call_soon_threadsafe(callback, channel_value)

        event_id = await asyncio.to_thread(
            proxy.subscribe_event,
            attribute,
            self._tango.EventType.CHANGE_EVENT,
            tango_callback,
        )

        sub_id = f"{channel_address}_{secrets.token_hex(8)}"
        self._subscriptions[sub_id] = (proxy, event_id)
        logger.debug(f"TANGO subscription created: {sub_id}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from TANGO attribute change events."""
        if subscription_id in self._subscriptions:
            proxy, event_id = self._subscriptions.pop(subscription_id)
            await asyncio.to_thread(proxy.unsubscribe_event, event_id)
            logger.debug(f"TANGO subscription removed: {subscription_id}")

    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        """Get metadata for a channel, enriched from the attribute config.

        The attribute config carries what a reading does not: units and the
        description. TANGO's ``min_value``/``max_value`` are write limits, not
        a display range, so they ride in ``raw_metadata`` rather than being
        reported as ``display_low``/``display_high`` — OSPREY's enforced
        bounds live in the limits database, not here.
        """
        channel_value = await self.read_channel(channel_address)
        metadata = channel_value.metadata

        device_name, attribute = _split_address(channel_address)
        try:
            proxy = self._get_proxy(device_name)
            info = await asyncio.to_thread(proxy.get_attribute_config, attribute)
        except Exception as exc:  # the reading's own metadata still stands
            logger.debug(f"Could not fetch attribute config for '{channel_address}': {exc}")
            return metadata

        unit = getattr(info, "unit", "") or ""
        if unit.lower() in ("no unit", "none"):  # TANGO's spellings of "unitless"
            unit = ""
        metadata.units = unit
        metadata.description = getattr(info, "description", None) or None
        if metadata.description in ("No description",):
            metadata.description = None
        raw = metadata.raw_metadata or {}
        raw.update(
            {
                "min_value": getattr(info, "min_value", None),
                "max_value": getattr(info, "max_value", None),
                "format": getattr(info, "format", None),
                "writable": str(getattr(info, "writable", None)),
            }
        )
        metadata.raw_metadata = raw
        return metadata

    async def validate_channel(self, channel_address: str) -> bool:
        """
        Check if the attribute exists and is readable.

        Args:
            channel_address: ``domain/family/member/attribute``

        Returns:
            True if the channel can be read
        """
        try:
            await self.read_channel(channel_address)
            return True
        except Exception as e:
            logger.debug(f"TANGO channel validation failed for {channel_address}: {e}")
            return False
