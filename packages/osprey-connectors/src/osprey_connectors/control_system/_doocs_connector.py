"""
DOOCS control system connector using doocs4py.

Provides interface to the DOOCS control system.

Author: Frank Mayet (DESY, MXL)
Date: 2026-07-01
"""

import asyncio
import secrets
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol

from typing_extensions import override

from osprey_connectors._contract_types import ContractAny
from osprey_connectors.call_timeout import DEFAULT_TIMEOUT_S
from osprey_connectors.control_system._base import (
    ChannelMetadata,
    ChannelValue,
    ControlSystemConnector,
    ControlSystemSettings,
    SendReport,
)
from osprey_connectors.control_system._limits_validator import (
    CurrentValueReader,
)
from osprey_connectors.errors import ChannelUnreachableError, ConnectorUnavailableError
from osprey_connectors.logger import get_logger

logger = get_logger("doocs_connector")


class _EqTimestamp(Protocol):
    """The timestamp of a doocs4py ``EqData`` reading."""

    def get_seconds_and_microseconds_since_epoch(self) -> tuple[int, int]: ...


class _EqData(Protocol):
    """A doocs4py ``EqData`` reading."""

    macropulse: object
    timestamp: _EqTimestamp

    def get_data(self) -> object: ...


class _Doocs4py(Protocol):
    """The surface of :mod:`doocs4py` the connector calls; doocs4py ships no types."""

    def names(self, pattern: str) -> Sequence[Sequence[str]]: ...

    def get(self, address: str) -> _EqData: ...

    def set(self, address: str, value: object) -> None: ...

    def Address(self, address: str) -> object: ...

    def subscribe(self, address: object, callback: Callable[[_EqData], None]) -> None: ...

    def unsubscribe(self, address: object) -> None: ...


class DOOCSConnector(ControlSystemConnector):
    """
    DOOCS control system connector using doocs4py

    Provides read/write access to DOOCS properties.
    """

    def __init__(self) -> None:
        self._connected: bool = False
        self._subscriptions: dict[str, object] = {}
        self._timeout_s: float = DEFAULT_TIMEOUT_S
        # doocs4py, set by connect().
        self._doocs4py: _Doocs4py | None = None

    @override
    async def _open(self, settings: ControlSystemSettings) -> None:
        """
        Configure DOOCS environment and test connection.

        Args:
            settings: The doocs block. Its one connector-specific key is
                ``timeout_s``: the seconds an ENS lookup, a property read or a
                property set is given (default 5.0). The ENS itself still comes
                from the DOOCS environment.

        Raises:
            ImportError: If doocs4py is not installed
            ValueError: If the block still carries ``timeout``, or if
                ``timeout_s`` is not a positive, finite number
            ConnectorUnavailableError: If the ENS does not answer
        """
        # Import doocs4py here and give clear error if not installed
        try:
            import doocs4py

            self._doocs4py = doocs4py
            logger.debug(f"DOOCS connector: doocs4py version {doocs4py.__version__} loaded")
        except ImportError:
            raise ImportError("doocs4py is required for the DOOCS connector.") from None

        self._timeout_s = self._call_timeout_s

        # Test connection using a doocs4py.names call, listing all FACILITYs,
        # off the event loop and within the bound.
        try:
            names = await asyncio.wait_for(
                asyncio.to_thread(self._doocs.names, "*"), self._timeout_s
            )
        except TimeoutError:
            raise ConnectorUnavailableError(
                f"DOOCS connector: the ENS did not answer within {self._timeout_s}s"
            ) from None
        except Exception as exc:
            raise ConnectorUnavailableError(
                f"DOOCS connector failed to connect to the ENS: {exc}"
            ) from exc
        facilities = [f[1] for f in names]
        logger.debug(
            f"DOOCS connector: ENS connection successful. Available FACILITIEs: {len(facilities)}"
        )

        self._connected = True
        logger.debug("DOOCS connector initialized")

    @override
    async def disconnect(self) -> None:
        """Cleanup DOOCS connections."""
        # Unsubscribe from all active subscriptions
        for sub_id in list(self._subscriptions.keys()):
            await self.unsubscribe(sub_id)

        self._connected = False
        logger.info("DOOCS connector disconnected")

    @property
    def _doocs(self) -> _Doocs4py:
        """doocs4py, as connect() imported it.

        Raises:
            ConnectorUnavailableError: connect() has not run, so there is no
                DOOCS client to call.
        """
        doocs4py = self._doocs4py
        if doocs4py is None:
            raise ConnectorUnavailableError(
                "No DOOCS client: connect() imports doocs4py for this connector."
            )
        return doocs4py

    @override
    async def read_channel(
        self,
        address: str,
        *,
        timeout_s: float | None = None,
    ) -> ChannelValue:
        """
        Read current value from a DOOCS property.

        Args:
            address: DOOCS address (e.g., 'FACILITY/DEVICE/LOCATION/PROPERTY')
            timeout_s: Seconds the read is given; the block's ``timeout_s`` when omitted

        Returns:
            ChannelValue with current value, ``sampled_at``, and metadata

        Raises:
            ChannelUnreachableError: If the property cannot be read, or does
                not answer within the bound
        """

        bound = self._bound(timeout_s)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._read_channel_sync, address), bound
            )
        except TimeoutError:
            raise ChannelUnreachableError(
                address, f"DOOCS property '{address}' did not answer within {bound}s"
            ) from None

    def _read_channel_sync(self, address: str) -> ChannelValue:
        """Synchronous DOOCS read (runs in thread pool).

        A failure of the ``get`` itself is reported as
        :class:`~osprey_connectors.errors.ChannelUnreachableError` naming the
        property, with doocs4py's own error kept as the cause: a doocs4py call
        holds no session, so a failed ``get`` is that property's, and nothing
        about the connector is replaced for it. The unwrapping below is outside
        the guard: a reading that cannot be unwrapped is not a property that
        cannot be reached.
        """
        try:
            data = self._doocs.get(address)
        except TimeoutError:
            raise
        except Exception as exc:
            raise ChannelUnreachableError(
                address, f"Failed to read DOOCS property '{address}': {exc}"
            ) from exc

        value = data.get_data()
        macropulse = data.macropulse
        timestamp_s, timestamp_us = data.timestamp.get_seconds_and_microseconds_since_epoch()
        timestamp_float = timestamp_s + timestamp_us / 1e6

        sampled_at = datetime.fromtimestamp(timestamp_float, UTC)

        # Compile metadata; DOOCS reports no alarm state, so the reading has none.
        metadata = ChannelMetadata(
            units="",
            precision=None,
            raw_metadata={
                "macropulse": macropulse,
                "type": type(value).__name__,
            },
        )

        return ChannelValue(address=address, value=value, sampled_at=sampled_at, metadata=metadata)

    @override
    def current_value_reader(self) -> CurrentValueReader:
        """The property's present value, read through this connector's own client.

        ``get_data()`` is what :meth:`_read_channel_sync` unwraps an ``EqData``
        with; the step check needs only that number, not the timestamps and
        metadata a full read builds around it.
        """

        def read_current(address: str) -> object:
            return self._doocs.get(address).get_data()

        return read_current

    @override
    async def _send(
        self, address: str, value: ContractAny, *, timeout_s: float, wait: bool
    ) -> SendReport:
        """Set the property, off the event loop and within *timeout_s*.

        DOOCS has no acknowledgement of its own — ``set()`` either returns or
        raises — so *wait* changes nothing, and a confirmed write is the set
        followed by one fresh read of the same property. A set that raises is
        one DOOCS did not take. One that outlives its bound is unacknowledged:
        the worker thread cannot be stopped, so the set may still arrive.
        """

        def set_value() -> Exception | None:
            try:
                self._doocs.set(address, value)
            except Exception as exc:
                return exc
            return None

        try:
            error = await asyncio.wait_for(asyncio.to_thread(set_value), timeout_s)
        except TimeoutError:
            logger.warning(f"DOOCS set of {address} did not return within {timeout_s}s")
            return SendReport(
                fault="unacknowledged",
                detail=f"DOOCS did not return from the set within {timeout_s}s",
            )
        if error is not None:
            return SendReport(fault="not_taken", detail=f"DOOCS did not take the value: {error}")
        return SendReport()

    @override
    async def subscribe(
        self,
        address: str,
        callback: Callable[[ChannelValue], None],
        *,
        on_connection: Callable[[bool], None]
        | None = None,  # accepted and ignored: DOOCS reports no connection state
    ) -> str:
        """
        Subscribe to property value changes.

        Args:
            address: DOOCS address
            callback: Function to call when value changes
            on_connection: Accepted for the base-class contract and ignored —
                DOOCS reports no connection state to this connector.

        Returns:
            Subscription ID for later unsubscription
        """
        loop = asyncio.get_event_loop()

        def doocs_callback(data: _EqData) -> None:
            """Wrapper to convert DOOCS callback to Osprey format."""
            value = data.get_data()
            macropulse = data.macropulse
            timestamp_s, timestamp_us = data.timestamp.get_seconds_and_microseconds_since_epoch()
            timestamp_float = timestamp_s + timestamp_us / 1e6

            sampled_at = datetime.fromtimestamp(timestamp_float, UTC)

            # Compile metadata; DOOCS reports no alarm state, so the reading has none.
            metadata = ChannelMetadata(
                units="",
                precision=None,
                raw_metadata={
                    "macropulse": macropulse,
                    "type": type(value).__name__,
                },
            )

            prop_value = ChannelValue(
                address=address, value=value, sampled_at=sampled_at, metadata=metadata
            )
            # Schedule callback in event loop
            loop.call_soon_threadsafe(callback, prop_value)

        # Subscribe
        doocs_address = self._doocs.Address(address)
        self._doocs.subscribe(doocs_address, doocs_callback)

        # Generate subscription ID
        sub_id = f"{address}_{secrets.token_hex(8)}"
        self._subscriptions[sub_id] = self._doocs.Address(address)

        logger.debug(f"DOOCS subscription created: {sub_id}")
        return sub_id

    @override
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from DOOCS property changes."""
        if subscription_id in self._subscriptions:
            address = self._subscriptions[subscription_id]
            self._doocs.unsubscribe(address)
            del self._subscriptions[subscription_id]
            logger.debug(f"DOOCS subscription removed: {subscription_id}")
