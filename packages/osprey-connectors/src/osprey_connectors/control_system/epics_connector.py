"""
EPICS control system connector using pyepics.

Provides interface to EPICS Channel Access (CA) control system.
Refactored from existing EPICS integration code.

"""

import asyncio
import fnmatch
import os
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from osprey_connectors.config import get_facility_timezone
from osprey_connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteVerification,
)
from osprey_connectors.logger import get_logger

logger = get_logger("epics_connector")


def _configure_pyepics_libca() -> None:
    """Point pyepics at epicscorelibs' per-architecture libca before it loads CA.

    pyepics's bundled ``clibs`` are x86_64-only: its ``find_libca()`` selects
    ``clibs/linux64`` for any 64-bit OS, so on an arm64 host it loads a
    mismatched-architecture ``libca.so`` and Channel Access initialization
    fails outright. ``epicscorelibs`` (already present via the CA stack) ships a
    correct per-architecture libca; exporting ``PYEPICS_LIBCA`` to it makes the
    connector's CA work in arm64 and amd64 containers alike.

    Only sets the variable when it is unset, so an operator's explicit
    ``PYEPICS_LIBCA`` override always wins. A no-op that leaves pyepics' own
    resolution in place when ``epicscorelibs`` is not installed.
    """
    if os.environ.get("PYEPICS_LIBCA"):
        return
    try:
        from epicscorelibs.path import get_lib

        os.environ["PYEPICS_LIBCA"] = get_lib("ca")
        logger.debug("Configured PYEPICS_LIBCA from epicscorelibs: %s", os.environ["PYEPICS_LIBCA"])
    except Exception:  # epicscorelibs absent/failed -> pyepics falls back to its own resolution
        logger.debug("epicscorelibs libca unavailable; using pyepics default libca resolution")


def _alarm_name(code: Any) -> str:
    """Map a Channel Access alarm status code onto its EPICS name.

    Channel Access reports the alarm status as a small integer; the rest of
    OSPREY (and ``ChannelMetadata.alarm_status``) speaks names, which is what
    PVAccess already emits. This is the one place codes become names, so the
    shared connector core stays free of EPICS constants.

    Anything the enum cannot map — an out-of-range code, ``None`` from a PV
    that has not reported yet — becomes ``"UNKNOWN"``. pyepics' enum already
    does that itself (verified on 3.5.9), but the guard keeps the read path
    from failing on a pyepics build that decides otherwise, or one where the
    import is unavailable.
    """
    try:
        from epics.dbr import AlarmStatus

        return str(AlarmStatus(code).name)
    except Exception:  # unmappable code, or pyepics unavailable
        return "UNKNOWN"


def _is_enum_type(pv_type: Any) -> bool:
    """True when a Channel Access field type names an enumeration.

    pyepics spells the type of an ``mbbi``/``bi``/``bo`` as ``"enum"`` or, in
    the connector's default ``time`` form, ``"time_enum"`` — hence the substring
    test rather than an equality check against one spelling.
    """
    return "enum" in str(pv_type)


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


def _ca_enum_labels(pv: Any, index: Any) -> tuple[list[str] | None, str | None]:
    """The state labels of an enum-typed PV, and the one this reading is in.

    ``pv.enum_strs`` triggers pyepics' lazy ``get_ctrlvars`` fetch, which is a
    round trip to the IOC and so can time out or fail on a PV that answered its
    value perfectly well. That is caught here: labels are an enrichment, never a
    precondition of the read.
    """
    try:
        labels = pv.enum_strs
    except Exception as exc:  # ctrlvars fetch failed — the value still stands
        logger.debug("Could not fetch enum labels for '%s': %s", getattr(pv, "pvname", pv), exc)
        return None, None
    return _enum_label_fields(labels, index)


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
    severity = raw.get("severity") if isinstance(raw, dict) else None
    if severity is not None:
        try:
            severity = int(severity)
        except (TypeError, ValueError):
            severity = None

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
    Channel Access keeps the pyepics ``PV`` (released with
    ``clear_callbacks()``), PVAccess keeps the p4p ``Subscription`` (released
    with ``close()`` — it has no ``clear_callbacks`` at all). Dispatching here
    is what lets ``unsubscribe`` and ``disconnect`` walk subscription ids
    without knowing which protocol opened them.

    ``close()`` is idempotent and best effort: closing twice is a no-op, and a
    handle that fails to release is logged rather than raised, so tearing down
    a whole connector cannot be derailed by one dead channel.
    """

    __slots__ = ("_closed", "handle", "kind")

    def __init__(self, kind: str, handle: Any) -> None:
        self.kind = kind
        self.handle = handle
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
            if self.kind == "pva":
                self.handle.close()
            else:
                self.handle.clear_callbacks()
        except Exception as exc:  # teardown is best effort, never fatal
            logger.debug(f"Subscription teardown failed for {self.kind} handle: {exc}")


class EPICSConnector(ControlSystemConnector):
    """
    EPICS control system connector using pyepics.

    Provides read/write access to EPICS Process Variables through
    Channel Access protocol. Supports gateway configuration for
    remote access and read-only/write-access gateways.

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
        >>>             'port': 5074,
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
        self._subscriptions: dict[str, Any] = {}
        self._pv_cache: dict[str, Any] = {}
        self._pv_cache_lock = threading.Lock()  # Thread safety for PV cache
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
                  matching any pattern are served over PVAccess (p4p) instead of
                  Channel Access. Absent or empty => pure Channel Access, exactly
                  as before: p4p is not imported and no PVA env var is touched.
                - pva_gateway: (optional) {address, port, use_name_server} for the
                  PVA gateway, mapping to EPICS_PVA_ADDR_LIST or
                  EPICS_PVA_NAME_SERVERS. Only read when pva_channels is non-empty.

        Raises:
            ImportError: If pyepics is not installed, or if PVA channels are
                configured and p4p is not installed
        """
        # Ensure pyepics loads a correct-architecture libca before first CA use.
        _configure_pyepics_libca()

        # Import epics here to give clear error if not installed
        try:
            import epics

            self._epics = epics
        except ImportError:
            raise ImportError(
                "pyepics is required for EPICS connector. Install with: pip install pyepics"
            ) from None

        # Select the CA gateway. EPICS uses one process-wide context, so the
        # connector points at a single gateway. A read-only gateway rejects
        # writes, so a write-enabled deployment must route through the
        # write-capable gateway. Defense-in-depth: only use write_access when
        # writes are enabled, so a read-only deployment also has its writes
        # rejected at the network layer (reinforcing the writes_enabled switch).
        from osprey_connectors.config import get_config_value

        gateways = config.get("gateways", {})
        try:
            writes_enabled = get_config_value("control_system.writes_enabled", False)
        except (FileNotFoundError, KeyError, RuntimeError):
            writes_enabled = False  # Can't tell -> assume the safe (read-only) path

        write_gateway = gateways.get("write_access") or {}
        # A readonly sandbox run stays on the read_only gateway even on a
        # write-enabled deployment: the per-run claim is enforced at the
        # network layer, so a raw caput issued after read_channel() in the
        # same process is rejected by the gateway rather than trusted.
        from osprey_connectors.control_system.base import is_readonly_run

        readonly_run = is_readonly_run()
        if writes_enabled and write_gateway and not readonly_run:
            gateway_config = write_gateway
            logger.debug("EPICS connector: routing through write_access gateway (writes enabled)")
        elif readonly_run and write_gateway:
            gateway_config = gateways.get("read_only", {})
            logger.debug("EPICS connector: readonly run, staying on read_only gateway")
        else:
            gateway_config = gateways.get("read_only", {})
            if writes_enabled and not write_gateway:
                logger.warning(
                    "control_system.writes_enabled is true but no gateways.write_access "
                    "is configured; routing through the read_only gateway, which may "
                    "reject writes. Configure gateways.write_access to enable hardware writes."
                )

        if gateway_config:
            address = gateway_config.get("address", "")
            port = gateway_config.get("port", 5064)
            # Explicit configuration for connection method
            # Config system automatically converts "true"/"false" strings to booleans
            use_name_server = gateway_config.get("use_name_server", False)

            # Configure EPICS environment variables
            # Clear conflicting variables first — having both CA_ADDR_LIST and
            # CA_NAME_SERVERS set causes TCP connection attempts that block
            # asyncio.to_thread() worker threads.
            if use_name_server:
                # Use CA_NAME_SERVERS (required for SSH tunnels and some gateway configurations)
                os.environ["EPICS_CA_NAME_SERVERS"] = f"{address}:{port}"
                os.environ.pop("EPICS_CA_ADDR_LIST", None)
                os.environ.pop("EPICS_CA_SERVER_PORT", None)
                logger.debug(f"Using EPICS_CA_NAME_SERVERS: {address}:{port}")
            else:
                # Use CA_ADDR_LIST (standard gateway configuration)
                os.environ["EPICS_CA_ADDR_LIST"] = address
                os.environ["EPICS_CA_SERVER_PORT"] = str(port)
                os.environ.pop("EPICS_CA_NAME_SERVERS", None)
                logger.debug(f"Using EPICS_CA_ADDR_LIST: {address}, CA_SERVER_PORT: {port}")

            os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

            # Clear EPICS cache to pick up new environment
            self._epics.ca.clear_cache()

            logger.debug(f"Configured EPICS gateway: {address}:{port}")
            self._epics_configured = True

        self._timeout = config.get("timeout", 5.0)

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
            # the PVA client is not installed, mirroring the pyepics guard above.
            try:
                import p4p
                import p4p.client.thread  # noqa: F401  (loads the client API)

                self._p4p = p4p
            except ImportError:
                raise ImportError(
                    "p4p is required for PVA channels (control_system.connector."
                    "epics.pva_channels). Install with: pip install p4p"
                ) from None

            pva_gateway = config.get("pva_gateway") or {}
            if pva_gateway:
                pva_address = pva_gateway.get("address", "")
                pva_port = pva_gateway.get("port", 5075)
                # Config system automatically converts "true"/"false" to booleans
                pva_use_name_server = pva_gateway.get("use_name_server", False)
                # PVA carries the port inside the address entry itself — there is
                # no client-side "server port" variable to set, unlike CA.
                pva_endpoint = f"{pva_address}:{pva_port}" if pva_port else pva_address
                if pva_use_name_server:
                    # Name servers make the client connect by TCP instead of
                    # UDP-searching — required for SSH tunnels.
                    os.environ["EPICS_PVA_NAME_SERVERS"] = pva_endpoint
                    os.environ.pop("EPICS_PVA_ADDR_LIST", None)
                    logger.debug(f"Using EPICS_PVA_NAME_SERVERS: {pva_endpoint}")
                else:
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

        # Initialize limits validator for automatic validation and verification config
        from osprey_connectors.control_system.limits_validator import LimitsValidator

        self._limits_validator = LimitsValidator.from_config()
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

        # Disconnect and clear cached PVs
        with self._pv_cache_lock:
            for pv in self._pv_cache.values():
                try:
                    pv.disconnect()
                except Exception:
                    pass  # Best effort cleanup
            self._pv_cache.clear()

        self._connected = False
        logger.info("EPICS connector disconnected")

    async def read_channel(
        self, channel_address: str, timeout: float | None = None
    ) -> ChannelValue:
        """
        Read current value from EPICS channel.

        Addresses matching one of the configured ``pva_channels`` globs are read
        over PVAccess; every other address is read over Channel Access. Both
        paths are blocking client calls, so both run in a worker thread.

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

        # Use asyncio.to_thread for blocking EPICS operations
        if self._is_pva_channel(channel_address):
            return await asyncio.to_thread(self._read_channel_pva, channel_address, timeout)

        pv_result = await asyncio.to_thread(self._read_channel_sync, channel_address, timeout)

        return pv_result

    def _read_channel_sync(self, pv_address: str, timeout: float) -> ChannelValue:
        """Synchronous PV read (runs in thread pool).

        Uses PV cache to reuse PV objects for the same channel address.
        This prevents subscription floods when reading the same channel rapidly,
        which can crash soft IOCs like caproto due to race conditions.
        """
        # Get or create cached PV object (thread-safe)
        with self._pv_cache_lock:
            if pv_address not in self._pv_cache:
                self._pv_cache[pv_address] = self._epics.PV(pv_address)
            pv = self._pv_cache[pv_address]

        pv.wait_for_connection(timeout=timeout)

        if not pv.connected:
            raise ConnectionError(
                f"Failed to connect to PV '{pv_address}' (timeout after {timeout}s)"
            )

        # Use pv.get() with explicit timeout instead of pv.value.
        # pv.value uses a 1s default timeout for ca.get() which is too short
        # when running in asyncio.to_thread() worker threads where the CA
        # context needs extra time to receive the first value.
        value = pv.get(timeout=timeout)

        # Get timestamp from EPICS (seconds since epoch), rendered in facility tz
        if pv.timestamp:
            timestamp = datetime.fromtimestamp(pv.timestamp, get_facility_timezone())
        else:
            timestamp = datetime.now(get_facility_timezone())

        # Extract metadata. The alarm status is reported by name; the raw CA
        # code stays in raw_metadata next to the severity so nothing is lost.
        alarm_code = getattr(pv, "status", None)
        pv_type = getattr(pv, "type", None)
        labels, label = _ca_enum_labels(pv, value) if _is_enum_type(pv_type) else (None, None)
        metadata = ChannelMetadata(
            units=getattr(pv, "units", "") or "",
            precision=getattr(pv, "precision", None),
            alarm_status=_alarm_name(alarm_code),
            timestamp=timestamp,
            enum_labels=labels,
            enum_label=label,
            raw_metadata={
                "status": alarm_code,
                "severity": getattr(pv, "severity", None),
                "type": pv_type,
                "count": getattr(pv, "count", None),
            },
        )

        return ChannelValue(value=value, timestamp=timestamp, metadata=metadata)

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

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,
        verification_level: str | None = None,
        tolerance: float | None = None,
    ) -> ChannelWriteResult:
        """
        Write value to EPICS channel with automatic limits validation and verification.

        The connector automatically:
        1. Validates limits (min/max/step/writable) if limits checking enabled
        2. Determines verification level from per-channel or global config
        3. Executes write with appropriate verification

        Args:
            channel_address: EPICS channel address
            value: Value to write
            timeout: Timeout in seconds
            verification_level: Optional override for verification level (auto-determined if None)
            tolerance: Optional override for tolerance (auto-calculated if None)

        Returns:
            ChannelWriteResult with write status and verification details

        Raises:
            ConnectionError: If channel cannot be connected
            TimeoutError: If operation times out
            ChannelLimitsViolationError: If limits validation fails (when enabled)
        """
        # Step 0: PVA-routed addresses are read-only. This check MUST stay the
        # first statement of the method: _get_verification_config() below calls
        # float(value), which raises TypeError on an array — the refusal would
        # never be reached for exactly the values PVA channels carry.
        if self._is_pva_channel(channel_address):
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                success=False,
                error_message=(
                    f"Write to '{channel_address}' refused: PVAccess writes are not supported. "
                    "This address matches a configured 'pva_channels' pattern, so it is routed "
                    "over PVAccess and is read-only in this deployment. No write was attempted."
                ),
                blocked=True,
                refusal_reason="VALIDATION_ERROR",
            )

        timeout = timeout or self._timeout

        # Import here to avoid circular dependency
        from osprey_connectors.errors import ChannelLimitsViolationError

        # Step 1: Auto-determine verification config if not provided (cheap, on-loop)
        if verification_level is None:
            verification_level, auto_tolerance = self._get_verification_config(
                channel_address, float(value)
            )
            if tolerance is None:
                tolerance = auto_tolerance
        elif verification_level == "readback" and tolerance is None:
            # An explicit level must not silently drop the configured tolerance:
            # resolve it from the same chain (channel limits entry -> limits
            # defaults -> global config) and keep the caller's level. Without
            # this, naming "readback" fell back to 0.001 absolute — far tighter
            # than the percentage tolerances shipped for setpoint channels.
            try:
                _, tolerance = self._get_verification_config(channel_address, float(value))
            except (TypeError, ValueError):
                # Non-numeric value: no percentage tolerance is computable. The
                # readback comparison below reports the failure in context.
                tolerance = None

        # Step 2: Validate the verification level up front — reject invalid levels
        # BEFORE issuing any caput (preserves the no-caput-on-invalid-level behavior).
        if verification_level not in ("none", "callback", "readback"):
            raise ValueError(
                f"Invalid verification_level: {verification_level}. Must be 'none', 'callback', or 'readback'"
            )

        # callback/readback wait for the IOC callback; none does not.
        wait = verification_level != "none"

        # Step 3: Validate limits (FAIL CLOSED) and issue the caput in ONE thread
        # offload, so a caller on the event loop is never stalled by the blocking
        # caget that max_step validation performs.

        # A control-system denial (IOC access security answering the put with
        # "Write access denied") is not a client bug and must not be reported as
        # one. It is caught by class, resolved off the module this connector
        # actually connected with — this module must never import pyepics
        # (tests/connectors/test_import_isolation.py, and the readonly runtime
        # forbids it outright). An empty tuple catches nothing, so a connector
        # whose client library does not expose the class behaves exactly as
        # before.
        _refusal_class = getattr(getattr(self._epics, "ca", None), "CASeverityException", None)
        control_system_refusals: tuple[type[BaseException], ...] = (
            (_refusal_class,)
            if isinstance(_refusal_class, type) and issubclass(_refusal_class, BaseException)
            else ()
        )

        def _validate_and_put():
            if self._limits_validator:
                try:
                    self._limits_validator.validate(channel_address, value)
                    logger.debug(f"✓ Limits validation passed: {channel_address}={value}")
                except ChannelLimitsViolationError:
                    raise  # limits refusal propagates unchanged (carries LIMITS semantics)
                except Exception as e:
                    # FAIL CLOSED: any other validation error refuses the write — no caput issued
                    logger.warning(
                        f"Validation error — refusing write (fail-closed): {channel_address}: {e}"
                    )
                    return ("refused", e)  # sentinel: no caput issued
            try:
                success = self._epics.caput(channel_address, value, wait=wait, timeout=timeout)
            except control_system_refusals as e:
                # The control system was asked and said no. Every other caput
                # error stays a raised failure: it leaves the outcome genuinely
                # unknown, and a refusal claims the opposite.
                return ("cs_refused", e)
            return ("ok", success)

        outcome, payload = await asyncio.to_thread(_validate_and_put)

        if outcome == "cs_refused":
            logger.warning(f"Control system refused write to {channel_address}: {payload}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                success=False,
                error_message=(
                    f"Write to '{channel_address}' refused by the control system "
                    f"(access security); no value was written: {payload}"
                ),
                blocked=True,
                refusal_reason="CONTROL_SYSTEM_REFUSED",
            )

        if outcome == "refused":
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                success=False,
                error_message=f"Write to '{channel_address}' refused: validation error: {payload}",
                blocked=True,
                refusal_reason="VALIDATION_ERROR",
            )

        success = payload

        # Step 4: Build the per-level result (observable output identical to before)
        if verification_level == "none":
            # Fast path - no verification, no wait for callback
            if not success:
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=False,
                    verification=WriteVerification(
                        level="none", verified=False, notes="Write command failed"
                    ),
                    error_message=f"Failed to write to channel '{channel_address}'",
                )

            logger.debug(f"EPICS write (no verification): {channel_address} = {value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                success=True,
                verification=WriteVerification(
                    level="none", verified=False, notes="No verification requested"
                ),
            )

        elif verification_level == "callback":
            # EPICS callback - the caput above waited for the IOC to confirm processing
            if not success:
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=False,
                    verification=WriteVerification(
                        level="callback", verified=False, notes="IOC callback failed or timeout"
                    ),
                    error_message=f"Failed to write to channel '{channel_address}'",
                )

            logger.debug(f"EPICS write (callback verified): {channel_address} = {value}")
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                success=True,
                verification=WriteVerification(
                    level="callback", verified=True, notes="IOC callback confirmed"
                ),
            )

        elif verification_level == "readback":
            # Full verification - the caput above waited (callback); now read back
            if not success:
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=False,
                    verification=WriteVerification(
                        level="readback", verified=False, notes="Write command failed"
                    ),
                    error_message=f"Failed to write to channel '{channel_address}'",
                )

            # Read back to verify
            try:
                readback = await self.read_channel(channel_address, timeout=timeout)

                # Check tolerance
                diff = abs(float(readback.value) - float(value))
                verified = diff <= (tolerance or 0.001)
                alarm_status, alarm_severity = _readback_alarm_fields(readback)

                logger.debug(
                    f"EPICS write (readback verified={verified}): {channel_address} = {value}, "
                    f"readback = {readback.value}, diff = {diff:.6f}, tolerance = {tolerance}"
                )

                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=True,
                    verification=WriteVerification(
                        level="readback",
                        verified=verified,
                        readback_value=float(readback.value),
                        tolerance_used=tolerance,
                        notes=(
                            f"Readback: {readback.value}, tolerance: ±{tolerance}, diff: {diff:.6f}"
                            if verified
                            else f"Readback mismatch: {readback.value} (expected {value}, diff: {diff:.6f} > tolerance {tolerance})"
                        ),
                        readback_alarm_status=alarm_status,
                        readback_alarm_severity=alarm_severity,
                        # A value mismatch is not a failure *kind*: the readback
                        # itself worked. Only the except branch below classifies.
                    ),
                )

            except Exception as e:
                logger.warning(f"EPICS readback failed for {channel_address}: {e}")
                return ChannelWriteResult(
                    channel_address=channel_address,
                    value_written=value,
                    success=True,  # Write succeeded, but readback failed
                    verification=WriteVerification(
                        level="readback",
                        verified=False,
                        notes=f"Readback failed: {str(e)}",
                        failure_kind="readback_failed",
                    ),
                    error_message=f"Readback verification failed: {str(e)}",
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
        """
        loop = asyncio.get_event_loop()

        if self._is_pva_channel(channel_address):
            return self._subscribe_pva(channel_address, callback, loop)

        def epics_callback(pvname=None, value=None, timestamp=None, **kwargs):
            """Wrapper to convert EPICS callback to our format.

            pyepics hands a monitor callback a copy of the PV's whole argument
            set, so ``enum_strs`` is among the kwargs — carrying the labels once
            the PV's control variables have been fetched, and ``None`` until
            then (pyepics does not fetch them on connect). Both cases are
            handled: an update with no labels reports the index alone rather
            than being dropped or delayed by a synchronous fetch on the
            transport's own callback thread.
            """
            alarm_code = kwargs.get("status")
            labels, label = _enum_label_fields(kwargs.get("enum_strs"), value)
            pv_value = ChannelValue(
                value=value,
                timestamp=(
                    datetime.fromtimestamp(timestamp, get_facility_timezone())
                    if timestamp
                    else datetime.now(get_facility_timezone())
                ),
                metadata=ChannelMetadata(
                    units=kwargs.get("units", ""),
                    alarm_status=_alarm_name(alarm_code),
                    enum_labels=labels,
                    enum_label=label,
                    raw_metadata={
                        "status": alarm_code,
                        "severity": kwargs.get("severity"),
                    },
                ),
            )
            # Schedule callback in event loop
            loop.call_soon_threadsafe(callback, pv_value)

        # Create PV and add callback
        pv = self._epics.PV(channel_address, callback=epics_callback)

        # Generate subscription ID
        sub_id = f"{channel_address}_{id(pv)}"
        self._subscriptions[sub_id] = _ChannelSubscription("ca", pv)

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
        because pyepics reports units and precision off the PV it just read.
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
        timeout = 2.0
        try:
            if self._is_pva_channel(channel_address):
                await asyncio.to_thread(self._read_metadata_pva, channel_address, timeout)
            else:
                await self.read_channel(channel_address, timeout=timeout)
            return True
        except Exception as e:
            logger.debug(f"Channel validation failed for {channel_address}: {e}")
            return False
