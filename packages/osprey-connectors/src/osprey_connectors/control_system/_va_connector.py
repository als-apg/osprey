"""
Virtual Accelerator control system connector.

Provides the ``virtual_accelerator`` connector type served over Channel
Access: a config-selectable seam distinct from ``epics`` for facilities
running a containerized PyAT-backed soft-IOC as their control system.

"""

from collections.abc import Mapping
from typing import TypeVar

from typing_extensions import override

from osprey_connectors._config_read import coerce_port
from osprey_connectors.config_sections import GatewayRow, Unreadable
from osprey_connectors.connector_types import DEFAULT_VA_PORT
from osprey_connectors.control_system._base import ControlSystemSettings, _settings_as
from osprey_connectors.control_system._epics_connector import EPICSConnector, EpicsSettings
from osprey_connectors.errors import ConfigurationError
from osprey_connectors.logger import get_logger
from osprey_connectors.model import evolve

logger = get_logger("va_connector")


def resolve_va_gateway_port(config_path: str | None = None) -> object:
    """Resolve the Channel Access port of the Virtual Accelerator to talk to.

    Reads ``services.virtual_accelerator.port`` — the same key the VA compose
    service publishes and sets ``EPICS_CA_SERVER_PORT`` from — so a project
    that moves the deployed soft-IOC's port moves the connector with it.
    Falls back to :data:`DEFAULT_VA_PORT` when the key is unset, and when the
    config cannot be read at all (no project context, unreadable file), so the
    connector still resolves a port without one.

    Args:
        config_path: The project config to read. ``None`` reads the one a child
            reads — ``OSPREY_CONFIG``, else ``build/config.yml`` or
            ``config.yml`` under the working directory.

    Returns:
        The port to default-fill unset gateway ports with.
    """
    from osprey_connectors.config import current_config, load_config
    from osprey_connectors.errors import ConfigLoadError

    try:
        config = current_config() if config_path is None else load_config(config_path)
        return config.get(_VA_PORT_KEY, DEFAULT_VA_PORT)
    except (ConfigLoadError, FileNotFoundError, KeyError, RuntimeError, ValueError):
        # Every way a config cannot be read: ConfigNotFoundError is a
        # FileNotFoundError, ConfigParseError and ConfigurationError ValueErrors,
        # and a directory a ConfigLoadError.
        return DEFAULT_VA_PORT


_Role = TypeVar("_Role", bound=str)

#: The key the deployed simulator's port is read from.
_VA_PORT_KEY = "services.virtual_accelerator.port"


def fill_gateway_ports(
    gateways: Mapping[_Role, GatewayRow | Unreadable], config_path: str | None = None
) -> dict[_Role, GatewayRow | Unreadable]:
    """Default-fill every gateway's ``port`` from the deployed VA's port.

    Precedence:
      1. An explicit gateway ``port`` wins, verbatim — that is how a project
         points at a VA it does not deploy itself (another host, or a
         container published on a different host port than it serves on).
      2. Otherwise the port comes from ``services.virtual_accelerator.port``,
         via :func:`resolve_va_gateway_port`.

    Args:
        gateways: The ``connector.virtual_accelerator`` block's gateways, by role.
        config_path: The project config ``services.virtual_accelerator.port``
            is read from; see :func:`resolve_va_gateway_port`.

    Returns:
        The rows, those that named no port carrying the VA's. An unreadable
        entry is returned as it is. The config file is read only when some row
        names no port.

    Raises:
        ConfigurationError: ``services.virtual_accelerator.port`` names no port.
    """
    unfilled = [
        role for role, row in gateways.items() if isinstance(row, GatewayRow) and row.port is None
    ]
    if not unfilled:
        return dict(gateways)
    written = resolve_va_gateway_port(config_path)
    port = coerce_port(written)
    if port is None:
        raise ConfigurationError(f"{_VA_PORT_KEY} is {written!r}, not a port", key=_VA_PORT_KEY)
    logger.debug(
        f"Virtual Accelerator gateways {sorted(unfilled)} have no explicit port; "
        f"using services.virtual_accelerator.port ({port})"
    )
    return {
        role: evolve(row, port=port) if role in unfilled and isinstance(row, GatewayRow) else row
        for role, row in gateways.items()
    }


class VirtualAcceleratorConnector(EPICSConnector):
    """
    Virtual Accelerator control system connector.

    A soft-IOC backed by a physics simulation engine (e.g. PyAT) is
    indistinguishable from real hardware to the Channel Access protocol
    stack, so this connector inherits :class:`EPICSConnector`'s
    read/write/subscribe behavior unmodified — the same gateway/name-server
    configuration applies.

    It differs in one respect: the VA is a service this project deploys, so
    an unset gateway ``port`` is default-filled from
    ``services.virtual_accelerator.port`` rather than left to the EPICS
    connector's generic default. A production EPICS gateway is external
    infrastructure and keeps its explicitly configured port.

    This subclass also exists as a seam: it gives the ``virtual_accelerator``
    control system type its own class identity, config namespace
    (``connector.virtual_accelerator``), and registration entry, so that
    future VA-specific behavior (e.g. simulation-aware metadata, scenario
    bookkeeping) can be added here without touching production EPICS
    behavior or requiring a config change for existing deployments.

    Example:
        >>> config = {
        >>>     'timeout_s': 5.0,
        >>>     'gateways': {
        >>>         'read_only': {
        >>>             'address': 'localhost',
        >>>             'use_name_server': True
        >>>         }
        >>>     }
        >>> }
        >>> connector = VirtualAcceleratorConnector()
        >>> await connector.connect(ConnectorContext(settings=config))
        >>> value = await connector.read_channel('BEAM:CURRENT')
    """

    @override
    async def _open(self, settings: ControlSystemSettings) -> None:
        """
        Configure the EPICS environment against the deployed Virtual Accelerator.

        Args:
            settings: Same shape as :meth:`EPICSConnector._open` takes, except
                that a gateway may omit ``port`` to follow
                ``services.virtual_accelerator.port``.
        """
        epics = _settings_as(settings, EpicsSettings)
        rows = fill_gateway_ports(epics.gateways)
        await super()._open(evolve(epics, gateways=rows))
