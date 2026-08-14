"""
Virtual Accelerator control system connector.

Provides the ``virtual_accelerator`` connector type: a config-selectable
seam distinct from ``mock`` and ``epics`` for facilities running a
containerized PyAT-backed soft-IOC as their control system.

"""

from typing import Any

from osprey_connectors.control_system.epics_connector import EPICSConnector
from osprey_connectors.logger import get_logger

logger = get_logger("va_connector")

#: Channel Access port the shipped Virtual Accelerator soft-IOC serves on.
DEFAULT_VA_PORT = 5064


def resolve_va_gateway_port() -> Any:
    """Resolve the Channel Access port of the Virtual Accelerator to talk to.

    Reads ``services.virtual_accelerator.port`` — the same key the VA compose
    service publishes and sets ``EPICS_CA_SERVER_PORT`` from — so a project
    that moves the deployed soft-IOC's port moves the connector with it.
    Falls back to :data:`DEFAULT_VA_PORT` when the key is unset, and when the
    config cannot be read at all (no project context, unreadable file), so the
    connector still resolves a port without one.

    Returns:
        The port to default-fill unset gateway ports with.
    """
    from osprey_connectors.config import get_config_value

    try:
        return get_config_value("services.virtual_accelerator.port", DEFAULT_VA_PORT)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError):
        return DEFAULT_VA_PORT


def fill_gateway_ports(config: dict[str, Any]) -> dict[str, Any]:
    """Default-fill every gateway's ``port`` from the deployed VA's port.

    Precedence:
      1. An explicit gateway ``port`` wins, verbatim — that is how a project
         points at a VA it does not deploy itself (another host, or a
         container published on a different host port than it serves on).
      2. Otherwise the port comes from ``services.virtual_accelerator.port``,
         via :func:`resolve_va_gateway_port`.

    Args:
        config: The ``connector.virtual_accelerator`` config block.

    Returns:
        A copy of ``config`` whose gateways all carry a ``port``. The caller's
        dict is never mutated. Returned unchanged when there is nothing to
        fill, so a fully explicit config never reads the config file.
    """
    gateways = config.get("gateways")
    if not isinstance(gateways, dict):
        return config

    unfilled = [
        name
        for name, gateway in gateways.items()
        if isinstance(gateway, dict) and "port" not in gateway
    ]
    if not unfilled:
        return config

    port = resolve_va_gateway_port()
    logger.debug(
        f"Virtual Accelerator gateways {sorted(unfilled)} have no explicit port; "
        f"using services.virtual_accelerator.port ({port})"
    )

    filled = {
        name: ({**gateway, "port": port} if name in unfilled else gateway)
        for name, gateway in gateways.items()
    }
    return {**config, "gateways": filled}


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
        >>>     'timeout': 5.0,
        >>>     'gateways': {
        >>>         'read_only': {
        >>>             'address': 'localhost',
        >>>             'use_name_server': True
        >>>         }
        >>>     }
        >>> }
        >>> connector = VirtualAcceleratorConnector()
        >>> await connector.connect(config)
        >>> value = await connector.read_channel('BEAM:CURRENT')
    """

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Configure the EPICS environment against the deployed Virtual Accelerator.

        Args:
            config: Same shape as :meth:`EPICSConnector.connect`, except that
                a gateway may omit ``port`` to follow
                ``services.virtual_accelerator.port``.
        """
        await super().connect(fill_gateway_ports(config))
