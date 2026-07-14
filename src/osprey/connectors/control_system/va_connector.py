"""
Virtual Accelerator control system connector.

Provides the ``virtual_accelerator`` connector type: a config-selectable
seam distinct from ``mock`` and ``epics`` for facilities running a
containerized PyAT-backed soft-IOC as their control system.

"""

from osprey.connectors.control_system.epics_connector import EPICSConnector
from osprey.utils.logger import get_logger

logger = get_logger("va_connector")


class VirtualAcceleratorConnector(EPICSConnector):
    """
    Virtual Accelerator control system connector.

    A soft-IOC backed by a physics simulation engine (e.g. PyAT) is
    indistinguishable from real hardware to the Channel Access protocol
    stack, so this connector inherits :class:`EPICSConnector` unmodified —
    the same gateway/name-server configuration and read/write/subscribe
    behavior apply.

    This subclass exists as a seam: it gives the ``virtual_accelerator``
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
        >>>             'port': 5074,
        >>>             'use_name_server': True
        >>>         }
        >>>     }
        >>> }
        >>> connector = VirtualAcceleratorConnector()
        >>> await connector.connect(config)
        >>> value = await connector.read_channel('BEAM:CURRENT')
    """
