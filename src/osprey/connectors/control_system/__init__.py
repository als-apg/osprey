"""Control system connector implementations."""

from osprey.connectors.control_system.base import PVMetadata  # Deprecated alias
from osprey.connectors.control_system.base import PVValue  # Deprecated alias
from osprey.connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteVerification,
)

__all__ = [
    "ControlSystemConnector",
    "ChannelValue",
    "ChannelMetadata",
    "ChannelWriteResult",
    "WriteVerification",
    # Deprecated aliases (backward compatibility)
    "PVValue",
    "PVMetadata",
]
