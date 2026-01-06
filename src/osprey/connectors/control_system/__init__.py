"""Control system connector implementations."""

from osprey.connectors.control_system.base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    PVMetadata,  # Deprecated alias
    PVValue,  # Deprecated alias
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
