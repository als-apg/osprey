"""Framework-native control system capabilities.

These capabilities were moved from Jinja2 templates to native framework
modules in v0.11. They provide the standard control system operations
that most accelerator/beamline applications need.

Capabilities:
    - channel_finding: Resolve natural-language queries to control system channel addresses
    - channel_read: Read current values from control system channels
    - channel_write: Write values to control system channels (requires human approval)
    - archiver_retrieval: Retrieve historical time-series data from archiver systems
"""

from osprey.capabilities.archiver_retrieval import ArchiverDataContext, ArchiverRetrievalCapability
from osprey.capabilities.channel_finding import ChannelAddressesContext, ChannelFindingCapability
from osprey.capabilities.channel_read import (
    ChannelReadCapability,
    ChannelValue,
    ChannelValuesContext,
)
from osprey.capabilities.channel_write import (
    ChannelWriteCapability,
    ChannelWriteResult,
    ChannelWriteResultsContext,
    WriteVerificationInfo,
)

__all__ = [
    # Capabilities
    "ChannelFindingCapability",
    "ChannelReadCapability",
    "ChannelWriteCapability",
    "ArchiverRetrievalCapability",
    # Context classes
    "ChannelAddressesContext",
    "ChannelValuesContext",
    "ChannelWriteResultsContext",
    "ArchiverDataContext",
    # Helper models
    "ChannelValue",
    "ChannelWriteResult",
    "WriteVerificationInfo",
]
