"""The control system's own directory of channels, behind the ``ChannelDirectory`` contract.

The backends are the underscore-private modules beside this one, which the
factory reaches from their type names.
"""

from osprey_connectors.directory._contract import (
    DEFAULT_DESCRIPTION_PROPERTY,
    ROW_FIELD_PROPERTIES,
    SERVED_SIGNAL_PROPERTY,
    ChannelDirectory,
    DirectoryPage,
    DirectoryQuery,
    DirectoryRecord,
    DirectoryRow,
)

__all__ = [
    "DEFAULT_DESCRIPTION_PROPERTY",
    "ROW_FIELD_PROPERTIES",
    "SERVED_SIGNAL_PROPERTY",
    "ChannelDirectory",
    "DirectoryPage",
    "DirectoryQuery",
    "DirectoryRecord",
    "DirectoryRow",
]
