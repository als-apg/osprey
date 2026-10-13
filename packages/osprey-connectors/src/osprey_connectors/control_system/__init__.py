"""The control-system connector contract.

``ControlSystemConnector`` and the value, metadata and write-result models it
returns, the write-outcome vocabulary, the per-call timeout keys, and the
limits validator every connector checks a write against. A new control system
subclasses the connector here; a configured one is built through
:class:`osprey_connectors.factory.ConnectorFactory` from its type name. The
concrete connectors are the ``_*_connector`` modules beside this one.
"""

from osprey_connectors.control_system._base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
    is_readonly_run,
    raise_for_write_result,
    values_match,
)
from osprey_connectors.control_system._call_timeout import (
    DEFAULT_TIMEOUT_S,
    TIMEOUT_KEY,
    refuse_renamed_timeout_keys,
)
from osprey_connectors.control_system._limits_validator import (
    DEFAULTS_FIELD,
    LIMITS_DATABASE_CONFIG_KEY,
    METADATA_FIELDS,
    ChannelLimitsConfig,
    LimitsValidator,
    mapping_config_lookup,
)

__all__ = [
    "ChannelLimitsConfig",
    "ChannelMetadata",
    "ChannelValue",
    "ChannelWriteResult",
    "ControlSystemConnector",
    "DEFAULTS_FIELD",
    "DEFAULT_TIMEOUT_S",
    "LIMITS_DATABASE_CONFIG_KEY",
    "LimitsValidator",
    "METADATA_FIELDS",
    "TIMEOUT_KEY",
    "WriteOutcome",
    "is_readonly_run",
    "mapping_config_lookup",
    "raise_for_write_result",
    "refuse_renamed_timeout_keys",
    "values_match",
]
