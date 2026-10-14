"""The control-system connector contract.

``ControlSystemConnector`` and the value, metadata and write-result models it
returns, the write-outcome vocabulary, and the limits validator every connector
checks a write against. A new control system subclasses the connector here; a
configured one is built through :class:`osprey_connectors.factory.ConnectorFactory`
from its type name. The concrete connectors are the ``_*_connector`` modules
beside this one.

The limits validator's names are resolved on first access (PEP 562). Every
import of :mod:`osprey_connectors.control_system.write_door` runs this file
first, and the raw-put guard reads the door inside executor sandboxes and
notebook kernels; importing the validator here would load its logging and YAML
stack into every one of them.
"""

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from osprey_connectors.control_system._limits_validator import (
        DEFAULTS_FIELD,
        LIMITS_DATABASE_CONFIG_KEY,
        METADATA_FIELDS,
        ChannelLimitsConfig,
        LimitsValidator,
        mapping_config_lookup,
    )

__all__ = [
    "DEFAULTS_FIELD",
    "LIMITS_DATABASE_CONFIG_KEY",
    "METADATA_FIELDS",
    "ChannelLimitsConfig",
    "ChannelMetadata",
    "ChannelValue",
    "ChannelWriteResult",
    "ControlSystemConnector",
    "LimitsValidator",
    "WriteOutcome",
    "is_readonly_run",
    "mapping_config_lookup",
    "raise_for_write_result",
    "values_match",
]

#: Public name -> the module that defines it, for the names resolved lazily.
_LAZY_EXPORTS: dict[str, str] = {
    "DEFAULTS_FIELD": "osprey_connectors.control_system._limits_validator",
    "LIMITS_DATABASE_CONFIG_KEY": "osprey_connectors.control_system._limits_validator",
    "METADATA_FIELDS": "osprey_connectors.control_system._limits_validator",
    "ChannelLimitsConfig": "osprey_connectors.control_system._limits_validator",
    "LimitsValidator": "osprey_connectors.control_system._limits_validator",
    "mapping_config_lookup": "osprey_connectors.control_system._limits_validator",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
