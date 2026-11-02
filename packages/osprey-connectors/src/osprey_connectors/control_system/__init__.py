"""The control-system connector contract.

``ControlSystemConnector`` and the value, metadata and write-result models it
returns, the write-outcome vocabulary, and the limits validator every connector
checks a write against. A new control system subclasses the connector here; a
configured one is built through :class:`osprey_connectors.factory.ConnectorFactory`
from its type name. The concrete connectors are the ``_*_connector`` modules
beside this one.

The limits validator's names are resolved on access (PEP 562). Every
import of :mod:`osprey_connectors.control_system.write_door` runs this file
first, and the raw-put guard reads the door inside executor sandboxes and
notebook kernels; importing the validator here would load its logging and YAML
stack into every one of them, and importing the EPICS connector for its stale
hold-down would load its Channel Access stack.
"""

from typing import TYPE_CHECKING, Any

from osprey_connectors.control_system._base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
    mark_long_lived_session,
    raise_for_write_result,
    values_match,
)

if TYPE_CHECKING:
    from osprey_connectors.control_system._epics_connector import STALE_HOLD_DOWN_S
    from osprey_connectors.control_system._limits_validator import (
        DEFAULT_STEP_READ_TIMEOUT_SECONDS,
        LIMITS_DATABASE_CONFIG_KEY,
        STEP_READ_TIMEOUT_KEY,
        ChannelLimitsConfig,
        LimitsValidator,
        mapping_config_lookup,
        step_read_timeout_seconds,
    )

__all__ = [
    "DEFAULT_STEP_READ_TIMEOUT_SECONDS",
    "LIMITS_DATABASE_CONFIG_KEY",
    "STALE_HOLD_DOWN_S",
    "STEP_READ_TIMEOUT_KEY",
    "ChannelLimitsConfig",
    "ChannelMetadata",
    "ChannelValue",
    "ChannelWriteResult",
    "ControlSystemConnector",
    "LimitsValidator",
    "WriteOutcome",
    "mapping_config_lookup",
    "mark_long_lived_session",
    "raise_for_write_result",
    "step_read_timeout_seconds",
    "values_match",
]

#: Public name -> the module that defines it, for the names resolved lazily.
_LAZY_EXPORTS: dict[str, str] = {
    "STALE_HOLD_DOWN_S": "osprey_connectors.control_system._epics_connector",
    "DEFAULT_STEP_READ_TIMEOUT_SECONDS": "osprey_connectors.control_system._limits_validator",
    "LIMITS_DATABASE_CONFIG_KEY": "osprey_connectors.control_system._limits_validator",
    "STEP_READ_TIMEOUT_KEY": "osprey_connectors.control_system._limits_validator",
    "ChannelLimitsConfig": "osprey_connectors.control_system._limits_validator",
    "LimitsValidator": "osprey_connectors.control_system._limits_validator",
    "mapping_config_lookup": "osprey_connectors.control_system._limits_validator",
    "step_read_timeout_seconds": "osprey_connectors.control_system._limits_validator",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module, on every access.

    Not cached here: the defining module stays the one place a name is bound,
    so a name replaced there (a test's monkeypatch, say) is what every later
    lookup through this namespace gets.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
