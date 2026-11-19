"""What is alarming now, per the ``AlarmState`` contract.

The backends are the underscore-private modules beside this one, which the
factory reaches from their type names.
"""

from osprey_connectors.alarm_state._contract import (
    AlarmCondition,
    AlarmConfiguration,
    AlarmEntry,
    AlarmSeverity,
    AlarmState,
    AlarmStates,
)

__all__ = [
    "AlarmCondition",
    "AlarmConfiguration",
    "AlarmEntry",
    "AlarmSeverity",
    "AlarmState",
    "AlarmStates",
]
