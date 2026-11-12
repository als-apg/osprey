"""What is alarming now, per the ``AlarmState`` contract.

The backends are the underscore-private modules beside this one, which the
factory reaches from their type names.
"""

from osprey_connectors.alarm_state._contract import (
    ALARM_STALE_WINDOW_S,
    AlarmConfigRecord,
    AlarmState,
    AlarmStateRecord,
)

__all__ = [
    "ALARM_STALE_WINDOW_S",
    "AlarmConfigRecord",
    "AlarmState",
    "AlarmStateRecord",
]
