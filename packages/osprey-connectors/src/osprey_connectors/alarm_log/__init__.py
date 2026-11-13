"""Logged alarm history and configuration, per the ``AlarmLog`` contract.

The backends are the underscore-private modules beside this one, which the
factory reaches from their type names.
"""

from osprey_connectors.alarm_log._contract import AlarmHistoryRow, AlarmLog

__all__ = [
    "AlarmHistoryRow",
    "AlarmLog",
]
