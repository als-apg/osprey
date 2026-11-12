"""The one place the framework imports the connectors' read-only service modules.

The framework runs on connectors releases that predate those modules, so the
names below are ``None`` there and :data:`DIRECTORY_CONNECTORS_AVAILABLE` is
false. Every other framework module imports this one, never the connectors'
modules, and checks the flag before touching a contract type.

:func:`unavailable_payload` and :func:`as_payload` hold no contract type, so they
work on both paths: a service answers the same JSON shape whether or not the
installed connectors have the modules.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import Enum
from typing import Any

try:
    from osprey_connectors.alarm_log import AlarmHistoryRow, AlarmLog
    from osprey_connectors.alarm_state import (
        ALARM_STALE_WINDOW_S,
        AlarmConfigRecord,
        AlarmState,
        AlarmStateRecord,
    )
    from osprey_connectors.directory import (
        DEFAULT_DESCRIPTION_PROPERTY,
        ROW_FIELD_PROPERTIES,
        SERVED_SIGNAL_PROPERTY,
        ChannelDirectory,
        DirectoryPage,
        DirectoryQuery,
        DirectoryRecord,
        DirectoryRow,
    )
    from osprey_connectors.put_log import PutLog, PutRow
    from osprey_connectors.service_answers import (
        BackendSpec,
        HistoryPage,
        Outcome,
        Source,
        Unavailable,
    )
    from osprey_connectors.service_targets import service_targets
    from osprey_connectors.simulation.view import ENGINEER, IOC_HEALTH_SIGNALS
    from osprey_connectors.simulator import has_simulator

    DIRECTORY_CONNECTORS_AVAILABLE = True
except ImportError:
    from osprey_connectors.types import CONTROL_TARGETS as _FLOOR_STAND_IN  # noqa: F401

    DIRECTORY_CONNECTORS_AVAILABLE = False
    ALARM_STALE_WINDOW_S = None  # type: ignore[assignment]
    DEFAULT_DESCRIPTION_PROPERTY = None  # type: ignore[assignment]
    ROW_FIELD_PROPERTIES = None  # type: ignore[assignment]
    AlarmConfigRecord = None  # type: ignore[assignment,misc]
    AlarmHistoryRow = None  # type: ignore[assignment,misc]
    AlarmLog = None  # type: ignore[assignment,misc]
    AlarmState = None  # type: ignore[assignment,misc]
    AlarmStateRecord = None  # type: ignore[assignment,misc]
    BackendSpec = None  # type: ignore[assignment,misc]
    ChannelDirectory = None  # type: ignore[assignment,misc]
    ENGINEER = None  # type: ignore[assignment]
    IOC_HEALTH_SIGNALS = None  # type: ignore[assignment]
    DirectoryPage = None  # type: ignore[assignment,misc]
    DirectoryQuery = None  # type: ignore[assignment,misc]
    DirectoryRecord = None  # type: ignore[assignment,misc]
    DirectoryRow = None  # type: ignore[assignment,misc]
    HistoryPage = None  # type: ignore[assignment,misc]
    Outcome = None  # type: ignore[assignment,misc]
    PutLog = None  # type: ignore[assignment,misc]
    PutRow = None  # type: ignore[assignment,misc]
    SERVED_SIGNAL_PROPERTY = None  # type: ignore[assignment]
    Source = None  # type: ignore[assignment,misc]
    Unavailable = None  # type: ignore[assignment,misc]
    has_simulator = None  # type: ignore[assignment]
    service_targets = None  # type: ignore[assignment]

__all__ = [
    "ALARM_STALE_WINDOW_S",
    "DEFAULT_DESCRIPTION_PROPERTY",
    "DIRECTORY_CONNECTORS_AVAILABLE",
    "ENGINEER",
    "FLOOR_REASON",
    "IOC_HEALTH_SIGNALS",
    "NO_SOURCE_REASON",
    "ROW_FIELD_PROPERTIES",
    "SERVED_SIGNAL_PROPERTY",
    "AlarmConfigRecord",
    "AlarmHistoryRow",
    "AlarmLog",
    "AlarmState",
    "AlarmStateRecord",
    "BackendSpec",
    "ChannelDirectory",
    "DirectoryPage",
    "DirectoryQuery",
    "DirectoryRecord",
    "DirectoryRow",
    "HistoryPage",
    "Outcome",
    "PutLog",
    "PutRow",
    "Source",
    "Unavailable",
    "as_payload",
    "has_simulator",
    "service_targets",
    "unavailable_payload",
]

#: The reason every service answers when the installed connectors lack the modules.
FLOOR_REASON = "installed osprey-connectors has no PV directory support (upgrade it)"

#: The reason a section with no source on the current target answers.
NO_SOURCE_REASON = "not available for this target"

_NO_SOURCE_LABEL = "All PVs"


def unavailable_payload(reason: str, *, target: str) -> dict[str, Any]:
    """The JSON form of a section with no source on *target*, giving *reason*."""
    return {
        "outcome": "unavailable",
        "source": {
            "target": target,
            "service": "none",
            "label": _NO_SOURCE_LABEL,
            "rehearsal": target == "standin",
        },
        "reason": reason,
    }


def as_payload(result: Any) -> Any:
    """*result*, a contract dataclass or a container of them, as plain JSON values.

    Dataclasses become mappings field by field, enum members their values,
    mappings and sequences their JSON forms; anything else is returned as it is.
    """
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return {
            field.name: as_payload(getattr(result, field.name))
            for field in dataclasses.fields(result)
        }
    if isinstance(result, Enum):
        return result.value
    if isinstance(result, Mapping):
        return {key: as_payload(value) for key, value in result.items()}
    if isinstance(result, (list, tuple)):
        return [as_payload(value) for value in result]
    return result
