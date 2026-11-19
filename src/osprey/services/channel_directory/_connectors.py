"""The one place the framework imports the connectors' read-only service modules.

Every other framework module imports this one, never the connectors' modules.
:func:`unavailable_payload`, :func:`as_payload` and :func:`logbook_source` turn
the contract types into the JSON shape every service answers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import Enum
from typing import Any

from osprey.services.channel_directory.vocabulary import (
    LOGBOOK_SERVICE,
    Words,
    source_label,
    words_of,
)
from osprey_connectors.alarm_log import AlarmHistoryRow, AlarmLog
from osprey_connectors.alarm_state import (
    AlarmConfigRecord,
    AlarmState,
    AlarmStateRecord,
    AlarmStateRow,
    AlarmStates,
    severity_rank,
)
from osprey_connectors.archive_targets import APPLIANCE_TARGETS
from osprey_connectors.directory import (
    ChannelDirectory,
    DirectoryPage,
    DirectoryQuery,
    DirectoryRecord,
    DirectoryRow,
    ServerRow,
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
from osprey_connectors.simulation.view import (
    ENGINEER,
    SERVED_SIGNAL_PROPERTY,
    SERVER_HEALTH_SIGNALS,
)
from osprey_connectors.simulator import has_simulator

__all__ = [
    "APPLIANCE_TARGETS",
    "ENGINEER",
    "NO_SOURCE_REASON",
    "SERVED_SIGNAL_PROPERTY",
    "SERVER_HEALTH_SIGNALS",
    "AlarmConfigRecord",
    "AlarmHistoryRow",
    "AlarmLog",
    "AlarmState",
    "AlarmStateRecord",
    "AlarmStateRow",
    "AlarmStates",
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
    "SERVED_FIELD",
    "ServerRow",
    "Unavailable",
    "as_payload",
    "has_simulator",
    "logbook_source",
    "service_targets",
    "severity_rank",
    "unavailable_payload",
]

#: The reason a section with no source on the current target answers.
NO_SOURCE_REASON = "not available for this target"

#: The served name of each contract field routes, pages and tool answers carry:
#: the contract's own name, except ``channel_type``, served as ``record_type``.
SERVED_FIELD = {"server": "server", "channel_type": "record_type", "channel": "channel"}

#: The records whose fields are served under :data:`SERVED_FIELD`.
_SERVED_RECORDS = (DirectoryRow, PutRow, AlarmStateRow)


def unavailable_payload(reason: str, *, target: str, words: Words | None = None) -> dict[str, Any]:
    """The JSON form of a section with no source on *target*, giving *reason*.

    Its label is the directory's title in *target*'s *words* (the neutral words
    when none are given).
    """
    return {
        "outcome": "unavailable",
        "source": {
            "target": target,
            "service": "none",
            "label": words_of(words, target).directory,
            "rehearsal": target == "standin",
        },
        "reason": reason,
    }


def as_payload(result: Any, words: Words | None = None) -> Any:
    """*result*, a contract dataclass or a container of them, as plain JSON values.

    A ``Source`` becomes ``{target, service, label, rehearsal}``, its label named
    by :func:`~osprey.services.channel_directory.vocabulary.source_label` in its
    target's *words* (the neutral words when none are given); a directory
    row, a put row and an alarm-state row name their fields by :data:`SERVED_FIELD`. Other
    dataclasses become mappings field by field, enum members their values,
    mappings and sequences their JSON forms; anything else is returned as it is.
    """
    if isinstance(result, Source):
        return {
            "target": result.target,
            "service": result.service,
            "label": source_label(result.kind, result.service, words_of(words, result.target)),
            "rehearsal": result.rehearsal,
        }
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        names = SERVED_FIELD if isinstance(result, _SERVED_RECORDS) else {}
        return {
            names.get(field.name, field.name): as_payload(getattr(result, field.name), words)
            for field in dataclasses.fields(result)
        }
    if isinstance(result, Enum):
        return result.value
    if isinstance(result, Mapping):
        return {key: as_payload(value, words) for key, value in result.items()}
    if isinstance(result, (list, tuple)):
        return [as_payload(value, words) for value in result]
    return result


def logbook_source(target: str, *, ariel_configured: bool) -> Any:
    """The logbook *target* reads, or ``None``.

    The logbook belongs to the facility's machine, as its archive does, so it
    answers on the appliance targets only, and on ``standin`` as a rehearsal.
    """
    if not ariel_configured or target not in APPLIANCE_TARGETS:
        return None
    return Source(
        kind="logbook",
        target=target,  # type: ignore[arg-type]
        service=LOGBOOK_SERVICE,
        rehearsal=target == "standin",
    )
