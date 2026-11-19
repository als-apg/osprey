"""The one place the framework imports the connectors' read-only service modules.

The framework runs on connectors releases that predate those modules, so the
names below are ``None`` there and :data:`DIRECTORY_CONNECTORS_AVAILABLE` is
false. Every other framework module imports this one, never the connectors'
modules, and checks the flag before touching a contract type.

:func:`unavailable_payload`, :func:`as_payload` and :func:`logbook_source` check
for the contract types they use, so they work on both paths: a service answers the same JSON shape whether or not the
installed connectors have the modules.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from osprey.services.pv_directory.vocabulary import DIRECTORY_LABEL, LOGBOOK_SERVICE, source_label

try:
    from osprey_connectors.alarm_log import AlarmLog, AlarmLogEntry
    from osprey_connectors.alarm_state import (
        AlarmCondition,
        AlarmConfiguration,
        AlarmEntry,
        AlarmState,
        AlarmStates,
    )
    from osprey_connectors.deployment.simulator import has_simulator
    from osprey_connectors.directory import (
        ChannelDirectory,
        DirectoryDetail,
        DirectoryEntry,
        DirectoryPage,
        DirectoryQuery,
    )
    from osprey_connectors.put_log import PutLog, PutLogEntry
    from osprey_connectors.service_answers import (
        BackendSpec,
        NotListed,
        Source,
        Unavailable,
    )
    from osprey_connectors.service_targets import APPLIANCE_TARGETS, service_targets
    from osprey_connectors.simulation.view import (
        ENGINEER,
        IOC_HEALTH_SIGNALS,
        SERVED_SIGNAL_PROPERTY,
    )

    DIRECTORY_CONNECTORS_AVAILABLE = True
except ImportError:
    from osprey_connectors.connector_types import CONTROL_TARGETS as _FLOOR_STAND_IN  # noqa: F401

    DIRECTORY_CONNECTORS_AVAILABLE = False
    APPLIANCE_TARGETS = None  # type: ignore[assignment]
    AlarmConfiguration = None  # type: ignore[assignment,misc]
    AlarmLogEntry = None  # type: ignore[assignment,misc]
    AlarmLog = None  # type: ignore[assignment,misc]
    AlarmState = None  # type: ignore[assignment,misc]
    AlarmCondition = None  # type: ignore[assignment,misc]
    AlarmEntry = None  # type: ignore[assignment,misc]
    AlarmStates = None  # type: ignore[assignment,misc]
    BackendSpec = None  # type: ignore[assignment,misc]
    ChannelDirectory = None  # type: ignore[assignment,misc]
    ENGINEER = None  # type: ignore[assignment,misc]
    IOC_HEALTH_SIGNALS = None  # type: ignore[assignment]
    DirectoryPage = None  # type: ignore[assignment,misc]
    DirectoryQuery = None  # type: ignore[assignment,misc]
    DirectoryDetail = None  # type: ignore[assignment,misc]
    DirectoryEntry = None  # type: ignore[assignment,misc]
    NotListed = None  # type: ignore[assignment,misc]
    PutLog = None  # type: ignore[assignment,misc]
    PutLogEntry = None  # type: ignore[assignment,misc]
    SERVED_SIGNAL_PROPERTY = None  # type: ignore[assignment]
    Source = None  # type: ignore[assignment,misc]
    Unavailable = None  # type: ignore[assignment,misc]
    has_simulator = None  # type: ignore[assignment]
    service_targets = None  # type: ignore[assignment]

__all__ = [
    "APPLIANCE_TARGETS",
    "DIRECTORY_CONNECTORS_AVAILABLE",
    "ENGINEER",
    "FLOOR_REASON",
    "IOC_HEALTH_SIGNALS",
    "NO_SOURCE_REASON",
    "SERVED_FIELD",
    "SERVED_INSTANT_FIELD",
    "SERVED_QUERY_FIELD",
    "SERVED_ROW_FIELD",
    "SERVED_SIGNAL_PROPERTY",
    "AlarmCondition",
    "AlarmConfiguration",
    "AlarmEntry",
    "AlarmLog",
    "AlarmLogEntry",
    "AlarmState",
    "AlarmStates",
    "BackendSpec",
    "ChannelDirectory",
    "DirectoryDetail",
    "DirectoryEntry",
    "DirectoryPage",
    "DirectoryQuery",
    "NotListed",
    "PutLog",
    "PutLogEntry",
    "Source",
    "Unavailable",
    "as_payload",
    "has_simulator",
    "logbook_source",
    "served_instant",
    "service_targets",
    "unavailable_payload",
]

#: The reason every service answers when the installed connectors lack the modules.
FLOOR_REASON = "installed osprey-connectors has no PV directory support (upgrade it)"

#: The reason a section with no source on the current target answers.
NO_SOURCE_REASON = "not available for this target"

#: The served name of each contract field the framework serves under its EPICS word,
#: or under the word the pages read: routes, pages and tool answers keep ``ioc``,
#: ``record_type`` and ``pv``, and a directory detail's entry is its ``row``.
SERVED_FIELD = {"server": "ioc", "channel_type": "record_type", "address": "pv", "entry": "row"}

#: A directory row's fields as served: its address is the row's ``name``.
SERVED_ROW_FIELD = {**SERVED_FIELD, "address": "name"}

#: The served keyword of each :class:`DirectoryQuery` field a search names
#: differently: the address glob is ``name``.
SERVED_QUERY_FIELD = {"server": "ioc", "channel_type": "record_type", "address_pattern": "name"}

#: The served name of each contract instant: the word the pages and tools have always
#: read, whatever record holds it. Each is served as :func:`served_instant` text, or,
#: where the record kept a stamp it could not place, as that ``recorded_stamp``'s text.
SERVED_INSTANT_FIELD = {
    "event_at": "time",
    "last_contact_at": "last_contact",
    "announced_at": "last_announced",
    "put_at": "at",
}

#: The field a record keeps an unplaced stamp in; it is served as the record's instant.
_RECORDED_STAMP = "recorded_stamp"

#: What the alarm logger appends to an acknowledged severity, served as it recorded it.
_ACK_SUFFIX = "_ACK"

#: The records whose fields are served under :data:`SERVED_FIELD`.
_SERVED_RECORDS = tuple(
    record for record in (PutLogEntry, AlarmEntry, DirectoryDetail) if record is not None
)


def _served_names(result: Any) -> Mapping[str, str]:
    """The served name of each of *result*'s fields that is served under another."""
    if DirectoryEntry is not None and isinstance(result, DirectoryEntry):
        return SERVED_ROW_FIELD
    return SERVED_FIELD if isinstance(result, _SERVED_RECORDS) else {}


def served_instant(moment: datetime | None) -> str | None:
    """*moment* as the services' served text: ISO-8601 in UTC to the millisecond, with ``Z``.

    The text the alarm feeds have always served, so a page reads every service's
    instants alike. ``None`` stays ``None``.
    """
    if moment is None:
        return None
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _model_payload(result: BaseModel) -> dict[str, Any]:
    """A contract record's fields as served: renamed, and an unplaced stamp as its instant."""
    names = {**SERVED_INSTANT_FIELD, **_served_names(result)}
    payload = {
        names.get(name, name): as_payload(value)
        for name, value in result
        if name != _RECORDED_STAMP
    }
    stamp = getattr(result, _RECORDED_STAMP, None)
    if stamp is not None:
        for name in type(result).model_fields:
            if name in SERVED_INSTANT_FIELD and payload[SERVED_INSTANT_FIELD[name]] is None:
                payload[SERVED_INSTANT_FIELD[name]] = stamp
    if AlarmLogEntry is not None and isinstance(result, AlarmLogEntry):
        _serve_recorded_severity(payload)
    return payload


def _serve_recorded_severity(payload: dict[str, Any]) -> None:
    """A logged alarm's severity as the logger recorded it: ``MAJOR_ACK``, not ``acknowledged``."""
    acknowledged = payload.pop("acknowledged")
    if acknowledged and payload["severity"] is not None:
        payload["severity"] = f"{payload['severity']}{_ACK_SUFFIX}"


def unavailable_payload(reason: str, *, target: str) -> dict[str, Any]:
    """The JSON form of a section with no source on *target*, giving *reason*."""
    return {
        "outcome": "unavailable",
        "source": {
            "target": target,
            "service": "none",
            "label": DIRECTORY_LABEL,
            "rehearsal": target == "standin",
        },
        "reason": reason,
    }


def as_payload(result: Any) -> Any:
    """*result*, a contract record or a container of them, as plain JSON values.

    A ``Source`` becomes ``{target, service, label, rehearsal}``, its label named
    by :func:`~osprey.services.pv_directory.vocabulary.source_label`; a directory
    entry names its fields by :data:`SERVED_ROW_FIELD`, a put-log entry, an alarm entry
    and a directory detail by :data:`SERVED_FIELD`, and every record its instants by
    :data:`SERVED_INSTANT_FIELD`; a logged alarm serves its severity as the logger
    recorded it, an acknowledged one with its ``_ACK``. Other records (pydantic models, and dataclasses)
    become mappings field by field, enum members their values, a ``datetime`` its
    :func:`served_instant` text, mappings and sequences their JSON forms; anything
    else is returned as it is.
    """
    if Source is not None and isinstance(result, Source):
        return {
            "target": result.target,
            "service": result.service,
            "label": source_label(result.kind, result.service),
            "rehearsal": result.rehearsal,
        }
    if isinstance(result, BaseModel):
        return _model_payload(result)
    if isinstance(result, datetime):
        return served_instant(result)
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        names = _served_names(result)
        return {
            names.get(field.name, field.name): as_payload(getattr(result, field.name))
            for field in dataclasses.fields(result)
        }
    if isinstance(result, Enum):
        return result.value
    if isinstance(result, Mapping):
        return {key: as_payload(value) for key, value in result.items()}
    if isinstance(result, (list, tuple)):
        return [as_payload(value) for value in result]
    return result


def logbook_source(target: str, *, ariel_configured: bool) -> Any:
    """The logbook *target* reads, or ``None``.

    The logbook belongs to the facility's machine, as its archive does, so it
    answers on the appliance targets only, and on ``standin`` as a rehearsal.
    Without the PV directory modules it is ``None``.
    """
    if Source is None or not ariel_configured or target not in APPLIANCE_TARGETS:
        return None
    return Source(
        kind="logbook",
        target=target,  # type: ignore[arg-type]
        service=LOGBOOK_SERVICE,
        rehearsal=target == "standin",
    )
