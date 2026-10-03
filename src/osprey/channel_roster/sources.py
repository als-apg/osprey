"""Which source a build's channel roster is enumerated from, and how it is read.

One question, answered once: *given this project's configuration, what does the
roster read?* The facility file a build writes at the root of every render
(``<render root>/facility.json``), and nothing else. The channel-finder mode
does not choose it and no config key declares it: a project with no
channel-finder mode enumerates the same file a graph-mode project does, so a
facility has one enumeration of its own channels whoever asks.

The render root is the directory holding the ``config.yml`` in play:
``config_dir`` when the config records one (:func:`_render_dir`), else the
parent of :func:`osprey_connectors.workspace.resolve_config_path`.

:func:`read_facility_roster` turns the file's channel records into the
roster's. Membership is the ``channels`` list verbatim --- the addresses a
simulator serves for its own models' status are not channel records and are
never listed. Direction is the record's ``role``: a ``setpoint`` is settable,
a ``readback`` is readable, and ``none`` states no direction. A setpoint's
readback is its ``pair``, unless the pair is the setpoint itself.

Failure is data, never an exception: a file no build has written is a
:attr:`~osprey.channel_roster.records.RosterAbsenceReason.FACILITY_NOT_BUILT`
absence, one that is there and cannot be read is
:attr:`~osprey.channel_roster.records.RosterAbsenceReason.CORRUPT_SOURCE`, and
one that holds no channel is
:attr:`~osprey.channel_roster.records.RosterAbsenceReason.FACILITY_EMPTY`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.channel_roster.records import (
    ChannelDirection,
    ChannelRecord,
    RosterAbsence,
    RosterAbsenceReason,
    RosterResult,
    RosterSource,
    RosterSourceKind,
)
from osprey.facility import FACILITY_FILE
from osprey.utils.logger import get_logger

logger = get_logger("channel_roster.sources")

#: The direction each facility-file ``role`` states. ``none`` states neither,
#: which the roster carries as an unknown rather than as readable.
_ROLE_DIRECTIONS: Mapping[str, ChannelDirection | None] = {
    "setpoint": "write",
    "readback": "read",
    "none": None,
}


@dataclass(frozen=True, slots=True)
class RosterSourceResolution:
    """What this build's roster reads, or the reason it reads nothing.

    Attributes:
        source: The resolved source, or ``None`` when there is none.
        absence: Why there is no source, or ``None`` when there is one.

    Raises:
        ValueError: If the resolution says both or neither (a source *and* an
            absence, or neither). A caller branching on ``source is None``
            would otherwise silently take the wrong arm.
    """

    source: RosterSource | None = None
    absence: RosterAbsence | None = None

    def __post_init__(self) -> None:
        if (self.source is None) == (self.absence is None):
            raise ValueError(
                "A roster source resolution names exactly one of a source or an absence."
            )


def _render_dir(config: dict) -> Path | None:
    """Directory a render-relative configured path is authored against.

    :func:`osprey.deployment.compose_generator.prepare_compose_files` records
    ``config_dir`` --- the directory the loaded ``config.yml`` sits in --- before
    any render helper runs, and a build records the render it is writing the
    same way. That directory is the render root, which is where the facility
    file sits.

    Returning None lets :func:`facility_file_path` fall back to the
    ``config.yml`` this process runs against --- ``OSPREY_CONFIG`` when set,
    else ``build/config.yml`` under the working directory when that file
    exists, else the working directory itself
    (:func:`osprey_connectors.workspace.resolve_config_path`).

    Deliberately NOT the ``project_root`` rung of
    :func:`~osprey.deployment.compose_generator._render_anchor_dir`:
    ``project_root`` is the repo root, which is the wrong anchor for a
    render-relative key.

    Args:
        config: Full project configuration dictionary.

    Returns:
        The recorded config directory, or None when the config carries none.
    """
    raw = config.get("config_dir")
    if isinstance(raw, str) and raw.strip():
        return Path(raw)
    return None


def facility_file_path(config: dict) -> Path:
    """Where the facility file of the render this config belongs to sits.

    Args:
        config: Full project configuration dictionary.

    Returns:
        ``<render root>/facility.json``. The file is not probed: whether a
        build has written it is the reader's answer.
    """
    render_dir = _render_dir(config)
    if render_dir is None:
        from osprey.utils.workspace import resolve_config_path

        render_dir = Path(resolve_config_path()).parent
    return render_dir / FACILITY_FILE


def resolve_roster_source(config: dict) -> RosterSourceResolution:
    """Decide what this project's channel roster is enumerated from.

    Every project resolves to its render's facility file, whatever
    channel-finder mode it configures and whether or not it configures one.

    Args:
        config: Full project configuration dictionary, as the build holds it.

    Returns:
        The resolution, naming the facility file. The source is spelled by its
        file name, because the resolved path of a render being built is a
        staging path nobody can retype.
    """
    path = facility_file_path(config)
    return RosterSourceResolution(
        source=RosterSource(kind=RosterSourceKind.FACILITY, path=path, spelled=path.name)
    )


def read_facility_roster(source: RosterSource) -> RosterResult:
    """Read every channel record the facility file holds.

    Args:
        source: The facility file, as :func:`resolve_roster_source` settled it.

    Returns:
        A :class:`~osprey.channel_roster.records.RosterResult` holding one
        record per channel, in the file's order; or one carrying a
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.FACILITY_NOT_BUILT`
        absence when the file is not there, a
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.CORRUPT_SOURCE`
        one when it is there and is not a facility file this reader can turn
        into records, or an
        :attr:`~osprey.channel_roster.records.RosterAbsenceReason.FACILITY_EMPTY`
        one when it holds no channel.
    """
    try:
        text = source.path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(
            f"The facility file {source.for_display()} is not built, so this build "
            "enumerates no channels."
        )
        return RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.FACILITY_NOT_BUILT,
                path=source.path,
                spelled=source.spelled,
            )
        )
    except (OSError, UnicodeDecodeError) as e:
        return _corrupt(source, str(e))

    try:
        records = tuple(_record(channel, source) for channel in json.loads(text)["channels"])
    except (ValueError, KeyError, TypeError) as e:
        return _corrupt(source, _failure(e))

    if not records:
        absence = RosterAbsence(
            reason=RosterAbsenceReason.FACILITY_EMPTY,
            path=source.path,
            spelled=source.spelled,
        )
        # A project that declares no channels is a state of the project, not a
        # fault of the read.
        logger.info(absence.message())
        return RosterResult(absence=absence)
    return RosterResult(records=records, source=source)


def _corrupt(source: RosterSource, detail: str) -> RosterResult:
    """The absence for a facility file that is there and cannot be used."""
    logger.warning(f"The facility file {source.for_display()} could not be read ({detail}).")
    return RosterResult(
        absence=RosterAbsence(
            reason=RosterAbsenceReason.CORRUPT_SOURCE,
            path=source.path,
            spelled=source.spelled,
            detail=detail,
        )
    )


def _failure(error: Exception) -> str:
    """Say what was wrong with the file, without a sentence terminator."""
    if isinstance(error, KeyError):
        return f"a record has no {error.args[0]!r}"
    return str(error).rstrip(".") or type(error).__name__


def _record(channel: Mapping[str, Any], source: RosterSource) -> ChannelRecord:
    """Turn one facility-file channel record into the roster's.

    Raises:
        KeyError: The record has no ``id`` or no ``role``.
        ValueError: The record's ``role`` is not one the facility file allows,
            or the record is not one :class:`ChannelRecord` accepts.
        TypeError: The record, or its ``on``, is not a mapping.
    """
    address = channel["id"]
    role = channel["role"]
    if role not in _ROLE_DIRECTIONS:
        raise ValueError(f"channel {address} has the role {role!r}")
    pair = channel.get("pair")
    return ChannelRecord(
        address=address,
        source=source,
        direction=_ROLE_DIRECTIONS[role],
        readback=pair if role == "setpoint" and pair != address else None,
        role=role,
        value_type=channel.get("value_type"),
        description=channel.get("description"),
        on=_on(channel.get("on")),
    )


def _on(target: Mapping[str, Any] | None) -> tuple[str, str] | None:
    """The one device or place an ``on`` names, as ``(kind, id)``."""
    if not target:
        return None
    for kind in ("device", "place"):
        if target.get(kind):
            return (kind, str(target[kind]))
    return None
