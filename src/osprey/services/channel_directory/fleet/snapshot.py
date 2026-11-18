"""One target's fleet snapshot (C5): the one definition every surface reads.

A refresher builds a :class:`Snapshot` at the end of each cycle and writes it
with :func:`write_snapshot`; the fleet service reads it with
:func:`read_snapshot` and serves :meth:`Snapshot.to_json` unchanged. The field
names of that JSON form are fixed here and nowhere else.

A snapshot has an ``outcome``:

* ``ok`` -- a cycle ran: the read path, the sources, the configured grouping
  field and one row per IOC, in C4 order (:func:`~.states.order_key`). Counts
  by state and by finding are derived from the rows, never stored apart from
  them.
* ``no_snapshot`` -- nothing has been written for the target yet (or what is
  on disk cannot be read).
* ``not_refreshed`` -- the target is not refreshed (D3, D12).
* ``cannot_list`` -- no listing of the target's directory has ever succeeded
  (D7, D8).

Every non-``ok`` outcome is a value of the same type carrying its reason, so a
reader handles one type. Times are seconds since the epoch.

The file is replaced whole (:func:`write_snapshot`); :func:`read_snapshot`
never raises on the file's content: a missing or unreadable file reads as
``no_snapshot`` with the reason.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from osprey.services.channel_directory.fleet.settings import GROUP_BY
from osprey.services.channel_directory.fleet.states import (
    Finding,
    HealthReading,
    HealthStatus,
    LivenessKind,
    ReadPath,
    State,
    order_key,
)
from osprey_connectors.safety_file import atomic_replace

__all__ = [
    "Counts",
    "Liveness",
    "Outcome",
    "ReadPathStatus",
    "ServerFleetRow",
    "Snapshot",
    "SnapshotFormatError",
    "SourceStatus",
    "read_snapshot",
    "snapshot_path",
    "write_snapshot",
]

logger = logging.getLogger(__name__)

_FILE_MODE = 0o644
_FILE_PREFIX = "fleet-snapshot-"
_FILE_SUFFIX = ".json"


class Outcome(StrEnum):
    OK = "ok"
    NO_SNAPSHOT = "no_snapshot"
    NOT_REFRESHED = "not_refreshed"
    CANNOT_LIST = "cannot_list"


class SnapshotFormatError(ValueError):
    """A snapshot's JSON form does not have the C5 shape."""


@dataclass(frozen=True, kw_only=True)
class ReadPathStatus:
    """The target's read-path PV this cycle; ``channel`` is ``None`` when none is configured."""

    status: ReadPath
    channel: str | None


@dataclass(frozen=True, kw_only=True)
class SourceStatus:
    """How one source of the cycle (the directory, a roster source) answered.

    ``outcome`` is the source's own word (``ok``, ``unavailable``, ...);
    ``reason`` says why when it is not ``ok``, and may carry a listing's age.
    """

    name: str
    outcome: str
    reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class Liveness:
    """An IOC's liveness PV and its last reading; ``channel`` is ``None`` when it has none.

    ``ambiguous`` is true when several PVs matched and the rule picked one.
    """

    channel: str | None
    kind: LivenessKind
    value: Any = None
    read_at: float | None = None
    ambiguous: bool = False


@dataclass(frozen=True, kw_only=True)
class ServerFleetRow:
    """One IOC's row.

    ``since`` is when it entered ``state``; ``directory_status`` the status the
    directory gives it (``None`` when it gives none); ``listed`` whether the
    directory's listing has it, false for an IOC known only from memory or a
    roster; ``sources`` the roster sources that list it.
    """

    name: str
    host: str | None
    class_name: str
    group: str | None
    state: State
    since: float | None
    reason: str
    findings: tuple[Finding, ...]
    directory_status: str | None
    listed: bool
    sources: tuple[str, ...]
    liveness: Liveness
    health: tuple[HealthReading, ...] = ()

    def __post_init__(self) -> None:
        if self.state is State.UNKNOWN and self.findings:
            raise ValueError(f"{self.name}: an unknown IOC carries no finding")

    def to_json(self) -> dict[str, Any]:
        """The row's C5 JSON form."""
        return {
            "name": self.name,
            "host": self.host,
            "class_name": self.class_name,
            "group": self.group,
            "state": self.state.value,
            "since": self.since,
            "reason": self.reason,
            "findings": [f.value for f in self.findings],
            "directory_status": self.directory_status,
            "listed": self.listed,
            "sources": list(self.sources),
            "liveness": {
                "channel": self.liveness.channel,
                "kind": self.liveness.kind.value,
                "value": self.liveness.value,
                "read_at": self.liveness.read_at,
                "ambiguous": self.liveness.ambiguous,
            },
            "health": [
                {"channel": h.channel, "value": h.value, "status": h.status.value}
                for h in self.health
            ],
        }


@dataclass(frozen=True, kw_only=True)
class Counts:
    """IOCs per state and per finding; every state and finding has an entry."""

    by_state: Mapping[str, int]
    by_finding: Mapping[str, int]

    @classmethod
    def of(cls, rows: Iterable[ServerFleetRow]) -> Counts:
        by_state = dict.fromkeys((s.value for s in State), 0)
        by_finding = dict.fromkeys((f.value for f in Finding), 0)
        for row in rows:
            by_state[row.state.value] += 1
            for finding in row.findings:
                by_finding[finding.value] += 1
        return cls(by_state=by_state, by_finding=by_finding)


@dataclass(frozen=True, kw_only=True)
class Snapshot:
    """One target's snapshot; build it with :meth:`ok` or :meth:`not_ok`."""

    outcome: Outcome
    target: str
    reason: str | None = None
    generated_at: float | None = None
    cycle: int | None = None
    period_s: float | None = None
    cycle_duration_s: float | None = None
    read_path: ReadPathStatus | None = None
    sources: tuple[SourceStatus, ...] = ()
    group_by: str | None = None
    servers: tuple[ServerFleetRow, ...] = ()
    counts: Counts = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("a snapshot names its target")
        if self.outcome is Outcome.OK:
            missing = [
                name
                for name in ("generated_at", "cycle", "period_s", "cycle_duration_s", "read_path")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"an ok snapshot needs {', '.join(missing)}")
            if self.reason is not None:
                raise ValueError("an ok snapshot carries no reason")
            if self.group_by not in GROUP_BY:
                raise ValueError(f"group_by must be one of {GROUP_BY}, got {self.group_by!r}")
            names = [row.name for row in self.servers]
            if len(set(names)) != len(names):
                raise ValueError("an IOC appears twice in the snapshot")
            ordered = tuple(sorted(self.servers, key=_row_key))
            object.__setattr__(self, "servers", ordered)
        else:
            if not self.reason:
                raise ValueError(f"a {self.outcome.value} snapshot needs a reason")
            if self.servers or self.sources or self.read_path is not None:
                raise ValueError(f"a {self.outcome.value} snapshot carries no rows")
        object.__setattr__(self, "counts", Counts.of(self.servers))

    # --- Construction -----------------------------------------------------------

    @classmethod
    def ok(
        cls,
        *,
        target: str,
        generated_at: float,
        cycle: int,
        period_s: float,
        cycle_duration_s: float,
        read_path: ReadPathStatus,
        sources: Iterable[SourceStatus],
        group_by: str,
        servers: Iterable[ServerFleetRow],
    ) -> Snapshot:
        """A cycle's snapshot; the rows are put in C4 order."""
        return cls(
            outcome=Outcome.OK,
            target=target,
            generated_at=generated_at,
            cycle=cycle,
            period_s=period_s,
            cycle_duration_s=cycle_duration_s,
            read_path=read_path,
            sources=tuple(sources),
            group_by=group_by,
            servers=tuple(servers),
        )

    @classmethod
    def not_ok(cls, outcome: Outcome | str, *, target: str, reason: str) -> Snapshot:
        """A ``no_snapshot``, ``not_refreshed`` or ``cannot_list`` answer for ``target``."""
        outcome = Outcome(outcome)
        if outcome is Outcome.OK:
            raise ValueError("use Snapshot.ok for an ok snapshot")
        return cls(outcome=outcome, target=target, reason=reason)

    # --- Reading ----------------------------------------------------------------

    def groups(self) -> dict[str | None, tuple[ServerFleetRow, ...]]:
        """The rows grouped by the configured field, each group in C4 order.

        Groups are ordered by their key, the group of rows with no value last.
        """
        if self.group_by is None:
            return {}
        grouped: dict[str | None, list[ServerFleetRow]] = {}
        for row in self.servers:
            grouped.setdefault(getattr(row, self.group_by), []).append(row)
        keys = sorted(grouped, key=lambda key: (key is None, key or ""))
        return {key: tuple(grouped[key]) for key in keys}

    def row(self, name: str) -> ServerFleetRow | None:
        """The row of IOC ``name``, or ``None`` when the snapshot has none."""
        return next((row for row in self.servers if row.name == name), None)

    # --- JSON form ----------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """The C5 JSON form."""
        if self.outcome is not Outcome.OK:
            return {"outcome": self.outcome.value, "target": self.target, "reason": self.reason}
        assert self.read_path is not None  # guaranteed by __post_init__
        return {
            "outcome": self.outcome.value,
            "target": self.target,
            "generated_at": self.generated_at,
            "cycle": self.cycle,
            "period_s": self.period_s,
            "cycle_duration_s": self.cycle_duration_s,
            "read_path": {"status": self.read_path.status.value, "channel": self.read_path.channel},
            "sources": [
                {"name": s.name, "outcome": s.outcome, "reason": s.reason} for s in self.sources
            ],
            "counts": {
                "by_state": dict(self.counts.by_state),
                "by_finding": dict(self.counts.by_finding),
            },
            "group_by": self.group_by,
            "servers": [row.to_json() for row in self.servers],
        }

    @classmethod
    def from_json(cls, document: Any) -> Snapshot:
        """The snapshot of a C5 JSON form; :class:`SnapshotFormatError` when it is not one."""
        try:
            return _decode(document)
        except SnapshotFormatError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotFormatError(f"not a snapshot: {exc!r}") from exc


def _row_key(row: ServerFleetRow) -> tuple[int, int, str]:
    return order_key(row.state, row.findings, row.name)


# --- Files ------------------------------------------------------------------------


def snapshot_path(directory: Path, target: str) -> Path:
    """The snapshot file of ``target`` in the state directory."""
    if not target or "/" in target or "\\" in target or target.startswith("."):
        raise ValueError(f"not a target name usable in a file name: {target!r}")
    return Path(directory) / f"{_FILE_PREFIX}{target}{_FILE_SUFFIX}"


def write_snapshot(directory: Path, snapshot: Snapshot) -> Path:
    """Replace the target's snapshot file, whole or not at all, and return its path.

    The directory is created when missing. A failure to write raises
    :class:`OSError` and leaves the previous file as it was.
    """
    path = snapshot_path(directory, snapshot.target)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(snapshot.to_json(), indent=2, allow_nan=False).encode("utf-8")
    atomic_replace(path, data, mode=_FILE_MODE, sync="required")
    return path


def read_snapshot(directory: Path, target: str) -> Snapshot:
    """The target's stored snapshot; ``no_snapshot`` with the reason when there is none to read."""
    path = snapshot_path(directory, target)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return Snapshot.not_ok(
            Outcome.NO_SNAPSHOT, target=target, reason="no snapshot has been written yet"
        )
    except OSError as exc:
        return _unreadable(target, f"snapshot file {path} could not be read ({exc})")
    try:
        snapshot = Snapshot.from_json(json.loads(raw.decode("utf-8")))
    except (SnapshotFormatError, ValueError) as exc:
        return _unreadable(target, f"snapshot file {path} is corrupt ({exc})")
    if snapshot.target != target:
        return _unreadable(
            target, f"snapshot file {path} names target {snapshot.target!r}, not {target!r}"
        )
    return snapshot


def _unreadable(target: str, problem: str) -> Snapshot:
    logger.warning("%s", problem)
    return Snapshot.not_ok(Outcome.NO_SNAPSHOT, target=target, reason=problem)


def _decode(document: Any) -> Snapshot:
    doc = _object(document, "snapshot")
    outcome = Outcome(doc["outcome"])
    target = _text(doc["target"])
    if outcome is not Outcome.OK:
        _exact_keys(doc, {"outcome", "target", "reason"}, "snapshot")
        return Snapshot.not_ok(outcome, target=target, reason=_text(doc["reason"]))
    _exact_keys(
        doc,
        {
            "outcome",
            "target",
            "generated_at",
            "cycle",
            "period_s",
            "cycle_duration_s",
            "read_path",
            "sources",
            "counts",
            "group_by",
            "servers",
        },
        "snapshot",
    )
    read_path = _object(doc["read_path"], "read_path")
    snapshot = Snapshot.ok(
        target=target,
        generated_at=_number(doc["generated_at"]),
        cycle=_integer(doc["cycle"]),
        period_s=_number(doc["period_s"]),
        cycle_duration_s=_number(doc["cycle_duration_s"]),
        read_path=ReadPathStatus(
            status=ReadPath(read_path["status"]), channel=_optional_text(read_path["channel"])
        ),
        sources=[
            SourceStatus(
                name=_text(s["name"]),
                outcome=_text(s["outcome"]),
                reason=_optional_text(s["reason"]),
            )
            for s in (_object(s, "source") for s in _list(doc["sources"], "sources"))
        ],
        group_by=_text(doc["group_by"]),
        servers=[_decode_row(row) for row in _list(doc["servers"], "servers")],
    )
    counts = _object(doc["counts"], "counts")
    stored = {"by_state": counts["by_state"], "by_finding": counts["by_finding"]}
    derived = {
        "by_state": dict(snapshot.counts.by_state),
        "by_finding": dict(snapshot.counts.by_finding),
    }
    if stored != derived:
        raise SnapshotFormatError("the counts do not match the rows")
    return snapshot


def _decode_row(value: Any) -> ServerFleetRow:
    row = _object(value, "IOC row")
    liveness = _object(row["liveness"], "liveness")
    return ServerFleetRow(
        name=_text(row["name"]),
        host=_optional_text(row["host"]),
        class_name=_text(row["class_name"]),
        group=_optional_text(row["group"]),
        state=State(row["state"]),
        since=_optional_number(row["since"]),
        reason=_text(row["reason"]),
        findings=tuple(Finding(f) for f in _list(row["findings"], "findings")),
        directory_status=_optional_text(row["directory_status"]),
        listed=_boolean(row["listed"]),
        sources=tuple(_text(s) for s in _list(row["sources"], "sources")),
        liveness=Liveness(
            channel=_optional_text(liveness["channel"]),
            kind=LivenessKind(liveness["kind"]),
            value=liveness["value"],
            read_at=_optional_number(liveness["read_at"]),
            ambiguous=_boolean(liveness["ambiguous"]),
        ),
        health=tuple(
            HealthReading(
                channel=_text(h["channel"]), value=h["value"], status=HealthStatus(h["status"])
            )
            for h in (_object(h, "health") for h in _list(row["health"], "health"))
        ),
    )


def _object(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SnapshotFormatError(f"{what} is not an object: {value!r}")
    return value


def _exact_keys(doc: Mapping[str, Any], keys: set[str], what: str) -> None:
    if set(doc) != keys:
        extra, missing = sorted(set(doc) - keys), sorted(keys - set(doc))
        raise SnapshotFormatError(f"{what} keys differ (extra {extra}, missing {missing})")


def _list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise SnapshotFormatError(f"{what} is not a list: {value!r}")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise SnapshotFormatError(f"expected text, found {value!r}")
    return value


def _optional_text(value: Any) -> str | None:
    return None if value is None else _text(value)


def _boolean(value: Any) -> bool:
    if not isinstance(value, bool):
        raise SnapshotFormatError(f"expected true or false, found {value!r}")
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise SnapshotFormatError(f"expected a number, found {value!r}")
    return float(value)


def _optional_number(value: Any) -> float | None:
    return None if value is None else _number(value)


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotFormatError(f"expected an integer, found {value!r}")
    return int(value)
