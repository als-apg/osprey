"""Fold alarm-server topic messages into what each PV's alarm is now.

The alarm server's compacted ``<config>`` topic carries ``state:/path`` and
``config:/path`` keys holding JSON, and a null value (or a ``{"delete": ...}``
marker just before it) when the item leaves the tree. Other keys such as
``command:`` say nothing about a PV. The Kafka and replay backends both feed
:func:`fold_message`, so a replay exercises the real fold.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from osprey_connectors.alarm_state._contract import (
    _ACK_SUFFIX,
    AlarmCondition,
    AlarmConfiguration,
    AlarmEntry,
    AlarmSeverity,
)
from osprey_connectors.errors import require_aware
from osprey_connectors.service_answers import NotListed, _flag, _instant, _text

FEED_LOADING = "alarm feed still loading"
FEED_EMPTY = "no message from the feed yet"

_UNESCAPED_SLASH = re.compile(r"(?<!\\)/")
_PV_SCHEME = re.compile(r"^(?:ca|pva)://")


@dataclass
class FeedSnapshot:
    """Every message folded so far; each PV maps its tree paths to their latest value."""

    messages: int = 0
    states: dict[str, dict[str, Mapping[str, object]]] = field(default_factory=dict)
    configs: dict[str, dict[str, Mapping[str, object]]] = field(default_factory=dict)


def fold_message(
    snapshot: FeedSnapshot,
    key: str | bytes | None,
    value: str | bytes | Mapping[str, object] | None,
    *,
    event_at: datetime | None = None,
) -> FeedSnapshot:
    """Fold one message into *snapshot* and return it.

    It touches nothing but *snapshot*, updated in place so that a topic of N
    messages folds in O(N). Every message counts towards ``messages``, including
    one that says nothing about a PV or cannot be parsed: the feed spoke.
    *event_at*, an aware instant when given, replaces the time the value records.
    """
    if event_at is not None:
        require_aware(event_at, name="event_at")
    snapshot.messages += 1
    kind, path = _split_key(key)
    if kind is None or path is None:
        return snapshot
    table = snapshot.states if kind == "state" else snapshot.configs
    pv = pv_name(_UNESCAPED_SLASH.split(path)[-1])
    body = _parse_value(value)
    if body is _UNPARSEABLE:
        return snapshot
    if body is None or "delete" in body:
        paths = table.get(pv)
        if paths is not None:
            paths.pop(path, None)
            if not paths:
                del table[pv]
        return snapshot
    if event_at is not None:
        body = {**body, "time": event_at}
    paths = table.setdefault(pv, {})
    paths.pop(path, None)  # the latest update is the last entry
    paths[path] = body
    return snapshot


def state_record(
    snapshot: FeedSnapshot, address: str, *, stale: bool
) -> AlarmCondition | NotListed:
    """The alarm state of *address*, or :class:`NotListed` when the feed holds none for it."""
    paths = snapshot.states.get(pv_name(address))
    if not paths:
        return NotListed(address=address)
    return _state_of(list(paths.values())[-1], stale=stale)


def state_rows(snapshot: FeedSnapshot, *, stale: bool) -> tuple[AlarmEntry, ...]:
    """One row per channel whose latest state is a leaf's, under the path it was reported on.

    A leaf's state carries ``current_severity``. The alarm server also publishes
    the state of each tree node, with ``severity`` alone, filed under the node's
    last path segment: that name is not a channel.
    """
    rows = []
    for channel, paths in snapshot.states.items():
        path, body = list(paths.items())[-1]
        if "current_severity" in body:
            rows.append(
                AlarmEntry(address=channel, path=path, condition=_state_of(body, stale=stale))
            )
    return tuple(rows)


def _state_of(body: Mapping[str, object], *, stale: bool) -> AlarmCondition:
    recorded = body.get("severity")
    severity, acknowledged = AlarmSeverity.of_recorded(recorded)
    bare = None if recorded is None else str(recorded).removesuffix(_ACK_SUFFIX)
    return AlarmCondition(
        severity=severity,
        status=_text(body.get("current_message")),
        latched=not acknowledged
        and _rank_or_worst(body.get("current_severity")) < _rank_or_worst(bare),
        acknowledged=acknowledged,
        message=_text(body.get("message")),
        event_at=_event_at(body.get("time")),
        stale=stale,
    )


def _rank_or_worst(word: object) -> int:
    """The rank of a recorded severity word, higher is worse.

    ``None`` and a word that is no severity rank above ``UNDEFINED``, so an
    unknown severity is never taken for a recovery.
    """
    try:
        return AlarmSeverity(word).rank if isinstance(word, str) else len(AlarmSeverity)
    except ValueError:
        return len(AlarmSeverity)


def config_record(snapshot: FeedSnapshot, address: str) -> AlarmConfiguration:
    """Where *address* sits in the alarm tree, as the feed last configured it.

    A state with no configuration beside it proves the path but not the tree's
    entry, so that answers ``"unknown"``; only no message at all answers ``"no"``.
    """
    name = pv_name(address)
    paths = snapshot.configs.get(name)
    if not paths:
        states = snapshot.states.get(name)
        if states:
            return AlarmConfiguration(in_tree="unknown", path=list(states)[-1])
        return AlarmConfiguration(in_tree="no")
    path, body = list(paths.items())[-1]
    return AlarmConfiguration(
        in_tree="yes",
        path=path,
        guidance=body.get("guidance"),
        displays=body.get("displays"),
        enabled=_flag(body.get("enabled")),
        latching=_flag(body.get("latching")),
    )


def pv_name(tree_name: str) -> str:
    """A tree item's name as the PV it watches: escaped slashes undone, scheme dropped."""
    return _PV_SCHEME.sub("", tree_name.replace("\\/", "/"))


class _Unparseable(Enum):
    """A value that is neither a mapping nor a tombstone."""

    VALUE = "unparseable"


_UNPARSEABLE = _Unparseable.VALUE


def _split_key(key: str | bytes | None) -> tuple[str | None, str | None]:
    if isinstance(key, bytes):
        key = key.decode("utf-8", errors="replace")
    if not key:
        return None, None
    kind, separator, path = key.partition(":")
    if not separator or kind not in ("state", "config") or not path.startswith("/"):
        return None, None
    return kind, path


def _parse_value(
    value: str | bytes | Mapping[str, object] | None,
) -> Mapping[str, object] | Literal[_Unparseable.VALUE] | None:
    """The value as a mapping, ``None`` for a tombstone, ``_UNPARSEABLE`` otherwise."""
    if isinstance(value, Mapping):
        return value
    if not value:
        return None
    try:
        parsed: object = json.loads(value)
    except (ValueError, UnicodeDecodeError):
        return _UNPARSEABLE
    return parsed if isinstance(parsed, dict) else _UNPARSEABLE


def _event_at(recorded: object) -> datetime | None:
    """The instant a state's ``time`` names: the alarm server's ``{seconds, nano}``, in UTC.

    An aware ``datetime`` (a replay's rebased time) or ISO-8601 text with a zone
    is read as it is; anything else, text with no zone included, names no instant.
    """
    if not isinstance(recorded, Mapping):
        return _instant(recorded)
    try:
        return datetime.fromtimestamp(recorded["seconds"] + recorded.get("nano", 0) / 1e9, UTC)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None
