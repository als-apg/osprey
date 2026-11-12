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
from typing import Any

from osprey_connectors.alarm_state._contract import AlarmConfigRecord, AlarmStateRecord

FEED_LOADING = "alarm feed still loading"
FEED_EMPTY = "no message from the feed yet"

_ACK_SUFFIX = "_ACK"
_SEVERITY_RANK = {"OK": 0, "MINOR": 1, "MAJOR": 2, "INVALID": 3, "UNDEFINED": 4}
_UNESCAPED_SLASH = re.compile(r"(?<!\\)/")
_PV_SCHEME = re.compile(r"^(?:ca|pva)://")


@dataclass
class FeedSnapshot:
    """Every message folded so far; each PV maps its tree paths to their latest value."""

    messages: int = 0
    states: dict[str, dict[str, Mapping[str, Any]]] = field(default_factory=dict)
    configs: dict[str, dict[str, Mapping[str, Any]]] = field(default_factory=dict)


def fold_message(
    snapshot: FeedSnapshot,
    key: str | bytes | None,
    value: str | bytes | Mapping[str, Any] | None,
    *,
    time: str | None = None,
) -> FeedSnapshot:
    """Fold one message into *snapshot* and return it.

    It touches nothing but *snapshot*, updated in place so that a topic of N
    messages folds in O(N). Every message counts towards ``messages``, including
    one that says nothing about a PV or cannot be parsed: the feed spoke. *time*,
    when given, replaces the time the value records.
    """
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
    if time is not None:
        body = {**body, "time": time}
    paths = table.setdefault(pv, {})
    paths.pop(path, None)  # the latest update is the last entry
    paths[path] = body
    return snapshot


def state_record(snapshot: FeedSnapshot, channel: str, *, stale: bool) -> AlarmStateRecord | None:
    """The alarm state of *channel*, or ``None`` when the feed holds none for it."""
    paths = snapshot.states.get(pv_name(channel))
    if not paths:
        return None
    body = list(paths.values())[-1]
    recorded = body.get("severity")
    severity = recorded.removesuffix(_ACK_SUFFIX) if isinstance(recorded, str) else recorded
    acknowledged = severity != recorded
    return AlarmStateRecord(
        severity=severity,
        status=body.get("current_message"),
        latched=not acknowledged and _rank(body.get("current_severity")) < _rank(severity),
        acknowledged=acknowledged,
        message=body.get("message"),
        time=_recorded_time(body.get("time")),
        stale=stale,
    )


def config_record(snapshot: FeedSnapshot, channel: str) -> AlarmConfigRecord:
    """Where *channel* sits in the alarm tree, as the feed last configured it.

    A state with no configuration beside it proves the path but not the tree's
    entry, so that answers ``"unknown"``; only no message at all answers ``"no"``.
    """
    name = pv_name(channel)
    paths = snapshot.configs.get(name)
    if not paths:
        states = snapshot.states.get(name)
        if states:
            return AlarmConfigRecord(in_tree="unknown", path=list(states)[-1])
        return AlarmConfigRecord(in_tree="no")
    path, body = list(paths.items())[-1]
    return AlarmConfigRecord(
        in_tree="yes",
        path=path,
        guidance=body.get("guidance"),
        displays=body.get("displays"),
        enabled=body.get("enabled"),
        latching=body.get("latching"),
    )


def pv_name(tree_name: str) -> str:
    """A tree item's name as the PV it watches: escaped slashes undone, scheme dropped."""
    return _PV_SCHEME.sub("", tree_name.replace("\\/", "/"))


_UNPARSEABLE: Any = object()


def _split_key(key: str | bytes | None) -> tuple[str | None, str | None]:
    if isinstance(key, bytes):
        key = key.decode("utf-8", errors="replace")
    if not key:
        return None, None
    kind, separator, path = key.partition(":")
    if not separator or kind not in ("state", "config") or not path.startswith("/"):
        return None, None
    return kind, path


def _parse_value(value: str | bytes | Mapping[str, Any] | None) -> Any:
    """The value as a mapping, ``None`` for a tombstone, ``_UNPARSEABLE`` otherwise."""
    if isinstance(value, Mapping):
        return value
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (ValueError, UnicodeDecodeError):
        return _UNPARSEABLE
    return parsed if isinstance(parsed, dict) else _UNPARSEABLE


def _rank(severity: Any) -> int:
    """The severity's rank; an unknown one ranks highest, so it never reads as recovered."""
    return _SEVERITY_RANK.get(severity, len(_SEVERITY_RANK))


def _recorded_time(recorded: Any) -> str | None:
    """The alarm server's ``{seconds, nano}`` as ISO-8601; a string stays as recorded."""
    if isinstance(recorded, str):
        return recorded
    if not isinstance(recorded, Mapping):
        return None
    try:
        moment = datetime.fromtimestamp(recorded["seconds"] + recorded.get("nano", 0) / 1e9, UTC)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")
