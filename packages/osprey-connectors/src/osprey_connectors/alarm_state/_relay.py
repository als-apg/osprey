"""Alarm state read over HTTP from an OSPREY alarm relay.

A broker's own protocol reaches only a host inside the broker's network. Where a
facility lets HTTP through a front end and nothing else, the relay
(:mod:`osprey_connectors.alarm_state.relay`) runs beside the broker, follows the
topic there, and serves what it folded. This backend reads that, so the answers
are the Kafka backend's own, one hop later: the relay sets ``stale`` and
``last_contact`` from its contact with the broker, and a relay that cannot be
reached is ``Unavailable``, never a clear state.

The wire shapes are written and read here, so the two ends cannot drift apart.
Every route answers JSON with status 200; a feed with no answer says
``{"unavailable": "<reason>"}``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from typing_extensions import override

from osprey_connectors._service_http import ReadOnlyTransport, Transport
from osprey_connectors.alarm_state._contract import (
    AlarmConfigRecord,
    AlarmState,
    AlarmStateRecord,
    AlarmStateRow,
    AlarmStates,
)
from osprey_connectors.service_answers import BackendSpec, Unavailable, _flag, _text

ROOT_PATH = "/"
STATES_PATH = "/states"
STATE_PATH = "/state"
CONFIG_PATH = "/config"
#: The query key the one-channel routes read.
CHANNEL_PARAM = "channel"
#: What the root names itself, so a reader can tell a relay from another service.
SERVICE_NAME = "osprey-alarm-relay"

_MALFORMED = Unavailable("malformed response")
_NOT_A_RELAY = Unavailable(f"the url does not answer as an {SERVICE_NAME}")
_IN_TREE = frozenset({"yes", "no", "unknown"})


def unavailable_payload(answer: Unavailable) -> dict[str, object]:
    """What a route answers when the feed has no answer."""
    return {"unavailable": answer.reason}


def states_payload(answer: AlarmStates | Unavailable) -> dict[str, object]:
    """Every channel's state as the relay serves it; ``stale`` is said once, for all rows."""
    if isinstance(answer, Unavailable):
        return unavailable_payload(answer)
    return {
        "stale": answer.stale,
        "last_contact": answer.last_contact,
        "counts": dict(answer.counts),
        "rows": [
            {"channel": row.channel, "path": row.path, "state": _state_body(row.state)}
            for row in answer.rows
        ],
    }


def state_payload(answer: AlarmStateRecord | Unavailable | None) -> dict[str, object]:
    """One channel's state; ``"state": null`` is a caught-up feed that holds none."""
    if isinstance(answer, Unavailable):
        return unavailable_payload(answer)
    if answer is None:
        return {"state": None}
    return {"state": _state_body(answer), "stale": answer.stale}


def config_payload(answer: AlarmConfigRecord | Unavailable) -> dict[str, object]:
    """One channel's place in the alarm tree."""
    if isinstance(answer, Unavailable):
        return unavailable_payload(answer)
    return {"config": asdict(answer)}


def _state_body(state: AlarmStateRecord) -> dict[str, object]:
    body = asdict(state)
    del body["stale"]
    return body


class RelayAlarmState(AlarmState):
    """Answer from a relay's folded snapshot; the relay follows the feed, not this.

    ``settings.url`` is the relay's address as the reader reaches it, a front
    end's path prefix included. :meth:`start` and :meth:`aclose` hold nothing:
    each read is one request, so a relay that comes back answers again with no
    reconnect here.
    """

    ALLOWED_REQUESTS = (
        ("GET", ROOT_PATH),
        ("GET", STATES_PATH),
        ("GET", STATE_PATH),
        ("GET", CONFIG_PATH),
    )

    def __init__(
        self,
        spec: BackendSpec,
        *,
        project_dir: Path | None = None,
        transport: Transport | None = None,
        **seams: object,
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        if transport is None:
            transport = ReadOnlyTransport(
                spec.settings, self.ALLOWED_REQUESTS, where=spec.settings_key
            )
        self._transport = transport

    @override
    async def start(self) -> None:
        return None

    @override
    async def aclose(self) -> None:
        return None

    @override
    async def state_for(
        self, channel: str, deadline: float
    ) -> AlarmStateRecord | Unavailable | None:
        answer = await self._read(STATE_PATH, deadline, channel=channel)
        if isinstance(answer, Unavailable):
            return answer
        if "state" not in answer:
            return _MALFORMED
        if answer["state"] is None:
            return None
        stale = answer.get("stale")
        if not isinstance(stale, bool):
            return _MALFORMED
        return _state_record(answer["state"], stale=stale) or _MALFORMED

    @override
    async def states(self, deadline: float) -> AlarmStates | Unavailable:
        answer = await self._read(STATES_PATH, deadline)
        if isinstance(answer, Unavailable):
            return answer
        stale, last_contact, served = (
            answer.get("stale"),
            answer.get("last_contact"),
            answer.get("rows"),
        )
        if (
            not isinstance(stale, bool)
            or not isinstance(last_contact, str | None)
            or not isinstance(served, list)
        ):
            return _MALFORMED
        rows = []
        for item in served:
            row = _state_row(item, stale=stale)
            if row is None:
                return _MALFORMED
            rows.append(row)
        return AlarmStates(rows=tuple(rows), stale=stale, last_contact=last_contact)

    @override
    async def config_for(self, channel: str, deadline: float) -> AlarmConfigRecord | Unavailable:
        answer = await self._read(CONFIG_PATH, deadline, channel=channel)
        if isinstance(answer, Unavailable):
            return answer
        body = answer.get("config")
        if not isinstance(body, Mapping) or body.get("in_tree") not in _IN_TREE:
            return _MALFORMED
        path = body.get("path")
        if not isinstance(path, str | None):
            return _MALFORMED
        return AlarmConfigRecord(
            in_tree=body["in_tree"],
            path=path,
            guidance=body.get("guidance"),
            displays=body.get("displays"),
            enabled=_flag(body.get("enabled")),
            latching=_flag(body.get("latching")),
        )

    @override
    async def probe(self, deadline: float) -> str | Unavailable:
        answer = await self._read(ROOT_PATH, deadline, relay_only=True)
        if isinstance(answer, Unavailable):
            return answer
        evidence = answer.get("evidence")
        if not isinstance(evidence, str):
            return _MALFORMED
        return f"relay answered: {evidence}"

    async def _read(
        self, path: str, deadline: float, *, relay_only: bool = False, channel: str | None = None
    ) -> Mapping[str, object] | Unavailable:
        """One route's answer as a mapping, or why there is none.

        With *relay_only*, an answer that does not name itself a relay is refused,
        so the probe of a wrong url does not read as a working feed.
        """
        params = None if channel is None else {CHANNEL_PARAM: channel}
        answer = await self._transport.request("GET", path, params=params, deadline=deadline)
        if isinstance(answer, Unavailable):
            return answer
        if not isinstance(answer, Mapping):
            return _MALFORMED
        if relay_only and answer.get("service") != SERVICE_NAME:
            return _NOT_A_RELAY
        reason = answer.get("unavailable")
        if reason is not None:
            return Unavailable(reason) if isinstance(reason, str) and reason else _MALFORMED
        return answer


def _state_record(body: object, *, stale: bool) -> AlarmStateRecord | None:
    """The served state as a record; ``None`` when a field is not what the relay writes."""
    if not isinstance(body, Mapping):
        return None
    latched, acknowledged = body.get("latched"), body.get("acknowledged")
    if not isinstance(latched, bool) or not isinstance(acknowledged, bool):
        return None
    for name in ("severity", "status", "message", "time"):
        if not isinstance(body.get(name), str | None):
            return None
    return AlarmStateRecord(
        severity=_text(body.get("severity")),
        status=_text(body.get("status")),
        latched=latched,
        acknowledged=acknowledged,
        message=_text(body.get("message")),
        time=_text(body.get("time")),
        stale=stale,
    )


def _state_row(item: object, *, stale: bool) -> AlarmStateRow | None:
    if not isinstance(item, Mapping):
        return None
    channel, path = item.get("channel"), item.get("path")
    if not isinstance(channel, str) or not channel or not isinstance(path, str | None):
        return None
    state = _state_record(item.get("state"), stale=stale)
    return None if state is None else AlarmStateRow(channel=channel, path=path, state=state)
