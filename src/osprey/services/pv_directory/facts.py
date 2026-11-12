"""What the PV page and the IOC page show, from the directory service alone.

:class:`Facts` answers two questions over a :class:`DirectoryService`:

* :meth:`Facts.pv_facts` gives one section per configured service: the
  directory record, the alarm state, the alarm configuration, the alarm history
  and the put history. Each runs under its own deadline, and a section that has
  no answer says why (``unavailable`` with its reason); a section the
  deployment did not configure is absent, never empty. Every section carries
  the ``caveats`` the deployment wrote on its configured source. The answer's
  own ``outcome`` says once whether the PV is known at all, for every reader. :meth:`Facts.alarm_state`,
  :meth:`Facts.alarm_config` and :meth:`Facts.history` answer one of those
  sections alone, as the PV page shows it, and :meth:`Facts.active_alarms`
  every channel the alarm feed holds in alarm now.
* :meth:`Facts.ioc_facts` gives one IOC's host, ports, PV counts, engineer, its
  IOC health PVs (each a name and its role: heartbeat, uptime or engineer) and,
  with an alarm feed configured, its PVs in alarm now.

Both return plain dicts on every connectors release. Names are carried, never
values: the IOC page reads the health PVs (and, on the virtual accelerator, the
ENGINEER PV) through the value path. People's names appear as the services
record them.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import Any
from urllib.parse import quote

from osprey.channel_roster import registered_channels
from osprey.services.pv_directory._connectors import (
    ENGINEER,
    IOC_HEALTH_SIGNALS,
    NO_SOURCE_REASON,
    SERVED_SIGNAL_PROPERTY,
    Unavailable,
    as_payload,
    severity_rank,
    unavailable_payload,
)
from osprey.services.pv_directory.resolve import SectionError, predicates, resolve_section
from osprey.services.pv_directory.service import DirectoryService
from osprey.services.pv_directory.vocabulary import CONFIG_LABELS

__all__ = ["ACTIVE_ALARMS_MAX_ROWS", "Facts", "panel_link"]

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 6.0
#: How many logged rows a history section shows unless the caller asks for another number.
_HISTORY_LIMIT = 20
#: The history kinds :meth:`Facts.history` reads.
_HISTORY_KINDS = ("alarm_log", "put_log")
#: The most rows one active-alarms answer serves, to the panel route and the agent tool alike.
ACTIVE_ALARMS_MAX_ROWS = 1000
_DEFAULT_ENGINEER_PROPERTY = "Engineer"
#: The measured RecSync properties holding an IOC's Channel Access and PV Access ports.
_PORT_PROPERTIES = ("caPort", "pvaPort")
_HEALTH_PAGE_SIZE = 50
#: The severities an IOC's alarms list: every one but OK and no state.
_ACTIVE_SEVERITIES = ("MAJOR", "INVALID", "UNDEFINED", "MINOR")
#: The most PVs in alarm one IOC answer lists; ``total`` counts them all.
_IOC_ALARMS_PAGE_SIZE = 50
#: The alarm fields an IOC's alarm row carries beside its PV name.
_IOC_ALARM_FIELDS = ("severity", "status", "latched", "acknowledged", "time")
#: How long past its deadline a backend call may run before the section gives up on it.
_GRACE_S = 0.5
_NO_HEALTH_PVS = "no IOC health PVs listed for this IOC"
_FEED_LOADING = "alarm feed still loading"
_LIVE_CONFIGURATION = CONFIG_LABELS["alarm_state"]
_LOGGED_CONFIGURATION = CONFIG_LABELS["alarm_log"]
#: The panel that shows directory results; its deep-link routes are ``#/pv/`` and ``#/ioc/``,
#: the name one segment encoded as ``encodeURIComponent`` encodes it.
_PANEL = "/panel/channel-finder/"

#: Each section after the directory, and the context key that configures it.
_SECTIONS = (
    ("alarm_state", "pv_directory_alarm_state"),
    ("alarm_config", "pv_directory_alarm_config"),
    ("alarm_history", "pv_directory_alarm_log"),
    ("put_history", "pv_directory_put_log"),
)


def panel_link(kind: str, name: str) -> dict[str, str]:
    """``{"panel": url}`` opening *name* on the panel's ``pv`` or ``ioc`` page."""
    segment = quote(name, safe="!*'()")
    return {"panel": f"{_PANEL}#/{kind}/{segment}"}


class _NoSource:
    """A section the current target has no backend for, and why."""

    def __init__(self, reason: str, **fields: Any) -> None:
        self.reason = reason
        self.fields = fields


class Facts:
    """The PV and IOC facts of one process, over its directory service.

    Args:
        service: The process's directory service; the only way to a backend.
        config: The full config mapping the service was built from.
    """

    def __init__(self, service: DirectoryService, config: Mapping[str, Any]) -> None:
        self._service = service
        self._config = config
        section = config.get("pv_directory")
        self._section: Mapping[str, Any] = section if isinstance(section, Mapping) else {}
        display = self._section.get("display")
        self._display: Mapping[str, Any] = display if isinstance(display, Mapping) else {}

    # -- the PV page -----------------------------------------------------

    async def pv_facts(self, name: str) -> dict[str, Any]:
        """Every configured section about the PV called *name*: ``{name, outcome, sections}``.

        ``outcome`` is ``found`` when the directory lists the PV, ``not_listed``
        when it does not but another section found a record or rows,
        ``unknown`` when nothing found it, and ``unavailable`` with the
        directory's ``reason`` when the directory gave no answer.
        """
        configured = self._configured()
        if configured is None:
            return {"name": name, **_unavailable(NO_SOURCE_REASON), "sections": []}
        deadline = self._timeout()
        readers: dict[str, Callable[[str, float], Awaitable[Any]]] = {
            "directory": self._directory,
            "alarm_state": self._alarm_state,
            "alarm_config": self._alarm_config,
            "alarm_history": lambda pv, within: self._history("alarm_log", pv, within),
            "put_history": lambda pv, within: self._history("put_log", pv, within),
        }
        keys = ("directory", *configured)
        answers = await asyncio.gather(*(_guarded(readers[key](name, deadline)) for key in keys))
        target = _target_of(answers[0]) or await self._target()
        sections = [
            {"section": key, **await self._finished(answer, target)}
            for key, answer in zip(keys, answers, strict=True)
        ]
        return {"name": name, **_page_outcome(sections), "sections": sections}

    async def alarm_state(self, name: str) -> dict[str, Any]:
        """The alarm state section of the PV called *name*: ``{outcome, source, record, reason?}``."""
        return await self._finished(await _guarded(self._alarm_state(name, self._timeout())))

    async def active_alarms(
        self, *, severities: Sequence[str] = (), limit: int | None = None
    ) -> dict[str, Any]:
        """Every channel the alarm feed holds in alarm now, with the counts per severity.

        ``counts`` and ``total`` cover every channel the feed holds a state for;
        ``rows`` are only those whose severity is not ``OK``, most severe first,
        then newest first.

        Args:
            severities: Only rows of these severities; empty serves every active row.
            limit: The most rows served; a cut sets ``truncated`` and never changes
                the counts. ``None`` serves every row.
        """
        read = self._active_alarms(self._timeout(), tuple(severities), limit)
        return await self._finished(await _guarded(read))

    async def alarm_config(self, name: str) -> dict[str, Any]:
        """The alarm configuration section of the PV called *name*, with its ``label``."""
        return await self._finished(await _guarded(self._alarm_config(name, self._timeout())))

    async def history(
        self,
        kind: str,
        name: str,
        *,
        since: datetime | None = None,
        limit: int = _HISTORY_LIMIT,
    ) -> dict[str, Any]:
        """The ``alarm_log`` or ``put_log`` history section of the PV called *name*.

        Args:
            kind: ``"alarm_log"`` or ``"put_log"``.
            name: The PV name.
            since: Only rows logged at or after this moment; ``None`` reads the newest.
            limit: The most rows the page holds.

        Raises:
            ValueError: *kind* is not a history kind.
        """
        if kind not in _HISTORY_KINDS:
            raise ValueError(f"{kind!r} is not one of {_HISTORY_KINDS}")
        read = self._history(kind, name, self._timeout(), since=since, limit=limit)
        return await self._finished(await _guarded(read))

    async def _directory(self, name: str, deadline: float) -> dict[str, Any]:  # noqa: ARG002
        """The directory record; ``describe`` runs under the service's own deadline."""
        described = await self._service.describe(name)
        backend, _source, _reason = await self._service.backend("pv_directory")
        record = described.get("record")
        return {
            **described,
            "caveats": _caveats(backend),
            "detail_properties": self._detail_properties(record),
        }

    async def _alarm_state(self, name: str, deadline: float) -> Any:
        backend, _source, reason = await self._service.backend("alarm_state")
        if backend is None:
            return _NoSource(reason or NO_SOURCE_REASON, record=None)
        snapshot = await self._service.snapshot()
        if snapshot is None:
            return _section(backend, "unavailable", record=None, reason="the target moved")
        answer = await _ask(snapshot.state_for(name, deadline), deadline)
        if answer is None:
            return _section(backend, "not_listed", record=None)
        if _is_unavailable(answer):
            return _section(backend, "unavailable", record=None, reason=answer.reason)
        return _section(backend, "found", record=as_payload(answer))

    async def _active_alarms(
        self, deadline: float, severities: tuple[str, ...], limit: int | None
    ) -> Any:
        backend, _source, reason = await self._service.backend("alarm_state")
        if backend is None:
            return _NoSource(reason or NO_SOURCE_REASON)
        snapshot = await self._service.snapshot()
        if snapshot is None:
            return _section(backend, "unavailable", reason="the target moved")
        answer = await _ask(snapshot.states(deadline), deadline)
        if _is_unavailable(answer):
            return _section(backend, "unavailable", reason=answer.reason)
        active = sorted(
            (row for row in answer.rows if row.state.severity not in ("OK", None)),
            key=lambda row: (severity_rank(row.state.severity), row.state.time or ""),
            reverse=True,
        )
        if severities:
            active = [row for row in active if row.state.severity in severities]
        served = active if limit is None else active[: max(limit, 0)]
        return _section(
            backend,
            "found",
            counts=dict(answer.counts),
            total=len(answer.rows),
            rows=[_alarm_row(row) for row in served],
            truncated=len(served) < len(active),
            stale=answer.stale,
            last_contact=answer.last_contact,
        )

    async def _alarm_config(self, name: str, deadline: float) -> Any:
        """The live configuration from the feed, else the logger's (D4); one source per page."""
        backend, _source, reason = await self._service.backend("alarm_state")
        if backend is not None:
            snapshot = await self._service.snapshot()
            answer = (
                await _ask(snapshot.config_for(name, deadline), deadline)
                if snapshot is not None
                else _Reason("the target moved")
            )
            if not _is_unavailable(answer):
                return _config_section(backend, answer)
            if answer.reason == _FEED_LOADING:
                return _failed_config(backend, _LIVE_CONFIGURATION, answer.reason)
            feed = (backend, answer.reason)
        else:
            feed = None
        log, _log_source, log_reason = await self._service.backend("alarm_log")
        if log is None:
            if feed is not None:
                return _failed_config(feed[0], _LIVE_CONFIGURATION, feed[1])
            return _NoSource(
                log_reason or reason or NO_SOURCE_REASON,
                record=_unknown_config(_LOGGED_CONFIGURATION),
                label=_LOGGED_CONFIGURATION,
            )
        answer = await _ask(log.config_for(name, deadline), deadline)
        if _is_unavailable(answer):
            return _failed_config(log, _LOGGED_CONFIGURATION, answer.reason)
        return _config_section(log, answer)

    async def _history(
        self,
        kind: str,
        name: str,
        deadline: float,
        *,
        since: datetime | None = None,
        limit: int = _HISTORY_LIMIT,
    ) -> Any:
        backend, _source, reason = await self._service.backend(kind)
        if backend is None:
            return _NoSource(reason or NO_SOURCE_REASON, rows=[])
        answer = await _ask(backend.history(name, since, limit, deadline), deadline)
        if _is_unavailable(answer):
            return _section(backend, "unavailable", rows=[], reason=answer.reason)
        page = as_payload(answer)
        fields = {
            key: value
            for key, value in page.items()
            if key not in ("outcome", "source") and (key != "reason" or value)
        }
        return _section(backend, page["outcome"], **fields)

    async def _finished(
        self, answer: dict[str, Any] | _NoSource, target: str | None = None
    ) -> dict[str, Any]:
        """*answer* as a section payload: a missing source answers for *target*, else the current one."""
        if not isinstance(answer, _NoSource):
            return answer
        payload = unavailable_payload(answer.reason, target=target or await self._target())
        return {**payload, "caveats": [], **answer.fields}

    def _detail_properties(self, record: Mapping[str, Any] | None) -> list[str]:
        """The properties the PV page shows first; every one when none are configured."""
        present = list((record or {}).get("properties") or {})
        names = self._display.get("detail_properties")
        if not isinstance(names, list):
            return sorted(present)
        wanted = [name for name in names if isinstance(name, str)]
        return [name for name in wanted if name in present] if record else wanted

    # -- the IOC page ----------------------------------------------------

    async def ioc_facts(self, ioc: str) -> dict[str, Any]:
        """The IOC called *ioc*, as the ``IocFacts`` payload."""
        links = panel_link("ioc", ioc)
        empty: dict[str, Any] = {
            "ioc": ioc,
            "host": None,
            "ports": None,
            "pv_count": None,
            "inactive_count": None,
            "curated_count": None,
            "engineer": None,
            "engineer_source": None,
            "health_pvs": [],
            "links": links,
        }
        total = await self._service.count(ioc=[ioc])
        if total["outcome"] == "unavailable":
            return {**empty, **{key: total[key] for key in ("outcome", "source", "reason")}}
        source = total["source"]
        if not total["total"]:
            return {
                **empty,
                "outcome": "not_listed",
                "source": source,
                "pv_count": 0,
                "inactive_count": 0,
                "curated_count": 0,
                "health_pvs_reason": _NO_HEALTH_PVS,
            }

        on_va = source.get("target") == "va"
        with_alarms = "alarm_state" in (self._configured() or [])
        inactive, curated, sample, health, alarms = await asyncio.gather(
            self._service.count(ioc=[ioc], status="Inactive"),
            self._curated_count(ioc),
            self._service.search(ioc=[ioc], page_size=1),
            self._health_pvs(ioc, on_va),
            self._ioc_alarms(ioc) if with_alarms else _absent(),
        )
        row = (sample.get("rows") or [None])[0]
        properties: Mapping[str, Any] = {}
        if row is not None:
            described = await self._service.describe(row["name"])
            properties = (described.get("record") or {}).get("properties") or {}
        health_pvs, health_reason, engineer = health

        facts = {
            **empty,
            "source": source,
            "host": row.get("host") if row else None,
            "ports": None if on_va else _ports(properties),
            "pv_count": total["total"],
            "inactive_count": inactive.get("total"),
            **curated,
            "health_pvs": health_pvs,
            **({"health_pvs_reason": health_reason} if not health_pvs else {}),
            **({"alarms": alarms} if alarms is not None else {}),
        }
        if on_va:
            if engineer is not None:
                facts.update(engineer=engineer, engineer_source="ENGINEER PV")
        else:
            prop = self._display.get("engineer_property") or _DEFAULT_ENGINEER_PROPERTY
            if properties.get(prop):
                facts.update(engineer=properties[prop], engineer_source="directory property")
        all_inactive = facts["inactive_count"] == total["total"]
        facts["outcome"] = "inactive" if all_inactive else "found"
        return facts

    async def _ioc_alarms(self, ioc: str) -> dict[str, Any]:
        """The IOC's PVs in alarm now: ``{outcome, total, rows, counts?}``, or ``unavailable``.

        One alarm-filtered search, so the panel and the agent read the answer the
        All PVs alarm filter gives; ``total`` counts the whole filtered set and
        ``counts`` each severity over the whole IOC, from the search's alarm
        facet (absent when the search answered no facets).
        """
        page = await self._service.search(
            ioc=[ioc], alarm=_ACTIVE_SEVERITIES, page_size=_IOC_ALARMS_PAGE_SIZE
        )
        if page["outcome"] == "unavailable":
            return {"outcome": "unavailable", "reason": page.get("reason")}
        feed = next(
            (entry for entry in page.get("partial") or () if entry["part"] == "alarm"), None
        )
        if feed is not None:
            return {"outcome": "unavailable", "reason": feed["reason"]}
        rows = [
            {
                "pv": row["name"],
                **{key: (row.get("alarm") or {}).get(key) for key in _IOC_ALARM_FIELDS},
            }
            for row in page.get("rows") or ()
        ]
        answer = {"outcome": page["outcome"], "total": page.get("total"), "rows": rows}
        facet = (page.get("facets") or {}).get("alarm")
        if facet is not None:
            answer["counts"] = {
                entry["value"]: entry["count"] for entry in facet if entry["value"] != "none"
            }
        return answer

    async def _curated_count(self, ioc: str) -> dict[str, Any]:
        """How many roster addresses this IOC's records carry, or why it cannot be counted."""
        roster = registered_channels(dict(self._config))
        if roster.absence is not None:
            return {"curated_count": None, "curated_reason": roster.absence.message()}
        counted = await self._service.count(ioc=[ioc], origin=["curated"])
        if counted["outcome"] == "unavailable":
            return {"curated_count": None, "curated_reason": counted["reason"]}
        return {"curated_count": counted["total"]}

    async def _health_pvs(
        self, ioc: str, on_va: bool
    ) -> tuple[list[dict[str, str]], str, str | None]:
        """The IOC's health PVs (name and role), the reason when none, its engineer.

        On ``va`` a health PV's role is the IOC health signal the simulator
        marks its row with, and the engineer is the row in the ``engineer``
        role; on any other target the role is the one whose configured suffix
        ends the name.
        """
        if on_va:
            asked = {
                _role(signal): {SERVED_SIGNAL_PROPERTY: signal} for signal in IOC_HEALTH_SIGNALS
            }
            pages = await asyncio.gather(
                *(
                    self._service.search(ioc=[ioc], properties=match, page_size=_HEALTH_PAGE_SIZE)
                    for match in asked.values()
                )
            )
            found = {
                (row["name"], role)
                for role, page in zip(asked, pages, strict=True)
                for row in page.get("rows") or ()
            }
        else:
            suffixes = self._health_suffixes()
            pages = await asyncio.gather(
                *(
                    self._service.search(ioc=[ioc], name=f"*{suffix}", page_size=_HEALTH_PAGE_SIZE)
                    for suffix in suffixes.values()
                )
            )
            found = {
                (row["name"], role)
                for (role, suffix), page in zip(suffixes.items(), pages, strict=True)
                for row in page.get("rows") or ()
                if row["name"].endswith(suffix)
            }
        health = [{"name": name, "role": role} for name, role in sorted(found)]
        engineer = (
            next((pv["name"] for pv in health if pv["role"] == _role(ENGINEER)), None)
            if on_va
            else None
        )
        failed = [page["reason"] for page in pages if page["outcome"] == "unavailable"]
        return health, failed[0] if failed and not health else _NO_HEALTH_PVS, engineer

    def _health_suffixes(self) -> dict[str, str]:
        """The record-name suffix of each IOC health role: ``display.ioc_health_suffixes``.

        A mapping of role to suffix; anything else, or a mapping naming no
        known role, answers the default: each role but the engineer's, ended
        by its signal's own word.
        """
        value = self._display.get("ioc_health_suffixes")
        signals = {_role(signal): signal for signal in IOC_HEALTH_SIGNALS}
        if isinstance(value, Mapping):
            listed = {
                role: suffix
                for role, suffix in value.items()
                if role in signals and isinstance(suffix, str) and suffix
            }
            if listed:
                return listed
        return {role: signal for role, signal in signals.items() if signal != ENGINEER}

    # -- small helpers ---------------------------------------------------

    def _configured(self) -> list[str] | None:
        """The configured sections after the directory, or ``None`` with no section."""
        try:
            resolved = resolve_section(self._config)
        except SectionError:
            resolved = None
        if resolved is None:
            return None
        keys = predicates(resolved)
        return [section for section, key in _SECTIONS if keys[key]]

    async def _target(self) -> str:
        """The current target, as the service's count names it in its source."""
        answer = await self._service.count()
        return str(answer["source"]["target"])

    def _timeout(self) -> float:
        settings = self._section.get("settings")
        value = settings.get("timeout_s") if isinstance(settings, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            return _DEFAULT_TIMEOUT_S
        return float(value)


class _Reason:
    """A reason with the shape of the connectors' ``Unavailable``."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


async def _ask(call: Awaitable[Any], deadline: float) -> Any:
    """*call*'s answer, or the reason it has none when it misses *deadline* or raises.

    The call was handed *deadline* and answers by it; the guard here, a grace
    later, only stops a backend that does not.
    """
    try:
        return await asyncio.wait_for(call, deadline + _GRACE_S)
    except TimeoutError:
        return _Reason(f"no answer within {deadline:g} s")
    except Exception as failure:
        logger.warning("PV facts backend failed", exc_info=True)
        return _Reason(f"the service failed: {failure}")


async def _absent() -> None:
    """The answer of a read the deployment did not configure."""


async def _guarded(call: Awaitable[Any]) -> Any:
    """*call*'s section; a section that raised says so instead of failing the page."""
    try:
        return await call
    except Exception as failure:
        logger.warning("PV facts section failed", exc_info=True)
        return _NoSource(f"the service failed: {failure}")


def _is_unavailable(answer: Any) -> bool:
    """Whether *answer* is a reason, not a record: the connectors' ``Unavailable`` or ours."""
    return isinstance(answer, _Reason) or (
        Unavailable is not None and isinstance(answer, Unavailable)
    )


def _section(backend: Any, outcome: str, **fields: Any) -> dict[str, Any]:
    """A section answered by *backend*, with its source and the caveats configured on it."""
    return {
        "outcome": outcome,
        "source": as_payload(backend.source),
        "caveats": _caveats(backend),
        **fields,
    }


def _alarm_row(row: Any) -> dict[str, Any]:
    """One active alarm as served: ``pv`` and ``path``, then its state's fields."""
    served = as_payload(row)
    state = served.pop("state")
    return {**served, **state}


def _page_outcome(sections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether the PV is known at all, from the directory section and what the others found."""
    directory, *others = sections
    outcome = directory["outcome"]
    if outcome in ("found", "inactive"):
        return {"outcome": "found"}
    if outcome == "not_listed":
        found = any(
            section["outcome"] == "found" and (section.get("record") or section.get("rows"))
            for section in others
        )
        return {"outcome": "not_listed" if found else "unknown"}
    return _unavailable(directory.get("reason") or NO_SOURCE_REASON)


def _unavailable(reason: str) -> dict[str, Any]:
    return {"outcome": "unavailable", "reason": reason}


def _caveats(backend: Any) -> list[str]:
    """The deployment's notes on *backend*'s source; none without a backend."""
    if backend is None:
        return []
    return list(backend.spec.settings.get("caveats") or ())


def _config_section(backend: Any, record: Any) -> dict[str, Any]:
    """The configuration *backend* answered, labelled by the kind of service it is."""
    label = CONFIG_LABELS[backend.spec.kind]
    return _section(backend, "found", record={**as_payload(record), "label": label}, label=label)


def _unknown_config(label: str) -> dict[str, Any]:
    """The record of a configuration step that failed: in the tree or not, unknown."""
    return {
        "in_tree": "unknown",
        "path": None,
        "guidance": None,
        "displays": None,
        "enabled": None,
        "latching": None,
        "label": label,
    }


def _failed_config(backend: Any, label: str, reason: str) -> dict[str, Any]:
    return _section(
        backend, "unavailable", record=_unknown_config(label), label=label, reason=reason
    )


def _target_of(section: Any) -> str | None:
    if isinstance(section, Mapping):
        source = section.get("source")
        target = source.get("target") if isinstance(source, Mapping) else None
        if isinstance(target, str):
            return target
    return None


def _ports(properties: Mapping[str, Any]) -> list[int] | None:
    """The IOC's recorded ports, Channel Access first; ``None`` when none is recorded."""
    ports = []
    for name in _PORT_PROPERTIES:
        try:
            ports.append(int(str(properties[name]).strip()))
        except (KeyError, ValueError):
            continue
    return ports or None


def _role(signal: str) -> str:
    """The IOC health role a simulator health signal serves: its word, lower case."""
    return signal.lower()
