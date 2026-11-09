"""What the PV page and the IOC page show, from the directory service alone.

:class:`Facts` answers two questions over a :class:`DirectoryService`:

* :meth:`Facts.pv_facts` gives one section per configured service: the
  directory record, the alarm state, the alarm configuration, the alarm history
  and the put history. Each runs under its own deadline, and a section that has
  no answer says why (``unavailable`` with its reason); a section the
  deployment did not configure is absent, never empty. :meth:`Facts.alarm_state`,
  :meth:`Facts.alarm_config` and :meth:`Facts.history` answer one of those
  sections alone, as the PV page shows it.
* :meth:`Facts.ioc_facts` gives one IOC's host, ports, PV counts, engineer and
  the names of its IOC health PVs.

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
    NO_SOURCE_REASON,
    Unavailable,
    as_payload,
    unavailable_payload,
)
from osprey.services.pv_directory.resolve import SectionError, predicates, resolve_section
from osprey.services.pv_directory.service import DirectoryService

__all__ = ["Facts", "panel_link"]

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 6.0
#: How many logged rows a history section shows unless the caller asks for another number.
_HISTORY_LIMIT = 20
#: The history kinds :meth:`Facts.history` reads.
_HISTORY_KINDS = ("alarm_log", "put_log")
_DEFAULT_ENGINEER_PROPERTY = "Engineer"
_DEFAULT_HEALTH_SUFFIXES = ("HEARTBEAT", "UPTIME")
#: The signals the simulator serves for each of its IOCs, ``<code>:SIM:<model>:<signal>``.
_SIMULATOR_HEALTH_SIGNALS = ("HEARTBEAT", "UPTIME", "ENGINEER")
_SIMULATOR_ENGINEER_SIGNAL = "ENGINEER"
#: The measured RecSync properties holding an IOC's Channel Access and PV Access ports.
_PORT_PROPERTIES = ("caPort", "pvaPort")
_HEALTH_PAGE_SIZE = 50
#: How long past its deadline a backend call may run before the section gives up on it.
_GRACE_S = 0.5
_NO_HEALTH_PVS = "no IOC health PVs listed for this IOC"
_FEED_LOADING = "alarm feed still loading"
_LIVE_CONFIGURATION = "live configuration"
_LOGGED_CONFIGURATION = "last logged configuration"
#: The panel that shows directory results; its deep-link routes are ``#/pv/`` and ``#/ioc/``.
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
    return {"panel": f"{_PANEL}#/{kind}/{quote(name, safe='')}"}


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
        """Every configured section about the PV called *name*: ``{name, sections}``."""
        configured = self._configured()
        if configured is None:
            return {"name": name, "sections": []}
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
        return {"name": name, "sections": sections}

    async def alarm_state(self, name: str) -> dict[str, Any]:
        """The alarm state section of the PV called *name*: ``{outcome, source, record, reason?}``."""
        return await self._finished(await _guarded(self._alarm_state(name, self._timeout())))

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
        record = described.get("record")
        return {**described, "detail_properties": self._detail_properties(record)}

    async def _alarm_state(self, name: str, deadline: float) -> Any:
        backend, source, reason = await self._service.backend("alarm_state")
        if backend is None:
            return _NoSource(reason or NO_SOURCE_REASON, record=None)
        snapshot = await self._service.snapshot()
        if snapshot is None:
            return _section(source, "unavailable", record=None, reason="the target moved")
        answer = await _ask(snapshot.state_for(name, deadline), deadline)
        if answer is None:
            return _section(source, "not_listed", record=None)
        if _is_unavailable(answer):
            return _section(source, "unavailable", record=None, reason=answer.reason)
        return _section(source, "found", record=as_payload(answer))

    async def _alarm_config(self, name: str, deadline: float) -> Any:
        """The live configuration from the feed, else the logger's (D4); one source per page."""
        backend, source, reason = await self._service.backend("alarm_state")
        if backend is not None:
            snapshot = await self._service.snapshot()
            answer = (
                await _ask(snapshot.config_for(name, deadline), deadline)
                if snapshot is not None
                else _Reason("the target moved")
            )
            if not _is_unavailable(answer):
                return _config_section(source, answer)
            if answer.reason == _FEED_LOADING:
                return _failed_config(source, _LIVE_CONFIGURATION, answer.reason)
            feed = (source, answer.reason)
        else:
            feed = None
        log, log_source, log_reason = await self._service.backend("alarm_log")
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
            return _failed_config(log_source, _LOGGED_CONFIGURATION, answer.reason)
        return _config_section(log_source, answer)

    async def _history(
        self,
        kind: str,
        name: str,
        deadline: float,
        *,
        since: datetime | None = None,
        limit: int = _HISTORY_LIMIT,
    ) -> Any:
        backend, source, reason = await self._service.backend(kind)
        if backend is None:
            return _NoSource(reason or NO_SOURCE_REASON, rows=[])
        answer = await _ask(backend.history(name, since, limit, deadline), deadline)
        if _is_unavailable(answer):
            return _section(source, "unavailable", rows=[], reason=answer.reason)
        page = as_payload(answer)
        return {key: value for key, value in page.items() if key != "reason" or value}

    async def _finished(
        self, answer: dict[str, Any] | _NoSource, target: str | None = None
    ) -> dict[str, Any]:
        """*answer* as a section payload: a missing source answers for *target*, else the current one."""
        if not isinstance(answer, _NoSource):
            return answer
        payload = unavailable_payload(answer.reason, target=target or await self._target())
        return {**payload, **answer.fields}

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
            return {**empty, **total}
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
        inactive, curated, sample, health = await asyncio.gather(
            self._service.count(ioc=[ioc], status="Inactive"),
            self._curated_count(ioc),
            self._service.search(ioc=[ioc], page_size=1),
            self._health_pvs(ioc, on_va),
        )
        row = (sample.get("rows") or [None])[0]
        properties: Mapping[str, Any] = {}
        if row is not None:
            described = await self._service.describe(row["name"])
            properties = (described.get("record") or {}).get("properties") or {}
        health_pvs, health_reason = health

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
        }
        if on_va:
            engineer = [name for name in health_pvs if _signal(name) == _SIMULATOR_ENGINEER_SIGNAL]
            if engineer:
                facts.update(engineer=engineer[0], engineer_source="ENGINEER PV")
        else:
            prop = self._display.get("engineer_property") or _DEFAULT_ENGINEER_PROPERTY
            if properties.get(prop):
                facts.update(engineer=properties[prop], engineer_source="directory property")
        all_inactive = facts["inactive_count"] == total["total"]
        facts["outcome"] = "inactive" if all_inactive else "found"
        return facts

    async def _curated_count(self, ioc: str) -> dict[str, Any]:
        """How many roster addresses this IOC's records carry, or why it cannot be counted."""
        roster = registered_channels(dict(self._config))
        if roster.absence is not None:
            return {"curated_count": None, "curated_reason": roster.absence.message()}
        counted = await self._service.count(ioc=[ioc], origin=["curated"])
        if counted["outcome"] == "unavailable":
            return {"curated_count": None, "curated_reason": counted["reason"]}
        return {"curated_count": counted["total"]}

    async def _health_pvs(self, ioc: str, on_va: bool) -> tuple[list[str], str]:
        """The IOC's health PV names by the rule of its target, and the reason when none."""
        if on_va:
            suffixes = tuple(f":{signal}" for signal in _SIMULATOR_HEALTH_SIGNALS)
        else:
            suffixes = _suffixes(self._display.get("ioc_health_suffixes"))
        pages = await asyncio.gather(
            *(
                self._service.search(ioc=[ioc], name=f"*{suffix}", page_size=_HEALTH_PAGE_SIZE)
                for suffix in suffixes
            )
        )
        failed = [page["reason"] for page in pages if page["outcome"] == "unavailable"]
        names = sorted(
            {
                row["name"]
                for page in pages
                for row in page.get("rows") or ()
                if row["name"].endswith(suffixes)
            }
        )
        return names, failed[0] if failed and not names else _NO_HEALTH_PVS

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


def _section(source: Any, outcome: str, **fields: Any) -> dict[str, Any]:
    return {"outcome": outcome, "source": as_payload(source), **fields}


def _config_section(source: Any, record: Any) -> dict[str, Any]:
    payload = as_payload(record)
    return _section(source, "found", record=payload, label=payload.get("label"))


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


def _failed_config(source: Any, label: str, reason: str) -> dict[str, Any]:
    return _section(
        source, "unavailable", record=_unknown_config(label), label=label, reason=reason
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


def _suffixes(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        listed = tuple(item for item in value if isinstance(item, str) and item)
        if listed:
            return listed
    return _DEFAULT_HEALTH_SUFFIXES


def _signal(name: str) -> str:
    return name.rpartition(":")[2]
