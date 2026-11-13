"""Core PV directory health categories.

Two categories share this module, split by cost class:

* ``pv_directory`` (poll): one row per configured directory, alarm logger and
  put log, from the directory service's probes on the current target. The
  alarm feed is left unstarted.
* ``pv_directory_alarm_feed`` (on_demand): the alarm feed, started through the
  service's alarm snapshot and given up to its deadline to catch up; the row
  carries the backend's probe words.

A row is ``ok`` with the service's evidence as its value, ``warning`` with the
reason when the service cannot answer (a connectors release without the PV
directory modules included), and ``skip`` when the service has no source on the
current target. With no ``pv_directory`` section, or no config, neither
category has rows. Both build the service inside the call and close it before
returning, so no feed task outlives the run.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.health.models import CheckResult, Status

CATEGORY = "pv_directory"
CATEGORY_ALARM_FEED = "pv_directory_alarm_feed"

_SECTION_KEY = "pv_directory"

#: Kind -> (row-name suffix, label).
_ROWS = {
    "pv_directory": ("directory", "directory"),
    "alarm_log": ("alarm_log", "alarm logger"),
    "put_log": ("put_log", "put log"),
    "alarm_state": ("alarm_feed", "alarm feed"),
}


def pv_directory(
    config: Mapping[str, Any] | None = None,
    context: Any = None,  # noqa: ARG001 - health category factory signature
    *,
    cwd: Path | None = None,
) -> Any:
    """Build the poll-class ``pv_directory`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
        context: Unused; the services are probed by the directory service.
        cwd: The rendered config's directory, which a relative service path
            in the config is relative to.
    """
    return _category(config, cwd, CATEGORY, ("pv_directory", "alarm_log", "put_log"))


def pv_directory_alarm_feed(
    config: Mapping[str, Any] | None = None,
    context: Any = None,  # noqa: ARG001 - health category factory signature
    *,
    cwd: Path | None = None,
) -> Any:
    """Build the on_demand ``pv_directory_alarm_feed`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable).
        context: Unused; the feed is read by the directory service.
        cwd: The rendered config's directory, which ``alarm_state.replay_file``
            may be relative to.
    """
    return _category(config, cwd, CATEGORY_ALARM_FEED, ("alarm_state",))


def _category(
    config: Mapping[str, Any] | None, cwd: Path | None, category: str, kinds: tuple[str, ...]
) -> Any:
    async def _run() -> list[CheckResult]:
        if not isinstance(config, Mapping) or config.get(_SECTION_KEY) is None:
            return []
        from osprey.services.pv_directory._connectors import NO_SOURCE_REASON
        from osprey.services.pv_directory.service import DirectoryService

        service = DirectoryService(config, project_dir=cwd)
        try:
            probes = await service.probes(kinds=kinds)
        finally:
            await service.aclose()
        return [
            _row(category, kind, payload, NO_SOURCE_REASON)
            for kind, payload in probes.items()
            if kind in kinds
        ]

    return _run


def _row(category: str, kind: str, payload: Mapping[str, Any], no_source: str) -> CheckResult:
    suffix, label = _ROWS[kind]
    source = payload.get("source") or {}
    service = source.get("service")
    details = f"{service} on {source.get('target')}" if service and service != "none" else ""
    name = f"{CATEGORY}.{suffix}"
    if payload.get("outcome") == "found":
        evidence = str(payload.get("evidence", ""))
        return CheckResult(
            name,
            category,
            Status.OK,
            f"{label}: {evidence}",
            value=evidence,
            details=details,
            probed=True,
        )
    reason = str(payload.get("reason", ""))
    status = Status.SKIP if reason == no_source else Status.WARNING
    return CheckResult(
        name,
        category,
        status,
        f"{label}: {reason}",
        details=details,
        probed=status is Status.WARNING,
    )
