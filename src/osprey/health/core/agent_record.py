"""Core ``agent_record`` health category.

The record's own account of itself, read from its ``GET /health/sources``
report through :class:`~osprey.agent_record.client.RecordClient` under the login
the calling process carries (:data:`~osprey.agent_record.client.URL_ENV` and
the login pair): the host login a host ``osprey web`` publishes, or a terminal's
or agent session's own login. Any login may read the report.

Rows, when the record answers:

* ``agent_record_backend`` — the store the record writes: a backend that is not
  ok, or a spool pass the backend refused, is a ``warning`` naming why.
* ``agent_record_spool`` — only when the spool set aside hostile paths or met
  ones it could not understand.
* ``agent_record_source.<identity>`` — one per spool directory: ``warning``
  when the source is **stopped** (a live session or pending backlog with nothing
  reaching the record for :data:`~osprey.agent_record.contract.STOPPED_AFTER`),
  when its lines were **refused**, when its writers logged errors or abandoned
  segments, or when its **backlog grows** (uncommitted lines older than
  :data:`BACKLOG_BEHIND_S`); ``ok`` and ``idle`` when it has no live session
  and nothing pending.
* ``agent_record_transcripts.<identity>`` — one per transcript root, the same
  way: stopped, unreadable, idle or receiving.
* ``agent_record_export`` — ``warning`` when more than one closed day is not
  exported, when the last export failed, or when the archive cannot be read.
  One day behind is normal: yesterday is exported by the next daily run.

Every problem row is advisory (``warning``): the record degrades, the rows say
where. When no record answers — nothing configured in this process (plain
``osprey health`` on a host, hello-world with nothing running) or nothing at
the configured address — the category answers one ``skip`` row saying so and
pointing at ``osprey agent-record health``, which asks the record on the host.
That is information, not an error.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from osprey.agent_record.client import ClientAnswer, RecordClient
from osprey.agent_record.contract import STOPPED_AFTER
from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "agent_record"

#: :data:`~osprey.agent_record.contract.STOPPED_AFTER` as the rows word it.
_STOPPED_AFTER_TEXT = f"over {int(STOPPED_AFTER.total_seconds() // 60)} min"

#: Seconds the report may take to answer.
_TIMEOUT_S = 5.0
#: Uncommitted lines older than this many seconds are a backlog that grows.
BACKLOG_BEHIND_S = 120.0
#: Closed days not exported beyond this many are an export that lags.
EXPORT_LAG_ALLOWED_DAYS = 1

_HEALTH_COMMAND = "osprey agent-record health"
_EXPORT_COMMAND = "osprey agent-record export"


class _HealthReader(Protocol):
    def health_sources(self) -> ClientAnswer: ...


def agent_record(
    config: Mapping[str, Any] | None = None,  # noqa: ARG001 - health category factory signature; the login comes from the environment
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; the record is read over HTTP
    *,
    client: _HealthReader | None = None,
    environ: Mapping[str, str] | None = None,
) -> CategoryCallable:
    """Build the ``agent_record`` category callable.

    Args:
        config: Parsed config mapping. Unused — the record and the login are
            named by the calling process's environment.
        context: Health runtime. Unused.
        client: The record client to ask, for tests; ``None`` builds one from
            *environ* when the callable runs.
        environ: The environment naming the record and the login; defaults to
            :data:`os.environ`, read when the callable runs.

    Returns:
        A no-argument sync callable returning the category's check results.
    """

    def _run() -> list[CheckResult]:
        reader = client
        if reader is None:
            env = os.environ if environ is None else environ
            reader = RecordClient.from_env(env, timeout=_TIMEOUT_S)
        answer = reader.health_sources()
        if not answer.available or not isinstance(answer.data, Mapping):
            return [_no_record(answer.unavailable or "the record answered no report")]
        return report_rows(answer.data)

    return _run


def report_rows(report: Mapping[str, Any]) -> list[CheckResult]:
    """The category's rows for one ``/health/sources`` *report* body."""
    spool = _mapping(report.get("spool"))
    backend = _mapping(report.get("backend"))
    rows = [_backend_row(backend, spool)]
    spool_row = _spool_row(spool)
    if spool_row is not None:
        rows.append(spool_row)
    rows.extend(_source_rows(_mapping(spool.get("sources"))))
    rows.extend(_transcript_rows(_mapping(report.get("transcripts"))))
    rows.append(_export_row(_mapping(report.get("export")), backend))
    return rows


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _no_record(reason: str) -> CheckResult:
    return CheckResult(
        "agent_record_report",
        CATEGORY,
        Status.SKIP,
        f"No agent record answers here: {reason}",
        details=f"`{_HEALTH_COMMAND}` asks the record on the host.",
        probed=False,
    )


def _row(name: str, status: Status, message: str, details: str = "") -> CheckResult:
    return CheckResult(name, CATEGORY, status, message, details=details, probed=True)


def _backend_row(backend: Mapping[str, Any], spool: Mapping[str, Any]) -> CheckResult:
    kind = backend.get("backend") or "record"
    refused = spool.get("backend_error") or ""
    if refused:
        return _row(
            "agent_record_backend",
            Status.WARNING,
            f"The {kind} store refused the last spool pass: {refused}",
            f"Lines stay in the spool until it accepts them; `{_HEALTH_COMMAND}` repeats this.",
        )
    if backend.get("ok") is not True:
        detail = backend.get("detail") or "no detail"
        return _row(
            "agent_record_backend",
            Status.WARNING,
            f"The {kind} store is not ok: {detail}",
            f"`{_HEALTH_COMMAND}` repeats this.",
        )
    entries = backend.get("entries")
    held = f"holds {entries} entries" if entries is not None else "answers"
    return _row("agent_record_backend", Status.OK, f"The {kind} store {held}")


def _spool_row(spool: Mapping[str, Any]) -> CheckResult | None:
    quarantined = _list(spool.get("quarantined"))
    problems = _list(spool.get("problems"))
    if not quarantined and not problems:
        return None
    lines = [f"set aside {pair[0]} -> {pair[1]}" for pair in quarantined if len(pair) == 2]
    lines.extend(str(problem) for problem in problems)
    return _row(
        "agent_record_spool",
        Status.WARNING,
        f"The spool set aside {len(quarantined)} path(s) and could not read {len(problems)}",
        "\n".join(lines),
    )


def _source_rows(sources: Mapping[str, Any]) -> list[CheckResult]:
    rows = []
    for path, raw in sorted(sources.items()):
        source = _mapping(raw)
        identity = str(source.get("identity") or path)
        rows.append(_source_row(identity, path, source))
    return rows


def _source_row(identity: str, path: str, source: Mapping[str, Any]) -> CheckResult:
    name = f"agent_record_source.{identity}"
    backlog = _int(source.get("backlog_lines"))
    problems: list[str] = []
    details: list[str] = [f"spool {source.get('directory') or path}"]

    if source.get("stopped"):
        waiting = f"{backlog} line(s) waiting, " if backlog else "a live session, "
        problems.append(f"stopped: {waiting}nothing reached the record for {_STOPPED_AFTER_TEXT}")
    else:
        age = source.get("oldest_backlog_age_s")
        if backlog and isinstance(age, int | float) and age > BACKLOG_BEHIND_S:
            problems.append(f"backlog growing: {backlog} line(s), the oldest {age:.0f} s old")

    refusals = source.get("refusals")
    spool_errors = _int(source.get("spool_errors"))
    if refusals is None:
        problems.append("refusals in the last 24 h unknown (the count cannot be read)")
    elif _int(refusals) or spool_errors:
        counted = max(_int(refusals), spool_errors)
        problems.append(f"refused {counted} line(s)")
    for refused in _list(source.get("refused"))[-3:]:
        entry = _mapping(refused)
        details.append(f"refused {entry.get('kind', '?')}: {entry.get('reason', '?')}")

    writer_errors = [str(line) for line in _list(source.get("writer_errors"))]
    if writer_errors:
        problems.append(f"{len(writer_errors)} writer error(s)")
        details.extend(f"writer: {line}" for line in writer_errors[-3:])
    abandoned = _list(source.get("abandoned"))
    if abandoned:
        problems.append(f"{len(abandoned)} abandoned segment(s)")

    if problems:
        return _row(name, Status.WARNING, f"{identity}: " + "; ".join(problems), "\n".join(details))
    if source.get("idle"):
        return _row(name, Status.OK, f"{identity}: idle", "\n".join(details))
    pending = f", {backlog} line(s) pending" if backlog else ""
    return _row(name, Status.OK, f"{identity}: receiving{pending}", "\n".join(details))


def _transcript_rows(transcripts: Mapping[str, Any]) -> list[CheckResult]:
    rows = []
    for root, raw in sorted(transcripts.items()):
        source = _mapping(raw)
        identity = str(source.get("identity") or root)
        name = f"agent_record_transcripts.{identity}"
        problems: list[str] = []
        if source.get("ok") is False or source.get("error"):
            problems.append(f"cannot be read: {source.get('error') or 'not ok'}")
        file_errors = _mapping(source.get("file_errors"))
        if file_errors:
            problems.append(f"{len(file_errors)} file(s) cannot be read")
        if source.get("stopped"):
            problems.append(
                f"stopped: a live transcript has not reached the record for {_STOPPED_AFTER_TEXT}"
            )
        details = "\n".join([f"transcripts {root}"] + [f"{k}: {v}" for k, v in file_errors.items()])
        if problems:
            rows.append(
                _row(
                    name, Status.WARNING, f"{identity} transcripts: " + "; ".join(problems), details
                )
            )
        elif source.get("idle"):
            rows.append(_row(name, Status.OK, f"{identity} transcripts: idle", details))
        else:
            rows.append(_row(name, Status.OK, f"{identity} transcripts: receiving", details))
    return rows


def _export_row(export: Mapping[str, Any], backend: Mapping[str, Any]) -> CheckResult:
    name = "agent_record_export"
    last = export.get("last_exported_day")
    through = f"last exported day {last}" if last else "no day exported yet"
    catch_up = f"`{_EXPORT_COMMAND}` runs the catch-up export now."
    archive_error = export.get("archive_error")
    if archive_error:
        return _row(name, Status.WARNING, f"Export: {archive_error}", catch_up)
    last_error = export.get("last_error")
    if last_error:
        return _row(name, Status.WARNING, f"Export failing: {last_error} ({through})", catch_up)
    lag = export.get("lag_days")
    if lag is None:
        reason = "the store is not ok" if backend.get("ok") is not True else "not reported"
        return _row(name, Status.WARNING, f"Export lag unknown: {reason} ({through})")
    if _int(lag) > EXPORT_LAG_ALLOWED_DAYS:
        return _row(
            name,
            Status.WARNING,
            f"Export {_int(lag)} closed day(s) behind ({through})",
            catch_up,
        )
    return _row(name, Status.OK, f"Export up to date ({through})")


# ---------------------------------------------------------------------------
# Reading a report the record wrote, without trusting its shape
# ---------------------------------------------------------------------------


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
