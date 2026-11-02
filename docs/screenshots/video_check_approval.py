"""Approval-log checks for the landing-page demo video recorder.

With hook debug on, every hook decision lands on the agent record as one
``debug`` entry whose body names ``source: "hook"`` and carries ``ts``,
``hook``, ``tool`` (the full tool name, ``mcp__<server>__<name>`` for MCP
tools), ``status`` and optionally ``tool_use_id`` and ``detail``.

The recorder takes a :func:`log_mark` before a step and afterwards reads the
hook debug lines written since with a :data:`HookLog` (:func:`record_log` for
the project's record); :func:`find_asks` names the approval prompts among them
and :func:`ask_record_count` counts every prompt, named or not.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

#: The ``source`` hook debug lines carry on the record.
HOOK_SOURCE = "hook"

_TOOL_TOKEN = re.compile(r"(?:^|\s)tool=(\S+)")
_MCP_PREFIX = re.compile(r"^mcp__.+?__")

#: Reads the hook debug bodies written since a mark, oldest first.
HookLog = Callable[[datetime], list[dict]]


class RecordUnavailable(RuntimeError):
    """The agent record did not answer a whole read of the hook debug lines."""


@dataclass(frozen=True)
class Ask:
    """One approval prompt: the tool call it gates and the tool's short name."""

    tool_use_id: str | None
    name: str


def log_mark() -> datetime:
    """The moment a step starts: the lines it raises are the ones written after."""
    return datetime.now(UTC)


def record_log(project: Path) -> HookLog:
    """A :data:`HookLog` over the agent record of the deployed *project* (its repo root).

    Each read holds the project's record (the running stack's, or a host-local
    one), drains ingest so a line the hook just wrote is counted, and raises
    :class:`RecordUnavailable` rather than report an unanswered read as no asks.
    """

    def read(since: datetime) -> list[dict]:
        import yaml

        from osprey.agent_record.client import RecordClient
        from osprey.agent_record.contract import KIND_DEBUG, Drain
        from osprey.agent_record.host import ensure_host_record

        config_path = Path(project) / "build" / "config.yml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        with ensure_host_record(project, config) as record:
            client = RecordClient(record.url, record.login, record.token)
            answer = client.entries(KIND_DEBUG, since=since, source=HOOK_SOURCE, wait=Drain())
        if not answer.available:
            raise RecordUnavailable(f"the agent record is unavailable: {answer.unavailable}")
        if not answer.complete:
            raise RecordUnavailable(f"the hook debug read is incomplete: {answer.reason}")
        return hook_bodies(answer.data, since)

    return read


def hook_bodies(items: object, since: datetime) -> list[dict]:
    """The hook debug bodies among record *items* written at or after *since*, oldest first.

    *items* is the record's newest-first entry list. Anything that is not a
    hook debug entry, or carries no readable ``ts``, is skipped.
    """
    if not isinstance(items, list):
        return []
    bodies: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        body = item.get("body")
        if not isinstance(body, dict) or body.get("source") != HOOK_SOURCE:
            continue
        written = _parse_ts(item.get("ts"))
        if written is None or written < since:
            continue
        bodies.append(body)
    bodies.reverse()
    return bodies


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _ask_name(record: dict) -> str | None:
    detail = record.get("detail")
    if isinstance(detail, str):
        match = _TOOL_TOKEN.search(detail)
        if match:
            return match.group(1)
    tool = record.get("tool")
    if not isinstance(tool, str) or not tool:
        return None
    return _MCP_PREFIX.sub("", tool, count=1)


def ask_records(records: list[dict]) -> list[dict]:
    """Every ``status == "ask"`` record among *records*, named or not."""
    return [r for r in records if isinstance(r, dict) and r.get("status") == "ask"]


def asks_in(records: list[dict]) -> list[Ask]:
    """The approval prompts among *records* that name a tool, in order."""
    asks: list[Ask] = []
    for record in ask_records(records):
        name = _ask_name(record)
        if name is None:
            continue
        tool_use_id = record.get("tool_use_id")
        asks.append(Ask(tool_use_id if isinstance(tool_use_id, str) else None, name))
    return asks


def find_asks(log: HookLog, mark: datetime) -> list[Ask]:
    """Every named approval prompt the hooks logged since *mark*, oldest first."""
    return asks_in(log(mark))


def ask_record_count(log: HookLog, mark: datetime) -> int:
    """How many approval prompts the hooks logged since *mark*, named or not."""
    return len(ask_records(log(mark)))
