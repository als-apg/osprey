"""Approval-log checks for the landing-page demo video recorder.

The approval hook appends one JSON record per decision to
``<project>/build/.claude/hooks/hook_debug.jsonl``. Each record carries
``ts``, ``hook``, ``tool`` (the full tool name, ``mcp__<server>__<name>`` for
MCP tools), ``status`` and optionally ``tool_use_id`` and ``detail``.

The recorder notes the file size before a step with :func:`log_offset` and
afterwards reads the approval prompts that step raised with :func:`find_asks`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_TOOL_TOKEN = re.compile(r"(?:^|\s)tool=(\S+)")
_MCP_PREFIX = re.compile(r"^mcp__.+?__")


@dataclass(frozen=True)
class Ask:
    """One approval prompt: the tool call it gates and the tool's short name."""

    tool_use_id: str | None
    name: str


def hook_log_path(project: Path) -> Path:
    """The approval hook's log inside the rendered project ``build/``."""
    return Path(project) / "build" / ".claude" / "hooks" / "hook_debug.jsonl"


def log_offset(path: Path) -> int:
    """The log's current size in bytes, or 0 while the file does not exist."""
    try:
        return Path(path).stat().st_size
    except FileNotFoundError:
        return 0


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


def find_asks(path: Path, offset: int) -> list[Ask]:
    """Every ``status == "ask"`` record appended to the log after ``offset``.

    A missing file yields no asks. Lines that are not a JSON object, lack a
    usable tool name, or are still being written (no trailing newline) are
    skipped.
    """
    try:
        with Path(path).open("rb") as fh:
            fh.seek(max(offset, 0))
            data = fh.read()
    except FileNotFoundError:
        return []

    asks: list[Ask] = []
    for raw in data.split(b"\n")[:-1]:
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("status") != "ask":
            continue
        name = _ask_name(record)
        if name is None:
            continue
        tool_use_id = record.get("tool_use_id")
        asks.append(Ask(tool_use_id if isinstance(tool_use_id, str) else None, name))
    return asks
