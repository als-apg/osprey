"""Turn-end check for the landing-page demo video recorder.

After the recorder submits a prompt it needs to know when Claude Code has
finished answering. The terminal screen is a poor witness; the session
transcript is a good one. :func:`turn_state` reads the newest user or
assistant entry of that JSONL file and classifies the turn:

- ``errored``: the newest entry is Claude Code's stand-in for a failed API
  call, an assistant entry whose ``message.model`` is ``"<synthetic>"``,
  written after the prompt was submitted.
- ``ended``: the newest entry is an assistant entry with
  ``message.stop_reason == "end_turn"``, written after the prompt was
  submitted, no background agent the session launched is still owed its
  completion notice, nothing queued for the session is still undelivered and
  nothing was delivered after it (a delivered report resumes the session),
  and the file has not changed for ``stable_s`` seconds.
  The agent ends its own turn while its background agents work, and their
  hand-back resumes it, so an ``end_turn`` alone does not end the work.
- ``running``: anything else, including a missing or unreadable file.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from osprey.agent_runner.project_paths import encode_claude_project_path
from osprey.mcp_server.workspace.transcript_reader import (
    TranscriptReader,
    _entry_epoch,
    is_bookkeeping_entry,
)

TurnState = Literal["running", "ended", "errored"]

SYNTHETIC_MODEL = "<synthetic>"


def transcript_path(
    build_dir: Path | str, session_id: str, config_dir: Path | None = None
) -> Path | None:
    """Return the transcript file of *session_id* for the project at *build_dir*.

    ``config_dir`` is the session's own Claude Code config dir, when it runs
    under one this process does not share; otherwise this process's is used.
    """
    if config_dir is None:
        return TranscriptReader(build_dir).find_transcript_by_id(session_id)
    path = config_dir / "projects" / encode_claude_project_path(build_dir) / f"{session_id}.jsonl"
    return path if path.is_file() else None


ASYNC_LAUNCH_MARKER = "Async agent launched"
_NOTIFIED_TOOL_USE = re.compile(r"<task-notification>.*?<tool-use-id>([^<]+)</tool-use-id>", re.S)


def _block_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _pending_agents(lines: list[str]) -> set[str]:
    """Tool-use ids of background agents launched but not yet reported complete."""
    launched: set[str] = set()
    notified: set[str] = set()
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        if entry.get("type") == "queue-operation" and isinstance(content, str):
            notified.update(_NOTIFIED_TOOL_USE.findall(content))
            continue
        message = entry.get("message")
        blocks = message.get("content") if isinstance(message, dict) else None
        if isinstance(blocks, str):
            notified.update(_NOTIFIED_TOOL_USE.findall(blocks))
            continue
        for block in blocks if isinstance(blocks, list) else []:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and ASYNC_LAUNCH_MARKER in _block_text(block.get("content"))
            ):
                launched.add(str(block.get("tool_use_id")))
    return launched - notified


def _queue_state(lines: list[str]) -> tuple[int, float | None]:
    """Items queued for the session but not yet delivered, and the last delivery.

    Background agents' reports are queued while the agent is busy and delivered
    to it once it is free, which resumes it; a removed item was taken up
    mid-turn.
    """
    queued = 0
    delivered_at: float | None = None
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("type") != "queue-operation":
            continue
        operation = entry.get("operation")
        if operation == "enqueue":
            queued += 1
        elif operation in ("dequeue", "remove"):
            queued -= 1
            stamp = _entry_epoch(entry)
            if operation == "dequeue" and stamp is not None:
                delivered_at = stamp if delivered_at is None else max(delivered_at, stamp)
    return max(queued, 0), delivered_at


def _newest_message(lines: list[str]) -> dict | None:
    """Return the newest user or assistant entry that is conversation.

    Blank lines, invalid JSON, non-message entry types (``system``, snapshots)
    and TUI bookkeeping (meta entries, slash-command echoes) are skipped.
    """
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(entry, dict) or entry.get("type") not in ("user", "assistant"):
            continue
        if is_bookkeeping_entry(entry):
            continue
        return entry
    return None


def turn_state(
    transcript_path: Path | str | None,
    submitted_at: float,
    stable_s: float = 2.0,
    now: Callable[[], float] = time.time,
) -> TurnState:
    """Classify the turn started at *submitted_at* (POSIX seconds)."""
    if transcript_path is None:
        return "running"
    path = Path(transcript_path)
    try:
        mtime = path.stat().st_mtime
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "running"

    entry = _newest_message(lines)
    if entry is None or entry.get("type") != "assistant":
        return "running"
    stamp = _entry_epoch(entry)
    if stamp is None or stamp <= submitted_at:
        return "running"

    message = entry.get("message")
    if not isinstance(message, dict):
        return "running"
    if message.get("model") == SYNTHETIC_MODEL:
        return "errored"
    queued, delivered_at = _queue_state(lines)
    if (
        message.get("stop_reason") == "end_turn"
        and now() - mtime >= stable_s
        and not _pending_agents(lines)
        and not queued
        and (delivered_at is None or stamp > delivered_at)
    ):
        return "ended"
    return "running"
