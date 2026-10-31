"""Claude Code transcript lines to record entries.

:func:`parse_line` is pure: it takes the source identity, the file's path
relative to ``projects/`` and one line's bytes, and returns the entries that
line holds. Reading files, offsets and generations belong to the tailer.

Each content block of a line becomes its own entry, keyed
``cc:<identity>:<sessionId>:<uuid>:<block index>`` (a line with no usable
``uuid`` is keyed by a hash of the line). The kinds are:

- ``prompt`` — a user line's text, or one of its image or document blocks;
- ``answer`` — an assistant text block, with the line's ``message.usage``
  token counts (the record's only copy of them), and ``api_error``, the kind
  of failure, on an answer the model API failed (``isApiErrorMessage``);
- ``thinking`` — the block's ``thinking`` text, kept even when empty (an
  ``omitted`` display writes the block with no text); ``redacted_thinking``
  is a ``thinking`` entry flagged ``redacted``;
- ``tool_call`` and ``tool_result`` — the whole block, images included;
- ``subagent`` — a tool result that names an ``agentId``, whatever the tool
  was called. It points at ``<session>/subagents/agent-<agentId>.jsonl``,
  whose own lines carry the same ``agent_id``, taken from the file stem;
- ``debug`` — everything else: other line types, unknown blocks and lines
  that are not JSON objects, holding the raw line (or block). Nothing is
  dropped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from posixpath import basename
from typing import Any

from osprey.agent_record.contract import (
    KIND_ANSWER,
    KIND_DEBUG,
    KIND_PROMPT,
    KIND_SUBAGENT,
    KIND_THINKING,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    Entry,
    EntryInvalid,
    check_projects_relpath,
    parse_ts,
    transcript_key,
)

__all__ = ["SUBAGENTS_DIR", "TRANSCRIPT_SUFFIX", "TranscriptPath", "parse_line", "split_relpath"]

#: The directory, beside a session's transcript, that holds its subagents' files.
SUBAGENTS_DIR = "subagents"
_AGENT_STEM_PREFIX = "agent-"
#: Only files with this suffix are transcripts.
TRANSCRIPT_SUFFIX = ".jsonl"

#: ``agentId: <id>`` as an Agent/Task result's text states it.
_AGENT_ID_IN_TEXT = re.compile(r"\bagentId:\s*([A-Za-z0-9._-]+)")


class TranscriptPath:
    """What a path relative to ``projects/`` says about its transcript.

    ``<project>/<session>.jsonl`` is a session's main transcript;
    ``<project>/<session>/subagents/agent-<id>.jsonl`` is one of its
    subagents, and ``agent_id`` is ``<id>``.
    """

    __slots__ = ("agent_id", "project", "relpath", "session_id")

    def __init__(self, relpath: str, project: str, session_id: str, agent_id: str) -> None:
        self.relpath = relpath
        self.project = project
        self.session_id = session_id
        self.agent_id = agent_id

    def subagent_relpath(self, session_id: str, agent_id: str) -> str:
        """The path of *agent_id*'s transcript under *session_id* in this project."""
        stem = f"{_AGENT_STEM_PREFIX}{agent_id}{TRANSCRIPT_SUFFIX}"
        parts = [p for p in (self.project, session_id, SUBAGENTS_DIR, stem) if p]
        return "/".join(parts)


def _stem(name: str) -> str:
    return name.removesuffix(TRANSCRIPT_SUFFIX)


def split_relpath(relpath: str) -> TranscriptPath:
    """Parse *relpath* (relative to ``projects/``) into project, session and agent.

    Raises:
        ValueError: If *relpath* is not a plain path relative to ``projects/``.
    """
    check_projects_relpath(relpath)
    parts = relpath.split("/")
    if len(parts) >= 3 and parts[-2] == SUBAGENTS_DIR:
        stem = _stem(parts[-1])
        agent_id = stem.removeprefix(_AGENT_STEM_PREFIX)
        return TranscriptPath(relpath, "/".join(parts[:-3]), parts[-3], agent_id)
    return TranscriptPath(relpath, "/".join(parts[:-1]), _stem(basename(relpath)), "")


def _usable(value: Any) -> str:
    """*value* when it can be part of a key (a non-empty string with no ``:``), else ``""``."""
    return value if isinstance(value, str) and value and ":" not in value else ""


class _Line:
    """One parsed line and the fields every entry from it shares."""

    def __init__(
        self,
        identity: str,
        path: TranscriptPath,
        raw: bytes,
        obj: Mapping[str, Any] | None,
        now: datetime,
    ) -> None:
        self.identity = identity
        self.path = path
        self.raw = raw
        self.obj: Mapping[str, Any] = obj or {}
        self.session_id = _usable(self.obj.get("sessionId")) or path.session_id
        self.uuid = _usable(self.obj.get("uuid")) or None
        self.agent_id = path.agent_id or _usable(self.obj.get("agentId"))
        try:
            self.ts = parse_ts(self.obj.get("timestamp"))
        except EntryInvalid:
            self.ts = now
        parent = _usable(self.obj.get("parentUuid"))
        self.parent_key = transcript_key(identity, self.session_id, parent, 0) if parent else ""

    def entry(
        self,
        kind: str,
        block_index: int,
        body: Mapping[str, Any],
        *,
        tool_use_id: str = "",
        agent_id: str | None = None,
    ) -> Entry:
        return Entry(
            key=transcript_key(
                self.identity, self.session_id, self.uuid, block_index, line=self.raw
            ),
            kind=kind,
            ts=self.ts,
            identity=self.identity,
            body=body,
            session_id=self.session_id,
            agent_id=self.agent_id if agent_id is None else agent_id,
            parent_key=self.parent_key,
            tool_use_id=tool_use_id if isinstance(tool_use_id, str) else "",
        )

    def debug(self, line_type: Any) -> list[Entry]:
        text = self.raw.decode("utf-8", errors="replace")
        return [self.entry(KIND_DEBUG, 0, {"type": line_type, "raw": text})]

    def debug_block(self, index: int, block: Any) -> Entry:
        return self.entry(KIND_DEBUG, index, {"type": self.obj.get("type"), "block": block})


def parse_line(
    identity: str, relpath: str, line: bytes, *, now: datetime | None = None
) -> list[Entry]:
    """The entries one transcript line holds, in block order.

    Args:
        identity: The trusted source identity of the transcript volume.
        relpath: The file's path relative to ``projects/``.
        line: One line of the file, with or without its line ending.
        now: The time given to a line that carries no usable ``timestamp``;
            defaults to the current time.

    Returns:
        One entry per content block; a single ``debug`` entry for a line the
        parser does not know; nothing for a blank line.

    Raises:
        ValueError: If *identity* or *relpath* is unusable.
    """
    path = split_relpath(relpath)
    raw = line.rstrip(b"\r\n")
    if not raw.strip():
        return []
    when = now if now is not None else datetime.now(UTC)
    try:
        obj = json.loads(raw)
    except ValueError:
        obj = None
    if not isinstance(obj, dict):
        return _Line(identity, path, raw, None, when).debug(None)
    parsed = _Line(identity, path, raw, obj, when)
    line_type = obj.get("type")
    message = obj.get("message")
    if not isinstance(message, Mapping):
        message = {}
    content = message.get("content")
    if line_type == "user" and content not in (None, "", []):
        return _user_entries(parsed, content)
    if line_type == "assistant" and isinstance(content, list) and content:
        return _assistant_entries(parsed, message, content)
    return parsed.debug(line_type)


def _user_entries(line: _Line, content: Any) -> list[Entry]:
    if isinstance(content, str):
        return [line.entry(KIND_PROMPT, 0, _prompt_body(line, {"text": content}))]
    if not isinstance(content, list):
        return line.debug(line.obj.get("type"))
    results = [b for b in content if isinstance(b, Mapping) and b.get("type") == "tool_result"]
    entries = []
    for index, block in enumerate(content):
        btype = block.get("type") if isinstance(block, Mapping) else None
        if btype == "text":
            entries.append(
                line.entry(KIND_PROMPT, index, _prompt_body(line, {"text": block.get("text", "")}))
            )
        elif btype in ("image", "document"):
            entries.append(line.entry(KIND_PROMPT, index, _prompt_body(line, {btype: dict(block)})))
        elif btype == "tool_result":
            entries.append(_tool_result(line, index, block, sole=len(results) == 1))
        else:
            entries.append(line.debug_block(index, block))
    return entries


def _prompt_body(line: _Line, body: dict[str, Any]) -> dict[str, Any]:
    if line.obj.get("isMeta") is True:
        body["is_meta"] = True
    return body


def _tool_result(line: _Line, index: int, block: Mapping[str, Any], *, sole: bool) -> Entry:
    body = {k: v for k, v in block.items() if k != "type"}
    tool_use_id = block.get("tool_use_id", "")
    structured = line.obj.get("toolUseResult") if sole else None
    agent_id = _result_agent_id(structured, block.get("content"))
    if not agent_id:
        return line.entry(KIND_TOOL_RESULT, index, body, tool_use_id=tool_use_id)
    body["agent_id"] = agent_id
    body["transcript"] = line.path.subagent_relpath(line.session_id, agent_id)
    if isinstance(structured, Mapping):
        for name, field_name in (("status", "status"), ("isAsync", "is_async")):
            if name in structured:
                body[field_name] = structured[name]
    return line.entry(KIND_SUBAGENT, index, body, tool_use_id=tool_use_id, agent_id=agent_id)


def _result_agent_id(structured: Any, content: Any) -> str:
    """The subagent a tool result names: its structured ``agentId``, else (with no
    structured result) the one its text states."""
    if isinstance(structured, Mapping):
        # A structured result is authoritative: ordinary tool output that
        # merely mentions an agentId (a grep, a file read) is not a subagent.
        return _usable(structured.get("agentId"))
    for text in _texts(content):
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, Mapping) and _usable(parsed.get("agentId")):
            return _usable(parsed["agentId"])
        match = _AGENT_ID_IN_TEXT.search(text)
        if match:
            return match.group(1)
    return ""


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            b["text"]
            for b in content
            if isinstance(b, Mapping) and b.get("type") == "text" and isinstance(b.get("text"), str)
        ]
    return []


def _assistant_entries(line: _Line, message: Mapping[str, Any], content: list) -> list[Entry]:
    entries = []
    for index, block in enumerate(content):
        btype = block.get("type") if isinstance(block, Mapping) else None
        if btype == "text":
            body: dict[str, Any] = {"text": block.get("text", "")}
            for name, field_name in (
                ("model", "model"),
                ("id", "message_id"),
                ("stop_reason", "stop_reason"),
                ("usage", "usage"),
            ):
                if message.get(name) is not None:
                    body[field_name] = message[name]
            if line.obj.get("requestId"):
                body["request_id"] = line.obj["requestId"]
            if line.obj.get("isApiErrorMessage") and isinstance(line.obj.get("error"), str):
                body["api_error"] = line.obj["error"]
            entries.append(line.entry(KIND_ANSWER, index, body))
        elif btype == "thinking":
            text = block.get("thinking")
            entries.append(
                line.entry(KIND_THINKING, index, {"text": text if isinstance(text, str) else ""})
            )
        elif btype == "redacted_thinking":
            entries.append(line.entry(KIND_THINKING, index, {"text": "", "redacted": True}))
        elif btype == "tool_use":
            body = {k: v for k, v in block.items() if k != "type"}
            entries.append(line.entry(KIND_TOOL_CALL, index, body, tool_use_id=block.get("id", "")))
        else:
            entries.append(line.debug_block(index, block))
    return entries
