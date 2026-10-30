"""Local liveness reads of Claude Code's native transcripts.

Claude Code writes each session to ``<config-dir>/projects/<encoded>/`` as
JSONL. What a session did — its tool calls, subagents and conversation — is
read from the agent record (:mod:`osprey.mcp_server.session`); this module
answers only what the local files alone can: which transcript is current or
names a session, and whether its tail shows a turn in flight. The text helpers
here are shared by the readers that turn record entries back into events.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from osprey.agent_runner.project_paths import claude_project_dir

logger = logging.getLogger("osprey.mcp_server.workspace.transcript_reader")

_MCP_TOOL_PREFIX = "mcp__"

MAX_RESULT_LENGTH = 500
MAX_ERROR_RESULT_LENGTH = 2000
MAX_CHAT_MESSAGE_LENGTH = 2000

# Claude Code writes this marker as a synthetic user entry when a turn is
# cancelled, and wraps slash-command echoes and their captured output in these
# tags. Both are the TUI's own bookkeeping, not conversation. A slash command's
# echo opens with ``<command-name>`` for a built-in and with
# ``<command-message>`` for a skill or a project command.
INTERRUPT_MARKER = "[Request interrupted by user"
COMMAND_NAME_PREFIX = "<command-name>"
COMMAND_MESSAGE_PREFIX = "<command-message>"
COMMAND_ECHO_PREFIXES = (COMMAND_NAME_PREFIX, COMMAND_MESSAGE_PREFIX)
LOCAL_COMMAND_STDOUT_PREFIX = "<local-command-stdout>"
LOCAL_COMMAND_STDERR_PREFIX = "<local-command-stderr>"
LOCAL_COMMAND_PREFIXES = (LOCAL_COMMAND_STDOUT_PREFIX, LOCAL_COMMAND_STDERR_PREFIX)
_BOOKKEEPING_PREFIXES = (*COMMAND_ECHO_PREFIXES, *LOCAL_COMMAND_PREFIXES)

# A transcript grows without bound, but the tail rule only ever looks at its
# newest message entry, so only this much of the end is read.
_TAIL_READ_BYTES = 256 * 1024


def _is_error_response(result_str: str) -> bool:
    """Check if a result string contains an OSPREY error envelope."""
    try:
        parsed = json.loads(result_str)
        return isinstance(parsed, dict) and parsed.get("error") is True
    except (json.JSONDecodeError, ValueError, TypeError):
        return False


def _truncate(text: str, is_error: bool) -> str:
    """Truncate result text using error-aware limits."""
    max_len = MAX_ERROR_RESULT_LENGTH if is_error else MAX_RESULT_LENGTH
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _extract_text(content_blocks: list) -> str:
    """Extract concatenated text from a list of tool_result content blocks."""
    parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts) if parts else ""


def _split_mcp_tool_name(tool_name: str) -> tuple[str, str, str] | None:
    """Split an MCP tool name into (short_name, server, full_name).

    Matches structurally on Claude Code's ``mcp__<server>__<tool>`` naming
    rather than an enumerated server list: every MCP server in a deployed
    session is deployment-declared (the registry and the profile render the
    agent's ``.mcp.json``), so the prefix alone is the audit boundary and a
    facility's custom server is captured without any OSPREY source edit. Only
    the first ``__`` after the prefix delimits the server — a further ``__``
    belongs to the tool name. Returns ``None`` for built-in (non-MCP) tools.
    """
    if not tool_name.startswith(_MCP_TOOL_PREFIX):
        return None
    server, sep, short_name = tool_name[len(_MCP_TOOL_PREFIX) :].partition("__")
    if not sep or not server or not short_name:
        return None
    return short_name, server, tool_name


def _conversation_text(entry: dict) -> str:
    """Extract only text content blocks from a user or assistant entry."""
    message = entry.get("message", {})
    content_blocks = message.get("content", [])
    if isinstance(content_blocks, str):
        return content_blocks.strip()
    parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


def is_bookkeeping_entry(entry: dict) -> bool:
    """Report whether an entry is the TUI's own accounting, not conversation.

    Covers meta entries, the slash-command echo and its captured output, and
    the marker written when a turn is cancelled. Replaying a conversation onto
    another surface shows none of these.
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("isMeta"):
        return True
    text = _conversation_text(entry)
    if not text:
        return False
    return text.startswith(INTERRUPT_MARKER) or text.startswith(_BOOKKEEPING_PREFIXES)


def _entry_epoch(entry: dict) -> float | None:
    """Parse an entry's ISO-8601 timestamp into POSIX seconds, or None."""
    raw = entry.get("timestamp")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.timestamp()


def _scan_last_message(lines: list[str]) -> dict | None:
    """Return the newest user or assistant entry among JSONL lines."""
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(entry, dict) and entry.get("type") in ("user", "assistant"):
            return entry
    return None


def _last_message_entry(path: Path) -> dict | None:
    """Read a transcript's newest user or assistant entry.

    Only the tail of the file is read; a transcript whose closing entries are
    all larger than that window falls back to a full read.
    """
    try:
        size = path.stat().st_size
        truncated = size > _TAIL_READ_BYTES
        with path.open("rb") as handle:
            if truncated:
                handle.seek(size - _TAIL_READ_BYTES)
                handle.readline()  # discard the partial line the seek landed in
            chunk = handle.read()
    except OSError:
        return None

    entry = _scan_last_message(chunk.decode("utf-8", "replace").splitlines())
    if entry is None and truncated:
        try:
            entry = _scan_last_message(path.read_text().splitlines())
        except OSError:
            return None
    return entry


def tail_state(path: Path, busy_since: float | None) -> Literal["busy", "idle", "unknown"]:
    """Classify a transcript's tail as a turn in flight, at rest, or unreadable.

    A text-bearing user entry at the end of the transcript is a prompt the
    agent has not answered yet, so the session is ``busy``. Two shapes say the
    opposite: the interrupt marker, and slash-command output written after
    ``busy_since`` (the caller's busy stamp, in POSIX seconds; ``None`` when
    the caller holds no busy stamp at all). Everything else — an assistant
    entry, a tool-result-only user entry, other bookkeeping — carries no
    evidence either way and returns ``unknown``, leaving the caller with
    whatever its own state store says.
    """
    entry = _last_message_entry(path)
    if entry is None or entry.get("type") != "user":
        return "unknown"

    text = _conversation_text(entry)
    if not text:
        return "unknown"
    if text.startswith(INTERRUPT_MARKER):
        return "idle"
    if text.startswith(LOCAL_COMMAND_PREFIXES):
        if busy_since is None:
            return "idle"
        stamp = _entry_epoch(entry)
        return "idle" if stamp is not None and stamp > busy_since else "unknown"
    if is_bookkeeping_entry(entry):
        return "unknown"
    return "busy"


def _transcript_file(directory: Path, name: str) -> Path | None:
    """The ``<name>.jsonl`` file directly inside *directory*, or ``None``.

    An id is a file name, never a path: it names a transcript only when the
    joined path, with symlinks followed, sits directly in *directory* with
    symlinks followed too. A separator, an absolute id, an empty id, a name
    the OS cannot represent, or a link leading out of the directory reads
    as a transcript that does not exist, so every caller gets the answer a
    missing file gives and no id reaches a file outside the directory.

    The unresolved joined path is what is returned, so callers see the same
    path the directory listing shows.
    """
    if not name:
        return None
    candidate = directory / f"{name}.jsonl"
    try:
        contained = candidate.resolve().parent == directory.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not contained or not candidate.is_file():
        return None
    return candidate


class TranscriptReader:
    """Locate a project's Claude Code transcripts on local disk."""

    def __init__(self, project_dir: Path | str) -> None:
        self.project_dir = Path(project_dir).resolve()

    def find_transcript_dir(self) -> Path | None:
        """Locate the Claude Code transcript directory for this project.

        Claude Code stores transcripts in ``<config-dir>/projects/<encoded>/``;
        see :func:`osprey.agent_runner.project_paths.claude_project_dir` for
        how the root and the encoded name are resolved.
        """
        transcript_dir = claude_project_dir(self.project_dir)
        if transcript_dir.is_dir():
            return transcript_dir
        return None

    def find_current_transcript(self) -> Path | None:
        """Find the most recently modified ``.jsonl`` transcript file."""
        transcript_dir = self.find_transcript_dir()
        if not transcript_dir:
            return None
        jsonl_files = list(transcript_dir.glob("*.jsonl"))
        if not jsonl_files:
            return None
        return max(jsonl_files, key=lambda p: p.stat().st_mtime)

    def find_transcript_by_id(self, session_id: str) -> Path | None:
        """Find the transcript file for a specific session ID.

        Claude Code uses session UUIDs as filenames, so this constructs
        the expected path directly without scanning; an id that is not a
        plain file name in the transcript directory reads as not found.

        Returns:
            Path to the ``.jsonl`` file, or None if not found.
        """
        transcript_dir = self.find_transcript_dir()
        if not transcript_dir:
            return None
        return _transcript_file(transcript_dir, session_id)
