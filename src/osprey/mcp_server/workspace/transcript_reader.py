"""Read Claude Code native transcripts to extract OSPREY tool-call and agent events.

Replaces the former PostToolUse audit hook (``osprey_audit.py``) and
SubagentStart/SubagentStop lifecycle hook (``osprey_subagent_lifecycle.py``).
Claude Code already logs every tool call to ``~/.claude/projects/<encoded>/``
as JSONL — this module reads those transcripts on demand.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("osprey.mcp_server.workspace.transcript_reader")

OSPREY_MCP_PREFIXES = (
    "mcp__controls__",
    "mcp__python__",
    "mcp__workspace__",
    "mcp__ariel__",
    "mcp__accelpapers__",
    "mcp__matlab__",
    "mcp__channel-finder__",
    "mcp__confluence__",
)

MAX_RESULT_LENGTH = 500
MAX_ERROR_RESULT_LENGTH = 2000
MAX_CHAT_MESSAGE_LENGTH = 2000


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


def _match_osprey_prefix(tool_name: str) -> tuple[str, str, str] | None:
    """Match a tool name against OSPREY prefixes.

    Returns (short_name, server, full_name) or None if no match.
    """
    for prefix in OSPREY_MCP_PREFIXES:
        if tool_name.startswith(prefix):
            short_name = tool_name[len(prefix) :]
            parts = tool_name.split("__")
            server = parts[1] if len(parts) >= 3 else "unknown"
            return short_name, server, tool_name
    return None


class TranscriptReader:
    """Read Claude Code native JSONL transcripts and extract OSPREY events."""

    def __init__(self, project_dir: Path | str) -> None:
        self.project_dir = Path(project_dir).resolve()

    def find_transcript_dir(self) -> Path | None:
        """Locate the Claude Code transcript directory for this project.

        Claude Code encodes the project path as: ``str(path).replace("/", "-")``
        and stores transcripts in ``~/.claude/projects/<encoded>/``.
        """
        encoded = str(self.project_dir).replace("/", "-")
        # Strip leading dash from the encoding
        if encoded.startswith("-"):
            encoded = encoded[1:]
        transcript_dir = Path.home() / ".claude" / "projects" / encoded
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

    def read_session(
        self,
        path: Path,
        *,
        include_subagents: bool = True,
        agent_id: str | None = None,
    ) -> list[dict]:
        """Read a single transcript file and extract OSPREY events.

        Args:
            path: Path to a ``.jsonl`` transcript file.
            include_subagents: If True, also read subagent transcripts.
            agent_id: If set, tag all tool_call events with this agent_id.

        Returns:
            List of event dicts (tool_call, agent_start, agent_stop).
        """
        if not path.is_file():
            return []

        events: list[dict] = []

        # Index: tool_use_id → (tool_use block, timestamp from assistant entry)
        tool_uses: dict[str, tuple[dict, str, str | None]] = {}
        # Index: tool_use_id → Task tool_use for subagent tracking
        task_uses: dict[str, tuple[dict, str]] = {}

        session_id: str | None = None

        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            entry_type = entry.get("type")
            if not entry_type:
                continue

            # Capture session_id from any entry
            if not session_id and entry.get("sessionId"):
                session_id = entry["sessionId"]

            timestamp = entry.get("timestamp", "")

            if entry_type == "assistant":
                message = entry.get("message", {})
                content_blocks = message.get("content", [])
                for block in content_blocks:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "")
                    if not tool_id:
                        continue

                    if tool_name == "Task":
                        task_uses[tool_id] = (block, timestamp)
                    elif _match_osprey_prefix(tool_name):
                        tool_uses[tool_id] = (block, timestamp, session_id)

            elif entry_type == "tool_result":
                # Standalone tool_result entries
                tool_use_id = entry.get("tool_use_id", "")
                content = entry.get("content", "")
                is_err = entry.get("is_error", False)

                if tool_use_id in tool_uses:
                    block, ts, sid = tool_uses.pop(tool_use_id)
                    events.append(self._make_tool_event(block, content, is_err, ts, sid, agent_id))
                elif tool_use_id in task_uses:
                    block, ts = task_uses.pop(tool_use_id)
                    task_events = self._make_task_events(block, content, is_err, ts)
                    events.extend(task_events)

            elif entry_type == "user":
                message = entry.get("message", {})
                content_blocks = message.get("content", [])
                for block in content_blocks:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_use_id = block.get("tool_use_id", "")
                    content = block.get("content", "")
                    is_err = block.get("is_error", False)

                    if tool_use_id in tool_uses:
                        tu_block, ts, sid = tool_uses.pop(tool_use_id)
                        events.append(
                            self._make_tool_event(tu_block, content, is_err, ts, sid, agent_id)
                        )
                    elif tool_use_id in task_uses:
                        tu_block, ts = task_uses.pop(tool_use_id)
                        task_events = self._make_task_events(tu_block, content, is_err, ts)
                        events.extend(task_events)

        # Read subagent transcripts
        if include_subagents:
            # Subagent transcripts are in <session-uuid>/subagents/
            # The session dir shares the stem of the transcript file
            session_stem = path.stem  # e.g., "abc-123"
            subagent_dir = path.parent / session_stem / "subagents"
            if subagent_dir and subagent_dir.is_dir():
                for agent_file in sorted(subagent_dir.glob("*.jsonl")):
                    # Extract agent_id from filename: agent-<id>.jsonl
                    agent_fname = agent_file.stem
                    sub_events = self.read_session(
                        agent_file,
                        include_subagents=False,
                        agent_id=agent_fname,
                    )
                    events.extend(sub_events)

        # Sort by timestamp
        events.sort(key=lambda e: e.get("timestamp", ""))
        return events

    def read_current_session(self) -> list[dict]:
        """Read the most recent transcript for this project.

        Returns:
            List of event dicts, or empty list if no transcript found.
        """
        path = self.find_current_transcript()
        if not path:
            return []
        return self.read_session(path)

    def read_chat_history(self, path: Path) -> list[dict]:
        """Extract conversation turns (user text + assistant text) from a transcript.

        Returns a chronologically ordered list of dicts with keys:
        ``role`` ("user" | "assistant"), ``content`` (str), ``timestamp`` (str).

        Only text content blocks are captured — tool_use / tool_result blocks
        are skipped.  Individual messages are truncated to
        ``MAX_CHAT_MESSAGE_LENGTH`` characters.
        """
        if not path.is_file():
            return []

        turns: list[dict] = []

        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            entry_type = entry.get("type")
            timestamp = entry.get("timestamp", "")

            if entry_type == "user":
                text = self._extract_conversation_text(entry)
                if text:
                    if len(text) > MAX_CHAT_MESSAGE_LENGTH:
                        text = text[:MAX_CHAT_MESSAGE_LENGTH] + "..."
                    turns.append({"role": "user", "content": text, "timestamp": timestamp})

            elif entry_type == "assistant":
                text = self._extract_conversation_text(entry)
                if text:
                    if len(text) > MAX_CHAT_MESSAGE_LENGTH:
                        text = text[:MAX_CHAT_MESSAGE_LENGTH] + "..."
                    turns.append(
                        {"role": "assistant", "content": text, "timestamp": timestamp}
                    )

        return turns

    def read_current_chat_history(self) -> list[dict]:
        """Read chat history from the most recent transcript.

        Returns:
            List of conversation turn dicts, or empty list if no transcript found.
        """
        path = self.find_current_transcript()
        if not path:
            return []
        return self.read_chat_history(path)

    @staticmethod
    def _extract_conversation_text(entry: dict) -> str:
        """Extract only text content blocks from a user or assistant entry."""
        message = entry.get("message", {})
        content_blocks = message.get("content", [])
        parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()

    def _make_tool_event(
        self,
        tool_use_block: dict,
        content: str | list,
        is_error: bool,
        timestamp: str,
        session_id: str | None,
        agent_id: str | None,
    ) -> dict:
        """Build a tool_call event dict from a tool_use + tool_result pair."""
        tool_name = tool_use_block.get("name", "")
        match = _match_osprey_prefix(tool_name)
        if not match:
            return {}

        short_name, server, full_name = match

        # Build result summary string
        if isinstance(content, list):
            result_str = _extract_text(content)
        elif isinstance(content, dict):
            result_str = json.dumps(content)
        else:
            result_str = str(content)

        # Detect OSPREY error envelope
        if not is_error:
            is_error = _is_error_response(result_str)

        result_summary = _truncate(result_str, is_error)

        return {
            "type": "tool_call",
            "timestamp": timestamp,
            "tool": short_name,
            "full_tool_name": full_name,
            "server": server,
            "is_error": is_error,
            "session_id": session_id,
            "agent_id": agent_id,
            "arguments": tool_use_block.get("input", {}),
            "result_summary": result_summary,
        }

    def _make_task_events(
        self,
        task_use_block: dict,
        content: str | list,
        is_error: bool,
        timestamp: str,
    ) -> list[dict]:
        """Build agent_start and agent_stop events from a Task tool_use/result pair."""
        events: list[dict] = []
        task_input = task_use_block.get("input", {})
        agent_type = task_input.get("subagent_type", task_input.get("description", ""))

        # Extract agent_id from the result
        result_agent_id: str | None = None
        agent_transcript_path: str | None = None
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    result_agent_id = parsed.get("agentId")
                    agent_transcript_path = parsed.get("transcriptPath")
            except (json.JSONDecodeError, ValueError):
                pass
        elif isinstance(content, list):
            text = _extract_text(content)
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    result_agent_id = parsed.get("agentId")
                    agent_transcript_path = parsed.get("transcriptPath")
            except (json.JSONDecodeError, ValueError):
                pass

        events.append(
            {
                "type": "agent_start",
                "timestamp": timestamp,
                "agent_id": result_agent_id or task_use_block.get("id", ""),
                "agent_type": agent_type,
            }
        )

        events.append(
            {
                "type": "agent_stop",
                "timestamp": timestamp,
                "agent_id": result_agent_id or task_use_block.get("id", ""),
                "agent_type": agent_type,
                "agent_transcript_path": agent_transcript_path,
            }
        )

        return events
