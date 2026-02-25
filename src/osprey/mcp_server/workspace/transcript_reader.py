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
        task_meta: list[dict] = []

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
                    if include_subagents:
                        self._collect_task_meta(task_meta, block, content, ts)
                    else:
                        events.extend(self._make_task_events(block, content, is_err, ts))

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
                        if include_subagents:
                            self._collect_task_meta(task_meta, tu_block, content, ts)
                        else:
                            events.extend(self._make_task_events(tu_block, content, is_err, ts))

        # In-flight Tasks (tool_use seen but no result yet) still carry
        # subagent_type in their input, so include them in task_meta so the
        # agent shows its proper name while the subagent is still running.
        for _tid, (block, ts) in task_uses.items():
            task_input = block.get("input", {})
            agent_type = task_input.get("subagent_type", task_input.get("description", ""))
            task_meta.append(
                {
                    "timestamp": ts,
                    "agent_type": agent_type,
                    "tool_use_id": block.get("id", ""),
                    "prompt": task_input.get("prompt", ""),
                }
            )

        # Read subagent transcripts and create lifecycle events with matching IDs
        if include_subagents:
            matched_task_indices: set[int] = set()
            session_stem = path.stem
            subagent_dir = path.parent / session_stem / "subagents"

            if subagent_dir and subagent_dir.is_dir():
                for agent_file in sorted(subagent_dir.glob("*.jsonl")):
                    agent_fname = agent_file.stem
                    sub_events = self.read_session(
                        agent_file,
                        include_subagents=False,
                        agent_id=agent_fname,
                    )

                    sub_prompt = self._read_first_user_text(agent_file)
                    agent_type = self._match_subagent_to_task(
                        agent_fname,
                        sub_events,
                        task_meta,
                        matched_task_indices,
                        subagent_prompt=sub_prompt,
                    )
                    start_ts = sub_events[0].get("timestamp", "") if sub_events else ""
                    stop_ts = sub_events[-1].get("timestamp", "") if sub_events else ""

                    events.append(
                        {
                            "type": "agent_start",
                            "timestamp": start_ts,
                            "agent_id": agent_fname,
                            "agent_type": agent_type,
                        }
                    )
                    events.extend(sub_events)
                    events.append(
                        {
                            "type": "agent_stop",
                            "timestamp": stop_ts,
                            "agent_id": agent_fname,
                            "agent_type": agent_type,
                        }
                    )

            # Create lifecycle events for tasks with no subagent transcript
            for i, meta in enumerate(task_meta):
                if i not in matched_task_indices:
                    fallback_id = meta.get("result_agent_id") or meta["tool_use_id"]
                    events.append(
                        {
                            "type": "agent_start",
                            "timestamp": meta["timestamp"],
                            "agent_id": fallback_id,
                            "agent_type": meta["agent_type"],
                        }
                    )
                    events.append(
                        {
                            "type": "agent_stop",
                            "timestamp": meta["timestamp"],
                            "agent_id": fallback_id,
                            "agent_type": meta["agent_type"],
                        }
                    )

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

    def find_transcript_by_id(self, session_id: str) -> Path | None:
        """Find the transcript file for a specific session ID.

        Claude Code uses session UUIDs as filenames, so this constructs
        the expected path directly without scanning.

        Returns:
            Path to the ``.jsonl`` file, or None if not found.
        """
        transcript_dir = self.find_transcript_dir()
        if not transcript_dir:
            return None
        path = transcript_dir / f"{session_id}.jsonl"
        if path.is_file():
            return path
        return None

    def read_session_by_id(self, session_id: str, **kwargs) -> list[dict]:
        """Read a specific session's events by session ID.

        Args:
            session_id: The Claude Code session UUID.
            **kwargs: Passed through to :meth:`read_session`
                (e.g. ``include_subagents``).

        Returns:
            List of event dicts, or empty list if session not found.
        """
        path = self.find_transcript_by_id(session_id)
        if not path:
            return []
        return self.read_session(path, **kwargs)

    def read_chat_history_by_id(self, session_id: str) -> list[dict]:
        """Read chat history for a specific session by ID.

        Returns:
            List of conversation turn dicts, or empty list if not found.
        """
        path = self.find_transcript_by_id(session_id)
        if not path:
            return []
        return self.read_chat_history(path)

    def read_agent_timeline(self, agent_id: str, *, session_id: str | None = None) -> list[dict]:
        """Read the full internal timeline of a subagent.

        Extracts every reasoning step, tool call, and result from the
        subagent's transcript — everything the agent "said to itself".

        Args:
            agent_id: The subagent filename stem.
            session_id: If provided, look up the parent transcript by
                session ID instead of using the most recent transcript.

        Returns a chronological list of timeline entries with ``kind``
        values: ``prompt``, ``reasoning``, ``tool_call``, ``tool_result``.
        """
        if session_id:
            transcript = self.find_transcript_by_id(session_id)
        else:
            transcript = self.find_current_transcript()
        if not transcript:
            return []

        subagent_dir = transcript.parent / transcript.stem / "subagents"
        agent_file = subagent_dir / f"{agent_id}.jsonl"
        if not agent_file.is_file():
            return []

        return self._parse_agent_timeline(agent_file)

    @staticmethod
    def _parse_agent_timeline(path: Path) -> list[dict]:
        """Parse a subagent transcript into a flat timeline."""
        timeline: list[dict] = []
        pending_tools: dict[str, dict] = {}
        is_first_user = True

        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            etype = entry.get("type")
            timestamp = entry.get("timestamp", "")

            if etype == "assistant":
                msg = entry.get("message", {})
                content = msg.get("content", [])
                if isinstance(content, str):
                    timeline.append(
                        {
                            "kind": "reasoning",
                            "timestamp": timestamp,
                            "text": content,
                        }
                    )
                    continue
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    bt = block.get("type", "")
                    if bt == "text":
                        text = block.get("text", "").strip()
                        if text:
                            timeline.append(
                                {
                                    "kind": "reasoning",
                                    "timestamp": timestamp,
                                    "text": text,
                                }
                            )
                    elif bt == "thinking":
                        text = block.get("thinking", "").strip()
                        if text:
                            timeline.append(
                                {
                                    "kind": "thinking",
                                    "timestamp": timestamp,
                                    "text": text,
                                }
                            )
                    elif bt == "tool_use":
                        tool_id = block.get("id", "")
                        tool_entry = {
                            "kind": "tool_call",
                            "timestamp": timestamp,
                            "tool": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        }
                        timeline.append(tool_entry)
                        if tool_id:
                            pending_tools[tool_id] = tool_entry

            elif etype == "user":
                msg = entry.get("message", {})
                content = msg.get("content", "")

                if is_first_user:
                    is_first_user = False
                    text = ""
                    if isinstance(content, str):
                        text = content.strip()
                    elif isinstance(content, list):
                        parts = []
                        for block in content:
                            if isinstance(block, str):
                                parts.append(block)
                            elif isinstance(block, dict) and block.get("type") == "text":
                                parts.append(block.get("text", ""))
                        text = "\n".join(parts).strip()
                    if text:
                        timeline.append(
                            {
                                "kind": "prompt",
                                "timestamp": timestamp,
                                "text": text,
                            }
                        )
                    # First user entry may also contain tool_results
                    # (fall through to process them)

                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_result":
                            continue
                        tool_use_id = block.get("tool_use_id", "")
                        result_content = block.get("content", "")
                        is_error = block.get("is_error", False)
                        result_text = ""
                        if isinstance(result_content, str):
                            result_text = result_content
                        elif isinstance(result_content, list):
                            result_text = _extract_text(result_content)

                        timeline.append(
                            {
                                "kind": "tool_result",
                                "timestamp": timestamp,
                                "tool_use_id": tool_use_id,
                                "text": result_text[:2000],
                                "is_error": is_error,
                            }
                        )

            elif etype == "tool_result":
                tool_use_id = entry.get("tool_use_id", "")
                result_content = entry.get("content", "")
                is_error = entry.get("is_error", False)
                result_text = ""
                if isinstance(result_content, str):
                    result_text = result_content
                elif isinstance(result_content, list):
                    result_text = _extract_text(result_content)

                timeline.append(
                    {
                        "kind": "tool_result",
                        "timestamp": timestamp,
                        "tool_use_id": tool_use_id,
                        "text": result_text[:2000],
                        "is_error": is_error,
                    }
                )

        return timeline

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
                    turns.append({"role": "assistant", "content": text, "timestamp": timestamp})

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
        if isinstance(content_blocks, str):
            return content_blocks.strip()
        parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()

    @staticmethod
    def _collect_task_meta(
        task_meta: list[dict],
        task_use_block: dict,
        content: str | list,
        timestamp: str,
    ) -> None:
        """Store Task metadata for deferred lifecycle event creation."""
        task_input = task_use_block.get("input", {})
        agent_type = task_input.get("subagent_type", task_input.get("description", ""))
        meta: dict = {
            "timestamp": timestamp,
            "agent_type": agent_type,
            "tool_use_id": task_use_block.get("id", ""),
            "prompt": task_input.get("prompt", ""),
        }

        text = content
        if isinstance(content, list):
            text = _extract_text(content)
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    if parsed.get("agentId"):
                        meta["result_agent_id"] = parsed["agentId"]
                    if parsed.get("transcriptPath"):
                        meta["transcript_path"] = parsed["transcriptPath"]
            except (json.JSONDecodeError, ValueError):
                pass

        task_meta.append(meta)

    @staticmethod
    def _read_first_user_text(path: Path) -> str:
        """Read the first user message from a transcript (the Task prompt)."""
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("type") != "user":
                    continue
                msg = entry.get("message", {})
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts: list[str] = []
                    for block in content:
                        if isinstance(block, str):
                            parts.append(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                    return "\n".join(parts).strip()
                return ""
        except Exception:
            return ""

    @staticmethod
    def _match_subagent_to_task(
        agent_fname: str,
        sub_events: list[dict],
        task_meta: list[dict],
        matched: set[int],
        *,
        subagent_prompt: str = "",
    ) -> str:
        """Match a subagent transcript to a Task metadata entry.

        Matching strategies (in priority order):
        1. Prompt comparison — the Task's prompt field should be the subagent's
           first user message.  This is deterministic and handles concurrent agents.
        2. Explicit result_agent_id / transcript_path from Task result JSON.
        3. Chronological order fallback.
        """
        if subagent_prompt:
            for i, meta in enumerate(task_meta):
                if i in matched:
                    continue
                task_prompt = meta.get("prompt", "")
                if task_prompt and task_prompt.strip() == subagent_prompt:
                    matched.add(i)
                    return meta["agent_type"]

        for i, meta in enumerate(task_meta):
            if i in matched:
                continue
            if meta.get("result_agent_id") == agent_fname:
                matched.add(i)
                return meta["agent_type"]

        for i, meta in enumerate(task_meta):
            if i in matched:
                continue
            tp = meta.get("transcript_path", "")
            if tp and agent_fname in tp:
                matched.add(i)
                return meta["agent_type"]

        for i, meta in enumerate(task_meta):
            if i not in matched:
                matched.add(i)
                return meta["agent_type"]

        return ""

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
            "tool_name": short_name,
            "full_tool_name": full_name,
            "server": server,
            "server_name": server,
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
