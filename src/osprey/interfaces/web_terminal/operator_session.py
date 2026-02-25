"""Operator Mode session management using Claude Agent SDK.

Provides OperatorSession (single SDK-backed conversation) and OperatorRegistry
(multi-session manager with cleanup) for the OSPREY Web Terminal operator mode.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ClaudeSDKError,
        CLIConnectionError,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
    )

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    ClaudeAgentOptions = dict  # type: ignore[assignment,misc]
    ClaudeSDKClient = object  # type: ignore[assignment,misc]
    AssistantMessage = object  # type: ignore[assignment,misc]
    ResultMessage = object  # type: ignore[assignment,misc]
    SystemMessage = object  # type: ignore[assignment,misc]
    TextBlock = object  # type: ignore[assignment,misc]
    ThinkingBlock = object  # type: ignore[assignment,misc]
    ToolResultBlock = object  # type: ignore[assignment,misc]
    ToolUseBlock = object  # type: ignore[assignment,misc]
    ClaudeSDKError = Exception  # type: ignore[assignment,misc]
    CLIConnectionError = Exception  # type: ignore[assignment,misc]

# Pattern for MCP tool name prefixes: mcp__<server>__<tool>
_MCP_PREFIX_RE = re.compile(r"^mcp__[^_]+__")


def _format_tool_name(raw: str) -> str:
    """Convert raw tool name to a human-readable display name.

    Strips ``mcp__<server>__`` prefix and title-cases the remainder,
    replacing underscores with spaces.

    Examples:
        >>> _format_tool_name("mcp__osprey__channel_read")
        'Channel Read'
        >>> _format_tool_name("Read")
        'Read'
    """
    name = _MCP_PREFIX_RE.sub("", raw)
    return name.replace("_", " ").title()


def _message_to_events(message: Any) -> list[dict[str, Any]]:
    """Convert a Claude SDK message to a list of structured events.

    Args:
        message: A message from ``client.receive_response()``.

    Returns:
        List of event dicts suitable for JSON serialisation over WebSocket.
    """
    events: list[dict[str, Any]] = []

    if isinstance(message, AssistantMessage):
        # Check for API-level errors on the message itself
        if message.error is not None:
            events.append(
                {
                    "type": "error",
                    "message": f"API error: {message.error}",
                    "error_type": "AssistantMessageError",
                }
            )

        for block in message.content:
            if isinstance(block, TextBlock):
                events.append({"type": "text", "content": block.text})
            elif isinstance(block, ThinkingBlock):
                events.append({"type": "thinking", "content": block.thinking})
            elif isinstance(block, ToolUseBlock):
                events.append(
                    {
                        "type": "tool_use",
                        "tool_name": _format_tool_name(block.name),
                        "tool_name_raw": block.name,
                        "tool_use_id": block.id,
                        "input": block.input,
                    }
                )
            elif isinstance(block, ToolResultBlock):
                events.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": bool(block.is_error),
                    }
                )

    elif isinstance(message, ResultMessage):
        events.append(
            {
                "type": "result",
                "is_error": message.is_error,
                "total_cost_usd": message.total_cost_usd,
                "duration_ms": message.duration_ms,
                "num_turns": message.num_turns,
            }
        )

    elif isinstance(message, SystemMessage):
        events.append({"type": "system", "subtype": message.subtype})

    # StreamEvent and other unknown types are silently ignored.
    return events


def build_clean_env(project_cwd: str | None = None) -> dict[str, str]:
    """Build a clean environment dict for the SDK subprocess.

    Mirrors the logic in ``pty_manager.py`` — strips ``CLAUDECODE``/``CLAUDE_CODE_*``
    variables and resolves auth token conflicts.

    Args:
        project_cwd: Optional project directory. When ``OSPREY_CONFIG`` is not
            already set and this directory contains ``config.yml``, the variable
            is set automatically so hooks can locate the configuration.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDECODE", "CLAUDE_CODE_"))}

    # When token-based auth is configured, strip ANTHROPIC_API_KEY to
    # prevent the "auth conflict" warning.
    if env.get("ANTHROPIC_AUTH_TOKEN"):
        env.pop("ANTHROPIC_API_KEY", None)

    # Auto-set OSPREY_CONFIG when a config.yml exists in the project directory
    if "OSPREY_CONFIG" not in env and project_cwd:
        config_path = Path(project_cwd) / "config.yml"
        if config_path.exists():
            env["OSPREY_CONFIG"] = str(config_path)

    # Propagate hooks.debug from config
    if "OSPREY_HOOK_DEBUG" not in env:
        config_file = env.get("OSPREY_CONFIG") or (
            str(Path(project_cwd) / "config.yml") if project_cwd else ""
        )
        if config_file:
            try:
                import yaml

                cfg_path = Path(config_file)
                if cfg_path.exists():
                    with open(cfg_path) as f:
                        cfg = yaml.safe_load(f) or {}
                    if cfg.get("hooks", {}).get("debug"):
                        env["OSPREY_HOOK_DEBUG"] = "1"
            except Exception:
                logger.warning("Failed to read hooks.debug from %s", config_file, exc_info=True)

    return env


def validate_project_directory(cwd: str) -> list[str]:
    """Check that the project directory contains expected OSPREY files.

    Returns a list of human-readable warning strings for any missing files.
    Does not raise — callers should log the warnings.
    """
    warnings: list[str] = []
    path = Path(cwd)

    expected = [
        (".mcp.json", "MCP server configuration"),
        ("CLAUDE.md", "Claude Code project instructions"),
        (".claude", "Claude Code settings directory"),
        ("config.yml", "OSPREY configuration"),
    ]

    for name, description in expected:
        target = path / name
        if not target.exists():
            warnings.append(f"Missing {description}: {name}")

    return warnings


class OperatorSession:
    """Wraps a ``ClaudeSDKClient`` for operator-mode conversation."""

    def __init__(self, cwd: str, env: dict[str, str] | None = None) -> None:
        self._cwd = cwd
        self._env = env
        self._client: ClaudeSDKClient | None = None
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._response_task: asyncio.Task | None = None
        self._started = False

    async def start(self) -> None:
        """Create and connect the SDK client."""
        if not CLAUDE_SDK_AVAILABLE:
            raise RuntimeError("claude-agent-sdk is not installed")

        # Warn about missing OSPREY project files
        for warning in validate_project_directory(self._cwd):
            logger.warning("Operator session: %s (cwd=%s)", warning, self._cwd)

        options = ClaudeAgentOptions(
            cwd=self._cwd,
            env=self._env,
            setting_sources=["project"],
        )
        self._client = ClaudeSDKClient(options=options)
        await self._client.__aenter__()
        self._started = True
        logger.info("OperatorSession started (cwd=%s)", self._cwd)

    async def send_prompt(self, prompt: str) -> None:
        """Send a prompt and start streaming the response into the queue."""
        if self._client is None:
            raise RuntimeError("Session not started")

        await self._client.query(prompt)
        self._response_task = asyncio.create_task(self._stream_response())

    async def _stream_response(self) -> None:
        """Iterate ``receive_response()`` and push events to the queue."""
        try:
            async for message in self._client.receive_response():
                for event in _message_to_events(message):
                    await self._queue.put(event)
        except (ClaudeSDKError, CLIConnectionError) as exc:
            await self._queue.put(
                {
                    "type": "error",
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._queue.put(
                {
                    "type": "error",
                    "message": f"Unexpected error: {exc}",
                    "error_type": type(exc).__name__,
                }
            )

    async def cancel(self) -> None:
        """Interrupt the current response."""
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass

        if self._client is not None:
            try:
                self._client.interrupt()
            except Exception:
                pass

    async def stop(self) -> None:
        """Disconnect the SDK client and cancel any in-flight response."""
        await self.cancel()

        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None

        self._started = False
        logger.info("OperatorSession stopped")

    @property
    def is_active(self) -> bool:
        return self._started and self._client is not None


class OperatorRegistry:
    """Manages multiple operator sessions keyed by session ID."""

    def __init__(self) -> None:
        self._sessions: dict[str, OperatorSession] = {}

    async def create_session(
        self, session_id: str, cwd: str, env: dict[str, str] | None = None
    ) -> OperatorSession:
        """Create and start a new operator session, replacing any existing one."""
        if session_id in self._sessions:
            await self._terminate_session_internal(session_id)

        session = OperatorSession(cwd=cwd, env=env)
        await session.start()
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> OperatorSession | None:
        return self._sessions.get(session_id)

    async def terminate_session(self, session_id: str) -> None:
        await self._terminate_session_internal(session_id)

    async def terminate_session_if_owner(self, session_id: str, owner: OperatorSession) -> None:
        """Terminate only if the caller still owns the session.

        Prevents a stale WebSocket's cleanup from killing a newer session
        that replaced it (e.g. on page reload or reconnection).
        """
        current = self._sessions.get(session_id)
        if current is owner:
            await self._terminate_session_internal(session_id)
        elif owner is not None:
            await owner.stop()

    async def cleanup_all(self) -> None:
        for session_id in list(self._sessions):
            await self._terminate_session_internal(session_id)

    async def _terminate_session_internal(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.stop()
