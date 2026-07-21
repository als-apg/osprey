"""Operator Mode session management using Claude Agent SDK.

Provides OperatorSession (single SDK-backed conversation) and OperatorRegistry
(multi-session manager with cleanup) for the OSPREY Web Terminal operator mode.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.interfaces.web_terminal.chat_session_pool import ChatSessionPool
from osprey.interfaces.web_terminal.env_utils import build_base_child_env
from osprey.interfaces.web_terminal.sdk_context import build_system_prompt
from osprey.utils.config import get_facility_timezone

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

# Bound (seconds) for draining an interrupted turn toward its terminal message
# before the reader is hard-cancelled. Enforced inside OperatorSession.cancel().
_QUIESCE_TIMEOUT_S = 5.0


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

    Layers the SDK-specific keys on top of :func:`build_base_child_env` (which
    strips ``CLAUDECODE``/``CLAUDE_CODE_*`` variables while preserving the
    telemetry master switch, resolves the auth-token conflict, and augments
    ``PATH``): auto-sets ``OSPREY_CONFIG`` from the project directory.

    Args:
        project_cwd: Optional project directory. When ``OSPREY_CONFIG`` is not
            already set and this directory contains ``config.yml``, the variable
            is set automatically so hooks can locate the configuration.
    """
    env = build_base_child_env()

    # Auto-set OSPREY_CONFIG when a config.yml exists in the project directory
    if "OSPREY_CONFIG" not in env and project_cwd:
        config_path = Path(project_cwd) / "config.yml"
        if config_path.exists():
            env["OSPREY_CONFIG"] = str(config_path)

    # Note: OSPREY_HOOK_DEBUG is intentionally NOT propagated here.
    # Hooks read config.yml directly for hot-reloadable debug toggle.

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


class TurnInProgressError(RuntimeError):
    """Raised by :meth:`OperatorSession.acquire_turn` when a turn is already active.

    Routes map this to HTTP 409 (a second prompt arrived while one is in flight).
    """


class TurnSilenceTimeout(RuntimeError):
    """Raised by :meth:`OperatorSession.run_turn` when the SDK goes silent past
    the turn deadline. Non-terminal: the turn is quiesced before this propagates.
    """


def is_terminal_event(event: dict[str, Any]) -> bool:
    """Return True if *event* ends a turn's event stream.

    Terminal = a ``result`` event, or an ``error`` other than the in-stream
    ``AssistantMessageError`` (which the SDK follows with more events).
    """
    etype = event.get("type")
    if etype == "result":
        return True
    if etype == "error" and event.get("error_type") != "AssistantMessageError":
        return True
    return False


class OperatorSession:
    """Wraps a ``ClaudeSDKClient`` for operator-mode conversation."""

    def __init__(self, cwd: str, env: dict[str, str] | None = None) -> None:
        self._cwd = cwd
        self._env = env
        self._client: ClaudeSDKClient | None = None
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._response_task: asyncio.Task | None = None
        self._quiesce_task: asyncio.Task | None = None
        self._started = False
        # Per-turn one-in-flight guard. ``_turn_epoch`` only ever increments and
        # names each turn ever started; ``_active_token`` holds the current
        # turn's epoch or None when idle. Manipulated only by the synchronous,
        # lock-free acquire_turn/release_turn pair — no awaits, so it is safe to
        # touch outside the registry lock (handler finally + teardown paths).
        self._turn_epoch = 0
        self._active_token: int | None = None
        # Monotonic timestamp of the last turn boundary. Initialized at
        # creation and re-stamped when a turn's reader completes; the chat
        # registry (issue: chat-registry) uses it as its idle predicate.
        self.last_activity = time.monotonic()

    # ---- Per-turn one-in-flight guard (synchronous, lock-free) ---- #

    def acquire_turn(self) -> int:
        """Mint an epoch token for a new turn.

        Raises :class:`TurnInProgressError` when a turn is already active (the
        route maps that to HTTP 409). Synchronous and lock-free — no awaits.
        """
        if self._active_token is not None:
            raise TurnInProgressError("a turn is already in flight for this session")
        self._turn_epoch += 1
        self._active_token = self._turn_epoch
        return self._active_token

    def release_turn(self, token: int) -> bool:
        """Owner-checked epoch compare-and-clear.

        Clears the active turn only when ``token`` matches the currently held
        epoch, so a stale token (from a turn that already ended and was
        replaced) is a no-op. Idempotent: a second release of the same token
        does nothing. Synchronous and lock-free — safe to call from a handler
        ``finally`` or teardown path outside the registry lock.

        Returns ``True`` iff this call cleared the active turn.
        """
        if self._active_token is not None and self._active_token == token:
            self._active_token = None
            return True
        return False

    @property
    def in_flight(self) -> bool:
        """True while a turn's epoch token is held (used by idle/eviction logic)."""
        return self._active_token is not None

    @property
    def is_busy(self) -> bool:
        """True while the session is genuinely working.

        Busy = the per-turn guard is held AND its reader or a quiesce task is
        still running. A *zombie* — guard held but both the reader and quiesce
        already done — is NOT busy, so pools may evict/reap it.
        """
        if not self.in_flight:
            return False
        handler = self._response_task
        quiesce = self._quiesce_task
        handler_running = handler is not None and not handler.done()
        quiesce_running = quiesce is not None and not quiesce.done()
        return handler_running or quiesce_running

    async def start(self) -> None:
        """Create and connect the SDK client."""
        if not CLAUDE_SDK_AVAILABLE:
            raise RuntimeError("claude-agent-sdk is not installed")

        # Warn about missing OSPREY project files
        for warning in validate_project_directory(self._cwd):
            logger.warning("Operator session: %s (cwd=%s)", warning, self._cwd)

        # Force a known session UUID and hand it to the workspace
        # provenance_locator tool via env, so a filed issue can point back to
        # this session's telemetry. The same value is forced onto the SDK
        # session (session_id below) so the OTEL emitter's session.id matches
        # what the locator returns. env is a build_clean_env dict in production
        # (which strips the harness's own CLAUDE_CODE_* id); inject into a copy.
        telemetry_session_id = str(uuid.uuid4())
        session_env = dict(self._env) if self._env is not None else None
        if session_env is not None:
            session_env["OSPREY_TELEMETRY_SESSION_ID"] = telemetry_session_id
            session_env["OSPREY_TELEMETRY_SESSION_START"] = datetime.now(UTC).isoformat()

        options = ClaudeAgentOptions(
            system_prompt=build_system_prompt(get_facility_timezone()),
            cwd=self._cwd,
            env=session_env,
            setting_sources=["project"],
            session_id=telemetry_session_id,
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
        finally:
            # Re-stamp at turn completion (normal, error, or cancellation).
            self.last_activity = time.monotonic()

    async def run_turn(
        self,
        prompt: str,
        token: int,
        *,
        timeout_s: float,
        heartbeat_s: float = 15.0,
    ):
        """Run one turn as an async generator: the sole queue consumer.

        Sends *prompt*, then yields every SDK event for the turn. While no event
        arrives for ``heartbeat_s``, a ``{"type": "heartbeat"}`` marker is
        yielded so a streaming consumer can keep its transport alive (buffered
        consumers just skip it — the marker never carries payload). ``keepalive``
        queue events are consumed silently. The deadline is SDK-silence based:
        it resets on every event; a lapse of ``timeout_s`` without any event
        raises :class:`TurnSilenceTimeout`.

        Owns the turn from just after :meth:`acquire_turn` (done by the caller,
        which needs the 409 mapping before a response starts) through release:

        * terminal event (see :func:`is_terminal_event`) — the generator returns
          and releases the guard only;
        * any other exit (silence timeout, consumer abandon/``GeneratorExit``,
          cancellation, error) — a detached quiesce is spawned FIRST, then the
          guard is released in a nested ``finally``.

        Callers must guard the never-started case (a generator that is closed
        before its first ``__anext__`` never runs this body): after consuming,
        ``if session.release_turn(token): session.spawn_quiesce()`` — both calls
        are synchronous, so the pair cannot be interleaved by the event loop.
        """
        terminal_seen = False
        loop = asyncio.get_running_loop()
        try:
            await self.send_prompt(prompt)

            deadline = loop.time() + timeout_s
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TurnSilenceTimeout("no SDK event before the turn deadline")
                try:
                    event = await asyncio.wait_for(
                        self._queue.get(), timeout=min(heartbeat_s, remaining)
                    )
                except TimeoutError:
                    if loop.time() >= deadline:
                        raise TurnSilenceTimeout("no SDK event before the turn deadline") from None
                    yield {"type": "heartbeat"}
                    continue

                deadline = loop.time() + timeout_s
                if event.get("type") == "keepalive":
                    continue

                # Flag BEFORE yielding: a consumer that stops right after the
                # terminal event (GeneratorExit at this yield) must still take
                # the release-only path, not a spurious quiesce.
                if is_terminal_event(event):
                    terminal_seen = True
                yield event
                if terminal_seen:
                    return
        finally:
            if terminal_seen:
                self.release_turn(token)
            else:
                # Spawn the detached quiesce FIRST, then release in a nested
                # finally so release still happens if spawn_quiesce is hit by a
                # re-delivered cancellation. Never await the quiesce here.
                try:
                    self.spawn_quiesce()
                finally:
                    self.release_turn(token)

    async def interrupt(self) -> None:
        """Signal-only interrupt: forward ``client.interrupt()`` if connected.

        Never drains the reader and never touches the turn guard — the consumer
        running the turn owns quiesce and release.
        """
        if self._client is not None:
            await self._client.interrupt()

    async def cancel(self) -> None:
        """Interrupt the in-flight turn and quiesce the reader.

        No-op short-circuit when there is no in-flight turn — ``_response_task``
        is ``None`` or already done — so an idle cancel never hangs. Otherwise:

        1. ``await self._client.interrupt()`` FIRST so the CLI stops generating
           (the previous implementation discarded this coroutine, letting the
           CLI keep running).
        2. Drain toward the interrupt's terminal message, bounded at
           ``_QUIESCE_TIMEOUT_S`` seconds via ``asyncio.wait_for``.
        3. Hard-cancel the reader regardless of whether the drain completed.
        """
        task = self._response_task
        if task is None or task.done():
            return

        # 1. Interrupt the SDK client first so the CLI stops generating.
        if self._client is not None:
            try:
                await self._client.interrupt()
            except Exception:
                pass

        # 2. Drain toward the interrupt's terminal message, bounded. Shield so
        #    the timeout cancels only our wait, not the reader itself — the
        #    hard-cancel below owns that.
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_QUIESCE_TIMEOUT_S)
        except TimeoutError:
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

        # 3. Hard-cancel the reader regardless of the drain's outcome.
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    def spawn_quiesce(self) -> asyncio.Task:
        """Spawn a detached task that quiesces the in-flight turn.

        Returns the created task (also stored on ``self._quiesce_task``). The
        5 s drain bound is enforced inside :meth:`cancel`, which the task
        awaits, so callers may fire-and-forget or await the returned task.
        """
        task = asyncio.create_task(self.cancel())
        self._quiesce_task = task
        return task

    async def teardown(self) -> None:
        """Await any in-flight quiesce (≤ the drain bound), then :meth:`stop`.

        ``stop()`` already interrupts, drains, and hard-cancels the reader; this
        additionally awaits a quiesce a consumer may have already spawned, so a
        busy session is quiesced before its client is closed.
        """
        quiesce = self._quiesce_task
        if quiesce is not None and not quiesce.done():
            try:
                await asyncio.wait_for(asyncio.shield(quiesce), timeout=_QUIESCE_TIMEOUT_S)
            except TimeoutError:
                pass
            except Exception:
                pass
        await self.stop()

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
    """Manages operator sessions and composes the Simple-mode chat pool.

    * ``_sessions`` — persistent ``/ws/operator`` sessions keyed by a
      connection-derived id. Uncapped, single-writer per key.
    * ``chats`` — a :class:`ChatSessionPool` of Simple-mode chat sessions
      (LRU-capped, idle-reaped). The ``*_chat_session`` methods below are a
      thin facade over it for route/app callers.
    """

    def __init__(self, chat_max_sessions: int = 5, chat_idle_seconds: float = 900.0) -> None:
        self._sessions: dict[str, OperatorSession] = {}
        # The factory resolves OperatorSession by name at call time, so tests
        # patching this module's OperatorSession still intercept creation.
        self.chats = ChatSessionPool(
            factory=lambda cwd, env: OperatorSession(cwd=cwd, env=env),
            max_sessions=chat_max_sessions,
            idle_seconds=chat_idle_seconds,
        )

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
        """Tear down every operator and chat session concurrently (shutdown)."""
        op_ids = list(self._sessions)
        await asyncio.gather(
            *(self._terminate_session_internal(sid) for sid in op_ids),
            self.chats.drain_all(),
        )

    async def _terminate_session_internal(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.stop()

    # ---- Chat pool facade (Simple-mode; see ChatSessionPool) ---- #

    async def get_or_create_chat_session(
        self, chat_id: str, cwd: str, env: dict[str, str] | None = None
    ) -> tuple[OperatorSession, bool]:
        return await self.chats.get_or_create(chat_id, cwd, env)

    def get_chat_session(self, chat_id: str) -> OperatorSession | None:
        return self.chats.get(chat_id)

    async def terminate_chat_session(self, chat_id: str) -> None:
        await self.chats.terminate(chat_id)

    async def reap_idle_chat_sessions(self) -> int:
        return await self.chats.reap_idle()
