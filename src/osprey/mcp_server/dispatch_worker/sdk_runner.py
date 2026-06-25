"""SDK runner for headless dispatch via claude_agent_sdk.

Wraps claude_agent_sdk.query() to execute agent prompts and return
structured results for use by the dispatch API endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger("osprey.mcp_server.dispatch_worker.sdk_runner")

# SDK imports -- guard so module loads even when SDK is not installed
# (SDK is only available inside the worker container).
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        query,
    )

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

_DEFAULT_PROJECT_DIR = "/app/project"

# Bounds on per-run captured output. A run's text/tool output is held in memory,
# persisted to JSON, and proxied to the dashboard, so an adversarial or runaway
# trigger could otherwise balloon RAM and disk. Oversized payloads are truncated
# with a marker rather than dropped.
_MAX_TEXT_OUTPUT = 256 * 1024  # total concatenated assistant text
_MAX_TOOL_RESULT = 16 * 1024  # per tool-result body
_MAX_TOOL_CALLS = 200  # number of tool calls retained

# Env-var name hints whose VALUES must be scrubbed from any text we persist or
# return (provider auth tokens, dispatch bearer tokens, etc.).
_SECRET_ENV_NAME_HINTS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY")
_MIN_SECRET_LEN = 12

# Inactivity watchdog: max seconds to wait for the *next* SDK message before
# treating the run as silently hung. A healthy run emits an init SystemMessage
# within seconds of CLI startup and then streams text/tool events continuously,
# so each message resets this clock — only a true stall trips it. The classic
# trigger is a bad/expired provider credential or an unreachable base URL: the
# bundled CLI does not fast-fail on those, it simply produces no output. Without
# this guard such a run stalls all the way to the dispatch worker's outer
# DISPATCH_TIMEOUT_SEC (default 300s) and surfaces only a generic "timed out"
# error that masks the auth cause. Generous default so legitimately slow single
# turns are not cut off; raise via DISPATCH_INACTIVITY_SEC for slow providers.
_INACTIVITY_TIMEOUT_SEC = float(os.environ.get("DISPATCH_INACTIVITY_SEC", "120"))


def _secret_values() -> list[str]:
    """Collect non-trivial secret env values to scrub from run output."""
    vals: set[str] = set()
    for name, val in os.environ.items():
        if (
            val
            and len(val) >= _MIN_SECRET_LEN
            and any(hint in name.upper() for hint in _SECRET_ENV_NAME_HINTS)
        ):
            vals.add(val)
    # Longest-first so a value that contains another is masked whole.
    return sorted(vals, key=len, reverse=True)


def _scrub(text: str | None, secrets: list[str]) -> str | None:
    """Replace known secret values in ``text`` with ``***`` (best-effort)."""
    if not text:
        return text
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, "***")
    return text


def _cap_text(text: str) -> str:
    """Truncate concatenated assistant text to _MAX_TEXT_OUTPUT with a marker."""
    if len(text) > _MAX_TEXT_OUTPUT:
        dropped = len(text) - _MAX_TEXT_OUTPUT
        return text[:_MAX_TEXT_OUTPUT] + f"\n…[truncated {dropped} chars]"
    return text


async def run_dispatch(
    prompt: str,
    allowed_tools: list[str],
    max_turns: int = 25,
    event_queue: asyncio.Queue | None = None,
    denied_tools: Iterable[str] = (),
) -> dict[str, Any]:
    """Run a prompt headlessly via the Claude Agent SDK.

    Args:
        prompt: The prompt to send to the agent.
        allowed_tools: List of tool names the agent may use.
        max_turns: Maximum number of agentic turns (default 25).
        denied_tools: Hard denylist enforced at the permission layer regardless
            of ``allowed_tools`` (defense-in-depth; the worker threads its
            ``DENIED_TOOLS`` here). Entries ending in ``*`` match by prefix.

    Returns:
        dict with keys:
            status: "completed" or "error"
            text_output: concatenated assistant text blocks
            tool_calls: list of {name, input, result} dicts
            error: error message string or None
            duration_sec: wall-clock seconds
            cost_usd: API cost (from ResultMessage, if available)
            num_turns: agentic turn count (from ResultMessage, if available)
    """
    if not HAS_SDK:
        return {
            "status": "error",
            "text_output": "",
            "tool_calls": [],
            "error": "claude_agent_sdk is not installed",
            "duration_sec": 0.0,
            "cost_usd": None,
            "num_turns": None,
        }

    project_dir = os.environ.get("OSPREY_PROJECT_DIR", _DEFAULT_PROJECT_DIR)
    stderr_lines: list[str] = []

    # Build env the same way the OSPREY web server does for operator sessions:
    # build_clean_env() strips CLAUDECODE/CLAUDE_CODE_* vars and resolves auth
    # conflicts.  Provider env (auth token, base URL, model tier IDs) is already
    # injected into os.environ by _inject_provider_env_once() at worker startup.
    from osprey.interfaces.web_terminal.operator_session import build_clean_env
    from osprey.interfaces.web_terminal.sdk_context import (
        build_system_prompt,
        make_tool_allowlist,
    )
    from osprey.utils.config import get_facility_timezone

    sdk_env = build_clean_env(project_cwd=project_dir)

    # Point OSPREY config resolution at the project explicitly. The worker
    # process CWD is the image WORKDIR (``/app`` in the container), not the
    # project dir, and ``osprey.utils.config`` falls back to ``CWD/config.yml``
    # when ``CONFIG_FILE`` is unset — so without this, the spawned agent and its
    # hook subprocesses (which ``ClaudeAgentOptions(cwd=...)`` does not relocate)
    # error with "No config.yml found in current directory" on every dispatch.
    sdk_env["CONFIG_FILE"] = os.path.join(project_dir, "config.yml")

    # The container sets CLAUDE_CONFIG_DIR=/data/claude-config (root-owned, used
    # by osprey-web).  The dispatch user can't write there, and the CLI hangs on
    # startup if it can't write session data.  Override to dispatch user's home.
    dispatch_home = os.environ.get("HOME", "/home/dispatch")
    sdk_env["CLAUDE_CONFIG_DIR"] = os.path.join(dispatch_home, ".claude")

    # NOTE: do NOT use permission_mode="bypassPermissions" — the CLI short-circuits
    # can_use_tool under bypass, so the allowlist would not be enforced. With the
    # default mode and can_use_tool set, the SDK auto-configures
    # permission_prompt_tool_name="stdio" (see client.py:122) which routes every
    # tool permission check to our allowlist callback.
    options = ClaudeAgentOptions(
        allowed_tools=allowed_tools,
        system_prompt=build_system_prompt(get_facility_timezone()),
        can_use_tool=make_tool_allowlist(allowed_tools, denied_tools),
        cwd=project_dir,
        env=sdk_env,
        max_turns=max_turns,
        stderr=lambda line: stderr_lines.append(line),
        setting_sources=["project"],
    )

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    # Map tool_use_id -> index in tool_calls for matching results to calls
    pending_tools: dict[str, int] = {}
    cost_usd: float | None = None
    num_turns: int | None = None
    # Snapshot secret values once so we can scrub them from anything we persist
    # or return (the SDK env carries provider/auth tokens).
    secret_values = _secret_values()

    def _finalize(result: dict[str, Any]) -> dict[str, Any]:
        """Scrub secrets and cap oversized fields before persist/return."""
        result["text_output"] = _cap_text(
            _scrub(result.get("text_output") or "", secret_values) or ""
        )
        if result.get("error"):
            result["error"] = _scrub(result["error"], secret_values)
        if result.get("stderr"):
            result["stderr"] = _scrub(result["stderr"], secret_values)
        for tc in result.get("tool_calls") or []:
            if tc.get("result"):
                tc["result"] = _scrub(tc["result"], secret_values)
        return result

    async def _push(event: dict[str, Any]) -> None:
        if event_queue is not None:
            await event_queue.put(event)

    # can_use_tool requires streaming-mode input (AsyncIterable), not str.
    async def _prompt_stream():
        yield {"type": "user", "message": {"role": "user", "content": prompt}}

    t0 = time.monotonic()
    agen = query(prompt=_prompt_stream(), options=options)
    try:
        # Drive the generator manually (rather than ``async for``) so each
        # ``__anext__`` is bounded by the inactivity watchdog. A full-window
        # silence means the provider never responded — fail fast with a clear
        # message instead of stalling to the worker's outer DISPATCH_TIMEOUT_SEC.
        while True:
            try:
                message = await asyncio.wait_for(agen.__anext__(), timeout=_INACTIVITY_TIMEOUT_SEC)
            except StopAsyncIteration:
                break
            except TimeoutError:
                try:
                    await agen.aclose()
                except Exception:
                    logger.debug("agen.aclose() raised after inactivity timeout", exc_info=True)
                duration_sec = time.monotonic() - t0
                msg = (
                    f"No response from the model provider for "
                    f"{_INACTIVITY_TIMEOUT_SEC:.0f}s — dispatch aborted. This usually "
                    "means an invalid or expired provider credential, or an "
                    "unreachable provider base URL."
                )
                logger.error("Dispatch aborted after %.1fs: %s", duration_sec, msg)
                await _push({"type": "error", "message": msg})
                return _finalize(
                    {
                        "status": "error",
                        "text_output": "".join(text_parts),
                        "tool_calls": tool_calls,
                        "error": msg,
                        "stderr": "\n".join(stderr_lines) if stderr_lines else None,
                        "duration_sec": round(duration_sec, 2),
                        "cost_usd": cost_usd,
                        "num_turns": num_turns,
                    }
                )

            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                        await _push({"type": "text", "content": block.text})
                    elif isinstance(block, ToolUseBlock):
                        # Bound the retained tool-call list; excess calls still
                        # stream as events but are not accumulated in memory.
                        if len(tool_calls) < _MAX_TOOL_CALLS:
                            entry: dict[str, Any] = {
                                "name": block.name,
                                "input": block.input,
                                "result": None,
                            }
                            idx = len(tool_calls)
                            tool_calls.append(entry)
                            pending_tools[block.id] = idx
                        await _push(
                            {"type": "tool_start", "name": block.name, "input": block.input}
                        )
                    elif isinstance(block, ToolResultBlock):
                        idx = pending_tools.get(block.tool_use_id)
                        result_text: str | None = None
                        if idx is not None:
                            content = block.content
                            if isinstance(content, str):
                                result_text = content
                            elif isinstance(content, list):
                                texts = [
                                    item.get("text", "")
                                    for item in content
                                    if isinstance(item, dict) and item.get("type") == "text"
                                ]
                                result_text = "\n".join(texts) if texts else str(content)
                            else:
                                result_text = str(content)
                            if result_text is not None and len(result_text) > _MAX_TOOL_RESULT:
                                dropped = len(result_text) - _MAX_TOOL_RESULT
                                result_text = (
                                    result_text[:_MAX_TOOL_RESULT]
                                    + f"\n…[truncated {dropped} chars]"
                                )
                            tool_calls[idx]["result"] = result_text
                        await _push(
                            {
                                "type": "tool_result",
                                "name": tool_calls[idx]["name"] if idx is not None else None,
                                "result": result_text,
                            }
                        )

            elif isinstance(message, ResultMessage):
                cost_usd = getattr(message, "cost_usd", None)
                num_turns = getattr(message, "num_turns", None)
                await _push({"type": "result", "cost_usd": cost_usd, "num_turns": num_turns})

            elif isinstance(message, SystemMessage):
                logger.debug("SystemMessage: %s", message)

        duration_sec = time.monotonic() - t0
        logger.info(
            "Dispatch completed: %d text blocks, %d tool calls, %.1fs",
            len(text_parts),
            len(tool_calls),
            duration_sec,
        )
        await _push({"type": "done"})
        return _finalize(
            {
                "status": "completed",
                "text_output": "".join(text_parts),
                "tool_calls": tool_calls,
                "error": None,
                "duration_sec": round(duration_sec, 2),
                "cost_usd": cost_usd,
                "num_turns": num_turns,
            }
        )

    except asyncio.CancelledError:
        # Close the SDK async generator so the CLI subprocess exits via its own
        # stdin-close path. Measured end-to-end cancel latency: ~0.6s; a few
        # seconds if a tool call is mid-flight and needs to unwind.
        try:
            await agen.aclose()
        except Exception:
            logger.debug("agen.aclose() raised during cancellation", exc_info=True)
        raise
    except Exception as exc:
        duration_sec = time.monotonic() - t0
        stderr_output = "\n".join(stderr_lines) if stderr_lines else None
        logger.error(
            "Dispatch failed after %.1fs: %s",
            duration_sec,
            exc,
            exc_info=True,
        )
        await _push({"type": "error", "message": _scrub(str(exc), secret_values)})
        return _finalize(
            {
                "status": "error",
                "text_output": "".join(text_parts),
                "tool_calls": tool_calls,
                "error": str(exc),
                "stderr": stderr_output,
                "duration_sec": round(duration_sec, 2),
                "cost_usd": cost_usd,
                "num_turns": num_turns,
            }
        )
