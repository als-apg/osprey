"""SDK runner for headless dispatch via claude_agent_sdk.

Wraps claude_agent_sdk.query() to execute agent prompts and return
structured results for use by the dispatch API endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import time
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from osprey.mcp_server.dispatch_worker import failure_class, run_stats

logger = logging.getLogger("osprey.mcp_server.dispatch_worker.sdk_runner")

# Any base64-ish run this long is not something this module ever intends to
# log — the sole source of such a run here is an inlined image input's bytes
# (content_b64). Redact it defensively so no current or future log statement in
# this module can leak inlined image bytes to a sink. Session/run UUIDs carry
# hyphens (max alnum run ~12 chars), so they are never touched. Hygiene
# invariant — unit-asserted alongside "never in the prompt string / run record".
_B64_RUN_RE = re.compile(r"[A-Za-z0-9+/]{64,}={0,2}")


class _Base64RedactingFilter(logging.Filter):
    """Redact long base64-ish runs from this logger's records (defense-in-depth)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = _B64_RUN_RE.sub("[redacted-b64]", message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


logger.addFilter(_Base64RedactingFilter())

# SDK imports -- guard so module loads even when SDK is not installed
# (SDK is only available inside the worker container).
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        HookMatcher,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
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


# ---------------------------------------------------------------------------
# Input-file prompt assembly
# ---------------------------------------------------------------------------

# Header introducing the per-input-file descriptor block appended to the prompt.
# Generic and mechanism-only: it names no channel and states no policy, so it
# never contradicts a channel-specific system-prompt fragment.
_INPUT_FILES_HEADER = "Files provided with this request:"


def _take_seam(run_id: str | None) -> list[dict[str, Any]]:
    """Pop this run's input-file seam exactly once (``[]`` when there is none).

    Deferred, function-local import of ``take_input_seam``: a top-level import
    would be circular (``dispatch_api`` imports ``sdk_runner``). Pop-once — the
    dispatch task populated the seam before invoking the run, and the task's
    ``finally`` drops anything left behind. A run with no ``run_id`` (unit tests,
    ad-hoc calls) has no seam.
    """
    if not run_id:
        return []
    from osprey.mcp_server.dispatch_worker.dispatch_api import take_input_seam

    return take_input_seam(run_id)


def _load_entry_b64(entry_id: str) -> str | None:
    """Return an ingested artifact's bytes as raw base64, or ``None`` if absent.

    Used to re-inline an ``ingest=True`` image input (its bytes already live in
    the ArtifactStore, so the seam item carries no ``content_b64``) as an image
    content block. The store copy is retained — the same entry stays readable
    later via ``data_read`` — so this only reads, never removes.
    """
    try:
        from osprey.agent_runner.artifact_resolve import get_run_store

        path = get_run_store().get_file_path(entry_id)
        if path is None or not path.exists():
            return None
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        logger.warning("Could not load input artifact %s for inlining", entry_id, exc_info=True)
        return None


def _descriptor_line(item: dict[str, Any], *, inlined: bool) -> str:
    """One mechanism-only descriptor line for an input file.

    An inlined image is marked ``image shown inline [shown_inline]`` — the
    literal ``shown_inline`` token is a consumed marker (a downstream channel
    prompt keys on it). A store-resident file is referenced by the mechanism a
    reader uses to open it: ``data_read("<entry_id>")``. No mention of
    base64/ingest/caps — mechanism only.
    """
    filename = item.get("filename")
    mime = item.get("mime")
    if inlined:
        return f"- {filename} ({mime}) — image shown inline [shown_inline]"
    entry_id = item.get("entry_id")
    if entry_id:
        return f'- {filename} ({mime}) — read with data_read("{entry_id}")'
    # Degenerate: a non-image seam item with neither an entry_id nor a way to
    # inline it. The seam contract makes text-mime inputs store-resident, so this
    # is not expected; emit a bare descriptor rather than a false access hint.
    return f"- {filename} ({mime})"


def _assemble_user_content(
    prompt: str, seam_items: list[dict[str, Any]]
) -> str | list[dict[str, Any]]:
    """Build the user-message ``content`` from the prompt and the input seam.

    Routing by mime: ``image/*`` items become base64 image content blocks in the
    same user message (from the seam's ``content_b64`` when present, else the
    stored bytes via ``entry_id``); every other item is referenced by descriptor
    only. Every input file — inlined or not — contributes one descriptor line to
    a block appended to the prompt text.

    Returns the raw ``prompt`` string when there are no input files (unchanged
    legacy shape), a ``[*image_blocks, {text}]`` list when at least one image is
    inlined, and a ``prompt + descriptor`` string when there are inputs but none
    inline. ``content_b64`` rides only an image block's ``source.data`` — never
    the returned text — so it cannot reach the prompt string or the run record.
    """
    if not seam_items:
        return prompt

    image_blocks: list[dict[str, Any]] = []
    descriptor_lines: list[str] = []
    for item in seam_items:
        mime = item.get("mime") or ""
        inlined = False
        if mime.startswith("image/"):
            data_b64 = item.get("content_b64")
            if data_b64 is None and item.get("entry_id"):
                data_b64 = _load_entry_b64(item["entry_id"])
            if data_b64:
                image_blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": data_b64},
                    }
                )
                inlined = True
        descriptor_lines.append(_descriptor_line(item, inlined=inlined))

    text = f"{prompt}\n\n{_INPUT_FILES_HEADER}\n" + "\n".join(descriptor_lines)
    if not image_blocks:
        return text
    # Image-then-text order verified by the SDK spike; block ordering is free.
    return [*image_blocks, {"type": "text", "text": text}]


async def run_dispatch(
    prompt: str,
    allowed_tools: list[str],
    max_turns: int = 25,
    event_queue: asyncio.Queue | None = None,
    denied_tools: Iterable[str] = (),
    run_id: str | None = None,
    surface_prompt: str | None = None,
    surface_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Run a prompt headlessly via the Claude Agent SDK.

    Args:
        prompt: The prompt to send to the agent.
        allowed_tools: List of tool names the agent may use.
        max_turns: Maximum number of agentic turns (default 25).
        denied_tools: Hard denylist enforced at the permission layer regardless
            of ``allowed_tools`` (defense-in-depth; the worker threads its
            ``DENIED_TOOLS`` here). Entries ending in ``*`` match by prefix.
        run_id: Dispatch run id. Exported to the agent (and the MCP tool
            subprocesses it spawns) as ``OSPREY_DISPATCH_RUN_ID`` so every
            artifact saved during the run is attributed to it.
        surface_prompt: Optional static fragment appended to the system prompt
            (via ``build_system_prompt``'s ``extra``) to describe the surface
            this dispatch was triggered from. Not templated/interpolated per
            run. When ``None`` (default), the system prompt is unchanged.
        surface_tools: Optional keep-list narrowing ``allowed_tools`` down to
            the subset a specific surface may use (via
            ``tool_policy.narrow_allowed_tools``). Can only remove tools from
            the trigger's allow set, never add one, and never touches
            ``denied_tools`` — the deny floor is enforced independently.
            ``None`` or empty is a no-op (``allowed_tools`` used as-is).

    Returns:
        dict with keys:
            status: "completed" or "error"
            text_output: concatenated assistant text blocks
            tool_calls: list of {name, input, result} dicts
            error: error message string or None
            duration_sec: wall-clock seconds
            cost_usd: API cost (from ResultMessage, if available)
            num_turns: agentic turn count (from ResultMessage, if available)
            session_id: the forced telemetry session UUID for this run — the
                value the OTEL emitter tags records with as session.id, so a
                consumer can locate this run's full provenance in the telemetry
                store (None if the SDK was unavailable and no run started).
            failure_class: (error results only) the class this failure was
                stamped with — one of ``failure_class.FAILURE_*``.
            num_tool_calls: (error results only) truthful count of tool calls
                the run made before failing (survives the retained-list cap).
    """
    if not HAS_SDK:
        return failure_class._stamp(
            {
                "status": "error",
                "text_output": "",
                "tool_calls": [],
                "error": "claude_agent_sdk is not installed",
                "duration_sec": 0.0,
                "cost_usd": None,
                "num_turns": None,
                "session_id": None,
            },
            failure_class.FAILURE_INFRASTRUCTURE,
            0,
        )

    project_dir = os.environ.get("OSPREY_PROJECT_DIR", _DEFAULT_PROJECT_DIR)
    stderr_lines: list[str] = []

    # Build env the same way the OSPREY web server does for operator sessions:
    # build_clean_env() strips CLAUDECODE/CLAUDE_CODE_* vars and resolves auth
    # conflicts.  Provider env (auth token, base URL, model tier IDs) is already
    # injected into os.environ by _inject_provider_env_once() at worker startup.
    from osprey.agent_runner.clean_env import build_clean_env
    from osprey.agent_runner.sdk_context import build_system_prompt
    from osprey.mcp_server.dispatch_worker.agent_surfaces import parse_project_agents
    from osprey.mcp_server.dispatch_worker.tool_policy import (
        make_backstop,
        make_pretooluse_hook,
        narrow_allowed_tools,
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

    # Marks this CLI session as a headless dispatch run for the project's own
    # hooks: osprey_approval must not emit explicit allow decisions here (CLI
    # hook aggregation is not deny-dominates, so an allow would override the
    # trigger-allowlist hook below).
    sdk_env["OSPREY_DISPATCH_RUN"] = "1"

    # Attribute every artifact this run saves to the run, so the worker can
    # later report and serve exactly this run's plots. NOT OSPREY_SESSION_ID:
    # that variable also relocates the artifact store into
    # _agent_data/sessions/<id>/ (resolve_agent_data_root), which would move
    # dispatch plots off the shared root the gallery reads.
    if run_id:
        sdk_env["OSPREY_DISPATCH_RUN_ID"] = run_id

    # Force a known session UUID for THIS run and hand it to the workspace
    # provenance_locator tool via env, so a filed issue can point back to this
    # run's telemetry. build_clean_env() strips CLAUDE_CODE_* (so the harness's
    # own session id never reaches the MCP subprocess headless); instead OSPREY
    # owns the id: the same value is forced onto the SDK session below
    # (ClaudeAgentOptions.session_id) so the OTEL emitter tags records with it,
    # making the returned locator resolve. Race-free — fixed per run at spawn,
    # never a shared-directory mtime pick.
    telemetry_session_id = str(uuid.uuid4())
    sdk_env["OSPREY_TELEMETRY_SESSION_ID"] = telemetry_session_id
    sdk_env["OSPREY_TELEMETRY_SESSION_START"] = datetime.now(UTC).isoformat()

    # Declared subagent tool surfaces from the provisioned .claude/agents/ —
    # each subagent is held to exactly its declared tools (web-terminal parity)
    # without the trigger having to enumerate them.
    agent_surfaces = parse_project_agents(project_dir)

    # Narrow the main thread's allow set to this surface's keep-list, if any.
    # Pure removal only — cannot add a tool absent from allowed_tools, and
    # never touches denied_tools (the deny floor below is independent).
    effective_tools = narrow_allowed_tools(allowed_tools, surface_tools)
    logger.info(
        "Dispatch tool policy: %d trigger tools (%d after surface narrowing), subagent surfaces %s",
        len(allowed_tools),
        len(effective_tools),
        {name: (len(s) if s is not None else None) for name, s in agent_surfaces.items()},
    )

    # NOTE: do NOT use permission_mode="bypassPermissions" — the CLI short-circuits
    # can_use_tool under bypass, so the allowlist would not be enforced. With the
    # default mode and can_use_tool set, the SDK auto-configures
    # permission_prompt_tool_name="stdio" (see client.py:122) which routes
    # unresolved permission checks to our backstop callback.
    #
    # The PreToolUse hook is the single authority: unlike can_use_tool — which
    # the CLI never consults for calls already permitted by settings.json
    # permissions.allow rules — the hook fires for EVERY tool call, including
    # inside subagents, so project settings cannot widen the trigger's surface.
    # Exact-name denied tools additionally go to disallowed_tools, which strips
    # them from the model's context entirely; prefix entries (``server__*``)
    # become server-level rules (``server``).
    disallowed = [t for t in denied_tools if not t.endswith("*")] + [
        t[: -len("__*")] for t in denied_tools if t.endswith("__*")
    ]
    # Any-typed so the SDK's HookCallback union (typed against its own
    # TypedDict inputs) accepts our dict-based callback.
    policy_hook: Any = make_pretooluse_hook(effective_tools, agent_surfaces, denied_tools)
    options = ClaudeAgentOptions(
        allowed_tools=effective_tools,
        system_prompt=build_system_prompt(get_facility_timezone(), extra=surface_prompt),
        can_use_tool=make_backstop(effective_tools, agent_surfaces, denied_tools),
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[policy_hook])]},
        disallowed_tools=sorted(disallowed),
        cwd=project_dir,
        env=sdk_env,
        max_turns=max_turns,
        stderr=lambda line: stderr_lines.append(line),
        setting_sources=["project"],
        # Force the session id = the value injected above so the OTEL emitter's
        # session.id matches what provenance_locator returns for this run.
        session_id=telemetry_session_id,
    )

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    # Map tool_use_id -> index in tool_calls for matching results to calls
    pending_tools: dict[str, int] = {}
    cost_usd: float | None = None
    num_turns: int | None = None
    # Terminal error state read off the ResultMessage. Captured in the handler
    # below and consulted once the generator drains, so the completed branch can
    # flip a failed run to status "error" from a single decision point (poll
    # body, SSE stream, persisted record, and counters all follow from it).
    result_is_error = False
    result_subtype: str | None = None
    result_error_text: str | None = None
    result_api_error_status: Any = None
    # Snapshot secret values once so we can scrub them from anything we persist
    # or return (the SDK env carries provider/auth tokens).
    secret_values = _secret_values()

    def _num_calls() -> int:
        """Truthful tool-call count for this run (survives the retained cap)."""
        if run_id:
            return run_stats.get_run_stats(run_id)["num_tool_calls"]
        return len(tool_calls)

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

    # Assemble the user-message content once, before streaming: pop this run's
    # input-file seam (pop-once) and route each item by type — image inputs
    # become base64 image content blocks in this same user message, every input
    # file adds a mechanism-only descriptor line to the prompt. content_b64
    # rides only an image block's source.data, never the prompt text or a log.
    user_content = _assemble_user_content(prompt, _take_seam(run_id))

    # can_use_tool requires streaming-mode input (AsyncIterable), not str.
    async def _prompt_stream():
        yield {"type": "user", "message": {"role": "user", "content": user_content}}

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
                # No bytes from the provider within the window — a provider fault.
                return failure_class._stamp(
                    _finalize(
                        {
                            "status": "error",
                            "text_output": "".join(text_parts),
                            "tool_calls": tool_calls,
                            "error": msg,
                            "stderr": "\n".join(stderr_lines) if stderr_lines else None,
                            "duration_sec": round(duration_sec, 2),
                            "cost_usd": cost_usd,
                            "num_turns": num_turns,
                            "session_id": telemetry_session_id,
                        }
                    ),
                    failure_class.FAILURE_PROVIDER,
                    _num_calls(),
                )

            # Tool RESULTS arrive as ToolResultBlock inside UserMessage (the
            # SDK's message_parser wraps tool_result content that way), while
            # text and ToolUseBlock arrive in AssistantMessage — handle both so
            # per-call results (including permission denials) are captured.
            if isinstance(message, AssistantMessage | UserMessage):
                msg_content = message.content
                blocks = msg_content if isinstance(msg_content, list) else []
                for block in blocks:
                    if isinstance(block, TextBlock) and isinstance(message, AssistantMessage):
                        text_parts.append(block.text)
                        await _push({"type": "text", "content": block.text})
                    elif isinstance(block, ToolUseBlock):
                        # Count every tool call for a truthful total, even beyond
                        # the retained-list cap below, so the stamp sites can
                        # report the real number rather than len(tool_calls).
                        if run_id:
                            run_stats.increment_tool_calls(run_id)
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
                result_is_error = bool(getattr(message, "is_error", False))
                result_subtype = getattr(message, "subtype", None)
                result_error_text = getattr(message, "result", None)
                result_api_error_status = getattr(message, "api_error_status", None)
                await _push({"type": "result", "cost_usd": cost_usd, "num_turns": num_turns})

            elif isinstance(message, SystemMessage):
                logger.debug("SystemMessage: %s", message)

        duration_sec = time.monotonic() - t0

        # The SDK reported a terminal error (``is_error`` or a resource-cap
        # subtype): flip the run to status "error" here rather than reporting a
        # false "completed". Doing it at this one point keeps the poll body, SSE
        # stream, persisted record, and per-class counters mutually consistent.
        if result_is_error or failure_class.is_budget_subtype(result_subtype):
            error_text = result_error_text or f"Agent run ended in error (subtype={result_subtype})"
            if result_api_error_status and str(result_api_error_status) not in error_text:
                error_text = f"{error_text} (api_error_status={result_api_error_status})"
            # Resource caps (max_turns / max_budget) are a run-level terminal
            # state; any other reported error is classified from its text so a
            # provider fault surfaced as a result (e.g. a 429) is still tagged
            # provider and therefore stays retryable downstream.
            if failure_class.is_budget_subtype(result_subtype):
                cls = failure_class.FAILURE_RUN
            else:
                cls = failure_class.classify_exception(Exception(error_text))
            logger.info(
                "Dispatch ended in error after %.1fs: subtype=%s class=%s",
                duration_sec,
                result_subtype,
                cls,
            )
            await _push({"type": "error", "message": error_text})
            return failure_class._stamp(
                _finalize(
                    {
                        "status": "error",
                        "text_output": "".join(text_parts),
                        "tool_calls": tool_calls,
                        "error": error_text,
                        "duration_sec": round(duration_sec, 2),
                        "cost_usd": cost_usd,
                        "num_turns": num_turns,
                    }
                ),
                cls,
                _num_calls(),
            )

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
                "session_id": telemetry_session_id,
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
        # Cause is not known from this catch-all site — let the table-driven
        # classifier decide provider-vs-run from the exception and its chain.
        return failure_class._stamp(
            _finalize(
                {
                    "status": "error",
                    "text_output": "".join(text_parts),
                    "tool_calls": tool_calls,
                    "error": str(exc),
                    "stderr": stderr_output,
                    "duration_sec": round(duration_sec, 2),
                    "cost_usd": cost_usd,
                    "num_turns": num_turns,
                    "session_id": telemetry_session_id,
                }
            ),
            failure_class.classify_exception(exc),
            _num_calls(),
        )
