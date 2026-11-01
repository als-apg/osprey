"""Operator Mode session management on the shared agent runner.

Provides OperatorSession (one conversation held open on the shared agent
runner) and OperatorRegistry (multi-session manager with cleanup) for the
OSPREY Web Terminal operator mode.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

from osprey.agent_runner import (
    HAS_SDK,
    AgentEvent,
    AgentRunError,
    AgentSession,
    ApiErrorEvent,
    DeltaEvent,
    ResultEvent,
    SystemEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    ToolUseEvent,
    agent_session,
)
from osprey.agent_runner.clean_env import build_clean_env
from osprey.agent_runner.sdk_context import (
    PermissionResultAllow,
    PermissionResultDeny,
    PermissionUpdate,
    build_system_prompt,
)
from osprey.agent_runner.write_tools import load_approval_prefixes, read_only_disallowed_tools
from osprey.audit.posture import OSPREY_AGENT_DATA_ROOT
from osprey.build.claude_code_telemetry import terminal_session_env
from osprey.deployment.process_table import ProcessRow
from osprey.interfaces.web_auth import PANEL_TOKEN_ENV, get_web_credentials
from osprey.interfaces.web_terminal import process_tree
from osprey.interfaces.web_terminal.chat_session_pool import ChatSessionPool
from osprey.interfaces.web_terminal.process_tree import ProcessGroup
from osprey.interfaces.web_terminal.session_key import is_posture_key
from osprey.interfaces.web_terminal.slash_commands import operator_skill, operator_skill_frames
from osprey.interfaces.web_terminal.tool_preview import (
    tool_result_preview,
    tool_result_refusal,
    tool_result_text,
)
from osprey.utils.tool_rules import matches_denylist
from osprey_connectors.config import get_facility_timezone

logger = logging.getLogger(__name__)


#: Env marker naming *where* the child's posture decision came from, and the
#: audit session id it was made under. The pair is what the audit envelope
#: records as ``posture_source`` and ``session``; a child that finds neither
#: was not spawned by a posture-aware surface at all and reports ``process``.
POSTURE_SOURCE_ENV = "OSPREY_POSTURE_SOURCE"
POSTURE_SESSION_ENV = "OSPREY_POSTURE_SESSION"

#: The two sources a *spawning* surface can claim. ``live`` is a session key
#: the posture store keeps answering for after the child is up (the chat pool's
#: ``chat_id``, a PTY pool key); ``spawn`` is a key minted for one child and
#: addressable by nobody, so its posture is whatever was true at spawn.
POSTURE_SOURCE_SPAWN = "spawn"
POSTURE_SOURCE_LIVE = "live"

#: The closed set the envelope accepts. ``app`` is stamped by the HTTP layer
#: and ``process`` is what a child with no marker at all reports; neither is
#: reachable from this seam, but validating against the whole set here keeps a
#: typo from reaching an audit record as a source nothing can interpret.
_POSTURE_SOURCES = frozenset({POSTURE_SOURCE_SPAWN, POSTURE_SOURCE_LIVE, "app", "process"})


def resolve_agent_data_root(app: Any = None) -> str:
    """The agent-data root this server resolved, for the child's env stamp.

    One derivation for both spawn sites — this module already owns the posture
    markers they share, and a second copy of the resolution is exactly how two
    sites come to disagree. It answers the SHARED root (never
    ``resolve_agent_data_root`` from the workspace module, which appends
    ``sessions/<OSPREY_SESSION_ID>``): the control-context record spans
    sessions, and a reader outside the session's environment could not
    reproduce a session-scoped path.

    The stamp exists because everything below the spawn re-derives this
    directory today — the controls server through config, the stdlib-only hooks
    through a repo-root guess and the literal ``var/agent_data`` — and those
    derivations disagree the moment a deployment moves ``agent_data.base_dir``.
    Handing the child the answer makes it authoritative for every one of them.

    A config load can fail transiently, and the pair must be stamped whole or
    not at all, so a failure falls back to the same place the posture store's
    own resolution does (``app.state.workspace_dir``, then the process CWD)
    rather than leaving the child a session key with no anchor. Every reader
    prefers this value, so writer and readers stay on ONE directory even when
    it is the fallback one.

    Args:
        app: The FastAPI app, consulted only for the fallback. Optional: the
            primary resolution reads config and needs nothing from the server.

    Returns:
        An absolute path as a string, ready to be stamped into a child env.
    """
    try:
        from osprey_connectors.workspace import resolve_shared_data_root

        return str(resolve_shared_data_root())
    except Exception:  # a spawn must not fail on a config load
        state = getattr(app, "state", None)
        fallback = getattr(state, "workspace_dir", None) or Path.cwd()
        logger.warning(
            "Could not resolve the shared agent-data root for the session stamp; "
            "falling back to %s",
            fallback,
            exc_info=True,
        )
        return str(Path(fallback))


def build_operator_child_env(
    project_cwd: str | None,
    session_key: str | None = None,
    app: Any = None,
    *,
    posture_source: str = POSTURE_SOURCE_LIVE,
) -> dict[str, str]:
    """Build the environment for an SDK-backed operator or chat session.

    Both surfaces that run an :class:`OperatorSession` — the ``/ws/operator``
    websocket and the ``POST /api/chat`` endpoint — call this instead of
    :func:`~osprey.agent_runner.clean_env.build_clean_env` directly, so the two
    cannot drift on what the agent child is allowed to hold.

    On top of the clean base it re-adds exactly one credential: the **panel
    token**. That is deliberate re-introduction, the SDK counterpart of what
    :func:`osprey.cli.chat_cmd.chat` does for the PTY-less ``osprey chat``
    child and what
    :func:`osprey.interfaces.web_terminal.routes.websocket._build_extra_env`
    does for the PTY child. The token is the weak, panel-tier-only credential
    (see :data:`osprey.interfaces.web_auth.PANEL_TIER_ROUTES`, which is panel
    arrangement and activity plus the terminal proxy's hop to the event
    dispatcher's MCP transport); without it the MCP panel tools, the
    SessionStart/UserPromptSubmit/approval hooks the agent spawns and that
    dispatch hop send no bearer at all and are answered 401, which is how the
    panel tier came to be dead on the default ``osprey web`` launch.

    The **operator secret** is emphatically not re-added, and nothing here
    reads it. Note what that means on this path: the SDK builds the child's
    environment as ``{**os.environ, **options.env}``, so a name this function
    omits is inherited from ``os.environ`` anyway — the operator secret is kept
    from the child by :func:`osprey.interfaces.web_auth.close_env_carriers`
    having removed it from ``os.environ`` at app construction, not by its
    absence here.

    It does **not** carry the write posture into the child: that posture is per
    control target, lives in the control-context record, and is read live at
    every write-time gate, so a narrowing applies to a chat already
    mid-conversation. A deployment-wide readonly marker still reaches the
    child exactly as it always has, through ``build_clean_env``'s copy of
    ``os.environ``; nothing here sets or removes one.

    What it stamps instead is the **audit pair** that tells the child where to
    read that store: ``OSPREY_POSTURE_SOURCE`` (this call site's
    *posture_source*) and ``OSPREY_POSTURE_SESSION`` (*session_key*). Two rules
    matter:

    * The pair is set **unconditionally** whenever there is a session key —
      a ``writes`` session exports both markers, as does a key the store has
      never held. They say *which* key a reader must look up and *who*
      spawned the child, and a session that was checked and came back
      ``writes`` is a different audit fact from one nobody ever asked about.
    * ``posture_source`` is passed in, never worked out here. It cannot be
      derived from the posture: ``writes`` and never-stored are the same
      value, and the source is a property of the *call site*, not of the
      answer the store gave. Every in-tree caller names it explicitly.

    Stamped in the same breath, and for the same reason it must not be stamped
    separately, is :data:`~osprey.audit.posture.OSPREY_AGENT_DATA_ROOT` —
    :func:`resolve_agent_data_root`'s answer. A session key tells a reader
    *whose* posture applies; the root tells it which directory holds the
    answer. The two are one fact split in half, so they are set together or
    not at all, and a test pins that.

    The same key is also stamped as ``OSPREY_SESSION_ID``, which is what
    :func:`osprey_connectors.workspace.resolve_agent_data_root` reads to scope
    a child's execution root to ``sessions/<key>`` — where the python and
    sandbox executors put their run directories. One key, one root: a
    conversation is one session whichever surface is showing it, so the chat
    child and the PTY child spawned under the same key work out of the same
    directory rather than each in a room of its own. Its PTY counterpart
    stamps the same name from the pool key
    (:func:`osprey.interfaces.web_terminal.routes.websocket._build_extra_env`).
    It rides with the audit pair for the same reason the root does — all three
    describe one session — and never appears without a key to name.

    Keeping the stamps here — instead of handing each call site a rule to
    compose — is what stops the two SDK surfaces drifting on what a child is
    told about its own posture, which is the same reason this function exists
    at all.

    Args:
        project_cwd: The project directory the session runs in, forwarded to
            :func:`~osprey.agent_runner.clean_env.build_clean_env` so
            ``OSPREY_CONFIG`` is resolved from it.
        session_key: The identity this session is pooled under — the chat
            pool's ``chat_id`` for ``POST /api/chat``, the minted
            ``operator-<hex8>`` key for ``/ws/operator``. It is the key the
            posture store is read under and the name the child's execution
            root is scoped by. Omitted, the child gets the render's baseline
            environment and no marker is added — nothing names a session for a
            reader to look up.
        app: The FastAPI app whose agent-data root the child is pointed at
            (:func:`resolve_agent_data_root`). Stamped alongside *session_key*;
            without an app the root falls back to the deployment derivation.
        posture_source: Which surface is spawning this child, from
            :data:`_POSTURE_SOURCES`. ``POST /api/chat`` passes
            :data:`POSTURE_SOURCE_LIVE` for a ``chat_id`` the posture surface
            can address, and ``process`` for one it cannot (a caller-chosen id
            outside the bare-UUID grammar, which no store will ever answer
            for); ``/ws/operator``
            passes :data:`POSTURE_SOURCE_SPAWN` (its ``operator-<hex8>`` key is
            minted per connection and matches no bare-UUID id a caller could
            send). The value names the stamping surface only — the posture
            itself is resolved from the control-context record at decision
            time, whichever value was stamped. Defaults to ``live``, but every
            call site in this tree states its own, and a test pins that.

    Returns:
        A fresh env dict for the agent child.

    Raises:
        ValueError: If *posture_source* is outside the envelope's closed set.
    """
    if posture_source not in _POSTURE_SOURCES:
        raise ValueError(
            f"posture_source must be one of {sorted(_POSTURE_SOURCES)}, got {posture_source!r}"
        )

    env = build_clean_env(project_cwd=project_cwd)
    env[PANEL_TOKEN_ENV] = get_web_credentials().panel_token

    # The audit pair, stamped after the strip (build_clean_env would drop it)
    # and outside the sandbox branch below — see the rules in the docstring.
    # The agent-data root travels WITH the session key, never without it: the
    # key names whose posture applies and the root names the directory the
    # answer is read out of, and a child holding one but not the other would
    # look for that state in a directory it had to guess.
    # The terminal session travels with the posture session and equals it.
    if session_key:
        env[POSTURE_SOURCE_ENV] = posture_source
        env[POSTURE_SESSION_ENV] = session_key
        env[OSPREY_AGENT_DATA_ROOT] = resolve_agent_data_root(app)
        env["OSPREY_SESSION_ID"] = session_key
        env.update(terminal_session_env(session_key, env))

    return env


# Pattern for MCP tool name prefixes: mcp__<server>__<tool>
# Non-greedy so the FIRST ``__`` after the server name ends the prefix — a
# server name may itself contain single underscores (osprey_workspace,
# osprey_facility_knowledge, or any facility-declared server named that way).
_MCP_PREFIX_RE = re.compile(r"^mcp__.+?__")

# Bound (seconds) for draining an interrupted turn toward its terminal message
# before the reader is hard-cancelled. Enforced inside OperatorSession.cancel().
_QUIESCE_TIMEOUT_S = 5.0

#: Bound (seconds) for draining a turn stopped while the main agent was idle
#: waiting on a background agent: the CLI answers the stop with a model turn
#: of its own, and that must be read off the stream before the next prompt.
_BACKGROUND_DRAIN_S = 30.0

#: Bound (seconds) for reading the context window at the end of a response. The
#: reading is best-effort: a CLI that does not answer in time costs the meter
#: its reading, and the turn still ends on its result.
_CONTEXT_READ_S = 5.0

#: The SDK tool whose permission request is a question for the operator, not a
#: tool approval: the answer goes back as ``answers`` in the tool's input.
_QUESTION_TOOL = "AskUserQuestion"

#: The tool the agent calls to leave plan mode; its request is the plan, for
#: the operator to approve or send back. Like a question, it is not a write.
_PLAN_TOOL = "ExitPlanMode"

#: How many frames a feed may fall behind before it is dropped; a dropped feed
#: reconnects and starts again from a snapshot.
_SUBSCRIBER_BUFFER = 2000

#: What approving a plan applies: back to the default mode, for this session.
_LEAVE_PLAN_MODE = (
    [PermissionUpdate(type="setMode", mode="default", destination="session")]
    if PermissionUpdate is not None
    else None
)

#: What a refused ExitPlanMode says when no plan is on file. The operator is never shown
#: a plan card with nothing on it: the agent writes the plan first, then asks again.
_NO_PLAN_MESSAGE = (
    "There is no plan to approve yet. Write the plan to the plan file, "
    "then call ExitPlanMode again."
)

#: How long a permission request waits for the operator before it is declined.
#: An unanswered card keeps its turn open, and the turn its pool slot; declining
#: is the safe direction, since the tool then never runs. One figure for every
#: kind of card; not a config key.
ASK_EXPIRY_S = 30 * 60.0

#: What the agent is told when a card expired, by the kind of request it was: the
#: tool was declined, the plan is kept in planning, the question got no answer.
_EXPIRED_MESSAGES = {
    "tool": "No answer from the operator within 30 minutes; the call was declined.",
    "plan": "No answer from the operator within 30 minutes; keep planning.",
    "question": "No answer from the operator within 30 minutes; the question was not answered.",
}

#: The tools that start a subagent (``Task`` was renamed ``Agent``), and the
#: text their result opens with when the agent went to the background — it then
#: reports in a later response of its own.
_AGENT_TOOLS = frozenset({"Task", "Agent"})
_ASYNC_LAUNCH_MARK = "Async agent launched"

#: What the chat's agent is told about drawing in its replies. Only this chat
#: draws a fenced ``svg`` block (the terminal would show its source), so the
#: guide rides on this session's system prompt, not on the shared rules.
_SVG_CALLOUT_GUIDE = (
    "Your replies are shown in the web terminal's operator chat, which renders markdown "
    "and draws a fenced `svg` code block as a picture in a box. Use a drawing only as a "
    "quick look at a result you saved as an artifact: the top few values, a handful of "
    "key numbers, or a small schematic of where something is. Make at most one drawing "
    "in a reply, and always name the artifact it comes from in your text as "
    "`artifact_id: <id>`, so the operator can open the full result. Every value in a "
    "drawing must come from a tool result or an artifact; never invent or estimate one. "
    "Do not draw what a plot tool should make: the artifact is the result, the drawing "
    "is a glance at it. Skip the drawing when the reply is one or two numbers. Decide "
    "whether to draw before you start writing the reply, and put the drawing in that "
    "reply: once a reply is sent it stands. Never call a tool you do not need, such as a "
    "read of an empty file or a subagent with nothing to do, only to keep your turn "
    "going.\n"
    "What to draw: bars only for values that share one unit, each bar's length "
    "proportional to its value and measured from a common zero, with the value written "
    "at its end. Values in different units are never bars: show them as a small grid of "
    "labelled numbers (a label, the value and its unit), one tile each. Give one colour to "
    "each kind of thing (every cavity the same colour), use var(--color-accent) for the "
    "item the reply is about and var(--text-muted) for the rest, and lay items out in one "
    "column or one row so the eye reads them in order.\n"
    "How to draw: one <svg> element with a viewBox no larger than 640 by 240; text at "
    "font-size 12 to 14; colours only currentColor, var(--text-primary), "
    "var(--text-secondary), var(--text-muted), var(--border-default), var(--bg-elevated), "
    "var(--color-accent), var(--color-accent-secondary), var(--color-success), "
    "var(--color-warning) and var(--color-error), set as fill and stroke attributes on "
    "each element; no background fill. Style elements, style attributes and any other "
    "colour are removed before the drawing is shown, and a part left with no colour draws "
    "in the reply's text colour. Scripts, foreignObject, links, images, <use> and "
    "animation are removed too."
)

#: Task statuses that end a background agent, across ``task_notification``'s
#: and ``task_updated``'s vocabularies (the SDK's TERMINAL_TASK_STATUSES).
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "stopped", "killed"})


#: The deny for an ask that is not a write on a render whose tier allows no writes:
#: nobody can approve a tool call on it, and it is not a write to refuse by reason.
_NOT_ARMED_MESSAGE = "This deployment does not arm writes, so the operator cannot approve it."


@dataclass(frozen=True)
class _RenderFacts:
    """What the chat reads from the render its child runs in, once at start.

    ``config`` is ``<project_dir>/config.yml`` and ``hook_config`` the render's
    ``.claude/hooks/hook_config.json``, each ``{}`` when it cannot be read.
    ``tier_allows`` is the tier term of the write verdict
    (:func:`osprey_connectors.types.tier_allows`) over the config's control
    section; it fails closed with the config.
    """

    config: dict[str, Any]
    hook_config: dict[str, Any]
    tier_allows: bool


#: The facts of a render that could not be read: no writes, no placement.
_NO_RENDER = _RenderFacts(config={}, hook_config={}, tier_allows=False)


def _read_render(project_dir: str) -> _RenderFacts:
    """The render facts of *project_dir*. BLOCKING; never raises.

    Read from ``<project_dir>``, the files the child's own ``OSPREY_CONFIG`` and
    hooks read (:func:`~osprey.agent_runner.clean_env.build_clean_env`), not
    whichever config the server process resolves. A render whose config cannot
    be read, or that states no control section, allows no writes; an unreadable
    hook_config places writes from the hooks' fallback list.
    """
    import json

    import yaml

    from osprey_connectors.types import tier_allows

    try:
        config = yaml.safe_load((Path(project_dir) / "config.yml").read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read the render's config; chat approvals stay off", exc_info=True)
        return _NO_RENDER
    if not isinstance(config, dict):
        return _NO_RENDER
    try:
        hook_path = Path(project_dir) / ".claude" / "hooks" / "hook_config.json"
        hook_config = json.loads(hook_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("No readable hook_config in %s", project_dir, exc_info=True)
        hook_config = {}
    return _RenderFacts(
        config=config,
        hook_config=hook_config if isinstance(hook_config, dict) else {},
        tier_allows=tier_allows(config.get("control_system")),
    )


def _place_write(
    render: _RenderFacts, tool_name: str, tool_input: dict[str, Any], stamped: bool | None
) -> tuple[bool, dict[str, str] | None]:
    """Whether a call writes, and C-FRAME's ``target`` for it. BLOCKING; never raises.

    Placed as the hooks place it (:func:`osprey_connectors.write_target.write_target_for_tool`):
    a call that writes nothing answers ``(False, None)``, a write nobody can place
    ``(True, None)``.
    """
    from osprey_connectors.write_target import frame_target, write_target_for_tool

    placed = write_target_for_tool(
        render.config, tool_name, tool_input, hook_config=render.hook_config, stamped=stamped
    )
    if placed is None:
        return False, None
    try:
        return True, frame_target(placed)
    except Exception:
        logger.debug("Could not name the machine of %s", tool_name, exc_info=True)
        return True, None


def _main_agent_active(event: AgentEvent) -> bool:
    """Whether *event* is the main agent itself working (not a subagent, not bookkeeping)."""
    if isinstance(event, (TextEvent, ThinkingEvent, ToolUseEvent)):
        return event.parent_tool_use_id is None
    return isinstance(event, SystemEvent) and event.subtype == "init"


def _starts_response(event: AgentEvent) -> bool:
    """Whether *event* means the CLI is answering something, not just reporting.

    A response opens with the ``init`` notice and carries the agent's records.
    The CLI's other notices — the SessionStart hooks it runs before any
    prompt, task progress, … — say nothing about a response being under way;
    counting them would park the next prompt behind a response that is not
    coming.
    """
    if isinstance(event, SystemEvent):
        return event.subtype == "init"
    return not isinstance(event, ResultEvent)


#: How many responses in a row the CLI may produce with no prompt behind them
#: before the pump stops reading. The CLI answers on its own only after a
#: background agent reports; a run of them means the stream is broken, and
#: reading on would spin and keep every one of them in memory.
_MAX_UNPROMPTED_RESPONSES = 20


@dataclass
class _PendingAsk:
    """One permission request waiting on the operator."""

    future: asyncio.Future[Any]
    tool_input: dict[str, Any]
    #: The ``permission_request`` frame, so a tab that joins late is shown it.
    frame: dict[str, Any]
    #: The rules "Allow for this session" adds; empty when it is not offered.
    standing: list[Any]


def _resolved_frame(
    request_id: str,
    *,
    allow: bool,
    for_session: bool = False,
    answers: dict[str, str] | None = None,
    expired: bool = False,
) -> dict[str, Any]:
    """The ``permission_resolved`` frame: how a request was answered, told to every tab.

    ``expired`` marks a request nobody answered in time, declined by the chat
    itself (:meth:`OperatorSession._expire`); a card still open shows that
    rather than an operator's decline.
    """
    return {
        "type": "permission_resolved",
        "request_id": request_id,
        "allow": allow,
        "for_session": for_session,
        "answers": answers,
        "expired": expired,
    }


#: The suffix of a whole-server matcher, ``mcp__<server>__.*``, as the hooks read
#: their write lists (``_MATCHER_WILDCARD`` in ``osprey_hook_log.py``).
_MATCHER_WILDCARD = ".*"


def _is_side_effect_tool(tool_name: str, side_effect_tools: Sequence[str]) -> bool:
    """Whether *tool_name* is on the side-effect surface, each entry read as it is meant.

    The surface is :func:`~osprey.agent_runner.write_tools.read_only_disallowed_tools`,
    the list the headless read-only run refuses. An exact name, or a name ending in
    ``*``, matches as in :func:`~osprey.utils.tool_rules.matches_denylist`; a
    facility server that declares ``writes:`` reaches the surface that way, one
    exact name per write tool, so only those tools are covered, not its reads. A
    whole-server matcher ``mcp__<server>__.*`` covers that server's tools, as the
    writes-check hook reads it. A server-level entry ``mcp__<server>`` covers every
    tool of that server, as the disallow engine reads it. A non-string entry covers
    nothing.
    """
    names = [entry for entry in side_effect_tools if isinstance(entry, str)]
    if matches_denylist(tool_name, names):
        return True
    for entry in names:
        if entry.endswith(_MATCHER_WILDCARD):
            prefix = entry[: -len(_MATCHER_WILDCARD)]
        elif entry.startswith("mcp__"):
            prefix = f"{entry}__"
        else:
            continue
        if tool_name.startswith(prefix):
            return True
    return False


def _standing_rules(
    tool_name: str, suggestions: list[Any], side_effect_tools: Sequence[str]
) -> list[Any]:
    """The CLI's suggested allow rules for *tool_name*, scoped to this session only.

    Never for a tool on the side-effect surface, so one that acts on the machine is
    approved per call every time. Never saved either: a suggestion aimed at the
    project's or the user's settings is narrowed to the session, so nothing an
    operator clicks in the chat outlives it. Never wider than the tool asked about
    either: only ``addRules`` suggestions are kept, and of their rules only those
    naming *tool_name* itself, so a suggestion that would also cover a server's
    other tools, or another tool, adds nothing.
    """
    if _is_side_effect_tool(tool_name, side_effect_tools):
        return []
    kept = []
    for suggestion in suggestions:
        if (
            getattr(suggestion, "type", None) != "addRules"
            or getattr(suggestion, "behavior", None) != "allow"
        ):
            continue
        rules = [
            rule
            for rule in getattr(suggestion, "rules", None) or []
            if getattr(rule, "tool_name", None) == tool_name
        ]
        if rules:
            kept.append(replace(suggestion, rules=rules, destination="session"))
    return kept


def format_tool_name(raw: str) -> str:
    """Convert raw tool name to a human-readable display name.

    Strips ``mcp__<server>__`` prefix and title-cases the remainder,
    replacing underscores with spaces; a built-in's CamelCase name is split
    into its words. The replay (``routes/session.py``) names tools through this
    same function.

    Examples:
        >>> format_tool_name("mcp__osprey__channel_read")
        'Channel Read'
        >>> format_tool_name("Read")
        'Read'
        >>> format_tool_name("ToolSearch")
        'Tool Search'
    """
    name = _MCP_PREFIX_RE.sub("", raw)
    if "_" not in name and name[:1].isupper():
        return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return name.replace("_", " ").title()


def _event_to_wire(event: AgentEvent) -> dict[str, Any] | None:
    """Convert one agent event record to the frame the operator stream carries.

    Args:
        event: A record from the agent runner's turn stream.

    Returns:
        An event dict suitable for JSON serialisation over WebSocket, or
        ``None`` for a record the stream does not carry.
    """
    if isinstance(event, ApiErrorEvent):
        return {
            "type": "error",
            "message": f"API error: {event.error}",
            "error_type": "AssistantMessageError",
        }
    # ``parent_tool_use_id`` names the subagent call a record belongs to (None
    # for the main agent's own), so the chat can file it under that agent.
    if isinstance(event, TextEvent):
        return {
            "type": "text",
            "content": event.text,
            "parent_tool_use_id": event.parent_tool_use_id,
        }
    if isinstance(event, ThinkingEvent):
        return {
            "type": "thinking",
            "content": event.text,
            "parent_tool_use_id": event.parent_tool_use_id,
        }
    if isinstance(event, ToolUseEvent):
        return {
            "type": "tool_use",
            "tool_name": format_tool_name(event.name),
            "tool_name_raw": event.name,
            "tool_use_id": event.tool_use_id,
            "input": event.input,
            "parent_tool_use_id": event.parent_tool_use_id,
        }
    if isinstance(event, ResultEvent):
        return {
            "type": "result",
            "is_error": event.is_error,
            "total_cost_usd": event.total_cost_usd,
            "duration_ms": event.duration_ms,
            "num_turns": event.num_turns,
        }
    if isinstance(event, SystemEvent):
        frame: dict[str, Any] = {"type": "system", "subtype": event.subtype}
        # The init message carries the id of the transcript the child is
        # actually writing to, which is the only place a resume's real
        # transcript id can be read back from — a resume may be answered with a
        # different id than the one asked for. Carried only when the message
        # has one, so a system message that says nothing about identity does
        # not look like one reporting a missing id.
        session_id = event.data.get("session_id")
        if session_id:
            frame["session_id"] = session_id
        frame.update(_notice_fields(event))
        return frame
    if isinstance(event, ToolResultEvent):
        # The refusal trailer ends the text, so it is read before the preview cut.
        refusal, shown = tool_result_refusal(tool_result_text(event.content), event.is_error)
        preview, truncated = tool_result_preview(shown)
        return {
            "type": "tool_result",
            "tool_use_id": event.tool_use_id,
            "is_error": event.is_error,
            "preview": preview,
            "truncated": truncated,
            "refusal": refusal,
        }
    if isinstance(event, DeltaEvent):
        return {
            "type": "delta",
            "kind": event.kind,
            "content": event.text,
            "parent_tool_use_id": event.parent_tool_use_id,
        }
    if isinstance(event, ToolStartEvent):
        # The chat's status line names the call while its input is still
        # being written; the row itself waits for the whole ``tool_use``.
        return {
            "type": "tool_start",
            "tool_name": format_tool_name(event.name),
            "tool_name_raw": event.name,
            "tool_use_id": event.tool_use_id,
            "parent_tool_use_id": event.parent_tool_use_id,
        }
    # Any other record (rate-limit notices, …) stays off the stream.
    return None


def _notice_fields(event: SystemEvent) -> dict[str, Any]:
    """The fields the chat shows for a CLI notice: compaction, API retries, status.

    ``compact_boundary`` says the conversation was summarised (what started it
    and how many tokens it had); ``api_retry`` that a request failed and is being
    retried; ``status`` what the CLI is doing now (``compacting``, or nothing).
    """
    data = event.data
    if event.subtype == "compact_boundary":
        meta = data.get("compact_metadata") or data.get("compactMetadata") or {}
        return {"trigger": meta.get("trigger"), "pre_tokens": meta.get("pre_tokens")}
    if event.subtype == "api_retry":
        return {
            key: data.get(key)
            for key in ("attempt", "max_retries", "retry_delay_ms", "error_status")
        }
    if event.subtype == "status":
        return {"status": data.get("status")}
    return {}


def _context_frame(usage: Any) -> dict[str, Any] | None:
    """The ``context`` frame for one reading of the context window, or ``None``.

    The chat shows the window's occupancy only: the tokens in use, the size of
    the window, and the percentage the CLI rounds from the two (its total tokens
    over the raw window). The rest of the reading, the file and tool breakdowns
    and the API usage with its cost, never reaches the stream.
    """
    if not isinstance(usage, dict):
        return None
    used = usage.get("totalTokens")
    window = usage.get("rawMaxTokens")
    percent = usage.get("percentage")
    if not _finite(used) or not _finite(window) or not _finite(percent) or window <= 0:
        return None
    return {
        "type": "context",
        "used_tokens": int(used),
        "window_tokens": int(window),
        "percent": max(0, min(100, round(percent))),
    }


def _finite(value: object) -> TypeGuard[float]:
    """Whether *value* is a finite number (a bool is not one)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


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
    return etype == "error" and event.get("error_type") != "AssistantMessageError"


@dataclass(frozen=True)
class _AgentTree:
    """What the chat child had started, read while it was alive."""

    groups: list[ProcessGroup]
    members: list[ProcessRow]
    # False when the process table could not be read: what the child started is unknown.
    read: bool


@dataclass(frozen=True)
class TreeEnding:
    """What a stop did about the processes the chat child had started.

    ``read`` is False when the process table could not be read while the child
    was alive: nothing was ended because nothing could be looked for, which is
    not the same as the child having started nothing. The ``ended`` fields
    hold what was signalled, the ``surviving`` ones what a read still found
    after SIGKILL, and the ``unknown`` ones what the table could not be read
    to re-identify: it was not signalled and may still be running.
    """

    read: bool
    ended_groups: tuple[ProcessGroup, ...] = ()
    ended_members: tuple[ProcessRow, ...] = ()
    surviving_groups: tuple[ProcessGroup, ...] = ()
    surviving_members: tuple[ProcessRow, ...] = ()
    unknown_groups: tuple[ProcessGroup, ...] = ()
    unknown_members: tuple[ProcessRow, ...] = ()


def _read_tree(pid: int, alive: Callable[[], bool]) -> _AgentTree:
    """Read the groups *pid*'s descendants are in and its descendants in the server's group.

    Both come from one process table, read while *alive* says *pid* runs and
    retried while it does. When the table could not be read the tree is
    returned unread and empty, and a warning names the child and the cause.
    Blocking (it reads the process table).
    """
    try:
        rows = process_tree.read_while_alive(alive)
    except process_tree.ProcessTableUnreadable as exc:
        logger.warning(
            "Could not read the process table while the chat child %d was alive (%s); "
            "what it started is left running",
            pid,
            exc.cause,
        )
        return _AgentTree(groups=[], members=[], read=False)
    return _AgentTree(
        groups=process_tree.tree_groups(pid, rows),
        members=process_tree.server_group_members(pid, rows),
        read=True,
    )


def _end_tree(pid: int, tree: _AgentTree) -> TreeEnding:
    """End what *tree* found under the chat child *pid*, and log what became of it. Blocking."""
    groups = process_tree.end_groups(tree.groups)
    members = process_tree.end_processes(tree.members)
    if groups.ended or members.ended:
        logger.info(
            "Ended %d process group(s) and %d process(es) the chat child %d started: %s",
            len(groups.ended),
            len(members.ended),
            pid,
            _describe(groups.ended, members.ended),
        )
    if groups.survivors or members.survivors:
        logger.warning(
            "Process(es) the chat child %d started survived SIGKILL: %s",
            pid,
            _describe(groups.survivors, members.survivors),
        )
    if groups.unknown or members.unknown:
        logger.warning(
            "Chat child %d: process table unreadable: started process(es) may still be running: %s",
            pid,
            _describe(groups.unknown, members.unknown),
        )
    return TreeEnding(
        read=tree.read,
        ended_groups=groups.ended,
        ended_members=members.ended,
        surviving_groups=groups.survivors,
        surviving_members=members.survivors,
        unknown_groups=groups.unknown,
        unknown_members=members.unknown,
    )


def _describe(groups: tuple[ProcessGroup, ...], members: tuple[ProcessRow, ...]) -> str:
    """One line naming *groups* and then *members*, either part left out when empty."""
    parts = (process_tree.describe(groups), process_tree.describe_processes(members))
    return "; ".join(part for part in parts if part)


class OperatorSession:
    """One operator-mode conversation, run on the shared agent runner."""

    def __init__(
        self,
        cwd: str,
        env: dict[str, str] | None = None,
        *,
        session_key: str | None = None,
    ) -> None:
        """Build an unstarted session.

        Args:
            cwd: The project directory the child runs in.
            env: The child's environment, or ``None`` to inherit the process
                environment. Fixed at spawn; see
                :func:`build_operator_child_env`.
            session_key: The identity this session is pooled and audited
                under — the shared session key, not a per-child id. It is the
                telemetry session id, and the SDK session id for a child that
                starts a conversation of its own. Omitted, one is minted here,
                so the key is stable for the life of the object rather than
                re-drawn on every :meth:`start`.
        """
        self._cwd = cwd
        self._env = env
        self._session_key = session_key or str(uuid.uuid4())
        # The runner's handle, kept after close so the child stays observable
        # (see :meth:`stop`, :attr:`pid` and :attr:`process_exited`).
        self._agent: AgentSession | None = None
        # Open exactly while the runner is connected; liveness keys on this.
        self._agent_scope: AsyncExitStack | None = None
        # What the last stop that found a live child did about what it started.
        self._tree_ending: TreeEnding | None = None
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._quiesce_task: asyncio.Task | None = None
        self._started = False
        # The one reader of the agent's stream, for the life of the session
        # (see ``_pump``), and where what it reads goes. ``_turn_open``: the
        # response being read answers a submitted prompt, so it goes to the
        # turn's consumer. ``_pending_turn``: a prompt arrived while the CLI was
        # mid-way through a response of its own; its turn opens when that one
        # ends. ``_in_response``: the pump has read a record since the last
        # result. ``_turn_closed`` is set whenever no turn is open.
        self._pump_task: asyncio.Task | None = None
        # Set when the pump stopped reading on its own (the child's stream
        # ended, or the pump gave up on it): the session is dead from then on.
        self._stream_ended = False
        self._turn_open = False
        self._pending_turn = False
        self._in_response = False
        self._turn_prompt: str | None = None
        self._turn_closed = asyncio.Event()
        self._turn_closed.set()
        # A turn whose consumer went away (the tab closed or reloaded) runs on;
        # its guard is held here and released when the turn ends.
        self._orphan_token: int | None = None
        # Tokens whose run_turn is reading the queue right now.
        self._consumers: set[int] = set()
        # Responses in a row the CLI produced with no prompt behind them.
        self._unprompted = 0
        # Live feeds of every frame (``subscribe``), and the frame counter
        # that lets a client holding two streams draw each frame once.
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._seq = 0
        # Per-turn one-in-flight guard. ``_turn_epoch`` only ever increments and
        # names each turn ever started; ``_active_token`` holds the current
        # turn's epoch or None when idle. Manipulated only by the synchronous,
        # lock-free acquire_turn/release_turn pair — no awaits, so it is safe to
        # touch outside the registry lock (handler finally + teardown paths).
        self._turn_epoch = 0
        self._active_token: int | None = None
        # Permission requests waiting on the operator, by request id. Filled by
        # ``_can_use_tool``; drained by ``resolve_permission`` or, on interrupt
        # and teardown, by ``_deny_pending``.
        self._permissions: dict[str, _PendingAsk] = {}
        # The render the child runs in, read at start: its config, hook_config and
        # whether its tier allows writes at all.
        self._render = _NO_RENDER
        # The side-effect surface this render refuses headless, read at start. No session
        # allow is ever offered for one: its approval is per call.
        self._side_effect_tools: tuple[str, ...] = ()
        # Background agents the current turn launched and that have not yet
        # reported, keyed by the tool_use id that launched them; the agent tool
        # calls seen this turn; and each started task's id -> its tool_use id.
        # While any background agent is open the turn stays open (see
        # ``_stream_response``).
        self._background: set[str] = set()
        self._agent_calls: set[str] = set()
        self._task_calls: dict[str, str] = {}
        # True while the main agent is idle waiting on a background agent's
        # report: a Stop then stops the agent and drains, never interrupts.
        self._idle_wait = False
        # The last context-window reading this child sent, which a tab joining the
        # feed is shown with its snapshot.
        self._last_context: dict[str, Any] | None = None
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

        Busy = the per-turn guard is held AND a turn is open (or waiting to
        open) on a live pump, or a quiesce is still running. A *zombie* — guard
        held but nothing reading — is NOT busy, so pools may evict/reap it.
        """
        if not self.in_flight:
            return False
        pump = self._pump_task
        quiesce = self._quiesce_task
        pump_running = pump is not None and not pump.done()
        quiesce_running = quiesce is not None and not quiesce.done()
        return (pump_running and (self._turn_open or self._pending_turn)) or quiesce_running

    async def start(self, *, resume_id: str | None = None) -> None:
        """Open the conversation on the shared agent runner.

        Args:
            resume_id: The transcript to continue. Given, the child resumes
                that conversation (``resume``) and the SDK keeps writing to it;
                omitted, the child starts a conversation of its own under the
                session key (``session_id``), or under an id it mints itself
                when the key is not one the CLI would accept. Resume and
                session id are mutually exclusive — a resume names the
                transcript, and naming a session id alongside it would ask the
                SDK to write one conversation under two identities. Which
                applies is the caller's decision, taken from the session's
                transcript state; this seam only offers the shapes.
        """
        if not HAS_SDK:
            raise RuntimeError("claude-agent-sdk is not installed")

        # Warn about missing OSPREY project files
        for warning in validate_project_directory(self._cwd):
            logger.warning("Operator session: %s (cwd=%s)", warning, self._cwd)

        # Hand the session key to the workspace provenance_locator tool via env,
        # so a filed issue can point back to this session's telemetry. It is the
        # key the session is pooled and audited under, never a value minted
        # here: a telemetry id that changed every time a surface re-acquired the
        # session would split one conversation's traces across several ids. env
        # is a build_clean_env dict in production (which strips the harness's own
        # CLAUDE_CODE_* id); inject into a copy.
        session_env = dict(self._env) if self._env is not None else None
        if session_env is not None:
            session_env["OSPREY_TELEMETRY_SESSION_ID"] = self._session_key
            session_env["OSPREY_TELEMETRY_SESSION_START"] = datetime.now(UTC).isoformat()
            # Mark the web surface this session serves. The panels-context
            # SessionStart hook reads this to tell
            # the agent which UI the operator is looking at (the PTY terminal
            # sets "terminal" — see routes/websocket.py's _build_extra_env).
            session_env["OSPREY_WEB_UX"] = "chat"

        # The launch names exactly one identity keyword, or none: the two are
        # mutually exclusive, and the runner refuses both together.
        if resume_id is not None:
            session_id = None
        else:
            # The key names the conversation only where the CLI would accept it
            # as one. POST /api/chat takes any string for chat_id — an embedder
            # keying its chats "user-42-chat-3" is a supported caller, and the
            # posture surface already treats such a key as unaddressable — but
            # here the key reaches the CLI as `--session-id`, which takes a
            # canonical UUID and exits non-zero on anything else. The SDK
            # imposes no format of its own, so an unchecked key would fail the
            # child on its first prompt rather than at a seam anyone can see.
            # None omits the flag and the child mints its own id, which is what
            # this surface did before it could resume at all.
            session_id = self._session_key if is_posture_key(self._session_key) else None

        # The chat routes its own run through the env it hands over (an empty
        # mapping inherits the process environment), sets no permission mode,
        # turn cap or budget, and waits for no MCP server. Every ask the
        # project's settings and hooks raise reaches ``_can_use_tool``, which
        # puts it in front of the operator on a write-enabled render.
        self._render = await asyncio.to_thread(_read_render, self._cwd)
        # The side-effect surface, read from the render the child runs in, plus every
        # prefix the render's approval hook asks about: the hook asks on each call, so
        # a standing allow there would do nothing or skip its ask. If either cannot be
        # read, no session allow is offered for any tool: "*" is the catch-all entry.
        try:
            surface = await asyncio.to_thread(read_only_disallowed_tools, Path(self._cwd))
            prefixes = await asyncio.to_thread(load_approval_prefixes, Path(self._cwd))
            self._side_effect_tools = (*surface, *(f"{prefix}*" for prefix in prefixes))
        except Exception:
            logger.warning(
                "Side-effect surface unreadable for %s; no session allows",
                self._cwd,
                exc_info=True,
            )
            self._side_effect_tools = ("*",)
        scope = AsyncExitStack()
        agent = await scope.enter_async_context(
            agent_session(
                Path(self._cwd),
                disallowed_tools=[],
                max_turns=None,
                max_budget_usd=None,
                permission_mode=None,
                setting_sources=["project"],
                system_prompt=build_system_prompt(get_facility_timezone(), _SVG_CALLOUT_GUIDE),
                env=session_env if session_env is not None else {},
                resume=resume_id,
                session_id=session_id,
                await_mcp_servers=(),
                can_use_tool=self._can_use_tool,
                include_partial_messages=True,
            )
        )
        self._agent = agent
        self._agent_scope = scope
        self._started = True
        self._stream_ended = False
        self._pump_task = asyncio.create_task(self._pump())
        logger.info("OperatorSession started (cwd=%s)", self._cwd)

    # ---- Operator approvals ---- #

    async def _can_use_tool(self, tool_name: str, tool_input: dict[str, Any], context: Any) -> Any:
        """Put one permission request in front of the operator and wait for the answer.

        Called by the SDK for every tool use the project's settings and hooks
        leave at ``ask``, for every ``AskUserQuestion`` and for ``ExitPlanMode``.
        The request goes out on the turn's stream and every feed as a
        ``permission_request`` frame; the turn waits here until
        :meth:`resolve_permission` answers it, until an interrupt or teardown
        declines it, or until :data:`ASK_EXPIRY_S` passes with no answer, when it
        is declined (see :meth:`_expire`).

        An ``ExitPlanMode`` whose input carries no plan is refused here without a card.
        """
        kind = {_QUESTION_TOOL: "question", _PLAN_TOOL: "plan"}.get(tool_name, "tool")
        target: dict[str, str] | None = None
        if kind == "tool":
            writes, target = await asyncio.to_thread(
                _place_write, self._render, tool_name, tool_input, self._child_stamped()
            )
            if not self._render.tier_allows:
                if not writes:
                    return PermissionResultDeny(message=_NOT_ARMED_MESSAGE)
                # The tier verdict (C-FRAME): the refusal text carries the trailer,
                # so the tool_result this deny becomes carries ``refusal``.
                from osprey_connectors.refusals import refusal_text

                machine = target["machine_kind"] if target else None
                return PermissionResultDeny(message=refusal_text("tier_never", machine))
        plan = tool_input.get("plan")
        if kind == "plan" and not (isinstance(plan, str) and plan.strip()):
            return PermissionResultDeny(message=_NO_PLAN_MESSAGE)
        standing = _standing_rules(
            tool_name, getattr(context, "suggestions", None) or [], self._side_effect_tools
        )
        request_id = uuid.uuid4().hex
        frame = {
            "type": "permission_request",
            "request_id": request_id,
            "kind": kind,
            "tool_name": format_tool_name(tool_name),
            "tool_name_raw": tool_name,
            "tool_use_id": getattr(context, "tool_use_id", None),
            "input": tool_input,
            "reason": getattr(context, "decision_reason", None),
            "title": getattr(context, "title", None),
            "description": getattr(context, "description", None),
            # The machine a write lands on (C-FRAME); None for a call that writes
            # nothing and for a write that cannot be placed.
            "target": target,
            # Whether "Allow for this session" is offered (see _standing_rules).
            "can_allow_session": bool(standing),
        }
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._permissions[request_id] = _PendingAsk(future, tool_input, frame, standing)
        try:
            await self._emit(frame)
            try:
                return await asyncio.wait_for(future, timeout=ASK_EXPIRY_S)
            except TimeoutError:
                return await self._expire(request_id, kind)
        finally:
            self._permissions.pop(request_id, None)

    def _child_stamped(self) -> bool | None:
        """Whether the child's agent-data root is stamped; ``None`` asks this process.

        The hooks in the child decide "no record" by their own environment, so
        the placement asks the env the child was handed.
        """
        if self._env is None:
            return None
        return bool(str(self._env.get(OSPREY_AGENT_DATA_ROOT) or "").strip())

    async def _expire(self, request_id: str, kind: str) -> Any:
        """Decline a request the operator never answered, and tell every tab so.

        The decline is the same one a "No" gives, so the turn goes on or ends
        as it would after one and the pool slot comes free; the agent is told
        there was no answer. The frame is the usual ``permission_resolved`` with
        ``expired`` set, so a card still open settles as expired rather than as
        answered. A late answer finds nothing waiting (:meth:`resolve_permission`).
        """
        logger.info("Chat %s: a %s request expired unanswered", self._session_key, kind)
        await self._emit(_resolved_frame(request_id, allow=False, expired=True))
        return PermissionResultDeny(message=_EXPIRED_MESSAGES[kind])

    async def resolve_permission(
        self,
        request_id: str,
        *,
        allow: bool,
        answers: dict[str, str] | None = None,
        message: str | None = None,
        for_session: bool = False,
    ) -> bool:
        """Answer one pending permission request. False when nothing waits under *request_id*.

        ``for_session`` also adds the CLI's suggested allow rules for the rest
        of this session — only where the request offered it. Every tab is told
        the request was answered, so a card open in another one settles too.
        """
        pending = self._permissions.get(request_id)
        if pending is None or pending.future.done():
            return False
        if not allow:
            result: Any = PermissionResultDeny(
                message=message
                or (
                    "The operator wants to keep planning."
                    if pending.frame["kind"] == "plan"
                    else "The operator declined this tool call."
                )
            )
        elif answers is not None:
            result = PermissionResultAllow(updated_input={**pending.tool_input, "answers": answers})
        elif pending.frame["kind"] == "plan":
            # Approving the plan leaves plan mode for the default one, where
            # every write still asks.
            result = PermissionResultAllow(updated_permissions=_LEAVE_PLAN_MODE)
        elif for_session and pending.standing:
            result = PermissionResultAllow(updated_permissions=pending.standing)
        else:
            result = PermissionResultAllow()
        pending.future.set_result(result)
        await self._emit(
            _resolved_frame(
                request_id,
                allow=allow,
                for_session=bool(allow and for_session and pending.standing),
                answers=answers,
            )
        )
        return True

    def _deny_pending(self) -> None:
        """Decline every request still waiting, so a stop never leaves the SDK hanging."""
        for pending in list(self._permissions.values()):
            if not pending.future.done():
                pending.future.set_result(
                    PermissionResultDeny(message="The operator stopped the turn.", interrupt=True)
                )

    # ---- Frame fan-out ---- #

    def subscribe(self) -> tuple[asyncio.Queue[dict[str, Any]], dict[str, Any]]:
        """Open a live feed of every frame this session emits, and the snapshot it starts from.

        The snapshot (a ``hello`` frame) says whether a turn is running, what it
        was asked, which permission requests are waiting and the last reading of
        the context window, so a tab that opens mid-turn can show the turn and
        answer its cards, and one that opens mid-session shows the meter. A
        response the CLI started on its own (no prompt behind it) is in flight too.
        """
        feed: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_SUBSCRIBER_BUFFER)
        self._subscribers.add(feed)
        hello = {
            "type": "hello",
            "seq": self._seq,
            "in_flight": self._turn_open or self._pending_turn or self._in_response,
            "prompt": self._turn_prompt,
            "pending": [pending.frame for pending in self._permissions.values()],
            "context": self._last_context,
        }
        return feed, hello

    def unsubscribe(self, feed: asyncio.Queue[dict[str, Any]]) -> None:
        """Close a feed opened by :meth:`subscribe`."""
        self._subscribers.discard(feed)

    def feed_open(self, feed: asyncio.Queue[dict[str, Any]]) -> bool:
        """False once *feed* fell too far behind and was dropped, or the session stopped."""
        return feed in self._subscribers and self.is_active

    async def _emit(self, frame: dict[str, Any]) -> None:
        """Number *frame*, hand it to every feed, and to the turn's consumer while it reads."""
        self._seq += 1
        frame = {**frame, "seq": self._seq}
        for feed in list(self._subscribers):
            try:
                feed.put_nowait(frame)
            except asyncio.QueueFull:
                self._subscribers.discard(feed)
        # Only a reading consumer gets the queue: one that is gone would leave
        # it to fill, and the pump would stall on a full queue.
        if self._turn_open and self._consumers and self._orphan_token is None:
            await self._queue.put(frame)

    # ---- Turn lifecycle ---- #

    async def _open_turn(self, prompt: str | None) -> None:
        """Route the responses from here on to the turn's consumer, and announce the turn.

        A prompt that runs a project skill by its slash command is followed by that
        skill's row: the CLI loads such a skill without a tool call, so the stream
        itself never says one ran.
        """
        self._turn_open = True
        self._turn_closed.clear()
        self._background.clear()
        self._agent_calls.clear()
        self._task_calls.clear()
        self._idle_wait = False
        await self._emit({"type": "turn_start", "prompt": prompt})
        skill = operator_skill(prompt, Path(self._cwd)) if prompt else None
        if skill is not None:
            for frame in operator_skill_frames(*skill, f"operator-skill-{uuid.uuid4().hex[:12]}"):
                await self._emit(frame)

    def _close_turn(self) -> None:
        """The open turn is over: stop routing to its consumer and free its guard if orphaned."""
        self._turn_open = False
        self._turn_prompt = None
        self._background.clear()
        self._idle_wait = False
        self.last_activity = time.monotonic()
        if self._orphan_token is not None:
            self.release_turn(self._orphan_token)
            self._orphan_token = None
        self._turn_closed.set()

    async def send_prompt(self, prompt: str) -> None:
        """Send a prompt; its response reaches the queue through :meth:`_pump`."""
        agent = self._connected_agent
        if agent is None:
            raise RuntimeError("Session not started")

        # A stopped turn may still be finishing (its quiesce waits for its
        # result); the new turn starts after it, never alongside it.
        quiesce = self._quiesce_task
        if quiesce is not None and not quiesce.done():
            with suppress(Exception):
                await asyncio.wait_for(
                    asyncio.shield(quiesce), timeout=_BACKGROUND_DRAIN_S + _QUIESCE_TIMEOUT_S
                )

        # Frames queued for a consumer that is gone belong to an earlier turn:
        # left queued, its result would end this turn at once.
        while not self._queue.empty():
            self._queue.get_nowait()
        self._turn_prompt = prompt
        self._unprompted = 0
        if self._in_response:
            # The CLI is mid-way through a response of its own and reads this
            # prompt after it, so this turn opens when that response ends.
            self._pending_turn = True
            self._turn_closed.clear()
        else:
            await self._open_turn(prompt)
        try:
            await agent.submit(prompt)
        except BaseException:
            self._pending_turn = False
            self._close_turn()
            raise

    def _track_background(self, event: AgentEvent) -> dict[str, Any] | None:
        """Follow the background agents this turn starts and the reports that end them.

        An agent runs in the background when the model asks for it or when
        its definition says so, and the input shows only the first; the call's
        result is what says it went async, so that is what marks one open.

        Returns an ``agent_done`` frame for the chat when *event* ends an agent
        call, naming the call by its ``tool_use_id``; ``None`` otherwise.
        """
        if isinstance(event, ToolUseEvent):
            if event.name in _AGENT_TOOLS:
                self._agent_calls.add(event.tool_use_id)
        elif isinstance(event, ToolResultEvent):
            if event.tool_use_id in self._agent_calls and _ASYNC_LAUNCH_MARK in (event.text or ""):
                self._background.add(event.tool_use_id)
        elif isinstance(event, SystemEvent):
            data = event.data
            task_id = data.get("task_id")
            if event.subtype == "task_started" and task_id and data.get("tool_use_id"):
                self._task_calls[task_id] = data["tool_use_id"]
            elif event.subtype == "task_notification" or (
                event.subtype == "task_updated"
                and (data.get("patch") or {}).get("status") in _TERMINAL_TASK_STATUSES
            ):
                call = data.get("tool_use_id") or self._task_calls.get(task_id or "")
                if call in self._background:
                    self._background.discard(call)
                    status = data.get("status") or (data.get("patch") or {}).get("status")
                    return {
                        "type": "system",
                        "subtype": "agent_done",
                        "tool_use_id": call,
                        "status": "stopped" if status == "killed" else status,
                        "summary": data.get("summary", ""),
                    }
        return None

    async def _stop_background(self) -> None:
        """Stop the background agents still open, so none reports after a Stop."""
        agent = self._connected_agent
        if agent is None:
            return
        for task_id, call in list(self._task_calls.items()):
            if call in self._background:
                try:
                    await agent.stop_task(task_id)
                except Exception:
                    logger.debug("Could not stop background task %s", task_id, exc_info=True)

    async def _pump(self) -> None:
        """Read the agent's stream for the life of the session and route every frame.

        One reader, always reading. A response to a submitted prompt goes to
        the turn's consumer and to every feed; a response the CLI produces on
        its own (a background agent's follow-up after a Stop, …) goes to the
        feeds alone, announced by a ``turn_start`` with no prompt. A turn that
        hands work to a background agent stays open past that response's result
        until the agent reports, and the next response is read into it.

        When the stream ends on its own (the child exited, or the stream
        failed), the session stops reading as active, so the pool and the
        hand-off door discard it and the next prompt starts a child that
        resumes the transcript. Every feed is told with an error frame
        (``AgentExited`` when the stream just closed), whether or not a turn is
        open, and an open turn ends with it.
        """
        agent = self._agent
        ended: dict[str, Any] = {
            "type": "error",
            "message": "The agent process ended.",
            "error_type": "AgentExited",
        }
        try:
            if agent is None:
                raise RuntimeError("Session not started")
            while True:
                saw_result = False
                async for event in agent.events():
                    if not self._in_response and _starts_response(event):
                        self._in_response = True
                        if not self._turn_open:
                            self._unprompted += 1
                            await self._emit({"type": "turn_start", "prompt": None})
                    done = self._track_background(event)
                    if done is not None:
                        await self._emit(done)
                    if _main_agent_active(event):
                        self._idle_wait = False
                    frame = _event_to_wire(event)
                    if frame is None:
                        continue
                    if frame["type"] == "result":
                        saw_result = True
                        if self._background and self._turn_open:
                            # The main agent is idle now, waiting on the report.
                            self._idle_wait = True
                            frame = {
                                "type": "system",
                                "subtype": "background_wait",
                                "tasks": len(self._background),
                            }
                        # Read before the result goes out: the turn's consumer releases
                        # the turn on its result, and the reading must already be on the
                        # stream. The response stays open across the read, so a prompt
                        # that arrives meanwhile waits behind this result (a pending
                        # turn) instead of opening a turn this result would then end.
                        await self._emit_context(agent)
                        self._in_response = False
                    await self._emit(frame)
                    if is_terminal_event(frame):
                        await self._end_response()
                if not saw_result:
                    break  # the stream closed without a result: the child is gone
                if self._unprompted > _MAX_UNPROMPTED_RESPONSES:
                    logger.error(
                        "Chat %s: %d responses with no prompt; no longer reading its stream",
                        self._session_key,
                        self._unprompted,
                    )
                    break
                # Give the loop back between responses, whatever the stream does.
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except AgentRunError as exc:
            ended = {"type": "error", "message": str(exc), "error_type": exc.error_type}
        except Exception as exc:
            ended = {
                "type": "error",
                "message": f"Unexpected error: {exc}",
                "error_type": type(exc).__name__,
            }
        # Not active from here: a prompt that arrives now gets a new child.
        self._stream_ended = True
        self._in_response = False
        in_turn = self._turn_open or self._pending_turn
        if in_turn:
            self._pending_turn = False
            self._turn_open = True
        await self._emit(ended)
        if in_turn:
            self._close_turn()

    async def _emit_context(self, agent: AgentSession) -> None:
        """Send the context window's reading as a response ends, when the CLI gives one.

        The reading is best-effort: one that does not come in time, or is not a
        usage record, costs the meter its update and nothing more.
        """
        try:
            usage = await asyncio.wait_for(agent.context_usage(), timeout=_CONTEXT_READ_S)
        except Exception:
            logger.debug(
                "The context window of chat %s was not read", self._session_key, exc_info=True
            )
            return
        frame = _context_frame(usage)
        if frame is None:
            return
        self._last_context = frame
        await self._emit(frame)

    async def _end_response(self) -> None:
        """A terminal frame went out: close the open turn, and open the one waiting behind it."""
        self._in_response = False
        self.last_activity = time.monotonic()
        if self._turn_open:
            self._close_turn()
        if self._pending_turn:
            self._pending_turn = False
            await self._open_turn(self._turn_prompt)

    async def run_turn(
        self,
        prompt: str,
        token: int,
        *,
        timeout_s: float,
        heartbeat_s: float = 15.0,
    ):
        """Run one turn as an async generator: the turn's consumer.

        Sends *prompt*, then yields every frame of the turn. While no frame
        arrives for ``heartbeat_s``, a ``{"type": "heartbeat"}`` marker is
        yielded so a streaming consumer can keep its transport alive (buffered
        consumers just skip it — the marker never carries payload). The
        deadline is SDK-silence based: it resets on every frame, and does not
        run while an approval or a background agent is open or the turn waits
        for a response of the CLI's own to end; a lapse of ``timeout_s``
        raises :class:`TurnSilenceTimeout`.

        Owns the turn from just after :meth:`acquire_turn` (done by the caller,
        which needs the 409 mapping before a response starts) through release:

        * terminal frame (see :func:`is_terminal_event`) — the guard is
          released before the frame is yielded, and the generator then returns;
        * the consumer goes away (``GeneratorExit``, cancellation) after the
          prompt was sent — the turn runs on: its guard is held as an orphan
          and released by the pump when the turn ends, and its frames still
          reach every feed, so a reloaded or second tab can follow it;
        * anything else (silence timeout, an error) — a detached quiesce is
          spawned FIRST, then the guard is released in a nested ``finally``.

        Callers must guard the never-started case (a generator that is closed
        before its first ``__anext__`` never runs this body) with
        :meth:`end_consumer`.
        """
        terminal_seen = False
        submitted = False
        consumer_gone = False
        self._consumers.add(token)
        loop = asyncio.get_running_loop()
        try:
            await self.send_prompt(prompt)
            submitted = True

            deadline = loop.time() + timeout_s
            while True:
                if self._permissions or self._background or self._pending_turn:
                    # An approval waits on the operator, a background agent on
                    # its own work, a pending turn on the CLI's own response —
                    # not on the SDK: the silence deadline does not run.
                    deadline = loop.time() + timeout_s
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

                # Flag and release BEFORE yielding. The consumer may hand the
                # terminal event to a client that sends its next prompt before
                # this generator resumes, and that prompt must find the turn
                # free. A consumer that stops right here (GeneratorExit at this
                # yield) still takes the release-only path, never a quiesce;
                # the release in ``finally`` is then an owner-checked no-op.
                if is_terminal_event(event):
                    terminal_seen = True
                    self.release_turn(token)
                yield event
                if terminal_seen:
                    return
        except (GeneratorExit, asyncio.CancelledError):
            consumer_gone = True
            raise
        finally:
            self._consumers.discard(token)
            if terminal_seen:
                self.release_turn(token)
            elif consumer_gone and submitted and (self._turn_open or self._pending_turn):
                # The tab went away; the turn does not. The pump frees the
                # guard when the turn ends (see _close_turn).
                self._orphan_token = token
            elif not submitted:
                self.release_turn(token)
            else:
                # Spawn the detached quiesce FIRST, then release in a nested
                # finally so release still happens if spawn_quiesce is hit by a
                # re-delivered cancellation. Never await the quiesce here.
                try:
                    self.spawn_quiesce()
                finally:
                    self.release_turn(token)

    def end_consumer(self, token: int) -> None:
        """The route that acquired *token* is done with it.

        Frees a guard whose :meth:`run_turn` never ran (the consumer closed
        before it started), and leaves alone one that run_turn released or
        handed to a running turn as an orphan.
        """
        if token in self._consumers or token == self._orphan_token:
            return
        self.release_turn(token)

    async def interrupt(self) -> None:
        """Signal-only interrupt: forward the runner's interrupt if connected.

        Never touches the turn guard — the pump closes the turn when the
        interrupt's result arrives, and whoever holds the guard releases it.
        """
        self._deny_pending()
        idle = self._idle_wait
        await self._stop_background()
        agent = self._connected_agent
        # Waiting on a background agent the main agent is idle: an interrupt
        # there answers with a result of its own that would end the turn before
        # the stopped agent's follow-up arrives, so only the agent stops. With
        # nothing running there is nothing to interrupt either.
        working = self._turn_open or self._pending_turn or self._in_response
        if agent is not None and working and not idle:
            await agent.interrupt()

    async def cancel(self) -> None:
        """Interrupt the in-flight turn and wait for it to end.

        No-op short-circuit when nothing is running, so an idle cancel never
        hangs. Otherwise:

        1. Stop any background agent, then await the runner's interrupt so the
           CLI stops generating — unless the main agent is idle waiting on a
           background agent; then only that agent stops (see :meth:`interrupt`).
        2. Wait for the turn to end, bounded at ``_QUIESCE_TIMEOUT_S`` (or
           ``_BACKGROUND_DRAIN_S`` when the CLI answers the stop with a model
           turn of its own).
        3. A turn that has not ended by then is abandoned: its consumer is
           told, and whatever it still says reaches the feeds alone.
        """
        self._deny_pending()
        if not (self._turn_open or self._pending_turn or self._in_response):
            return

        idle = self._idle_wait
        await self._stop_background()
        agent = self._connected_agent
        if agent is not None and not idle:
            with suppress(Exception):
                await agent.interrupt()

        if not (self._turn_open or self._pending_turn):
            return
        drain_s = _BACKGROUND_DRAIN_S if idle else _QUIESCE_TIMEOUT_S
        try:
            await asyncio.wait_for(self._turn_closed.wait(), timeout=drain_s)
        except TimeoutError:
            if self._turn_open or self._pending_turn:
                self._pending_turn = False
                self._turn_open = True
                await self._emit(
                    {
                        "type": "error",
                        "message": "The turn did not stop in time and was abandoned.",
                        "error_type": "TurnAbandoned",
                    }
                )
                self._close_turn()

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
        """End the child and every process descending from it, cancelling any in-flight response.

        That is the runner closed, the groups the agent split off for its
        commands ended, and its descendants in the server's own group ended
        one by one; never the server or the server's group. When started
        processes ignore SIGTERM this blocks a worker thread, off the loop,
        for up to about ten more seconds. What it ended is in the log line and
        in :attr:`tree_ending`.

        The process table is read while the child is alive, and a failed read
        is retried while it is, for up to ``process_tree.READ_WAIT_S``. When it
        still cannot be read, nothing is signalled: what the child started is
        left running, a warning names the child, and :attr:`tree_ending` says
        ``read=False``. What a later read could not re-identify is left
        unsignalled and named in :attr:`tree_ending`.

        The runner keeps the child's process handle after it closes, so a
        caller that has to wait for the child to be *gone* — not merely asked
        to leave — can still read :attr:`process_exited` afterwards. A second
        ``stop()`` does not erase the first one's answer.

        A child still running once the runner is closed is sent SIGKILL.
        Closing the runner already escalates from stdin EOF through SIGTERM to
        SIGKILL, each with a bounded wait, so this only matters for a child
        that outlived that escalation — and it is what makes a second
        ``stop()`` on a session whose runner is already closed kill the child
        again rather than merely re-read its status.
        """
        await self.cancel()
        pump = self._pump_task
        if pump is not None and not pump.done():
            pump.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pump

        # Read while the child is alive and before the runner closes: a process
        # the agent started in a session of its own keeps its parent link only
        # until the agent exits.
        pid = self.pid
        tree = (
            await asyncio.to_thread(_read_tree, pid, lambda: self.process_exited is False)
            if pid is not None
            else None
        )

        scope = self._agent_scope
        if scope is not None:
            self._agent_scope = None
            with suppress(Exception):
                await scope.aclose()

        if self._agent is not None:
            try:
                self._agent.kill_child()
            except Exception:
                logger.warning(
                    "OperatorSession could not signal its lingering child", exc_info=True
                )
        if pid is not None and tree is not None:
            if tree.groups or tree.members:
                self._tree_ending = await asyncio.to_thread(_end_tree, pid, tree)
            else:
                self._tree_ending = TreeEnding(read=tree.read)
        self._started = False
        logger.info("OperatorSession stopped")

    def started_commands(self) -> list[ProcessGroup]:
        """The commands the agent started in groups of their own and still running.

        What ending this session would also end, beyond the agent itself and
        its helpers in the server's group. Empty once the child has exited.
        Blocking (it reads the process table): call it off the loop.

        Raises:
            process_tree.ProcessTableUnreadable: The process table could not
                be read while the child was alive: what it started is unknown.
        """
        pid = self.pid
        if pid is None:
            return []
        return process_tree.started_groups(pid, lambda: self.process_exited is False)

    @property
    def _connected_agent(self) -> AgentSession | None:
        """The runner while its scope is open, else ``None``.

        The handle outlives the scope (:attr:`pid` and :attr:`process_exited`
        read it after :meth:`stop`), so a call that reaches the runner keys on
        the scope; :meth:`start` sets the two together.
        """
        return self._agent if self._agent_scope is not None else None

    @property
    def is_active(self) -> bool:
        return self._started and self._connected_agent is not None and not self._stream_ended

    @property
    def pid(self) -> int | None:
        """The pid of this session's child process, or ``None`` when none is running.

        The chat child is what holds a session on the chat surface, the way
        a PTY holds one on the terminal surface, so a caller addressing
        something *to* the process behind a session key has to be able to ask
        which process that is. The runner keeps the handle after it closes, so
        a session whose runner has been closed still answers while its child
        is alive.

        ``None`` once the child has a return code, and ``None`` for a session
        that never started or never captured a handle: a pid nothing is
        running under names no process, and returning one would invite a
        caller to address it.
        """
        return self._agent.pid if self._agent is not None else None

    @property
    def tree_ending(self) -> TreeEnding | None:
        """What the last :meth:`stop` that found a live child ended of what it started.

        ``None`` until a stop has found one. A later stop that finds no child
        keeps the answer, as :attr:`process_exited` keeps the child's.
        """
        return self._tree_ending

    @property
    def process_exited(self) -> bool | None:
        """Whether this session's child process has actually exited.

        ``True`` once the child has a return code, ``False`` while it is still
        running, and ``None`` when no process handle was ever captured — the
        session was never started, or runs against a client that exposes none.

        ``None`` is deliberately not ``False``: a caller sequencing a handover
        on "the outgoing child is gone" must be able to tell *not exited* from
        *nothing to observe*, and collapsing the two would make an unanswerable
        session look like a live one forever.
        """
        return self._agent.child_exited if self._agent is not None else None


class OperatorRegistry:
    """Manages operator sessions and composes the operator-chat pool.

    * ``_sessions`` — persistent ``/ws/operator`` sessions keyed by a
      connection-derived id. Uncapped, single-writer per key.
    * ``chats`` — a :class:`ChatSessionPool` of operator-chat sessions
      (LRU-capped, idle-reaped). The ``*_chat_session`` methods below are a
      thin facade over it for route/app callers.
    """

    def __init__(self, chat_max_sessions: int = 5, chat_idle_seconds: float = 900.0) -> None:
        self._sessions: dict[str, OperatorSession] = {}
        # The factory resolves OperatorSession by name at call time, so tests
        # patching this module's OperatorSession still intercept creation. The
        # pool's key IS the session key: a chat child is pooled, audited and
        # given telemetry under the shared session key, never under an id of
        # its own. The transcript to resume is not bound here — it is decided
        # per launch and reaches the child through ``start``.
        self.chats = ChatSessionPool(
            factory=lambda cwd, env, session_key: OperatorSession(
                cwd=cwd, env=env, session_key=session_key
            ),
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

    # ---- Chat pool facade (see ChatSessionPool) ---- #

    async def get_or_create_chat_session(
        self,
        chat_id: str,
        cwd: str,
        env: dict[str, str] | Callable[[], dict[str, str] | None] | None = None,
        *,
        resume_id: str | None = None,
    ) -> tuple[OperatorSession, bool]:
        """Pass-through to :meth:`ChatSessionPool.get_or_create`.

        *env* may be a mapping or a zero-arg builder; the builder form is what
        keeps a caller's environment read atomic with the pool's registration
        of the creation (``routes/chat.py`` relies on it for the runtime
        posture).

        *resume_id* is the transcript the child should continue — the caller's
        answer to "what is this key's conversation right now", not something
        the pool can work out for itself. Omitted, the child starts a
        conversation of its own under *chat_id*. A live child built from a
        different transcript is rebuilt, exactly as an environment change
        rebuilds it.
        """
        return await self.chats.get_or_create(chat_id, cwd, env, resume_id=resume_id)

    def get_chat_session(self, chat_id: str) -> OperatorSession | None:
        return self.chats.get(chat_id)

    def has_chat_key(self, chat_id: str) -> bool:
        """Pass-through to :meth:`ChatSessionPool.has_key`.

        The *addressability* question, as opposed to the "what did we
        terminate" question :meth:`get_chat_session` answers: this one also
        says yes while a creation is still inside ``start()``, which is the
        window a posture flip has to be able to name.
        """
        return self.chats.has_key(chat_id)

    async def terminate_chat_session(self, chat_id: str) -> OperatorSession | None:
        """Pass-through to :meth:`ChatSessionPool.terminate`.

        Hands back the torn-down session so a caller can read
        :attr:`OperatorSession.process_exited` on it; ``None`` when the key held
        nothing poppable.
        """
        return await self.chats.terminate(chat_id)

    async def reap_idle_chat_sessions(self) -> int:
        return await self.chats.reap_idle()
