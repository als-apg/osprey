"""The agent engine: a read-only headless agent judges a prompt against the built project.

The run goes straight through :func:`osprey.agent_runner.stream_query`, with the
deployment's own settings layer off (``setting_sources=[]``) and only the MCP
servers it names, so no project hook runs. Read-only holds at four layers:

* ``disallowed_tools`` is the headless read-only floor, which binds under
  ``bypassPermissions``;
* a deny-only PreToolUse hook and a ``can_use_tool`` backstop hold the run to
  the watchdog's read tools plus ``Read``/``Glob``/``Grep``, and scope those three
  to the watchdog's references, because ``bypassPermissions`` ignores path scope;
* its MCP servers run with ``OSPREY_EXECUTION_MODE=readonly``, under which the
  servers refuse a write themselves;
* its environment blanks the keys that would tie it to a session: a judge is
  never held, and never reaches a panel.

Every tool call refused on the way is reported in ``refused_tool_calls``.
:meth:`AgentEngine.judge` never raises: a run that times out, hits its budget,
fails, or answers with no readable verdict is ``unavailable``. Neither the session
it read nor the answer is logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import time
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from osprey.agent_runner import (
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolUseEvent,
    build_agent_options,
    read_only_disallowed_tools,
    stream_query,
)
from osprey.agent_runner.launcher import RENDERED_MCP_CONFIG
from osprey.agent_runner.runner import WATCHDOG_SKIP_ENV_VAR
from osprey.agent_runner.tool_policy import (
    PermissionResultDeny,
    make_backstop,
    make_pretooluse_hook,
)
from osprey.services.watchdog.config import WatchdogDefinition
from osprey.services.watchdog.history import (
    SessionNodes,
    SessionView,
    as_session_nodes,
    build_session_view_from_nodes,
    rough_tokens,
)
from osprey.services.watchdog.verdict import (
    UNBOUND_ID,
    ErrorType,
    InputDigest,
    RefusedToolCall,
    Verdict,
)
from osprey_connectors.session_hold import HOLD_DIR_ENV_VAR

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_VIEW_TOKENS", "AgentEngine"]

#: The size of the session view a judge reads, by :func:`rough_tokens`.
DEFAULT_VIEW_TOKENS = 60_000

#: The file tools a judge always has, scoped to its references.
_FILE_TOOLS = ("Read", "Glob", "Grep")

#: The servers a judge's read tools come from, beside any its ``tools`` name.
_BASE_SERVERS = ("controls", "osprey_facility_knowledge")

#: The result subtype of a run stopped at its budget.
_BUDGET_SUBTYPE = "error_max_budget_usd"

#: How long a cancelled run gets to unwind its agent process before the judge
#: answers without it.
_TEARDOWN_GRACE_S = 5.0

#: Stamps of one executor sandbox child. A judge resolves its target from the
#: owner record instead.
_CONTROL_TARGET_PREFIX = "OSPREY_CONTROL_TARGET"

#: The environment every process of the run gets. The SDK overlays the run's env
#: on its parent's and cannot delete a key, so a key that must not reach the run
#: is set blank.
_RUN_ENV_OVERLAY: Mapping[str, str] = {
    "OSPREY_EXECUTION_MODE": "readonly",
    "OSPREY_SESSION_ID": "",
    "OSPREY_POSTURE_SESSION": "",
    HOLD_DIR_ENV_VAR: "",
    "OSPREY_PANEL_TOKEN": "",
    WATCHDOG_SKIP_ENV_VAR: "1",
}

_REFUSED_DISALLOWED = "disallowed"
_REFUSED_NOT_ALLOWED = "not_allowed"
_REFUSED_OUTSIDE_REFERENCES = "outside_references"


class _Answer(BaseModel):
    """The verdict document a judge answers with."""

    model_config = ConfigDict(extra="forbid")

    flagged: StrictBool
    explanation: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)


def _reason(answer: _Answer) -> str:
    """The verdict's reason: the explanation, with what it cites."""
    if not answer.sources:
        return answer.explanation
    return f"{answer.explanation} (Sources: {', '.join(answer.sources)})"


def _parse_answer(text: str) -> _Answer | None:
    """The last verdict document in *text*, prose and fences around it ignored."""
    decoder = json.JSONDecoder()
    found: _Answer | None = None
    position = text.find("{")
    while position != -1:
        try:
            document, end = decoder.raw_decode(text, position)
        except ValueError:
            position = text.find("{", position + 1)
            continue
        with contextlib.suppress(ValidationError):
            found = _Answer.model_validate(document)
        position = text.find("{", end)
    return found


def _named_by(tool: str, entries: Iterable[str]) -> bool:
    """Whether *tool* falls under a tool list: by name, whole server, or trailing wildcard."""
    for entry in entries:
        rule = entry.replace(".*", "*")
        if tool == rule or tool.startswith(f"{rule}__"):
            return True
        if rule.endswith("*") and tool.startswith(rule[:-1]):
            return True
    return False


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class _ReferenceScope:
    """What the file tools of a run may touch: its references, and nothing else."""

    def __init__(self, references: Iterable[Path], cwd: Path) -> None:
        self._cwd = cwd
        self._files = frozenset(reference.resolve() for reference in references)
        parents = {str(file.parent) for file in self._files}
        self._root = Path(os.path.commonpath(sorted(parents))) if parents else None

    def _resolve(self, value: str) -> Path:
        return (self._cwd / value).resolve()

    def _searchable(self, value: Any) -> bool:
        if not isinstance(value, str) or not value or self._root is None:
            return False
        return _within(self._resolve(value), self._root)

    def allows(self, tool: str, tool_input: Mapping[str, Any]) -> bool:
        """Whether this file-tool call stays inside the references."""
        if tool == "Read":
            value = tool_input.get("file_path")
            return isinstance(value, str) and bool(value) and self._resolve(value) in self._files
        pattern = tool_input.get("pattern" if tool == "Glob" else "glob")
        if pattern is not None and (
            not isinstance(pattern, str)
            or Path(pattern).is_absolute()
            or ".." in Path(pattern).parts
        ):
            return False
        return self._searchable(tool_input.get("path"))


@dataclass
class _RunState:
    """What a run has produced so far; a timed-out run still reports it."""

    refused: list[RefusedToolCall] = field(default_factory=list)
    refused_ids: set[str] = field(default_factory=set)
    tool_names: dict[str, str] = field(default_factory=dict)
    text: list[str] = field(default_factory=list)
    result: ResultEvent | None = None

    def refuse(self, tool: str, reason: str, tool_use_id: str | None) -> None:
        """Note one refused call, once however many layers report it."""
        if tool_use_id:
            if tool_use_id in self.refused_ids:
                return
            self.refused_ids.add(tool_use_id)
        self.refused.append(RefusedToolCall(tool=tool, reason=reason))


class _ToolGate:
    """The deny-only callbacks of one run, reporting each refusal to its state."""

    def __init__(
        self,
        read_tools: Iterable[str],
        disallowed: Collection[str],
        scope: _ReferenceScope,
        state: _RunState,
    ) -> None:
        self.allowed = [*(tool.replace(".*", "*") for tool in read_tools), *_FILE_TOOLS]
        self._disallowed = disallowed
        self._scope = scope
        self._state = state
        self._exact = frozenset(self.allowed)

    def _admitted(self, tool: str) -> tuple[str, ...]:
        """The allow set for one call, a wildcard entry spelled out as the tool it covers."""
        extra = (tool,) if tool not in self._exact and _named_by(tool, self.allowed) else ()
        return (*self.allowed, *extra)

    def refusal(self, tool: str, tool_input: Mapping[str, Any]) -> str | None:
        """Why this call is refused, ``None`` when nothing here refuses it."""
        if _named_by(tool, self._disallowed):
            return _REFUSED_DISALLOWED
        if tool in _FILE_TOOLS and not self._scope.allows(tool, tool_input):
            return _REFUSED_OUTSIDE_REFERENCES
        return None

    def refused_result(self, tool: str) -> str | None:
        """Why a failed tool result counts as a refusal, ``None`` for a tool's own failure."""
        if _named_by(tool, self._disallowed):
            return _REFUSED_DISALLOWED
        return None if _named_by(tool, self.allowed) else _REFUSED_NOT_ALLOWED

    async def hook(
        self, input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        """PreToolUse: deny what is out of bounds, and decide nothing otherwise."""
        tool = str(input_data.get("tool_name") or "")
        try:
            tool_input = input_data.get("tool_input")
            reason = self.refusal(tool, tool_input if isinstance(tool_input, Mapping) else {})
        except Exception:
            logger.exception("Watchdog tool gate failed; denying the call")
            reason = _REFUSED_NOT_ALLOWED
        if reason is None:
            policy = make_pretooluse_hook(self._admitted(tool), {}, self._disallowed)
            decision = await policy(input_data, tool_use_id, context)
            if decision:
                self._state.refuse(tool, _REFUSED_NOT_ALLOWED, tool_use_id)
            return decision
        self._state.refuse(tool, reason, tool_use_id)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"A watchdog run may not use {tool!r}: {reason}",
            }
        }

    async def backstop(self, tool: str, tool_input: dict[str, Any], context: Any) -> Any:
        """``can_use_tool``: the same bounds, under the hook."""
        tool_use_id = getattr(context, "tool_use_id", None)
        reason = self.refusal(tool, tool_input)
        if reason is not None:
            self._state.refuse(tool, reason, tool_use_id)
            return PermissionResultDeny(message=f"A watchdog run may not use {tool!r}: {reason}")
        policy = make_backstop(self._admitted(tool), {}, self._disallowed)
        decision = await policy(tool, tool_input, context)
        if isinstance(decision, PermissionResultDeny):
            self._state.refuse(tool, _REFUSED_NOT_ALLOWED, tool_use_id)
        return decision


def _named_servers(tools: Iterable[str]) -> tuple[set[str], set[str]]:
    """The servers a tool list names exactly, and the prefixes its wildcards name."""
    exact: set[str] = set()
    prefixes: set[str] = set()
    for entry in tools:
        if not entry.startswith("mcp__"):
            continue
        name = entry[len("mcp__") :].replace(".*", "*").split("__", 1)[0]
        if name.endswith("*"):
            prefixes.add(name[:-1])
        else:
            exact.add(name)
    return exact, prefixes


def _run_env(env: Mapping[str, str]) -> dict[str, str]:
    """*env* under the overlay, with no control-target stamp."""
    kept = {key: value for key, value in env.items() if not key.startswith(_CONTROL_TARGET_PREFIX)}
    return {**kept, **_RUN_ENV_OVERLAY}


def _mcp_servers(build_dir: Path, tools: Iterable[str]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The run's MCP servers from the render, and the ones it cannot do without.

    Raises:
        OSError: When the render's server file cannot be read.
        ValueError: When it is malformed, or a server the tools name is not in it:
            a judge without its tools cannot read what it was written to read.
    """
    document = json.loads((build_dir / RENDERED_MCP_CONFIG).read_text(encoding="utf-8"))
    declared = document.get("mcpServers") if isinstance(document, dict) else None
    if not isinstance(declared, dict):
        raise ValueError(f"{RENDERED_MCP_CONFIG} declares no mcpServers")
    exact, prefixes = _named_servers(tools)
    missing = exact - set(declared)
    if missing:
        raise ValueError(f"tools name servers the build lacks: {', '.join(sorted(missing))}")
    required = exact | {
        name for name in declared if any(name.startswith(prefix) for prefix in prefixes)
    }
    servers: dict[str, Any] = {}
    for name in sorted(required | (set(_BASE_SERVERS) & set(declared))):
        config = dict(declared[name])
        config["env"] = _run_env(config.get("env") or {})
        servers[name] = config
    return servers, tuple(sorted(required))


def _prompt(defn: WatchdogDefinition, view: SessionView) -> str:
    """The run's one user turn: instructions, references, the session, the answer's shape."""
    references = "\n".join(f"- {path}" for path in defn.references) or "(none)"
    return (
        "You are a background safety watchdog. You observe an agent session and judge the "
        "operator's latest prompt, the last USER entry below. You only read: nothing you do "
        "changes the session or the machine, and any write you attempt is refused and "
        "recorded.\n\n"
        f"## Your instructions\n\n{defn.instructions.strip()}\n\n"
        "## Your references\n\n"
        "Read them with Read, or search them with Glob or Grep given a `path` inside them. "
        f"No other file is open to you.\n\n{references}\n\n"
        f"## The session\n\n{view.text}\n\n"
        "## Your answer\n\n"
        "End with one JSON object and nothing after it: "
        '{"flagged": true or false, "explanation": "why, for the operator", '
        '"sources": ["what you read that supports it"]}'
    )


class AgentEngine:
    """Judges prompts with a read-only agent run in one built project."""

    def __init__(self, build_dir: Path, *, view_tokens: int = DEFAULT_VIEW_TOKENS) -> None:
        self._build_dir = Path(build_dir)
        self._view_tokens = view_tokens

    def _run_options(
        self, defn: WatchdogDefinition, gate_state: _RunState
    ) -> tuple[_ToolGate, dict[str, Any]]:
        """The run's gate and its keywords. Blocking: it reads the build and may start a proxy."""
        routed = build_agent_options(
            self._build_dir, disallowed_tools=[], model=defn.model, setting_sources=[]
        )
        disallowed = read_only_disallowed_tools(self._build_dir)
        servers, required = _mcp_servers(self._build_dir, defn.tools)
        gate = _ToolGate(
            defn.tools,
            disallowed,
            _ReferenceScope(defn.references, self._build_dir),
            gate_state,
        )
        return gate, {
            "disallowed_tools": disallowed,
            "allowed_tools": gate.allowed,
            "pre_tool_use_hooks": [gate.hook],
            "can_use_tool": gate.backstop,
            "permission_mode": "bypassPermissions",
            "setting_sources": [],
            "mcp_servers": servers,
            "await_mcp_servers": required,
            "require_mcp_servers": required,
            "env": _run_env(routed.env),
            "model": routed.model,
            "max_budget_usd": defn.max_cost_usd,
        }

    async def _run(self, defn: WatchdogDefinition, prompt: str, state: _RunState) -> None:
        """Drive one run to its end, folding its events into *state*."""
        gate, options = await asyncio.to_thread(self._run_options, defn, state)
        async with contextlib.aclosing(stream_query(self._build_dir, prompt, **options)) as events:
            async for event in events:
                if isinstance(event, TextEvent):
                    if event.parent_tool_use_id is None:
                        state.text.append(event.text)
                elif isinstance(event, ToolUseEvent):
                    state.tool_names[event.tool_use_id] = event.name
                elif isinstance(event, ToolResultEvent):
                    name = state.tool_names.get(event.tool_use_id)
                    if event.is_error and name is not None:
                        reason = gate.refused_result(name)
                        if reason is not None:
                            state.refuse(name, reason, event.tool_use_id)
                elif isinstance(event, ResultEvent):
                    state.result = event

    async def _bounded_run(
        self, defn: WatchdogDefinition, prompt: str, state: _RunState, timeout: float
    ) -> ErrorType | None:
        """Run within *timeout* seconds of wall clock; the error type if it did not finish.

        The bound holds even when the agent process does not unwind: past the
        grace period the judge answers and leaves the cancelled task behind.
        """
        task = asyncio.create_task(self._run(defn, prompt, state))
        task.add_done_callback(lambda done: done.cancelled() or done.exception())
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
        finally:
            if not task.done():
                task.cancel()
        if not done:
            await asyncio.wait({task}, timeout=_TEARDOWN_GRACE_S)
            return "timeout"
        failure = task.exception()
        if failure is not None:
            logger.warning("Watchdog %r agent run failed: %s", defn.name, type(failure).__name__)
            return "budget" if _at_budget(state) else "run_error"
        return None

    async def judge(
        self,
        defn: WatchdogDefinition,
        nodes: SessionNodes | Sequence[Mapping[str, Any]],
        prompt: str,
        *,
        deadline: float,
        session_key: str | None = None,
        record_session_id: str = UNBOUND_ID,
        submission_id: str = UNBOUND_ID,
    ) -> Verdict:
        """Judge *prompt* against the session in *nodes*. Never raises.

        Args:
            defn: The watchdog.
            nodes: The session's history: one read's answer, or its bare nodes.
            prompt: The operator prompt to judge.
            deadline: A :func:`time.monotonic` value no run outlasts, whatever
                the watchdog's own ``timeout_s``.
            session_key: The web terminal pool key the verdict belongs to.
            record_session_id: The Claude record session it belongs to.
            submission_id: The id of the prompt's submission.

        Returns:
            The verdict, ``unavailable`` when the run gave no readable answer.
        """
        started = time.monotonic()
        state = _RunState()
        read = as_session_nodes(nodes)
        digest = InputDigest(
            drain_timed_out=read.drain_timed_out, record_available=read.record_available
        )

        def verdict(
            *,
            error: ErrorType | None = None,
            answer: _Answer | None = None,
        ) -> Verdict:
            cost = state.result.total_cost_usd if state.result is not None else None
            return Verdict(
                watchdog=defn.name,
                engine="agent",
                session_key=session_key,
                record_session_id=record_session_id,
                submission_id=submission_id,
                status="unavailable" if answer is None else "ok",
                flagged=answer.flagged if answer is not None else False,
                reason=_reason(answer) if answer is not None else None,
                input_digest=digest,
                refused_tool_calls=list(state.refused),
                latency_s=time.monotonic() - started,
                cost_usd=cost,
                error_type=error,
            )

        try:
            view = build_session_view_from_nodes(
                read.nodes, prompt, max_tokens=self._view_tokens, count_tokens=rough_tokens
            )
            digest = digest.model_copy(
                update={
                    "sha256": hashlib.sha256(view.text.encode("utf-8")).hexdigest(),
                    "entries_kept": view.entries_kept,
                    "entries_cut": view.entries_cut,
                    "tokens_estimate": view.tokens_estimate,
                }
            )
            remaining = min(defn.timeout_s, deadline - started)
            if remaining <= 0:
                return verdict(error="timeout")
            error = await self._bounded_run(defn, _prompt(defn, view), state, remaining)
            if error is not None:
                return verdict(error=error)
            if _at_budget(state):
                return verdict(error="budget")
            if state.result is None or state.result.is_error:
                return verdict(error="run_error")
            answer = _parse_answer(state.result.result or "\n".join(state.text))
            if answer is None:
                return verdict(error="parse_error")
            return verdict(answer=answer)
        except Exception as exc:
            logger.warning("Watchdog %r agent run failed: %s", defn.name, type(exc).__name__)
            return verdict(error="run_error")


def _at_budget(state: _RunState) -> bool:
    return state.result is not None and state.result.subtype == _BUDGET_SUBTYPE
