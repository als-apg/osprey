"""Multi-turn agent conversations over the Claude Agent SDK.

``osprey.agent_runner.runner.run_query`` sends one prompt and closes the
session.  Some callers need to hold a session open and choose each message from
the agent's previous reply — an operator-style evaluation, an adaptive
red-team loop, or a benchmark that branches on what the agent answered.

:class:`AgentSession` provides that: a handle over one persistent SDK client
that sends a single user turn at a time, accumulating transcript and cost and
enforcing a session-wide budget.  Open one for a project with
:func:`agent_session`; for the simple case of a fixed, pre-decided script use
:func:`run_turns`.  Both reuse the same option building, readiness barrier and
event records as ``run_query`` (via ``primitives.build_agent_options``,
``primitives._ready_mcp`` and ``events.translate_message``).

Cost accounting: the SDK reports ``ResultMessage.total_cost_usd`` as the
cumulative session cost to date — the field is a running ``total_`` and the SDK
documents its usage counters as cumulative for the session.  A turn's
incremental cost is therefore the delta from the previous turn's total, and the
session total is the latest turn's ``total_cost_usd``.  The session budget is
enforced from that cumulative figure and is also passed to the SDK as a
backstop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Collection, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_agent_sdk import (
        CanUseTool,
        HookCallback,
        McpServerConfig,
        PermissionMode,
        SettingSource,
    )
    from claude_agent_sdk.types import SystemPromptFile, SystemPromptPreset

# SDK import — keep module importable even when SDK is absent.
try:
    from claude_agent_sdk import ClaudeSDKClient, ClaudeSDKError

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

from osprey.agent_runner.errors import from_sdk_error
from osprey.agent_runner.events import (
    AgentEvent,
    ResultEvent,
    SystemEvent,
    translate_message,
)
from osprey.agent_runner.primitives import (
    ToolTrace,
    _ready_mcp,
    _record_event,
    _send_turn,
    build_agent_options,
)


class AgentSessionBudgetExceeded(RuntimeError):
    """Raised when :meth:`AgentSession.send` is called after the session's
    cumulative cost has reached ``max_budget_usd``.  The offending turn is
    refused *before* it is sent, so it costs nothing."""


@dataclass
class TurnResult:
    """The outcome of a single conversation turn.

    ``text_blocks``/``tool_traces``/``system_messages``/``result`` cover only
    this turn; the running view lives on :class:`AgentSession`. Every field
    holds plain data.
    """

    index: int
    text_blocks: list[str] = field(default_factory=list)
    tool_traces: list[ToolTrace] = field(default_factory=list)
    system_messages: list[SystemEvent] = field(default_factory=list)
    result: ResultEvent | None = None
    #: This turn's incremental cost (delta from the previous turn), or ``None``
    #: when the SDK did not report a cost for the turn.
    cost_usd: float | None = None
    #: Cumulative cost as of this turn (the SDK's ``total_cost_usd``).
    cumulative_cost_usd: float | None = None

    @property
    def text(self) -> str:
        """This turn's assistant text, concatenated."""
        return "".join(self.text_blocks)

    @property
    def tool_names(self) -> list[str]:
        """Names of the tools called during this turn, in order."""
        return [t.name for t in self.tool_traces]


class AgentSession:
    """A live multi-turn conversation driven one turn at a time.

    Construct via :func:`agent_session`, which wires provider routing and waits
    for MCP servers to register; then drive turns with :meth:`send`, or with
    :meth:`submit` followed by :meth:`events` to read a turn as it streams.
    The SDK client is injected rather than built here, so the turn/cost/budget
    logic is unit-testable with a fake client and no live model.

    The agent child process stays observable through :attr:`pid`,
    :attr:`child_exited` and :meth:`kill_child`, also after the client is
    closed.

    Attributes:
        turns: Completed :class:`TurnResult` records, in order.
        mcp_servers: The MCP status snapshot captured before the first turn.
    """

    def __init__(
        self,
        client: ClaudeSDKClient,
        *,
        max_budget_usd: float | None = None,
        mcp_servers: list[Any] | None = None,
    ) -> None:
        self._client = client
        self._max_budget_usd = max_budget_usd
        self.mcp_servers: list[Any] = mcp_servers if mcp_servers is not None else []
        self.turns: list[TurnResult] = []
        self._cumulative_cost: float = 0.0
        self._process: Any = self._transport_process()

    @property
    def total_cost_usd(self) -> float:
        """Cumulative cost across all completed turns."""
        return self._cumulative_cost

    @property
    def num_turns(self) -> int:
        """Number of turns completed so far."""
        return len(self.turns)

    @property
    def budget_remaining(self) -> float | None:
        """Remaining budget, or ``None`` when uncapped."""
        if self._max_budget_usd is None:
            return None
        return max(0.0, self._max_budget_usd - self._cumulative_cost)

    @property
    def budget_exhausted(self) -> bool:
        """True once the cumulative cost has reached a finite budget cap."""
        return self._max_budget_usd is not None and self._cumulative_cost >= self._max_budget_usd

    async def submit(self, message: str | Sequence[Mapping[str, Any]]) -> None:
        """Send one operator turn without reading its response.

        Read the response with :meth:`events`. The message joins the same
        conversation as all prior turns, so the agent keeps their context.

        Args:
            message: The turn's text, or a sequence of content blocks (text,
                images, …) sent as one user message.

        Raises:
            AgentSessionBudgetExceeded: When the budget is already spent; the
                turn is refused before it is sent.
            AgentRunError: When the agent SDK fails to send it.
        """
        if self.budget_exhausted:
            spent = self._cumulative_cost
            cap = self._max_budget_usd
            raise AgentSessionBudgetExceeded(
                f"session budget ${cap:.2f} reached (spent ${spent:.4f}); refusing further turns"
            )
        try:
            await _send_turn(self._client, message)
        except ClaudeSDKError as exc:
            raise from_sdk_error(exc) from exc

    async def events(self) -> AsyncIterator[AgentEvent]:
        """Yield the submitted turn's event records, up to and including its result.

        The client's response stream ends with the result, and this one with it.

        Each record is folded into the turn before it is yielded. When the
        :class:`~osprey.agent_runner.events.ResultEvent` arrives the turn's cost
        is accounted and the turn is appended to :attr:`turns`; a stream that
        ends without one records no turn.

        Yields:
            This turn's records, in stream order.

        Raises:
            AgentRunError: When the agent SDK fails mid-stream.
        """
        text_blocks: list[str] = []
        tool_traces: list[ToolTrace] = []
        system_messages: list[SystemEvent] = []
        pending: dict[str, ToolTrace] = {}
        try:
            async for message in self._client.receive_response():
                for event in translate_message(message):
                    _record_event(event, text_blocks, tool_traces, pending)
                    if isinstance(event, SystemEvent):
                        system_messages.append(event)
                    elif isinstance(event, ResultEvent):
                        self._record_turn(text_blocks, tool_traces, system_messages, event)
                    yield event
        except ClaudeSDKError as exc:
            raise from_sdk_error(exc) from exc

    def _record_turn(
        self,
        text_blocks: list[str],
        tool_traces: list[ToolTrace],
        system_messages: list[SystemEvent],
        result: ResultEvent,
    ) -> None:
        cumulative = result.total_cost_usd  # SDK total_cost_usd (cumulative to date)
        if cumulative is None:
            incremental: float | None = None
        else:
            # SDK documents total_cost_usd as monotonically cumulative; guard a
            # non-monotonic report so a spurious dip neither records a negative
            # turn cost nor rolls back the budgeted total (which would raise
            # budget_remaining and let spend continue past the cap).
            incremental = max(0.0, cumulative - self._cumulative_cost)
            self._cumulative_cost = max(self._cumulative_cost, cumulative)

        self.turns.append(
            TurnResult(
                index=len(self.turns),
                text_blocks=text_blocks,
                tool_traces=tool_traces,
                system_messages=system_messages,
                result=result,
                cost_usd=incremental,
                cumulative_cost_usd=cumulative,
            )
        )

    async def send(self, message: str | Sequence[Mapping[str, Any]]) -> TurnResult:
        """Send one operator turn and return this turn's result.

        :meth:`submit` followed by draining :meth:`events`. Cost is accumulated
        and the budget enforced.

        Args:
            message: The user/operator message for this turn.

        Returns:
            The :class:`TurnResult` for this turn (also appended to ``turns``).

        Raises:
            AgentSessionBudgetExceeded: When the budget is already spent; the
                turn is refused before it is sent.
            AgentRunError: When the agent SDK fails.
            RuntimeError: When the response ends without a result.
        """
        recorded = len(self.turns)
        await self.submit(message)
        async for _event in self.events():
            pass
        if len(self.turns) == recorded:
            raise RuntimeError("the agent response ended without a result")
        return self.turns[-1]

    async def interrupt(self) -> None:
        """Ask the agent to stop the turn in flight.

        The turn's remaining records, ending in its result, still arrive
        through :meth:`events`; nothing here reads them or changes turn state.

        Raises:
            AgentRunError: When the agent SDK fails to deliver the request.
        """
        try:
            await self._client.interrupt()
        except ClaudeSDKError as exc:
            raise from_sdk_error(exc) from exc

    def _transport_process(self) -> Any:
        """The child process handle the client's transport holds, if any.

        Read from the SDK's private subprocess transport; this is the one place
        the package reaches it.
        """
        return getattr(getattr(self._client, "_transport", None), "_process", None)

    def _child(self) -> Any:
        """The live transport's handle, else the one retained."""
        process = self._transport_process()
        if process is not None:
            self._process = process
        return self._process

    @property
    def pid(self) -> int | None:
        """The agent child's pid; ``None`` once it has exited or when no handle
        was captured."""
        process = self._child()
        if process is None or process.returncode is not None:
            return None
        pid = getattr(process, "pid", None)
        return pid if isinstance(pid, int) and pid > 0 else None

    @property
    def child_exited(self) -> bool | None:
        """Whether the agent child has exited.

        ``True`` once it has a return code, ``False`` while it runs, and
        ``None`` when no process handle was ever captured — *nothing to
        observe*, which a caller waiting for the child to be gone must be able
        to tell from *still running*.
        """
        process = self._child()
        if process is None:
            return None
        return process.returncode is not None

    def kill_child(self) -> None:
        """Send SIGKILL to the agent child when it has no return code yet.

        Nothing is awaited: :attr:`child_exited` shows the signal landing. A
        child already gone is the outcome wanted, so its
        ``ProcessLookupError`` is swallowed.
        """
        process = self._child()
        if process is None or process.returncode is not None:
            return
        try:
            process.kill()
        except ProcessLookupError:
            return


@asynccontextmanager
async def agent_session(
    project_dir: Path,
    *,
    disallowed_tools: Sequence[str],
    max_turns: int | None = 25,
    max_budget_usd: float | None = 5.0,
    model: str | None = None,
    permission_mode: PermissionMode | None = "bypassPermissions",
    setting_sources: list[SettingSource] | None = None,
    allowed_tools: Sequence[str] = (),
    system_prompt: str | SystemPromptPreset | SystemPromptFile | None = None,
    env: Mapping[str, str] | None = None,
    provider: str | None = None,
    mcp_servers: Mapping[str, McpServerConfig] | Path | None = None,
    session_id: str | None = None,
    resume: str | None = None,
    can_use_tool: CanUseTool | None = None,
    pre_tool_use_hooks: Sequence[HookCallback] = (),
    stderr: Callable[[str], None] | None = None,
    await_mcp_servers: Collection[str] | None = None,
) -> AsyncIterator[AgentSession]:
    """Open a multi-turn :class:`AgentSession` for *project_dir*.

    Builds options (the same path as ``run_query``), opens one SDK client,
    waits for the MCP servers to register, and yields the handle. The client
    is closed when the context exits. The agent options are built by
    ``build_agent_options`` from the keywords of the same names; see it for
    each one.

    Args:
        project_dir: Path to an initialized OSPREY project.
        disallowed_tools: Tool names forbidden at the SDK level (the read-only
            guard; forwarded as ``--disallowedTools``).
        max_turns: Maximum agentic turns per response; ``None`` sets no cap.
        max_budget_usd: Budget across all turns, enforced here and passed to
            the SDK as a backstop; ``None`` sets no budget.
        model: Model id; the project's main model when ``None``.
        permission_mode: SDK permission mode (``"bypassPermissions"`` for a
            read-only run; ``None`` when an approval callback mediates).
        await_mcp_servers: The MCP servers to wait for before the first turn;
            ``None`` for the project's declared ones, empty to skip the wait.

    Yields:
        An :class:`AgentSession` ready to :meth:`~AgentSession.send` turns.

    Raises:
        ImportError: When ``claude_agent_sdk`` is not installed.
    """
    if not HAS_SDK:
        raise ImportError(
            "claude_agent_sdk is required for agent_session. "
            "Install it with: pip install claude-agent-sdk"
        )

    options = build_agent_options(
        project_dir,
        disallowed_tools=disallowed_tools,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        model=model,
        permission_mode=permission_mode,
        setting_sources=setting_sources,
        allowed_tools=allowed_tools,
        system_prompt=system_prompt,
        env=env,
        provider=provider,
        mcp_servers=mcp_servers,
        session_id=session_id,
        resume=resume,
        can_use_tool=can_use_tool,
        pre_tool_use_hooks=pre_tool_use_hooks,
        stderr=stderr,
    )

    async with ClaudeSDKClient(options=options) as client:
        mcp_snapshot = await _ready_mcp(client, project_dir, await_mcp_servers=await_mcp_servers)
        yield AgentSession(client, max_budget_usd=max_budget_usd, mcp_servers=mcp_snapshot)


async def run_turns(
    project_dir: Path,
    prompts: list[str],
    *,
    disallowed_tools: list[str],
    max_turns: int = 25,
    max_budget_usd: float = 5.0,
    model: str | None = None,
    permission_mode: PermissionMode = "bypassPermissions",
) -> list[TurnResult]:
    """Run a fixed sequence of *prompts* as one conversation.

    Convenience over :func:`agent_session` for non-adaptive callers (a scripted
    scenario or smoke test).  Stops early if the session budget is exhausted
    mid-sequence, returning the turns completed so far.

    Args:
        project_dir: Path to an initialized OSPREY project.
        prompts: Operator messages to send in order, one per turn.
        disallowed_tools: Tool names forbidden at the SDK level.
        max_turns: Maximum agentic turns per response.
        max_budget_usd: Session budget across all turns.
        model: Model id; the project's main model when ``None``.
        permission_mode: SDK permission mode.

    Returns:
        One :class:`TurnResult` per prompt actually sent (fewer than
        ``len(prompts)`` if the budget cut the conversation short).
    """
    results: list[TurnResult] = []
    async with agent_session(
        project_dir,
        disallowed_tools=disallowed_tools,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        model=model,
        permission_mode=permission_mode,
    ) as session:
        for prompt in prompts:
            if session.budget_exhausted:
                break
            results.append(await session.send(prompt))
    return results
