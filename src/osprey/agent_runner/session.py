"""Multi-turn agent conversations over the Claude Agent SDK.

``osprey.agent_runner.runner.run_query`` sends one prompt and closes the
session.  Some callers need to hold a session open and choose each message from
the agent's previous reply — an operator-style evaluation, an adaptive
red-team loop, or a benchmark that branches on what the agent answered.

:class:`AgentSession` provides that: a handle over one persistent SDK client
that sends a single user turn at a time, accumulating transcript and cost and
enforcing a session-wide budget.  Open one for a project with
:func:`agent_session`; for the simple case of a fixed, pre-decided script use
:func:`run_turns`.  Both reuse the same provider-routing and stream-parsing as
``run_query`` (via ``primitives.build_agent_options`` / ``_drain_response``).

Cost accounting: the SDK reports ``ResultMessage.total_cost_usd`` as the
cumulative session cost to date — the field is a running ``total_`` and the SDK
documents its usage counters as cumulative for the session.  A turn's
incremental cost is therefore the delta from the previous turn's total, and the
session total is the latest turn's ``total_cost_usd``.  The session budget is
enforced from that cumulative figure and is also passed to the SDK as a
backstop.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from claude_agent_sdk import (
        ClaudeSDKClient,
        PermissionMode,
        ResultMessage,
        ToolPermissionContext,
    )

# SDK import — keep module importable even when SDK is absent.
try:
    from claude_agent_sdk import ClaudeSDKClient

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

from osprey.agent_runner.primitives import (
    SDKWorkflowResult,
    ToolTrace,
    _await_mcp_ready,
    _drain_response,
    _expected_mcp_servers,
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
    this turn; the running session view lives on :class:`AgentSession`.
    """

    index: int
    text_blocks: list[str] = field(default_factory=list)
    tool_traces: list[ToolTrace] = field(default_factory=list)
    system_messages: list[Any] = field(default_factory=list)
    result: ResultMessage | None = None
    #: This turn's incremental cost (delta from the previous turn), or ``None``
    #: when the SDK did not report a cost for the turn.
    cost_usd: float | None = None
    #: Cumulative session cost as of this turn (the SDK's ``total_cost_usd``).
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
    for MCP servers to register; then drive turns with :meth:`send`.  The SDK
    client is injected rather than built here, so the turn/cost/budget logic is
    unit-testable with a fake client and no live model.

    Attributes:
        turns: Completed :class:`TurnResult` records, in order.
        mcp_servers: The MCP status snapshot captured when the session opened.
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
        """Remaining budget, or ``None`` when the session is uncapped."""
        if self._max_budget_usd is None:
            return None
        return max(0.0, self._max_budget_usd - self._cumulative_cost)

    @property
    def budget_exhausted(self) -> bool:
        """True once the cumulative cost has reached a finite budget cap."""
        return self._max_budget_usd is not None and self._cumulative_cost >= self._max_budget_usd

    async def send(self, message: str) -> TurnResult:
        """Send one operator turn and return this turn's result.

        The message is sent on the same session as all prior turns, so the
        agent retains full conversation context.  Cost is accumulated and the
        session budget enforced.

        Args:
            message: The user/operator message for this turn.

        Returns:
            The :class:`TurnResult` for this turn (also appended to ``turns``).

        Raises:
            AgentSessionBudgetExceeded: When the budget is already spent; the
                turn is refused before it is sent.
        """
        if self.budget_exhausted:
            spent = self._cumulative_cost
            cap = self._max_budget_usd
            raise AgentSessionBudgetExceeded(
                f"session budget ${cap:.2f} reached (spent ${spent:.4f}); refusing further turns"
            )

        turn_workflow = SDKWorkflowResult()
        await self._client.query(message)
        await _drain_response(self._client, turn_workflow)

        cumulative = turn_workflow.cost_usd  # SDK total_cost_usd (session-to-date)
        if cumulative is None:
            incremental: float | None = None
        else:
            # SDK documents total_cost_usd as monotonically cumulative; guard a
            # non-monotonic report so a spurious dip neither records a negative
            # turn cost nor rolls back the budgeted total (which would raise
            # budget_remaining and let spend continue past the cap).
            incremental = max(0.0, cumulative - self._cumulative_cost)
            self._cumulative_cost = max(self._cumulative_cost, cumulative)

        turn = TurnResult(
            index=len(self.turns),
            text_blocks=turn_workflow.text_blocks,
            tool_traces=turn_workflow.tool_traces,
            system_messages=turn_workflow.system_messages,
            result=turn_workflow.result,
            cost_usd=incremental,
            cumulative_cost_usd=cumulative,
        )
        self.turns.append(turn)
        return turn


@asynccontextmanager
async def agent_session(
    project_dir: Path,
    *,
    disallowed_tools: list[str],
    max_turns: int = 25,
    max_budget_usd: float = 5.0,
    model: str | None = None,
    permission_mode: PermissionMode = "bypassPermissions",
    can_use_tool: Callable[[str, dict[str, Any], ToolPermissionContext], Any] | None = None,
) -> AsyncIterator[AgentSession]:
    """Open a multi-turn :class:`AgentSession` for *project_dir*.

    Builds provider-routed options (the same path as ``run_query``), opens one
    SDK client, waits for the project's MCP servers to register, and yields a
    session handle.  The client is closed when the context exits.

    Args:
        project_dir: Path to an initialized OSPREY project.
        disallowed_tools: Tool names forbidden at the SDK level (the read-only
            guard; forwarded as ``--disallowedTools``).
        max_turns: Maximum agentic turns per response.
        max_budget_usd: Session budget, enforced across all turns and passed to
            the SDK as a backstop.
        model: Model id; resolved from the project's haiku tier when ``None``.
        permission_mode: SDK permission mode (``"bypassPermissions"`` for a
            read-only run; ``"default"`` when an approval callback mediates).
        can_use_tool: Optional SDK permission callback, forwarded to
            :func:`~osprey.agent_runner.primitives.build_agent_options`
            unchanged. Only meaningful alongside ``permission_mode="default"``.
            ``None`` (the default) preserves prior behavior exactly.

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
        can_use_tool=can_use_tool,
    )

    async with ClaudeSDKClient(options=options) as client:
        mcp_servers = await _await_mcp_ready(client, _expected_mcp_servers(project_dir))
        yield AgentSession(client, max_budget_usd=max_budget_usd, mcp_servers=mcp_servers)


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
        model: Model id; resolved from the project's haiku tier when ``None``.
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
