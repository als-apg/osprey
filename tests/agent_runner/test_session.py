"""Unit tests for osprey.agent_runner.session (multi-turn AgentSession).

Every test drives a scripted fake ClaudeSDKClient — no live model, no API keys.
The fake yields a pre-scripted message stream per turn and records the messages
it was asked to send, so the tests can prove turn ordering, cross-turn cost
accumulation, and budget enforcement without touching a provider.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from osprey.agent_runner.session import (
    AgentSession,
    AgentSessionBudgetExceeded,
    run_turns,
)

TOOL_ID = "tool-1"


def _result(total_cost_usd: float | None, num_turns: int = 1) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=num_turns,
        session_id="sess-fake",
        total_cost_usd=total_cost_usd,
        stop_reason="end_turn",
    )


def _tool_turn(text: str, cost: float | None) -> list:
    """A turn that calls a tool, gets a result, then answers — ending in a
    ResultMessage carrying the session-cumulative cost."""
    return [
        AssistantMessage(
            content=[
                ToolUseBlock(id=TOOL_ID, name="mcp__controls__channel_read", input={"channel": "X"})
            ],
            model="m",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id=TOOL_ID, content="42")]),
        AssistantMessage(content=[TextBlock(text=text)], model="m"),
        _result(cost),
    ]


def _text_turn(text: str, cost: float | None) -> list:
    """A plain text turn ending in a ResultMessage."""
    return [AssistantMessage(content=[TextBlock(text=text)], model="m"), _result(cost)]


class _FakeClient:
    """Scripted async client. ``turns[i]`` is the message list yielded for the
    i-th ``receive_response()``; ``queries`` records what was sent."""

    def __init__(self, turns: list[list]) -> None:
        self._turns = turns
        self._i = 0
        self.queries: list[str] = []

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def query(self, message: str, **kwargs: object) -> None:
        self.queries.append(message)

    def receive_response(self):
        messages = self._turns[self._i]
        self._i += 1

        async def _gen():
            for message in messages:
                yield message

        return _gen()


@pytest.mark.asyncio
async def test_send_accumulates_turns_and_cost() -> None:
    """Each send() records a turn; text/tools are captured per turn; the per-turn
    cost is the delta and the session total is the latest cumulative figure."""
    client = _FakeClient([_tool_turn("first answer", 0.001), _text_turn("second answer", 0.003)])
    session = AgentSession(client, max_budget_usd=None)

    t0 = await session.send("turn 0")
    assert t0.index == 0
    assert t0.text == "first answer"
    assert t0.tool_names == ["mcp__controls__channel_read"]
    assert t0.tool_traces[0].result == "42"
    assert t0.cost_usd == pytest.approx(0.001)
    assert t0.cumulative_cost_usd == pytest.approx(0.001)

    t1 = await session.send("turn 1")
    assert t1.index == 1
    assert t1.text == "second answer"
    assert t1.cost_usd == pytest.approx(0.002)  # 0.003 cumulative − 0.001 prior
    assert t1.cumulative_cost_usd == pytest.approx(0.003)

    assert session.num_turns == 2
    assert session.total_cost_usd == pytest.approx(0.003)
    assert [t.text for t in session.turns] == ["first answer", "second answer"]
    assert client.queries == ["turn 0", "turn 1"]


@pytest.mark.asyncio
async def test_send_issues_one_query_per_turn_on_one_session() -> None:
    """A conversation is N sequential queries on the SAME client — not a batch."""
    client = _FakeClient([_text_turn("a", 0.001), _text_turn("b", 0.002), _text_turn("c", 0.003)])
    session = AgentSession(client, max_budget_usd=None)

    for i in range(3):
        await session.send(f"msg {i}")

    assert client.queries == ["msg 0", "msg 1", "msg 2"]
    assert session.num_turns == 3


@pytest.mark.asyncio
async def test_budget_refuses_further_turns_before_spending() -> None:
    """Once cumulative cost reaches the cap the next send() raises and never
    reaches the client, so the refused turn costs nothing."""
    client = _FakeClient([_text_turn("a", 0.001), _text_turn("b", 0.005)])
    session = AgentSession(client, max_budget_usd=0.004)

    await session.send("first")  # cumulative 0.001 < 0.004 → allowed
    assert not session.budget_exhausted
    await session.send("second")  # cumulative 0.005 ≥ 0.004 → now exhausted
    assert session.budget_exhausted
    assert session.budget_remaining == pytest.approx(0.0)

    with pytest.raises(AgentSessionBudgetExceeded):
        await session.send("third")

    assert client.queries == ["first", "second"]  # third never sent


@pytest.mark.asyncio
async def test_missing_cost_leaves_total_unchanged() -> None:
    """A turn whose ResultMessage reports no cost yields cost_usd=None without
    corrupting the running total."""
    client = _FakeClient([_text_turn("a", None)])
    session = AgentSession(client, max_budget_usd=None)

    t0 = await session.send("q")

    assert t0.cost_usd is None
    assert t0.cumulative_cost_usd is None
    assert session.total_cost_usd == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_non_monotonic_cost_is_guarded() -> None:
    """A turn whose reported cumulative dips below the prior turn's must not
    record a negative per-turn cost nor roll back the session total (which would
    weaken the budget cap)."""
    client = _FakeClient([_text_turn("a", 0.005), _text_turn("b", 0.003)])
    session = AgentSession(client, max_budget_usd=None)

    t0 = await session.send("first")
    assert t0.cost_usd == pytest.approx(0.005)

    t1 = await session.send("second")  # cumulative dips 0.005 → 0.003
    assert t1.cost_usd == pytest.approx(0.0)  # clamped, not negative
    assert t1.cumulative_cost_usd == pytest.approx(0.003)  # raw SDK figure preserved
    assert session.total_cost_usd == pytest.approx(0.005)  # monotone, not rolled back


@pytest.mark.asyncio
async def test_uncapped_session_reports_no_budget_limit() -> None:
    """An uncapped session (max_budget_usd=None) never reports a remaining
    budget or an exhausted state, before or after spending."""
    client = _FakeClient([_text_turn("a", 0.001)])
    session = AgentSession(client, max_budget_usd=None)

    assert session.budget_remaining is None
    assert not session.budget_exhausted
    await session.send("q")
    assert session.budget_remaining is None
    assert not session.budget_exhausted


@pytest.mark.asyncio
async def test_run_turns_stops_early_when_budget_exhausted(tmp_path: Path) -> None:
    """run_turns stops sending once the session budget is exhausted mid-sequence,
    returning only the turns completed so far rather than raising."""
    client = _FakeClient([_text_turn("a", 0.001), _text_turn("b", 0.005)])
    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=client)
    async_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("osprey.agent_runner.session.build_agent_options", return_value=MagicMock()),
        patch("osprey.agent_runner.session.ClaudeSDKClient", return_value=async_cm),
        patch("osprey.agent_runner.session._await_mcp_ready", new=AsyncMock(return_value=[])),
        patch("osprey.agent_runner.session._expected_mcp_servers", return_value=set()),
    ):
        results = await run_turns(
            tmp_path, ["p0", "p1", "p2"], disallowed_tools=[], max_budget_usd=0.004
        )

    assert [r.text for r in results] == ["a", "b"]  # p2 never sent
    assert client.queries == ["p0", "p1"]


@pytest.mark.asyncio
async def test_run_turns_runs_fixed_script(tmp_path: Path) -> None:
    """run_turns opens a session (SDK wiring faked) and sends each prompt in
    order, returning one TurnResult per prompt."""
    client = _FakeClient([_text_turn("a", 0.001), _text_turn("b", 0.002)])
    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=client)
    async_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("osprey.agent_runner.session.build_agent_options", return_value=MagicMock()),
        patch("osprey.agent_runner.session.ClaudeSDKClient", return_value=async_cm),
        patch("osprey.agent_runner.session._await_mcp_ready", new=AsyncMock(return_value=[])),
        patch("osprey.agent_runner.session._expected_mcp_servers", return_value=set()),
    ):
        results = await run_turns(tmp_path, ["p0", "p1"], disallowed_tools=[])

    assert [r.text for r in results] == ["a", "b"]
    assert client.queries == ["p0", "p1"]
