"""The context frame the chat's meter reads.

A chat child is asked for its context window as each response ends, and the
reading goes out as a ``context`` frame before the result, carrying the window's
occupancy and never a cost. These tests pin that frame at the server: when it is
sent, what it carries, what happens when the child cannot answer, and that a
prompt landing during the read waits behind the result it belongs to. What the
chat's stream filter lets through of the frame is a row of ``_STRIP_CASES`` in
``test_chat_routes.py``.
"""

from __future__ import annotations

import asyncio

import pytest
from claude_agent_sdk import TextBlock

from osprey.interfaces.web_terminal import operator_session
from osprey.interfaces.web_terminal.operator_session import _context_frame
from tests.interfaces.web_terminal._fakes import (
    assistant_message,
    collect_turn,
    result_message,
    started_session,
    system_message,
    unnumbered,
)

#: The session key one browser tab keeps: the bare UUID the chat child starts under.
KEY = "11111111-2222-3333-4444-555555555555"

#: What the window reads as on a reading the CLI makes; ``apiUsage`` carries a cost.
USAGE = {
    "categories": [{"name": "Messages", "tokens": 80_000, "color": "#000"}],
    "totalTokens": 84_000,
    "maxTokens": 180_000,
    "rawMaxTokens": 200_000,
    "percentage": 42,
    "model": "claude-test",
    "isAutoCompactEnabled": True,
    "memoryFiles": [],
    "mcpTools": [],
    "agents": [],
    "gridRows": [],
    "apiUsage": {"input_tokens": 9, "total_cost_usd": 1.23},
}


def _init(session_id: str):
    return system_message("init", {"session_id": session_id})


def _result(session_id: str):
    return result_message(session_id=session_id)


def _say(text: str):
    return assistant_message([TextBlock(text)])


class _Child:
    """A scripted CLI child: each prompt is answered by the next list of messages.

    It answers no context-window question, as an SDK client without the method does.
    """

    def __init__(self, turns) -> None:
        self._turns = list(turns)
        self._prompts: asyncio.Queue[str] = asyncio.Queue()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        self._prompts.put_nowait(prompt)

    async def interrupt(self):
        pass

    async def receive_response(self):
        await self._prompts.get()
        for message in self._turns.pop(0):
            yield message


class _ReadingChild(_Child):
    """A scripted child that also answers the context-window question with *usage*."""

    def __init__(self, turns, usage=None) -> None:
        super().__init__(turns)
        self.usage = usage

    async def get_context_usage(self):
        return self.usage


class _RefusingChild(_ReadingChild):
    """A child whose CLI refuses the context-window question."""

    async def get_context_usage(self):
        raise RuntimeError("the control request was refused")


class _SilentChild(_ReadingChild):
    """A child whose CLI never answers the context-window question."""

    async def get_context_usage(self):
        await asyncio.sleep(3600)


async def _collect(frames) -> list[dict]:
    return [frame async for frame in frames]


def _types(frames: list[dict]) -> list[str]:
    return [frame["type"] for frame in frames]


class TestTheContextMeterReading:
    async def test_the_reading_goes_out_before_the_result_with_the_window_only(self):
        child = _ReadingChild([[_init(KEY), _say("done"), _result(KEY)]], USAGE)
        async with started_session(child, session_key=KEY) as session:
            frames = await collect_turn(session)
        assert _types(frames) == ["turn_start", "system", "text", "context", "result"]
        (context,) = [f for f in frames if f["type"] == "context"]
        assert unnumbered(context) == {
            "type": "context",
            "used_tokens": 84_000,
            "window_tokens": 200_000,
            "percent": 42,
        }

    async def test_the_last_reading_is_in_the_feed_snapshot(self):
        child = _ReadingChild([[_result(KEY)]], USAGE)
        async with started_session(child, session_key=KEY) as session:
            await collect_turn(session)
            feed, hello = session.subscribe()
            session.unsubscribe(feed)
        assert hello["context"] == {
            "type": "context",
            "used_tokens": 84_000,
            "window_tokens": 200_000,
            "percent": 42,
        }

    @pytest.mark.parametrize(
        "child_type",
        [
            pytest.param(_Child, id="no-such-question"),
            pytest.param(_RefusingChild, id="refused"),
            pytest.param(_SilentChild, id="never-answered"),
        ],
    )
    async def test_a_reading_that_does_not_come_costs_the_meter_only(self, monkeypatch, child_type):
        """No ``context`` frame goes out, and the turn still ends on its result.

        The read is given a millisecond, so a child that never answers ends the
        turn at once rather than after the five seconds a live chat allows.
        """
        monkeypatch.setattr(operator_session, "_CONTEXT_READ_S", 0.001)
        async with started_session(child_type([[_result(KEY)]]), session_key=KEY) as session:
            frames = await asyncio.wait_for(collect_turn(session), timeout=5)
        assert _types(frames) == ["turn_start", "result"]

    async def test_a_prompt_sent_during_the_reading_waits_behind_the_result(self):
        """The response is not over until its result has gone out.

        A response the CLI produced on its own ends while its context reading is
        still pending; a prompt that lands in that window opens its turn after the
        old result, so the old result cannot end the new turn.
        """
        release = asyncio.Event()
        prompted = asyncio.Event()

        class OwnResponseChild(_ReadingChild):
            """Answers first with a response nobody asked for, whose reading waits."""

            async def query(self, prompt):
                await super().query(prompt)
                prompted.set()

            async def receive_response(self):
                if self._turns and len(self._turns) == 2:
                    for message in self._turns.pop(0):
                        yield message
                    return
                async for message in super().receive_response():
                    yield message

            async def get_context_usage(self):
                if not release.is_set():
                    await release.wait()
                return self.usage

        child = OwnResponseChild(
            [
                [_init(KEY), _say("on my own"), _result(KEY)],
                [_init(KEY), _say("answer"), _result(KEY)],
            ],
            USAGE,
        )
        async with started_session(child, session_key=KEY) as session:
            feed, _hello = session.subscribe()
            # Once the own response's text is on the feed the pump has reached its
            # result and is waiting in the context read.
            seen: list[dict] = []
            while not seen or seen[-1]["type"] != "text":
                seen.append(await asyncio.wait_for(feed.get(), timeout=5))
            session.unsubscribe(feed)
            token = session.acquire_turn()
            consumer = asyncio.create_task(
                _collect(session.run_turn("hello", token, timeout_s=5.0))
            )
            await asyncio.wait_for(prompted.wait(), timeout=5)
            release.set()
            frames = await asyncio.wait_for(consumer, timeout=5)
        assert _types(seen) == ["turn_start", "system", "text"]
        assert seen[0]["prompt"] is None
        assert _types(frames) == ["turn_start", "system", "text", "context", "result"]
        assert frames[0]["prompt"] == "hello"
        assert frames[2]["content"] == "answer"


class TestContextFrame:
    @pytest.mark.parametrize(
        "usage",
        [
            None,
            "42%",
            [],
            {},
            {"totalTokens": 1, "rawMaxTokens": 0, "percentage": 0},
            {"totalTokens": True, "rawMaxTokens": 10, "percentage": 1},
            {"totalTokens": 1, "rawMaxTokens": 10, "percentage": float("nan")},
        ],
    )
    def test_a_reading_that_is_not_a_window_yields_no_frame(self, usage):
        assert _context_frame(usage) is None

    @pytest.mark.parametrize(
        ("percentage", "shown"), [(41.6, 42), (0, 0), (100, 100), (150, 100), (-3, 0)]
    )
    def test_the_percent_is_a_whole_number_on_the_scale(self, percentage, shown):
        frame = _context_frame({"totalTokens": 1, "rawMaxTokens": 10, "percentage": percentage})
        assert frame["percent"] == shown
