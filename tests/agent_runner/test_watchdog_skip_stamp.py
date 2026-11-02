"""Headless runs carry the watchdog skip marker; an interactive session does not.

The marker has to land on the built environment. Passing it as ``env`` would tell
the options builder the caller has routed the run, and the run would lose its
provider, so these tests also pin what the builder was asked.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.agent_runner import runner, session
from osprey.agent_runner.runner import WATCHDOG_SKIP_ENV_VAR

ROUTED = {"ANTHROPIC_BASE_URL": "https://routed.example"}


class _Harness:
    """Stands in for the options builder and the SDK client, in one module."""

    def __init__(self, module: Any) -> None:
        self.options = SimpleNamespace(env=dict(ROUTED))
        self.build = MagicMock(return_value=self.options)

        async def _no_messages():
            return
            yield

        client = MagicMock()
        client.receive_response = _no_messages
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=False)
        self.patches = [
            patch.object(module, "build_agent_options", self.build),
            patch.object(module, "ClaudeSDKClient", return_value=context),
            patch.object(module, "_ready_mcp", AsyncMock(return_value=[])),
        ]
        if module is runner:
            self.patches.append(patch.object(module, "_send_turn", AsyncMock()))

    def __enter__(self) -> _Harness:
        for active in self.patches:
            active.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        for active in self.patches:
            active.stop()


def _assert_stamped_and_routed(harness: _Harness) -> None:
    assert harness.options.env == {**ROUTED, WATCHDOG_SKIP_ENV_VAR: "1"}
    assert harness.build.call_args.kwargs.get("env") is None


@pytest.mark.asyncio
async def test_stream_query_stamps_the_marker(tmp_path: Path) -> None:
    with _Harness(runner) as harness:
        async for _event in runner.stream_query(tmp_path, "q", disallowed_tools=[]):
            pass

    _assert_stamped_and_routed(harness)


@pytest.mark.asyncio
async def test_run_query_stamps_the_marker(tmp_path: Path) -> None:
    with _Harness(runner) as harness:
        await runner.run_query(tmp_path, "q", disallowed_tools=[])

    _assert_stamped_and_routed(harness)


@pytest.mark.asyncio
async def test_a_caller_routed_query_is_stamped_too(tmp_path: Path) -> None:
    caller_env = {"ANTHROPIC_BASE_URL": "https://caller.example"}
    with _Harness(runner) as harness:
        await runner.run_query(tmp_path, "q", disallowed_tools=[], env=caller_env)

    assert harness.build.call_args.kwargs["env"] == caller_env
    assert harness.options.env[WATCHDOG_SKIP_ENV_VAR] == "1"
    assert WATCHDOG_SKIP_ENV_VAR not in caller_env


@pytest.mark.asyncio
async def test_run_turns_stamps_the_marker(tmp_path: Path) -> None:
    with _Harness(session) as harness:
        assert await session.run_turns(tmp_path, [], disallowed_tools=[]) == []

    _assert_stamped_and_routed(harness)


@pytest.mark.asyncio
async def test_simple_modes_session_is_not_stamped(tmp_path: Path) -> None:
    with _Harness(session) as harness:
        async with session.agent_session(tmp_path, disallowed_tools=[]):
            pass

    assert harness.options.env == ROUTED
