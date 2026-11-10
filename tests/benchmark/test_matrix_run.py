"""The single-model probe (scripts/benchmark/matrix_run.py) starts the proxy it is meant to.

No network and no ``claude`` CLI: the proxy and the agent SDK's ``query`` are
replaced, and what is checked is the call the probe makes into
:func:`osprey.infrastructure.proxy.lifecycle.start_proxy`. That call is bound
against the real function's signature, so a change to the proxy's arguments
fails here instead of as a ``TypeError`` at the start of a benchmark run.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from pathlib import Path

import claude_agent_sdk
import pytest

from osprey.agent_runner import launcher
from osprey.infrastructure.proxy import lifecycle

_RUN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark" / "matrix_run.py"
_spec = importlib.util.spec_from_file_location("benchmark_matrix_run", _RUN_PATH)
matrix_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(matrix_run)


@pytest.fixture
def proxy_calls(monkeypatch):
    """Record every ``start_proxy`` call, refusing one the real function would refuse."""
    real = inspect.signature(lifecycle.start_proxy)
    calls: list[inspect.BoundArguments] = []

    def _start(*args, **kwargs):
        calls.append(real.bind(*args, **kwargs))
        return 4321

    monkeypatch.setattr(lifecycle, "start_proxy", _start)
    monkeypatch.setattr(lifecycle, "stop_proxy", lambda: None)
    monkeypatch.setattr(launcher, "agent_cli_path", lambda _cfg: "claude")
    return calls


def _answer(output_tokens: int):
    async def _query(*, prompt, options):  # noqa: ARG001 - the SDK's keywords
        yield claude_agent_sdk.AssistantMessage(
            content=[claude_agent_sdk.TextBlock(text="BPM:01:X = 3.14159 mm")],
            model="m",
        )
        yield claude_agent_sdk.ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=5,
            is_error=False,
            num_turns=2,
            session_id="s",
            usage={"input_tokens": 100, "output_tokens": output_tokens},
        )

    return _query


def test_probe_starts_the_proxy_with_the_arguments_the_lifecycle_requires(proxy_calls, monkeypatch):
    monkeypatch.setattr(claude_agent_sdk, "query", _answer(57))

    asyncio.run(matrix_run._drive("http://127.0.0.1:8000/v1", "", "kolibri-1", "vllm"))

    (call,) = proxy_calls
    assert call.arguments["upstream_base_url"] == "http://127.0.0.1:8000/v1"
    assert call.arguments["upstream_api_key"] == ""
    assert call.arguments["provider"] == "vllm"
    assert tuple(call.arguments["forward_headers"]) == ()
    assert call.arguments["supports_images"] is None


@pytest.mark.usefixtures("proxy_calls")
def test_probe_reports_the_output_tokens_the_proxy_returned(monkeypatch):
    monkeypatch.setattr(claude_agent_sdk, "query", _answer(57))

    _calls, text, turns, output_tokens = asyncio.run(
        matrix_run._drive("https://gateway.example/v1", "k", "gpt-oss-20b", "cborg")
    )

    assert matrix_run.SENTINEL in text
    assert turns == 2
    assert output_tokens == 57
