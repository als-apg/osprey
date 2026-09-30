"""Tests for SDK helpers extracted to benchmarks.sdk."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from osprey.services.channel_finder.benchmarks.sdk import (
    SDKWorkflowResult,
    ToolTrace,
    combined_text,
    run_sdk_query,
)


class TestToolTrace:
    """Tests for the ToolTrace dataclass."""

    def test_defaults(self):
        trace = ToolTrace(name="read_file", input={"path": "/tmp/x"})
        assert trace.name == "read_file"
        assert trace.input == {"path": "/tmp/x"}
        assert trace.result is None
        assert trace.is_error is False
        assert trace.tool_use_id is None
        assert trace.parent_tool_use_id is None

    def test_all_fields(self):
        trace = ToolTrace(
            name="write_file",
            input={"path": "/tmp/x"},
            result="ok",
            is_error=True,
            tool_use_id="tu_123",
            parent_tool_use_id="tu_000",
        )
        assert trace.result == "ok"
        assert trace.is_error is True
        assert trace.tool_use_id == "tu_123"
        assert trace.parent_tool_use_id == "tu_000"


class TestSDKWorkflowResult:
    """Tests for the SDKWorkflowResult dataclass."""

    def test_empty(self):
        r = SDKWorkflowResult()
        assert r.tool_traces == []
        assert r.text_blocks == []
        assert r.system_messages == []
        assert r.result is None
        assert r.tool_names == []
        assert r.cost_usd is None
        assert r.num_turns is None

    def test_tool_names(self):
        r = SDKWorkflowResult(
            tool_traces=[
                ToolTrace(name="read_channel", input={}),
                ToolTrace(name="write_channel", input={}),
            ]
        )
        assert r.tool_names == ["read_channel", "write_channel"]

    def test_tools_matching(self):
        r = SDKWorkflowResult(
            tool_traces=[
                ToolTrace(name="read_channel", input={}),
                ToolTrace(name="write_channel", input={}),
                ToolTrace(name="read_file", input={}),
            ]
        )
        matches = r.tools_matching("read")
        assert len(matches) == 2
        assert matches[0].name == "read_channel"
        assert matches[1].name == "read_file"

    def test_cost_and_turns_from_result(self):
        """cost_usd and num_turns delegate to result object via duck typing."""

        class FakeResult:
            total_cost_usd = 0.05
            num_turns = 7

        r = SDKWorkflowResult(result=FakeResult())
        assert r.cost_usd == 0.05
        assert r.num_turns == 7


class TestCombinedText:
    """Tests for the combined_text helper."""

    def test_text_blocks_only(self):
        r = SDKWorkflowResult(text_blocks=["Hello", "World"])
        assert combined_text(r) == "hello world"

    def test_includes_tool_results(self):
        r = SDKWorkflowResult(
            text_blocks=["Found channels:"],
            tool_traces=[
                ToolTrace(name="query", input={}, result="SR01C:H1"),
                ToolTrace(name="other", input={}, result=None),
            ],
        )
        assert "sr01c:h1" in combined_text(r)

    def test_empty_result(self):
        r = SDKWorkflowResult()
        assert combined_text(r) == ""


# ---------------------------------------------------------------------------
# run_sdk_query
# ---------------------------------------------------------------------------

_RUN_QUERY = "osprey.services.channel_finder.benchmarks.sdk.run_query"
_AGENT_BODY = "You find channels."
_CF_ENTRY: dict[str, Any] = {"command": "osprey-channel-finder-mcp", "args": []}
_CONTROLS_ENTRY: dict[str, Any] = {"command": "osprey-controls-mcp"}


def _write_project(root: Path, servers: dict[str, Any]) -> Path:
    agents = root / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "channel-finder.md").write_text(
        f"---\nname: channel-finder\ndescription: finds channels\n---\n{_AGENT_BODY}\n"
    )
    (root / ".mcp.json").write_text(json.dumps({"mcpServers": servers}))
    return root


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A built project whose ``.mcp.json`` declares controls and channel-finder."""
    return _write_project(tmp_path, {"controls": _CONTROLS_ENTRY, "channel-finder": _CF_ENTRY})


class TestRunSdkQuery:
    """What ``run_sdk_query`` hands the shared runner and how it reports failure."""

    async def test_runs_as_the_channel_finder_agent_through_the_runner(self, project: Path):
        expected = SDKWorkflowResult()
        runner = AsyncMock(return_value=expected)
        with patch(_RUN_QUERY, new=runner):
            result = await run_sdk_query(
                project, "q", max_turns=7, max_budget_usd=0.5, model="m", provider="p"
            )

        assert result is expected
        runner.assert_awaited_once()
        args, kwargs = runner.call_args
        assert args == (project, "q")
        stderr = kwargs.pop("stderr")
        assert callable(stderr)
        assert kwargs == {
            "disallowed_tools": [],
            "allowed_tools": ["mcp__channel-finder__*"],
            "system_prompt": _AGENT_BODY,
            "mcp_servers": {"channel-finder": _CF_ENTRY},
            "await_mcp_servers": {"channel-finder"},
            "setting_sources": [],
            "max_turns": 7,
            "max_budget_usd": 0.5,
            "model": "m",
            "provider": "p",
        }

    async def test_waits_only_for_the_channel_finder_server(self, project: Path):
        runner = AsyncMock(return_value=SDKWorkflowResult())
        with patch(_RUN_QUERY, new=runner):
            await run_sdk_query(project, "q", model="m")

        kwargs = runner.call_args.kwargs
        assert kwargs["mcp_servers"] == {"channel-finder": _CF_ENTRY}
        assert kwargs["await_mcp_servers"] == {"channel-finder"}

    async def test_without_a_channel_finder_entry_the_project_mcp_file_is_loaded(
        self, tmp_path: Path
    ):
        project = _write_project(tmp_path, {"controls": _CONTROLS_ENTRY})
        runner = AsyncMock(return_value=SDKWorkflowResult())
        with patch(_RUN_QUERY, new=runner):
            await run_sdk_query(project, "q", model="m")

        kwargs = runner.call_args.kwargs
        assert kwargs["mcp_servers"] == project / ".mcp.json"
        assert kwargs["await_mcp_servers"] is None

    async def test_model_and_provider_reach_the_runner_unresolved(self, project: Path):
        runner = AsyncMock(return_value=SDKWorkflowResult())
        with patch(_RUN_QUERY, new=runner):
            await run_sdk_query(project, "q", model=None, provider="cborg")

        kwargs = runner.call_args.kwargs
        assert kwargs["model"] is None
        assert kwargs["provider"] == "cborg"

    async def test_a_failed_query_carries_the_cli_stderr(self, project: Path):
        cause = ValueError("boom")

        async def _fail(*_args: Any, **kwargs: Any) -> SDKWorkflowResult:
            kwargs["stderr"]("line 1")
            raise RuntimeError("SDK query failed: boom") from cause

        with patch(_RUN_QUERY, new=_fail), pytest.raises(RuntimeError) as info:
            await run_sdk_query(project, "q", model="m")

        assert str(info.value) == "SDK query failed: boom\n\nCLI stderr:\nline 1"
        assert info.value.__cause__ is cause

    async def test_a_failed_query_without_stderr_says_none_was_captured(self, project: Path):
        runner = AsyncMock(side_effect=RuntimeError("SDK query failed: boom"))
        with patch(_RUN_QUERY, new=runner), pytest.raises(RuntimeError) as info:
            await run_sdk_query(project, "q", model="m")

        assert str(info.value) == "SDK query failed: boom\n\nCLI stderr:\n(no stderr captured)"

    async def test_errors_other_than_runtime_errors_pass_through(self, project: Path):
        error = ValueError("bad options")
        runner = AsyncMock(side_effect=error)
        with patch(_RUN_QUERY, new=runner), pytest.raises(ValueError) as info:
            await run_sdk_query(project, "q", model="m")

        assert info.value is error


# ---------------------------------------------------------------------------
# run_sdk_query on the real runner, with the agent SDK client faked
# ---------------------------------------------------------------------------

_ENV = {"CLAUDECODE": "", "ANTHROPIC_AUTH_TOKEN": "sk-test"}
_TOOL_USE_ID = "tool-cf-1"
_TOOL_TEXT = "SR01C:BPM1:X"
_ANSWER = "The channel is SR01C:BPM1:X."
_SNAPSHOT = [{"name": "channel-finder", "status": "connected", "tools": [{"name": "query"}]}]


async def _scripted_stream() -> AsyncIterator[object]:
    yield AssistantMessage(
        content=[ToolUseBlock(id=_TOOL_USE_ID, name="mcp__channel-finder__query", input={})],
        model="m",
    )
    yield UserMessage(content=[ToolResultBlock(tool_use_id=_TOOL_USE_ID, content=_TOOL_TEXT)])
    yield AssistantMessage(content=[TextBlock(text=_ANSWER)], model="m")
    yield ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=2,
        session_id="sess-bench",
        total_cost_usd=0.0123,
        stop_reason="end_turn",
    )


class _FakeSpec:
    def __init__(self, *, needs_proxy: bool) -> None:
        self.needs_proxy = needs_proxy
        self.auth_env_var = "ANTHROPIC_AUTH_TOKEN"
        self.upstream_base_url = "https://openai.example/v1"
        self.provider = "p"
        self.supports_images = None


class _Harness:
    """The fakes around ``run_query``: a capturing SDK client and the routing seams."""

    def __init__(self, spec: _FakeSpec | None = None) -> None:
        self.captured: list[ClaudeAgentOptions] = []
        client = MagicMock()
        client.query = AsyncMock(return_value=None)
        client.receive_response = MagicMock(return_value=_scripted_stream())
        self._async_cm = MagicMock()
        self._async_cm.__aenter__ = AsyncMock(return_value=client)
        self._async_cm.__aexit__ = AsyncMock(return_value=False)
        self.await_mcp_ready = AsyncMock(return_value=_SNAPSHOT)
        self.sdk_env = MagicMock(side_effect=lambda *_a, **_k: dict(_ENV))
        self.start_proxy = MagicMock(return_value=8123)
        self.spec = spec

    def _client(self, options: ClaudeAgentOptions) -> MagicMock:
        self.captured.append(options)
        return self._async_cm

    async def run(self, project: Path, **kwargs: Any) -> SDKWorkflowResult:
        with (
            patch("osprey.agent_runner.runner.ClaudeSDKClient", side_effect=self._client),
            patch("osprey.agent_runner.primitives.await_mcp_ready", new=self.await_mcp_ready),
            patch("osprey.agent_runner.primitives.sdk_env", new=self.sdk_env),
            patch("osprey.agent_runner.primitives._resolve_project_spec", return_value=self.spec),
            patch("osprey.infrastructure.proxy.lifecycle.start_proxy", new=self.start_proxy),
        ):
            return await run_sdk_query(project, "q", **kwargs)


class TestRunSdkQueryOnTheRunner:
    """The options and result of a benchmark query driven through ``run_query``."""

    async def test_agent_options_equal_the_one_shot_backend_options(self, project: Path):
        harness = _Harness()
        await harness.run(project, max_turns=7, max_budget_usd=0.5, model="m", provider="p")

        (captured,) = harness.captured
        assert callable(captured.stderr)
        assert dataclasses.replace(captured, stderr=None) == ClaudeAgentOptions(
            model="m",
            cwd=str(project),
            permission_mode="bypassPermissions",
            max_turns=7,
            max_budget_usd=0.5,
            env=_ENV,
            system_prompt=_AGENT_BODY,
            mcp_servers={"channel-finder": _CF_ENTRY},
            allowed_tools=["mcp__channel-finder__*"],
            setting_sources=[],
            disallowed_tools=[],
            strict_mcp_config=True,
        )
        harness.sdk_env.assert_called_once_with(project, provider="p")

    async def test_tool_results_reach_the_traces(self, project: Path):
        harness = _Harness()
        result = await harness.run(project, model="m")

        assert result.tool_traces[0].result == _TOOL_TEXT
        assert result.text_blocks == [_ANSWER]
        assert result.cost_usd == 0.0123
        assert result.mcp_servers == _SNAPSHOT
        harness.await_mcp_ready.assert_awaited_once()
        assert harness.await_mcp_ready.call_args.args[1] == {"channel-finder"}

    async def test_an_openai_protocol_provider_is_routed_through_the_proxy(self, project: Path):
        harness = _Harness(spec=_FakeSpec(needs_proxy=True))
        await harness.run(project, model="m", provider="p")

        harness.start_proxy.assert_called_once()
        assert harness.start_proxy.call_args.args[0] == "https://openai.example/v1"
        (captured,) = harness.captured
        assert captured.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8123"
