"""Unit tests for osprey.agent_runner.runner.run_query.

All tests mock ClaudeSDKClient and await_mcp_ready so no live model or API
keys are required.  The fake message stream exercises the full collection
logic: tool call → tool result (via UserMessage) → text block → ResultMessage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
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

from osprey.agent_runner import (
    AgentRunError,
    McpNotReadyError,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolUseEvent,
)
from osprey.agent_runner.primitives import SDKWorkflowResult
from osprey.agent_runner.runner import run_query, stream_query

# ---------------------------------------------------------------------------
# Scripted message stream
# ---------------------------------------------------------------------------

FAKE_TOOL_USE_ID = "tool-abc-123"
FAKE_TOOL_NAME = "mcp__controls__channel_read"
FAKE_TOOL_INPUT: dict = {"channel": "BL1:PHOTON_ENERGY"}
FAKE_TOOL_RESULT = "12345.6 eV"
FAKE_TEXT = "The photon energy is 12345.6 eV."
FAKE_MCP_SERVERS = [
    {"name": "controls", "status": "connected", "tools": [{"name": "channel_read"}]}
]


async def _scripted_stream() -> AsyncIterator:
    """Yield a scripted sequence: tool call → tool result → text → ResultMessage."""
    # Turn 1: assistant issues a tool call
    yield AssistantMessage(
        content=[ToolUseBlock(id=FAKE_TOOL_USE_ID, name=FAKE_TOOL_NAME, input=FAKE_TOOL_INPUT)],
        model="claude-haiku-4-5-20251001",
    )
    # Turn 1: user returns the tool result
    yield UserMessage(
        content=[ToolResultBlock(tool_use_id=FAKE_TOOL_USE_ID, content=FAKE_TOOL_RESULT)]
    )
    # Turn 2: assistant produces a text reply
    yield AssistantMessage(
        content=[TextBlock(text=FAKE_TEXT)],
        model="claude-haiku-4-5-20251001",
    )
    # Final: result message
    yield ResultMessage(
        subtype="success",
        duration_ms=500,
        duration_api_ms=400,
        is_error=False,
        num_turns=2,
        session_id="sess-fake-001",
        total_cost_usd=0.001,
        stop_reason="end_turn",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Minimal OSPREY project skeleton sufficient for run_query."""
    # .mcp.json declares the "controls" server so expected_mcp_servers parses it.
    (tmp_path / ".mcp.json").write_text(
        '{"mcpServers": {"controls": {"command": "osprey-controls-mcp"}}}'
    )
    # config.yml must exist (read by sdk_env → provider_env_for_project).
    # We stub sdk_env so this file is never actually parsed in these tests.
    (tmp_path / "config.yml").write_text("api:\n  providers: {}\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Helper: build a mock ClaudeSDKClient async context manager
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    """Return a mock that behaves as ``async with ClaudeSDKClient(...) as client``."""
    client = MagicMock()
    client.query = AsyncMock(return_value=None)
    client.receive_response = MagicMock(return_value=_scripted_stream())

    # Async context manager protocol
    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=client)
    async_cm.__aexit__ = AsyncMock(return_value=False)
    return async_cm, client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_query_collects_tool_traces_and_text(project_dir: Path) -> None:
    """run_query returns an SDKWorkflowResult with the expected tool traces and text."""
    async_cm, mock_client = _make_mock_client()

    with (
        patch("osprey.agent_runner.runner.ClaudeSDKClient", return_value=async_cm),
        patch(
            "osprey.agent_runner.primitives.await_mcp_ready",
            new=AsyncMock(return_value=FAKE_MCP_SERVERS),
        ),
        patch("osprey.agent_runner.primitives.sdk_env", return_value={"CLAUDECODE": ""}),
        patch(
            "osprey.agent_runner.primitives.resolve_default_model",
            return_value="claude-haiku-4-5-20251001",
        ),
        patch(
            "osprey.agent_runner.primitives.expected_mcp_servers",
            return_value={"controls"},
        ),
    ):
        result = await run_query(
            project_dir,
            "What is the photon energy?",
            disallowed_tools=["mcp__controls__channel_write"],
        )

    assert isinstance(result, SDKWorkflowResult)
    # Tool traces
    assert len(result.tool_traces) == 1
    trace = result.tool_traces[0]
    assert trace.name == FAKE_TOOL_NAME
    assert trace.input == FAKE_TOOL_INPUT
    assert trace.tool_use_id == FAKE_TOOL_USE_ID
    assert trace.result == FAKE_TOOL_RESULT
    assert trace.is_error is False
    # Text blocks
    assert result.text_blocks == [FAKE_TEXT]
    # MCP servers
    assert result.mcp_servers == FAKE_MCP_SERVERS
    # ResultMessage
    assert result.result is not None
    assert result.result.num_turns == 2
    assert result.result.total_cost_usd == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_run_query_passes_disallowed_tools_to_options(project_dir: Path) -> None:
    """disallowed_tools is forwarded to ClaudeAgentOptions as-is."""
    async_cm, _ = _make_mock_client()
    captured_options: list[ClaudeAgentOptions] = []

    def _capture_client(options: ClaudeAgentOptions) -> MagicMock:
        captured_options.append(options)
        return async_cm

    write_tools = ["mcp__controls__channel_write", "mcp__controls__channel_put"]

    with (
        patch("osprey.agent_runner.runner.ClaudeSDKClient", side_effect=_capture_client),
        patch(
            "osprey.agent_runner.primitives.await_mcp_ready",
            new=AsyncMock(return_value=[]),
        ),
        patch("osprey.agent_runner.primitives.sdk_env", return_value={"CLAUDECODE": ""}),
        patch(
            "osprey.agent_runner.primitives.resolve_default_model",
            return_value="claude-haiku-4-5-20251001",
        ),
        patch(
            "osprey.agent_runner.primitives.expected_mcp_servers",
            return_value=set(),
        ),
    ):
        await run_query(project_dir, "query", disallowed_tools=write_tools)

    assert len(captured_options) == 1
    opts = captured_options[0]
    assert opts.disallowed_tools == write_tools
    assert opts.permission_mode == "bypassPermissions"
    assert opts.setting_sources == ["project"]
    assert opts.max_turns == 25
    assert opts.max_budget_usd == 2.0


@pytest.mark.asyncio
async def test_run_query_uses_resolved_model_when_none(project_dir: Path) -> None:
    """When model=None, the project's main model is resolved from its config."""
    async_cm, _ = _make_mock_client()
    captured_options: list[ClaudeAgentOptions] = []

    def _capture_client(options: ClaudeAgentOptions) -> MagicMock:
        captured_options.append(options)
        return async_cm

    with (
        patch("osprey.agent_runner.runner.ClaudeSDKClient", side_effect=_capture_client),
        patch(
            "osprey.agent_runner.primitives.await_mcp_ready",
            new=AsyncMock(return_value=[]),
        ),
        patch("osprey.agent_runner.primitives.sdk_env", return_value={"CLAUDECODE": ""}),
        patch(
            "osprey.agent_runner.primitives.resolve_default_model",
            return_value="claude-haiku-4-5-20251001",
        ),
        patch(
            "osprey.agent_runner.primitives.expected_mcp_servers",
            return_value=set(),
        ),
    ):
        await run_query(project_dir, "query", disallowed_tools=[], model=None)

    assert captured_options[0].model == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_run_query_uses_explicit_model_when_supplied(project_dir: Path) -> None:
    """When model is explicitly provided it is passed through unchanged."""
    async_cm, _ = _make_mock_client()
    captured_options: list[ClaudeAgentOptions] = []

    def _capture_client(options: ClaudeAgentOptions) -> MagicMock:
        captured_options.append(options)
        return async_cm

    with (
        patch("osprey.agent_runner.runner.ClaudeSDKClient", side_effect=_capture_client),
        patch(
            "osprey.agent_runner.primitives.await_mcp_ready",
            new=AsyncMock(return_value=[]),
        ),
        patch("osprey.agent_runner.primitives.sdk_env", return_value={"CLAUDECODE": ""}),
        patch(
            "osprey.agent_runner.primitives.resolve_default_model",
            return_value="claude-haiku-4-5-20251001",
        ),
        patch(
            "osprey.agent_runner.primitives.expected_mcp_servers",
            return_value=set(),
        ),
    ):
        await run_query(project_dir, "query", disallowed_tools=[], model="claude-sonnet-4-6")

    assert captured_options[0].model == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_run_query_mcp_servers_populated(project_dir: Path) -> None:
    """MCP server snapshot from await_mcp_ready is stored in the result."""
    async_cm, _ = _make_mock_client()
    fake_servers = [
        {"name": "controls", "status": "connected", "tools": []},
        {"name": "python", "status": "connected", "tools": []},
    ]

    with (
        patch("osprey.agent_runner.runner.ClaudeSDKClient", return_value=async_cm),
        patch(
            "osprey.agent_runner.primitives.await_mcp_ready",
            new=AsyncMock(return_value=fake_servers),
        ),
        patch("osprey.agent_runner.primitives.sdk_env", return_value={"CLAUDECODE": ""}),
        patch(
            "osprey.agent_runner.primitives.resolve_default_model",
            return_value="claude-haiku-4-5-20251001",
        ),
        patch(
            "osprey.agent_runner.primitives.expected_mcp_servers",
            return_value={"controls", "python"},
        ),
    ):
        result = await run_query(project_dir, "query", disallowed_tools=[])

    assert result.mcp_servers == fake_servers
    assert result.mcp_server_status == {"controls": "connected", "python": "connected"}


@pytest.mark.asyncio
async def test_run_query_wraps_sdk_exception(project_dir: Path) -> None:
    """SDK errors are re-raised as RuntimeError with a descriptive message."""
    broken_cm = MagicMock()
    broken_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("connection refused"))
    broken_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("osprey.agent_runner.runner.ClaudeSDKClient", return_value=broken_cm),
        patch("osprey.agent_runner.primitives.sdk_env", return_value={"CLAUDECODE": ""}),
        patch(
            "osprey.agent_runner.primitives.resolve_default_model",
            return_value="claude-haiku-4-5-20251001",
        ),
        patch(
            "osprey.agent_runner.primitives.expected_mcp_servers",
            return_value=set(),
        ),
    ):
        with pytest.raises(RuntimeError, match="SDK query failed"):
            await run_query(project_dir, "query", disallowed_tools=[])


# ---------------------------------------------------------------------------
# Tool-result parsing branches: list content, is_error, AssistantMessage-embedded
# ---------------------------------------------------------------------------


async def _list_content_stream() -> AsyncIterator:
    """A run where the tool result arrives as a list of content blocks and is an error.

    Also exercises the path where a ToolResultBlock is embedded directly in an
    AssistantMessage (rather than a UserMessage) — the SDK forwards it that way.
    """
    yield AssistantMessage(
        content=[ToolUseBlock(id=FAKE_TOOL_USE_ID, name=FAKE_TOOL_NAME, input=FAKE_TOOL_INPUT)],
        model="claude-haiku-4-5-20251001",
    )
    # Tool result embedded in an AssistantMessage, with list-shaped content and is_error.
    yield AssistantMessage(
        content=[
            ToolResultBlock(
                tool_use_id=FAKE_TOOL_USE_ID,
                content=[{"type": "text", "text": "channel offline"}],
                is_error=True,
            )
        ],
        model="claude-haiku-4-5-20251001",
    )
    yield ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=10,
        is_error=False,
        num_turns=1,
        session_id="sess-list-001",
    )


@pytest.mark.asyncio
async def test_run_query_parses_list_content_and_is_error(project_dir: Path) -> None:
    """List-shaped tool-result content is joined to text; is_error is captured;
    a ToolResultBlock embedded in an AssistantMessage is ingested."""
    client = MagicMock()
    client.query = AsyncMock(return_value=None)
    client.receive_response = MagicMock(return_value=_list_content_stream())
    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=client)
    async_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("osprey.agent_runner.runner.ClaudeSDKClient", return_value=async_cm),
        patch(
            "osprey.agent_runner.primitives.await_mcp_ready",
            new=AsyncMock(return_value=[]),
        ),
        patch("osprey.agent_runner.primitives.sdk_env", return_value={"CLAUDECODE": ""}),
        patch(
            "osprey.agent_runner.primitives.resolve_default_model",
            return_value="claude-haiku-4-5-20251001",
        ),
        patch("osprey.agent_runner.primitives.expected_mcp_servers", return_value=set()),
    ):
        result = await run_query(project_dir, "q", disallowed_tools=[])

    assert len(result.tool_traces) == 1
    trace = result.tool_traces[0]
    assert trace.result == "channel offline"
    assert trace.is_error is True


@pytest.mark.asyncio
async def test_run_query_raises_when_sdk_absent(project_dir: Path) -> None:
    """When claude_agent_sdk is not installed, run_query raises ImportError."""
    with patch("osprey.agent_runner.runner.HAS_SDK", False):
        with pytest.raises(ImportError, match="claude_agent_sdk is required"):
            await run_query(project_dir, "q", disallowed_tools=[])


# ---------------------------------------------------------------------------
# Translation proxy start for non-native (OpenAI-protocol) providers (#307)
# ---------------------------------------------------------------------------


class _FakeSpec:
    def __init__(
        self,
        *,
        needs_proxy: bool,
        auth_env_var: str = "ANTHROPIC_AUTH_TOKEN",
        upstream_base_url: str | None = "https://argo.example/v1",
        provider: str = "argo",
        supports_images: bool | None = None,
    ) -> None:
        self.needs_proxy = needs_proxy
        self.auth_env_var = auth_env_var
        self.upstream_base_url = upstream_base_url
        self.provider = provider
        self.supports_images = supports_images


def _capture(captured: list, async_cm):
    def _capture_client(options: ClaudeAgentOptions):
        captured.append(options)
        return async_cm

    return _capture_client


@pytest.mark.asyncio
async def test_run_query_starts_proxy_for_non_native_provider(project_dir: Path) -> None:
    """needs_proxy spec → start_proxy(spec.upstream_base_url, key-from-env-dict).

    The proxy upstream MUST come from spec.upstream_base_url (the OpenAI root
    with /v1), NOT from env["ANTHROPIC_BASE_URL"] — which the resolver strips of
    /v1 for Claude Code (issue #312). The env var here is deliberately the
    stripped form to prove the two are not conflated.
    """
    async_cm, _ = _make_mock_client()
    captured: list[ClaudeAgentOptions] = []
    proxy = MagicMock(return_value=8123)
    proxy_env = {
        "CLAUDECODE": "",
        "ANTHROPIC_BASE_URL": "https://argo.example",  # stripped (Claude-Code-facing)
        "ANTHROPIC_AUTH_TOKEN": "sk-argo",
        "ANTHROPIC_CUSTOM_HEADERS": "x-litellm-end-user-id: alice\nX-Corp-Trace: abc123",
    }

    with (
        patch(
            "osprey.agent_runner.runner.ClaudeSDKClient", side_effect=_capture(captured, async_cm)
        ),
        patch("osprey.agent_runner.primitives.await_mcp_ready", new=AsyncMock(return_value=[])),
        patch("osprey.agent_runner.primitives.sdk_env", return_value=proxy_env),
        patch("osprey.agent_runner.primitives.resolve_default_model", return_value="m"),
        patch(
            "osprey.agent_runner.primitives._resolve_project_spec",
            return_value=_FakeSpec(needs_proxy=True, upstream_base_url="https://argo.example/v1"),
        ),
        patch("osprey.agent_runner.primitives.start_proxy", proxy),
        patch("osprey.agent_runner.primitives.expected_mcp_servers", return_value=set()),
    ):
        await run_query(project_dir, "q", disallowed_tools=[])

    # Proxy upstream = spec.upstream_base_url (WITH /v1), NOT the stripped env var.
    # api_key sourced from the env dict (not os.environ) on this path.
    proxy.assert_called_once_with(
        "https://argo.example/v1",
        "sk-argo",
        provider="argo",
        forward_headers=frozenset({"x-litellm-end-user-id", "x-corp-trace"}),
        supports_images=None,
    )
    assert captured[0].env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8123"


@pytest.mark.asyncio
async def test_run_query_warns_when_proxy_auth_token_missing(project_dir: Path, caplog) -> None:
    """Proxy needed but auth token absent from env → warn (else it surfaces as a 401)."""
    import logging

    async_cm, _ = _make_mock_client()
    proxy = MagicMock(return_value=8123)
    # ANTHROPIC_BASE_URL present (proxy will start) but no ANTHROPIC_AUTH_TOKEN.
    proxy_env = {"CLAUDECODE": "", "ANTHROPIC_BASE_URL": "https://argo.example/v1"}

    with (
        patch("osprey.agent_runner.runner.ClaudeSDKClient", return_value=async_cm),
        patch("osprey.agent_runner.primitives.await_mcp_ready", new=AsyncMock(return_value=[])),
        patch("osprey.agent_runner.primitives.sdk_env", return_value=proxy_env),
        patch("osprey.agent_runner.primitives.resolve_default_model", return_value="m"),
        patch(
            "osprey.agent_runner.primitives._resolve_project_spec",
            return_value=_FakeSpec(needs_proxy=True, provider="argo"),
        ),
        patch("osprey.agent_runner.primitives.start_proxy", proxy),
        patch("osprey.agent_runner.primitives.expected_mcp_servers", return_value=set()),
        caplog.at_level(logging.WARNING, logger="osprey.agent_runner.primitives"),
    ):
        await run_query(project_dir, "q", disallowed_tools=[])

    # Proxy still starts (best-effort), but the user is warned about auth.
    proxy.assert_called_once()
    assert any(
        "ANTHROPIC_AUTH_TOKEN" in r.message and "argo" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    ), "expected a warning naming the missing auth var and provider"


@pytest.mark.asyncio
async def test_run_query_no_proxy_for_native_provider(project_dir: Path) -> None:
    """A native spec (needs_proxy=False) starts no proxy and leaves the base URL alone."""
    async_cm, _ = _make_mock_client()
    captured: list[ClaudeAgentOptions] = []
    proxy = MagicMock(return_value=9999)
    native_env = {"CLAUDECODE": "", "ANTHROPIC_BASE_URL": "https://api.example.com"}

    with (
        patch(
            "osprey.agent_runner.runner.ClaudeSDKClient", side_effect=_capture(captured, async_cm)
        ),
        patch("osprey.agent_runner.primitives.await_mcp_ready", new=AsyncMock(return_value=[])),
        patch("osprey.agent_runner.primitives.sdk_env", return_value=native_env),
        patch("osprey.agent_runner.primitives.resolve_default_model", return_value="m"),
        patch(
            "osprey.agent_runner.primitives._resolve_project_spec",
            return_value=_FakeSpec(needs_proxy=False),
        ),
        patch("osprey.agent_runner.primitives.start_proxy", proxy),
        patch("osprey.agent_runner.primitives.expected_mcp_servers", return_value=set()),
    ):
        await run_query(project_dir, "q", disallowed_tools=[])

    proxy.assert_not_called()
    assert captured[0].env["ANTHROPIC_BASE_URL"] == "https://api.example.com"


@pytest.mark.asyncio
async def test_run_query_no_proxy_when_upstream_absent(project_dir: Path) -> None:
    """needs_proxy spec but no upstream_base_url (base_url-less provider) → no proxy."""
    async_cm, _ = _make_mock_client()
    captured: list[ClaudeAgentOptions] = []
    proxy = MagicMock(return_value=1)

    with (
        patch(
            "osprey.agent_runner.runner.ClaudeSDKClient", side_effect=_capture(captured, async_cm)
        ),
        patch("osprey.agent_runner.primitives.await_mcp_ready", new=AsyncMock(return_value=[])),
        patch("osprey.agent_runner.primitives.sdk_env", return_value={"CLAUDECODE": ""}),
        patch("osprey.agent_runner.primitives.resolve_default_model", return_value="m"),
        patch(
            "osprey.agent_runner.primitives._resolve_project_spec",
            return_value=_FakeSpec(needs_proxy=True, upstream_base_url=None),
        ),
        patch("osprey.agent_runner.primitives.start_proxy", proxy),
        patch("osprey.agent_runner.primitives.expected_mcp_servers", return_value=set()),
    ):
        await run_query(project_dir, "q", disallowed_tools=[])

    proxy.assert_not_called()


# ---------------------------------------------------------------------------
# stream_query: event records, prompt shapes, the readiness set and SDK errors
# ---------------------------------------------------------------------------


def _routing_patches(async_cm: MagicMock | None = None, *, client_factory=None):
    """The patches every stream_query test needs: a fake client, no provider lookups."""
    from contextlib import ExitStack

    stack = ExitStack()
    if client_factory is not None:
        stack.enter_context(
            patch("osprey.agent_runner.runner.ClaudeSDKClient", side_effect=client_factory)
        )
    else:
        stack.enter_context(
            patch("osprey.agent_runner.runner.ClaudeSDKClient", return_value=async_cm)
        )
    stack.enter_context(
        patch("osprey.agent_runner.primitives.sdk_env", return_value={"CLAUDECODE": ""})
    )
    stack.enter_context(
        patch("osprey.agent_runner.primitives.resolve_default_model", return_value="m")
    )
    stack.enter_context(
        patch("osprey.agent_runner.primitives._resolve_project_spec", return_value=None)
    )
    return stack


async def _collect(project_dir: Path, prompt, **kwargs) -> list:
    return [event async for event in stream_query(project_dir, prompt, **kwargs)]


@pytest.mark.asyncio
async def test_stream_query_yields_event_records_in_stream_order(project_dir: Path) -> None:
    async_cm, _ = _make_mock_client()

    with (
        _routing_patches(async_cm),
        patch(
            "osprey.agent_runner.primitives.await_mcp_ready",
            new=AsyncMock(return_value=FAKE_MCP_SERVERS),
        ),
    ):
        events = await _collect(project_dir, "q", disallowed_tools=[])

    assert events == [
        ToolUseEvent(
            tool_use_id=FAKE_TOOL_USE_ID,
            name=FAKE_TOOL_NAME,
            input=FAKE_TOOL_INPUT,
            parent_tool_use_id=None,
        ),
        ToolResultEvent(
            tool_use_id=FAKE_TOOL_USE_ID,
            content=FAKE_TOOL_RESULT,
            is_error=False,
            parent_tool_use_id=None,
        ),
        TextEvent(text=FAKE_TEXT, parent_tool_use_id=None),
        ResultEvent(
            subtype="success",
            is_error=False,
            num_turns=2,
            duration_ms=500,
            session_id="sess-fake-001",
            total_cost_usd=0.001,
            usage=None,
            result=None,
            api_error_status=None,
        ),
    ]


@pytest.mark.asyncio
async def test_closing_the_stream_early_closes_the_client_before_returning(
    project_dir: Path,
) -> None:
    async_cm, _ = _make_mock_client()

    with (
        _routing_patches(async_cm),
        patch(
            "osprey.agent_runner.primitives.await_mcp_ready",
            new=AsyncMock(return_value=FAKE_MCP_SERVERS),
        ),
    ):
        stream = stream_query(project_dir, "q", disallowed_tools=[])
        first = await anext(stream)
        async_cm.__aexit__.assert_not_awaited()
        await stream.aclose()

    assert isinstance(first, ToolUseEvent)
    async_cm.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_query_sends_content_blocks_as_one_user_message(project_dir: Path) -> None:
    async_cm, client = _make_mock_client()
    sent: list = []

    async def _capture_query(prompt) -> None:
        sent.append([envelope async for envelope in prompt])

    client.query = AsyncMock(side_effect=_capture_query)
    blocks = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
        {"type": "text", "text": "what is in the picture?"},
    ]

    with _routing_patches(async_cm):
        await _collect(project_dir, blocks, disallowed_tools=[], await_mcp_servers=())

    assert sent == [[{"type": "user", "message": {"role": "user", "content": blocks}}]]


@pytest.mark.asyncio
async def test_stream_query_sends_a_text_prompt_as_a_string(project_dir: Path) -> None:
    async_cm, client = _make_mock_client()

    with _routing_patches(async_cm):
        await _collect(project_dir, "plain question", disallowed_tools=[], await_mcp_servers=())

    client.query.assert_awaited_once_with("plain question")


@pytest.mark.asyncio
async def test_an_empty_readiness_set_skips_the_barrier(project_dir: Path) -> None:
    async_cm, _ = _make_mock_client()
    barrier = AsyncMock(return_value=FAKE_MCP_SERVERS)
    statuses: list = []

    with (
        _routing_patches(async_cm),
        patch("osprey.agent_runner.primitives.await_mcp_ready", new=barrier),
    ):
        await _collect(
            project_dir,
            "q",
            disallowed_tools=[],
            await_mcp_servers=(),
            on_mcp_status=statuses.append,
        )

    barrier.assert_not_awaited()
    assert statuses == [[]]


@pytest.mark.asyncio
async def test_an_explicit_readiness_set_replaces_the_declared_one(project_dir: Path) -> None:
    async_cm, client = _make_mock_client()
    barrier = AsyncMock(return_value=[{"name": "python", "status": "connected"}])

    with (
        _routing_patches(async_cm),
        patch("osprey.agent_runner.primitives.await_mcp_ready", new=barrier),
    ):
        await _collect(project_dir, "q", disallowed_tools=[], await_mcp_servers=["python"])

    barrier.assert_awaited_once_with(client, {"python"})


@pytest.mark.asyncio
async def test_a_required_server_not_connected_refuses_before_the_prompt(
    project_dir: Path,
) -> None:
    async_cm, client = _make_mock_client()
    snapshot = [{"name": "controls", "status": "pending"}]
    order: list[str] = []

    def _on_status(_servers: list) -> None:
        order.append("status")

    client.query = AsyncMock(side_effect=lambda *_a: order.append("query"))

    with (
        _routing_patches(async_cm),
        patch(
            "osprey.agent_runner.primitives.await_mcp_ready", new=AsyncMock(return_value=snapshot)
        ),
        pytest.raises(McpNotReadyError) as refused,
    ):
        await _collect(
            project_dir,
            "q",
            disallowed_tools=[],
            require_mcp_servers={"controls"},
            on_mcp_status=_on_status,
        )

    assert "controls (pending)" in str(refused.value)
    assert refused.value.servers == snapshot
    assert refused.value.missing == ["controls"]
    assert order == ["status"]
    client.query.assert_not_awaited()
    async_cm.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failed_required_server_is_named_with_its_error(project_dir: Path) -> None:
    async_cm, _ = _make_mock_client()
    snapshot = [{"name": "controls", "status": "failed", "error": "spawn ENOENT"}]

    with (
        _routing_patches(async_cm),
        patch(
            "osprey.agent_runner.primitives.await_mcp_ready", new=AsyncMock(return_value=snapshot)
        ),
        pytest.raises(McpNotReadyError, match=r"controls \(failed: spawn ENOENT\)"),
    ):
        await _collect(project_dir, "q", disallowed_tools=[], require_mcp_servers=["controls"])


@pytest.mark.asyncio
async def test_an_optional_server_not_connected_is_logged_and_the_run_proceeds(
    project_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    async_cm, client = _make_mock_client()
    snapshot = [
        {"name": "controls", "status": "connected"},
        {"name": "python", "status": "failed"},
    ]

    with (
        _routing_patches(async_cm),
        patch(
            "osprey.agent_runner.primitives.await_mcp_ready", new=AsyncMock(return_value=snapshot)
        ),
        caplog.at_level(logging.WARNING, logger="osprey.agent_runner.primitives"),
    ):
        events = await _collect(
            project_dir,
            "q",
            disallowed_tools=[],
            await_mcp_servers={"controls", "python"},
            require_mcp_servers={"controls"},
        )

    client.query.assert_awaited_once()
    assert isinstance(events[-1], ResultEvent)
    assert any("python" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


@pytest.mark.asyncio
async def test_agent_sdk_errors_surface_as_agent_run_errors(project_dir: Path) -> None:
    from claude_agent_sdk import ProcessError

    original = ProcessError("CLI exited", exit_code=1, stderr="boom")

    async def _failing_stream():
        raise original
        yield  # pragma: no cover - makes this an async generator

    async_cm, client = _make_mock_client()
    client.receive_response = MagicMock(return_value=_failing_stream())

    with _routing_patches(async_cm), pytest.raises(AgentRunError) as failed:
        await _collect(project_dir, "q", disallowed_tools=[], await_mcp_servers=())

    assert failed.value.error_type == "ProcessError"
    assert str(failed.value) == str(original)
    assert failed.value.__cause__ is original


@pytest.mark.asyncio
async def test_run_query_passes_caller_options_through(project_dir: Path) -> None:
    async_cm, _ = _make_mock_client()
    captured: list[ClaudeAgentOptions] = []

    def _sink(_line: str) -> None:
        return None

    caller_env = {"ANTHROPIC_BASE_URL": "https://caller.example"}

    with _routing_patches(client_factory=_capture(captured, async_cm)):
        await run_query(
            project_dir,
            "q",
            disallowed_tools=[],
            allowed_tools=["mcp__channel-finder__*"],
            system_prompt="find channels",
            setting_sources=[],
            stderr=_sink,
            env=caller_env,
            await_mcp_servers=(),
        )

    [options] = captured
    assert options.allowed_tools == ["mcp__channel-finder__*"]
    assert options.system_prompt == "find channels"
    assert options.setting_sources == []
    assert options.stderr is _sink
    assert options.env == caller_env


@pytest.mark.asyncio
async def test_run_query_with_no_readiness_set_does_not_poll(project_dir: Path) -> None:
    async_cm, client = _make_mock_client()
    client.get_mcp_status = AsyncMock(return_value={"mcpServers": []})

    with _routing_patches(async_cm):
        result = await run_query(project_dir, "q", disallowed_tools=[], await_mcp_servers=())

    client.get_mcp_status.assert_not_awaited()
    assert result.mcp_servers == []
    assert result.text_blocks == [FAKE_TEXT]
