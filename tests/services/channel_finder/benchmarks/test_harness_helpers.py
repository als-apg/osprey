"""Unit tests for the pure helpers in the benchmark harness.

Only the provider-free, SDK-free logic is exercised here: LiteLLM model-string
routing (including the Ollama OpenAI-compat remapping foot-gun), MCP->OpenAI
tool conversion, and ReAct result text aggregation. The async MCP/LLM driving
paths are the human-babysat benchmark surface and are not unit-tested.
"""

from __future__ import annotations

from types import SimpleNamespace

from osprey.agent_runner import ToolTrace
from osprey.services.channel_finder.benchmarks import harness as mod
from osprey.services.channel_finder.benchmarks.harness import (
    ReactWorkflowResult,
    _litellm_call_kwargs,
    _litellm_model_name,
    _mcp_tool_to_openai,
    combined_text_from_react,
)


class TestLitellmCallKwargs:
    def test_ollama_remapped_to_openai_endpoint(self):
        kwargs = _litellm_call_kwargs("ollama/llama3")
        assert kwargs["model"] == "openai/llama3"
        assert kwargs["api_base"] == mod.OLLAMA_OPENAI_BASE
        assert kwargs["api_key"] == "ollama-local"

    def test_ollama_chat_prefix_also_remapped(self):
        kwargs = _litellm_call_kwargs("ollama_chat/qwen2.5")
        assert kwargs["model"] == "openai/qwen2.5"
        assert kwargs["api_base"] == mod.OLLAMA_OPENAI_BASE

    def test_non_ollama_model_passthrough(self):
        assert _litellm_call_kwargs("anthropic/claude-x") == {"model": "anthropic/claude-x"}

    def test_model_name_helper_unwraps_kwargs(self):
        assert _litellm_model_name("ollama/llama3") == "openai/llama3"
        assert _litellm_model_name("gpt-4o") == "gpt-4o"


class TestMcpToolToOpenai:
    def test_conversion_shape(self):
        tool = SimpleNamespace(
            name="find_channel",
            description="Find a channel",
            inputSchema={"type": "object", "properties": {}},
        )
        result = _mcp_tool_to_openai(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "find_channel"
        assert result["function"]["description"] == "Find a channel"
        assert result["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_missing_description_becomes_empty_string(self):
        tool = SimpleNamespace(name="t", description=None, inputSchema={})
        result = _mcp_tool_to_openai(tool)
        assert result["function"]["description"] == ""


class TestCombinedTextFromReact:
    def test_combines_text_blocks_and_tool_results_lowercased(self):
        result = ReactWorkflowResult(
            text_blocks=["Found CHANNEL", "Second Block"],
            tool_traces=[
                ToolTrace(name="search", input={}, result="SR01:BPM:X"),
                ToolTrace(name="empty", input={}, result=None),
            ],
        )
        combined = combined_text_from_react(result)
        assert "found channel" in combined
        assert "second block" in combined
        assert "sr01:bpm:x" in combined
        # None result is skipped, not stringified.
        assert "none" not in combined

    def test_empty_result_is_empty_string(self):
        assert combined_text_from_react(ReactWorkflowResult()) == ""

    def test_dataclass_defaults(self):
        result = ReactWorkflowResult()
        assert result.tool_traces == []
        assert result.text_blocks == []
        assert result.cost_usd == 0.0
        assert result.num_turns == 0
