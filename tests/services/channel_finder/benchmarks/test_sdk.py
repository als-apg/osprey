"""Tests for SDK helpers extracted to benchmarks.sdk."""

from __future__ import annotations

from osprey.services.channel_finder.benchmarks.sdk import (
    SDKWorkflowResult,
    ToolTrace,
    combined_text,
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
