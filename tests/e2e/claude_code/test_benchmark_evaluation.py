"""Unit tests for the benchmark evaluation helpers.

These tests validate the two-stage evaluation pipeline
(programmatic recall check → LLM precision judge) without
making any real API calls.
"""

from __future__ import annotations

from unittest.mock import patch

from tests.e2e.claude_code.test_channel_finder_mcp_benchmarks import (
    ChannelExtractionResult,
    evaluate_channel_response,
    get_response_text,
    programmatic_recall_check,
)
from tests.e2e.sdk_helpers import SDKWorkflowResult, ToolTrace


def _make_result(text_blocks: list[str], tool_traces: list[ToolTrace] | None = None):
    """Create a minimal SDKWorkflowResult for testing."""
    result = SDKWorkflowResult()
    result.text_blocks = text_blocks
    if tool_traces:
        result.tool_traces = tool_traces
    return result


# ---------------------------------------------------------------------------
# get_response_text
# ---------------------------------------------------------------------------


class TestGetResponseText:
    def test_preserves_case(self):
        result = _make_result(["Here is SR:MAG:DIPOLE:B05:CURRENT:SP"])
        text = get_response_text(result)
        assert "SR:MAG:DIPOLE:B05:CURRENT:SP" in text

    def test_concatenates_text_and_tool_results(self):
        traces = [ToolTrace(name="get_options", input={}, result="Channel: FOO:BAR")]
        result = _make_result(["Agent text here."], tool_traces=traces)
        text = get_response_text(result)
        assert "Agent text here." in text
        assert "Channel: FOO:BAR" in text

    def test_skips_none_tool_results(self):
        traces = [ToolTrace(name="get_options", input={}, result=None)]
        result = _make_result(["Only text."], tool_traces=traces)
        text = get_response_text(result)
        assert text == "Only text."


# ---------------------------------------------------------------------------
# programmatic_recall_check
# ---------------------------------------------------------------------------


class TestProgrammaticRecallCheck:
    def test_all_found(self):
        text = "The channels are SR:MAG:DIPOLE:B05:CURRENT:SP and SR:RF:CAVITY:C2:TUNER:RB."
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP", "SR:RF:CAVITY:C2:TUNER:RB"]
        found, missing = programmatic_recall_check(text, expected)
        assert found == expected
        assert missing == []

    def test_partial(self):
        text = "Found SR:MAG:DIPOLE:B05:CURRENT:SP but not the other."
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP", "SR:RF:CAVITY:C2:TUNER:RB"]
        found, missing = programmatic_recall_check(text, expected)
        assert found == ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        assert missing == ["SR:RF:CAVITY:C2:TUNER:RB"]

    def test_case_insensitive(self):
        text = "channel is sr:mag:dipole:b05:current:sp in lowercase"
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        found, missing = programmatic_recall_check(text, expected)
        assert found == ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        assert missing == []

    def test_none_found(self):
        text = "No relevant channels here."
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        found, missing = programmatic_recall_check(text, expected)
        assert found == []
        assert missing == ["SR:MAG:DIPOLE:B05:CURRENT:SP"]


# ---------------------------------------------------------------------------
# evaluate_channel_response (integrated two-stage)
# ---------------------------------------------------------------------------


class TestEvaluateChannelResponse:
    def test_recall_fail_skips_llm(self):
        """Stage 1 failure: missing channels → no LLM call."""
        result = _make_result(["Found SR:MAG:DIPOLE:B05:CURRENT:SP only."])
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP", "SR:RF:CAVITY:C2:TUNER:RB"]

        predicted, meta = evaluate_channel_response(result, expected)
        assert predicted == ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        assert meta["stage"] == 1
        assert meta["evaluation"] == "programmatic_recall_fail"
        assert "SR:RF:CAVITY:C2:TUNER:RB" in meta["missing"]

    @patch(
        "osprey.models.providers.litellm_adapter.execute_litellm_completion",
    )
    def test_llm_judge_called_when_all_found(self, mock_completion):
        """Stage 2: all channels in text → LLM judge extracts final list."""
        mock_completion.return_value = ChannelExtractionResult(
            recommended_channels=["SR:MAG:DIPOLE:B05:CURRENT:SP", "SR:RF:CAVITY:C2:TUNER:RB"],
            reasoning="Both channels are in the final recommendation.",
        )

        result = _make_result(
            [
                "After exploring, the final channels are:\n"
                "- SR:MAG:DIPOLE:B05:CURRENT:SP\n"
                "- SR:RF:CAVITY:C2:TUNER:RB"
            ]
        )
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP", "SR:RF:CAVITY:C2:TUNER:RB"]

        predicted, meta = evaluate_channel_response(result, expected)
        assert predicted == ["SR:MAG:DIPOLE:B05:CURRENT:SP", "SR:RF:CAVITY:C2:TUNER:RB"]
        assert meta["stage"] == 2
        assert meta["evaluation"] == "llm_judge"
        mock_completion.assert_called_once()

    @patch(
        "osprey.models.providers.litellm_adapter.execute_litellm_completion",
    )
    def test_llm_judge_error_falls_back(self, mock_completion):
        """Stage 2 error: LLM call fails → fall back to found channels."""
        mock_completion.side_effect = RuntimeError("API error")

        result = _make_result(["SR:MAG:DIPOLE:B05:CURRENT:SP is the final channel."])
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP"]

        predicted, meta = evaluate_channel_response(result, expected)
        assert predicted == ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        assert meta["stage"] == 2
        assert meta["evaluation"] == "llm_judge_error"
        assert "API error" in meta["llm_error"]
