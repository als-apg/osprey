"""Tests for the two-stage evaluation pipeline.

Tests cover:
  - programmatic_recall_check: substring matching (full, partial, case-insensitive)
  - compute_f1: edge cases (perfect, both empty, one empty, partial, case-insensitive)
  - evaluate_response: stage 1 failure path and stage 2 with mocked LLM
  - llm_extract_channels: mocked LLM structured output
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osprey.services.channel_finder.benchmarks.evaluation import (
    ChannelExtractionResult,
    compute_f1,
    evaluate_response,
    llm_extract_channels,
    programmatic_recall_check,
)

# ---------------------------------------------------------------------------
# programmatic_recall_check
# ---------------------------------------------------------------------------


class TestProgrammaticRecallCheck:
    """Tests for stage-1 programmatic recall."""

    def test_all_found(self):
        """All expected channels present in text."""
        text = (
            "The recommended channels are SR:MAG:DIPOLE:B05:CURRENT:SP "
            "and SR:MAG:QUAD:Q1:CURRENT:RB."
        )
        expected = [
            "SR:MAG:DIPOLE:B05:CURRENT:SP",
            "SR:MAG:QUAD:Q1:CURRENT:RB",
        ]
        found, missing = programmatic_recall_check(text, expected)
        assert found == expected
        assert missing == []

    def test_partial(self):
        """Some expected channels missing from text."""
        text = "Found channel SR:MAG:DIPOLE:B05:CURRENT:SP in the system."
        expected = [
            "SR:MAG:DIPOLE:B05:CURRENT:SP",
            "SR:MAG:QUAD:Q1:CURRENT:RB",
        ]
        found, missing = programmatic_recall_check(text, expected)
        assert found == ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        assert missing == ["SR:MAG:QUAD:Q1:CURRENT:RB"]

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        text = "The channel sr:mag:dipole:b05:current:sp is available."
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        found, missing = programmatic_recall_check(text, expected)
        assert found == expected
        assert missing == []

    def test_none_found(self):
        """No expected channels in text."""
        text = "No relevant channels were found."
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        found, missing = programmatic_recall_check(text, expected)
        assert found == []
        assert missing == expected

    def test_empty_expected(self):
        """Empty expected list returns empty found and missing."""
        found, missing = programmatic_recall_check("some text", [])
        assert found == []
        assert missing == []


# ---------------------------------------------------------------------------
# compute_f1
# ---------------------------------------------------------------------------


class TestComputeF1:
    """Tests for precision/recall/F1 computation."""

    def test_perfect(self):
        """Predicted equals expected -> (1.0, 1.0, 1.0)."""
        channels = ["A", "B", "C"]
        precision, recall, f1 = compute_f1(channels, channels)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_empty_both(self):
        """Both empty -> (1.0, 1.0, 1.0)."""
        precision, recall, f1 = compute_f1([], [])
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_empty_predicted(self):
        """Predicted empty, expected non-empty -> (0.0, 0.0, 0.0)."""
        precision, recall, f1 = compute_f1([], ["A", "B"])
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_empty_expected(self):
        """Expected empty, predicted non-empty -> (0.0, 0.0, 0.0)."""
        precision, recall, f1 = compute_f1(["A", "B"], [])
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_partial(self):
        """Partial overlap gives correct precision/recall/F1."""
        predicted = ["A", "B", "C"]  # 3 predicted
        expected = ["A", "B", "D"]  # 3 expected, 2 overlap (A, B)

        precision, recall, f1 = compute_f1(predicted, expected)
        # tp=2, precision=2/3, recall=2/3, f1=2/3
        assert precision == pytest.approx(2 / 3)
        assert recall == pytest.approx(2 / 3)
        assert f1 == pytest.approx(2 / 3)

    def test_precision_and_recall_differ(self):
        """Different precision and recall values."""
        predicted = ["A", "B"]  # 2 predicted
        expected = ["A", "B", "C", "D"]  # 4 expected, 2 overlap

        precision, recall, f1 = compute_f1(predicted, expected)
        # tp=2, precision=2/2=1.0, recall=2/4=0.5
        assert precision == 1.0
        assert recall == 0.5
        expected_f1 = 2 * 1.0 * 0.5 / (1.0 + 0.5)
        assert f1 == pytest.approx(expected_f1)

    def test_case_insensitive(self):
        """F1 scoring is case-insensitive (EPICS PV convention)."""
        predicted = ["sr:mag:dipole:b05:current:sp"]
        expected = ["SR:MAG:DIPOLE:B05:CURRENT:SP"]
        precision, recall, f1 = compute_f1(predicted, expected)
        assert f1 == 1.0


# ---------------------------------------------------------------------------
# llm_extract_channels (mocked)
# ---------------------------------------------------------------------------


class TestLlmExtractChannels:
    """Tests for LLM-based channel extraction with mocked completions."""

    @patch("osprey.models.providers.litellm_adapter.execute_litellm_completion")
    def test_returns_extracted_channels(self, mock_completion):
        """Structured output returns recommended channels."""
        mock_completion.return_value = ChannelExtractionResult(
            recommended_channels=["CH:A", "CH:B"],
            reasoning="These are the final channels.",
        )
        result = llm_extract_channels("some response text", ["CH:A", "CH:B"])
        assert result == ["CH:A", "CH:B"]
        mock_completion.assert_called_once()

    @patch("osprey.models.providers.litellm_adapter.execute_litellm_completion")
    def test_non_pydantic_returns_empty(self, mock_completion):
        """Non-Pydantic return value falls back to empty list."""
        mock_completion.return_value = "raw string response"
        result = llm_extract_channels("some response", ["CH:A"])
        assert result == []


# ---------------------------------------------------------------------------
# evaluate_response
# ---------------------------------------------------------------------------


class TestEvaluateResponse:
    """Tests for the combined two-stage evaluation pipeline."""

    def test_missing_channels_skips_llm(self):
        """Stage 1 failure path: missing channels, no LLM call."""
        text = "Found channel CH:A in the response."
        expected = ["CH:A", "CH:B"]

        predicted, meta = evaluate_response(text, expected)

        assert predicted == ["CH:A"]
        assert meta["stage"] == 1
        assert meta["evaluation"] == "programmatic_recall_fail"
        assert meta["found"] == ["CH:A"]
        assert meta["missing"] == ["CH:B"]

        # Verify precision contract: stage-1 fallback always yields precision=1.0
        # because `found` is a subset of `expected` by construction.
        precision, recall, f1 = compute_f1(predicted, expected)
        assert precision == 1.0
        assert recall < 1.0

    @patch("osprey.services.channel_finder.benchmarks.evaluation.llm_extract_channels")
    def test_all_found_invokes_llm_judge(self, mock_llm):
        """Stage 2 path: all expected found AND opt-in flag set → LLM judge invoked."""
        mock_llm.return_value = ["CH:A", "CH:B"]
        text = "The channels are CH:A and CH:B in the final answer."
        expected = ["CH:A", "CH:B"]

        predicted, meta = evaluate_response(text, expected, use_llm_judge=True)

        assert predicted == ["CH:A", "CH:B"]
        assert meta["stage"] == 2
        assert meta["evaluation"] == "llm_judge"
        assert meta["llm_extracted"] == ["CH:A", "CH:B"]
        mock_llm.assert_called_once_with(text, expected)

    @patch("osprey.services.channel_finder.benchmarks.evaluation.llm_extract_channels")
    def test_llm_error_falls_back_to_found(self, mock_llm):
        """LLM judge failure falls back to programmatic found list (opt-in path)."""
        mock_llm.side_effect = RuntimeError("API error")
        text = "Channels CH:A and CH:B are recommended."
        expected = ["CH:A", "CH:B"]

        predicted, meta = evaluate_response(text, expected, use_llm_judge=True)

        assert predicted == ["CH:A", "CH:B"]
        assert meta["stage"] == 2
        assert meta["evaluation"] == "llm_judge_error"
        assert "API error" in meta["llm_error"]

    def test_no_channels_found(self):
        """No expected channels found in text at all."""
        text = "I could not find any relevant channels."
        expected = ["CH:A", "CH:B"]

        predicted, meta = evaluate_response(text, expected)

        assert predicted == []
        assert meta["stage"] == 1
        assert meta["evaluation"] == "programmatic_recall_fail"
        assert meta["missing"] == ["CH:A", "CH:B"]
