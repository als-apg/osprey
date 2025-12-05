"""Tests for Gateway approval detection with explicit yes/no checks."""

from unittest.mock import MagicMock, patch

import pytest

from osprey.infrastructure.gateway import Gateway


@pytest.fixture
def gateway():
    """Create a Gateway instance for testing."""
    return Gateway()


class TestExplicitApprovalDetection:
    """Test explicit yes/no detection in Gateway."""

    def test_explicit_yes(self, gateway):
        """Test that 'yes' is detected as approval without LLM call."""
        with patch("osprey.infrastructure.gateway.get_chat_completion") as mock_llm:
            result = gateway._detect_approval_response("yes")

            # Should not call LLM for simple "yes"
            mock_llm.assert_not_called()

            assert result is not None
            assert result["approved"] is True
            assert result["type"] == "approval"

    def test_explicit_yes_with_punctuation(self, gateway):
        """Test that 'yes!' and 'yes.' are detected as approval."""
        test_cases = ["yes!", "yes.", "yes?", "Yes!", "YES."]

        with patch("osprey.infrastructure.gateway.get_chat_completion") as mock_llm:
            for test_input in test_cases:
                result = gateway._detect_approval_response(test_input)

                # Should not call LLM for simple "yes" variations
                assert mock_llm.call_count == 0, f"LLM called for '{test_input}'"

                assert result is not None, f"No result for '{test_input}'"
                assert result["approved"] is True, f"Not approved for '{test_input}'"
                assert result["type"] == "approval", f"Wrong type for '{test_input}'"

    def test_explicit_yes_variations(self, gateway):
        """Test various explicit approval words."""
        test_cases = ["y", "yep", "yeah", "ok", "okay", "OK!", "OKAY."]

        with patch("osprey.infrastructure.gateway.get_chat_completion") as mock_llm:
            for test_input in test_cases:
                result = gateway._detect_approval_response(test_input)

                # Should not call LLM for simple approval variations
                assert mock_llm.call_count == 0, f"LLM called for '{test_input}'"

                assert result is not None, f"No result for '{test_input}'"
                assert result["approved"] is True, f"Not approved for '{test_input}'"
                assert result["type"] == "approval", f"Wrong type for '{test_input}'"

    def test_explicit_no(self, gateway):
        """Test that 'no' is detected as rejection without LLM call."""
        with patch("osprey.infrastructure.gateway.get_chat_completion") as mock_llm:
            result = gateway._detect_approval_response("no")

            # Should not call LLM for simple "no"
            mock_llm.assert_not_called()

            assert result is not None
            assert result["approved"] is False
            assert result["type"] == "rejection"

    def test_explicit_no_with_punctuation(self, gateway):
        """Test that 'no!' and 'no.' are detected as rejection."""
        test_cases = ["no!", "no.", "no?", "No!", "NO."]

        with patch("osprey.infrastructure.gateway.get_chat_completion") as mock_llm:
            for test_input in test_cases:
                result = gateway._detect_approval_response(test_input)

                # Should not call LLM for simple "no" variations
                assert mock_llm.call_count == 0, f"LLM called for '{test_input}'"

                assert result is not None, f"No result for '{test_input}'"
                assert result["approved"] is False, f"Incorrectly approved for '{test_input}'"
                assert result["type"] == "rejection", f"Wrong type for '{test_input}'"

    def test_explicit_no_variations(self, gateway):
        """Test various explicit rejection words."""
        test_cases = ["n", "nope", "nah", "cancel", "CANCEL!", "Nope."]

        with patch("osprey.infrastructure.gateway.get_chat_completion") as mock_llm:
            for test_input in test_cases:
                result = gateway._detect_approval_response(test_input)

                # Should not call LLM for simple rejection variations
                assert mock_llm.call_count == 0, f"LLM called for '{test_input}'"

                assert result is not None, f"No result for '{test_input}'"
                assert result["approved"] is False, f"Incorrectly approved for '{test_input}'"
                assert result["type"] == "rejection", f"Wrong type for '{test_input}'"

    def test_complex_response_uses_llm(self, gateway):
        """Test that complex responses fall back to LLM detection."""
        with patch("osprey.infrastructure.gateway.get_chat_completion") as mock_llm:
            with patch("osprey.infrastructure.gateway.get_model_config") as mock_config:
                # Setup mocks
                mock_config.return_value = {"model": "test-model"}
                mock_llm.return_value = MagicMock(approved=True)

                result = gateway._detect_approval_response("I think we should proceed with this")

                # Should call LLM for complex response
                mock_llm.assert_called_once()

                assert result is not None
                assert result["approved"] is True

    def test_whitespace_handling(self, gateway):
        """Test that whitespace doesn't affect detection."""
        test_cases = ["  yes  ", "\tyes\t", "\nyes\n", "  yes!  ", "  no.  "]
        expected_results = [True, True, True, True, False]

        with patch("osprey.infrastructure.gateway.get_chat_completion") as mock_llm:
            for test_input, expected_approved in zip(test_cases, expected_results):
                result = gateway._detect_approval_response(test_input)

                # Should not call LLM for simple variations with whitespace
                assert mock_llm.call_count == 0, f"LLM called for '{test_input}'"

                assert result is not None, f"No result for '{test_input}'"
                assert result["approved"] == expected_approved, f"Wrong approval for '{test_input}'"

    def test_case_insensitivity(self, gateway):
        """Test that case doesn't matter for explicit yes/no."""
        test_cases = ["YES", "Yes", "yEs", "NO", "No", "nO"]
        expected_results = [True, True, True, False, False, False]

        with patch("osprey.infrastructure.gateway.get_chat_completion") as mock_llm:
            for test_input, expected_approved in zip(test_cases, expected_results):
                result = gateway._detect_approval_response(test_input)

                # Should not call LLM for case variations
                assert mock_llm.call_count == 0, f"LLM called for '{test_input}'"

                assert result is not None, f"No result for '{test_input}'"
                assert result["approved"] == expected_approved, f"Wrong approval for '{test_input}'"
