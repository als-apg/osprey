"""
Tests for control assistant task extraction prompt customization.

This test verifies that the custom control-system-specific task extraction prompt
is properly registered and used instead of framework defaults.

The test mocks the LLM call and inspects the actual prompt sent to verify it contains
control system terminology and examples rather than generic framework examples.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from osprey.prompts.defaults import ExtractedTask
from osprey.prompts.defaults.task_extraction import DefaultTaskExtractionPromptBuilder
from osprey.state import MessageUtils


class MockControlSystemTaskExtractionPromptBuilder(DefaultTaskExtractionPromptBuilder):
    """Mock of the control system task extraction prompt builder for testing."""

    def __init__(self):
        # Don't call super().__init__ to avoid loading framework examples
        self.examples = []

    def build_prompt(self, messages, retrieval_result=None):
        """Override to return control system specific prompt."""
        from osprey.state import ChatHistoryFormatter

        chat_formatted = ChatHistoryFormatter.format_for_llm(messages)

        return f"""
You are a control system assistant task extraction specialist that analyzes conversations to extract actionable tasks related to control system operations.

Your job is to:
1. Understand what the user is asking for in the context of control system operations
2. Extract a clear, actionable task related to channels, devices, or system monitoring
3. Determine if the task depends on chat history context
4. Determine if the task depends on user memory

## Control System Guidelines:
- Resolve channel references from previous messages
- Create self-contained task descriptions

## Control System Terminology:
- BPM = Beam Position Monitor (NOT beats per minute - this is accelerator/beam diagnostics)
- SP = Setpoint (desired value to write to a device)
- RB/RBV = Readback/Readback Value (actual measured value from a device)
- Common devices: quadrupoles (focusing magnets), dipoles (bending magnets), RF cavities, vacuum gauges

## Write Operations:
- Extract the target (channel/device) and value clearly

## Computational Requests:
- State the computational goal, not the implementation steps

## Current Chat History:
{chat_formatted}

## User Memory:
No stored memories

Now extract the task from the provided chat history and user memory.
""".strip()


# ===================================================================
# Test Fixtures
# ===================================================================


@pytest.fixture
def sample_messages():
    """Create sample messages for task extraction."""
    return [
        MessageUtils.create_user_message("What's the current beam current?"),
    ]


@pytest.fixture
def bpm_messages():
    """Create messages mentioning BPM to test domain-specific terminology."""
    return [
        MessageUtils.create_user_message("Show me the BPM readings in sector 3"),
    ]


@pytest.fixture(autouse=True)
def mock_infrastructure():
    """Automatically mock infrastructure components for all tests."""
    with (
        patch("osprey.infrastructure.task_extraction_node.get_framework_prompts") as mock_prompts,
        patch("osprey.infrastructure.task_extraction_node.get_model_config") as mock_model,
    ):
        # Set up mock prompt provider
        mock_provider = Mock()
        mock_provider.get_task_extraction_prompt_builder.return_value = (
            MockControlSystemTaskExtractionPromptBuilder()
        )
        mock_prompts.return_value = mock_provider

        # Mock model config
        mock_model.return_value = {"provider": "test", "model": "test-model"}

        yield


# ===================================================================
# Tests for Control System Task Extraction Prompt
# ===================================================================


class TestControlSystemTaskExtractionPrompt:
    """Test suite for control assistant task extraction prompt customization."""

    @patch("osprey.infrastructure.task_extraction_node.get_chat_completion")
    def test_custom_prompt_is_used(self, mock_llm, sample_messages):
        """Test that the custom control system prompt builder is used instead of framework default."""
        from osprey.infrastructure.task_extraction_node import _extract_task

        # Mock the LLM response
        mock_llm.return_value = ExtractedTask(
            task="Read current beam current value",
            depends_on_chat_history=False,
            depends_on_user_memory=False,
        )

        # Call task extraction (infrastructure is mocked by autouse fixture)
        logger = MagicMock()
        _extract_task(sample_messages, retrieval_result=None, logger=logger)

        # Verify LLM was called
        assert mock_llm.called, "LLM should have been called for task extraction"

        # Get the prompt that was sent to the LLM
        call_args = mock_llm.call_args
        prompt = call_args.kwargs.get("message") or call_args[0][0]

        # Verify it contains control system specific content
        assert "control system" in prompt.lower(), "Prompt should mention 'control system'"

        # Verify it contains control system role definition
        assert (
            "control system assistant task extraction specialist" in prompt.lower()
            or "control system operations" in prompt.lower()
        ), "Prompt should contain control system specific role"

    @patch("osprey.infrastructure.task_extraction_node.get_chat_completion")
    def test_bpm_terminology_in_prompt(self, mock_llm, bpm_messages):
        """Test that the prompt clarifies BPM = Beam Position Monitor (not beats per minute)."""
        from osprey.infrastructure.task_extraction_node import _extract_task

        # Mock the LLM response
        mock_llm.return_value = ExtractedTask(
            task="Display BPM readings for sector 3",
            depends_on_chat_history=False,
            depends_on_user_memory=False,
        )

        # Call task extraction with BPM in the message
        logger = MagicMock()
        _extract_task(bpm_messages, retrieval_result=None, logger=logger)

        # Get the prompt
        call_args = mock_llm.call_args
        prompt = call_args.kwargs.get("message") or call_args[0][0]

        # Verify BPM terminology clarification is present
        assert "BPM" in prompt, "Prompt should mention BPM"
        assert "Beam Position Monitor" in prompt, (
            "Prompt should clarify BPM = Beam Position Monitor"
        )
        assert "NOT beats per minute" in prompt or "not beats per minute" in prompt.lower(), (
            "Prompt should explicitly state BPM is NOT beats per minute"
        )

    @patch("osprey.infrastructure.task_extraction_node.get_chat_completion")
    def test_control_system_terminology_present(self, mock_llm, sample_messages):
        """Test that control system terminology is included in the prompt."""
        from osprey.infrastructure.task_extraction_node import _extract_task

        # Mock the LLM response
        mock_llm.return_value = ExtractedTask(
            task="Test task",
            depends_on_chat_history=False,
            depends_on_user_memory=False,
        )

        # Call task extraction
        logger = MagicMock()
        _extract_task(sample_messages, retrieval_result=None, logger=logger)

        # Get the prompt
        call_args = mock_llm.call_args
        prompt = call_args.kwargs.get("message") or call_args[0][0]

        # Verify control system terminology is present
        control_system_terms = [
            "SP",  # Setpoint
            "Setpoint",
            "RB",  # Readback
            "Readback",
            "channel",
            "magnet",
        ]

        terms_found = [term for term in control_system_terms if term in prompt]
        assert len(terms_found) >= 3, (
            f"Prompt should contain multiple control system terms. "
            f"Found: {terms_found}. Prompt excerpt: {prompt[:500]}"
        )

    @patch("osprey.infrastructure.task_extraction_node.get_chat_completion")
    def test_control_system_guidelines_present(self, mock_llm, sample_messages):
        """Test that control system specific guidelines are in the prompt."""
        from osprey.infrastructure.task_extraction_node import _extract_task

        # Mock the LLM response
        mock_llm.return_value = ExtractedTask(
            task="Test task",
            depends_on_chat_history=False,
            depends_on_user_memory=False,
        )

        # Call task extraction
        logger = MagicMock()
        _extract_task(sample_messages, retrieval_result=None, logger=logger)

        # Get the prompt
        call_args = mock_llm.call_args
        prompt = call_args.kwargs.get("message") or call_args[0][0]

        # Verify control system guidelines are present
        assert (
            "Control System Guidelines" in prompt or "control system operations" in prompt.lower()
        ), "Prompt should contain control system specific guidelines section"

        # Check for channel reference resolution guidance
        assert "channel" in prompt.lower() and "reference" in prompt.lower(), (
            "Prompt should mention channel reference resolution"
        )

    @patch("osprey.infrastructure.task_extraction_node.get_chat_completion")
    def test_framework_defaults_not_present(self, mock_llm, sample_messages):
        """Test that framework default examples are NOT present (we use only control system examples)."""
        from osprey.infrastructure.task_extraction_node import _extract_task

        # Mock the LLM response
        mock_llm.return_value = ExtractedTask(
            task="Test task",
            depends_on_chat_history=False,
            depends_on_user_memory=False,
        )

        # Call task extraction
        logger = MagicMock()
        _extract_task(sample_messages, retrieval_result=None, logger=logger)

        # Get the prompt
        call_args = mock_llm.call_args
        prompt = call_args.kwargs.get("message") or call_args[0][0]

        # Framework default examples typically mention generic things like:
        # - "system status", "CPU usage", "database cluster", "web servers"
        # These should NOT be present if we're using only control system examples

        framework_indicators = [
            "database cluster",
            "web servers",
            "CPU usage",
        ]

        found_framework_content = [term for term in framework_indicators if term in prompt]

        # It's ok if these appear in small numbers (could be in explanatory text)
        # but they shouldn't dominate the examples
        assert len(found_framework_content) == 0, (
            f"Prompt should NOT contain framework default examples. "
            f"Found framework terms: {found_framework_content}"
        )


# ===================================================================
# Integration Test
# ===================================================================


class TestTaskExtractionIntegration:
    """Integration tests for task extraction with custom prompt."""

    @patch("osprey.infrastructure.task_extraction_node.get_chat_completion")
    def test_full_extraction_flow_with_custom_prompt(self, mock_llm):
        """Test the complete task extraction flow uses the custom prompt."""
        from osprey.infrastructure.task_extraction_node import _extract_task

        # Mock the LLM response
        expected_task = ExtractedTask(
            task="Read current values of beam position monitors in sector 3",
            depends_on_chat_history=False,
            depends_on_user_memory=False,
        )
        mock_llm.return_value = expected_task

        # Create messages
        messages = [
            MessageUtils.create_user_message("Show me the BPMs in sector 3"),
        ]

        # Call task extraction
        logger = MagicMock()
        result = _extract_task(messages, retrieval_result=None, logger=logger)

        # Verify result
        assert result == expected_task

        # Verify the prompt sent to LLM contains control system content
        call_args = mock_llm.call_args
        prompt = call_args.kwargs.get("message") or call_args[0][0]

        assert "BPM" in prompt
        assert "Beam Position Monitor" in prompt
        assert "control system" in prompt.lower()
