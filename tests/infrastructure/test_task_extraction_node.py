"""Tests for task extraction node."""

from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from osprey.base.errors import ErrorSeverity
from osprey.infrastructure.task_extraction_node import (
    TaskExtractionNode,
    _build_task_extraction_prompt,
    _extract_task,
    _format_task_context,
)
from osprey.prompts.defaults.task_extraction import ExtractedTask


class TestTaskExtractionNode:
    """Test TaskExtractionNode infrastructure node."""

    def test_node_exists_and_is_callable(self):
        """Verify TaskExtractionNode can be instantiated."""
        node = TaskExtractionNode()
        assert node is not None
        assert hasattr(node, "execute")

    def test_has_langgraph_node_attribute(self):
        """Test that TaskExtractionNode has langgraph_node from decorator."""
        assert hasattr(TaskExtractionNode, "langgraph_node")
        assert callable(TaskExtractionNode.langgraph_node)

    def test_node_name_and_description(self):
        """Test node has correct name and description."""
        assert TaskExtractionNode.name == "task_extraction"
        assert TaskExtractionNode.description is not None
        assert len(TaskExtractionNode.description) > 0


class TestTaskExtractionErrorClassification:
    """Test error classification for task extraction operations."""

    def test_classify_connection_error(self):
        """Test connection errors are classified as retriable."""
        exc = ConnectionError("Network error")
        context = {"operation": "task_extraction"}

        classification = TaskExtractionNode.classify_error(exc, context)

        assert classification.severity == ErrorSeverity.RETRIABLE
        assert "retrying" in classification.user_message.lower()

    def test_classify_timeout_error(self):
        """Test timeout errors are classified as retriable."""
        exc = TimeoutError("LLM timeout")
        context = {"operation": "task_extraction"}

        classification = TaskExtractionNode.classify_error(exc, context)

        assert classification.severity == ErrorSeverity.RETRIABLE
        assert "retrying" in classification.user_message.lower()

    def test_classify_value_error(self):
        """Test ValueError is classified as critical."""
        exc = ValueError("Invalid configuration")
        context = {"operation": "task_extraction"}

        classification = TaskExtractionNode.classify_error(exc, context)

        assert classification.severity == ErrorSeverity.CRITICAL
        assert "configuration" in classification.user_message.lower()

    def test_classify_type_error(self):
        """Test TypeError is classified as critical."""
        exc = TypeError("Invalid type")
        context = {"operation": "task_extraction"}

        classification = TaskExtractionNode.classify_error(exc, context)

        assert classification.severity == ErrorSeverity.CRITICAL

    def test_classify_import_error(self):
        """Test ImportError is classified as critical."""
        exc = ImportError("Missing module")
        context = {"operation": "task_extraction"}

        classification = TaskExtractionNode.classify_error(exc, context)

        assert classification.severity == ErrorSeverity.CRITICAL
        assert "dependencies" in classification.user_message.lower()

    def test_classify_module_not_found_error(self):
        """Test ModuleNotFoundError is classified as critical."""
        exc = ModuleNotFoundError("Module not found")
        context = {"operation": "task_extraction"}

        classification = TaskExtractionNode.classify_error(exc, context)

        assert classification.severity == ErrorSeverity.CRITICAL

    def test_classify_unknown_error(self):
        """Test unknown errors are classified as critical."""
        exc = RuntimeError("Unknown error")
        context = {"operation": "task_extraction"}

        classification = TaskExtractionNode.classify_error(exc, context)

        assert classification.severity == ErrorSeverity.CRITICAL
        assert "Unknown" in classification.user_message

    def test_error_metadata_contains_technical_details(self):
        """Test that error classification includes technical details."""
        exc = ValueError("Test error message")
        context = {"operation": "task_extraction"}

        classification = TaskExtractionNode.classify_error(exc, context)

        assert "technical_details" in classification.metadata
        assert "Test error message" in classification.metadata["technical_details"]


class TestTaskExtractionRetryPolicy:
    """Test retry policy for task extraction."""

    def test_get_retry_policy_returns_dict(self):
        """Test that get_retry_policy returns a dictionary."""
        policy = TaskExtractionNode.get_retry_policy()
        assert isinstance(policy, dict)

    def test_retry_policy_has_max_attempts(self):
        """Test retry policy includes max_attempts."""
        policy = TaskExtractionNode.get_retry_policy()
        assert "max_attempts" in policy
        assert policy["max_attempts"] == 3

    def test_retry_policy_has_delay(self):
        """Test retry policy includes delay_seconds."""
        policy = TaskExtractionNode.get_retry_policy()
        assert "delay_seconds" in policy
        assert policy["delay_seconds"] == 1.0

    def test_retry_policy_has_backoff(self):
        """Test retry policy includes backoff_factor."""
        policy = TaskExtractionNode.get_retry_policy()
        assert "backoff_factor" in policy
        assert policy["backoff_factor"] == 1.5


class TestFormatTaskContext:
    """Test _format_task_context helper function."""

    def test_format_task_context_without_data(self):
        """Test formatting task context without retrieval data."""
        messages = [
            HumanMessage(content="What is the weather?"),
            AIMessage(content="Let me check that for you."),
        ]
        logger = Mock()

        with patch("osprey.state.messages.ChatHistoryFormatter") as mock_formatter:
            mock_formatter.format_for_llm.return_value = "Formatted chat history"

            result = _format_task_context(messages, None, logger)

        assert isinstance(result, ExtractedTask)
        assert "Formatted chat history" in result.task
        assert result.depends_on_chat_history is True
        assert result.depends_on_user_memory is True

    def test_format_task_context_with_retrieval_data(self):
        """Test formatting task context with retrieval data."""
        messages = [HumanMessage(content="Test message")]
        logger = Mock()

        # Mock retrieval result
        mock_context = Mock()
        mock_context.format_for_prompt.return_value = "Formatted data content"

        mock_result = Mock()
        mock_result.has_data = True
        mock_result.context_data = {"data_source_1": mock_context}
        mock_result.get_summary.return_value = "Summary"

        with patch("osprey.state.messages.ChatHistoryFormatter") as mock_formatter:
            mock_formatter.format_for_llm.return_value = "Chat"

            result = _format_task_context(messages, mock_result, logger)

        assert isinstance(result, ExtractedTask)
        assert "Formatted data content" in result.task
        assert "data_source_1" in result.task

    def test_format_task_context_handles_formatting_errors(self):
        """Test that formatting errors are handled gracefully."""
        messages = [HumanMessage(content="Test")]
        logger = Mock()

        # Mock context that raises error on format
        mock_context = Mock()
        mock_context.format_for_prompt.side_effect = Exception("Format error")

        mock_result = Mock()
        mock_result.has_data = True
        mock_result.context_data = {"bad_source": mock_context}
        mock_result.get_summary.return_value = "Summary"

        with patch("osprey.state.messages.ChatHistoryFormatter") as mock_formatter:
            mock_formatter.format_for_llm.return_value = "Chat"

            result = _format_task_context(messages, mock_result, logger)

        # Should still return a result with fallback to summary
        assert isinstance(result, ExtractedTask)
        logger.warning.assert_called()

    def test_format_task_context_logs_bypass_mode(self):
        """Test that bypass mode is logged."""
        messages = [HumanMessage(content="Test")]
        logger = Mock()

        with patch("osprey.state.messages.ChatHistoryFormatter") as mock_formatter:
            mock_formatter.format_for_llm.return_value = "Chat"

            _format_task_context(messages, None, logger)

        logger.info.assert_called_with("Bypass mode: skipping LLM, using formatted context as task")


class TestBuildTaskExtractionPrompt:
    """Test _build_task_extraction_prompt helper function."""

    def test_build_prompt_without_retrieval_data(self):
        """Test building prompt without retrieval data."""
        messages = [HumanMessage(content="Test message")]

        with patch(
            "osprey.infrastructure.task_extraction_node.get_framework_prompts"
        ) as mock_get_prompts:
            mock_builder = Mock()
            mock_builder.get_system_instructions.return_value = "System prompt"
            mock_prompts = Mock()
            mock_prompts.get_task_extraction_prompt_builder.return_value = mock_builder
            mock_get_prompts.return_value = mock_prompts

            result = _build_task_extraction_prompt(messages, None)

        assert result == "System prompt"
        mock_builder.get_system_instructions.assert_called_once_with(
            messages=messages, retrieval_result=None
        )

    def test_build_prompt_with_retrieval_data(self):
        """Test building prompt with retrieval data."""
        messages = [HumanMessage(content="Test")]
        mock_result = Mock()

        with patch(
            "osprey.infrastructure.task_extraction_node.get_framework_prompts"
        ) as mock_get_prompts:
            mock_builder = Mock()
            mock_builder.get_system_instructions.return_value = "Prompt with data"
            mock_prompts = Mock()
            mock_prompts.get_task_extraction_prompt_builder.return_value = mock_builder
            mock_get_prompts.return_value = mock_prompts

            result = _build_task_extraction_prompt(messages, mock_result)

        assert result == "Prompt with data"
        mock_builder.get_system_instructions.assert_called_once_with(
            messages=messages, retrieval_result=mock_result
        )


class TestExtractTask:
    """Test _extract_task helper function."""

    def test_extract_task_without_data(self):
        """Test extracting task without retrieval data."""
        messages = [HumanMessage(content="Turn on the lights")]
        logger = Mock()
        expected_task = ExtractedTask(
            task="Turn on the lights",
            depends_on_chat_history=False,
            depends_on_user_memory=False,
        )

        with (
            patch("osprey.infrastructure.task_extraction_node.get_framework_prompts"),
            patch("osprey.infrastructure.task_extraction_node.get_model_config") as mock_config,
            patch("osprey.infrastructure.task_extraction_node.get_chat_completion") as mock_llm,
        ):
            mock_config.return_value = {"model": "gpt-4"}
            mock_llm.return_value = expected_task

            result = _extract_task(messages, None, logger)

        assert result == expected_task
        mock_llm.assert_called_once()

    def test_extract_task_with_retrieval_data(self):
        """Test extracting task with retrieval data."""
        messages = [HumanMessage(content="Test")]
        logger = Mock()
        mock_result = Mock()
        mock_result.has_data = True
        mock_result.get_summary.return_value = "Data summary"

        expected_task = ExtractedTask(
            task="Test task",
            depends_on_chat_history=True,
            depends_on_user_memory=False,
        )

        with (
            patch("osprey.infrastructure.task_extraction_node.get_framework_prompts"),
            patch("osprey.infrastructure.task_extraction_node.get_model_config"),
            patch("osprey.infrastructure.task_extraction_node.get_chat_completion") as mock_llm,
        ):
            mock_llm.return_value = expected_task

            result = _extract_task(messages, mock_result, logger)

        assert result == expected_task
        logger.debug.assert_called_with("Injecting data sources into task extraction: Data summary")

    def test_extract_task_uses_correct_model_config(self):
        """Test that task extraction uses correct model configuration."""
        messages = [HumanMessage(content="Test")]
        logger = Mock()

        with (
            patch("osprey.infrastructure.task_extraction_node.get_framework_prompts"),
            patch("osprey.infrastructure.task_extraction_node.get_model_config") as mock_config,
            patch("osprey.infrastructure.task_extraction_node.get_chat_completion") as mock_llm,
        ):
            mock_config.return_value = {"model": "test-model"}
            mock_llm.return_value = ExtractedTask(
                task="Test", depends_on_chat_history=False, depends_on_user_memory=False
            )

            _extract_task(messages, None, logger)

        mock_config.assert_called_once_with("task_extraction")

    def test_extract_task_passes_output_model(self):
        """Test that ExtractedTask is passed as output_model to LLM."""
        messages = [HumanMessage(content="Test")]
        logger = Mock()

        with (
            patch("osprey.infrastructure.task_extraction_node.get_framework_prompts"),
            patch("osprey.infrastructure.task_extraction_node.get_model_config"),
            patch("osprey.infrastructure.task_extraction_node.get_chat_completion") as mock_llm,
        ):
            mock_llm.return_value = ExtractedTask(
                task="Test", depends_on_chat_history=False, depends_on_user_memory=False
            )

            _extract_task(messages, None, logger)

        # Verify that output_model=ExtractedTask was passed
        call_args = mock_llm.call_args
        assert "output_model" in call_args.kwargs
        assert call_args.kwargs["output_model"] == ExtractedTask
