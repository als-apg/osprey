"""Tests for error response generation infrastructure."""

from unittest.mock import Mock, patch

import pytest

from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.infrastructure.error_node import (
    ErrorContext,
    ErrorNode,
    _build_structured_error_report,
    _create_error_context_from_state,
    _create_fallback_response,
    _generate_error_response,
    _generate_llm_explanation,
    _populate_error_context,
)
from osprey.state import AgentState, StateManager


class TestErrorContext:
    """Test ErrorContext data structure."""

    def test_error_context_initialization(self):
        """Test basic ErrorContext creation."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Test error",
            metadata={"test": "data"},
        )

        context = ErrorContext(
            error_classification=classification,
            current_task="Test task",
            failed_operation="test_op",
            total_operations=3,
        )

        assert context.error_classification == classification
        assert context.current_task == "Test task"
        assert context.failed_operation == "test_op"
        assert context.total_operations == 3
        assert context.successful_steps == []
        assert context.failed_steps == []

    def test_error_context_post_init_defaults(self):
        """Test that None list fields are initialized to empty lists."""
        classification = ErrorClassification(
            severity=ErrorSeverity.RETRIABLE,
            user_message="Test",
        )

        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
            successful_steps=None,
            failed_steps=None,
        )

        assert context.successful_steps == []
        assert context.failed_steps == []

    def test_error_context_with_steps(self):
        """Test ErrorContext with execution steps."""
        classification = ErrorClassification(
            severity=ErrorSeverity.REPLANNING,
            user_message="Planning failed",
        )

        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
            successful_steps=["Step 1", "Step 2"],
            failed_steps=["Step 3 - Failed"],
        )

        assert len(context.successful_steps) == 2
        assert len(context.failed_steps) == 1

    def test_error_severity_property(self):
        """Test error_severity property extracts severity from classification."""
        classification = ErrorClassification(
            severity=ErrorSeverity.FATAL,
            user_message="Fatal error",
        )

        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
        )

        assert context.error_severity == ErrorSeverity.FATAL

    def test_error_message_property(self):
        """Test error_message property returns user message."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Custom error message",
        )

        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
        )

        assert context.error_message == "Custom error message"

    def test_error_message_fallback(self):
        """Test error_message returns fallback when no message."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=None,
        )

        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
        )

        assert context.error_message == "Unknown error occurred"

    def test_capability_name_property(self):
        """Test capability_name property access."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Test",
        )

        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
        )

        # Initially None
        assert context.capability_name is None

        # Set dynamically
        context._capability_name = "test_capability"
        assert context.capability_name == "test_capability"


class TestErrorNode:
    """Test ErrorNode infrastructure."""

    def test_error_node_name_and_description(self):
        """Test ErrorNode has correct name and description."""
        assert ErrorNode.name == "error"
        assert ErrorNode.description == "Error Response Generation"

    def test_classify_error(self):
        """Test ErrorNode.classify_error creates FATAL classification."""
        exc = RuntimeError("Test error")
        context = {"node_name": "error", "execution_time": 1.5}

        classification = ErrorNode.classify_error(exc, context)

        assert classification.severity == ErrorSeverity.FATAL
        assert "Error node failed" in classification.user_message
        assert "Test error" in classification.metadata["technical_details"]

    def test_error_node_can_be_instantiated(self):
        """Test ErrorNode can be instantiated."""
        node = ErrorNode()
        assert node is not None
        assert hasattr(node, "execute")

    def test_has_langgraph_node_attribute(self):
        """Test that ErrorNode has langgraph_node from decorator."""
        assert hasattr(ErrorNode, "langgraph_node")
        assert callable(ErrorNode.langgraph_node)


class TestCreateErrorContextFromState:
    """Test _create_error_context_from_state function."""

    def test_create_context_with_full_state(self):
        """Test creating error context from complete state."""
        state = AgentState()
        state["control_error_info"] = {
            "classification": ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Network timeout",
                metadata={"timeout": 30},
            ),
            "capability_name": "api_call",
            "execution_time": 5.2,
        }
        state["task_current_task"] = "Fetch data"
        state["control_retry_count"] = 2

        with patch.object(StateManager, "get_current_step_index", return_value=3):
            context = _create_error_context_from_state(state)

        assert context.error_severity == ErrorSeverity.RETRIABLE
        assert context.current_task == "Fetch data"
        assert context.failed_operation == "api_call"
        assert context.execution_time == 5.2
        assert context.retry_count == 2
        assert context.total_operations == 4  # step_index + 1
        assert context.capability_name == "api_call"

    def test_create_context_with_minimal_state(self):
        """Test creating context with minimal error information."""
        state = AgentState()
        state["control_error_info"] = {
            "original_error": "Unknown system error",
        }

        with patch.object(StateManager, "get_current_step_index", return_value=0):
            context = _create_error_context_from_state(state)

        # Should create fallback classification
        assert context.error_severity == ErrorSeverity.CRITICAL
        assert "Unknown system error" in context.error_message
        assert context.failed_operation == "Unknown operation"

    def test_create_context_with_node_name_fallback(self):
        """Test that node_name is used if capability_name is missing."""
        state = AgentState()
        state["control_error_info"] = {
            "classification": ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Test error",
            ),
            "node_name": "test_node",
        }
        state["task_current_task"] = "Test task"

        with patch.object(StateManager, "get_current_step_index", return_value=0):
            context = _create_error_context_from_state(state)

        assert context.failed_operation == "test_node"
        assert context.capability_name == "test_node"


class TestPopulateErrorContext:
    """Test _populate_error_context function."""

    def test_populate_with_execution_steps(self):
        """Test populating error context with execution history."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Error",
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
        )

        state = AgentState()
        state["execution_step_results"] = {
            "step_0": {
                "step_index": 0,
                "capability": "input_validation",
                "task_objective": "Validate input",
                "success": True,
            },
            "step_1": {
                "step_index": 1,
                "capability": "data_processing",
                "task_objective": "Process data",
                "success": True,
            },
            "step_2": {
                "step_index": 2,
                "capability": "database_query",
                "task_objective": "Query database",
                "success": False,
            },
        }

        _populate_error_context(context, state)

        assert len(context.successful_steps) == 2
        assert len(context.failed_steps) == 1
        assert "Step 1: Validate input" in context.successful_steps
        assert "Step 2: Process data" in context.successful_steps
        assert "Step 3: Query database - Failed" in context.failed_steps

    def test_populate_with_no_execution_steps(self):
        """Test populating context when no execution steps exist."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Error",
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
        )

        state = AgentState()
        # No execution_step_results

        _populate_error_context(context, state)

        assert context.successful_steps == []
        assert context.failed_steps == []

    def test_populate_maintains_chronological_order(self):
        """Test that steps are ordered by step_index."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Error",
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
        )

        state = AgentState()
        # Add steps out of order
        state["execution_step_results"] = {
            "step_2": {
                "step_index": 2,
                "capability": "step_c",
                "task_objective": "Third",
                "success": True,
            },
            "step_0": {
                "step_index": 0,
                "capability": "step_a",
                "task_objective": "First",
                "success": True,
            },
            "step_1": {
                "step_index": 1,
                "capability": "step_b",
                "task_objective": "Second",
                "success": True,
            },
        }

        _populate_error_context(context, state)

        # Verify chronological order
        assert context.successful_steps[0] == "Step 1: First"
        assert context.successful_steps[1] == "Step 2: Second"
        assert context.successful_steps[2] == "Step 3: Third"


class TestBuildStructuredErrorReport:
    """Test _build_structured_error_report function."""

    def test_basic_error_report(self):
        """Test building basic structured error report."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Database connection failed",
            metadata={"host": "db.example.com"},
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Update user profile",
            failed_operation="database_connection",
        )

        report = _build_structured_error_report(context)

        assert "ERROR REPORT" in report
        assert "CRITICAL" in report
        assert "Update user profile" in report
        assert "database_connection" in report

    def test_report_with_capability_name(self):
        """Test report includes capability name when present."""
        classification = ErrorClassification(
            severity=ErrorSeverity.RETRIABLE,
            user_message="Temporary error",
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
        )
        context._capability_name = "test_capability"

        report = _build_structured_error_report(context)

        assert "test_capability" in report

    def test_report_with_execution_statistics(self):
        """Test report includes execution statistics."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Error",
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
            total_operations=5,
            execution_time=12.3,
            retry_count=2,
        )

        report = _build_structured_error_report(context)

        assert "Execution Stats" in report
        assert "Total operations: 5" in report
        assert "12.3s" in report
        assert "Retry attempts: 2" in report

    def test_report_with_execution_summary(self):
        """Test report includes execution summary with steps."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Error",
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
            successful_steps=["Step 1: Init", "Step 2: Process"],
            failed_steps=["Step 3: Save - Failed"],
        )

        report = _build_structured_error_report(context)

        assert "Execution Summary" in report
        assert "Completed successfully" in report
        assert "Step 1: Init" in report
        assert "Failed steps" in report
        assert "Step 3: Save - Failed" in report

    def test_report_uses_format_for_llm(self):
        """Test report uses ErrorClassification.format_for_llm() when available."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Test error",
            metadata={"key": "value"},
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
        )

        with patch.object(
            classification,
            "format_for_llm",
            return_value="FORMATTED ERROR DETAILS",
        ) as mock_format:
            report = _build_structured_error_report(context)

            mock_format.assert_called_once()
            assert "FORMATTED ERROR DETAILS" in report


class TestGenerateLLMExplanation:
    """Test _generate_llm_explanation function."""

    def test_llm_explanation_success(self):
        """Test successful LLM explanation generation."""
        classification = ErrorClassification(
            severity=ErrorSeverity.REPLANNING,
            user_message="Rate limit exceeded",
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Fetch data",
            failed_operation="api_call",
        )

        with (
            patch("osprey.infrastructure.error_node.get_registry") as mock_registry,
            patch("osprey.infrastructure.error_node.get_framework_prompts") as mock_prompts,
            patch("osprey.infrastructure.error_node.get_chat_completion") as mock_llm,
            patch("osprey.infrastructure.error_node.get_model_config") as mock_config,
        ):
            # Mock dependencies
            mock_registry.return_value.get_capabilities_overview.return_value = (
                "Capabilities overview"
            )

            mock_builder = Mock()
            mock_builder.build_prompt.return_value = "Analysis prompt"
            mock_prompts.return_value.get_error_analysis_prompt_builder.return_value = mock_builder

            mock_llm.return_value = "The error occurred because of rate limiting. Try again later."
            mock_config.return_value = {"model": "gpt-4"}

            explanation = _generate_llm_explanation(context)

            assert "Analysis:" in explanation
            assert "rate limiting" in explanation

    def test_llm_explanation_with_empty_response(self):
        """Test handling of empty LLM response."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Error",
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
        )

        with (
            patch("osprey.infrastructure.error_node.get_registry"),
            patch("osprey.infrastructure.error_node.get_framework_prompts"),
            patch("osprey.infrastructure.error_node.get_chat_completion") as mock_llm,
            patch("osprey.infrastructure.error_node.get_model_config"),
        ):
            mock_llm.return_value = ""  # Empty response

            explanation = _generate_llm_explanation(context)

            assert "Analysis:" in explanation
            assert "review the recovery options" in explanation

    def test_llm_explanation_handles_exceptions(self):
        """Test that LLM generation exceptions are handled gracefully."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Error",
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Task",
            failed_operation="op",
        )

        with patch("osprey.infrastructure.error_node.get_registry") as mock_registry:
            mock_registry.side_effect = Exception("Registry unavailable")

            explanation = _generate_llm_explanation(context)

            # Should return fallback message
            assert "Analysis:" in explanation
            assert "structured report" in explanation


class TestGenerateErrorResponse:
    """Test _generate_error_response function."""

    @pytest.mark.asyncio
    async def test_generate_complete_error_response(self):
        """Test generating complete error response with report and analysis."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Database error",
        )
        context = ErrorContext(
            error_classification=classification,
            current_task="Save data",
            failed_operation="db_write",
            execution_time=3.5,
        )

        with (
            patch("osprey.infrastructure.error_node._build_structured_error_report") as mock_report,
            patch("osprey.infrastructure.error_node._generate_llm_explanation") as mock_explain,
        ):
            mock_report.return_value = "STRUCTURED REPORT"
            mock_explain.return_value = "LLM ANALYSIS"

            response = await _generate_error_response(context)

            assert "STRUCTURED REPORT" in response
            assert "LLM ANALYSIS" in response
            mock_report.assert_called_once_with(context)
            mock_explain.assert_called_once_with(context)


class TestCreateFallbackResponse:
    """Test _create_fallback_response function."""

    def test_fallback_response_with_full_error_info(self):
        """Test creating fallback response with complete error information."""
        state = AgentState()
        state["control_error_info"] = {
            "original_error": "Connection timeout",
            "capability_name": "database_query",
        }
        generation_error = Exception("LLM API unavailable")

        response = _create_fallback_response(state, generation_error)

        assert "System Error During Error Handling" in response
        assert "Original Issue" in response
        assert "database_query" in response
        assert "Connection timeout" in response
        assert "Secondary Issue" in response
        assert "LLM API unavailable" in response

    def test_fallback_response_with_minimal_error_info(self):
        """Test creating fallback response with minimal information."""
        state = AgentState()
        # Empty error info
        generation_error = Exception("Generation failed")

        response = _create_fallback_response(state, generation_error)

        assert "Original Issue" in response
        assert "unknown operation" in response
        assert "Unknown error occurred" in response
        assert "Generation failed" in response

    def test_fallback_response_structure(self):
        """Test fallback response has expected structure."""
        state = AgentState()
        state["control_error_info"] = {
            "original_error": "Test error",
            "capability_name": "test_cap",
        }
        generation_error = Exception("Test")

        response = _create_fallback_response(state, generation_error)

        # Should have clear sections
        assert response.count("**") >= 4  # Multiple bold sections
        assert "⚠️" in response  # Warning emoji
        lines = response.split("\n")
        assert len(lines) > 3  # Multi-line structure
