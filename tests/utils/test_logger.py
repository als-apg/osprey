"""
Tests for unified logging and streaming system.

Tests cover:
- Logger without streaming context (module-level usage)
- Logger with streaming context (capability usage)
- Status method and automatic streaming
- Graceful degradation when LangGraph unavailable
- TypedEvent emission through all logging methods
"""

from unittest.mock import MagicMock, patch

import pytest

from osprey.utils.logger import ComponentLogger, get_logger


class TestComponentLoggerBasic:
    """Test basic ComponentLogger functionality without streaming."""

    def test_logger_creation_without_state(self):
        """Test that logger can be created without state (module-level usage)."""
        logger = get_logger("test_component")
        assert isinstance(logger, ComponentLogger)
        assert logger.component_name == "test_component"
        assert logger._state is None

    def test_logger_creation_with_custom_params(self):
        """Test custom logger creation with explicit parameters."""
        logger = get_logger(name="custom_logger", color="blue")
        assert isinstance(logger, ComponentLogger)
        assert logger.component_name == "custom_logger"
        assert logger.color == "blue"

    def test_basic_logging_methods(self):
        """Test that basic logging methods work without crashing."""
        logger = get_logger("test_component")

        # These should all work without streaming
        logger.info("Info message")
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.success("Success message")
        logger.key_info("Key info message")
        logger.timing("Timing message")
        logger.approval("Approval message")
        logger.resume("Resume message")

    def test_status_without_streaming_context(self):
        """Test that status() works gracefully without streaming context."""
        logger = get_logger("test_component")

        # Should not raise even though no stream writer available
        logger.status("Test status message")


class TestComponentLoggerStreaming:
    """Test ComponentLogger streaming functionality with TypedEvents."""

    @patch("langgraph.config.get_stream_writer")
    def test_logger_with_streaming_context(self, mock_get_stream_writer):
        """Test that logger streams TypedEvents when LangGraph context available."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer

        mock_state = {"planning_execution_plan": {"steps": [{}, {}]}}
        logger = get_logger("test_component", state=mock_state)

        logger.status("Test message")

        # Verify stream event was emitted as TypedEvent
        assert mock_writer.called
        event = mock_writer.call_args[0][0]
        assert event["message"] == "Test message"
        assert event["event_class"] == "StatusEvent"
        assert event["component"] == "test_component"
        assert event["level"] == "status"

    @patch("langgraph.config.get_stream_writer")
    def test_status_method_streams_automatically(self, mock_get_stream_writer):
        """Test that status() method streams automatically as StatusEvent."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer

        logger = get_logger("test_component", state={})
        logger.status("Creating execution plan...")

        assert mock_writer.called
        event = mock_writer.call_args[0][0]
        assert event["event_class"] == "StatusEvent"
        assert event["level"] == "status"
        assert event["message"] == "Creating execution plan..."

    @patch("langgraph.config.get_stream_writer")
    def test_error_streams_automatically(self, mock_get_stream_writer):
        """Test that error() method streams automatically as ErrorEvent."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer

        logger = get_logger("test_component", state={})
        logger.error("Error occurred")

        assert mock_writer.called
        event = mock_writer.call_args[0][0]
        assert event["event_class"] == "ErrorEvent"
        assert event["error_message"] == "Error occurred"

    @patch("langgraph.config.get_stream_writer")
    def test_success_streams_by_default(self, mock_get_stream_writer):
        """Test that success() method streams by default as StatusEvent."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer

        logger = get_logger("test_component", state={})
        logger.success("Operation completed")

        assert mock_writer.called
        event = mock_writer.call_args[0][0]
        assert event["event_class"] == "StatusEvent"
        assert event["level"] == "success"

    @patch("langgraph.config.get_stream_writer")
    def test_warning_streams_by_default(self, mock_get_stream_writer):
        """Test that warning() method streams by default as StatusEvent."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer

        logger = get_logger("test_component", state={})
        logger.warning("Warning message")

        assert mock_writer.called
        event = mock_writer.call_args[0][0]
        assert event["event_class"] == "StatusEvent"
        assert event["level"] == "warning"

    @patch("langgraph.config.get_stream_writer")
    def test_info_streams_as_typed_event(self, mock_get_stream_writer):
        """Test that info() streams as TypedEvent in unified pipeline."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer

        logger = get_logger("test_component", state={})
        logger.info("Info message")

        # In unified TypedEvent pipeline, ALL methods emit events
        assert mock_writer.called
        event = mock_writer.call_args[0][0]
        assert event["event_class"] == "StatusEvent"
        assert event["level"] == "info"
        assert event["message"] == "Info message"

    @patch("langgraph.config.get_stream_writer")
    def test_debug_streams_as_typed_event(self, mock_get_stream_writer):
        """Test that debug() streams as TypedEvent in unified pipeline."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer

        logger = get_logger("test_component", state={})
        logger.debug("Debug message")

        # In unified TypedEvent pipeline, ALL methods emit events
        assert mock_writer.called
        event = mock_writer.call_args[0][0]
        assert event["event_class"] == "StatusEvent"
        assert event["level"] == "debug"
        assert event["message"] == "Debug message"

    @patch("langgraph.config.get_stream_writer")
    def test_metadata_in_status_event(self, mock_get_stream_writer):
        """Test that StatusEvent includes message content correctly."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer

        logger = get_logger("test_component", state={})
        logger.status("Processing batch 2/5")

        assert mock_writer.called
        event = mock_writer.call_args[0][0]
        assert event["message"] == "Processing batch 2/5"
        assert event["event_class"] == "StatusEvent"


class TestStreamingGracefulDegradation:
    """Test graceful degradation when LangGraph unavailable."""

    @patch("langgraph.config.get_stream_writer")
    def test_graceful_degradation_on_runtime_error(self, mock_get_stream_writer):
        """Test that logger handles RuntimeError gracefully (not in LangGraph context)."""
        mock_get_stream_writer.side_effect = RuntimeError("Not in LangGraph context")

        logger = get_logger("test_component", state={})

        # Should not raise, even though stream writer unavailable
        logger.status("Test message")
        logger.error("Error message")
        logger.success("Success message")

    @patch("langgraph.config.get_stream_writer")
    def test_graceful_degradation_on_import_error(self, mock_get_stream_writer):
        """Test that logger handles ImportError gracefully (LangGraph not installed)."""
        mock_get_stream_writer.side_effect = ImportError("LangGraph not available")

        logger = get_logger("test_component", state={})

        # Should not raise
        logger.status("Test message")
        logger.error("Error message")

    @patch("langgraph.config.get_stream_writer")
    def test_emitter_initialization_on_first_call(self, mock_get_stream_writer):
        """Test that emitter is initialized when first stream call is made."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer

        logger = get_logger("test_component", state={})

        # First status call should trigger emitter initialization
        logger.status("Test")

        # Verify the stream writer was called
        assert mock_writer.called


class TestStepInfoExtraction:
    """Test step info extraction from state."""

    @patch("langgraph.config.get_stream_writer")
    def test_task_preparation_step_info(self, mock_get_stream_writer):
        """Test that task preparation components get hard-coded step info."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer

        # Test orchestrator (step 3 of task preparation)
        logger = get_logger("orchestrator", state={})
        logger.status("Creating plan")

        event = mock_writer.call_args[0][0]
        assert event["step"] == 3
        assert event["total_steps"] == 3
        assert event["phase"] == "Task Preparation"

    @patch("langgraph.config.get_stream_writer")
    @patch("osprey.state.state_manager.StateManager.get_current_step_index")
    def test_execution_phase_step_info(self, mock_get_step_index, mock_get_stream_writer):
        """Test that execution phase components extract step info from state."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer
        mock_get_step_index.return_value = 1  # Second step (0-indexed)

        mock_state = {
            "planning_execution_plan": {
                "steps": [{"capability": "step1"}, {"capability": "step2"}, {"capability": "step3"}]
            }
        }

        logger = get_logger("python_executor", state=mock_state)
        logger.status("Executing code")

        event = mock_writer.call_args[0][0]
        assert event["step"] == 2  # 1-based for display
        assert event["total_steps"] == 3
        assert event["phase"] == "Execution"

    @patch("langgraph.config.get_stream_writer")
    def test_phase_fallback_when_no_plan(self, mock_get_stream_writer):
        """Test fallback when no execution plan in state."""
        mock_writer = MagicMock()
        mock_get_stream_writer.return_value = mock_writer

        logger = get_logger("custom_component", state={})
        logger.status("Working")

        event = mock_writer.call_args[0][0]
        # step and total_steps can be None in TypedEvent
        assert event["phase"] == "Custom Component"  # Title case of component name


class TestBackwardCompatibility:
    """Test backward compatibility with old API."""

    def test_deprecated_source_parameter(self):
        """Test that deprecated source parameter still works with warning."""
        with pytest.warns(DeprecationWarning):
            logger = get_logger("test_component", source="framework")

        assert isinstance(logger, ComponentLogger)
        assert logger.component_name == "test_component"

    def test_critical_and_exception_methods(self):
        """Test that critical and exception methods still work."""
        logger = get_logger("test_component")

        # These methods should still work
        logger.critical("Critical message")
        logger.exception("Exception message")

    def test_logger_properties(self):
        """Test that logger properties are accessible."""
        logger = get_logger("test_component")

        assert logger.name == "test_component"
        assert isinstance(logger.level, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
