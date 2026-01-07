"""
Tests for AgentWorker and BaseWorker

Tests the background worker thread functionality including async execution,
signal emission, error handling, and thread safety.
"""

from unittest.mock import Mock, patch

import pytest

# Skip if PyQt5 is not available
pytest.importorskip("PyQt5")

from PyQt5.QtCore import QThread

from osprey.interfaces.pyqt.base_worker import BaseWorker
from osprey.interfaces.pyqt.worker_thread import AgentWorker


class TestBaseWorkerInitialization:
    """Test suite for BaseWorker initialization."""

    def test_init_creates_worker(self):
        """Test initialization creates worker instance."""
        worker = BaseWorker()

        assert isinstance(worker, QThread)
        assert worker._loop is None
        assert worker._should_stop is False

    def test_init_has_signals(self):
        """Test initialization includes required signals."""
        worker = BaseWorker()

        assert hasattr(worker, "error_occurred")
        assert hasattr(worker, "processing_complete")
        # Signals are bound methods on instances, not pyqtSignal type
        assert hasattr(worker.error_occurred, "emit")
        assert hasattr(worker.processing_complete, "emit")


class TestBaseWorkerExecute:
    """Test suite for BaseWorker execute method."""

    def test_execute_raises_not_implemented(self):
        """Test execute raises NotImplementedError if not overridden."""
        worker = BaseWorker()

        with pytest.raises(NotImplementedError):
            worker.execute()


class TestBaseWorkerRunAsync:
    """Test suite for BaseWorker run_async method."""

    def test_run_async_without_loop_raises_error(self):
        """Test run_async raises error when loop not initialized."""
        worker = BaseWorker()

        async def dummy_coro():
            return "result"

        with pytest.raises(RuntimeError, match="Event loop not initialized"):
            worker.run_async(dummy_coro())

    @patch("asyncio.new_event_loop")
    def test_run_async_executes_coroutine(self, mock_new_loop):
        """Test run_async executes coroutine in event loop."""
        mock_loop = Mock()
        mock_loop.run_until_complete = Mock(return_value="test_result")
        mock_new_loop.return_value = mock_loop

        worker = BaseWorker()
        worker._loop = mock_loop

        async def test_coro():
            return "test_result"

        result = worker.run_async(test_coro())

        assert result == "test_result"
        mock_loop.run_until_complete.assert_called_once()


class TestBaseWorkerStop:
    """Test suite for BaseWorker stop functionality."""

    def test_stop_sets_flag(self):
        """Test stop sets the should_stop flag."""
        worker = BaseWorker()

        assert worker.should_stop() is False

        worker.stop()

        assert worker.should_stop() is True

    def test_should_stop_returns_correct_value(self):
        """Test should_stop returns correct value."""
        worker = BaseWorker()

        assert worker.should_stop() is False

        worker._should_stop = True
        assert worker.should_stop() is True


class TestBaseWorkerHandleError:
    """Test suite for BaseWorker error handling."""

    @patch("osprey.interfaces.pyqt.base_worker.logger")
    def test_handle_error_logs_and_emits(self, mock_logger):
        """Test handle_error logs and emits signal."""
        worker = BaseWorker()
        error_emitted = []
        worker.error_occurred.connect(lambda msg: error_emitted.append(msg))

        error = ValueError("Test error")
        worker.handle_error(error, "Test context")

        mock_logger.exception.assert_called_once()
        assert len(error_emitted) == 1
        assert "Test context" in error_emitted[0]
        assert "Test error" in error_emitted[0]

    @patch("osprey.interfaces.pyqt.base_worker.logger")
    def test_handle_error_without_context(self, mock_logger):
        """Test handle_error works without context."""
        worker = BaseWorker()
        error_emitted = []
        worker.error_occurred.connect(lambda msg: error_emitted.append(msg))

        error = ValueError("Test error")
        worker.handle_error(error)

        assert len(error_emitted) == 1
        assert error_emitted[0] == "Test error"


class TestAgentWorkerInitialization:
    """Test suite for AgentWorker initialization."""

    def test_init_stores_parameters(self):
        """Test initialization stores all parameters."""
        gateway = Mock()
        graph = Mock()
        config = {"test": "config"}
        message = "test message"

        worker = AgentWorker(gateway, graph, config, message)

        assert worker.gateway is gateway
        assert worker.graph is graph
        assert worker.config == config
        assert worker.user_message == message

    def test_init_has_additional_signals(self):
        """Test initialization includes AgentWorker-specific signals."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test"

        worker = AgentWorker(gateway, graph, config, message)

        assert hasattr(worker, "message_received")
        assert hasattr(worker, "status_update")
        assert hasattr(worker, "llm_detail")
        assert hasattr(worker, "tool_usage")


class TestAgentWorkerExecute:
    """Test suite for AgentWorker execute method."""

    @patch.object(AgentWorker, "run_async")
    def test_execute_processes_message(self, mock_run_async):
        """Test execute processes message through gateway."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test message"

        # Mock gateway result
        result = Mock()
        result.error = None
        result.resume_command = None
        result.agent_state = {"test": "state"}
        mock_run_async.return_value = result

        worker = AgentWorker(gateway, graph, config, message)

        # Mock _execute_graph to avoid complex graph execution
        worker._execute_graph = Mock()

        worker.execute()

        # Verify gateway was called
        assert mock_run_async.called

    @patch.object(AgentWorker, "run_async")
    def test_execute_handles_error_result(self, mock_run_async):
        """Test execute handles error from gateway."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test message"

        # Mock gateway result with error
        result = Mock()
        result.error = "Test error"
        mock_run_async.return_value = result

        worker = AgentWorker(gateway, graph, config, message)
        error_emitted = []
        worker.error_occurred.connect(lambda msg: error_emitted.append(msg))

        worker.execute()

        assert len(error_emitted) == 1
        assert "Test error" in error_emitted[0]

    @patch.object(AgentWorker, "run_async")
    @patch.object(AgentWorker, "_execute_graph")
    def test_execute_handles_resume_command(self, mock_execute_graph, mock_run_async):
        """Test execute handles resume command."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test message"

        # Mock gateway result with resume command
        result = Mock()
        result.error = None
        result.resume_command = {"resume": "data"}
        result.agent_state = None
        mock_run_async.return_value = result

        worker = AgentWorker(gateway, graph, config, message)
        status_updates = []
        worker.status_update.connect(lambda msg, comp, info: status_updates.append((msg, comp)))

        worker.execute()

        mock_execute_graph.assert_called_once_with({"resume": "data"})
        assert any("Resuming" in msg for msg, _ in status_updates)

    @patch.object(AgentWorker, "run_async")
    @patch.object(AgentWorker, "_execute_graph")
    def test_execute_handles_agent_state(self, mock_execute_graph, mock_run_async):
        """Test execute handles agent state."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test message"

        # Mock gateway result with agent state
        result = Mock()
        result.error = None
        result.resume_command = None
        result.agent_state = {"state": "data"}
        mock_run_async.return_value = result

        worker = AgentWorker(gateway, graph, config, message)

        worker.execute()

        mock_execute_graph.assert_called_once_with({"state": "data"})

    @patch.object(AgentWorker, "run_async")
    def test_execute_handles_no_action_required(self, mock_run_async):
        """Test execute handles case when no action required."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test message"

        # Mock gateway result with no action
        result = Mock()
        result.error = None
        result.resume_command = None
        result.agent_state = None
        mock_run_async.return_value = result

        worker = AgentWorker(gateway, graph, config, message)
        messages = []
        worker.message_received.connect(lambda msg: messages.append(msg))

        worker.execute()

        assert len(messages) == 1
        assert "No action required" in messages[0]

    @patch.object(AgentWorker, "run_async")
    @patch.object(AgentWorker, "handle_error")
    def test_execute_handles_exception(self, mock_handle_error, mock_run_async):
        """Test execute handles exceptions."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test message"

        # Mock exception
        mock_run_async.side_effect = Exception("Test exception")

        worker = AgentWorker(gateway, graph, config, message)

        worker.execute()

        mock_handle_error.assert_called_once()


class TestAgentWorkerExecuteGraph:
    """Test suite for AgentWorker _execute_graph method."""

    @patch.object(AgentWorker, "run_async")
    def test_execute_graph_streams_execution(self, mock_run_async):
        """Test _execute_graph streams graph execution."""
        gateway = Mock()
        graph = Mock()
        graph.get_state = Mock(return_value=Mock(values={"messages": []}, interrupts=[]))
        config = {}
        message = "test"

        worker = AgentWorker(gateway, graph, config, message)
        worker._extract_and_emit_execution_info = Mock()

        # Mock async stream
        async def mock_stream(*args, **kwargs):
            yield {"event_type": "status", "message": "Processing", "component": "test"}

        graph.astream = mock_stream

        worker._execute_graph({"test": "input"})

        # Verify graph.get_state was called
        graph.get_state.assert_called_once()

    @patch.object(AgentWorker, "run_async")
    def test_execute_graph_emits_status_updates(self, mock_run_async):
        """Test _execute_graph emits status updates."""
        gateway = Mock()
        graph = Mock()
        graph.get_state = Mock(return_value=Mock(values={"messages": []}, interrupts=[]))
        config = {}
        message = "test"

        worker = AgentWorker(gateway, graph, config, message)
        worker._extract_and_emit_execution_info = Mock()

        status_updates = []
        llm_details = []
        worker.status_update.connect(
            lambda msg, comp, info: status_updates.append((msg, comp, info))
        )
        worker.llm_detail.connect(
            lambda detail, event_type: llm_details.append((detail, event_type))
        )

        # Mock async stream with status event
        async def mock_stream(*args, **kwargs):
            yield {
                "event_type": "status",
                "message": "Test status",
                "component": "test_component",
                "model_provider": "openai",
                "model_id": "gpt-4",
            }

        graph.astream = mock_stream

        # Mock run_async to actually execute the async function
        def run_async_impl(coro):
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        mock_run_async.side_effect = run_async_impl

        worker._execute_graph({"test": "input"})

        assert len(status_updates) > 0
        msg, comp, info = status_updates[0]
        assert msg == "Test status"
        assert comp == "test_component"
        assert info.get("model_provider") == "openai"

    @patch.object(AgentWorker, "run_async")
    def test_execute_graph_handles_interrupts(self, mock_run_async):
        """Test _execute_graph handles interrupts."""
        gateway = Mock()
        graph = Mock()

        # Mock interrupt
        interrupt = Mock()
        interrupt.value = {"user_message": "Input needed"}

        graph.get_state = Mock(return_value=Mock(values={"messages": []}, interrupts=[interrupt]))
        config = {}
        message = "test"

        worker = AgentWorker(gateway, graph, config, message)
        worker._extract_and_emit_execution_info = Mock()

        messages = []
        worker.message_received.connect(lambda msg: messages.append(msg))

        # Mock async stream
        async def mock_stream(*args, **kwargs):
            return
            yield  # Make it a generator

        graph.astream = mock_stream

        worker._execute_graph({"test": "input"})

        assert len(messages) == 1
        assert "Input needed" in messages[0]

    @patch.object(AgentWorker, "run_async")
    def test_execute_graph_emits_final_message(self, mock_run_async):
        """Test _execute_graph emits final AI message."""
        gateway = Mock()
        graph = Mock()

        # Mock final message
        ai_message = Mock()
        ai_message.content = "Final response"
        ai_message.type = "ai"

        graph.get_state = Mock(return_value=Mock(values={"messages": [ai_message]}, interrupts=[]))
        config = {}
        message = "test"

        worker = AgentWorker(gateway, graph, config, message)
        worker._extract_and_emit_execution_info = Mock()

        messages = []
        worker.message_received.connect(lambda msg: messages.append(msg))

        # Mock async stream
        async def mock_stream(*args, **kwargs):
            return
            yield  # Make it a generator

        graph.astream = mock_stream

        worker._execute_graph({"test": "input"})

        assert len(messages) == 1
        assert "Final response" in messages[0]

    @patch.object(AgentWorker, "run_async")
    def test_execute_graph_stops_when_requested(self, mock_run_async):
        """Test _execute_graph stops when stop is requested."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test"

        worker = AgentWorker(gateway, graph, config, message)
        worker._should_stop = True

        # Mock async stream
        async def mock_stream(*args, **kwargs):
            yield {"event_type": "status", "message": "Processing"}

        graph.astream = mock_stream

        worker._execute_graph({"test": "input"})

        # Should not call get_state if stopped
        graph.get_state.assert_not_called()


class TestAgentWorkerExtractExecutionInfo:
    """Test suite for AgentWorker _extract_and_emit_execution_info method."""

    def test_extract_execution_info_emits_tool_usage(self):
        """Test extraction emits tool usage events."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test"

        worker = AgentWorker(gateway, graph, config, message)

        tool_usage = []
        worker.tool_usage.connect(lambda tool, info: tool_usage.append((tool, info)))

        state_values = {
            "execution_step_results": {
                "step1": {
                    "step_index": 0,
                    "capability": "test_capability",
                    "task_objective": "Test task",
                    "success": True,
                    "execution_time": 1.5,
                }
            }
        }

        worker._extract_and_emit_execution_info(state_values)

        assert len(tool_usage) == 1
        tool, info = tool_usage[0]
        assert tool == "test_capability"
        assert "Test task" in info
        assert "1.50s" in info

    def test_extract_execution_info_handles_empty_results(self):
        """Test extraction handles empty execution results."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test"

        worker = AgentWorker(gateway, graph, config, message)

        tool_usage = []
        worker.tool_usage.connect(lambda tool, info: tool_usage.append((tool, info)))

        state_values = {"execution_step_results": {}}

        worker._extract_and_emit_execution_info(state_values)

        assert len(tool_usage) == 0

    def test_extract_execution_info_orders_by_step_index(self):
        """Test extraction orders results by step_index."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test"

        worker = AgentWorker(gateway, graph, config, message)

        tool_usage = []
        worker.tool_usage.connect(lambda tool, info: tool_usage.append((tool, info)))

        state_values = {
            "execution_step_results": {
                "step2": {
                    "step_index": 1,
                    "capability": "second",
                    "task_objective": "Second task",
                    "success": True,
                    "execution_time": 1.0,
                },
                "step1": {
                    "step_index": 0,
                    "capability": "first",
                    "task_objective": "First task",
                    "success": True,
                    "execution_time": 1.0,
                },
            }
        }

        worker._extract_and_emit_execution_info(state_values)

        assert len(tool_usage) == 2
        assert tool_usage[0][0] == "first"
        assert tool_usage[1][0] == "second"

    def test_extract_execution_info_handles_failure(self):
        """Test extraction handles failed steps."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test"

        worker = AgentWorker(gateway, graph, config, message)

        tool_usage = []
        worker.tool_usage.connect(lambda tool, info: tool_usage.append((tool, info)))

        state_values = {
            "execution_step_results": {
                "step1": {
                    "step_index": 0,
                    "capability": "test_capability",
                    "task_objective": "Failed task",
                    "success": False,
                    "execution_time": 0.5,
                }
            }
        }

        worker._extract_and_emit_execution_info(state_values)

        assert len(tool_usage) == 1
        tool, info = tool_usage[0]
        assert "❌" in info  # Failure icon


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_multiple_workers_independent(self):
        """Test multiple worker instances are independent."""
        gateway1 = Mock()
        gateway2 = Mock()
        graph1 = Mock()
        graph2 = Mock()

        worker1 = AgentWorker(gateway1, graph1, {}, "message1")
        worker2 = AgentWorker(gateway2, graph2, {}, "message2")

        assert worker1.gateway is not worker2.gateway
        assert worker1.user_message != worker2.user_message

    def test_worker_stop_before_execution(self):
        """Test stopping worker before execution."""
        gateway = Mock()
        graph = Mock()

        worker = AgentWorker(gateway, graph, {}, "test")
        worker.stop()

        assert worker.should_stop() is True

    @patch("osprey.interfaces.pyqt.worker_thread.logger")
    def test_extract_execution_info_handles_exception(self, mock_logger):
        """Test extraction handles exceptions gracefully."""
        gateway = Mock()
        graph = Mock()
        config = {}
        message = "test"

        worker = AgentWorker(gateway, graph, config, message)

        # Invalid state values that will cause exception
        state_values = {"execution_step_results": "invalid"}

        # Should not raise exception
        worker._extract_and_emit_execution_info(state_values)

        # Should log warning
        mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
