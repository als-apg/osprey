"""Integration tests for TimeRangeParsingCapability instance method pattern."""

import inspect
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.capabilities.time_range_parsing import (
    TimeRangeOutput,
    TimeRangeParsingCapability,
)


class TestTimeRangeParsingCapabilityMigration:
    """Test TimeRangeParsingCapability instance method migration."""

    def test_uses_instance_method_not_static(self):
        """Verify execute() migrated from @staticmethod to instance method."""
        execute_method = inspect.getattr_static(TimeRangeParsingCapability, "execute")
        assert not isinstance(execute_method, staticmethod)

        sig = inspect.signature(TimeRangeParsingCapability.execute)
        params = list(sig.parameters.keys())
        assert params == ["self"]

    @pytest.mark.asyncio
    async def test_execute_with_state_injection(self, mock_state, mock_step, monkeypatch):
        """Test execute() accesses self._state and self._step correctly."""
        # Mock streamer
        mock_streamer = MagicMock()
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_streamer",
            MagicMock(return_value=mock_streamer),
        )

        # Mock StateManager
        mock_sm = MagicMock()
        mock_sm.store_context.return_value = {"context_data": {"TIME_RANGE": "stored"}}
        monkeypatch.setattr("osprey.capabilities.time_range_parsing.StateManager", mock_sm)

        # Mock get_model_config
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_model_config",
            MagicMock(return_value={"model": "gpt-4"}),
        )

        # Mock get_chat_completion to return valid time range
        mock_time_output = TimeRangeOutput(
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 2, 0, 0, 0),
            found=True,
        )

        async def mock_to_thread(func, *args, **kwargs):
            """Mock asyncio.to_thread to return our mocked response."""
            return mock_time_output

        monkeypatch.setattr("asyncio.to_thread", mock_to_thread)

        # Mock TimeRangeContext
        mock_context_instance = MagicMock()
        mock_context_class = MagicMock(return_value=mock_context_instance)
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeContext", mock_context_class
        )

        # Mock logger
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_logger", lambda x: MagicMock()
        )

        # Create instance and inject state/step
        capability = TimeRangeParsingCapability()
        capability._state = mock_state
        capability._step = mock_step

        # Execute
        result = await capability.execute()

        # Verify it executed and returned state updates
        assert isinstance(result, dict)
        assert "context_data" in result

        # Verify TimeRangeContext was created
        mock_context_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_time_parsing_with_llm(self, mock_state, mock_step, monkeypatch):
        """Test time range parsing using LLM."""
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_streamer",
            MagicMock(return_value=MagicMock()),
        )

        mock_sm = MagicMock()
        mock_sm.store_context.return_value = {"context_data": {"TIME_RANGE": "stored"}}
        monkeypatch.setattr("osprey.capabilities.time_range_parsing.StateManager", mock_sm)

        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_model_config",
            MagicMock(return_value={"model": "gpt-4"}),
        )

        # Mock LLM response
        mock_time_output = TimeRangeOutput(
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 2, 0, 0, 0),
            found=True,
        )

        async def mock_to_thread(func, *args, **kwargs):
            return mock_time_output

        monkeypatch.setattr("asyncio.to_thread", mock_to_thread)

        mock_context_instance = MagicMock(
            start_time=datetime(2024, 1, 1, 0, 0, 0),
            end_time=datetime(2024, 1, 2, 0, 0, 0),
        )
        mock_context_class = MagicMock(return_value=mock_context_instance)
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeContext", mock_context_class
        )

        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_logger", lambda x: MagicMock()
        )

        capability = TimeRangeParsingCapability()
        capability._state = mock_state
        capability._step = mock_step

        result = await capability.execute()

        assert isinstance(result, dict)
        assert "context_data" in result

    @pytest.mark.asyncio
    async def test_context_storage(self, mock_state, mock_step, monkeypatch):
        """Test that time range context is properly stored."""
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_streamer",
            MagicMock(return_value=MagicMock()),
        )

        mock_sm = MagicMock()
        expected_update = {"context_data": {"TIME_RANGE": "stored_context"}}
        mock_sm.store_context.return_value = expected_update
        monkeypatch.setattr("osprey.capabilities.time_range_parsing.StateManager", mock_sm)

        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_model_config",
            MagicMock(return_value={"model": "gpt-4"}),
        )

        mock_time_output = TimeRangeOutput(
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 2, 0, 0, 0),
            found=True,
        )

        async def mock_to_thread(func, *args, **kwargs):
            return mock_time_output

        monkeypatch.setattr("asyncio.to_thread", mock_to_thread)

        mock_context_instance = MagicMock()
        mock_context_class = MagicMock(return_value=mock_context_instance)
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeContext", mock_context_class
        )

        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_logger", lambda x: MagicMock()
        )

        capability = TimeRangeParsingCapability()
        capability._state = mock_state
        capability._step = mock_step

        result = await capability.execute()

        # Verify StateManager.store_context was called
        mock_sm.store_context.assert_called_once()
        assert result == expected_update


class TestTimeRangeParsingCapabilityDecoratorIntegration:
    """Test TimeRangeParsing capability works via @capability_node decorator."""

    @pytest.mark.asyncio
    async def test_via_langgraph_node(self, mock_state, monkeypatch):
        """Test execution via decorator-created langgraph_node."""
        # Mock all dependencies
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_streamer",
            MagicMock(return_value=MagicMock()),
        )

        mock_sm = MagicMock()
        mock_sm.get_current_step.return_value = {
            "context_key": "test_key",
            "task_objective": "Parse time range for last 24 hours",
        }
        mock_sm.store_context.return_value = {"context_data": {}}
        monkeypatch.setattr("osprey.capabilities.time_range_parsing.StateManager", mock_sm)
        monkeypatch.setattr("osprey.state.StateManager", mock_sm)

        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_model_config",
            MagicMock(return_value={"model": "gpt-4"}),
        )

        mock_time_output = TimeRangeOutput(
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 2, 0, 0, 0),
            found=True,
        )

        async def mock_to_thread(func, *args, **kwargs):
            return mock_time_output

        monkeypatch.setattr("asyncio.to_thread", mock_to_thread)

        mock_context_instance = MagicMock()
        mock_context_class = MagicMock(return_value=mock_context_instance)
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeContext", mock_context_class
        )

        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_logger", lambda x: MagicMock()
        )

        # Mock LangGraph streaming
        monkeypatch.setattr("osprey.base.decorators.get_stream_writer", lambda: None)

        # Call via decorator
        node_func = TimeRangeParsingCapability.langgraph_node
        result = await node_func(mock_state)

        assert isinstance(result, dict)
        assert "planning_current_step_index" in result
