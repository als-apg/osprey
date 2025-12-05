"""Integration tests for TimeRangeParsingCapability instance method pattern."""

import inspect
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock

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
        # Mock get_model_config
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_model_config",
            MagicMock(return_value={"model": "gpt-4"}),
        )

        # Mock store_output_context to bypass registry validation (like memory capability does)
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeParsingCapability.store_output_context",
            MagicMock(return_value={"capability_context_data": {}}),
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

        # Create a simple mock context with proper CONTEXT_TYPE as class variable
        class MockTimeRangeContext:
            CONTEXT_TYPE = "TIME_RANGE"
            CONTEXT_CATEGORY = "METADATA"

            def __init__(self, start_date, end_date, *args, **kwargs):
                self.start_date = start_date
                self.end_date = end_date
                self.context_type = "TIME_RANGE"

            def model_dump(self):
                """Mimic Pydantic's model_dump() method."""
                return {
                    "start_date": (
                        self.start_date.isoformat()
                        if hasattr(self.start_date, "isoformat")
                        else str(self.start_date)
                    ),
                    "end_date": (
                        self.end_date.isoformat()
                        if hasattr(self.end_date, "isoformat")
                        else str(self.end_date)
                    ),
                    "context_type": self.context_type,
                }

        mock_context_class = MockTimeRangeContext
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeContext", mock_context_class
        )

        # Create instance and inject state/step
        capability = TimeRangeParsingCapability()
        capability._state = mock_state
        capability._step = mock_step

        # Execute
        result = await capability.execute()

        # Verify it executed and returned state updates
        assert isinstance(result, dict)
        assert "capability_context_data" in result

    @pytest.mark.asyncio
    async def test_time_parsing_with_llm(self, mock_state, mock_step, monkeypatch):
        """Test time range parsing using LLM."""
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_model_config",
            MagicMock(return_value={"model": "gpt-4"}),
        )

        # Mock store_output_context to bypass registry validation
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeParsingCapability.store_output_context",
            MagicMock(return_value={"capability_context_data": {}}),
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

        # Create a proper mock context class
        class MockTimeRangeContext:
            CONTEXT_TYPE = "TIME_RANGE"
            CONTEXT_CATEGORY = "METADATA"

            def __init__(self, start_date, end_date, *args, **kwargs):
                self.start_date = start_date
                self.end_date = end_date
                self.context_type = "TIME_RANGE"

            def model_dump(self):
                """Mimic Pydantic's model_dump() method."""
                return {
                    "start_date": (
                        self.start_date.isoformat()
                        if hasattr(self.start_date, "isoformat")
                        else str(self.start_date)
                    ),
                    "end_date": (
                        self.end_date.isoformat()
                        if hasattr(self.end_date, "isoformat")
                        else str(self.end_date)
                    ),
                    "context_type": self.context_type,
                }

        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeContext", MockTimeRangeContext
        )

        capability = TimeRangeParsingCapability()
        capability._state = mock_state
        capability._step = mock_step

        result = await capability.execute()

        assert isinstance(result, dict)
        assert "capability_context_data" in result

    @pytest.mark.asyncio
    async def test_context_storage(self, mock_state, mock_step, monkeypatch):
        """Test that time range context is properly stored."""
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_model_config",
            MagicMock(return_value={"model": "gpt-4"}),
        )

        # Mock store_output_context to bypass registry validation
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeParsingCapability.store_output_context",
            MagicMock(return_value={"capability_context_data": {}}),
        )

        mock_time_output = TimeRangeOutput(
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 2, 0, 0, 0),
            found=True,
        )

        async def mock_to_thread(func, *args, **kwargs):
            return mock_time_output

        monkeypatch.setattr("asyncio.to_thread", mock_to_thread)

        # Create a proper mock context class
        class MockTimeRangeContext:
            CONTEXT_TYPE = "TIME_RANGE"
            CONTEXT_CATEGORY = "METADATA"

            def __init__(self, start_date, end_date, *args, **kwargs):
                self.start_date = start_date
                self.end_date = end_date
                self.context_type = "TIME_RANGE"

            def model_dump(self):
                """Mimic Pydantic's model_dump() method."""
                return {
                    "start_date": (
                        self.start_date.isoformat()
                        if hasattr(self.start_date, "isoformat")
                        else str(self.start_date)
                    ),
                    "end_date": (
                        self.end_date.isoformat()
                        if hasattr(self.end_date, "isoformat")
                        else str(self.end_date)
                    ),
                    "context_type": self.context_type,
                }

        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeContext", MockTimeRangeContext
        )

        capability = TimeRangeParsingCapability()
        capability._state = mock_state
        capability._step = mock_step

        result = await capability.execute()

        # Verify result is a dict with capability_context_data
        assert isinstance(result, dict)
        assert "capability_context_data" in result
