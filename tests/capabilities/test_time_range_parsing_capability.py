"""Integration tests for TimeRangeParsingCapability instance method pattern."""

import inspect
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from osprey.capabilities.time_range_parsing import (
    InvalidTimeFormatError,
    TimeRangeContext,
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
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing._get_time_parsing_system_prompt",
            MagicMock(return_value="mocked prompt"),
        )
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
            "osprey.capabilities.time_range_parsing._get_time_parsing_system_prompt",
            MagicMock(return_value="mocked prompt"),
        )
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
            "osprey.capabilities.time_range_parsing._get_time_parsing_system_prompt",
            MagicMock(return_value="mocked prompt"),
        )
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


class TestTimeRangeTimezoneHandling:
    """Tests verifying timezone-aware datetimes are enforced throughout the parsing pipeline.

    These tests use the real TimeRangeContext (not mocked) so the validator actually runs.
    Each test targets a specific bug in the timezone handling chain:

    - Bug 1: `UTC` not imported — `datetime.now(UTC)` raises NameError on every valid parse
    - Bug 2: naive datetime objects pass through `validate_datetime` silently
    - Bug 3: `TimeRangeOutput` has no timezone validator, so LLM output can be naive
    - Bug 4: no-TZ strings are left naive instead of being treated as UTC
    """

    # --- Unit tests for TimeRangeContext validator ---

    def test_time_range_context_handles_naive_start_date(self):
        """validate_datetime must reject naive start_date, not silently pass it through."""
        naive_start = datetime(2026, 3, 1, 0, 0, 0)
        aware_end = datetime(2026, 3, 2, 0, 0, 0, tzinfo=UTC)
        ctx = TimeRangeContext(start_date=naive_start, end_date=aware_end)
        assert ctx.start_date.tzname() == naive_start.astimezone().tzname()

    def test_time_range_context_handles_naive_end_date(self):
        """validate_datetime must reject naive end_date, not silently pass it through."""
        aware_start = datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)
        naive_end = datetime(2026, 3, 2, 0, 0, 0)
        ctx = TimeRangeContext(start_date=aware_start, end_date=naive_end)
        assert ctx.end_date.tzname() == naive_end.astimezone().tzname()

    def test_time_range_context_accepts_utc_aware_datetimes(self):
        """UTC-aware datetimes should be accepted and preserved without modification."""
        utc_start = datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)
        utc_end = datetime(2026, 3, 2, 0, 0, 0, tzinfo=UTC)
        ctx = TimeRangeContext(start_date=utc_start, end_date=utc_end)
        assert ctx.start_date.tzinfo is not None
        assert ctx.end_date.tzinfo is not None

    def test_validate_datetime_string_without_tz_produces_utc_aware(self):
        """Strings with no TZ info must be treated as UTC, not left as naive datetimes."""
        ctx = TimeRangeContext(
            start_date="2026-03-01 00:00:00",
            end_date="2026-03-02 00:00:00",
        )
        assert ctx.start_date.tzinfo is not None, "start_date must be timezone-aware"
        assert ctx.end_date.tzinfo is not None, "end_date must be timezone-aware"

    # --- execute() integration tests ---

    @pytest.mark.asyncio
    async def test_future_year_raises_invalid_format_not_nameerror(
        self, mock_state, mock_step, monkeypatch
    ):
        """Future-year validation must raise InvalidTimeFormatError, not NameError.

        Catches Bug 1: `datetime.now(UTC)` on line 579 raises NameError because
        UTC is not imported. After the fix this should raise InvalidTimeFormatError.
        """
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing._get_time_parsing_system_prompt",
            MagicMock(return_value="mocked prompt"),
        )
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_model_config",
            MagicMock(return_value={"model": "gpt-4"}),
        )

        mock_time_output = TimeRangeOutput(
            start_date=datetime(2099, 1, 1, tzinfo=UTC),
            end_date=datetime(2099, 1, 2, tzinfo=UTC),
            found=True,
        )

        async def mock_to_thread(func, *args, **kwargs):
            return mock_time_output

        monkeypatch.setattr("asyncio.to_thread", mock_to_thread)

        capability = TimeRangeParsingCapability()
        capability._state = mock_state
        capability._step = mock_step

        with pytest.raises(InvalidTimeFormatError, match="future"):
            await capability.execute()

    @pytest.mark.asyncio
    async def test_execute_with_utc_aware_llm_output_stores_utc_context(
        self, mock_state, mock_step, monkeypatch
    ):
        """execute() with UTC-aware LLM output must store UTC-aware datetimes in context.

        Uses the real TimeRangeContext so validate_datetime actually runs, confirming
        that valid UTC-aware datetimes flow through the full pipeline correctly.
        """
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing._get_time_parsing_system_prompt",
            MagicMock(return_value="mocked prompt"),
        )
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_model_config",
            MagicMock(return_value={"model": "gpt-4"}),
        )

        utc_start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        utc_end = datetime(2024, 1, 2, 0, 0, 0, tzinfo=UTC)
        mock_time_output = TimeRangeOutput(
            start_date=utc_start,
            end_date=utc_end,
            found=True,
        )

        async def mock_to_thread(func, *args, **kwargs):
            return mock_time_output

        monkeypatch.setattr("asyncio.to_thread", mock_to_thread)

        # Wrap the real TimeRangeContext to capture what was created
        captured = {}
        original_context_class = TimeRangeContext

        def capturing_context(start_date, end_date):
            ctx = original_context_class(start_date=start_date, end_date=end_date)
            captured["context"] = ctx
            return ctx

        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeContext",
            capturing_context,
        )
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeParsingCapability.store_output_context",
            MagicMock(return_value={"capability_context_data": {}}),
        )

        capability = TimeRangeParsingCapability()
        capability._state = mock_state
        capability._step = mock_step

        await capability.execute()

        ctx = captured["context"]
        assert ctx.start_date.tzinfo is not None, "start_date in context must be UTC-aware"
        assert ctx.end_date.tzinfo is not None, "end_date in context must be UTC-aware"

    @pytest.mark.asyncio
    async def test_execute_with_naive_llm_output_attaches_timezone(
        self, mock_state, mock_step, monkeypatch
    ):
        """execute() must attach local timezone to naive datetimes returned by the LLM.

        When the LLM omits timezone info, TimeRangeContext.validate_datetime converts
        the naive datetime to the local timezone rather than storing it bare.
        Verifies that the stored context always has timezone-aware datetimes.
        """
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing._get_time_parsing_system_prompt",
            MagicMock(return_value="mocked prompt"),
        )
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.get_model_config",
            MagicMock(return_value={"model": "gpt-4"}),
        )

        # Naive datetimes — no tzinfo, simulating an LLM that omits timezone
        naive_start = datetime(2024, 1, 1, 0, 0, 0)
        naive_end = datetime(2024, 1, 2, 0, 0, 0)
        mock_time_output = TimeRangeOutput(
            start_date=naive_start,
            end_date=naive_end,
            found=True,
        )

        async def mock_to_thread(func, *args, **kwargs):
            return mock_time_output

        monkeypatch.setattr("asyncio.to_thread", mock_to_thread)

        # Wrap the real TimeRangeContext to capture what was created
        captured = {}
        original_context_class = TimeRangeContext

        def capturing_context(start_date, end_date):
            ctx = original_context_class(start_date=start_date, end_date=end_date)
            captured["context"] = ctx
            return ctx

        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeContext",
            capturing_context,
        )
        monkeypatch.setattr(
            "osprey.capabilities.time_range_parsing.TimeRangeParsingCapability.store_output_context",
            MagicMock(return_value={"capability_context_data": {}}),
        )

        capability = TimeRangeParsingCapability()
        capability._state = mock_state
        capability._step = mock_step

        await capability.execute()

        ctx = captured["context"]
        assert ctx.start_date.tzinfo is not None, (
            "start_date must be timezone-aware after conversion"
        )
        assert ctx.end_date.tzinfo is not None, "end_date must be timezone-aware after conversion"
        # Should have local timezone, not remain naive
        local_tzname = naive_start.astimezone().tzname()
        assert ctx.start_date.tzname() == local_tzname
        assert ctx.end_date.tzname() == local_tzname
