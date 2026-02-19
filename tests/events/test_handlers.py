"""Tests for TUI and CLI event handlers."""

import re
from io import StringIO
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from osprey.events import (
    CapabilityCompleteEvent,
    CapabilityStartEvent,
    ErrorEvent,
    PhaseCompleteEvent,
    PhaseStartEvent,
    ResultEvent,
    StatusEvent,
)
from osprey.interfaces.cli.event_handler import CLIEventHandler


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


# =============================================================================
# Test CLI Event Handler Initialization
# =============================================================================


class TestCLIEventHandlerInitialization:
    """Test CLIEventHandler initialization."""

    def test_handler_creates_console_if_not_provided(self):
        """Test that handler creates a Console if none provided."""
        handler = CLIEventHandler()
        assert handler.console is not None
        assert isinstance(handler.console, Console)

    def test_handler_uses_provided_console(self):
        """Test that handler uses provided Console."""
        console = Console()
        handler = CLIEventHandler(console=console)
        assert handler.console is console

    def test_handler_verbose_default_false(self):
        """Test that verbose defaults to False."""
        handler = CLIEventHandler()
        assert handler.verbose is False

    def test_handler_verbose_can_be_set(self):
        """Test that verbose can be set."""
        handler = CLIEventHandler(verbose=True)
        assert handler.verbose is True

    def test_handler_show_timing_default_true(self):
        """Test that show_timing defaults to True."""
        handler = CLIEventHandler()
        assert handler.show_timing is True


# =============================================================================
# Test CLI Handler - Capability Events
# =============================================================================


class TestCLIHandlerCapabilityEvents:
    """Test CLIEventHandler handling of capability events.

    In the unified TypedEvent system, CapabilityStart/Complete are structural
    events used by TUI for progress tracking. The CLI handler intentionally
    ignores them - status updates flow through StatusEvent instead.
    """

    @pytest.mark.asyncio
    async def test_handle_capability_start_silently_ignored(self):
        """Test that capability start is silently ignored (structural event)."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console)

        event = CapabilityStartEvent(
            capability_name="python_executor",
            step_number=2,
            total_steps=5,
            description="Running Python code",
        )

        await handler.handle(event)

        printed = output.getvalue()
        assert printed == ""

    @pytest.mark.asyncio
    async def test_handle_capability_complete_silently_ignored(self):
        """Test that capability complete is silently ignored (structural event)."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console)

        event = CapabilityCompleteEvent(
            capability_name="python_executor",
            success=True,
            duration_ms=500,
        )

        await handler.handle(event)

        printed = output.getvalue()
        assert printed == ""

    @pytest.mark.asyncio
    async def test_handle_capability_failure_silently_ignored(self):
        """Test that capability failure is silently ignored (errors come via ErrorEvent)."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console)

        event = CapabilityCompleteEvent(
            capability_name="python_executor",
            success=False,
            duration_ms=100,
            error_message="Execution failed: syntax error",
        )

        await handler.handle(event)

        printed = output.getvalue()
        assert printed == ""


# =============================================================================
# Test CLI Handler - Status Events
# =============================================================================


class TestCLIHandlerStatusEvents:
    """Test CLIEventHandler handling of status events."""

    @pytest.mark.asyncio
    async def test_handle_status_error_always_shown(self):
        """Test that error level status is always shown."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console, verbose=False)

        event = StatusEvent(message="Something went wrong", level="error")

        await handler.handle(event)

        printed = output.getvalue()
        assert "Something went wrong" in printed

    @pytest.mark.asyncio
    async def test_handle_status_warning_always_shown(self):
        """Test that warning level status is always shown."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console, verbose=False)

        event = StatusEvent(message="This is a warning", level="warning")

        await handler.handle(event)

        printed = output.getvalue()
        assert "This is a warning" in printed

    @pytest.mark.asyncio
    async def test_handle_status_info_always_shown(self):
        """Test that info level status is always shown (unified TypedEvent pipeline)."""
        # In unified TypedEvent pipeline, all events are shown to the user
        # (clients filter what they need, not the emitter)
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console, verbose=False)

        event = StatusEvent(message="Info message", level="info")
        await handler.handle(event)

        printed = output.getvalue()
        assert "Info message" in printed


# =============================================================================
# Test CLI Handler - Phase Events
# =============================================================================


class TestCLIHandlerPhaseEvents:
    """Test CLIEventHandler handling of phase events.

    In the unified TypedEvent system, PhaseStart/Complete are structural
    events used by TUI for progress tracking. The CLI handler intentionally
    ignores them - phase updates flow through StatusEvent instead.
    """

    @pytest.mark.asyncio
    async def test_handle_phase_start_silently_ignored(self):
        """Test that phase start is silently ignored (structural event)."""
        output1 = StringIO()
        console1 = Console(file=output1, force_terminal=True)
        handler1 = CLIEventHandler(console=console1, verbose=False)

        event = PhaseStartEvent(phase="classification", description="Classifying")
        await handler1.handle(event)
        assert output1.getvalue() == ""

        # Also ignored in verbose mode
        output2 = StringIO()
        console2 = Console(file=output2, force_terminal=True)
        handler2 = CLIEventHandler(console=console2, verbose=True)
        await handler2.handle(event)
        assert output2.getvalue() == ""

    @pytest.mark.asyncio
    async def test_handle_phase_complete_silently_ignored(self):
        """Test that phase complete is silently ignored (structural event)."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console, verbose=True)

        event = PhaseCompleteEvent(phase="classification", success=True, duration_ms=150)

        await handler.handle(event)
        assert output.getvalue() == ""


# =============================================================================
# Test CLI Handler - Result Events
# =============================================================================


class TestCLIHandlerResultEvents:
    """Test CLIEventHandler handling of result events.

    In the unified TypedEvent system, ResultEvent is a structural event
    for bookkeeping. The CLI handler ignores it - final responses are
    delivered via LLM token streaming, not ResultEvent.
    """

    @pytest.mark.asyncio
    async def test_handle_result_silently_ignored(self):
        """Test that result success is silently ignored (response comes via streaming)."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console)

        event = ResultEvent(
            success=True,
            response="Task completed successfully",
        )

        await handler.handle(event)
        assert output.getvalue() == ""

    @pytest.mark.asyncio
    async def test_handle_result_failure_silently_ignored(self):
        """Test that result failure is silently ignored (errors come via ErrorEvent)."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console)

        event = ResultEvent(
            success=False,
            response="Something went wrong",
        )

        await handler.handle(event)
        assert output.getvalue() == ""


# =============================================================================
# Test CLI Handler - Error Events
# =============================================================================


class TestCLIHandlerErrorEvents:
    """Test CLIEventHandler handling of error events."""

    @pytest.mark.asyncio
    async def test_handle_error_event(self):
        """Test that error event is printed."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console)

        event = ErrorEvent(
            error_type="ValidationError",
            error_message="Invalid input format",
            recoverable=True,
        )

        await handler.handle(event)

        printed = output.getvalue()
        assert "ValidationError" in printed
        assert "Invalid input format" in printed


# =============================================================================
# Test CLI Handler - Sync Method
# =============================================================================


class TestCLIHandlerSyncMethod:
    """Test CLIEventHandler synchronous handling."""

    def test_handle_sync_method(self):
        """Test that handle_sync works with StatusEvent."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console)

        event = StatusEvent(
            message="Processing step 1",
            level="info",
            component="test",
        )

        # Should not raise
        handler.handle_sync(event)

        printed = strip_ansi(output.getvalue())
        assert "Processing step 1" in printed


# =============================================================================
# Test CLI Handler - Unknown Events
# =============================================================================


class TestCLIHandlerUnknownEvents:
    """Test CLIEventHandler handling of unknown events."""

    @pytest.mark.asyncio
    async def test_unknown_event_silently_skipped(self):
        """Test that unknown events are silently skipped."""
        output = StringIO()
        console = Console(file=output, force_terminal=True)
        handler = CLIEventHandler(console=console)

        # Use a structural event type that the CLI handler doesn't display
        from osprey.events import ApprovalRequiredEvent

        event = ApprovalRequiredEvent(
            action_description="Test approval",
            approval_type="execution",
        )

        # Should not raise
        await handler.handle(event)

        # Should print nothing for unhandled event
        printed = output.getvalue()
        assert printed == ""


# =============================================================================
# Test TUI Event Handler (Basic)
# =============================================================================


class TestTUIEventHandlerBasic:
    """Test TUIEventHandler basic functionality.

    Note: Full TUI testing requires Textual test harness.
    These tests verify basic handler initialization and method structure.
    """

    def test_handler_initialization(self):
        """Test TUIEventHandler initialization."""
        from osprey.interfaces.tui.event_handler import TUIEventHandler

        mock_display = MagicMock()
        handler = TUIEventHandler(display=mock_display)

        assert handler.display is mock_display
        assert handler.shared_data == {}
        assert handler.current_blocks == {}
        assert handler.current_phase is None

    def test_handler_with_shared_data(self):
        """Test TUIEventHandler with shared data."""
        from osprey.interfaces.tui.event_handler import TUIEventHandler

        mock_display = MagicMock()
        shared = {"task": "Test task", "capability_names": ["python"]}
        handler = TUIEventHandler(display=mock_display, shared_data=shared)

        assert handler.shared_data == shared

    @pytest.mark.asyncio
    async def test_handle_status_event(self):
        """Test TUIEventHandler handles status events."""
        from osprey.interfaces.tui.event_handler import TUIEventHandler

        mock_display = MagicMock()
        handler = TUIEventHandler(display=mock_display)

        event = StatusEvent(
            message="Processing",
            level="info",
            component="router",
        )

        # Should not raise
        await handler.handle(event)

    def test_handle_legacy_event(self):
        """Test TUIEventHandler handles legacy dict events."""
        from osprey.interfaces.tui.event_handler import TUIEventHandler

        mock_display = MagicMock()
        handler = TUIEventHandler(display=mock_display)

        legacy_dict = {
            "event_class": "StatusEvent",
            "message": "Test",
            "level": "info",
        }

        event = handler.handle_legacy_event(legacy_dict)
        assert isinstance(event, StatusEvent)

    def test_handle_legacy_event_invalid(self):
        """Test TUIEventHandler returns None for invalid legacy events."""
        from osprey.interfaces.tui.event_handler import TUIEventHandler

        mock_display = MagicMock()
        handler = TUIEventHandler(display=mock_display)

        invalid_dict = {"message": "Test"}  # Missing event_class

        event = handler.handle_legacy_event(invalid_dict)
        assert event is None

    def test_extract_shared_data(self):
        """Test TUIEventHandler extract_shared_data method."""
        from osprey.interfaces.tui.event_handler import TUIEventHandler

        mock_display = MagicMock()
        handler = TUIEventHandler(display=mock_display)

        event = StatusEvent(
            message="Test",
            component="task_extraction",
        )

        # Should not raise - method exists and accepts event
        handler.extract_shared_data(event)
