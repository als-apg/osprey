"""
Tests for MessageHandlers

Tests the message handling functionality including message routing,
status updates, error handling, LLM details, and tool usage tracking.
"""

from unittest.mock import Mock, patch

import pytest

from osprey.interfaces.pyqt.enums import Colors, EventTypes
from osprey.interfaces.pyqt.message_handlers import MessageHandlers


class TestMessageHandlersInitialization:
    """Test suite for MessageHandlers initialization."""

    def test_init_stores_parameters(self):
        """Test initialization stores event bus and conversation provider."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)

        assert handlers.event_bus is event_bus
        assert handlers.conversation_id_provider is provider

    def test_init_with_different_providers(self):
        """Test initialization with different provider types."""
        event_bus = Mock()

        # Function provider
        def provider1():
            return "conv1"

        handlers1 = MessageHandlers(event_bus, provider1)
        assert handlers1.conversation_id_provider() == "conv1"

        # Mock provider
        provider2 = Mock(return_value="conv2")
        handlers2 = MessageHandlers(event_bus, provider2)
        assert handlers2.conversation_id_provider() == "conv2"


class TestOnMessageReceived:
    """Test suite for on_message_received handler."""

    def test_on_message_received_publishes_events(self):
        """Test message received publishes correct events."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_message_received("Test message")

        # Should publish 4 events
        assert event_bus.publish.call_count == 4

        # Check event types
        calls = event_bus.publish.call_args_list
        assert calls[0][0][0] == EventTypes.MESSAGE_RECEIVED
        assert calls[1][0][0] == EventTypes.CONVERSATION_UPDATED
        assert calls[2][0][0] == "save_conversation_history"
        assert calls[3][0][0] == "display_message"

    def test_on_message_received_includes_conversation_id(self):
        """Test message received includes conversation ID."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_message_received("Test message")

        # Check MESSAGE_RECEIVED event data
        call_args = event_bus.publish.call_args_list[0]
        assert call_args[0][0] == EventTypes.MESSAGE_RECEIVED
        data = call_args[0][1]
        assert data["conversation_id"] == "conv123"
        assert data["message_type"] == "agent"
        assert data["content"] == "Test message"

    def test_on_message_received_without_conversation_id(self):
        """Test message received when no conversation ID."""
        event_bus = Mock()
        provider = Mock(return_value=None)

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_message_received("Test message")

        # Should only publish display_message event (1 call)
        assert event_bus.publish.call_count == 1
        assert event_bus.publish.call_args[0][0] == "display_message"

    def test_on_message_received_success_color(self):
        """Test message with success indicator uses success color."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_message_received("✅ Task completed successfully")

        # Check display_message event
        display_call = [
            c for c in event_bus.publish.call_args_list if c[0][0] == "display_message"
        ][0]
        data = display_call[0][1]
        assert data["color"] == Colors.SUCCESS_MESSAGE

    def test_on_message_received_completed_color(self):
        """Test message with 'completed' uses success color."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_message_received("Task completed")

        # Check display_message event
        display_call = [
            c for c in event_bus.publish.call_args_list if c[0][0] == "display_message"
        ][0]
        data = display_call[0][1]
        assert data["color"] == Colors.SUCCESS_MESSAGE

    def test_on_message_received_normal_color(self):
        """Test normal message uses agent message color."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_message_received("Normal message")

        # Check display_message event
        display_call = [
            c for c in event_bus.publish.call_args_list if c[0][0] == "display_message"
        ][0]
        data = display_call[0][1]
        assert data["color"] == Colors.AGENT_MESSAGE

    def test_on_message_received_auto_open_plots(self):
        """Test message enables auto-open plots."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_message_received("Test message")

        # Check display_message event
        display_call = [
            c for c in event_bus.publish.call_args_list if c[0][0] == "display_message"
        ][0]
        data = display_call[0][1]
        assert data["auto_open_plots"] is True


class TestOnStatusUpdate:
    """Test suite for on_status_update handler."""

    def test_on_status_update_publishes_events(self):
        """Test status update publishes correct events."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_status_update("Processing...", "orchestrator")

        # Should publish 2 events
        assert event_bus.publish.call_count == 2

        # Check event types
        calls = event_bus.publish.call_args_list
        assert calls[0][0][0] == EventTypes.STATUS_UPDATE
        assert calls[1][0][0] == "update_status_bar"

    def test_on_status_update_includes_component(self):
        """Test status update includes component type."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_status_update("Processing...", "router")

        # Check STATUS_UPDATE event data
        call_args = event_bus.publish.call_args_list[0]
        data = call_args[0][1]
        assert data["status"] == "Processing..."
        assert data["component"] == "router"

    def test_on_status_update_with_model_info(self):
        """Test status update with model information."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        model_info = {"model_provider": "openai", "model_id": "gpt-4"}
        handlers = MessageHandlers(event_bus, provider)
        handlers.on_status_update("Processing...", "orchestrator", model_info)

        # Check STATUS_UPDATE event data
        call_args = event_bus.publish.call_args_list[0]
        data = call_args[0][1]
        assert data["model_info"] == model_info

    def test_on_status_update_without_model_info(self):
        """Test status update without model information."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_status_update("Processing...")

        # Check STATUS_UPDATE event data
        call_args = event_bus.publish.call_args_list[0]
        data = call_args[0][1]
        assert data["model_info"] == {}
        assert data["component"] == "base"

    def test_on_status_update_status_bar(self):
        """Test status update updates status bar."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_status_update("Test status")

        # Check update_status_bar event
        status_bar_call = [
            c for c in event_bus.publish.call_args_list if c[0][0] == "update_status_bar"
        ][0]
        data = status_bar_call[0][1]
        assert data["message"] == "Test status"


class TestOnError:
    """Test suite for on_error handler."""

    def test_on_error_publishes_events(self):
        """Test error publishes correct events."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_error("Test error")

        # Should publish 3 events
        assert event_bus.publish.call_count == 3

        # Check event types
        calls = event_bus.publish.call_args_list
        assert calls[0][0][0] == EventTypes.ERROR_OCCURRED
        assert calls[1][0][0] == "display_error"
        assert calls[2][0][0] == EventTypes.STATUS_UPDATE

    def test_on_error_includes_error_message(self):
        """Test error includes error message."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_error("Connection failed")

        # Check ERROR_OCCURRED event
        error_call = event_bus.publish.call_args_list[0]
        assert error_call[0][1]["error"] == "Connection failed"

        # Check display_error event
        display_call = event_bus.publish.call_args_list[1]
        assert display_call[0][1]["error"] == "Connection failed"

    def test_on_error_updates_status(self):
        """Test error updates status with error component."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_error("Test error")

        # Check STATUS_UPDATE event
        status_call = event_bus.publish.call_args_list[2]
        data = status_call[0][1]
        assert "Error: Test error" in data["status"]
        assert data["component"] == "error"


class TestOnLLMDetail:
    """Test suite for on_llm_detail handler."""

    @patch("osprey.interfaces.pyqt.message_handlers.datetime")
    def test_on_llm_detail_publishes_event(self, mock_datetime):
        """Test LLM detail publishes event with timestamp."""
        mock_now = Mock()
        mock_now.strftime.return_value = "12:34:56.789"
        mock_datetime.now.return_value = mock_now

        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_llm_detail("LLM processing", "llm_start")

        # Should publish 1 event
        assert event_bus.publish.call_count == 1
        assert event_bus.publish.call_args[0][0] == EventTypes.LLM_DETAIL

    @patch("osprey.interfaces.pyqt.message_handlers.datetime")
    def test_on_llm_detail_includes_timestamp(self, mock_datetime):
        """Test LLM detail includes timestamp."""
        mock_now = Mock()
        # strftime returns full microseconds, then [:-3] truncates to milliseconds
        mock_now.strftime.return_value = "12:34:56.789123"
        mock_datetime.now.return_value = mock_now

        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_llm_detail("Test detail", "llm_stream")

        # Check event data
        data = event_bus.publish.call_args[0][1]
        assert data["detail"] == "Test detail"
        assert data["event_type"] == "llm_stream"
        assert data["timestamp"] == "12:34:56.789"

    @patch("osprey.interfaces.pyqt.message_handlers.datetime")
    def test_on_llm_detail_default_event_type(self, mock_datetime):
        """Test LLM detail with default event type."""
        mock_now = Mock()
        mock_now.strftime.return_value = "12:34:56.789"
        mock_datetime.now.return_value = mock_now

        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_llm_detail("Test detail")

        # Check event data
        data = event_bus.publish.call_args[0][1]
        assert data["event_type"] == "base"

    @patch("osprey.interfaces.pyqt.message_handlers.datetime")
    def test_on_llm_detail_different_event_types(self, mock_datetime):
        """Test LLM detail with different event types."""
        mock_now = Mock()
        mock_now.strftime.return_value = "12:34:56.789"
        mock_datetime.now.return_value = mock_now

        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)

        event_types = ["llm_start", "llm_end", "llm_stream", "classification"]
        for event_type in event_types:
            event_bus.reset_mock()
            handlers.on_llm_detail("Detail", event_type)
            data = event_bus.publish.call_args[0][1]
            assert data["event_type"] == event_type


class TestOnToolUsage:
    """Test suite for on_tool_usage handler."""

    @patch("osprey.interfaces.pyqt.message_handlers.datetime")
    def test_on_tool_usage_publishes_event(self, mock_datetime):
        """Test tool usage publishes event."""
        mock_now = Mock()
        mock_now.strftime.return_value = "12:34:56"
        mock_datetime.now.return_value = mock_now

        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_tool_usage("test_tool", "Tool reasoning")

        # Should publish 1 event
        assert event_bus.publish.call_count == 1
        assert event_bus.publish.call_args[0][0] == EventTypes.TOOL_USAGE

    @patch("osprey.interfaces.pyqt.message_handlers.datetime")
    def test_on_tool_usage_includes_details(self, mock_datetime):
        """Test tool usage includes tool name and reasoning."""
        mock_now = Mock()
        mock_now.strftime.return_value = "12:34:56"
        mock_datetime.now.return_value = mock_now

        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_tool_usage("weather_api", "Fetching weather data")

        # Check event data
        data = event_bus.publish.call_args[0][1]
        assert data["tool_name"] == "weather_api"
        assert data["reasoning"] == "Fetching weather data"
        assert data["timestamp"] == "12:34:56"

    @patch("osprey.interfaces.pyqt.message_handlers.datetime")
    def test_on_tool_usage_timestamp_format(self, mock_datetime):
        """Test tool usage timestamp format."""
        mock_now = Mock()
        mock_now.strftime.return_value = "09:15:30"
        mock_datetime.now.return_value = mock_now

        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_tool_usage("tool", "reasoning")

        # Verify strftime was called with correct format
        mock_now.strftime.assert_called_once_with("%H:%M:%S")


class TestOnProcessingComplete:
    """Test suite for on_processing_complete handler."""

    def test_on_processing_complete_publishes_event(self):
        """Test processing complete publishes event."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_processing_complete()

        # Should publish 1 event
        assert event_bus.publish.call_count == 1
        assert event_bus.publish.call_args[0][0] == EventTypes.PROCESSING_COMPLETE

    def test_on_processing_complete_empty_data(self):
        """Test processing complete sends empty data."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_processing_complete()

        # Check event data is empty dict
        data = event_bus.publish.call_args[0][1]
        assert data == {}


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_multiple_handlers_independent(self):
        """Test multiple handler instances are independent."""
        event_bus1 = Mock()
        event_bus2 = Mock()
        provider1 = Mock(return_value="conv1")
        provider2 = Mock(return_value="conv2")

        handlers1 = MessageHandlers(event_bus1, provider1)
        handlers2 = MessageHandlers(event_bus2, provider2)

        handlers1.on_message_received("Message 1")
        handlers2.on_message_received("Message 2")

        # Each should only publish to their own event bus
        assert event_bus1.publish.called
        assert event_bus2.publish.called

        # Check conversation IDs are different
        conv1_call = [
            c for c in event_bus1.publish.call_args_list if c[0][0] == EventTypes.MESSAGE_RECEIVED
        ][0]
        conv2_call = [
            c for c in event_bus2.publish.call_args_list if c[0][0] == EventTypes.MESSAGE_RECEIVED
        ][0]

        assert conv1_call[0][1]["conversation_id"] == "conv1"
        assert conv2_call[0][1]["conversation_id"] == "conv2"

    def test_empty_message(self):
        """Test handling empty message."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_message_received("")

        # Should still publish events
        assert event_bus.publish.called

    def test_empty_error(self):
        """Test handling empty error."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)
        handlers.on_error("")

        # Should still publish events
        assert event_bus.publish.call_count == 3

    def test_conversation_id_changes(self):
        """Test handling when conversation ID changes."""
        event_bus = Mock()
        provider = Mock(side_effect=["conv1", "conv2"])

        handlers = MessageHandlers(event_bus, provider)

        handlers.on_message_received("Message 1")
        event_bus.reset_mock()
        handlers.on_message_received("Message 2")

        # Second message should use conv2
        msg_call = [
            c for c in event_bus.publish.call_args_list if c[0][0] == EventTypes.MESSAGE_RECEIVED
        ][0]
        assert msg_call[0][1]["conversation_id"] == "conv2"

    def test_special_characters_in_messages(self):
        """Test handling special characters in messages."""
        event_bus = Mock()
        provider = Mock(return_value="conv123")

        handlers = MessageHandlers(event_bus, provider)

        special_messages = [
            "Message with emoji 🎉",
            "Message with\nnewlines\n",
            "Message with\ttabs",
            "Message with 'quotes' and \"double quotes\"",
        ]

        for msg in special_messages:
            event_bus.reset_mock()
            handlers.on_message_received(msg)

            # Should handle without errors
            assert event_bus.publish.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
