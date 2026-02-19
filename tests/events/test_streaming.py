"""Tests for multi-mode streaming helpers."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from osprey.events.streaming import (
    LLMToken,
    StateUpdate,
    StreamMode,
    consume_custom_events,
    consume_stream,
)

# =============================================================================
# Test StreamMode Enum
# =============================================================================


class TestStreamMode:
    """Test StreamMode enum."""

    def test_stream_mode_values(self):
        """Verify StreamMode enum values match LangGraph."""
        assert StreamMode.CUSTOM.value == "custom"
        assert StreamMode.MESSAGES.value == "messages"
        assert StreamMode.UPDATES.value == "updates"
        assert StreamMode.VALUES.value == "values"

    def test_stream_mode_is_string_enum(self):
        """Verify StreamMode inherits from str."""
        assert isinstance(StreamMode.CUSTOM, str)
        assert StreamMode.CUSTOM == "custom"


# =============================================================================
# Test LLMToken Dataclass
# =============================================================================


class TestLLMToken:
    """Test LLMToken dataclass."""

    def test_creation_with_content(self):
        """Test LLMToken creation with content."""
        token = LLMToken(content="Hello")
        assert token.content == "Hello"
        assert token.metadata is None

    def test_creation_with_metadata(self):
        """Test LLMToken creation with metadata."""
        metadata = {"run_id": "abc123", "model": "gpt-4"}
        token = LLMToken(content="World", metadata=metadata)
        assert token.content == "World"
        assert token.metadata == metadata

    def test_empty_content(self):
        """Test LLMToken with empty content."""
        token = LLMToken(content="")
        assert token.content == ""


# =============================================================================
# Test StateUpdate Dataclass
# =============================================================================


class TestStateUpdate:
    """Test StateUpdate dataclass."""

    def test_creation(self):
        """Test StateUpdate creation."""
        state = {"key": "value", "count": 42}
        update = StateUpdate(node_name="classifier", state=state)
        assert update.node_name == "classifier"
        assert update.state == state

    def test_empty_state(self):
        """Test StateUpdate with empty state."""
        update = StateUpdate(node_name="router", state={})
        assert update.node_name == "router"
        assert update.state == {}


# =============================================================================
# Test consume_stream Function
# =============================================================================


class TestConsumeStream:
    """Test consume_stream async generator."""

    @pytest.mark.asyncio
    async def test_consume_stream_yields_osprey_events(self):
        """Test that consume_stream yields OspreyEvents from custom mode."""
        from osprey.events import StatusEvent

        mock_graph = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield (
                "custom",
                {
                    "event_class": "StatusEvent",
                    "message": "Test status",
                    "level": "info",
                    "component": "test",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        mock_graph.astream = mock_astream

        events = []
        async for output in consume_stream(mock_graph, {}, {}):
            events.append(output)

        assert len(events) == 1
        assert isinstance(events[0], StatusEvent)
        assert events[0].message == "Test status"

    @pytest.mark.asyncio
    async def test_consume_stream_yields_llm_tokens(self):
        """Test that consume_stream yields LLMTokens from messages mode."""
        mock_graph = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Hello token"

        async def mock_astream(*args, **kwargs):
            yield "messages", (mock_message, {"run_id": "test"})

        mock_graph.astream = mock_astream

        outputs = []
        async for output in consume_stream(mock_graph, {}, {}):
            outputs.append(output)

        assert len(outputs) == 1
        assert isinstance(outputs[0], LLMToken)
        assert outputs[0].content == "Hello token"
        assert outputs[0].metadata == {"run_id": "test"}

    @pytest.mark.asyncio
    async def test_consume_stream_yields_state_updates(self):
        """Test that consume_stream yields StateUpdates from updates mode."""
        mock_graph = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield "updates", {"classifier": {"active_capabilities": ["python"]}}

        mock_graph.astream = mock_astream

        outputs = []
        async for output in consume_stream(mock_graph, {}, {}, modes=[StreamMode.UPDATES]):
            outputs.append(output)

        assert len(outputs) == 1
        assert isinstance(outputs[0], StateUpdate)
        assert outputs[0].node_name == "classifier"
        assert outputs[0].state == {"active_capabilities": ["python"]}

    @pytest.mark.asyncio
    async def test_consume_stream_default_modes(self):
        """Test that default modes are custom and messages."""

        mock_graph = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Token"

        async def mock_astream(*args, **kwargs):
            modes = kwargs.get("stream_mode", [])
            assert "custom" in modes
            assert "messages" in modes
            yield (
                "custom",
                {
                    "event_class": "StatusEvent",
                    "message": "Test",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            yield "messages", (mock_message, {})

        mock_graph.astream = mock_astream

        outputs = []
        async for output in consume_stream(mock_graph, {}, {}):
            outputs.append(output)

        assert len(outputs) == 2

    @pytest.mark.asyncio
    async def test_consume_stream_filters_invalid_events(self):
        """Test that invalid chunks are skipped."""
        mock_graph = MagicMock()

        async def mock_astream(*args, **kwargs):
            # Invalid: not a dict
            yield "custom", "not a dict"
            # Invalid: no event_class
            yield "custom", {"message": "test"}
            # Valid
            yield (
                "custom",
                {
                    "event_class": "StatusEvent",
                    "message": "Valid",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        mock_graph.astream = mock_astream

        outputs = []
        async for output in consume_stream(mock_graph, {}, {}):
            outputs.append(output)

        # Only the valid event should be yielded
        assert len(outputs) == 1

    @pytest.mark.asyncio
    async def test_consume_stream_handles_message_without_tuple(self):
        """Test handling of messages that aren't tuples."""
        mock_graph = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Direct token"

        async def mock_astream(*args, **kwargs):
            yield "messages", mock_message

        mock_graph.astream = mock_astream

        outputs = []
        async for output in consume_stream(mock_graph, {}, {}):
            outputs.append(output)

        assert len(outputs) == 1
        assert isinstance(outputs[0], LLMToken)
        assert outputs[0].content == "Direct token"

    @pytest.mark.asyncio
    async def test_consume_stream_skips_empty_tokens(self):
        """Test that empty LLM tokens are skipped."""
        mock_graph = MagicMock()
        mock_message = MagicMock()
        mock_message.content = ""

        async def mock_astream(*args, **kwargs):
            yield "messages", (mock_message, {})

        mock_graph.astream = mock_astream

        outputs = []
        async for output in consume_stream(mock_graph, {}, {}):
            outputs.append(output)

        # Empty content should be skipped
        assert len(outputs) == 0


# =============================================================================
# Test consume_custom_events Function
# =============================================================================


class TestConsumeCustomEvents:
    """Test consume_custom_events async generator."""

    @pytest.mark.asyncio
    async def test_consume_custom_events_only_custom(self):
        """Test that consume_custom_events only yields OspreyEvents."""
        from osprey.events import PhaseStartEvent, StatusEvent

        mock_graph = MagicMock()

        async def mock_astream(*args, **kwargs):
            # Verify only custom mode is used
            assert kwargs.get("stream_mode") == "custom"
            yield {
                "event_class": "StatusEvent",
                "message": "Status",
                "timestamp": datetime.now().isoformat(),
            }
            yield {
                "event_class": "PhaseStartEvent",
                "phase": "execution",
                "timestamp": datetime.now().isoformat(),
            }

        mock_graph.astream = mock_astream

        events = []
        async for event in consume_custom_events(mock_graph, {}, {}):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], StatusEvent)
        assert isinstance(events[1], PhaseStartEvent)

    @pytest.mark.asyncio
    async def test_consume_custom_events_filters_non_events(self):
        """Test that non-event dicts are skipped."""
        mock_graph = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield "not a dict"
            yield {"no_event_class": "test"}
            yield {
                "event_class": "StatusEvent",
                "message": "Valid",
                "timestamp": datetime.now().isoformat(),
            }

        mock_graph.astream = mock_astream

        events = []
        async for event in consume_custom_events(mock_graph, {}, {}):
            events.append(event)

        assert len(events) == 1


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestStreamingEdgeCases:
    """Test edge cases in streaming helpers."""

    @pytest.mark.asyncio
    async def test_consume_stream_with_string_modes(self):
        """Test consume_stream accepts string modes."""
        mock_graph = MagicMock()

        async def mock_astream(*args, **kwargs):
            modes = kwargs.get("stream_mode", [])
            assert "custom" in modes
            # Return nothing
            return
            yield  # Make it a generator

        mock_graph.astream = mock_astream

        outputs = []
        async for output in consume_stream(mock_graph, {}, {}, modes=["custom"]):
            outputs.append(output)

        assert len(outputs) == 0

    @pytest.mark.asyncio
    async def test_consume_stream_with_enum_modes(self):
        """Test consume_stream accepts enum modes."""
        mock_graph = MagicMock()

        async def mock_astream(*args, **kwargs):
            modes = kwargs.get("stream_mode", [])
            assert "custom" in modes
            assert "messages" in modes
            return
            yield

        mock_graph.astream = mock_astream

        outputs = []
        async for output in consume_stream(
            mock_graph, {}, {}, modes=[StreamMode.CUSTOM, StreamMode.MESSAGES]
        ):
            outputs.append(output)

        assert len(outputs) == 0

    @pytest.mark.asyncio
    async def test_consume_stream_multiple_state_updates(self):
        """Test that multiple nodes in one update are yielded separately."""
        mock_graph = MagicMock()

        async def mock_astream(*args, **kwargs):
            yield (
                "updates",
                {
                    "classifier": {"caps": ["python"]},
                    "orchestrator": {"plan": []},
                },
            )

        mock_graph.astream = mock_astream

        outputs = []
        async for output in consume_stream(mock_graph, {}, {}, modes=[StreamMode.UPDATES]):
            outputs.append(output)

        assert len(outputs) == 2
        node_names = {out.node_name for out in outputs}
        assert "classifier" in node_names
        assert "orchestrator" in node_names
