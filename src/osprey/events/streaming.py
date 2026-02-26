"""Multi-mode streaming helpers for Osprey event consumption.

This module provides helper functions and async generators for consuming
multi-mode streams from LangGraph, combining custom OspreyEvents with
LLM token streaming.

Usage:
    from osprey.events.streaming import consume_stream, StreamMode

    async for event_or_token in consume_stream(graph, input_data, config):
        if isinstance(event_or_token, OspreyEvent):
            # Handle typed event
            await handler.handle(event_or_token)
        elif isinstance(event_or_token, LLMToken):
            # Handle LLM token
            print(event_or_token.content, end="", flush=True)
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .parser import parse_event
from .types import OspreyEvent


class StreamMode(StrEnum):
    """Available LangGraph stream modes."""

    CUSTOM = "custom"
    MESSAGES = "messages"
    UPDATES = "updates"
    VALUES = "values"


@dataclass
class LLMToken:
    """Represents an LLM token from the messages stream.

    Attributes:
        content: The token content string
        metadata: Additional metadata from the stream
    """

    content: str
    metadata: dict[str, Any] | None = None


@dataclass
class StateUpdate:
    """Represents a state update from the updates stream.

    Attributes:
        node_name: The node that produced this update
        state: The updated state values
    """

    node_name: str
    state: dict[str, Any]


# Type alias for stream output
StreamOutput = OspreyEvent | LLMToken | StateUpdate


async def consume_stream(
    graph,
    input_data: dict[str, Any],
    config: dict[str, Any],
    modes: list[StreamMode] | None = None,
) -> AsyncGenerator[StreamOutput, None]:
    """Consume a multi-mode stream and yield typed outputs.

    This async generator simplifies consuming multi-mode LangGraph streams
    by parsing chunks into typed outputs: OspreyEvents, LLMTokens, or StateUpdates.

    Args:
        graph: Compiled LangGraph StateGraph
        input_data: Input data for the graph
        config: LangGraph config (with thread_id, etc.)
        modes: Stream modes to enable. Defaults to ["custom", "messages"]

    Yields:
        OspreyEvent: For custom events (status, capability start/complete, etc.)
        LLMToken: For LLM token streaming
        StateUpdate: For state updates (if "updates" mode enabled)

    Example:
        from osprey.events.streaming import consume_stream, StreamMode

        async for output in consume_stream(graph, input_data, config):
            match output:
                case OspreyEvent() as event:
                    await handler.handle(event)
                case LLMToken(content=text):
                    print(text, end="", flush=True)
                case StateUpdate(node_name=node):
                    logger.debug(f"Node {node} updated state")

    Note:
        This helper abstracts away the tuple unpacking required for multi-mode
        streams, providing a cleaner consumption pattern.
    """
    if modes is None:
        modes = [StreamMode.CUSTOM, StreamMode.MESSAGES]

    # Convert to string list for LangGraph
    mode_strings = [m.value if isinstance(m, StreamMode) else m for m in modes]

    # Use multi-mode streaming
    async for mode, chunk in graph.astream(input_data, config=config, stream_mode=mode_strings):
        if mode == StreamMode.CUSTOM.value:
            # Parse as typed OspreyEvent
            event = parse_event(chunk) if isinstance(chunk, dict) else None
            if event:
                yield event

        elif mode == StreamMode.MESSAGES.value:
            # Extract LLM token content
            if isinstance(chunk, tuple) and len(chunk) == 2:
                message_chunk, metadata = chunk
                if hasattr(message_chunk, "content") and message_chunk.content:
                    yield LLMToken(content=message_chunk.content, metadata=metadata)
            elif hasattr(chunk, "content") and chunk.content:
                yield LLMToken(content=chunk.content)

        elif mode == StreamMode.UPDATES.value:
            # State update - chunk is dict of {node_name: state_update}
            if isinstance(chunk, dict):
                for node_name, state in chunk.items():
                    yield StateUpdate(node_name=node_name, state=state)

        # "values" mode yields final state - not typically used in streaming


async def consume_custom_events(
    graph,
    input_data: dict[str, Any],
    config: dict[str, Any],
) -> AsyncGenerator[OspreyEvent, None]:
    """Consume only custom events from a stream.

    Simpler helper for when you only need typed OspreyEvents without
    LLM token streaming.

    Args:
        graph: Compiled LangGraph StateGraph
        input_data: Input data for the graph
        config: LangGraph config

    Yields:
        OspreyEvent: Typed events from the stream

    Example:
        async for event in consume_custom_events(graph, input_data, config):
            await handler.handle(event)
    """
    async for chunk in graph.astream(
        input_data, config=config, stream_mode=StreamMode.CUSTOM.value
    ):
        event = parse_event(chunk) if isinstance(chunk, dict) else None
        if event:
            yield event
