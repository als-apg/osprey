"""Framework State - Message and Memory Management.

This module provides comprehensive message and memory management utilities for the
Osprey Agent Framework, built on LangGraph's native message types for
optimal compatibility and performance. The message system handles conversation
state, chat history formatting, and persistent user memory across sessions.

**Core Components:**

- :class:`MessageUtils`: LangGraph-native message creation and manipulation utilities
- :class:`ChatHistoryFormatter`: Conversation formatting for LLM consumption and UI display
- :class:`UserMemories`: Persistent user context and memory management

**Message Architecture:**

The message system is built on LangGraph's native message types (HumanMessage,
AIMessage) to ensure seamless integration with LangGraph's checkpointing and
serialization systems. All message operations maintain compatibility with
LangGraph's automatic message handling.

**Key Features:**

1. **Native Compatibility**: Full integration with LangGraph's MessagesState
2. **Timestamp Support**: Automatic timestamp metadata for all messages
3. **Flexible Formatting**: Multiple formatting options for different contexts
4. **Memory Integration**: Persistent user context across conversation sessions
5. **Type Safety**: Comprehensive type annotations and validation

**Usage Patterns:**

The message utilities are designed for use throughout the framework:
- Infrastructure components for message creation
- Capabilities for accessing conversation history
- UI components for displaying formatted conversations
- Memory systems for persistent user context

.. note::
   All message utilities work with LangGraph's native message types to ensure
   compatibility with automatic checkpointing and state management.

.. seealso::
   :class:`osprey.state.AgentState` : Main state containing message history
   :mod:`langchain_core.messages` : LangGraph native message types
   :class:`osprey.state.StateManager` : State management using message utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

# LangGraph native message types for checkpointing compatibility
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class MessageUtils:
    """Comprehensive utilities for LangGraph-native message creation and manipulation.

    This utility class provides a complete suite of static methods for creating,
    manipulating, and extracting information from LangGraph's native message types.
    The utilities ensure consistent message formatting, metadata handling, and
    timestamp management throughout the framework.

    **Core Functionality:**

    - **Message Creation**: Create properly formatted HumanMessage and AIMessage instances
    - **Metadata Management**: Handle timestamp and custom metadata consistently
    - **Information Extraction**: Extract roles, timestamps, and content from messages
    - **Type Safety**: Comprehensive type checking and validation

    **LangGraph Integration:**

    All MessageUtils methods work exclusively with LangGraph's native message types
    to ensure seamless integration with MessagesState, checkpointing, and serialization.
    The utilities maintain compatibility with LangGraph's automatic message handling.

    .. note::
       All methods in MessageUtils are static utilities that do not require
       instantiation. The class serves as a namespace for message operations.

    .. seealso::
       :mod:`langchain_core.messages` : LangGraph native message types
       :class:`osprey.state.AgentState` : State using MessageUtils for message handling
    """

    @staticmethod
    def create_user_message(content: str, timestamp: datetime | None = None) -> HumanMessage:
        """Create a user message with optional timestamp metadata."""
        metadata = {"timestamp": timestamp.isoformat()} if timestamp else {}
        return HumanMessage(content=content, additional_kwargs=metadata)

    @staticmethod
    def create_assistant_message(content: str, timestamp: datetime | None = None) -> AIMessage:
        """Create an assistant message with optional timestamp metadata."""
        metadata = {"timestamp": timestamp.isoformat()} if timestamp else {}
        return AIMessage(content=content, additional_kwargs=metadata)

    @staticmethod
    def get_timestamp(message: BaseMessage) -> datetime | None:
        """Extract timestamp from message metadata."""
        timestamp_str = message.additional_kwargs.get("timestamp")
        if timestamp_str:
            return datetime.fromisoformat(timestamp_str)
        return None

    @staticmethod
    def get_role(message: BaseMessage) -> str:
        """Get the role of a message (user or assistant)."""
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        return "unknown"


class ChatHistoryFormatter:
    """Formatting utilities for chat history using native message types."""

    @staticmethod
    def format_for_llm(messages: list[BaseMessage]) -> str:
        """Format the entire chat history for LLM consumption.

        Creates a numbered, formatted representation of all messages in the
        conversation suitable for processing by language models.

        :param messages: List of LangGraph native messages
        :return: Formatted string representation of the chat history
        """
        if not messages:
            return "No conversation history available"

        formatted_messages = []
        for i, message in enumerate(messages, 1):
            role = MessageUtils.get_role(message)
            timestamp = MessageUtils.get_timestamp(message)
            timestamp_str = f" ({timestamp.strftime('%H:%M:%S')})" if timestamp else ""
            formatted_messages.append(f"{i}. {role.upper()}{timestamp_str}: {message.content}")

        return "\n".join(formatted_messages)

    @staticmethod
    def get_latest_user_message(messages: list[BaseMessage]) -> str | None:
        """Get the content of the most recent user message.

        Searches the message history in reverse order to find the most recent
        message from a user and returns its content.

        :param messages: List of LangGraph native messages
        :return: Content of the latest user message, or None if no user messages exist
        """
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content
        return None

    @staticmethod
    def format_for_prompt(messages: list[BaseMessage]) -> str:
        """Format chat history for inclusion in prompts.

        Creates a well-formatted representation of the chat history optimized
        for inclusion in LLM prompts, with proper indentation and role formatting.

        :param messages: List of LangGraph native messages
        :return: Formatted string representation optimized for prompt inclusion
        """
        if not messages:
            return "**No messages in chat history**"

        formatted_messages = []

        for msg in messages:
            role = MessageUtils.get_role(msg)
            timestamp = MessageUtils.get_timestamp(msg)

            # Format timestamp if available
            timestamp_str = ""
            if timestamp:
                timestamp_str = f" [{timestamp.strftime('%H:%M:%S')}]"

            # Format role
            role_display = role.title()

            # Format content with proper indentation for multi-line messages
            content_lines = msg.content.split("\n")
            if len(content_lines) == 1:
                content_formatted = content_lines[0]
            else:
                content_formatted = (
                    content_lines[0] + "\n" + "\n".join(f"    {line}" for line in content_lines[1:])
                )

            formatted_messages.append(f"**{role_display}{timestamp_str}:** {content_formatted}")

        return "\n\n".join(formatted_messages)


@dataclass
class UserMemories:
    """Collection of user memory entries for persistent user context.

    Manages a collection of memory entries that provide persistent context
    about the user across conversations and sessions.

    :param entries: List of memory entry strings
    :type entries: list[str]
    """

    entries: list[str]

    def __bool__(self) -> bool:
        """Check if memory has any entries.

        :return: True if memory contains any entries
        :rtype: bool
        """
        return bool(self.entries)

    def __len__(self) -> int:
        """Get the number of memory entries.

        :return: Number of memory entries
        :rtype: int
        """
        return len(self.entries)

    def format_for_prompt(self) -> str:
        """Format memory entries for inclusion in LLM prompts.

        Creates a formatted representation of all memory entries suitable for
        inclusion in LLM prompts as context about the user.

        :return: Formatted string of memory entries, or empty string if no entries
        :rtype: str
        """
        if not self.entries:
            return ""
        return "\n".join(f"- {entry}" for entry in self.entries)
