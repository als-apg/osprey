"""
Conversation Manager for Osprey GUI

Handles conversation history management including:
- Creating, switching, deleting, and renaming conversations
- Saving and loading conversation history (JSON or PostgreSQL)
- Managing conversation metadata and messages
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from osprey.utils.logger import get_logger

logger = get_logger("conversation_manager")


@dataclass
class ConversationMessage:
    """A single message in a conversation."""

    type: str  # 'user' or 'agent'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    formatting: str | None = None  # 'orchestrated' or None for special formatting

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.formatting:
            result["formatting"] = self.formatting
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except (ValueError, TypeError):
                timestamp = datetime.now()
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()

        return cls(
            type=data["type"],
            content=data["content"],
            timestamp=timestamp,
            formatting=data.get("formatting"),
        )


@dataclass
class Conversation:
    """A conversation with its metadata and messages."""

    thread_id: str
    name: str
    messages: list[ConversationMessage] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "thread_id": self.thread_id,
            "name": self.name,
            "messages": [msg.to_dict() for msg in self.messages],
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except (ValueError, TypeError):
                timestamp = datetime.now()
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()

        messages = [ConversationMessage.from_dict(msg) for msg in data.get("messages", [])]

        return cls(
            thread_id=data["thread_id"], name=data["name"], messages=messages, timestamp=timestamp
        )


class ConversationManager:
    """
    Manages conversation history for the GUI.

    Handles:
    - Creating and managing conversations
    - Saving/loading from JSON or PostgreSQL
    - Conversation metadata and messages
    """

    def __init__(self, storage_mode: str = "json"):
        """
        Initialize conversation manager.

        Args:
            storage_mode: 'json' or 'postgresql'
        """
        self.storage_mode = storage_mode
        self.conversations: dict[str, Conversation] = {}
        self.current_conversation_id: str | None = None

    def create_conversation(self, name: str | None = None) -> str:
        """
        Create a new conversation.

        Args:
            name: Optional name for the conversation

        Returns:
            Thread ID of the new conversation
        """
        thread_id = f"gui_session_{uuid.uuid4().hex[:8]}"

        if name is None:
            conv_number = len(self.conversations) + 1
            name = f"Conversation {conv_number}"

        conversation = Conversation(
            thread_id=thread_id, name=name, messages=[], timestamp=datetime.now()
        )

        self.conversations[thread_id] = conversation
        self.current_conversation_id = thread_id

        logger.info(f"Created new conversation: {name} ({thread_id})")
        return thread_id

    def get_conversation(self, thread_id: str) -> Conversation | None:
        """Get a conversation by thread ID."""
        return self.conversations.get(thread_id)

    def get_current_conversation(self) -> Conversation | None:
        """Get the current active conversation."""
        if self.current_conversation_id:
            return self.conversations.get(self.current_conversation_id)
        return None

    def switch_conversation(self, thread_id: str) -> bool:
        """
        Switch to a different conversation.

        Args:
            thread_id: Thread ID to switch to

        Returns:
            True if successful, False if conversation not found
        """
        if thread_id not in self.conversations:
            logger.warning(f"Conversation {thread_id} not found")
            return False

        self.current_conversation_id = thread_id
        logger.info(f"Switched to conversation: {self.conversations[thread_id].name}")
        return True

    def delete_conversation(self, thread_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            thread_id: Thread ID to delete

        Returns:
            True if successful, False if conversation not found or is the last one
        """
        if thread_id not in self.conversations:
            logger.warning(f"Conversation {thread_id} not found")
            return False

        # Don't allow deleting the last conversation
        if len(self.conversations) == 1:
            logger.warning("Cannot delete the last conversation")
            return False

        # If deleting current conversation, switch to another one
        if thread_id == self.current_conversation_id:
            # Find another conversation to switch to
            for other_id in self.conversations:
                if other_id != thread_id:
                    self.current_conversation_id = other_id
                    break

        conv_name = self.conversations[thread_id].name
        del self.conversations[thread_id]
        logger.info(f"Deleted conversation: {conv_name}")
        return True

    def rename_conversation(self, thread_id: str, new_name: str) -> bool:
        """
        Rename a conversation.

        Args:
            thread_id: Thread ID to rename
            new_name: New name for the conversation

        Returns:
            True if successful, False if conversation not found
        """
        if thread_id not in self.conversations:
            logger.warning(f"Conversation {thread_id} not found")
            return False

        old_name = self.conversations[thread_id].name
        self.conversations[thread_id].name = new_name.strip()
        logger.info(f"Renamed conversation: '{old_name}' → '{new_name}'")
        return True

    def add_message(
        self, thread_id: str, message_type: str, content: str, formatting: str | None = None
    ) -> bool:
        """
        Add a message to a conversation.

        Args:
            thread_id: Thread ID to add message to
            message_type: 'user' or 'agent'
            content: Message content
            formatting: Optional formatting hint ('orchestrated', etc.)

        Returns:
            True if successful, False if conversation not found
        """
        if thread_id not in self.conversations:
            logger.warning(f"Conversation {thread_id} not found")
            return False

        message = ConversationMessage(
            type=message_type, content=content, timestamp=datetime.now(), formatting=formatting
        )

        self.conversations[thread_id].messages.append(message)
        self.conversations[thread_id].timestamp = datetime.now()
        return True

    def get_messages(self, thread_id: str) -> list[ConversationMessage]:
        """Get all messages from a conversation."""
        conversation = self.conversations.get(thread_id)
        if conversation:
            return conversation.messages
        return []

    def list_conversations(self, sort_by_timestamp: bool = True) -> list[Conversation]:
        """
        List all conversations.

        Args:
            sort_by_timestamp: If True, sort by timestamp (newest first)

        Returns:
            List of conversations
        """
        conversations = list(self.conversations.values())

        if sort_by_timestamp:
            conversations.sort(key=lambda c: c.timestamp, reverse=True)

        return conversations

    def save_to_json(self, file_path: Path) -> bool:
        """
        Save conversations to JSON file.

        Args:
            file_path: Path to save JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert all conversations to dictionaries
            data = {thread_id: conv.to_dict() for thread_id, conv in self.conversations.items()}

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.conversations)} conversation(s) to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save conversations to JSON: {e}")
            return False

    def load_from_json(self, file_path: Path) -> bool:
        """
        Load conversations from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            if not file_path.exists():
                logger.debug(f"No conversation history file found at {file_path}")
                return False

            with open(file_path) as f:
                data = json.load(f)

            # Convert dictionaries to Conversation objects
            self.conversations = {
                thread_id: Conversation.from_dict(conv_data)
                for thread_id, conv_data in data.items()
            }

            logger.info(f"Loaded {len(self.conversations)} conversation(s) from {file_path}")

            # Set current conversation to the most recent one
            if self.conversations and not self.current_conversation_id:
                conversations = self.list_conversations(sort_by_timestamp=True)
                self.current_conversation_id = conversations[0].thread_id

            return True

        except Exception as e:
            logger.error(f"Failed to load conversations from JSON: {e}")
            return False

    def clear_all(self):
        """Clear all conversations and create a new default one."""
        self.conversations.clear()
        self.current_conversation_id = None
        self.create_conversation("Initial Conversation")
        logger.info("Cleared all conversations and created new default")
