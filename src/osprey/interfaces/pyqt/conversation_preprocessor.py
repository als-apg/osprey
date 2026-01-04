"""
Conversation History Preprocessor for GUI

Preprocesses user messages to resolve conversation history references before
sending to the framework. This allows the GUI to leverage its conversation
history to resolve temporal references like "earlier" or "when I asked before".
"""

import re
from datetime import datetime
from typing import Optional

from osprey.utils.logger import get_logger

logger = get_logger("conversation_preprocessor")


class ConversationPreprocessor:
    """Preprocesses user messages to resolve conversation history references."""

    # Patterns that indicate reference to conversation history
    HISTORY_PATTERNS = [
        r"\b(earlier|before|previously|last time)\b",
        r"\bwhen\s+i\s+asked\b",
        r"\bmy\s+(previous|last)\s+question\b",
    ]

    @classmethod
    def should_preprocess(cls, message: str) -> bool:
        """Check if message contains conversation history references.
        
        Args:
            message: User message to check
            
        Returns:
            True if message should be preprocessed
        """
        message_lower = message.lower()
        return any(re.search(pattern, message_lower) for pattern in cls.HISTORY_PATTERNS)

    @classmethod
    def preprocess_message(
        cls, message: str, conversation_manager, current_conversation_id: str
    ) -> tuple[str, Optional[str]]:
        """Preprocess message to resolve conversation history references.
        
        Args:
            message: Original user message
            conversation_manager: ConversationManager instance with conversation history
            current_conversation_id: ID of current conversation
            
        Returns:
            Tuple of (preprocessed_message, explanation)
            - preprocessed_message: Message with history references resolved
            - explanation: Human-readable explanation of what was resolved (or None)
        """
        if not cls.should_preprocess(message):
            return message, None

        # Get conversation history
        conversation = conversation_manager.get_conversation(current_conversation_id)
        if not conversation or not conversation.messages:
            return message, None

        # Find the most recent user message (excluding the current one)
        previous_user_messages = [
            msg for msg in conversation.messages if msg.get("role") == "user"
        ]
        
        if len(previous_user_messages) < 1:
            # No previous messages to reference
            return message, None

        # Get the most recent previous user message
        previous_message = previous_user_messages[-1]
        previous_content = previous_message.get("content", "")
        previous_timestamp = previous_message.get("timestamp")

        if not previous_timestamp:
            # No timestamp available
            return message, None

        # Parse timestamp
        try:
            if isinstance(previous_timestamp, str):
                timestamp_dt = datetime.fromisoformat(previous_timestamp.replace("Z", "+00:00"))
            else:
                timestamp_dt = previous_timestamp
            
            timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.warning(f"Failed to parse timestamp: {e}")
            return message, None

        # Create enhanced message with explicit timestamp
        enhanced_message = cls._enhance_message_with_timestamp(
            message, previous_content, timestamp_str
        )

        # Create explanation
        explanation = (
            f"📝 Resolved conversation reference:\n"
            f"   Previous question: \"{previous_content[:100]}{'...' if len(previous_content) > 100 else ''}\"\n"
            f"   Asked at: {timestamp_str}"
        )

        return enhanced_message, explanation

    @classmethod
    def _enhance_message_with_timestamp(
        cls, message: str, previous_content: str, timestamp_str: str
    ) -> str:
        """Enhance message by adding explicit timestamp information.
        
        Args:
            message: Original message
            previous_content: Content of previous message
            timestamp_str: Timestamp string in format "YYYY-MM-DD HH:MM:SS"
            
        Returns:
            Enhanced message with explicit timestamp
        """
        # Check what kind of reference this is
        message_lower = message.lower()
        
        # Pattern 1: "when I asked earlier" or similar
        if re.search(r"when\s+i\s+asked", message_lower):
            # Replace the temporal reference with explicit timestamp
            enhanced = re.sub(
                r"when\s+i\s+asked\s+(earlier|before|this\s+question\s+earlier)",
                f"at {timestamp_str} when I asked",
                message,
                flags=re.IGNORECASE
            )
            return enhanced
        
        # Pattern 2: "earlier" or "before" without "when I asked"
        if re.search(r"\b(earlier|before)\b", message_lower):
            # Add timestamp context
            return f"{message} (referring to {timestamp_str})"
        
        # Default: append timestamp information
        return f"{message} (previous question was at {timestamp_str})"