"""
Conversation History Preprocessor for GUI

Preprocesses user messages to resolve conversation history references before
sending to the framework. This allows the GUI to leverage its conversation
history to answer questions about previous interactions directly, without
needing to query the framework.
"""

import re
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
    ) -> tuple[str, Optional[str], Optional[str]]:
        """Preprocess message to resolve conversation history references.
        
        This method checks if the user is asking about a previous conversation.
        If so, it retrieves the previous question and answer from history and
        returns them directly, bypassing the framework entirely.
        
        Args:
            message: Original user message
            conversation_manager: ConversationManager instance with conversation history
            current_conversation_id: ID of current conversation
            
        Returns:
            Tuple of (preprocessed_message, explanation, direct_answer)
            - preprocessed_message: Message with history references resolved (or original)
            - explanation: Human-readable explanation of what was resolved (or None)
            - direct_answer: Direct answer from conversation history (or None if framework should handle)
        """
        if not cls.should_preprocess(message):
            return message, None, None

        # Get conversation history
        conversation = conversation_manager.get_conversation(current_conversation_id)
        if not conversation or not conversation.messages:
            return message, None, None

        # Find the most recent user-assistant exchange
        messages = conversation.messages
        if len(messages) < 2:
            return message, None, None

        # Look for the most recent user message and its assistant response
        previous_user_msg = None
        previous_assistant_msg = None
        
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.type == "user" and previous_user_msg is None:
                previous_user_msg = msg
            elif msg.type == "assistant" and previous_user_msg is not None and previous_assistant_msg is None:
                previous_assistant_msg = msg
                break

        if not previous_user_msg or not previous_assistant_msg:
            return message, None, None

        # Check if the current message is asking about the previous exchange
        # Patterns like "what was X when I asked earlier" or "do you know what X was when I asked"
        message_lower = message.lower()
        is_asking_about_previous = any([
            "when i asked" in message_lower and ("earlier" in message_lower or "before" in message_lower or "question" in message_lower),
            "what was" in message_lower and "when" in message_lower,
            "do you know what" in message_lower and "when" in message_lower,
        ])

        if is_asking_about_previous:
            # User is asking about a previous conversation - provide the answer directly
            logger.info(f"Detected question about previous conversation. Providing direct answer from history.")
            
            explanation = (
                f"📝 Resolved conversation reference:\n"
                f"   Previous question: \"{previous_user_msg.content[:100]}{'...' if len(previous_user_msg.content) > 100 else ''}\"\n"
                f"   Providing previous answer directly from conversation history"
            )
            
            # Format the direct answer
            direct_answer = (
                f"Based on our previous conversation:\n\n"
                f"**Your question:** {previous_user_msg.content}\n\n"
                f"**My answer:** {previous_assistant_msg.content}"
            )
            
            return message, explanation, direct_answer

        # Not asking about previous conversation - let framework handle it
        return message, None, None