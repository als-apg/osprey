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

        messages = conversation.messages
        if len(messages) < 2:
            return message, None, None

        # Check if the current message is asking about a previous exchange
        # Patterns like "what was X when I asked earlier" or "do you know what X was when I asked"
        message_lower = message.lower()
        is_asking_about_previous = any([
            "when i asked" in message_lower and ("earlier" in message_lower or "before" in message_lower or "question" in message_lower or "this" in message_lower),
            "what was" in message_lower and ("when" in message_lower or "earlier" in message_lower),
            "do you know what" in message_lower and ("when" in message_lower or "earlier" in message_lower),
            "when did i ask" in message_lower,
            re.search(r"(earlier|before|previously)\s+(question|query|message)", message_lower),
        ])

        if not is_asking_about_previous:
            # Not asking about previous conversation - let framework handle it
            logger.debug("Message references history but not asking about previous conversation")
            return message, None, None

        # User IS asking about a previous conversation - find the most recent user-assistant exchange
        logger.info(f"Detected question about previous conversation: '{message}'")
        
        # Look back through the last few exchanges to find user-response pairs
        # Response can be from 'agent', 'assistant', or 'project'
        user_response_pairs = []
        response_types = {"agent", "assistant", "project"}
        
        for i in range(len(messages) - 1):
            if messages[i].type == "user" and i + 1 < len(messages) and messages[i + 1].type in response_types:
                user_response_pairs.append((messages[i], messages[i + 1]))
        
        if not user_response_pairs:
            logger.debug("No previous user-response exchanges found")
            return message, None, None
        
        # Use the most recent exchange (last in the list)
        previous_user_msg, previous_response_msg = user_response_pairs[-1]
        
        logger.info(f"Found previous exchange - User: '{previous_user_msg.content[:50]}...', Response: '{previous_response_msg.content[:50]}...'")
        
        explanation = (
            f"📝 Resolved conversation reference:\n"
            f"   Previous question: \"{previous_user_msg.content[:100]}{'...' if len(previous_user_msg.content) > 100 else ''}\"\n"
            f"   Providing previous answer directly from conversation history"
        )
        
        # Format the direct answer
        direct_answer = (
            f"Based on our previous conversation:\n\n"
            f"**Your question:** {previous_user_msg.content}\n\n"
            f"**My answer:** {previous_response_msg.content}"
        )
        
        return message, explanation, direct_answer