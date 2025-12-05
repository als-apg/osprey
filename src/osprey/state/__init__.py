"""Framework State Management - LangGraph Native Implementation.

This module provides comprehensive state management for the Osprey Agent Framework
using LangGraph's native patterns for conversation state, execution tracking, and
context persistence. The state system is designed for optimal performance with
LangGraph's checkpointing and serialization mechanisms.

The state management system follows a clear separation of concerns:

**Core State Components:**
- :class:`AgentState`: Main conversational state with LangGraph-native message handling
- :class:`StateManager`: Utilities for state creation and management
- :class:`StateUpdate`: Type alias for LangGraph state update dictionaries

**Execution and Control:**
- :class:`ApprovalRequest`: Approval workflow management for sensitive operations
- :class:`AgentControlState`: Runtime configuration and execution control
- Control flow utilities for slash commands and agent behavior

**Message and Session Management:**
- :class:`MessageUtils`: LangGraph-native message creation and manipulation
- :class:`ChatHistoryFormatter`: Conversation formatting for LLM consumption
- :class:`UserMemories`: Persistent user context across conversations
- :class:`SessionContext`: Session-specific metadata and configuration

**Key Design Principles:**

1. **LangGraph Native**: Built on MessagesState with automatic message handling
2. **Selective Persistence**: Only capability_context_data persists across conversations
3. **Execution Scoped**: All other fields reset automatically between graph invocations
4. **Type Safety**: Comprehensive TypedDict definitions with proper type hints
5. **Serialization Ready**: Pure dictionary structures compatible with checkpointing

**State Lifecycle:**

The state system operates on a conversation-turn basis where:
- Fresh state is created for each new conversation turn
- Only capability context data accumulates across turns
- All execution-scoped fields reset to defaults
- Message history is managed automatically by LangGraph

**Usage Patterns:**

Basic state creation::

    from osprey.state import StateManager

    # Create fresh state for new conversation
    state = StateManager.create_fresh_state(
        user_input="Find beam current PV addresses",
        current_state=previous_state  # Optional, preserves context
    )

Context storage in capabilities::

    from osprey.state import StateManager

    # Store capability results
    return StateManager.store_context(
        state, "PV_ADDRESSES", context_key, pv_data
    )

Message handling::

    from osprey.state import MessageUtils

    # Create properly formatted messages
    user_msg = MessageUtils.create_user_message(user_input)
    response_msg = MessageUtils.create_assistant_message(response)

.. note::
   The state system is optimized for LangGraph's native patterns and should be
   used through the provided utilities rather than direct state manipulation.

.. warning::
   Direct modification of state dictionaries may interfere with LangGraph's
   checkpointing and serialization. Always use StateManager utilities for
   state operations.

.. seealso::
   :mod:`osprey.context` : Context management and capability data storage
   :mod:`osprey.base.planning` : Execution planning and step management
   :mod:`osprey.infrastructure.gateway` : Main entry point for state processing
"""

from .control import AgentControlState, apply_slash_commands_to_agent_control_state
from .execution import ApprovalRequest  # Keep as dataclass
from .messages import ChatHistoryFormatter, MessageUtils, UserMemories
from .session import SessionContext  # Keep as simple utility
from .state import (
    AgentState,
    StateUpdate,
    create_progress_event,
    create_status_update,
    merge_capability_context_data,
)
from .state_manager import StateManager, get_execution_steps_summary

# Export everything for convenient importing
__all__ = [
    # Core state (LangGraph native)
    "AgentState",
    "StateUpdate",
    "StateManager",
    # Utility functions
    "create_status_update",
    "create_progress_event",
    "get_execution_steps_summary",
    "merge_capability_context_data",
    # Session management (simplified)
    "SessionContext",
    # Execution components (keep as dataclasses)
    "ApprovalRequest",
    # Message utilities
    "MessageUtils",
    "ChatHistoryFormatter",
    "UserMemories",
    # Control state (simplified)
    "AgentControlState",
    "apply_slash_commands_to_agent_control_state",
]
