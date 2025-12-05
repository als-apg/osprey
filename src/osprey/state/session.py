"""Framework State - Session and Context Management.

This module provides session context management for the Osprey Agent Framework,
handling user identification, session metadata, and configuration overrides. The
session system maintains context across conversation turns and provides integration
points for user-specific customization.

**Core Components:**

- :class:`SessionContext`: Type-safe session context with user and configuration data

**Session Architecture:**

The session context system provides a lightweight, type-safe structure for managing
session-specific information including user identification, chat context, and
user-specific configuration overrides. The system integrates with the main
AgentState to provide persistent session context.

**Key Features:**

1. **Type Safety**: Comprehensive TypedDict definitions with proper type annotations
2. **LangGraph Compatibility**: Supports partial updates for LangGraph state management
3. **User Context**: User identification and session tracking
4. **Configuration Overrides**: User-specific valve configurations and preferences
5. **Optional Fields**: All fields optional to support flexible session management

**Integration Points:**

The session context integrates with:
- User authentication and identification systems
- Configuration override mechanisms (user valves)
- Chat and conversation tracking systems
- Session persistence and restoration

.. note::
   All session context fields are optional to support partial updates and
   flexible session management patterns in LangGraph state systems.

.. seealso::
   :class:`osprey.state.AgentState` : Main state structure using session context
   :mod:`osprey.infrastructure.gateway` : Gateway managing session context
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class SessionContext(TypedDict, total=False):
    """Type-safe session context for user identification and configuration management.

    This TypedDict class provides a comprehensive structure for managing session-specific
    information including user identification, chat context, session metadata, and
    user-specific configuration overrides. The session context enables personalized
    agent behavior and maintains continuity across conversation sessions.

    **Session Management:**

    The SessionContext supports various session management patterns:

    - **User Identification**: Track users across sessions for personalization
    - **Chat Context**: Maintain chat-specific state and preferences
    - **Session Persistence**: Enable session restoration and continuity
    - **Configuration Overrides**: Support user-specific valve configurations

    **Integration with Agent State:**

    SessionContext integrates with the main AgentState to provide session-aware
    agent behavior. The context is preserved across conversation turns and enables
    personalized responses based on user preferences and history.

    **LangGraph Compatibility:**

    All fields are optional (total=False) to support LangGraph's partial state
    update patterns. This enables efficient session context updates without
    requiring complete context reconstruction.

    **Default Values:**

    When fields are not provided, the following defaults apply:

    - user_id: None (anonymous session)
    - chat_id: None (no chat-specific context)
    - session_id: None (no session persistence)
    - session_url: None (no URL context)
    - user_valves: None (no user-specific overrides)

    .. note::
       The SessionContext is designed to be lightweight and optional. Agents can
       operate without session context, but user-specific features require proper
       user identification through the user_id field.

    .. warning::
       User identification data should be handled according to privacy policies
       and security requirements. Avoid storing sensitive user information in
       session context without proper security measures.

    Examples:
        Basic session context::

            >>> context = SessionContext(
            ...     user_id="user123",
            ...     chat_id="chat456"
            ... )
            >>> context['user_id']
            'user123'

        Session with user valve overrides::

            >>> context = SessionContext(
            ...     user_id="user123",
            ...     user_valves={
            ...         "planning_mode": True,
            ...         "epics_writes": False
            ...     }
            ... )

        Anonymous session::

            >>> context = SessionContext()
            >>> # All fields None by default

    .. seealso::
       :class:`osprey.state.AgentState` : Main state using session context
       :class:`osprey.state.AgentControlState` : Control state affected by user valves
    """

    user_id: str | None  # Unique identifier for the user
    chat_id: str | None  # Unique identifier for the chat session
    session_id: str | None  # Unique identifier for the session
    session_url: str | None  # URL associated with the session
    user_valves: Any | None  # User-specific configuration overrides
