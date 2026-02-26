"""
Context Management Tools for ReAct Agents

This module provides LangChain tools that ReAct agents can use to interact with
the agent's accumulated context data during direct chat mode. These tools enable
agents to read and save context, supporting multi-turn conversations with
context awareness.

The tools are designed to be added to ReAct agents when they execute in either
direct chat mode or orchestrated mode, providing consistent access to accumulated
context data from previous operations.

Example usage::

    from osprey.capabilities.context_tools import create_context_tools

    # In capability execute method:
    context_tools = create_context_tools(state, "my_capability")
    all_tools = list(domain_tools) + context_tools
    agent = create_react_agent(llm, all_tools)
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from osprey.context import ContextManager
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.state import AgentState

logger = get_logger("context_tools")


def create_context_tools(state: AgentState, capability_name: str) -> list:
    """Create context management tools for a ReAct agent.

    These tools allow the ReAct agent to:
    - Read accumulated context from previous operations
    - List available context
    - Save results to context when explicitly requested by the user

    The tools are closures that capture the current state and capability name,
    allowing them to access and modify context data during agent execution.

    :param state: Current agent state containing context data
    :type state: AgentState
    :param capability_name: Name of the capability (for context key generation)
    :type capability_name: str
    :return: List of LangChain Tool instances for context management
    :rtype: list

    Example::

        >>> from osprey.capabilities.context_tools import create_context_tools
        >>> tools = create_context_tools(state, "weather")
        >>> len(tools)
        6  # read, list, save, remove, clear_type, summary

    .. seealso::
       :class:`osprey.context.ContextManager` : Context data management
       :class:`osprey.state.AgentState` : Agent state structure
    """

    @tool
    def read_context(context_type: str, context_key: str | None = None) -> str:
        """Read accumulated context data from the agent state.

        Use this tool when the user asks to use or reference data from previous
        operations. This enables multi-turn conversations where the agent can
        access results from earlier in the session.

        Args:
            context_type: Type of context to read (e.g., "PV_ADDRESSES", "WEATHER_RESULTS")
            context_key: Optional specific key. If not provided, returns all keys of that type

        Returns:
            Structured summary of context data with access guidance

        Examples:
            - read_context("PV_ADDRESSES", "beam_current")
            - read_context("WEATHER_RESULTS")  # Lists all weather results
        """
        try:
            context_mgr = ContextManager(state)

            if context_key:
                # Get specific context
                context_obj = context_mgr.get_context(context_type, context_key)
                if context_obj:
                    # Use the context's built-in summary method if available
                    if hasattr(context_obj, "get_summary"):
                        summary = context_obj.get_summary()
                    else:
                        # Fallback to model_dump for Pydantic models
                        summary = (
                            context_obj.model_dump()
                            if hasattr(context_obj, "model_dump")
                            else str(context_obj)
                        )

                    # Format for display
                    result = f"Context {context_type}.{context_key}:\n\n"
                    result += "📊 Summary:\n"
                    if isinstance(summary, dict):
                        result += json.dumps(summary, indent=2, default=str)
                    else:
                        result += str(summary)

                    return result
                else:
                    return f"Context {context_type}.{context_key} not found"
            else:
                # List all keys of this type
                all_of_type = context_mgr.get_all_of_type(context_type)
                if all_of_type:
                    keys = list(all_of_type.keys())
                    return (
                        f"Available {context_type} keys: {', '.join(keys)}\n"
                        f"Use read_context('{context_type}', '<key>') to read specific data"
                    )
                else:
                    return f"No context data of type {context_type} found"
        except Exception as e:
            logger.error(f"Error reading context: {e}")
            return f"Error reading context: {str(e)}"

    @tool
    def list_available_context() -> str:
        """List all available context types and keys in the agent state.

        Use this when the user asks "what data do we have?" or similar questions
        about accumulated context from previous operations.

        Returns:
            Summary of available context organized by type
        """
        try:
            context_mgr = ContextManager(state)
            context_data = state.get("capability_context_data", {})

            if not context_data:
                return "No context data available yet"

            summary = "Available Context:\n\n"
            for context_type, keys_dict in context_data.items():
                summary += f"📦 {context_type}:\n"
                for key in keys_dict.keys():
                    # Try to get context object for description
                    ctx_obj = context_mgr.get_context(context_type, key)
                    if ctx_obj and hasattr(ctx_obj, "get_summary"):
                        try:
                            ctx_summary = ctx_obj.get_summary()
                            desc = (
                                ctx_summary.get("description", "N/A")
                                if isinstance(ctx_summary, dict)
                                else "N/A"
                            )
                            summary += f"  - {key}: {desc}\n"
                        except Exception:
                            summary += f"  - {key}\n"
                    else:
                        summary += f"  - {key}\n"
                summary += "\n"

            return summary
        except Exception as e:
            logger.error(f"Error listing context: {e}")
            return f"Error listing context: {str(e)}"

    @tool
    def save_result_to_context(context_key: str, description: str | None = None) -> str:
        """Save the most recent result from this conversation to the agent's context.

        WHEN TO USE THIS TOOL:
        Call this tool when the user asks to save/store/remember something using phrases like:
        - "save that" / "save it" / "save this"
        - "remember that" / "store this"
        - "save that as X" (explicit key provided)

        DO NOT call this tool automatically or preemptively - only when the user asks.

        CHOOSING THE CONTEXT KEY:
        - If user provides explicit name ("save as tokyo_weather") → use that name
        - If user just says "save it" without a name → INFER a descriptive key from context

        When inferring a key, create a short, descriptive snake_case name based on:
        - The data being saved (e.g., weather data → "hamburg_weather")
        - The conversation context (e.g., "current weather in Hamburg" → "hamburg_current_weather")
        - Keep it concise but meaningful (2-4 words max)

        Args:
            context_key: Name to save under - infer from context if user didn't specify
            description: Optional human-readable description of what was saved

        Returns:
            Confirmation message that data was saved

        Examples:
            User: "Save that as tokyo_weather" → context_key="tokyo_weather"
            User: "Save it" (after Hamburg weather query) → context_key="hamburg_weather"
            User: "Remember this" (after beam current lookup) → context_key="beam_current"
        """
        try:
            from osprey.registry import get_registry
            from osprey.state import StateManager

            # Get the last result from session_state
            session_state = state.get("session_state", {})
            last_result = session_state.get("last_direct_chat_result")

            if not last_result:
                return (
                    "❌ No recent result to save. "
                    "Please generate a result first, then ask me to save it."
                )

            # Extract details from last result
            result_capability = last_result.get("capability", capability_name)
            context_type = last_result.get("context_type")

            if not context_type:
                return "❌ Cannot save: no context type defined for this capability's results"

            # Create context with namespace
            context_key_full = f"chat:{context_key}"

            # Get the appropriate context class from registry
            registry = get_registry()
            context_class = registry.get_context_class(context_type)

            if not context_class:
                # Try to use a generic context storage approach
                logger.warning(f"Context type {context_type} not registered, using raw storage")
                # Store as raw dict in capability_context_data
                context_data = state.get("capability_context_data", {})
                if context_type not in context_data:
                    context_data[context_type] = {}

                context_data[context_type][context_key_full] = {
                    "tool": "react_agent",
                    "results": last_result.get("full_response"),
                    "description": description or f"Saved from direct chat: {context_key}",
                    "origin": "direct_chat",
                    "capability": result_capability,
                    "timestamp": last_result.get("timestamp", time.time()),
                }

                state["capability_context_data"] = context_data
                logger.info(f"Saved direct chat result as {context_key_full} (raw)")
                return f"✓ Saved to context as {context_key_full}"

            # Create context object using registered class
            try:
                context = context_class(
                    tool="react_agent",
                    results=last_result.get("full_response"),
                    description=description or f"Saved from direct chat: {context_key}",
                    origin="direct_chat",
                    capability=result_capability,
                    timestamp=last_result.get("timestamp"),
                )

                # Store context using StateManager
                context_updates = StateManager.store_context(
                    state, context_type, context_key_full, context
                )

                # Update state
                state["capability_context_data"] = context_updates.get(
                    "capability_context_data", state.get("capability_context_data", {})
                )

                logger.info(f"Saved direct chat result as {context_key_full}")
                return f"✓ Saved to context as {context_key_full}"

            except Exception as e:
                logger.error(f"Error creating context object: {e}")
                return f"❌ Error saving result: {str(e)}"

        except Exception as e:
            logger.error(f"Error saving result: {e}")
            return f"❌ Error saving result: {str(e)}"

    @tool
    def remove_context(context_type: str, context_key: str) -> str:
        """Remove specific context data from the agent state.

        Only use this when the user explicitly asks to delete or remove context.

        Args:
            context_type: Type of context (e.g., "PV_ADDRESSES", "WEATHER_RESULTS")
            context_key: Specific key to remove

        Returns:
            Confirmation of what was deleted

        Examples:
            User: "Delete the old weather data" → remove_context("WEATHER_RESULTS", "old_key")
        """
        try:
            context_data = state.get("capability_context_data", {})

            if context_type not in context_data:
                return f"❌ Context type '{context_type}' not found"

            if context_key not in context_data[context_type]:
                return f"❌ Context key '{context_key}' not found in {context_type}"

            # Delete the context
            del context_data[context_type][context_key]

            # If type is now empty, remove it
            if not context_data[context_type]:
                del context_data[context_type]

            logger.info(f"Removed context: {context_type}.{context_key}")
            return f"✓ Removed {context_type}.{context_key} from context"

        except Exception as e:
            logger.error(f"Error removing context: {e}")
            return f"❌ Error removing context: {str(e)}"

    @tool
    def clear_context_type(context_type: str) -> str:
        """Remove ALL context data of a specific type.

        Use only when user explicitly wants to clear all data of one type.

        Args:
            context_type: Type of context to clear (e.g., "WEATHER_RESULTS")

        Returns:
            Confirmation of what was deleted
        """
        try:
            context_data = state.get("capability_context_data", {})

            if context_type not in context_data:
                return f"❌ No context data of type '{context_type}'"

            count = len(context_data[context_type])
            keys = list(context_data[context_type].keys())

            # Delete the context type
            del context_data[context_type]

            logger.info(f"Cleared context type: {context_type} ({count} items)")
            return f"✓ Removed all {count} entries of type {context_type}: {', '.join(keys[:5])}{'...' if len(keys) > 5 else ''}"

        except Exception as e:
            logger.error(f"Error clearing context: {e}")
            return f"❌ Error clearing context: {str(e)}"

    @tool
    def get_context_summary() -> str:
        """Get a high-level summary of accumulated context.

        Use this for a quick overview of how much context data has accumulated
        across all types. Good for answering "how much data do we have?"

        Returns:
            Summary with counts by type and total items
        """
        context_data = state.get("capability_context_data", {})

        if not context_data:
            return "No context data accumulated"

        summary = "Context Summary:\n\n"
        total_items = 0

        for context_type, keys_dict in context_data.items():
            count = len(keys_dict)
            total_items += count
            summary += f"📦 {context_type}: {count} items\n"

        summary += f"\nTotal: {total_items} context items across {len(context_data)} types"
        return summary

    return [
        read_context,
        list_available_context,
        save_result_to_context,
        remove_context,
        clear_context_type,
        get_context_summary,
    ]
