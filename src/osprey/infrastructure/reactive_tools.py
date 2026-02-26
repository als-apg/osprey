"""Reactive Orchestrator Tool Utilities.

Converts LangChain tools and registered capabilities into OpenAI function-calling
format for use by the reactive orchestrator. Provides two tiers:

- **Lightweight tools**: Read-only inspection tools that execute inline within
  the orchestrator's LLM loop (no graph cycle needed).
- **Capability tools**: Registered capabilities converted to tool definitions
  that trigger exit to the router for execution.

.. seealso::
   :mod:`osprey.capabilities.context_tools` : Context inspection tools
   :mod:`osprey.capabilities.state_tools` : State inspection tools
   :class:`osprey.infrastructure.reactive_orchestrator_node.ReactiveOrchestratorNode`
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.state import AgentState

logger = logging.getLogger(__name__)

# Read-only tools safe for inline execution by the orchestrator
_LIGHTWEIGHT_TOOL_NAMES = frozenset(
    {
        "read_context",
        "list_available_context",
        "get_context_summary",
        "get_session_info",
        "get_execution_status",
        "list_system_capabilities",
    }
)


def langchain_tool_to_openai_schema(tool) -> dict:
    """Convert a LangChain ``@tool`` to OpenAI function-calling format.

    :param tool: LangChain tool with ``.name``, ``.description``, ``.args_schema``
    :return: OpenAI-format tool definition
    """
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        schema = args_schema.model_json_schema()
        parameters["properties"] = schema.get("properties", {})
        required = schema.get("required", [])
        if required:
            parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": parameters,
        },
    }


def build_lightweight_tools(
    state: AgentState,
) -> tuple[list[dict], dict]:
    """Create read-only inspection tools from context_tools and state_tools.

    Filters to only the 6 read-only tools safe for inline orchestrator execution.

    :param state: Current agent state
    :return: ``(openai_tool_defs, name_to_langchain_tool_map)``
    """
    from osprey.capabilities.context_tools import create_context_tools
    from osprey.capabilities.state_tools import create_state_tools

    all_tools = create_context_tools(state, "orchestrator") + create_state_tools(state)

    tool_defs = []
    tool_map = {}

    for t in all_tools:
        if t.name in _LIGHTWEIGHT_TOOL_NAMES:
            tool_defs.append(langchain_tool_to_openai_schema(t))
            tool_map[t.name] = t

    return tool_defs, tool_map


def build_capability_tools(active_capabilities: list) -> list[dict]:
    """Convert registered capabilities to OpenAI tool definitions.

    Each capability becomes a tool with ``task_objective``, ``context_key``,
    ``expected_output``, and ``success_criteria`` parameters.

    :param active_capabilities: List of capability instances from the registry
    :return: List of OpenAI-format tool definitions
    """
    # Build a map: context_type -> providing capability name
    provider_map: dict[str, str] = {}
    for cap in active_capabilities:
        for provided_type in getattr(cap, "provides", []) or []:
            provider_map[provided_type] = getattr(cap, "name", "unknown")

    tools = []
    for cap in active_capabilities:
        name = getattr(cap, "name", "unknown")
        description = getattr(cap, "description", "")

        # Enrich description with dependency info
        requires = getattr(cap, "requires", []) or []
        provides = getattr(cap, "provides", []) or []
        dep_parts: list[str] = []
        if requires:
            req_items = []
            for req in requires:
                ctx_type = req[0] if isinstance(req, tuple) else req
                provider = provider_map.get(ctx_type)
                if provider:
                    req_items.append(f"{ctx_type} (from {provider})")
                else:
                    req_items.append(ctx_type)
            dep_parts.append(f"Requires: {', '.join(req_items)}.")
        if provides:
            dep_parts.append(f"Provides: {', '.join(provides)}.")
        if dep_parts:
            description = f"{description} {' '.join(dep_parts)}"

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_objective": {
                                "type": "string",
                                "description": "What this step should accomplish",
                            },
                            "context_key": {
                                "type": "string",
                                "description": "Unique identifier for step output in context",
                            },
                            "expected_output": {
                                "type": "string",
                                "description": "Expected output context type",
                            },
                            "success_criteria": {
                                "type": "string",
                                "description": "How to determine if step succeeded",
                            },
                        },
                        "required": ["task_objective", "context_key"],
                    },
                },
            }
        )

    return tools


def execute_lightweight_tool(
    name: str,
    arguments: str | dict,
    tool_map: dict,
) -> str:
    """Execute a lightweight tool inline and return its result.

    :param name: Tool function name
    :param arguments: JSON string or dict of arguments
    :param tool_map: Map of tool name -> LangChain tool
    :return: Tool result as string
    """
    tool = tool_map.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown lightweight tool: {name}"})

    try:
        if isinstance(arguments, str):
            args = json.loads(arguments) if arguments else {}
        else:
            args = arguments

        result = tool.invoke(args)
        return str(result) if result is not None else ""
    except Exception as e:
        logger.warning(f"Lightweight tool '{name}' raised: {e}")
        return f"Error executing {name}: {e}"
