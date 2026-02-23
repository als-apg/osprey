"""MCP Capability Generator for Osprey Framework.

Generates complete, working Osprey capabilities from MCP servers.
Everything in one file: capability class, guides, context class, error handling.
"""

import json
from typing import Any

from .base_generator import BaseCapabilityGenerator
from .models import ClassifierAnalysis, OrchestratorAnalysis

# Try MCP client (optional - can work in simulated mode)
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


# =============================================================================
# Simulated Tools (for testing without MCP server)
# =============================================================================

SIMULATED_TOOLS = [
    {
        "name": "get_current_weather",
        "description": "Get current weather conditions for a location. If location is not provided, defaults to San Francisco.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "default": "San Francisco",
                    "description": "City name or coordinates (defaults to San Francisco if not provided)",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                    "description": "Temperature units",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_forecast",
        "description": "Get weather forecast for upcoming days. If location is not provided, defaults to San Francisco.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "default": "San Francisco",
                    "description": "City name or coordinates (defaults to San Francisco if not provided)",
                },
                "days": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 7,
                    "description": "Number of forecast days (1-7)",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                    "description": "Temperature units",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_weather_alerts",
        "description": "Get active weather alerts and warnings for a location. If location is not provided, defaults to San Francisco.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "default": "San Francisco",
                    "description": "City name or coordinates (defaults to San Francisco if not provided)",
                },
                "severity": {
                    "type": "string",
                    "enum": ["all", "severe", "moderate", "minor"],
                    "default": "all",
                    "description": "Filter by alert severity level",
                },
            },
            "required": [],
        },
    },
]


# =============================================================================
# MCP Capability Generator
# =============================================================================


class MCPCapabilityGenerator(BaseCapabilityGenerator):
    """Generate complete MCP capability from MCP server tools."""

    def __init__(
        self,
        capability_name: str,
        server_name: str,
        verbose: bool = False,
        provider: str | None = None,
        model_id: str | None = None,
    ):
        """Initialize generator.

        Args:
            capability_name: Name for the generated capability (e.g., 'slack_mcp')
            server_name: Human-readable server name (e.g., 'Slack')
            verbose: Whether to print progress messages
            provider: Optional LLM provider override
            model_id: Optional model ID override
        """
        super().__init__(capability_name, verbose, provider, model_id)
        self.server_name = server_name
        self.tools: list[dict[str, Any]] = []
        self.mcp_url: str | None = None

    async def discover_tools(
        self, mcp_url: str | None = None, simulated: bool = False
    ) -> list[dict[str, Any]]:
        """Discover tools from MCP server or use simulated tools.

        Args:
            mcp_url: MCP server URL (e.g., 'http://localhost:3001')
            simulated: If True, use simulated weather tools instead of connecting to server

        Returns:
            List of tool dictionaries with name, description, and inputSchema

        Raises:
            RuntimeError: If MCP client not installed or server unreachable
        """
        if simulated:
            if self.verbose:
                print("Using simulated tools (no MCP server needed)")
            self.tools = SIMULATED_TOOLS
            self.mcp_url = "http://localhost:3001"  # Placeholder
        else:
            if not MCP_AVAILABLE:
                raise RuntimeError(
                    "MCP client not installed. Use simulated mode or install: pip install mcp"
                )

            if self.verbose:
                print(f"Connecting to MCP server: {mcp_url}")

            self.mcp_url = mcp_url

            # FastMCP SSE endpoint is at /sse
            sse_url = mcp_url if mcp_url.endswith("/sse") else f"{mcp_url}/sse"

            # Pre-flight check: Try to connect to the server first
            try:
                import httpx

                # Quick check to see if server is reachable
                async with httpx.AsyncClient(timeout=2.0) as client:
                    try:
                        # Try a simple GET to the base URL
                        _ = await client.get(mcp_url)
                        # We expect this might fail, but at least we know server is responding
                    except httpx.ConnectError as e:
                        # Server is not reachable at all
                        error_msg = (
                            f"\n❌ Cannot connect to MCP server at {mcp_url}\n\n"
                            f"The server is not running or not reachable.\n\n"
                            f"To start the demo MCP server:\n"
                            f"  1. Generate the server: osprey generate mcp-server\n"
                            f"  2. Run it: python demo_mcp_server.py\n\n"
                            f"Or use simulated mode (no server needed):\n"
                            f"  osprey generate capability --from-mcp simulated --name {self.capability_name}\n\n"
                            f"Connection error: {e}"
                        )
                        raise RuntimeError(error_msg) from e
            except ImportError:
                # httpx not available, skip pre-flight check
                pass

            try:
                # Use native MCP client to get tools in standardized format
                async with sse_client(sse_url) as (read, write):
                    async with ClientSession(read, write) as session:
                        # Initialize the session
                        await session.initialize()

                        # List tools using standard MCP protocol
                        tools_result = await session.list_tools()

                        # Convert from MCP's Pydantic models to dicts for JSON serialization
                        self.tools = []
                        for tool in tools_result.tools:
                            tool_dict = {
                                "name": tool.name,
                                "description": tool.description or "",
                                "inputSchema": (
                                    tool.inputSchema if hasattr(tool, "inputSchema") else {}
                                ),
                            }
                            self.tools.append(tool_dict)

                if self.verbose:
                    print(f"✓ Discovered {len(self.tools)} tools")

            except (ConnectionError, ConnectionRefusedError, TimeoutError, OSError) as e:
                # Connection-specific errors - likely server not running
                error_msg = (
                    f"\n❌ Cannot connect to MCP server at {sse_url}\n\n"
                    f"The MCP server appears to be down or not responding.\n\n"
                    f"To start the demo MCP server:\n"
                    f"  1. Generate the server: osprey generate mcp-server\n"
                    f"  2. Run it: python demo_mcp_server.py\n\n"
                    f"Or use simulated mode (no server needed):\n"
                    f"  osprey generate capability --from-mcp simulated --name {self.capability_name}\n\n"
                    f"Error: {type(e).__name__}: {e}"
                )
                raise RuntimeError(error_msg) from e
            except Exception as e:
                # Check if this looks like a connection/TaskGroup error (common when server is down)
                error_str = str(e).lower()
                if any(
                    term in error_str
                    for term in [
                        "taskgroup",
                        "connection",
                        "refused",
                        "timeout",
                        "unreachable",
                        "connecterror",
                        "incomplete chunked read",
                        "peer closed",
                    ]
                ):
                    error_msg = (
                        f"\n❌ Cannot connect to MCP server at {sse_url}\n\n"
                        f"The MCP server appears to be down, crashed during handshake, or is incompatible.\n\n"
                        f"To start the demo MCP server:\n"
                        f"  1. Generate the server: osprey generate mcp-server\n"
                        f"  2. Install FastMCP: pip install fastmcp\n"
                        f"  3. Run it: python demo_mcp_server.py\n\n"
                        f"Or use simulated mode (no server needed):\n"
                        f"  osprey generate capability --from-mcp simulated --name {self.capability_name}\n\n"
                        f"Technical details: {type(e).__name__}"
                    )
                    raise RuntimeError(error_msg) from e
                else:
                    # Some other error during tool discovery
                    error_msg = (
                        f"\n❌ Failed to discover tools from MCP server at {sse_url}\n\n"
                        f"Error: {type(e).__name__}: {e}"
                    )
                    raise RuntimeError(error_msg) from e

        return self.tools

    async def generate_guides(self) -> tuple[ClassifierAnalysis, OrchestratorAnalysis]:
        """Generate classifier and orchestrator guides using LLM.

        Uses the configured orchestrator model (or overrides if specified)
        to analyze the discovered tools and generate activation guides.

        Returns:
            Tuple of (classifier_analysis, orchestrator_analysis)

        Raises:
            RuntimeError: If generation fails after all retry attempts
        """
        if self.verbose:
            print("\n🤖 Analyzing tools with LLM...")

        tools_json = json.dumps(self.tools, indent=2)

        # Generate classifier analysis
        classifier_prompt = f"""You are an expert at analyzing tool capabilities and generating task classification rules.

I have a capability called "{self.capability_name}" that wraps a {self.server_name} MCP server.

Here are ALL the tools this MCP server provides:

{tools_json}

Your task: Analyze these tools and generate a comprehensive classifier guide.

Generate:
- Clear activation criteria
- Key terms/patterns that indicate this capability is needed
- 5-7 realistic positive examples (queries that SHOULD activate) with reasoning
- 3-4 realistic negative examples (queries that SHOULD NOT activate) with reasoning
- Edge cases to watch for

Make the examples natural and varied - think about real users.
Output as JSON matching the ClassifierAnalysis schema.
"""

        classifier_analysis = await self._call_llm(classifier_prompt, ClassifierAnalysis)

        # Generate orchestrator analysis
        orchestrator_prompt = f"""You are an expert at high-level task planning without business logic.

I have a capability called "{self.capability_name}" that wraps a {self.server_name} MCP server.

The capability uses a ReAct agent internally that autonomously selects from these MCP tools:

{tools_json}

Your task: Generate a SIMPLE orchestrator guide for HIGH-LEVEL planning only.

The orchestrator should NOT know about specific MCP tools (the ReAct agent handles that).
The orchestrator should ONLY know:
- When to invoke the {self.capability_name} capability (what types of user requests)
- How to formulate clear task_objective descriptions
- General patterns for {self.server_name} operations

Generate:
- 3-5 example scenarios showing WHAT users might ask for (not HOW to implement with tools)
- Each example should have a clear task_objective that describes the goal, not the implementation
- **IMPORTANT**: Each example MUST have a descriptive context_key that captures the essence of the step
  - Good: "current_weather_sf", "severe_alerts_boston", "forecast_tokyo_5day"
  - Bad: "result_1", "weather_data", "output"
  - The context_key should be specific, descriptive, and use snake_case
- **CRITICAL**: If any tool parameters have default values, include AT LEAST ONE example that demonstrates using those defaults (omitting the parameter from task_objective). This teaches the orchestrator that parameters with defaults are truly optional.
- Important notes about formulating good task objectives for the capability

Do NOT include tool_name in examples - the ReAct agent decides which tools to use.
The tool_name field can be empty or contain a generic placeholder.

Output as JSON matching the OrchestratorAnalysis schema.
"""

        orchestrator_analysis = await self._call_llm(orchestrator_prompt, OrchestratorAnalysis)

        if self.verbose:
            print("✓ Guides generated")

        return classifier_analysis, orchestrator_analysis

    def generate_capability_code(
        self, classifier_analysis: ClassifierAnalysis, orchestrator_analysis: OrchestratorAnalysis
    ) -> str:
        """Generate complete capability Python code.

        Args:
            classifier_analysis: Classifier guide analysis from LLM
            orchestrator_analysis: Orchestrator guide analysis from LLM

        Returns:
            Complete Python source code for the capability
        """
        timestamp = self._get_timestamp()
        class_name = self._to_class_name(self.capability_name)
        context_class_name = self._to_class_name(self.capability_name, suffix="ResultsContext")
        context_type = f"{self.server_name.upper()}_RESULTS"

        # Build examples using base class methods
        classifier_examples_code = self._build_classifier_examples_code(classifier_analysis)
        orchestrator_examples_code = self._build_orchestrator_examples_code(
            orchestrator_analysis, context_type
        )

        # Build tools list for documentation
        tools_list = "\n".join(
            [f"        - {t['name']}: {t.get('description', 'N/A')}" for t in self.tools]
        )

        # Ensure SSE endpoint path is included
        sse_url = self.mcp_url if self.mcp_url.endswith("/sse") else f"{self.mcp_url}/sse"

        code = f'''"""
{self.server_name} MCP Capability

Auto-generated MCP capability for Osprey Framework.
Generated: {timestamp}

MCP Server: {self.server_name}
Server URL: {sse_url}
Tools: {len(self.tools)}

This file contains everything needed to integrate the {self.server_name} MCP server:
- Capability class with ReAct agent execution pattern
- LangGraph ReAct agent with MCP tools integration
- Direct chat mode support for conversational interaction
- Classifier guide (when to activate)
- Orchestrator guide (high-level planning only)
- Context class for results
- Error handling

DIRECT CHAT MODE:
This capability supports direct chat mode, allowing conversational interaction
with the ReAct agent without going through task extraction/orchestration.

Usage:
  /chat:{self.capability_name}    # Enter direct chat mode

In direct chat mode:
  - Messages go directly to the ReAct agent
  - Full conversation history maintained
  - Say "save that as <key>" to store results in context
  - Agent can read accumulated context via tools

ARCHITECTURE:
This capability uses a ReAct agent pattern that follows standard MCP usage:
- The orchestrator simply decides to invoke this capability and provides a task_objective
- The capability's ReAct agent autonomously:
  * Sees all available MCP tools from the server
  * Reasons about which tool(s) to call
  * Executes tools and observes results
  * Continues until the task is complete
- This keeps business logic out of orchestration (proper separation of concerns)

NEXT STEPS:
1. Review the generated code
2. **IMPORTANT: Customize {context_class_name}** - The generated context class is a minimal placeholder.
   Customize it based on your MCP server's actual data structure. See the TODO comments
   and documentation link in the context class section.
3. **Configure ReAct Agent Model** - Add to your config.yml:
   ```yaml
   models:
     {self.capability_name}_react:  # Optional - dedicated model for {self.capability_name} tool execution
       provider: anthropic  # or openai, cborg, amsc, etc.
       model_id: claude-sonnet-4
       max_tokens: 4096
   ```
   If not configured, falls back to using the "orchestrator" model.
   The ReAct agent needs good reasoning for autonomous tool selection.
4. Adjust MCP server URL and transport in MCP_SERVER_CONFIG if needed
5. Install required dependencies:
   - pip install langchain-mcp-adapters langgraph
   - pip install langchain-anthropic  # If using Anthropic provider
   - pip install langchain-openai     # If using OpenAI or CBORG provider
6. Consider moving {context_class_name} to a shared context_classes.py
7. Add to your registry.py (see registration snippet at bottom)
8. Test with real queries

Generated by: osprey generate capability --from-mcp
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, ClassVar, TYPE_CHECKING
import textwrap

if TYPE_CHECKING:
    from osprey.state import AgentState

# Framework imports
from osprey.base.decorators import capability_node
from osprey.base.capability import BaseCapability
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.planning import PlannedStep
from osprey.base.examples import OrchestratorGuide, OrchestratorExample, TaskClassifierGuide, ClassifierExample, ClassifierActions
from osprey.context import CapabilityContext
from osprey.state import StateManager
from osprey.registry import get_registry
from osprey.utils.logger import get_logger

# MCP and LangGraph imports
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent
except ImportError:
    raise ImportError(
        "MCP adapters not installed. Install with: "
        "pip install langchain-mcp-adapters langgraph"
    )

# Get model for ReAct agent
from osprey.utils.config import get_model_config


# =============================================================================
# Context Class - CUSTOMIZE THIS FOR YOUR USE CASE
# =============================================================================
# NOTE: This is a MINIMAL PLACEHOLDER. You should customize this based on your
# actual MCP server's data structure and your application's needs.
#
# 📚 Documentation: For detailed guidance on creating context classes, see:
# https://als-apg.github.io/osprey/developer-guides/03_core-framework-systems/02_context-management-system.html

class {context_class_name}(CapabilityContext):
    """
    Context for {self.server_name} MCP results.

    TODO: Customize this class based on your MCP server's actual data structure.
    """

    CONTEXT_TYPE: ClassVar[str] = "{context_type}"
    CONTEXT_CATEGORY: ClassVar[str] = "EXTERNAL_DATA"

    tool: str
    results: Dict[str, Any]
    description: str

    def get_access_details(self, key: str) -> Dict[str, Any]:
        """Tell the LLM how to access this context data."""
        return {{
            "tool_used": self.tool,
            "description": self.description,
            "data_structure": "Dict[str, Any]",
            "access_pattern": f"context.{{self.CONTEXT_TYPE}}.{{key}}.results",
            "available_fields": list(self.results.keys()) if isinstance(self.results, dict) else "results",
        }}

    def get_summary(self) -> Dict[str, Any]:
        """Format data for human display."""
        return {{
            "type": "{self.server_name} Results",
            "tool": self.tool,
            "description": self.description,
            "results": self.results,
        }}


# =============================================================================
# Error Classes
# =============================================================================

class {class_name}Error(Exception):
    """Base error for {self.server_name} MCP operations."""
    pass


class {self.server_name}ConnectionError({class_name}Error):
    """MCP server connection failed."""
    pass


class {self.server_name}ToolError({class_name}Error):
    """MCP tool execution failed."""
    pass


# =============================================================================
# Capability Implementation
# =============================================================================

@capability_node
class {class_name}(BaseCapability):
    """
    {self.server_name} MCP capability.

    Integrates with {self.server_name} MCP server to provide:
{tools_list}
    """

    name = "{self.capability_name}"
    description = "{self.server_name} operations via MCP server"
    provides = ["{context_type}"]
    requires = []

    # Enable direct chat mode for conversational interaction
    direct_chat_enabled = True

    # MCP server configuration
    MCP_SERVER_URL = "{sse_url}"
    MCP_SERVER_CONFIG = {{
        "{self.capability_name}_server": {{
            "url": "{sse_url}",
            "transport": "sse",
        }}
    }}

    # Class-level client and agent cache
    _mcp_client: Optional[MultiServerMCPClient] = None
    _react_agent = None

    @classmethod
    async def _get_react_agent(cls, state: Optional['AgentState'] = None, include_context_tools: bool = False):
        """Get or create ReAct agent with MCP tools.

        Args:
            state: Agent state (required when include_context_tools=True)
            include_context_tools: If True, adds context management tools to the agent.
                                   When True, agent is NOT cached since tools are state-dependent.

        Returns:
            ReAct agent configured with requested tools
        """
        # When context tools are included, we can't cache (state-dependent)
        # Use cached agent only when no context tools needed
        if not include_context_tools and cls._react_agent is not None:
            return cls._react_agent

        if cls._react_agent is None or include_context_tools:
            # Get logger for initialization (classmethod doesn't have self)
            logger = get_logger("{self.capability_name}")

            try:
                # Initialize MCP client
                if cls._mcp_client is None:
                    cls._mcp_client = MultiServerMCPClient(cls.MCP_SERVER_CONFIG)
                    logger.info(f"Connected to MCP server: {{cls.MCP_SERVER_URL}}")

                # Get tools from MCP server
                mcp_tools = await cls._mcp_client.get_tools()
                logger.info(f"Loaded {{len(mcp_tools)}} tools from MCP server")

                # Optionally add context management tools
                all_tools = list(mcp_tools)
                if include_context_tools and state is not None:
                    from osprey.capabilities.context_tools import create_context_tools
                    context_tools = create_context_tools(state, cls.name)
                    all_tools.extend(context_tools)
                    logger.info(f"Added {{len(context_tools)}} context management tools")

                # Get LLM instance for ReAct agent
                # Try to use dedicated "{self.capability_name}_react" model first, fallback to "orchestrator"
                model_config = get_model_config("{self.capability_name}_react")
                if not model_config or not model_config.get("provider"):
                    # Model config doesn't exist or is incomplete, fallback to orchestrator
                    model_config = get_model_config("orchestrator")

                provider = model_config.get("provider")
                model_id = model_config.get("model_id")

                if not provider:
                    raise ValueError(
                        f"No provider configured for {{self.capability_name}}_react or orchestrator model. "
                        f"Please configure a model in config.yml under models section."
                    )

                # Create LangChain model using osprey's unified factory
                from osprey.models import get_langchain_model
                llm = get_langchain_model(
                    provider=provider,
                    model_id=model_id,
                    max_tokens=model_config.get("max_tokens", 4096)
                )

                # Create ReAct agent with all tools
                agent = create_react_agent(llm, all_tools)
                logger.info(f"ReAct agent initialized with {{len(all_tools)}} total tools")

                # Only cache if no context tools (context tools are state-dependent)
                if not include_context_tools:
                    cls._react_agent = agent

                return agent

            except (ConnectionError, ConnectionRefusedError, TimeoutError, OSError) as e:
                # Connection-specific errors - likely server not running
                error_msg = (
                    f"{self.server_name} MCP server is not reachable at {{cls.MCP_SERVER_URL}}. "
                    f"Please ensure the MCP server is running. Error: {{type(e).__name__}}: {{e}}"
                )
                logger.error(error_msg)
                raise {self.server_name}ConnectionError(error_msg) from e
            except Exception as e:
                # Check if this looks like a connection/TaskGroup error (common when server is down)
                error_str = str(e).lower()
                if any(term in error_str for term in ["taskgroup", "connection", "refused", "timeout", "unreachable"]):
                    error_msg = (
                        f"{self.server_name} MCP server appears to be down or unreachable at {{cls.MCP_SERVER_URL}}. "
                        f"Please start the MCP server before using this capability. "
                        f"Technical details: {{type(e).__name__}}: {{e}}"
                    )
                    logger.error(error_msg)
                    raise {self.server_name}ConnectionError(error_msg) from e
                else:
                    # Some other initialization error
                    error_msg = f"Failed to initialize {self.server_name} ReAct agent: {{type(e).__name__}}: {{e}}"
                    logger.error(error_msg)
                    raise {self.server_name}ConnectionError(error_msg) from e

        return cls._react_agent

    async def execute(self) -> Dict[str, Any]:
        """Execute {self.server_name} MCP capability in orchestrated OR direct chat mode."""
        import time as time_module
        from osprey.state import MessageUtils

        # Get unified logger with automatic streaming support
        logger = self.get_logger()
        state = self._state

        # Detect execution mode
        session_state = state.get('session_state', {{}})
        direct_chat_mode = session_state.get('direct_chat_capability') == "{self.capability_name}"

        # Get input based on mode
        if direct_chat_mode:
            # Direct chat: use raw user message and preserve conversation history
            messages = state.get('messages', [])
            if not messages:
                return {{"messages": [MessageUtils.create_assistant_message("No input provided")]}}

            last_message = messages[-1]
            user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
            logger.info(f"Direct chat mode: {{user_input[:50]}}...")

            # Use full message history for conversation context
            agent_messages = messages
        else:
            # Orchestrated mode: use task objective
            task_objective = self.get_task_objective()
            user_input = task_objective
            logger.info(f"Orchestrated mode: {{user_input}}")

            # Single message for orchestrated mode
            agent_messages = [{{"role": "user", "content": user_input}}]

        logger.status(f"Initializing {self.server_name} ReAct agent...")

        try:
            # Get ReAct agent - only include context tools in direct chat mode
            # (orchestrated mode handles context automatically via store_output_context)
            agent = await {class_name}._get_react_agent(
                state,
                include_context_tools=direct_chat_mode
            )
            logger.status(f"Agent reasoning about task...")

            # Invoke ReAct agent
            response = await agent.ainvoke({{"messages": agent_messages}})
            logger.info(f"ReAct agent completed task")

            # Extract final result
            final_message = response["messages"][-1]
            result_content = final_message.content if hasattr(final_message, 'content') else str(final_message)

            # Build state updates
            state_updates = {{}}

            # In direct chat mode, store last result for potential saving via tool
            if direct_chat_mode:
                state_updates['session_state'] = {{
                    **session_state,
                    'last_direct_chat_result': {{
                        'content': result_content,
                        'full_response': response,
                        'timestamp': time_module.time(),
                        'capability': "{self.capability_name}",
                        'context_type': "{context_type}"
                    }}
                }}
                # Add response message for direct chat
                state_updates['messages'] = [MessageUtils.create_assistant_message(result_content)]

                # CRITICAL: Include capability_context_data in state_updates
                # Context tools (save_result_to_context, etc.) modify state["capability_context_data"]
                # in-place during ReAct execution. We must return these changes to LangGraph.
                if state.get("capability_context_data"):
                    state_updates['capability_context_data'] = state["capability_context_data"]

            # Store context in orchestrated mode (store_output_context handles context_key internally)
            if not direct_chat_mode:
                context = {context_class_name}(
                    tool="react_agent",
                    results={{"final_output": result_content, "full_response": response}},
                    description=f"{self.server_name} ReAct agent: {{user_input[:50]}}..."
                )
                context_updates = self.store_output_context(context)
                state_updates.update(context_updates)

            logger.status(f"{self.server_name} ReAct agent complete")
            return state_updates

        except {self.server_name}ConnectionError:
            raise
        except Exception as e:
            error_msg = f"{self.server_name} ReAct agent failed: {{str(e)}}"
            logger.error(error_msg)

            if direct_chat_mode:
                return {{"messages": [MessageUtils.create_assistant_message(f"❌ Error: {{error_msg}}")]}}
            else:
                raise {self.server_name}ToolError(error_msg) from e

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify {self.server_name} MCP errors."""
        if isinstance(exc, {self.server_name}ConnectionError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"{self.server_name} MCP server unavailable: {{str(exc)}}",
                metadata={{
                    "technical_details": str(exc),
                    "safety_abort_reason": f"Cannot connect to {self.server_name} MCP server"
                }}
            )
        elif isinstance(exc, {self.server_name}ToolError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message=f"{self.server_name} operation failed: {{str(exc)}}",
                metadata={{
                    "technical_details": str(exc),
                    "replanning_reason": f"{self.server_name} tool execution failed"
                }}
            )
        else:
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Unexpected {self.server_name} error: {{exc}}",
                metadata={{
                    "technical_details": str(exc),
                    "safety_abort_reason": "Unhandled MCP error"
                }}
            )

    def _create_classifier_guide(self) -> Optional[TaskClassifierGuide]:
        """Classifier guide: When should this capability be activated?"""
        return TaskClassifierGuide(
            instructions=textwrap.dedent("""
                {classifier_analysis.activation_criteria}

                Activate if the query involves:
                {chr(10).join("- " + kw for kw in classifier_analysis.keywords[:10])}

                Do NOT activate for purely informational queries about {self.server_name}.
            """).strip(),
            examples=[
{classifier_examples_code}
            ],
            actions_if_true=ClassifierActions()
        )

    def _create_orchestrator_guide(self) -> Optional[OrchestratorGuide]:
        """Orchestrator guide: How should steps be planned?"""
        return OrchestratorGuide(
            instructions=textwrap.dedent("""
                **When to plan "{self.capability_name}" steps:**
                {orchestrator_analysis.when_to_use}

                **How This Capability Works:**
                This capability uses an internal ReAct agent that autonomously selects and calls MCP tools.

                **Your Role:**
                You only need to:
                1. Decide when to invoke this capability
                2. Formulate clear task_objective descriptions

                **Step Structure:**
                - context_key: Descriptive identifier (e.g., "current_weather_sf", "forecast_tokyo_5day")
                - capability: "{self.capability_name}"
                - task_objective: Clear description of WHAT the user wants
                - expected_output: "{context_type}"

                **Output:** {context_type}
                Contains results from the {self.server_name} ReAct agent.
            """).strip(),
            examples=[
{orchestrator_examples_code}
            ],
            priority=2
        )


# =============================================================================
# Registry Registration
# =============================================================================
"""
Add this to your registry.py:

from osprey.registry import RegistryConfigProvider, extend_framework_registry
from osprey.registry.base import CapabilityRegistration, ContextClassRegistration

class MyAppRegistryProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="{self.capability_name}",
                    module_path="your_app.capabilities.{self.capability_name}",
                    class_name="{class_name}",
                    provides=["{context_type}"],
                    requires=[]
                ),
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="{context_type}",
                    module_path="your_app.capabilities.{self.capability_name}",
                    class_name="{context_class_name}"
                ),
            ]
        )
"""
'''

        return code
