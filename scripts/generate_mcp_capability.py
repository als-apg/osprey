#!/usr/bin/env python3
"""
MCP Capability Generator - Complete Single-File Output

Generates a complete, working Osprey capability from an MCP server.
Everything in one file: capability class, guides, context class, error handling.

QUICK START (Demo Mode - No MCP Server Needed):
    python scripts/generate_mcp_capability.py

This creates a GitHub MCP capability example in: generated_capabilities/github_mcp.py

FROM PROJECT DIRECTORY (uses your config.yml):
    cd my-project
    python ../scripts/generate_mcp_capability.py

STANDALONE MODE (uses framework defaults + override):
    python scripts/generate_mcp_capability.py \
        --provider cborg \
        --model anthropic/claude-sonnet

REAL MODE (with your MCP server):
    python scripts/generate_mcp_capability.py \
        --mcp-url http://localhost:3001 \
        --capability-name slack_mcp \
        --server-name Slack \
        --output-file my_app/capabilities/slack_mcp.py

NOTES:
- Registry is always initialized (falls back to framework-only if no app config)
- Uses orchestrator model config from registry by default
- --provider and --model override the registry config
- Providers come from registry, so you need Osprey installed with valid provider configs
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Try MCP client (optional - can work in simulated mode)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Osprey imports
try:
    from pydantic import BaseModel, Field

    from osprey.models.completion import get_chat_completion
    from osprey.registry import initialize_registry
    from osprey.utils.config import get_model_config
except ImportError:
    print("ERROR: Osprey not installed or not in PYTHONPATH")
    sys.exit(1)


# =============================================================================
# Pydantic Models for LLM Analysis
# =============================================================================

class ClassifierExampleRaw(BaseModel):
    """Raw classifier example from LLM."""
    query: str = Field(description="User query example")
    reason: str = Field(description="Why this should/shouldn't activate")


class ClassifierAnalysis(BaseModel):
    """LLM analysis for classifier guide generation."""
    activation_criteria: str = Field(description="When to activate")
    keywords: List[str] = Field(description="Key indicators")
    positive_examples: List[ClassifierExampleRaw] = Field(description="Should activate")
    negative_examples: List[ClassifierExampleRaw] = Field(description="Should not activate")
    edge_cases: List[str] = Field(description="Tricky scenarios")


class ToolPattern(BaseModel):
    """Tool usage pattern from LLM."""
    tool_name: str = Field(description="Tool name")
    typical_scenario: str = Field(description="When to use this tool")


class ExampleStepRaw(BaseModel):
    """Raw example step from LLM."""
    tool_name: str = Field(description="Tool to invoke")
    task_objective: str = Field(description="What user wants to accomplish")
    scenario: str = Field(description="Real-world scenario description")


class OrchestratorAnalysis(BaseModel):
    """LLM analysis for orchestrator guide generation."""
    when_to_use: str = Field(description="General guidance")
    tool_usage_patterns: List[ToolPattern] = Field(description="Tool patterns")
    example_steps: List[ExampleStepRaw] = Field(description="Example steps")
    common_sequences: List[str] = Field(description="Common patterns")
    important_notes: List[str] = Field(description="Important reminders")


# =============================================================================
# Simulated Tools (for testing without MCP server)
# =============================================================================

SIMULATED_GITHUB_TOOLS = [
    {
        "name": "search_repositories",
        "description": "Search for GitHub repositories using various criteria",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "sort": {"type": "string", "enum": ["stars", "forks", "updated"]},
                "limit": {"type": "integer", "default": 30}
            },
            "required": ["query"]
        }
    },
    {
        "name": "create_or_update_file",
        "description": "Create or update a single file in a repository",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "path": {"type": "string", "description": "Path where to create/update the file"},
                "content": {"type": "string", "description": "Content of the file"},
                "message": {"type": "string", "description": "Commit message"},
                "branch": {"type": "string", "description": "Branch name (optional)"},
                "sha": {"type": "string", "description": "SHA of file being replaced (optional)"}
            },
            "required": ["owner", "repo", "path", "content", "message"]
        }
    },
    {
        "name": "create_pull_request",
        "description": "Create a new pull request",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "title": {"type": "string", "description": "Pull request title"},
                "body": {"type": "string", "description": "Pull request description"},
                "head": {"type": "string", "description": "Branch with your changes"},
                "base": {"type": "string", "description": "Branch to merge into"}
            },
            "required": ["owner", "repo", "title", "head", "base"]
        }
    },
    {
        "name": "create_issue",
        "description": "Create a new issue in a repository",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["owner", "repo", "title"]
        }
    },
]


# =============================================================================
# MCP Capability Generator
# =============================================================================

class MCPCapabilityGenerator:
    """Generate complete MCP capability from MCP server tools."""

    def __init__(
        self,
        capability_name: str,
        server_name: str,
        verbose: bool = False,
        provider: str = None,
        model_id: str = None
    ):
        self.capability_name = capability_name
        self.server_name = server_name
        self.verbose = verbose
        self.tools = []
        self.mcp_url = None
        self.provider = provider
        self.model_id = model_id

    async def discover_tools(self, mcp_url: str = None, simulated: bool = False) -> List[Dict[str, Any]]:
        """Discover tools from MCP server or use simulated tools."""
        if simulated:
            if self.verbose:
                print("Using simulated tools (no MCP server needed)")
            self.tools = SIMULATED_GITHUB_TOOLS
            self.mcp_url = "http://localhost:3001"  # Placeholder
        else:
            if not MCP_AVAILABLE:
                raise RuntimeError(
                    "MCP client not installed. Use --simulated or install: "
                    "pip install mcp"
                )

            if self.verbose:
                print(f"Connecting to MCP server: {mcp_url}")

            self.mcp_url = mcp_url

            # FastMCP SSE endpoint is at /sse
            sse_url = mcp_url if mcp_url.endswith('/sse') else f"{mcp_url}/sse"

            # Use native MCP client to get tools in standardized format
            # MCP protocol guarantees consistent tool structure - no adapter guessing needed!
            async with sse_client(sse_url) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()

                    # List tools using standard MCP protocol
                    # This returns tools in the MCP-standardized format:
                    # - name: string
                    # - description: string
                    # - inputSchema: JSON Schema object
                    tools_result = await session.list_tools()

                    # Convert from MCP's Pydantic models to dicts for JSON serialization
                    self.tools = []
                    for tool in tools_result.tools:
                        tool_dict = {
                            "name": tool.name,
                            "description": tool.description or "",
                            "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        }
                        self.tools.append(tool_dict)

            if self.verbose:
                print(f"✓ Discovered {len(self.tools)} tools")

        return self.tools

    async def generate_guides(self) -> tuple[ClassifierAnalysis, OrchestratorAnalysis]:
        """Generate classifier and orchestrator guides using LLM."""
        if self.verbose:
            print("\n🤖 Analyzing tools with LLM...")

        tools_json = json.dumps(self.tools, indent=2)

        # Get model config from registry (orchestrator model)
        # Registry should already be initialized (framework-only if no app config)
        model_config = get_model_config("orchestrator")

        # Allow explicit provider/model override
        if self.provider and self.model_id:
            if self.verbose:
                print(f"   Using explicit model: {self.provider}/{self.model_id}")
            model_kwargs = {
                "provider": self.provider,
                "model_id": self.model_id,
                "max_tokens": model_config.get("max_tokens", 4096)
            }
        else:
            # Use orchestrator config from registry
            if self.verbose:
                provider = model_config.get("provider", "unknown")
                model_id = model_config.get("model_id", "unknown")
                print(f"   Using orchestrator model: {provider}/{model_id}")
            model_kwargs = {"model_config": model_config}

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

        classifier_analysis = await asyncio.to_thread(
            get_chat_completion,
            message=classifier_prompt,
            **model_kwargs,
            output_model=ClassifierAnalysis
        )

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
- Important notes about formulating good task objectives for the capability

Do NOT include tool_name in examples - the ReAct agent decides which tools to use.

Output as JSON matching the OrchestratorAnalysis schema (but tool_name field can be empty or generic).
"""

        orchestrator_analysis = await asyncio.to_thread(
            get_chat_completion,
            message=orchestrator_prompt,
            **model_kwargs,
            output_model=OrchestratorAnalysis
        )

        if self.verbose:
            print("✓ Guides generated")

        return classifier_analysis, orchestrator_analysis

    def generate_capability_code(
        self,
        classifier_analysis: ClassifierAnalysis,
        orchestrator_analysis: OrchestratorAnalysis
    ) -> str:
        """Generate complete capability Python code."""

        timestamp = datetime.now().isoformat()
        class_name = ''.join(word.title() for word in self.capability_name.split('_')) + 'Capability'
        context_class_name = ''.join(word.title() for word in self.capability_name.split('_')) + 'ResultsContext'
        context_type = f"{self.server_name.upper()}_RESULTS"

        # Build classifier examples
        classifier_examples = []
        for ex in classifier_analysis.positive_examples:
            classifier_examples.append(
                f"            ClassifierExample(\n"
                f"                query=\"{ex.query}\",\n"
                f"                result=True,\n"
                f"                reason=\"{ex.reason}\"\n"
                f"            )"
            )
        for ex in classifier_analysis.negative_examples:
            classifier_examples.append(
                f"            ClassifierExample(\n"
                f"                query=\"{ex.query}\",\n"
                f"                result=False,\n"
                f"                reason=\"{ex.reason}\"\n"
                f"            )"
            )
        classifier_examples_code = ",\n".join(classifier_examples)

        # Build orchestrator examples
        orchestrator_examples = []
        for i, ex in enumerate(orchestrator_analysis.example_steps):
            # Use generic context keys since tool selection is internal
            context_key = f"{self.capability_name}_result_{i+1}"
            orchestrator_examples.append(
                f"            OrchestratorExample(\n"
                f"                step=PlannedStep(\n"
                f"                    context_key=\"{context_key}\",\n"
                f"                    capability=\"{self.capability_name}\",\n"
                f"                    task_objective=\"{ex.task_objective}\",\n"
                f"                    expected_output=\"{context_type}\",\n"
                f"                    success_criteria=\"Successfully completed {self.server_name} operation\",\n"
                f"                    inputs=[]\n"
                f"                ),\n"
                f"                scenario_description=\"{ex.scenario}\",\n"
                f"                notes=\"The ReAct agent autonomously selects appropriate MCP tools based on task_objective\"\n"
                f"            )"
            )
        orchestrator_examples_code = ",\n".join(orchestrator_examples)

        # Build tools list for documentation
        tools_list = "\n".join([f"        - {t['name']}: {t.get('description', 'N/A')}" for t in self.tools])
        tool_patterns = "\n".join([f"        - {p.tool_name}: {p.typical_scenario}" for p in orchestrator_analysis.tool_usage_patterns[:5]])

        # Ensure SSE endpoint path is included
        sse_url = self.mcp_url if self.mcp_url.endswith('/sse') else f"{self.mcp_url}/sse"

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
- Classifier guide (when to activate)
- Orchestrator guide (high-level planning only)
- Context class for results
- Error handling

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
3. Adjust MCP server URL and transport in MCP_SERVER_CONFIG if needed
4. Install required dependencies: pip install langchain-mcp-adapters langgraph
5. Consider moving {context_class_name} to a shared context_classes.py
6. Add to your registry.py (see registration snippet at bottom)
7. Test with real queries

Generated by: scripts/generate_mcp_capability.py
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
from osprey.base.planning import PlannedStep, OrchestratorGuide, OrchestratorExample
from osprey.base.examples import TaskClassifierGuide, ClassifierExample, ClassifierActions
from osprey.context import CapabilityContext
from osprey.state import StateManager
from osprey.registry import get_registry
from osprey.utils.streaming import get_streamer
from osprey.utils.logger import get_logger

# MCP and LangGraph imports
# Note: We use langchain-mcp-adapters here for LangGraph integration
# The adapter converts MCP's standardized tool format to LangChain's format
# which is required by LangGraph's create_react_agent
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
from osprey.models.completion import _get_llm_instance


logger = get_logger("{self.capability_name}")
registry = get_registry()


# =============================================================================
# Context Class - CUSTOMIZE THIS FOR YOUR USE CASE
# =============================================================================
# NOTE: This is a MINIMAL PLACEHOLDER. You should customize this based on your
# actual MCP server's data structure and your application's needs.
#
# Consider moving this to a shared context_classes.py file in your project.
#
# 📚 Documentation: For detailed guidance on creating context classes, see:
# https://als-apg.github.io/osprey/developer-guides/03_core-framework-systems/02_context-management-system.html
#
# Key requirements:
# 1. Inherit from CapabilityContext (Pydantic BaseModel)
# 2. Define CONTEXT_TYPE and CONTEXT_CATEGORY as ClassVar
# 3. Use JSON-serializable Pydantic fields only
# 4. Implement get_access_details() - tells LLM how to use the data
# 5. Implement get_summary() - formats data for human display

class {context_class_name}(CapabilityContext):
    """
    Context for {self.server_name} MCP results.

    TODO: Customize this class based on your MCP server's actual data structure.
    This is a generic placeholder that stores raw tool results.

    For production use, you should:
    - Define specific Pydantic fields matching your data structure
    - Implement rich get_access_details() with clear access patterns
    - Implement get_summary() for better human-readable output
    - Choose appropriate CONTEXT_CATEGORY (see examples in docs)
    """

    # Framework integration constants
    CONTEXT_TYPE: ClassVar[str] = "{context_type}"
    CONTEXT_CATEGORY: ClassVar[str] = "EXTERNAL_DATA"  # TODO: Choose appropriate category

    # Data fields - customize based on your MCP server's output
    tool: str  # Which MCP tool was called
    results: Dict[str, Any]  # Raw results from MCP server - TODO: make this more specific
    description: str  # Human-readable description

    def get_access_details(self, key_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Tell the LLM how to access and use this context data.

        TODO: Customize this to provide clear, specific access patterns for your data structure.
        The LLM uses this to understand how to reference the data in Python code.
        """
        key_ref = key_name if key_name else "key_name"
        return {{
            "tool_used": self.tool,
            "description": self.description,
            "data_structure": "Dict[str, Any] - TODO: describe your actual structure",
            "access_pattern": f"context.{{self.CONTEXT_TYPE}}.{{key_ref}}.results",
            "example_usage": f"context.{{self.CONTEXT_TYPE}}.{{key_ref}}.results['field_name']",
            "available_fields": list(self.results.keys()) if isinstance(self.results, dict) else "results",
        }}

    def get_summary(self, key_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Format data for human display and LLM consumption.

        TODO: Customize this to provide meaningful summaries of your data.
        This is what users see in the UI and what the LLM sees when generating responses.
        """
        return {{
            "type": "{self.server_name} Results",
            "tool": self.tool,
            "description": self.description,
            "results": self.results,  # TODO: consider summarizing large results
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

    Pattern: Just like ChannelFindingCapability calls ChannelFinderService,
    this capability calls the {self.server_name} MCP server.
    """

    name = "{self.capability_name}"
    description = "{self.server_name} operations via MCP server"
    provides = ["{context_type}"]
    requires = []

    # MCP server configuration
    # TODO: Move these to config.yml
    MCP_SERVER_URL = "{sse_url}"
    MCP_SERVER_CONFIG = {{
        "{self.capability_name}_server": {{
            "url": "{sse_url}",
            "transport": "sse",  # or "stdio" depending on your MCP server
        }}
    }}

    # Class-level client and agent cache (shared across all executions)
    _mcp_client: Optional[MultiServerMCPClient] = None
    _react_agent = None

    @classmethod
    async def _get_react_agent(cls):
        """Get or create ReAct agent with MCP tools (cached)."""
        if cls._react_agent is None:
            try:
                # Initialize MCP client
                if cls._mcp_client is None:
                    cls._mcp_client = MultiServerMCPClient(cls.MCP_SERVER_CONFIG)
                    logger.info(f"Connected to MCP server: {{cls.MCP_SERVER_URL}}")

                # Get tools from MCP server
                tools = await cls._mcp_client.get_tools()
                logger.info(f"Loaded {{len(tools)}} tools from MCP server")

                # Get LLM instance for ReAct agent
                model_config = get_model_config("orchestrator")
                llm = _get_llm_instance(model_config)

                # Create ReAct agent with MCP tools
                cls._react_agent = create_react_agent(llm, tools)
                logger.info("ReAct agent initialized with MCP tools")

            except Exception as e:
                logger.error(f"Failed to initialize ReAct agent: {{e}}")
                raise {self.server_name}ConnectionError(f"Cannot initialize ReAct agent: {{e}}")

        return cls._react_agent

    @staticmethod
    async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
        """
        Execute {self.server_name} MCP capability using ReAct agent.

        Pattern: Uses a ReAct agent that autonomously selects and calls MCP tools.
        1. Extract task objective from step
        2. Get ReAct agent with MCP tools
        3. Let agent autonomously reason and act to complete the task
        4. Format final results as context
        5. Return state updates

        The ReAct agent internally:
        - Sees all available MCP tools
        - Reasons about which tool(s) to call
        - Executes tools and observes results
        - Continues until task is complete
        """
        # Extract current step
        step = StateManager.get_current_step(state)
        task_objective = step.get('task_objective', 'unknown')

        streamer = get_streamer("{self.capability_name}", state)

        logger.info(f"{self.server_name} MCP: {{task_objective}}")
        streamer.status(f"Initializing {self.server_name} ReAct agent...")

        try:
            # Get ReAct agent with MCP tools
            agent = await {class_name}._get_react_agent()

            streamer.status(f"Agent reasoning about task: {{task_objective[:50]}}...")

            # Invoke ReAct agent with task objective
            # The agent will autonomously:
            # 1. Reason about what tool(s) to use
            # 2. Call the appropriate MCP tool(s)
            # 3. Observe results and decide next steps
            # 4. Repeat until task is complete
            response = await agent.ainvoke({{
                "messages": [{{
                    "role": "user",
                    "content": task_objective
                }}]
            }})

            logger.info(f"ReAct agent completed task")

            # Extract final result from agent response
            # The agent's last message contains the final answer
            final_message = response["messages"][-1]
            result_content = final_message.content if hasattr(final_message, 'content') else str(final_message)

            # Format as context
            context = {context_class_name}(
                tool="react_agent",  # Multiple tools may have been used
                results={{"final_output": result_content, "full_response": response}},
                description=f"{self.server_name} ReAct agent: {{task_objective}}"
            )

            # Store in state
            state_updates = StateManager.store_context(
                state,
                registry.context_types.{context_type},
                step.get("context_key"),
                context
            )

            streamer.status(f"{self.server_name} ReAct agent complete")

            return state_updates

        except {self.server_name}ConnectionError:
            raise  # Re-raise connection errors
        except Exception as e:
            error_msg = f"{self.server_name} ReAct agent failed: {{str(e)}}"
            logger.error(error_msg)
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
        """
        Classifier guide: When should this capability be activated?

        Auto-generated from MCP tool analysis.
        """
        return TaskClassifierGuide(
            instructions=textwrap.dedent("""
                {classifier_analysis.activation_criteria}

                Activate if the query involves:
                {chr(10).join('- ' + kw for kw in classifier_analysis.keywords[:10])}

                Do NOT activate for purely informational queries about {self.server_name}.
            """).strip(),
            examples=[
{classifier_examples_code}
            ],
            actions_if_true=ClassifierActions()
        )

    def _create_orchestrator_guide(self) -> Optional[OrchestratorGuide]:
        """
        Orchestrator guide: How should steps be planned?

        Auto-generated from MCP tool analysis.
        Uses ReAct pattern - orchestrator describes WHAT, not HOW.
        """
        return OrchestratorGuide(
            instructions=textwrap.dedent("""
                **When to plan "{self.capability_name}" steps:**
                {orchestrator_analysis.when_to_use}

                **How This Capability Works:**
                This capability uses an internal ReAct agent that autonomously:
                - Sees all available {self.server_name} MCP tools
                - Reasons about which tool(s) to use
                - Executes tools and observes results
                - Continues until the task is complete

                **Your Role as Orchestrator:**
                You do NOT need to know about specific MCP tools or how to use them.
                You only need to:
                1. Decide when to invoke this capability (based on user needs)
                2. Formulate clear task_objective descriptions

                **Step Structure:**
                - context_key: Unique identifier for output (e.g., "{self.capability_name}_result_1")
                - capability: "{self.capability_name}"
                - task_objective: Clear description of WHAT the user wants (not HOW to implement it)
                - expected_output: "{context_type}"

                **Important Notes:**
                {chr(10).join('- ' + note for note in orchestrator_analysis.important_notes)}
                - Do NOT specify tool_name - the ReAct agent decides tool selection
                - Focus task_objective on the user's goal, not implementation details
                - The ReAct agent handles all {self.server_name} MCP tool interactions

                **Output:** {context_type}
                Contains results from the {self.server_name} ReAct agent's autonomous execution.
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


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Generate complete MCP capability for Osprey",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DEMO MODE (simulated GitHub MCP server)
  python scripts/generate_mcp_capability.py

  # FROM PROJECT (uses your project's config.yml)
  cd my-project
  python ../scripts/generate_mcp_capability.py

  # WITH MODEL OVERRIDE (uses framework defaults + custom model)
  python scripts/generate_mcp_capability.py \\
      --provider cborg \\
      --model anthropic/claude-sonnet

  # REAL MCP SERVER
  python scripts/generate_mcp_capability.py \\
      --mcp-url http://localhost:3001 \\
      --capability-name slack_mcp \\
      --server-name Slack

Notes:
  - Registry is always initialized (framework-only if no app config)
  - Uses orchestrator model config from registry by default
  - --provider and --model override the registry config
        """
    )

    parser.add_argument(
        '--mcp-url',
        help='MCP server URL (e.g., http://localhost:3001)'
    )

    parser.add_argument(
        '--simulated',
        action='store_true',
        help='Use simulated tools instead of real MCP server'
    )

    parser.add_argument(
        '--capability-name',
        default='github_mcp',
        help='Name for the capability (default: github_mcp)'
    )

    parser.add_argument(
        '--server-name',
        default='GitHub',
        help='Human-readable server name (default: GitHub)'
    )

    parser.add_argument(
        '--output-file',
        default='generated_capabilities/github_mcp.py',
        help='Output file path (default: generated_capabilities/github_mcp.py)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )

    parser.add_argument(
        '--provider',
        help='LLM provider override (e.g., cborg, anthropic, openai). Defaults to orchestrator config'
    )

    parser.add_argument(
        '--model',
        help='Model ID override (e.g., anthropic/claude-sonnet, gpt-4o). Defaults to orchestrator config'
    )

    args = parser.parse_args()

    # Default to simulated mode if no MCP URL provided
    if not args.mcp_url:
        args.simulated = True

    # Create output directory
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"MCP Capability Generator for Osprey")
    print("=" * 70)
    print(f"Capability: {args.capability_name}")
    print(f"Server: {args.server_name}")
    print(f"Mode: {'Simulated' if args.simulated else f'Real MCP ({args.mcp_url})'}")
    print(f"Output: {output_file}")
    print("=" * 70)

    # Initialize Osprey registry (required for providers)
    # Falls back to framework-only registry if no application config found
    if not args.quiet:
        print("\n📚 Initializing registry...")

    try:
        initialize_registry()
        if not args.quiet:
            print("✓ Registry initialized")
    except Exception as e:
        print(f"\n❌ ERROR: Could not initialize registry: {e}")
        print(f"   The registry is required for LLM provider configuration.")
        print(f"   Please ensure you have a valid Osprey installation.")
        return 1

    # Initialize generator
    generator = MCPCapabilityGenerator(
        capability_name=args.capability_name,
        server_name=args.server_name,
        verbose=not args.quiet,
        provider=args.provider,
        model_id=args.model
    )

    try:
        # Step 1: Discover tools
        print("\n📡 Step 1: Discovering MCP tools...")
        tools = await generator.discover_tools(
            mcp_url=args.mcp_url,
            simulated=args.simulated
        )

        if not tools:
            print("❌ ERROR: No tools found")
            return 1

        print(f"✓ Found {len(tools)} tools:")
        for tool in tools[:5]:  # Show first 5
            print(f"  - {tool['name']}")
        if len(tools) > 5:
            print(f"  ... and {len(tools) - 5} more")

        # Step 2: Generate guides with LLM
        print("\n🤖 Step 2: Generating guides with LLM...")
        classifier_analysis, orchestrator_analysis = await generator.generate_guides()
        print(f"✓ Generated {len(classifier_analysis.positive_examples) + len(classifier_analysis.negative_examples)} classifier examples")
        print(f"✓ Generated {len(orchestrator_analysis.example_steps)} orchestrator examples")

        # Step 3: Generate capability code
        print("\n📝 Step 3: Generating capability code...")
        code = generator.generate_capability_code(classifier_analysis, orchestrator_analysis)

        # Step 4: Write to file
        print("\n💾 Step 4: Writing output file...")
        with open(output_file, 'w') as f:
            f.write(code)

        print(f"✓ Written: {output_file} ({len(code)} bytes)")

        # Success summary
        print("\n" + "=" * 70)
        print("✅ SUCCESS! MCP Capability Generated")
        print("=" * 70)
        print("\n📋 What was created:")
        print(f"  ✓ Capability class: {args.capability_name}")
        print(f"  ✓ MCP client integration (inline)")
        print(f"  ✓ Classifier guide with {len(classifier_analysis.positive_examples) + len(classifier_analysis.negative_examples)} examples")
        print(f"  ✓ Orchestrator guide with {len(orchestrator_analysis.example_steps)} examples")
        print(f"  ✓ Context class for results")
        print(f"  ✓ Error handling")
        print(f"  ✓ Registry registration snippet")
        print("\n📋 Next Steps:")
        print(f"  1. Review: {output_file}")
        print(f"  2. Adjust MCP_SERVER_URL if needed")
        print(f"  3. Add to your registry.py (see snippet at bottom of file)")
        print(f"  4. Test with: osprey chat (after registering)")
        print("\n💡 Pro Tip: The generated code follows the same pattern as")
        print("   ChannelFindingCapability - it's just calling a different service!")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

