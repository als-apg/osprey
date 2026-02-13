"""
Middle Layer React Agent Pipeline

Implements channel finding using LangGraph's ReAct agent with database query tools.
This approach mimics production accelerator control systems where an agent
explores a functional hierarchy (System → Family → Field) using tools rather
than navigating a tree structure.

Key features:
- Uses LangGraph's create_react_agent for autonomous exploration
- Agent queries database using LangChain tools
- Channel addresses are retrieved from database (not built from patterns)
- Organization is by function (Monitor, Setpoint) not naming pattern
- Supports device/sector filtering and subfield navigation

This pipeline is based on the MATLAB Middle Layer (MML) pattern used in
production at facilities like ALS, ESRF, and others.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from osprey.models import get_chat_completion
from osprey.utils.config import get_config_builder

# LangGraph and LangChain imports
try:
    from langchain_core.tools import StructuredTool, tool
    from langgraph.prebuilt import create_react_agent
except ImportError as err:
    raise ImportError(
        "LangGraph not installed. Install with: pip install langgraph langchain-core"
    ) from err

from ...core.base_pipeline import BasePipeline
from ...core.models import ChannelFinderResult, ChannelInfo, QuerySplitterOutput
from ...utils.prompt_loader import load_prompts

logger = logging.getLogger(__name__)


# === Structured Output Models ===


class ChannelSearchResult(BaseModel):
    """Structured result from channel search with validation."""

    channels: list[str] = Field(
        description="List of channel addresses found (e.g., ['SR:DCCT:Current', 'SR01C:BPM1:X']). Empty list if none found."
    )
    description: str = Field(
        description="Brief 1-2 sentence summary. State: (1) what channel(s) were found, (2) which system/family/field. Omit detailed search steps."
    )


# === Tool Support Functions ===


def _save_prompt_to_file(prompt: str, stage: str, query: str = "") -> None:
    """Save prompt to temporary file for inspection."""
    config_builder = get_config_builder()
    if not config_builder.get("debug.save_prompts", False):
        return

    prompts_dir = config_builder.get("debug.prompts_dir", "temp_prompts")
    project_root = Path(config_builder.get("project_root"))
    temp_dir = project_root / prompts_dir
    temp_dir.mkdir(exist_ok=True, parents=True)

    # Map stage names
    if stage == "query_split":
        filename = "prompt_stage1_query_split.txt"
    elif stage == "pv_query":
        filename = "prompt_stage2_pv_query.txt"
    else:
        filename = f"prompt_{stage}.txt"

    filepath = temp_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"=== STAGE: {stage.upper()} ===\n")
        f.write(f"=== TIMESTAMP: {datetime.now().isoformat()} ===\n")
        if query:
            f.write(f"=== QUERY: {query} ===\n")
        f.write("=" * 80 + "\n\n")
        f.write(prompt)

    logger.debug(f"  [dim]Saved prompt to: {filepath}[/dim]")


# === Pipeline Implementation ===


class MiddleLayerPipeline(BasePipeline):
    """
    Middle Layer React Agent Pipeline.

    Uses React-style agent with database query tools to find channels.
    """

    def __init__(
        self,
        database,  # MiddleLayerDatabase
        model_config: dict,
        facility_name: str = "control system",
        facility_description: str = "",
        query_splitting: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize middle layer pipeline.

        Args:
            database: MiddleLayerDatabase instance
            model_config: LLM model configuration
            facility_name: Name of facility
            facility_description: Facility description for context
            query_splitting: Whether to split multi-part queries (disable for facility-specific lingo)
            **kwargs: Additional pipeline arguments
        """
        super().__init__(database, model_config, **kwargs)
        self.facility_name = facility_name
        self.facility_description = facility_description
        self.query_splitting = query_splitting

        # Load query splitter prompt (only if enabled)
        config_builder = get_config_builder()
        prompts_module = load_prompts(
            config_builder.raw_config, require_query_splitter=query_splitting
        )
        self.query_splitter = (
            getattr(prompts_module, "query_splitter", None) if query_splitting else None
        )

        # Agent will be created lazily on first use
        self._agent = None

    @property
    def pipeline_name(self) -> str:
        """Return the pipeline name."""
        return "Middle Layer React Agent"

    def _create_tools(self) -> list:
        """Create LangChain tools for database queries."""
        # Use the @tool decorator for each database operation
        # We'll capture self.database in closures

        @tool
        def list_systems() -> list[dict[str, str]]:
            """Get list of all available systems in the control system.

            Returns:
                List of dicts with 'name' and 'description' keys.
                Example: [
                    {'name': 'SR', 'description': 'Storage Ring - main synchrotron light source'},
                    {'name': 'BR', 'description': 'Booster Ring - accelerates beam to 1.9 GeV'},
                    {'name': 'BTS', 'description': ''}  # Empty string if no description
                ]
            """
            logger.info("Tool: list_systems() called")
            result = self.database.list_systems()
            logger.debug(f"  → Returned {len(result)} systems")
            return result

        @tool
        def list_families(system: str) -> list[dict[str, str]]:
            """Get list of device families in a specific system.

            Args:
                system: System name (e.g., 'SR', 'BR')

            Returns:
                List of dicts with 'name' and 'description' keys.
                Example: [
                    {'name': 'BPM', 'description': 'Beam Position Monitors - measure beam X/Y position'},
                    {'name': 'HCM', 'description': 'Horizontal Corrector Magnets'},
                    {'name': 'DCCT', 'description': ''}  # Empty string if no description
                ]
            """
            logger.info(f"Tool: list_families(system='{system}') called")
            try:
                result = self.database.list_families(system)
                logger.debug(f"  → Returned {len(result)} families")
                return result
            except ValueError as e:
                return {"error": str(e)}

        @tool
        def inspect_fields(
            system: str, family: str, field: str = None
        ) -> dict[str, dict[str, str]]:
            """Inspect the structure of fields within a family.

            Use this to discover what fields and subfields are available
            before querying for channel names. Includes descriptions when available.

            Args:
                system: System name
                family: Family name
                field: Optional field name to inspect subfields (if None, shows top-level fields)

            Returns:
                Dict mapping field/subfield names to dicts with 'type' and 'description'.
                Example: {
                    'Monitor': {
                        'type': 'ChannelNames',
                        'description': 'Position readback values in mm'
                    },
                    'Setpoint': {
                        'type': 'dict (has subfields)',
                        'description': 'Position setpoint controls'
                    },
                    'OnControl': {
                        'type': 'ChannelNames',
                        'description': ''  # Empty string if no description provided
                    }
                }
            """
            logger.info(
                f"Tool: inspect_fields(system='{system}', family='{family}', field='{field}') called"
            )
            try:
                result = self.database.inspect_fields(system, family, field)
                logger.debug(f"  → Returned {len(result)} fields")
                return result
            except ValueError as e:
                return {"error": str(e)}

        @tool
        def list_channel_names(
            system: str,
            family: str,
            field: str,
            subfield: str = None,
            sectors: list[int] = None,
            devices: list[int] = None,
        ) -> list[str]:
            """Get the actual PV addresses for a specific field or subfield.

            This is the main tool for retrieving channel names. Use after
            exploring the structure with inspect_fields.

            Args:
                system: System name (e.g., 'SR')
                family: Family name (e.g., 'BPM')
                field: Field name (e.g., 'Monitor', 'Setpoint')
                subfield: Optional subfield name (e.g., 'X', 'Y' under Setpoint)
                sectors: Optional list of sector numbers to filter by
                devices: Optional list of device numbers to filter by

            Returns:
                List of PV addresses (e.g., ['SR01C:BPM1:X', 'SR01C:BPM2:X'])
            """
            logger.info(
                f"Tool: list_channel_names(system='{system}', family='{family}', "
                f"field='{field}', subfield='{subfield}', sectors={sectors}, devices={devices}) called"
            )
            try:
                result = self.database.list_channel_names(
                    system, family, field, subfield, sectors, devices
                )
                logger.debug(f"  → Returned {len(result)} channels")
                return result
            except ValueError as e:
                return {"error": str(e)}

        @tool
        def get_common_names(system: str, family: str) -> list[str]:
            """Get friendly/common names for devices in a family.

            Useful for understanding what devices exist before filtering
            by sectors/devices.

            Args:
                system: System name
                family: Family name

            Returns:
                List of common names (e.g., ['BPM 1', 'BPM 2', ...])
                Returns empty list if not available.
            """
            logger.info(f"Tool: get_common_names(system='{system}', family='{family}') called")
            result = self.database.get_common_names(system, family)
            if result is None:
                logger.debug("  → No common names available")
                return []
            logger.debug(f"  → Returned {len(result)} common names")
            return result

        # Create the report_results tool with structured input
        def report_results_func(channels: list[str], description: str) -> str:
            """Report your final search results (REQUIRED - call this when done).

            This tool MUST be called when you've completed your search to report findings.
            Provide the channel addresses you found and a BRIEF description.

            Args:
                channels: List of channel addresses found. Empty list if none found.
                    Example: ["SR:DCCT:Current", "SR01C:BPM1:X", "SR01C:BPM2:X"]
                description: Brief 1-2 sentence summary stating what was found and where (system/family/field).
                    Keep it concise - omit detailed search steps.

            Returns:
                Confirmation message
            """
            # Results will be extracted from tool call args in the response messages
            # No need for side-channel storage - this is now thread-safe!
            return f"✓ Results reported: {len(channels)} channel(s) found"

        # Create a StructuredTool with explicit schema for validation
        report_results = StructuredTool.from_function(
            func=report_results_func,
            name="report_results",
            description=report_results_func.__doc__,
            args_schema=ChannelSearchResult,
        )

        # Return list of all tools (report_results must be last per instructions to agent)
        return [
            list_systems,
            list_families,
            inspect_fields,
            list_channel_names,
            get_common_names,
            report_results,
        ]

    async def _get_agent(self):
        """Get or create ReAct agent with database tools (cached)."""
        if self._agent is None:
            # Create tools
            tools = self._create_tools()

            # Get LLM instance for ReAct agent - use model_config passed to __init__
            # Create LangChain model using osprey's unified factory
            # model_config already contains provider, model_id, api_key, base_url, etc.
            from osprey.models import get_langchain_model

            llm = get_langchain_model(
                provider=self.model_config.get("provider"),
                model_id=self.model_config.get("model_id"),
                provider_config={
                    "api_key": self.model_config.get("api_key"),
                    "base_url": self.model_config.get("base_url"),
                },
                max_tokens=self.model_config.get("max_tokens", 4096),
            )

            # Create ReAct agent with LangGraph
            self._agent = create_react_agent(llm, tools)
            logger.info("Middle Layer ReAct agent initialized")

        return self._agent

    def _get_system_prompt(self) -> str:
        """Build system prompt for the React agent."""
        facility_context = ""
        if self.facility_description:
            facility_context = f"\nFacility Context:\n{self.facility_description}\n"

        prompt = f"""You are a specialized agent for finding process variable (PV) addresses in the {self.facility_name} control system.
{facility_context}
Your task is to explore the control system database using the provided tools and find the correct PV addresses that match the user's query.

Database Organization:
The database follows a Middle Layer (MML) functional hierarchy:
- **Systems**: Top-level accelerator systems (e.g., SR=Storage Ring, BR=Booster Ring, BTS=Booster-to-Storage transport)
- **Families**: Device families within systems (e.g., BPM=Beam Position Monitors, HCM=Horizontal Corrector Magnets, DCCT=Beam Current)
- **Fields**: Functional categories (e.g., Monitor=readback values, Setpoint=control values)
- **Subfields**: Optional nested organization within fields (e.g., X/Y positions, different signal types)
- **ChannelNames**: The actual EPICS PV addresses you need to return

Using Descriptions:
The database MAY include optional description fields at various levels (systems, families, fields, subfields).
When descriptions are present:
- READ THEM CAREFULLY - they provide crucial context about what each component does
- USE THEM to match user queries - descriptions often contain domain-specific terminology
- PRIORITIZE matches based on description content over just names
- Example: If query asks for "beam position readback", a field with description "Position readback values in mm" is likely correct

When descriptions are ABSENT (empty strings):
- Fall back to interpreting the names themselves
- Use common patterns and domain knowledge
- Descriptive names like "Monitor", "Setpoint", "X", "Y" are often self-explanatory

Strategy:
1. Start by exploring available systems with list_systems() - check descriptions if available
2. Find relevant families with list_families(system) - read family descriptions to understand device types
3. Inspect field structure with inspect_fields(system, family) - use descriptions to understand field purposes
4. For nested structures, use inspect_fields(system, family, field) to explore subfields
5. Retrieve channel names with list_channel_names(system, family, field, ...)
6. Use filtering (sectors, devices) when query specifies specific devices
7. Use subfield parameter when fields have nested structure

Common Patterns (when descriptions are not available):
- "beam current" → Usually in DCCT family, Monitor field
- "BPM positions" → BPM family, Monitor or Setpoint fields, may have X/Y subfields
- "corrector magnets" → HCM (horizontal) or VCM (vertical) families
- "quadrupole" or "quad" → QF (focusing) or QD (defocusing) families, Monitor for readbacks
- "sextupole" or "sext" → SF (focusing) or SD (defocusing) families, Monitor for readbacks
- "focusing" magnets → QF (quadrupole) or SF (sextupole)
- "defocusing" magnets → QD (quadrupole) or SD (sextupole)
- "readback" or "monitor" or "current" → Use Monitor field
- "setpoint" or "control" → Use Setpoint field
- Specific device numbers (e.g., "BPM 1") → Use devices parameter for filtering

Important:
- Always use tools to explore the database - don't guess channel names
- Descriptions are OPTIONAL - some databases have them, some don't
- When available, descriptions are the BEST source of information
- If query mentions specific systems/devices, focus your search there
- Return ALL matching channels if query is general (e.g., "all BPM positions")
- Use filtering to narrow down when query specifies particular devices
- **CRITICAL**: When you've completed your search, you MUST call the `report_results` tool with:
  - channels: List of channel addresses you found (empty list if none)
  - description: Brief 1-2 sentence summary - state what was found and where (system/family/field). Keep it concise.

Example workflow:
1. list_systems() → see what's available
2. list_families('SR') → find relevant families
3. inspect_fields('SR', 'DCCT') → check field structure
4. list_channel_names('SR', 'DCCT', 'Monitor') → get actual PV addresses
5. **report_results(channels=["SR:DCCT:Current"], description="Found beam current in SR:DCCT:Monitor field in the Storage Ring system.")**
"""
        return prompt

    async def process_query(self, query: str) -> ChannelFinderResult:
        """
        Execute middle layer pipeline.

        Stages:
        0. Detect explicit channel addresses (optimization)
        1. Split query (if needed)
        2. For each sub-query, run React agent with tools
        3. Aggregate results

        Args:
            query: Natural language query

        Returns:
            ChannelFinderResult with found channels
        """
        # Handle empty query
        if not query or not query.strip():
            return ChannelFinderResult(
                query=query, channels=[], total_channels=0, processing_notes="Empty query provided"
            )

        # Stage 0: Check for explicit channel addresses (optimization)
        logger.info("[bold cyan]Pre-check:[/bold cyan] Detecting explicit channel addresses...")
        detection_result = await self._detect_explicit_channels(query)

        # Track explicit channels separately
        explicit_channels = []

        if detection_result.has_explicit_addresses:
            logger.info(
                f"  → Detected {len(detection_result.channel_addresses)} explicit address(es)"
            )
            logger.info(f"  → [dim]{detection_result.reasoning}[/dim]")

            # Validate detected addresses (mode: strict/lenient/skip)
            valid_channels, invalid_channels = self._validate_explicit_channels(
                detection_result.channel_addresses
            )

            explicit_channels = valid_channels

            # Check if we need additional search
            if valid_channels and not detection_result.needs_additional_search:
                # Found valid explicit addresses and no additional search needed - done!
                logger.info(
                    f"[green]✓[/green] Found {len(valid_channels)} channel(s) from explicit addresses "
                    f"(skipped agent search)"
                )
                return self._build_result(query, valid_channels)
            elif valid_channels and detection_result.needs_additional_search:
                # Have some explicit channels but need to search for more
                logger.info(
                    f"  → Found {len(valid_channels)} explicit channel(s), but query requires additional search"
                )
                logger.info("  → Proceeding with agent search for additional channels")
            elif invalid_channels:
                # Had explicit addresses but none were valid - fall back to agent search
                logger.info(
                    "  → Explicit addresses not found in database, falling back to agent search"
                )
        else:
            logger.info(f"  → [dim]{detection_result.reasoning}[/dim]")
            logger.info("  → Proceeding with agent search")

        # Stage 1: Split query into atomic queries (optional)
        if self.query_splitting:
            atomic_queries = await self._split_query(query)
            logger.info(
                f"[bold cyan]Stage 1:[/bold cyan] Split into {len(atomic_queries)} atomic quer{'y' if len(atomic_queries) == 1 else 'ies'}"
            )

            if len(atomic_queries) > 1:
                for i, aq in enumerate(atomic_queries, 1):
                    logger.info(f"  → Query {i}: {aq}")
        else:
            atomic_queries = [query]
            logger.info(
                "[bold cyan]Stage 1:[/bold cyan] Query splitting disabled, using original query"
            )

        # Stage 2: Process each query with React agent
        all_channels = []
        for i, atomic_query in enumerate(atomic_queries, 1):
            if len(atomic_queries) == 1:
                logger.info("[bold cyan]Stage 2:[/bold cyan] Querying database with React agent...")
            else:
                logger.info(
                    f"[bold cyan]Stage 2 - Query {i}/{len(atomic_queries)}:[/bold cyan] {atomic_query}"
                )

            try:
                result = await self._query_with_agent(atomic_query)
                all_channels.extend(result["channels"])
                logger.info(f"  → Found {len(result['channels'])} channel(s)")
                logger.info(f"  [dim]→ {result['description']}[/dim]")
            except Exception as e:
                logger.error(f"  [red]✗[/red] Error processing query: {e}")
                continue

        # Stage 3: Merge explicit channels with search results and deduplicate
        all_channels.extend(explicit_channels)
        unique_channels = list(dict.fromkeys(all_channels))  # Preserve order while deduplicating

        return self._build_result(query, unique_channels)

    async def _split_query(self, query: str) -> list[str]:
        """Split query into atomic sub-queries."""
        prompt = self.query_splitter.get_prompt(facility_name=self.facility_name)
        message = f"{prompt}\n\nQuery to process: {query}"

        _save_prompt_to_file(message, "query_split", query)

        # Set caller context for API call logging
        from osprey.models import set_api_call_context

        set_api_call_context(
            function="_split_query",
            module="middle_layer.pipeline",
            class_name="MiddleLayerPipeline",
            extra={"stage": "query_split"},
        )

        response = await asyncio.to_thread(
            get_chat_completion,
            message=message,
            model_config=self.model_config,
            output_model=QuerySplitterOutput,
        )

        return response.queries

    async def _query_with_agent(self, query: str) -> dict[str, Any]:
        """Query database using React agent with tools."""
        # Set caller context for API call logging
        from osprey.models import set_api_call_context

        set_api_call_context(
            function="_query_with_agent",
            module="middle_layer.pipeline",
            class_name="MiddleLayerPipeline",
            extra={"stage": "channel_query"},
        )

        # Get agent
        agent = await self._get_agent()

        # Build prompt with system context
        system_prompt = self._get_system_prompt()
        full_query = f"{system_prompt}\n\nUser Query: {query}"

        # Run LangGraph agent (no stdout suppression needed - we extract from tool call)
        response = await agent.ainvoke({"messages": [{"role": "user", "content": full_query}]})

        # Extract structured result from report_results tool call in messages
        # This is thread-safe and works with concurrent queries
        for message in response.get("messages", []):
            # Check if message has tool_calls attribute
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "report_results":
                        # Extract args directly from the tool call
                        args = tool_call.get("args", {})
                        return {
                            "channels": args.get("channels", []),
                            "description": args.get("description", "No description provided"),
                        }

            # Also check for ToolMessage responses (where tool results are stored)
            if hasattr(message, "name") and message.name == "report_results":
                # Tool was called, check if result is available
                # The actual structured data should be in the tool_calls, but keep this as fallback
                pass

        # Fallback: If agent didn't call report_results, log warning and return empty
        logger.warning("Agent did not call report_results tool. Returning empty result.")
        final_message = response["messages"][-1] if response.get("messages") else None
        fallback_description = (
            final_message.content
            if final_message and hasattr(final_message, "content")
            else "Agent completed without calling report_results"
        )

        return {
            "channels": [],
            "description": f"WARNING: Agent did not report results properly. Agent response: {fallback_description}",
        }

    def _build_result(self, query: str, channels_list: list[str]) -> ChannelFinderResult:
        """
        Build final result object.

        Note: Channels passed here have already been validated according to
        explicit_validation_mode config. No additional validation needed.

        For middle layer database: channel name = address, descriptions are at
        hierarchy branch points (system/family/field), not individual PVs.
        """
        channel_infos = [
            ChannelInfo(
                channel=channel_address,
                address=channel_address,
                description=None,
            )
            for channel_address in channels_list
        ]

        notes = (
            f"Processed query using React agent with database tools. "
            f"Found {len(channel_infos)} channels."
        )

        return ChannelFinderResult(
            query=query,
            channels=channel_infos,
            total_channels=len(channel_infos),
            processing_notes=notes,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Return pipeline statistics."""
        db_stats = self.database.get_statistics()
        return {
            "total_channels": db_stats.get("total_channels", 0),
            "systems": db_stats.get("systems", 0),
            "families": db_stats.get("families", 0),
        }
