"""
Hierarchical Pipeline Implementation

Iterative navigation through structured channel hierarchy.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

# Use Osprey's config system
from osprey.utils.config import get_config_builder

from ...core.base_pipeline import BasePipeline
from ...core.exceptions import HierarchicalNavigationError
from ...core.models import ChannelFinderResult, ChannelInfo, QuerySplitterOutput
from ...llm import get_chat_completion
from ...utils.prompt_loader import load_prompts
from .models import NOTHING_FOUND_MARKER, create_selection_model

logger = logging.getLogger(__name__)


def _save_prompt_to_file(prompt: str, stage: str, level: str = "", query: str = ""):
    """Save prompt to temporary file for inspection.

    Only saves if debug.save_prompts is enabled in config.yml.

    Args:
        prompt: The prompt text to save
        stage: Stage name (e.g., "query_split", "level_selection")
        level: Optional level name for hierarchical stages
        query: Optional query identifier
    """
    config_builder = get_config_builder()
    if not config_builder.get("debug.save_prompts", False):
        return

    # Get temp directory from config or use default
    prompts_dir = config_builder.get("debug.prompts_dir", "temp_prompts")
    project_root = Path(config_builder.get("project_root"))
    temp_dir = project_root / prompts_dir
    temp_dir.mkdir(exist_ok=True, parents=True)

    # Map stage names to descriptive filenames
    if stage == "query_split":
        filename = "prompt_stage1_query_split.txt"
    elif stage == "level_selection" and level:
        # Dynamic level mapping based on hierarchy definition
        config_builder = get_config_builder()
        hierarchy_levels = config_builder.get(
            "channel_finder.pipelines.hierarchical.database.hierarchy_levels", []
        )

        try:
            level_idx = hierarchy_levels.index(level) + 1
            filename = f"prompt_stage2_level{level_idx}_{level}.txt"
        except (ValueError, AttributeError):
            filename = f"prompt_stage2_level_{level}.txt"
    else:
        filename = f"prompt_{stage}.txt"

    filepath = temp_dir / filename

    # Save prompt
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"=== STAGE: {stage.upper()} ===\n")
        if level:
            f.write(f"=== LEVEL: {level.upper()} ===\n")
        f.write(f"=== TIMESTAMP: {datetime.now().isoformat()} ===\n")
        if query:
            f.write(f"=== QUERY: {query} ===\n")
        f.write("=" * 80 + "\n\n")
        f.write(prompt)

    logger.debug(f"  [dim]Saved prompt to: {filepath}[/dim]")


class HierarchicalPipeline(BasePipeline):
    """
    Hierarchical navigation pipeline.

    Navigates through hierarchy levels iteratively:
    SYSTEM → FAMILY → DEVICE → FIELD → SUBFIELD
    """

    def __init__(
        self,
        database,  # HierarchicalChannelDatabase
        model_config: dict,
        facility_name: str = "control system",
        facility_description: str = "",
        query_splitting: bool = True,
        **kwargs,
    ):
        """
        Initialize hierarchical pipeline.

        Args:
            database: HierarchicalChannelDatabase instance
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

        # Load query splitter from shared prompts (only if query splitting is enabled)
        config_builder = get_config_builder()
        prompts_module = load_prompts(
            config_builder.raw_config, require_query_splitter=query_splitting
        )
        self.query_splitter = (
            getattr(prompts_module, "query_splitter", None) if query_splitting else None
        )

        # Load hierarchical navigation context from prompts (not database - separation of data vs instructions)
        if hasattr(prompts_module, "hierarchical_context"):
            self.hierarchical_context = (
                prompts_module.hierarchical_context.get_hierarchical_context()
            )
        else:
            # Fallback for facilities without hierarchical context
            self.hierarchical_context = {}

    @property
    def pipeline_name(self) -> str:
        """Return the pipeline name."""
        return "Hierarchical Navigation"

    async def process_query(self, query: str) -> ChannelFinderResult:
        """
        Execute hierarchical pipeline.

        Stages:
        0. Detect explicit channel addresses (optimization)
        1. Split query (if needed)
        2. For each sub-query, navigate hierarchy
        3. Build channel names from selections
        4. Aggregate results
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
                    f"(skipped hierarchical navigation)"
                )
                return self._build_result(query, valid_channels)
            elif valid_channels and detection_result.needs_additional_search:
                # Have some explicit channels but need to search for more
                logger.info(
                    f"  → Found {len(valid_channels)} explicit channel(s), but query requires additional search"
                )
                logger.info("  → Proceeding with hierarchical navigation for additional channels")
            elif invalid_channels:
                # Had explicit addresses but none were valid - fall back to hierarchical search
                logger.info(
                    "  → Explicit addresses not found in database, falling back to hierarchical search"
                )
        else:
            logger.info(f"  → [dim]{detection_result.reasoning}[/dim]")
            logger.info("  → Proceeding with hierarchical navigation")

        # Stage 1: Split query into atomic queries (optional)
        if self.query_splitting:
            atomic_queries = await self._split_query(query)
            logger.info(
                f"[bold cyan]Stage 1:[/bold cyan] Split into {len(atomic_queries)} atomic quer{'y' if len(atomic_queries) == 1 else 'ies'}"
            )

            # Only show individual queries if there are multiple (avoid redundancy for single query)
            if len(atomic_queries) > 1:
                for i, aq in enumerate(atomic_queries, 1):
                    logger.info(f"  → Query {i}: {aq}")
        else:
            atomic_queries = [query]
            logger.info(
                "[bold cyan]Stage 1:[/bold cyan] Query splitting disabled, using original query"
            )

        # Stage 2: Navigate hierarchy for each atomic query
        all_channels = []
        for i, atomic_query in enumerate(atomic_queries, 1):
            # For single query, keep it concise since we already know the task from capability logs
            if len(atomic_queries) == 1:
                logger.info("[bold cyan]Stage 2:[/bold cyan] Navigating hierarchy...")
            else:
                logger.info(
                    f"[bold cyan]Stage 2 - Query {i}/{len(atomic_queries)}:[/bold cyan] {atomic_query}"
                )
            try:
                channels = await self._navigate_hierarchy(atomic_query)
                all_channels.extend(channels)
                logger.info(f"  → Found {len(channels)} channel(s)")
            except HierarchicalNavigationError as e:
                logger.warning(f"  [yellow]⚠[/yellow] Navigation failed: {e}")
                continue
            except Exception as e:
                logger.error(f"  [red]✗[/red] Error processing query: {e}")
                continue

        # Merge explicit channels with search results and deduplicate
        all_channels.extend(explicit_channels)
        unique_channels = list(set(all_channels))

        # Build result (detailed display happens in capability layer)
        return self._build_result(query, unique_channels)

    async def _split_query(self, query: str) -> list[str]:
        """Split query into atomic sub-queries (reuse from in-context)."""
        prompt = self.query_splitter.get_prompt(facility_name=self.facility_name)
        message = f"{prompt}\n\nQuery to process: {query}"

        # Save prompt for debugging
        _save_prompt_to_file(message, stage="query_split", query=query)

        # Set caller context for API call logging (propagates through asyncio.to_thread)
        from osprey.models import set_api_call_context

        set_api_call_context(
            function="_split_query",
            module="hierarchical.pipeline",
            class_name="HierarchicalPipeline",
            extra={"stage": "query_split"},
        )

        response = await asyncio.to_thread(
            get_chat_completion,
            message=message,
            model_config=self.model_config,
            output_model=QuerySplitterOutput,
        )

        return response.queries

    async def _navigate_hierarchy(self, query: str) -> list[str]:
        """
        Navigate through hierarchy levels for a single atomic query.

        Uses recursive branching to handle multiple selections at system/family/field levels.

        Returns:
            List of fully-qualified channel names
        """
        levels = self.database.get_hierarchy_definition()

        # Start recursive navigation from root
        all_channels = await self._navigate_recursive(
            query=query,
            remaining_levels=levels,
            selections={},
            branch_path=[],
            branch_num=1,
            total_branches=1,
        )

        # Validate all channels exist
        valid_channels = [ch for ch in all_channels if self.database.validate_channel(ch)]

        if len(valid_channels) != len(all_channels):
            logger.warning(
                f"  [yellow]⚠[/yellow] {len(all_channels) - len(valid_channels)} invalid channels discarded"
            )

        return valid_channels

    async def _navigate_recursive(
        self,
        query: str,
        remaining_levels: list[str],
        selections: dict,
        branch_path: list[str],
        branch_num: int,
        total_branches: int,
    ) -> list[str]:
        """
        Recursively navigate hierarchy with automatic branching.

        Branches occur when multiple selections are made at system/family/field levels.
        Devices don't cause branching as they're structurally identical within a family.

        Args:
            query: Original user query
            remaining_levels: Levels still to navigate
            selections: Selections made so far (single values only)
            branch_path: Human-readable path for this branch (for logging)
            branch_num: Current branch number (for visualization)
            total_branches: Total number of branches at this level (for visualization)

        Returns:
            List of channel names found in this branch and all sub-branches
        """
        # Base case: no more levels to navigate
        if not remaining_levels:
            # Build channels from this complete path
            channels = self.database.build_channels_from_selections(selections)
            return channels

        # Get current level and remaining
        level = remaining_levels[0]
        next_levels = remaining_levels[1:]

        # Indent based on branch depth
        indent = "  " * (len(self.database.get_hierarchy_definition()) - len(remaining_levels) + 1)

        # Get available options at this level
        options = self.database.get_options_at_level(level, selections)

        # Get level configuration
        level_config = self.database.hierarchy_config["levels"][level]
        is_optional = level_config.get("optional", False)

        if not options:
            # NO OPTIONS AT THIS LEVEL
            # If this is an optional level, skip it and continue
            if is_optional:
                logger.info(
                    f"{indent}[yellow]No options available at level {level} - skipping optional level[/yellow]"
                )
                # If no more levels, build channels from current selections
                if not next_levels:
                    return self.database.build_channels_from_selections(selections)
                # Otherwise continue to next level
                return await self._navigate_recursive(
                    query=query,
                    remaining_levels=next_levels,
                    selections=selections,
                    branch_path=branch_path,
                    branch_num=branch_num,
                    total_branches=total_branches,
                )
            else:
                # Required level has no options - this is an error
                logger.warning(f"{indent}[yellow]No options available at level {level}[/yellow]")
                return []

        logger.info(f"{indent}Level: {level}")
        logger.info(f"{indent}  Available options: {len(options)}")

        # Make LLM selection
        selected = await self._select_at_level(query, level, options, selections)

        if not selected:
            # OPTIONAL LEVEL HANDLING: If this is an optional level and NOTHING_FOUND was returned,
            # skip this level and continue to the next level
            if is_optional:
                logger.info(f"{indent}  [yellow]→ Skipping optional level '{level}'[/yellow]")
                # Continue to next level without adding this level to selections
                return await self._navigate_recursive(
                    query=query,
                    remaining_levels=next_levels,
                    selections=selections,  # Don't add current level
                    branch_path=branch_path,
                    branch_num=branch_num,
                    total_branches=total_branches,
                )
            else:
                logger.warning(f"{indent}  [yellow]No selection made at level {level}[/yellow]")
                return []

        logger.info(
            f"{indent}  → Selected: {selected[:3] if len(selected) > 3 else selected}{'...' if len(selected) > 3 else ''}"
        )

        # DIRECT SIGNAL DETECTION AT OPTIONAL LEVELS:
        # If this is an optional level and the selection is a leaf node (direct signal),
        # treat it as if it belongs to the next level and skip the current optional level.
        if is_optional and len(selected) == 1:
            # Check if the selected option is a leaf node
            current_node = self.database._navigate_to_node(level, selections)
            if current_node:
                selected_node = current_node.get(selected[0])
                level_idx = self.database.hierarchy_levels.index(level)
                if selected_node and self.database._is_leaf_node(selected_node, level_idx + 1):
                    # This is a direct signal! Skip optional level and add to next level instead
                    logger.info(
                        f"{indent}  [cyan]→ '{selected[0]}' is a direct signal - skipping optional level '{level}'[/cyan]"
                    )
                    # Find the next non-optional level to assign this selection to
                    next_level = next_levels[0] if next_levels else None
                    if next_level:
                        new_selections = selections.copy()
                        new_selections[next_level] = selected
                        # Skip both current optional level AND the next level (since we just filled it)
                        return await self._navigate_recursive(
                            query=query,
                            remaining_levels=next_levels[1:],  # Skip next level
                            selections=new_selections,
                            branch_path=branch_path,
                            branch_num=branch_num,
                            total_branches=total_branches,
                        )

        # Determine if we should branch based on level type
        # Tree levels allow branching, instance levels don't
        allow_branching = level_config["type"] == "tree"

        should_branch = len(selected) > 1 and allow_branching

        if should_branch:
            # BRANCHING: Multiple selections at a branch-point level
            num_branches = len(selected)
            logger.info(
                f"{indent}  [cyan]⚡ Branching:[/cyan] {num_branches} {level}(s) selected - exploring each separately"
            )

            all_results = []
            for i, single_selection in enumerate(selected, 1):
                # Create new branch path for visualization
                new_branch_path = branch_path + [f"{level}={single_selection}"]
                branch_path_str = " → ".join(new_branch_path)

                # Log branch start
                logger.info(
                    f"{indent}  [bold cyan]Branch {i}/{num_branches}:[/bold cyan] {branch_path_str}"
                )

                # OPTIONAL LEVEL LEAF DETECTION IN BRANCHING:
                # For each branch, check if the selection is a leaf node at an optional level.
                # If so, skip the optional level and assign to next level instead.
                # This handles the case where multiple direct signals are selected together.
                if is_optional and next_levels:
                    current_node = self.database._navigate_to_node(level, selections)
                    if current_node:
                        selected_node = current_node.get(single_selection)
                        level_idx = self.database.hierarchy_levels.index(level)
                        if selected_node and self.database._is_leaf_node(
                            selected_node, level_idx + 1
                        ):
                            # This branch is a direct signal! Skip optional level
                            logger.info(
                                f"{indent}    [cyan]→ '{single_selection}' is a direct signal - skipping optional level '{level}'[/cyan]"
                            )
                            # Create selections that skip this optional level
                            branch_selections = selections.copy()
                            next_level = next_levels[0]
                            branch_selections[next_level] = single_selection

                            # Navigate with next level skipped (since we just filled it)
                            branch_results = await self._navigate_recursive(
                                query=query,
                                remaining_levels=next_levels[1:],
                                selections=branch_selections,
                                branch_path=new_branch_path,
                                branch_num=i,
                                total_branches=num_branches,
                            )

                            # Log branch completion
                            logger.info(
                                f"{indent}    [green]✓[/green] Branch {i}/{num_branches}: Found {len(branch_results)} channel(s)"
                            )

                            all_results.extend(branch_results)
                            continue  # Skip normal branch processing

                # Normal branch processing (not a leaf or not optional)
                branch_selections = selections.copy()
                branch_selections[level] = single_selection

                # Recursively navigate this branch
                branch_results = await self._navigate_recursive(
                    query=query,
                    remaining_levels=next_levels,
                    selections=branch_selections,
                    branch_path=new_branch_path,
                    branch_num=i,
                    total_branches=num_branches,
                )

                # Log branch completion
                logger.info(
                    f"{indent}    [green]✓[/green] Branch {i}/{num_branches}: Found {len(branch_results)} channel(s)"
                )

                all_results.extend(branch_results)

            return all_results

        else:
            # NO BRANCHING: Single selection OR device level (homogeneous)
            # Continue down the same path
            new_selections = selections.copy()
            new_selections[level] = selected

            return await self._navigate_recursive(
                query=query,
                remaining_levels=next_levels,
                selections=new_selections,
                branch_path=branch_path,
                branch_num=branch_num,
                total_branches=total_branches,
            )

    async def _select_at_level(
        self, query: str, level: str, options: list[dict], previous_selections: dict
    ) -> list[str]:
        """
        Use LLM to select option(s) at current hierarchy level.

        Args:
            query: Original atomic query
            level: Current level name
            options: Available options at this level
            previous_selections: Selections made at previous levels

        Returns:
            List of selected option names (can be multiple for wildcards).
            Empty list if NOTHING_FOUND is selected.
        """
        # Get option names for dynamic model
        option_names = [opt["name"] for opt in options]

        # Create dynamic model for this specific level
        SelectionModel = create_selection_model(level, option_names, allow_multiple=True)

        # Build prompt for level selection
        prompt = self._build_level_prompt(query, level, options, previous_selections)

        # Save prompt for debugging
        _save_prompt_to_file(prompt, stage="level_selection", level=level, query=query)

        # Set caller context for API call logging (propagates through asyncio.to_thread)
        from osprey.models import set_api_call_context

        set_api_call_context(
            function="_select_at_level",
            module="hierarchical.pipeline",
            class_name="HierarchicalPipeline",
            extra={"stage": "level_selection", "level": level},
        )

        # Get LLM response with dynamic model
        response = await asyncio.to_thread(
            get_chat_completion,
            message=prompt,
            model_config=self.model_config,
            output_model=SelectionModel,
        )

        # Extract string values from Enum objects
        selection_values = [sel.value for sel in response.selections]

        # Check if nothing was found
        if NOTHING_FOUND_MARKER in selection_values:
            logger.info(f"    [yellow]→ {NOTHING_FOUND_MARKER} - aborting search[/yellow]")
            return []

        # All selections are guaranteed to be valid by the dynamic model
        return selection_values

    def _build_level_prompt(
        self, query: str, level: str, options: list[dict], previous_selections: dict
    ) -> str:
        """Build prompt for hierarchical level selection with path-aware domain context."""
        # Format options (limit to reasonable number for prompt)
        max_options_display = 100
        display_options = options[:max_options_display]
        truncated = len(options) > max_options_display

        # Build options list with descriptions
        options_list = []
        for opt in display_options:
            opt_name = opt["name"]
            opt_desc = opt.get("description", "N/A")

            # Description is now complete (merged from _description and _detailed)
            options_list.append(f"- {opt_name}: {opt_desc}")

        options_str = "\n".join(options_list)

        if truncated:
            options_str += f"\n... and {len(options) - max_options_display} more options"

        # Format previous selections
        if previous_selections:
            path_items = []
            for k, v in previous_selections.items():
                if isinstance(v, str):
                    path_items.append(f"{k}: {v}")
                elif isinstance(v, list):
                    # Show multiple selections clearly
                    if len(v) == 1:
                        path_items.append(f"{k}: {v[0]}")
                    else:
                        path_items.append(f"{k}: [{', '.join(v)}]")
                else:
                    path_items.append(f"{k}: {v}")
            path_str = " → ".join(path_items)
        else:
            path_str = "ROOT"

        # Check if this level is optional
        level_config = self.database.hierarchy_config["levels"][level]
        is_optional = level_config.get("optional", False)

        # Get level-specific instructions if available
        level_instruction = ""
        if self.hierarchical_context and level in self.hierarchical_context:
            level_instruction = f"\n{self.hierarchical_context[level].strip()}\n"
        else:
            # Fallback guidance based on level type for arbitrary hierarchies
            if level_config["type"] == "instances":
                level_instruction = f"""
Select specific instance(s) of {level} based on the query.

Guidelines:
- Specific number/name mentioned → select that instance exactly
- "all" or no specific instance → select all available instances
- Range mentioned → select instances in that range
- Multiple instances can be selected - they share the same structure
"""
            elif level_config["type"] == "tree":
                level_instruction = f"""
Select the {level} category/categories that match the query.

BRANCHING BEHAVIOR: If you select multiple options, each will be explored
separately with its own subtree. This is important when different options
have different structures or children.

Select multiple when the query spans multiple categories or is ambiguous.
"""
            else:
                # Fallback for legacy container mode
                level_instruction = f"""
Select the {level} option that best matches the query context.
"""

        # Add optional level guidance if applicable
        optional_level_guidance = ""
        if is_optional:
            optional_level_guidance = f"""
⚠️  OPTIONAL LEVEL NOTICE:
This '{level}' level is OPTIONAL in the hierarchy and can be skipped if not relevant to the query.

IMPORTANT: Some channels exist at the parent level and skip this {level} level entirely.
If the query asks for something that is NOT specifically related to any of the {level}
options shown above, you should skip this level by selecting '{NOTHING_FOUND_MARKER}'.

WHEN TO SKIP (select '{NOTHING_FOUND_MARKER}'):
- The query asks for something that is NOT specifically mentioned in any of the {level} descriptions above
- The query does not specify any {level}-specific terminology or features
- The query explicitly asks to skip this level (e.g., "skip {level}", "directly at device level")
- None of the {level} options have descriptions that match what the user is asking for
- The user's query terms do not appear in ANY of the {level} option names or descriptions

WHEN NOT TO SKIP:
- The query explicitly mentions a specific {level} name shown in the options above
- One or more option descriptions clearly match what the query is asking for
- The query uses terms that appear in the {level} option descriptions
- The query uses general terms like "all" that could apply to multiple {level} options

CRITICAL: Read the option descriptions carefully. If what the user is asking for does NOT
appear in any of the option descriptions, it likely exists at a different level and you
should select '{NOTHING_FOUND_MARKER}' to skip this level.
"""

        # Add facility description for system-level navigation (provides device naming conventions)
        facility_context = ""
        if level == "system" and self.facility_description:
            facility_context = f"\nFACILITY CONTEXT:\n{self.facility_description}\n\n"

        # Build prompt
        prompt = f"""You are navigating a hierarchical channel database for {self.facility_name}.
{facility_context}
User Query: "{query}"

Current Level: {level}
Previous Path: {path_str}

Available Options ({len(options)} total):
{options_str}
{level_instruction}
{optional_level_guidance}
Instructions:
1. Select one or more options that best match the query
2. You can select multiple options if the query is ambiguous or uses words like "all" or "both"
3. Consider the context from previous selections and the detailed descriptions above
4. If the query mentions specific numbers or names, select only those
5. Use your understanding of the domain terminology to make accurate selections
6. If NONE of the options match the query, select '{NOTHING_FOUND_MARKER}' to abort the search

Return your selection as a JSON object with:
- selections: list of selected option names (must match exactly from the list above)

Important: Only select options that appear in the Available Options list above. Names must match exactly.
If no options are relevant, use '{NOTHING_FOUND_MARKER}'."""

        return prompt

    def _get_single_selection(self, selections: dict, key: str) -> str | None:
        """Get single selection value (handle lists by taking first element)."""
        value = selections.get(key)
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def _build_result(self, query: str, channels: list[str]) -> ChannelFinderResult:
        """
        Build final result object.

        Note: Channels passed here have already been validated according to
        explicit_validation_mode config. No additional validation needed.

        For hierarchical database: channel name = address, no descriptions stored.
        """
        channel_infos = [
            ChannelInfo(
                channel=channel_name,
                address=channel_name,
                description=None,
            )
            for channel_name in channels
        ]

        notes = (
            f"Processed query using hierarchical navigation. Found {len(channel_infos)} channels."
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
            "hierarchy_levels": db_stats.get("hierarchy_levels", []),
        }
