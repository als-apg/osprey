"""
Hierarchical Pipeline Implementation

Iterative navigation through structured channel hierarchy.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from ...core.base_pipeline import BasePipeline
from ...core.models import ChannelFinderResult, ChannelInfo, QuerySplitterOutput
from ...core.exceptions import HierarchicalNavigationError
from ...llm import get_chat_completion
from ...utils.prompt_loader import load_prompts
from .models import create_selection_model, NOTHING_FOUND_MARKER

# Use Osprey's config system
from osprey.utils.config import _get_config

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
    config_builder = _get_config()
    if not config_builder.get('debug.save_prompts', False):
        return

    # Get temp directory from config or use default
    prompts_dir = config_builder.get('debug.prompts_dir', 'temp_prompts')
    project_root = Path(config_builder.get('project_root'))
    temp_dir = project_root / prompts_dir
    temp_dir.mkdir(exist_ok=True, parents=True)

    # Map stage names to descriptive filenames
    if stage == 'query_split':
        filename = 'prompt_stage1_query_split.txt'
    elif stage == 'level_selection' and level:
        # Stage 2 is hierarchical navigation with 5 levels
        level_map = {
            'system': 'prompt_stage2_level1_system.txt',
            'family': 'prompt_stage2_level2_family.txt',
            'device': 'prompt_stage2_level3_device.txt',
            'field': 'prompt_stage2_level4_field.txt',
            'subfield': 'prompt_stage2_level5_subfield.txt',
        }
        filename = level_map.get(level, f'prompt_stage2_level_{level}.txt')
    else:
        filename = f'prompt_{stage}.txt'

    filepath = temp_dir / filename

    # Save prompt
    with open(filepath, 'w', encoding='utf-8') as f:
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
        **kwargs
    ):
        """
        Initialize hierarchical pipeline.

        Args:
            database: HierarchicalChannelDatabase instance
            model_config: LLM model configuration
            facility_name: Name of facility
            facility_description: Facility description for context
            **kwargs: Additional pipeline arguments
        """
        super().__init__(database, model_config, **kwargs)
        self.facility_name = facility_name
        self.facility_description = facility_description

        # Load query splitter from shared prompts
        config_builder = _get_config()
        prompts_module = load_prompts(config_builder.raw_config)
        self.query_splitter = prompts_module.query_splitter

        # Load hierarchical navigation context from prompts (not database - separation of data vs instructions)
        if hasattr(prompts_module, 'hierarchical_context'):
            self.hierarchical_context = prompts_module.hierarchical_context.get_hierarchical_context()
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
        1. Split query (if needed)
        2. For each sub-query, navigate hierarchy
        3. Build channel names from selections
        4. Aggregate results
        """
        # Handle empty query
        if not query or not query.strip():
            return ChannelFinderResult(
                query=query,
                channels=[],
                total_channels=0,
                processing_notes="Empty query provided"
            )

        # Stage 1: Split query into atomic queries
        atomic_queries = await self._split_query(query)
        logger.info(f"[bold cyan]Stage 1:[/bold cyan] Split into {len(atomic_queries)} atomic quer{'y' if len(atomic_queries) == 1 else 'ies'}")

        # Only show individual queries if there are multiple (avoid redundancy for single query)
        if len(atomic_queries) > 1:
            for i, aq in enumerate(atomic_queries, 1):
                logger.info(f"  → Query {i}: {aq}")

        # Stage 2: Navigate hierarchy for each atomic query
        all_channels = []
        for i, atomic_query in enumerate(atomic_queries, 1):
            # For single query, keep it concise since we already know the task from capability logs
            if len(atomic_queries) == 1:
                logger.info(f"[bold cyan]Stage 2:[/bold cyan] Navigating hierarchy...")
            else:
                logger.info(f"[bold cyan]Stage 2 - Query {i}/{len(atomic_queries)}:[/bold cyan] {atomic_query}")
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

        # Deduplicate
        unique_channels = list(set(all_channels))

        # Build result (detailed display happens in capability layer)
        return self._build_result(query, unique_channels)

    async def _split_query(self, query: str) -> List[str]:
        """Split query into atomic sub-queries (reuse from in-context)."""
        prompt = self.query_splitter.get_prompt(facility_name=self.facility_name)
        message = f"{prompt}\n\nQuery to process: {query}"

        # Save prompt for debugging
        _save_prompt_to_file(message, stage='query_split', query=query)

        response = await asyncio.to_thread(
            get_chat_completion,
            message=message,
            model_config=self.model_config,
            output_model=QuerySplitterOutput
        )

        return response.queries


    async def _navigate_hierarchy(self, query: str) -> List[str]:
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
            total_branches=1
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
        remaining_levels: List[str],
        selections: Dict,
        branch_path: List[str],
        branch_num: int,
        total_branches: int
    ) -> List[str]:
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

        if not options:
            logger.warning(f"{indent}[yellow]No options available at level {level}[/yellow]")
            return []

        logger.info(f"{indent}Level: {level}")
        logger.info(f"{indent}  Available options: {len(options)}")

        # Make LLM selection
        selected = await self._select_at_level(query, level, options, selections)

        if not selected:
            logger.warning(f"{indent}  [yellow]No selection made at level {level}[/yellow]")
            return []

        logger.info(f"{indent}  → Selected: {selected[:3] if len(selected) > 3 else selected}{'...' if len(selected) > 3 else ''}")

        # Determine if we should branch
        should_branch = (
            len(selected) > 1 and
            level in ['system', 'family', 'field']  # Branching levels
        )

        if should_branch:
            # BRANCHING: Multiple selections at a branch-point level
            num_branches = len(selected)
            logger.info(f"{indent}  [cyan]⚡ Branching:[/cyan] {num_branches} {level}(s) selected - exploring each separately")

            all_results = []
            for i, single_selection in enumerate(selected, 1):
                # Create new branch path for visualization
                new_branch_path = branch_path + [f"{level}={single_selection}"]
                branch_path_str = " → ".join(new_branch_path)

                # Log branch start
                logger.info(f"{indent}  [bold cyan]Branch {i}/{num_branches}:[/bold cyan] {branch_path_str}")

                # Create new selections dict with single value
                branch_selections = selections.copy()
                branch_selections[level] = single_selection

                # Recursively navigate this branch
                branch_results = await self._navigate_recursive(
                    query=query,
                    remaining_levels=next_levels,
                    selections=branch_selections,
                    branch_path=new_branch_path,
                    branch_num=i,
                    total_branches=num_branches
                )

                # Log branch completion
                logger.info(f"{indent}    [green]✓[/green] Branch {i}/{num_branches}: Found {len(branch_results)} channel(s)")

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
                total_branches=total_branches
            )

    async def _select_at_level(
        self,
        query: str,
        level: str,
        options: List[Dict],
        previous_selections: Dict
    ) -> List[str]:
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
        option_names = [opt['name'] for opt in options]

        # Create dynamic model for this specific level
        SelectionModel = create_selection_model(level, option_names, allow_multiple=True)

        # Build prompt for level selection
        prompt = self._build_level_prompt(query, level, options, previous_selections)

        # Save prompt for debugging
        _save_prompt_to_file(prompt, stage='level_selection', level=level, query=query)

        # Get LLM response with dynamic model
        response = await asyncio.to_thread(
            get_chat_completion,
            message=prompt,
            model_config=self.model_config,
            output_model=SelectionModel
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
        self,
        query: str,
        level: str,
        options: List[Dict],
        previous_selections: Dict
    ) -> str:
        """Build prompt for hierarchical level selection with path-aware domain context."""
        # Format options (limit to reasonable number for prompt)
        max_options_display = 100
        display_options = options[:max_options_display]
        truncated = len(options) > max_options_display

        # Build options list with descriptions
        options_list = []
        for opt in display_options:
            opt_name = opt['name']
            opt_desc = opt.get('description', 'N/A')

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

        # Get level-specific instructions if available
        level_instruction = ""
        if self.hierarchical_context and level in self.hierarchical_context:
            level_instruction = f"\n{self.hierarchical_context[level].strip()}\n"

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

    def _get_single_selection(self, selections: Dict, key: str) -> Optional[str]:
        """Get single selection value (handle lists by taking first element)."""
        value = selections.get(key)
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def _build_result(self, query: str, channels: List[str]) -> ChannelFinderResult:
        """Build final result object."""
        channel_infos = []

        for channel_name in channels:
            channel_data = self.database.get_channel(channel_name)
            if channel_data:
                # Hierarchical database might not have description
                channel_infos.append(ChannelInfo(
                    channel=channel_name,
                    address=channel_data.get('address', channel_name),
                    description=channel_data.get('description')
                ))

        notes = (
            f"Processed query using hierarchical navigation. "
            f"Found {len(channel_infos)} channels."
        )

        return ChannelFinderResult(
            query=query,
            channels=channel_infos,
            total_channels=len(channel_infos),
            processing_notes=notes
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Return pipeline statistics."""
        db_stats = self.database.get_statistics()
        return {
            'total_channels': db_stats.get('total_channels', 0),
            'hierarchy_levels': db_stats.get('hierarchy_levels', [])
        }

