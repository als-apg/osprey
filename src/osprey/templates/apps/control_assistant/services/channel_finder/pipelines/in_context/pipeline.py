"""
Processing Pipeline for Generic Channel Finder

Implements the async multi-stage processing pipeline with chunking support.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from ...core.base_pipeline import BasePipeline
from ...core.models import (
    QuerySplitterOutput,
    ChannelMatchOutput,
    ChannelCorrectionOutput,
    ChannelFinderResult,
    ChannelInfo
)
from ...llm import get_chat_completion
from ...utils.prompt_loader import load_prompts

# Use Osprey's config system
from osprey.utils.config import _get_config

logger = logging.getLogger(__name__)


def _save_prompt_to_file(prompt: str, stage: str, query: str = "", chunk_num: int = None):
    """Save prompt to temporary file for inspection.
    
    Only saves if debug.save_prompts is enabled in config.yml.
    
    Args:
        prompt: The prompt text to save
        stage: Stage name (e.g., "query_split", "channel_match", "correction")
        query: Optional query identifier
        chunk_num: Optional chunk number
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
    stage_filename_map = {
        'query_split': 'prompt_stage1_query_split',
        'channel_match': 'prompt_stage2_channel_match',
        'correction': 'prompt_stage3_correction'
    }
    
    # Get descriptive filename or use stage name as fallback
    base_filename = stage_filename_map.get(stage, stage)
    
    # Build simple filename (latest will overwrite)
    if chunk_num is not None:
        filename = f"{base_filename}_chunk{chunk_num}.txt"
    else:
        filename = f"{base_filename}.txt"
    
    filepath = temp_dir / filename
    
    # Save prompt
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"=== STAGE: {stage.upper()} ===\n")
        f.write(f"=== TIMESTAMP: {datetime.now().isoformat()} ===\n")
        if query:
            f.write(f"=== QUERY: {query} ===\n")
        if chunk_num is not None:
            f.write(f"=== CHUNK: {chunk_num} ===\n")
        f.write("=" * 80 + "\n\n")
        f.write(prompt)
    
    logger.debug(f"  [dim]Saved prompt to: {filepath}[/dim]")


class InContextPipeline(BasePipeline):
    """In-context semantic search pipeline."""
    
    def __init__(
        self,
        database,
        model_config: dict,
        chunk_dictionary: bool = False,
        chunk_size: int = 50,
        max_correction_iterations: int = 2,
        facility_name: str = "control system",
        facility_description: str = "",
        **kwargs
    ):
        """
        Initialize the channel finder pipeline.
        
        Args:
            database: ChannelDatabase instance
            model_config: Model configuration dict (provider, model_id, api_key, etc.)
            chunk_dictionary: Whether to use chunked processing
            chunk_size: Number of channels per chunk (if chunked)
            max_correction_iterations: Maximum correction attempts for invalid channels
            facility_name: Name of the facility (e.g., "UCSB FEL", "ALS")
            facility_description: Optional facility context for better matching
            **kwargs: Additional pipeline-specific arguments
        """
        super().__init__(database, model_config, **kwargs)
        self.max_correction_iterations = max_correction_iterations
        self.chunk_dictionary = chunk_dictionary
        self.chunk_size = chunk_size
        self.facility_name = facility_name
        self.facility_description = facility_description
        
        # Load prompts dynamically based on configuration
        config_builder = _get_config()
        prompts_module = load_prompts(config_builder.raw_config)
        self.query_splitter = prompts_module.query_splitter
        self.channel_matcher = prompts_module.channel_matcher
        self.correction = prompts_module.correction
        
        # Save database presentation preview if debug mode is enabled
        self._save_database_preview_if_debug()
    
    @property
    def pipeline_name(self) -> str:
        """Return the pipeline name."""
        return "In-Context Semantic Search"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return pipeline statistics."""
        db_stats = self.database.get_statistics()
        return {
            'total_channels': db_stats.get('total_channels', 0),
            'chunk_mode': self.chunk_dictionary,
            'chunk_size': self.chunk_size if self.chunk_dictionary else 'N/A',
            'presentation_mode': getattr(self.database, 'presentation_mode', 'N/A'),
            'database_format': db_stats.get('format', 'unknown')
        }
    
    async def process_query(self, query: str) -> ChannelFinderResult:
        """Execute the complete pipeline with chunking as outer loop.
        
        Args:
            query: Natural language query string
            
        Returns:
            ChannelFinderResult with found channels and metadata
        """
        logger.info(f"[cyan]Query:[/cyan] {query}")
        
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
        for i, aq in enumerate(atomic_queries, 1):
            logger.debug(f"  → Query {i}: {aq}")
        
        # Prepare chunks (single chunk if not chunked mode)
        if self.chunk_dictionary:
            chunks = self.database.chunk_database(self.chunk_size)
            logger.info(f"[bold cyan]Stage 2:[/bold cyan] Chunked mode - {len(chunks)} chunks × {self.chunk_size} channels")
        else:
            chunks = [self.database.get_all_channels()]  # Full DB as single chunk
            logger.info(f"[bold cyan]Stage 2:[/bold cyan] Full database mode - {len(chunks[0])} channels")
        
        # Process each chunk through complete pipeline
        all_valid_channels = []
        for chunk_idx, chunk in enumerate(chunks, 1):
            logger.debug(f"  Processing chunk {chunk_idx}/{len(chunks)}...")
            chunk_valid_channels = await self._process_chunk(
                atomic_queries, chunk, chunk_idx
            )
            all_valid_channels.extend(chunk_valid_channels)
            logger.debug(f"  → Found {len(chunk_valid_channels)} valid channel(s) in chunk {chunk_idx}")
        
        # Stage 4: Aggregate and format (no deduplication - chunks are disjoint)
        result = self._aggregate_results(query, all_valid_channels)
        logger.info(f"[bold green]Result:[/bold green] {result.total_channels} channel(s) found")
        
        return result
    
    async def _process_chunk(
        self, 
        atomic_queries: List[str], 
        chunk: List[Dict],
        chunk_num: int = 1
    ) -> List[str]:
        """Process all atomic queries against a single chunk.
        
        Args:
            atomic_queries: List of atomic query strings
            chunk: List of channel dictionaries for this chunk
            chunk_num: Chunk number for logging/debugging
            
        Returns:
            List of valid channel names from this chunk
        """
        # Stage 2: Match all atomic queries against this chunk
        chunk_channels = await self._match_queries_in_chunk(atomic_queries, chunk, chunk_num)
        
        # Stage 3: Validate and correct within this chunk
        valid_channels = await self._validate_and_correct_chunk(
            atomic_queries, chunk_channels, chunk, chunk_num
        )
        
        return valid_channels
    
    async def _split_query(self, query: str) -> List[str]:
        """Stage 1: Split query into atomic sub-queries.
        
        Args:
            query: Original user query
            
        Returns:
            List of atomic query strings
        """
        prompt = self.query_splitter.get_prompt(facility_name=self.facility_name)
        message = f"{prompt}\n\nQuery to process: {query}"
        
        # Save prompt for inspection
        _save_prompt_to_file(message, "query_split", query)
        
        response = await asyncio.to_thread(
            get_chat_completion,
            message=message,
            model_config=self.model_config,
            output_model=QuerySplitterOutput
        )
        
        return response.queries
    
    async def _match_queries_in_chunk(
        self, 
        atomic_queries: List[str], 
        chunk: List[Dict],
        chunk_num: int = 1
    ) -> List[str]:
        """Stage 2: Match all atomic queries against a single chunk.
        
        Args:
            atomic_queries: List of atomic query strings
            chunk: List of channel dictionaries for this chunk
            chunk_num: Chunk number for debugging
            
        Returns:
            Deduplicated list of channel names found in this chunk
        """
        # Format chunk once for all queries
        chunk_formatted = self.database.format_chunk_for_prompt(
            chunk, include_addresses=False
        )
        
        # Process each atomic query sequentially
        all_channels = []
        for i, query in enumerate(atomic_queries, 1):
            try:
                logger.debug(f"  Matching query {i}/{len(atomic_queries)}: [dim]{query}[/dim]")
                result = await self._match_single_query_in_chunk(query, chunk_formatted, chunk_num)
                if result.channels_found:
                    preview = ', '.join(result.channels[:3])
                    if len(result.channels) > 3:
                        preview += f' +{len(result.channels) - 3} more'
                    logger.debug(f"    [green]✓[/green] {len(result.channels)} match(es): [dim]{preview}[/dim]")
                    all_channels.extend(result.channels)
                else:
                    logger.debug(f"    [dim]No matches[/dim]")
            except Exception as e:
                # Log error but continue processing other queries
                logger.warning(f"[yellow]⚠[/yellow] Query failed: {query} - {e}")
                continue
        
        # Deduplicate channels (same query might match same channel)
        unique_channels = []
        seen = set()
        for channel in all_channels:
            if channel not in seen:
                unique_channels.append(channel)
                seen.add(channel)
        
        if len(all_channels) != len(unique_channels):
            logger.debug(f"  Removed {len(all_channels) - len(unique_channels)} duplicate(s) → {len(unique_channels)} unique")
        
        return unique_channels
    
    async def _match_single_query_in_chunk(
        self, 
        atomic_query: str, 
        chunk_formatted: str,
        chunk_num: int = 1
    ) -> ChannelMatchOutput:
        """Match a single atomic query against formatted chunk.
        
        Args:
            atomic_query: Atomic query string
            chunk_formatted: Formatted channel database string
            chunk_num: Chunk number for debugging
            
        Returns:
            ChannelMatchOutput with match results
        """
        prompt = self.channel_matcher.get_prompt(
            atomic_query, 
            chunk_formatted,
            facility_name=self.facility_name,
            facility_description=self.facility_description
        )
        
        # Save prompt for inspection
        _save_prompt_to_file(prompt, "channel_match", atomic_query, chunk_num)
        
        response = await asyncio.to_thread(
            get_chat_completion,
            message=prompt,
            model_config=self.model_config,
            output_model=ChannelMatchOutput
        )
        
        return response
    
    async def _validate_and_correct_chunk(
        self,
        atomic_queries: List[str],
        channels: List[str],
        chunk: List[Dict],
        chunk_num: int = 1
    ) -> List[str]:
        """Stage 3: Validate channels against chunk and correct if needed.
        
        Args:
            atomic_queries: Original queries for context
            channels: Channel names to validate
            chunk: Current database chunk
            chunk_num: Chunk number for debugging
            
        Returns:
            List of valid channel names only
        """
        if not channels:
            logger.debug("  [dim]No channels to validate[/dim]")
            return []
        
        logger.debug(f"  Validating {len(channels)} channel(s)...")
        
        # Create temporary channel map for this chunk
        chunk_channel_map = {ch['channel']: ch for ch in chunk}
        
        # Validate all channels
        validation_results = []
        for channel_name in channels:
            validation_results.append({
                'channel': channel_name,
                'valid': channel_name in chunk_channel_map
            })
        
        # Check if any invalid channels exist
        has_invalid = any(not entry['valid'] for entry in validation_results)
        valid_count = sum(1 for r in validation_results if r['valid'])
        invalid_count = len(validation_results) - valid_count
        
        if not has_invalid:
            # All valid, return immediately
            logger.debug(f"    [green]✓[/green] All valid")
            return channels
        
        logger.info(f"  [yellow]⚠[/yellow] Found {invalid_count} invalid channel(s) - attempting correction...")
        
        # Attempt correction with full context
        for iteration in range(self.max_correction_iterations):
            logger.debug(f"    Correction attempt {iteration + 1}/{self.max_correction_iterations}")
            corrected = await self._correct_channels_with_context(
                atomic_queries, validation_results, chunk, chunk_num
            )
            
            # Re-validate corrected channels
            validation_results = []
            for channel_name in corrected.corrected_channels:
                validation_results.append({
                    'channel': channel_name,
                    'valid': channel_name in chunk_channel_map
                })
            
            # Check if all now valid
            has_invalid = any(not entry['valid'] for entry in validation_results)
            valid_count = sum(1 for r in validation_results if r['valid'])
            invalid_count = len(validation_results) - valid_count
            
            if not has_invalid:
                logger.info(f"  [green]✓[/green] Correction successful - all {valid_count} channel(s) valid")
                break
            else:
                logger.debug(f"      Still {invalid_count} invalid")
        
        # Return only valid channels
        valid_channels = [
            entry['channel'] 
            for entry in validation_results 
            if entry['valid']
        ]
        
        if invalid_count > 0:
            logger.warning(f"  [yellow]⚠[/yellow] {invalid_count} channel(s) remain invalid after correction")
        
        return valid_channels
    
    async def _correct_channels_with_context(
        self,
        atomic_queries: List[str],
        validation_results: List[Dict],
        chunk: List[Dict],
        chunk_num: int = 1
    ) -> ChannelCorrectionOutput:
        """Correct channels using full context (queries + validation flags + chunk).
        
        Args:
            atomic_queries: Original queries for context
            validation_results: List of {channel, valid} dicts
            chunk: Current database chunk
            chunk_num: Chunk number for debugging
            
        Returns:
            ChannelCorrectionOutput with corrected channels
        """
        # Format chunk for prompt
        chunk_formatted = self.database.format_chunk_for_prompt(
            chunk, include_addresses=False
        )
        
        prompt = self.correction.get_prompt(
            atomic_queries, validation_results, chunk_formatted,
            facility_name=self.facility_name
        )
        
        # Save prompt for inspection
        query_str = ", ".join(atomic_queries[:2])  # First 2 queries for filename
        _save_prompt_to_file(prompt, "correction", query_str, chunk_num)
        
        response = await asyncio.to_thread(
            get_chat_completion,
            message=prompt,
            model_config=self.model_config,
            output_model=ChannelCorrectionOutput
        )
        
        return response
    
    def _aggregate_results(
        self, query: str, valid_channels: List[str]
    ) -> ChannelFinderResult:
        """Stage 4: Aggregate results and map to addresses.
        
        No deduplication needed - chunks are disjoint, so each channel
        appears in exactly one chunk.
        
        Args:
            query: Original user query
            valid_channels: List of valid channel names
            
        Returns:
            ChannelFinderResult with complete information
        """
        channel_infos = []
        
        for channel_name in valid_channels:
            channel_data = self.database.get_channel(channel_name)
            if channel_data:
                channel_infos.append(ChannelInfo(
                    channel=channel_data['channel'],
                    address=channel_data['address'],
                    description=channel_data.get('description')
                ))
        
        # Enhanced processing notes
        mode = "chunked" if self.chunk_dictionary else "full dictionary"
        notes = (
            f"Processed query in {mode} mode. "
            f"Found {len(channel_infos)} channels matching the query."
        )
        
        return ChannelFinderResult(
            query=query,
            channels=channel_infos,
            total_channels=len(channel_infos),
            processing_notes=notes
        )
    
    def _save_database_preview_if_debug(self):
        """Save formatted database preview if debug mode is enabled."""
        config_builder = _get_config()
        
        # Check if debug mode and save_prompts are enabled
        if not config_builder.get('debug.save_prompts', False):
            return
        
        try:
            # Get output directory
            prompts_dir = config_builder.get('debug.prompts_dir', 'temp_prompts')
            project_root = Path(config_builder.get('project_root'))
            output_dir = project_root / prompts_dir
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Get all channels
            all_channels = self.database.get_all_channels()
            
            # Save full database preview
            full_preview = self.database.format_chunk_for_prompt(
                all_channels, include_addresses=False
            )
            
            preview_path = output_dir / 'db_llm_format_runtime.txt'
            with open(preview_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("DATABASE PRESENTATION (as sent to LLM)\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total Channels: {len(all_channels)}\n")
                f.write(f"Chunked Mode: {self.chunk_dictionary}\n")
                if self.chunk_dictionary:
                    f.write(f"Chunk Size: {self.chunk_size}\n")
                f.write("=" * 80 + "\n\n")
                f.write(full_preview)
            
            logger.debug(f"Saved database presentation preview to: {preview_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save database preview: {e}")

