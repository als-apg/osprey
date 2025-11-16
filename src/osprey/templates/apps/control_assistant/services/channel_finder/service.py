"""
Service Interface for Generic Channel Finder

Provides a high-level service interface supporting multiple pipeline modes.
"""

import logging
from pathlib import Path
from .pipelines.in_context import InContextPipeline
from .pipelines.hierarchical import HierarchicalPipeline
from .databases import LegacyChannelDatabase, TemplateChannelDatabase, HierarchicalChannelDatabase
from .core.models import ChannelFinderResult
from .core.exceptions import (
    PipelineModeError,
    DatabaseLoadError,
    ConfigurationError
)
from .utils.prompt_loader import load_prompts

# Use Osprey's config system
from osprey.utils.config import _get_config, get_provider_config

logger = logging.getLogger(__name__)


class ChannelFinderService:
    """
    Unified service interface supporting multiple pipeline architectures.

    Automatically selects and initializes the appropriate pipeline based on
    configuration settings.
    """

    def __init__(
        self,
        db_path: str = None,
        model_config: dict = None,
        pipeline_mode: str = None,
        **kwargs
    ):
        """
        Initialize the Channel Finder service.

        Args:
            db_path: Path to database file (None = use config.yml)
            model_config: Model configuration dict (None = use config.yml)
            pipeline_mode: Override pipeline mode from config ('in_context' or 'hierarchical')
            **kwargs: Pipeline-specific configuration

        Raises:
            PipelineModeError: If invalid pipeline mode specified
            DatabaseLoadError: If database cannot be loaded
            ConfigurationError: If configuration is invalid
        """
        # Load configuration from Osprey
        config_builder = _get_config()
        config = config_builder.raw_config

        # Determine pipeline mode
        if pipeline_mode is None:
            pipeline_mode = config_builder.get('channel_finder.pipeline_mode', 'in_context')

        # Validate pipeline mode
        valid_modes = ['in_context', 'hierarchical']
        if pipeline_mode not in valid_modes:
            raise PipelineModeError(
                f"Invalid pipeline mode: {pipeline_mode}. "
                f"Valid options: {', '.join(valid_modes)}"
            )

        self.pipeline_mode = pipeline_mode

        # Load model config
        if model_config is None:
            model_config = self._load_model_config(config)

        self.model_config = model_config

        # Initialize appropriate pipeline
        if pipeline_mode == 'in_context':
            self.pipeline = self._init_in_context_pipeline(config, db_path, model_config, **kwargs)

        elif pipeline_mode == 'hierarchical':
            self.pipeline = self._init_hierarchical_pipeline(config, db_path, model_config, **kwargs)

        else:
            raise PipelineModeError(
                f"Unknown pipeline mode: {pipeline_mode}. "
                f"Valid options: 'in_context', 'hierarchical'"
            )

    def _load_model_config(self, config: dict) -> dict:
        """Load model configuration using Osprey's config system."""
        config_builder = _get_config()

        # Get model configuration from Osprey config
        provider = config_builder.get('model.provider', 'cborg')
        model_id = config_builder.get('model.model_id', 'anthropic/claude-haiku')
        max_tokens = config_builder.get('model.max_tokens', 4096)

        # Get provider configuration using Osprey's utility
        provider_config = get_provider_config(provider)

        return {
            'provider': provider,
            'model_id': model_id,
            'max_tokens': max_tokens,
            **provider_config
        }

    def _resolve_path(self, path_str: str) -> str:
        """Resolve path relative to project root using Osprey config."""
        config_builder = _get_config()
        project_root = Path(config_builder.get('project_root'))
        path = Path(path_str)

        if path.is_absolute():
            return str(path)
        return str(project_root / path)

    def _init_in_context_pipeline(
        self,
        config: dict,
        db_path: str,
        model_config: dict,
        **kwargs
    ):
        """Initialize in-context pipeline."""
        # Get in-context specific config
        in_context_config = config.get('channel_finder', {}).get('pipelines', {}).get('in_context', {})
        db_config = in_context_config.get('database', {})
        processing_config = in_context_config.get('processing', {})

        # Determine database path
        if db_path is None:
            db_path = db_config.get('path')
            if not db_path:
                raise ConfigurationError("No database path provided for in-context pipeline")
            db_path = self._resolve_path(db_path)

        # Load appropriate database
        db_type = db_config.get('type', 'template')
        presentation_mode = db_config.get('presentation_mode', 'explicit')

        try:
            if db_type == 'template':
                database = TemplateChannelDatabase(db_path, presentation_mode=presentation_mode)
            elif db_type == 'legacy':
                database = LegacyChannelDatabase(db_path)
            else:
                raise ConfigurationError(f"Unknown database type: {db_type}")
        except FileNotFoundError as e:
            raise DatabaseLoadError(f"Database file not found: {db_path}") from e

        # Load facility config
        facility_config = config.get('facility', {})
        facility_name = facility_config.get('name', 'control system')

        # Load facility description from loaded prompts module
        facility_description = ''
        prompts_module = load_prompts(config)
        if hasattr(prompts_module, 'system') and hasattr(prompts_module.system, 'facility_description'):
            facility_description = prompts_module.system.facility_description
            logger.info(f"[dim]✓ Loaded facility context from prompts ({len(facility_description)} chars)[/dim]")

        # Initialize pipeline
        return InContextPipeline(
            database=database,
            model_config=model_config,
            chunk_dictionary=processing_config.get('chunk_dictionary', False),
            chunk_size=processing_config.get('chunk_size', 50),
            max_correction_iterations=processing_config.get('max_correction_iterations', 2),
            facility_name=facility_name,
            facility_description=facility_description,
            **kwargs
        )

    def _init_hierarchical_pipeline(
        self,
        config: dict,
        db_path: str,
        model_config: dict,
        **kwargs
    ):
        """Initialize hierarchical pipeline."""
        # Get hierarchical specific config
        hierarchical_config = config.get('channel_finder', {}).get('pipelines', {}).get('hierarchical', {})
        db_config = hierarchical_config.get('database', {})
        processing_config = hierarchical_config.get('processing', {})

        # Determine database path
        if db_path is None:
            db_path = db_config.get('path')
            if not db_path:
                raise ConfigurationError("No database path provided for hierarchical pipeline")
            db_path = self._resolve_path(db_path)

        # Load hierarchical database
        try:
            database = HierarchicalChannelDatabase(db_path)
        except FileNotFoundError as e:
            raise DatabaseLoadError(f"Database file not found: {db_path}") from e

        # Load facility config
        facility_config = config.get('facility', {})
        facility_name = facility_config.get('name', 'control system')

        # Load facility description from loaded prompts module
        facility_description = ''
        prompts_module = load_prompts(config)
        if hasattr(prompts_module, 'system') and hasattr(prompts_module.system, 'facility_description'):
            facility_description = prompts_module.system.facility_description
            logger.info(f"[dim]✓ Loaded facility context from prompts ({len(facility_description)} chars)[/dim]")

        # Initialize pipeline
        return HierarchicalPipeline(
            database=database,
            model_config=model_config,
            facility_name=facility_name,
            facility_description=facility_description,
            **kwargs
        )

    async def find_channels(self, query: str) -> ChannelFinderResult:
        """
        Find channels based on natural language query.

        This method works with any pipeline type.

        Args:
            query: Natural language query string

        Returns:
            ChannelFinderResult with found channels and metadata
        """
        return await self.pipeline.process_query(query)

    def get_pipeline_info(self) -> dict:
        """
        Get information about the current pipeline.

        Returns:
            Dict with pipeline name, mode, and statistics
        """
        return {
            'pipeline_mode': self.pipeline_mode,
            'pipeline_name': self.pipeline.pipeline_name,
            'statistics': self.pipeline.get_statistics()
        }
