"""
Service Interface for Generic Channel Finder

Provides a high-level service interface supporting multiple pipeline modes.
Supports pluggable custom pipelines and databases via registration.
"""

import logging
from pathlib import Path

# Use Osprey's config system
from osprey.utils.config import get_config_builder, get_provider_config

from .core.base_database import BaseDatabase
from .core.base_pipeline import BasePipeline
from .core.exceptions import ConfigurationError, DatabaseLoadError, PipelineModeError
from .core.models import ChannelFinderResult
from .databases import (
    FlatChannelDatabase,
    HierarchicalChannelDatabase,
    MiddleLayerDatabase,
    TemplateChannelDatabase,
)
from .pipelines.hierarchical import HierarchicalPipeline
from .pipelines.in_context import InContextPipeline
from .pipelines.middle_layer import MiddleLayerPipeline
from .utils.prompt_loader import load_prompts

logger = logging.getLogger(__name__)


class ChannelFinderService:
    """
    Unified service interface supporting multiple pipeline architectures.

    Automatically selects and initializes the appropriate pipeline based on
    configuration settings. Supports custom pipelines and databases via
    the registration system.

    Custom Extensions:
        Register custom pipelines and databases before instantiating the service:

        >>> from my_facility.pipelines.rag import RAGPipeline
        >>> from my_facility.databases.postgres import PostgreSQLDatabase
        >>>
        >>> # Register custom implementations
        >>> ChannelFinderService.register_pipeline('rag', RAGPipeline)
        >>> ChannelFinderService.register_database('postgres', PostgreSQLDatabase)
        >>>
        >>> # Now use in config.yml:
        >>> # channel_finder:
        >>> #   pipeline_mode: "rag"
        >>> #   pipelines:
        >>> #     rag:
        >>> #       database:
        >>> #         type: "postgres"
        >>> #         path: "config/postgres.json"

    Built-in Pipelines:
        - in_context: Semantic search with full database context
        - hierarchical: Iterative navigation through hierarchy

    Built-in Databases:
        - flat: Simple flat list format (base implementation for in-context databases)
        - template: Template-based expansion with dual presentation (extends flat)
        - hierarchical: Tree structure for large hierarchies
        - legacy: Alias for 'flat' (backward compatibility)
    """

    # Class-level registries for custom implementations
    _custom_pipelines: dict[str, type[BasePipeline]] = {}
    _custom_databases: dict[str, type[BaseDatabase]] = {}

    @classmethod
    def register_pipeline(cls, name: str, pipeline_class: type[BasePipeline]) -> None:
        """
        Register a custom pipeline implementation.

        Allows users to add custom channel finding pipelines (e.g., RAG-based,
        hybrid approaches, vector search) without modifying the channel finder
        source code.

        Args:
            name: Unique identifier for the pipeline (used in config.yml)
            pipeline_class: Class implementing BasePipeline interface

        Example:
            >>> from my_facility.pipelines.rag import RAGChannelPipeline
            >>>
            >>> # Register custom pipeline
            >>> ChannelFinderService.register_pipeline('rag', RAGChannelPipeline)
            >>>
            >>> # Use in config.yml:
            >>> # channel_finder:
            >>> #   pipeline_mode: "rag"
            >>> #   pipelines:
            >>> #     rag:
            >>> #       database:
            >>> #         type: "template"
            >>> #         path: "data/channels.json"
            >>> #       top_k: 20
            >>> #       embedding_model: "text-embedding-3-small"

        Note:
            Registration should happen before service instantiation, typically
            in your application's __init__.py or registry module.
        """
        if not issubclass(pipeline_class, BasePipeline):
            raise TypeError(
                f"Pipeline class must inherit from BasePipeline. Got: {pipeline_class.__name__}"
            )

        cls._custom_pipelines[name] = pipeline_class
        logger.info(f"Registered custom pipeline: {name} ({pipeline_class.__name__})")

    @classmethod
    def register_database(cls, name: str, database_class: type[BaseDatabase]) -> None:
        """
        Register a custom database implementation.

        Allows users to add custom database backends (e.g., PostgreSQL, MongoDB,
        REST API, GraphQL) without modifying the channel finder source code.

        Args:
            name: Unique identifier for the database (used in config.yml)
            database_class: Class implementing BaseDatabase interface

        Example:
            >>> from my_facility.databases.postgres import PostgreSQLChannelDatabase
            >>>
            >>> # Register custom database
            >>> ChannelFinderService.register_database('postgres', PostgreSQLChannelDatabase)
            >>>
            >>> # Use in config.yml:
            >>> # channel_finder:
            >>> #   pipeline_mode: "in_context"
            >>> #   pipelines:
            >>> #     in_context:
            >>> #       database:
            >>> #         type: "postgres"
            >>> #         path: "config/postgres_config.json"
            >>> #         connection_string: "postgresql://..."

        Note:
            The database class must implement all abstract methods from BaseDatabase.
            For in_context pipeline, also implement: chunk_database(), format_chunk_for_prompt().
            For hierarchical pipeline, also implement: get_hierarchy_definition(),
            get_options_at_level(), build_channels_from_selections().
        """
        if not issubclass(database_class, BaseDatabase):
            raise TypeError(
                f"Database class must inherit from BaseDatabase. Got: {database_class.__name__}"
            )

        cls._custom_databases[name] = database_class
        logger.info(f"Registered custom database: {name} ({database_class.__name__})")

    @classmethod
    def list_available_pipelines(cls) -> dict[str, str]:
        """
        List all available pipelines (built-in + custom).

        Returns:
            Dict mapping pipeline names to descriptions
        """
        pipelines = {
            "in_context": "Built-in: Semantic search with full database context",
            "hierarchical": "Built-in: Iterative navigation through hierarchy",
            "middle_layer": "Built-in: React agent with database query tools (MML style)",
        }

        for name, pipeline_class in cls._custom_pipelines.items():
            desc = getattr(pipeline_class, "__doc__", "Custom pipeline")
            desc = desc.split("\n")[0] if desc else "Custom pipeline"
            pipelines[name] = f"Custom: {desc.strip()}"

        return pipelines

    @classmethod
    def list_available_databases(cls) -> dict[str, str]:
        """
        List all available database types (built-in + custom).

        Returns:
            Dict mapping database names to descriptions
        """
        databases = {
            "flat": "Built-in: Simple flat list format (base implementation)",
            "legacy": "Built-in: Alias for 'flat' (backward compatibility)",
            "template": "Built-in: Template-based expansion with dual presentation",
            "hierarchical": "Built-in: Tree structure for large hierarchies",
            "middle_layer": "Built-in: MML functional hierarchy (System→Family→Field)",
        }

        for name, db_class in cls._custom_databases.items():
            desc = getattr(db_class, "__doc__", "Custom database")
            desc = desc.split("\n")[0] if desc else "Custom database"
            databases[name] = f"Custom: {desc.strip()}"

        return databases

    def __init__(
        self, db_path: str = None, model_config: dict = None, pipeline_mode: str = None, **kwargs
    ):
        """
        Initialize the Channel Finder service.

        Args:
            db_path: Path to database file (None = use config.yml)
            model_config: Model configuration dict (None = use config.yml)
            pipeline_mode: Override pipeline mode from config ('in_context', 'hierarchical', or custom)
            **kwargs: Pipeline-specific configuration

        Raises:
            PipelineModeError: If invalid pipeline mode specified
            DatabaseLoadError: If database cannot be loaded
            ConfigurationError: If configuration is invalid
        """
        # Load configuration from Osprey
        config_builder = get_config_builder()
        config = config_builder.raw_config

        # Determine pipeline mode
        if pipeline_mode is None:
            pipeline_mode = config_builder.get("channel_finder.pipeline_mode", "in_context")

        self.pipeline_mode = pipeline_mode

        # Load model config
        if model_config is None:
            model_config = self._load_model_config(config)

        self.model_config = model_config

        # Initialize appropriate pipeline (custom first, then built-in)
        if pipeline_mode in self._custom_pipelines:
            # Custom pipeline
            self.pipeline = self._init_custom_pipeline(
                pipeline_mode, config, db_path, model_config, **kwargs
            )

        elif pipeline_mode == "in_context":
            # Built-in in-context pipeline
            self.pipeline = self._init_in_context_pipeline(config, db_path, model_config, **kwargs)

        elif pipeline_mode == "hierarchical":
            # Built-in hierarchical pipeline
            self.pipeline = self._init_hierarchical_pipeline(
                config, db_path, model_config, **kwargs
            )

        elif pipeline_mode == "middle_layer":
            # Built-in middle layer pipeline
            self.pipeline = self._init_middle_layer_pipeline(
                config, db_path, model_config, **kwargs
            )

        else:
            # Unknown pipeline
            available = list(self.list_available_pipelines().keys())
            raise PipelineModeError(
                f"Unknown pipeline mode: '{pipeline_mode}'. "
                f"Available pipelines: {', '.join(available)}"
            )

    def _load_model_config(self, config: dict) -> dict:
        """Load model configuration using Osprey's config system.

        Raises:
            ConfigurationError: If channel_finder model is not configured
        """
        from osprey.utils.config import get_model_config

        # Use Osprey's standard model config utility
        # This returns the entire config dict with all parameters
        model_config = get_model_config("channel_finder")

        # Validate that configuration exists
        if not model_config:
            raise ConfigurationError(
                "Channel finder model not configured. "
                "Please add 'channel_finder' to the 'models' section in config.yml:\n\n"
                "models:\n"
                "  channel_finder:\n"
                "    provider: openai  # or anthropic, cborg, etc.\n"
                "    model_id: gpt-4   # your preferred model\n"
                "    max_tokens: 4096\n"
            )

        # Validate required fields
        if "provider" not in model_config or "model_id" not in model_config:
            raise ConfigurationError(
                "Channel finder model configuration incomplete. "
                f"Missing required fields: provider={model_config.get('provider')}, "
                f"model_id={model_config.get('model_id')}. "
                "Please ensure both 'provider' and 'model_id' are set in config.yml"
            )

        # Get provider configuration and merge
        provider_config = get_provider_config(model_config["provider"])

        # Return complete config (preserves all parameters from config.yml)
        return {**model_config, **provider_config}

    def _resolve_path(self, path_str: str) -> str:
        """Resolve path relative to project root using Osprey config."""
        config_builder = get_config_builder()
        project_root = Path(config_builder.get("project_root"))
        path = Path(path_str)

        if path.is_absolute():
            return str(path)
        return str(project_root / path)

    def _init_in_context_pipeline(self, config: dict, db_path: str, model_config: dict, **kwargs):
        """Initialize in-context pipeline."""
        # Get in-context specific config
        in_context_config = (
            config.get("channel_finder", {}).get("pipelines", {}).get("in_context", {})
        )
        db_config = in_context_config.get("database", {})
        processing_config = in_context_config.get("processing", {})

        # Determine database source and path
        source = db_config.get("source")
        if source == "google_sheets":
            spreadsheet_id = db_config.get("spreadsheet_id")
            if not spreadsheet_id:
                raise ConfigurationError(
                    "Google Sheets database requires 'spreadsheet_id' in database config"
                )

            try:
                from .databases.google_sheets import GoogleSheetsChannelDatabase
            except ImportError as e:
                raise ConfigurationError(
                    "Google Sheets database requires gspread. "
                    "Install it with: pip install osprey-framework[sheets]"
                ) from e

            database = GoogleSheetsChannelDatabase(
                spreadsheet_id=spreadsheet_id,
                worksheet=db_config.get("worksheet"),
                credentials_path=db_config.get("credentials_path"),
            )
        else:
            if db_path is None:
                db_path = db_config.get("path")
                if not db_path:
                    raise ConfigurationError("No database path provided for in-context pipeline")
                db_path = self._resolve_path(db_path)

            # Load appropriate database (custom first, then built-in)
            db_type = db_config.get("type", "template")
            presentation_mode = db_config.get("presentation_mode", "explicit")

            try:
                if db_type in self._custom_databases:
                    # Custom database - pass full db_config for custom parameters
                    database_class = self._custom_databases[db_type]
                    database = database_class(db_path, **db_config)
                    logger.info(f"Loaded custom database: {db_type} ({database_class.__name__})")

                elif db_type == "template":
                    database = TemplateChannelDatabase(db_path, presentation_mode=presentation_mode)

                elif db_type in ("flat", "legacy"):
                    # 'legacy' is an alias for 'flat' (backward compatibility)
                    database = FlatChannelDatabase(db_path)

                else:
                    available = list(self.list_available_databases().keys())
                    raise ConfigurationError(
                        f"Unknown database type: '{db_type}'. "
                        f"Available databases: {', '.join(available)}"
                    )
            except FileNotFoundError as e:
                raise DatabaseLoadError(f"Database file not found: {db_path}") from e

        # Load facility config
        facility_config = config.get("facility", {})
        facility_name = facility_config.get("name", "control system")

        # Load facility description: try framework prompts first, fall back to app prompts
        facility_description = ""
        try:
            from osprey.prompts.loader import get_framework_prompts

            prompt_builder = get_framework_prompts().get_channel_finder_in_context_prompt_builder()
            facility_description = prompt_builder.get_combined_description()
            logger.info(
                f"[dim]✓ Loaded facility context from framework prompts ({len(facility_description)} chars)[/dim]"
            )
        except (ValueError, NotImplementedError):
            # Fall back to app-level prompts module
            prompts_module = load_prompts(config)
            if hasattr(prompts_module, "system") and hasattr(
                prompts_module.system, "facility_description"
            ):
                facility_description = prompts_module.system.facility_description
                logger.info(
                    f"[dim]✓ Loaded facility context from prompts ({len(facility_description)} chars)[/dim]"
                )

        # Initialize pipeline
        return InContextPipeline(
            database=database,
            model_config=model_config,
            chunk_dictionary=processing_config.get("chunk_dictionary", False),
            chunk_size=processing_config.get("chunk_size", 50),
            max_correction_iterations=processing_config.get("max_correction_iterations", 2),
            facility_name=facility_name,
            facility_description=facility_description,
            **kwargs,
        )

    def _init_hierarchical_pipeline(self, config: dict, db_path: str, model_config: dict, **kwargs):
        """Initialize hierarchical pipeline."""
        # Get hierarchical specific config
        hierarchical_config = (
            config.get("channel_finder", {}).get("pipelines", {}).get("hierarchical", {})
        )
        db_config = hierarchical_config.get("database", {})
        hierarchical_config.get("processing", {})

        # Determine database path
        if db_path is None:
            db_path = db_config.get("path")
            if not db_path:
                raise ConfigurationError("No database path provided for hierarchical pipeline")
            db_path = self._resolve_path(db_path)

        # Load hierarchical database (custom first, then built-in)
        db_type = db_config.get("type", "hierarchical")

        try:
            if db_type in self._custom_databases:
                # Custom database - pass full db_config for custom parameters
                database_class = self._custom_databases[db_type]
                database = database_class(db_path, **db_config)
                logger.info(f"Loaded custom database: {db_type} ({database_class.__name__})")

            elif db_type == "hierarchical":
                database = HierarchicalChannelDatabase(db_path)

            else:
                available = list(self.list_available_databases().keys())
                raise ConfigurationError(
                    f"Unknown database type: '{db_type}'. "
                    f"Available databases: {', '.join(available)}. "
                    f"Hierarchical pipeline typically uses 'hierarchical' database type."
                )
        except FileNotFoundError as e:
            raise DatabaseLoadError(f"Database file not found: {db_path}") from e

        # Load facility config
        facility_config = config.get("facility", {})
        facility_name = facility_config.get("name", "control system")

        # Load facility description: try framework prompts first, fall back to app prompts
        facility_description = ""
        try:
            from osprey.prompts.loader import get_framework_prompts

            prompt_builder = (
                get_framework_prompts().get_channel_finder_hierarchical_prompt_builder()
            )
            facility_description = prompt_builder.get_combined_description()
            logger.info(
                f"[dim]✓ Loaded facility context from framework prompts ({len(facility_description)} chars)[/dim]"
            )
        except (ValueError, NotImplementedError):
            # Fall back to app-level prompts module
            prompts_module = load_prompts(config)
            if hasattr(prompts_module, "system") and hasattr(
                prompts_module.system, "facility_description"
            ):
                facility_description = prompts_module.system.facility_description
                logger.info(
                    f"[dim]✓ Loaded facility context from prompts ({len(facility_description)} chars)[/dim]"
                )

        # Initialize pipeline
        return HierarchicalPipeline(
            database=database,
            model_config=model_config,
            facility_name=facility_name,
            facility_description=facility_description,
            **kwargs,
        )

    def _init_middle_layer_pipeline(self, config: dict, db_path: str, model_config: dict, **kwargs):
        """Initialize middle layer pipeline."""
        # Get middle_layer specific config
        middle_layer_config = (
            config.get("channel_finder", {}).get("pipelines", {}).get("middle_layer", {})
        )
        db_config = middle_layer_config.get("database", {})

        # Determine database path
        if db_path is None:
            db_path = db_config.get("path")
            if not db_path:
                raise ConfigurationError("No database path provided for middle_layer pipeline")
            db_path = self._resolve_path(db_path)

        # Load middle_layer database (custom first, then built-in)
        db_type = db_config.get("type", "middle_layer")

        try:
            if db_type in self._custom_databases:
                # Custom database - pass full db_config for custom parameters
                database_class = self._custom_databases[db_type]
                database = database_class(db_path, **db_config)
                logger.info(f"Loaded custom database: {db_type} ({database_class.__name__})")

            elif db_type == "middle_layer":
                database = MiddleLayerDatabase(db_path)

            else:
                available = list(self.list_available_databases().keys())
                raise ConfigurationError(
                    f"Unknown database type: '{db_type}'. "
                    f"Available databases: {', '.join(available)}. "
                    f"Middle layer pipeline typically uses 'middle_layer' database type."
                )
        except FileNotFoundError as e:
            raise DatabaseLoadError(f"Database file not found: {db_path}") from e

        # Load facility config
        facility_config = config.get("facility", {})
        facility_name = facility_config.get("name", "control system")

        # Load facility description: try framework prompts first, fall back to app prompts
        facility_description = ""
        try:
            from osprey.prompts.loader import get_framework_prompts

            prompt_builder = (
                get_framework_prompts().get_channel_finder_middle_layer_prompt_builder()
            )
            facility_description = prompt_builder.get_combined_description()
            logger.info(
                f"[dim]✓ Loaded facility context from framework prompts ({len(facility_description)} chars)[/dim]"
            )
        except (ValueError, NotImplementedError):
            # Fall back to app-level prompts module
            prompts_module = load_prompts(config)
            if hasattr(prompts_module, "system") and hasattr(
                prompts_module.system, "facility_description"
            ):
                facility_description = prompts_module.system.facility_description
                logger.info(
                    f"[dim]✓ Loaded facility context from prompts ({len(facility_description)} chars)[/dim]"
                )

        # Initialize pipeline
        return MiddleLayerPipeline(
            database=database,
            model_config=model_config,
            facility_name=facility_name,
            facility_description=facility_description,
            **kwargs,
        )

    def _init_custom_pipeline(
        self, pipeline_name: str, config: dict, db_path: str, model_config: dict, **kwargs
    ):
        """Initialize custom registered pipeline."""
        pipeline_class = self._custom_pipelines[pipeline_name]

        # Get pipeline-specific config
        pipeline_config = (
            config.get("channel_finder", {}).get("pipelines", {}).get(pipeline_name, {})
        )
        db_config = pipeline_config.get("database", {})
        processing_config = pipeline_config.get("processing", {})

        # Determine database path
        if db_path is None:
            db_path = db_config.get("path")
            if not db_path:
                raise ConfigurationError(f"No database path provided for {pipeline_name} pipeline")
            db_path = self._resolve_path(db_path)

        # Load database (custom or built-in)
        db_type = db_config.get("type", "template")

        try:
            if db_type in self._custom_databases:
                # Custom database
                database_class = self._custom_databases[db_type]
                database = database_class(db_path, **db_config)
                logger.info(f"Loaded custom database: {db_type} ({database_class.__name__})")

            elif db_type == "template":
                presentation_mode = db_config.get("presentation_mode", "explicit")
                database = TemplateChannelDatabase(db_path, presentation_mode=presentation_mode)

            elif db_type in ("flat", "legacy"):
                # 'legacy' is an alias for 'flat' (backward compatibility)
                database = FlatChannelDatabase(db_path)

            elif db_type == "hierarchical":
                database = HierarchicalChannelDatabase(db_path)

            else:
                available = list(self.list_available_databases().keys())
                raise ConfigurationError(
                    f"Unknown database type: '{db_type}'. "
                    f"Available databases: {', '.join(available)}"
                )
        except FileNotFoundError as e:
            raise DatabaseLoadError(f"Database file not found: {db_path}") from e

        # Load facility config
        facility_config = config.get("facility", {})
        facility_name = facility_config.get("name", "control system")

        # Load facility description from prompts (if available)
        facility_description = ""
        try:
            prompts_module = load_prompts(config)
            if hasattr(prompts_module, "system") and hasattr(
                prompts_module.system, "facility_description"
            ):
                facility_description = prompts_module.system.facility_description
                logger.info(
                    f"[dim]✓ Loaded facility context from prompts ({len(facility_description)} chars)[/dim]"
                )
        except Exception as e:
            logger.debug(f"Could not load facility description: {e}")

        # Initialize custom pipeline with all config
        logger.info(f"Initializing custom pipeline: {pipeline_name} ({pipeline_class.__name__})")

        return pipeline_class(
            database=database,
            model_config=model_config,
            facility_name=facility_name,
            facility_description=facility_description,
            **processing_config,  # Pipeline-specific processing config
            **kwargs,  # Override parameters
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
            "pipeline_mode": self.pipeline_mode,
            "pipeline_name": self.pipeline.pipeline_name,
            "statistics": self.pipeline.get_statistics(),
        }
