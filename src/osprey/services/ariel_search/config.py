"""ARIEL configuration classes.

This module defines typed configuration classes for ARIEL components.
Configuration is loaded from the `ariel:` section of config.yml.

"""

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for a single embedding model.

    Used in text_embedding enhancement to specify which models
    to embed with during ingestion.

    Attributes:
        name: Model name (e.g., "nomic-embed-text")
        dimension: Embedding dimension (must match model output)
        max_input_tokens: Maximum input tokens for the model (optional)
    """

    name: str
    dimension: int
    max_input_tokens: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        return cls(
            name=data["name"],
            dimension=data["dimension"],
            max_input_tokens=data.get("max_input_tokens"),
        )


@dataclass
class PipelineModuleConfig:
    """Configuration for a pipeline (rag, agent).

    Pipelines compose search modules into higher-level execution strategies.

    Attributes:
        enabled: Whether pipeline is active
        retrieval_modules: Which search modules this pipeline uses
        settings: Pipeline-specific settings
    """

    enabled: bool = True
    retrieval_modules: list[str] = field(default_factory=lambda: ["keyword", "semantic"])
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineModuleConfig":
        """Create PipelineModuleConfig from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            retrieval_modules=data.get("retrieval_modules", ["keyword", "semantic"]),
            settings=data.get("settings", {}),
        )


@dataclass
class SearchModuleConfig:
    """Configuration for a single search module (keyword, semantic).

    Attributes:
        enabled: Whether module is active
        provider: Provider name for embeddings (references api.providers section)
        model: Model identifier for semantic modules - which model's table to query
        settings: Module-specific settings
    """

    enabled: bool
    provider: str | None = None
    model: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchModuleConfig":
        """Create SearchModuleConfig from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            provider=data.get("provider"),
            model=data.get("model"),
            settings=data.get("settings", {}),
        )


@dataclass
class EnhancementModuleConfig:
    """Configuration for a single enhancement module (text_embedding, semantic_processor).

    Attributes:
        enabled: Whether module is active
        provider: Provider name for embeddings (references api.providers section)
        models: List of model configurations (for text_embedding)
        settings: Module-specific settings
    """

    enabled: bool
    provider: str | None = None
    models: list[ModelConfig] | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnhancementModuleConfig":
        """Create EnhancementModuleConfig from dictionary."""
        models = None
        if "models" in data:
            models = [ModelConfig.from_dict(m) for m in data["models"]]

        # Capture all extra keys as settings (not just explicit settings: block)
        reserved_keys = {"enabled", "provider", "models", "settings"}
        extra_settings = {k: v for k, v in data.items() if k not in reserved_keys}

        # Merge explicit settings with extra keys (explicit takes precedence)
        settings = {**extra_settings, **data.get("settings", {})}

        return cls(
            enabled=data.get("enabled", False),
            provider=data.get("provider"),
            models=models,
            settings=settings,
        )


@dataclass
class WatchConfig:
    """Configuration for the watch (live polling) mode.

    Attributes:
        require_initial_ingest: Require at least one successful ingest before watching
        max_consecutive_failures: Stop after this many consecutive poll failures
        backoff_multiplier: Multiply interval by this on consecutive failures
        max_interval_seconds: Maximum poll interval after backoff
    """

    require_initial_ingest: bool = True
    max_consecutive_failures: int = 10
    backoff_multiplier: float = 2.0
    max_interval_seconds: int = 3600

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchConfig":
        """Create WatchConfig from dictionary."""
        return cls(
            require_initial_ingest=data.get("require_initial_ingest", True),
            max_consecutive_failures=data.get("max_consecutive_failures", 10),
            backoff_multiplier=data.get("backoff_multiplier", 2.0),
            max_interval_seconds=data.get("max_interval_seconds", 3600),
        )


@dataclass
class IngestionConfig:
    """Configuration for logbook ingestion.

    Attributes:
        adapter: Adapter name (e.g., "als_logbook", "generic_json")
        source_url: URL for source system API (optional)
        poll_interval_seconds: Polling interval for incremental ingestion
        proxy_url: SOCKS proxy URL (e.g., "socks5://localhost:9095")
        verify_ssl: Whether to verify SSL certificates (default: False for internal servers)
        chunk_days: Days per API request for time windowing (default: 365)
        request_timeout_seconds: Timeout for HTTP requests (default: 60)
        max_retries: Maximum retry attempts for failed requests (default: 3)
        retry_delay_seconds: Base delay between retries (default: 5)
        watch: Watch mode configuration
    """

    adapter: str
    source_url: str | None = None
    poll_interval_seconds: int = 3600
    proxy_url: str | None = None
    verify_ssl: bool = False
    chunk_days: int = 365
    request_timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: int = 5
    watch: WatchConfig = field(default_factory=WatchConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngestionConfig":
        """Create IngestionConfig from dictionary."""
        proxy_url = data.get("proxy_url") or os.environ.get("ARIEL_SOCKS_PROXY")

        watch = WatchConfig()
        if "watch" in data:
            watch = WatchConfig.from_dict(data["watch"])

        return cls(
            adapter=data.get("adapter", "generic"),
            source_url=data.get("source_url"),
            poll_interval_seconds=data.get("poll_interval_seconds", 3600),
            proxy_url=proxy_url,
            verify_ssl=data.get("verify_ssl", False),
            chunk_days=data.get("chunk_days", 365),
            request_timeout_seconds=data.get("request_timeout_seconds", 60),
            max_retries=data.get("max_retries", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 5),
            watch=watch,
        )


@dataclass
class DatabaseConfig:
    """Configuration for ARIEL database connection.

    Attributes:
        uri: PostgreSQL connection URI (e.g., "postgresql://localhost:5432/ariel")
    """

    uri: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatabaseConfig":
        """Create DatabaseConfig from dictionary."""
        return cls(uri=data["uri"])


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation.

    Attributes:
        provider: Provider name (uses central Osprey config)
    """

    provider: str = "ollama"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingConfig":
        """Create EmbeddingConfig from dictionary."""
        return cls(provider=data.get("provider", "ollama"))


@dataclass
class ReasoningConfig:
    """Configuration for agentic reasoning behavior.

    Uses Osprey's provider configuration system for credentials.
    The `provider` field references api.providers for api_key and base_url.

    Attributes:
        provider: Provider name (references api.providers section)
        model_id: LLM model identifier (default: "gpt-4o-mini")
        max_iterations: Maximum ReAct cycles (default: 5)
        temperature: LLM temperature (default: 0.1)
        tool_timeout_seconds: Per-tool call timeout (default: 30)
        total_timeout_seconds: Total agent execution timeout (default: 120)
    """

    provider: str = "openai"
    model_id: str = "gpt-4o-mini"
    max_iterations: int = 5
    temperature: float = 0.1
    tool_timeout_seconds: int = 30
    total_timeout_seconds: int = 120

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReasoningConfig":
        """Create ReasoningConfig from dictionary."""
        return cls(
            provider=data.get("provider", "openai"),
            model_id=data.get("model_id", "gpt-4o-mini"),
            max_iterations=data.get("max_iterations", 5),
            temperature=data.get("temperature", 0.1),
            tool_timeout_seconds=data.get("tool_timeout_seconds", 30),
            total_timeout_seconds=data.get("total_timeout_seconds", 120),
        )


@dataclass
class ARIELConfig:
    """Root configuration for ARIEL service.

    Attributes:
        database: Database connection configuration
        search_modules: Search module configurations by name
        enhancement_modules: Enhancement module configurations by name
        ingestion: Ingestion configuration
        reasoning: Agentic reasoning configuration
        embedding: Embedding provider configuration
        default_max_results: Default maximum results to return
        cache_embeddings: Whether to cache embeddings
    """

    database: DatabaseConfig
    search_modules: dict[str, SearchModuleConfig] = field(default_factory=dict)
    pipelines: dict[str, PipelineModuleConfig] = field(default_factory=dict)
    enhancement_modules: dict[str, EnhancementModuleConfig] = field(default_factory=dict)
    ingestion: IngestionConfig | None = None
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    default_max_results: int = 10
    cache_embeddings: bool = True

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ARIELConfig":
        """Create ARIELConfig from config.yml dictionary.

        Args:
            config_dict: The 'ariel' section from config.yml

        Returns:
            ARIELConfig instance
        """
        database = DatabaseConfig.from_dict(config_dict["database"])

        search_modules: dict[str, SearchModuleConfig] = {}
        for name, data in config_dict.get("search_modules", {}).items():
            search_modules[name] = SearchModuleConfig.from_dict(data)

        pipelines: dict[str, PipelineModuleConfig] = {}
        for name, data in config_dict.get("pipelines", {}).items():
            pipelines[name] = PipelineModuleConfig.from_dict(data)

        enhancement_modules: dict[str, EnhancementModuleConfig] = {}
        for name, data in config_dict.get("enhancement_modules", {}).items():
            enhancement_modules[name] = EnhancementModuleConfig.from_dict(data)

        ingestion = None
        if "ingestion" in config_dict:
            ingestion = IngestionConfig.from_dict(config_dict["ingestion"])

        reasoning = ReasoningConfig()
        if "reasoning" in config_dict:
            reasoning = ReasoningConfig.from_dict(config_dict["reasoning"])

        embedding = EmbeddingConfig()
        if "embedding" in config_dict:
            embedding = EmbeddingConfig.from_dict(config_dict["embedding"])

        return cls(
            database=database,
            search_modules=search_modules,
            pipelines=pipelines,
            enhancement_modules=enhancement_modules,
            ingestion=ingestion,
            reasoning=reasoning,
            embedding=embedding,
            default_max_results=config_dict.get("default_max_results", 10),
            cache_embeddings=config_dict.get("cache_embeddings", True),
        )

    def is_search_module_enabled(self, name: str) -> bool:
        """Check if a search module is enabled.

        Args:
            name: Module name (keyword, semantic)

        Returns:
            True if the module is enabled
        """
        module = self.search_modules.get(name)
        return module is not None and module.enabled

    def get_enabled_search_modules(self) -> list[str]:
        """Get list of enabled search module names.

        Returns:
            List of enabled module names
        """
        return [name for name, config in self.search_modules.items() if config.enabled]

    def is_pipeline_enabled(self, name: str) -> bool:
        """Check if a pipeline is enabled.

        If no pipeline config exists for the name, defaults to True
        (pipelines are always-on unless explicitly disabled).

        Args:
            name: Pipeline name (rag, agent)

        Returns:
            True if the pipeline is enabled
        """
        pipeline = self.pipelines.get(name)
        if pipeline is None:
            return True  # Pipelines default to enabled
        return pipeline.enabled

    def get_enabled_pipelines(self) -> list[str]:
        """Get list of enabled pipeline names.

        Returns configured pipelines that are enabled, plus default
        pipelines (rag, agent) if not explicitly configured.

        Returns:
            List of enabled pipeline names
        """
        # Start with defaults
        defaults = {"rag", "agent"}
        result: list[str] = []

        # Add explicitly configured pipelines that are enabled
        for name, config in self.pipelines.items():
            if config.enabled:
                result.append(name)
            defaults.discard(name)

        # Add defaults that weren't explicitly configured
        result.extend(sorted(defaults))

        return result

    def get_pipeline_retrieval_modules(self, name: str) -> list[str]:
        """Get retrieval modules configured for a pipeline.

        If the pipeline has no explicit config, falls back to
        the list of enabled search modules.

        Args:
            name: Pipeline name (rag, agent)

        Returns:
            List of search module names to use for retrieval
        """
        pipeline = self.pipelines.get(name)
        if pipeline is not None:
            return list(pipeline.retrieval_modules)
        # Fall back to enabled search modules
        return self.get_enabled_search_modules()

    def is_enhancement_module_enabled(self, name: str) -> bool:
        """Check if an enhancement module is enabled.

        Args:
            name: Module name (text_embedding, semantic_processor, figure_embedding)

        Returns:
            True if the module is enabled
        """
        module = self.enhancement_modules.get(name)
        return module is not None and module.enabled

    def get_enabled_enhancement_modules(self) -> list[str]:
        """Get list of enabled enhancement module names.

        Returns:
            List of enabled module names
        """
        return [name for name, config in self.enhancement_modules.items() if config.enabled]

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Validate database URI
        if not self.database.uri:
            errors.append("database.uri is required")

        # Validate semantic search requires model
        if self.is_search_module_enabled("semantic"):
            semantic_config = self.search_modules.get("semantic")
            if semantic_config and not semantic_config.model:
                errors.append(
                    "search_modules.semantic.model is required when semantic search is enabled"
                )

        # Validate text_embedding requires models list
        if self.is_enhancement_module_enabled("text_embedding"):
            text_emb_config = self.enhancement_modules.get("text_embedding")
            if text_emb_config and not text_emb_config.models:
                errors.append(
                    "enhancement_modules.text_embedding.models is required "
                    "when text_embedding enhancement is enabled"
                )

        # Validate reasoning config
        if self.reasoning.max_iterations < 1:
            errors.append("reasoning.max_iterations must be >= 1")
        if self.reasoning.total_timeout_seconds < 1:
            errors.append("reasoning.total_timeout_seconds must be >= 1")

        return errors

    def get_search_model(self) -> str | None:
        """Get the configured search model name.

        Returns the model configured for semantic search, which is also
        used for RAG search when enabled.

        Returns:
            Model name or None if semantic search is not enabled
        """
        if self.is_search_module_enabled("semantic"):
            semantic_config = self.search_modules.get("semantic")
            if semantic_config:
                return semantic_config.model
        return None

    def get_enhancement_module_config(self, name: str) -> dict[str, Any] | None:
        """Get configuration dictionary for an enhancement module.

        Returns the raw configuration that can be passed to module.configure().

        Args:
            name: Module name (text_embedding, semantic_processor)

        Returns:
            Configuration dictionary or None if module not configured
        """
        module_config = self.enhancement_modules.get(name)
        if not module_config:
            return None

        # Convert back to dict for configure() method
        config: dict[str, Any] = {"enabled": module_config.enabled}

        # Include provider (falls back to embedding.provider if not set)
        provider = module_config.provider or self.embedding.provider
        config["provider"] = provider

        if module_config.models:
            config["models"] = [
                {
                    "name": m.name,
                    "dimension": m.dimension,
                    "max_input_tokens": m.max_input_tokens,
                }
                for m in module_config.models
            ]

        config.update(module_config.settings)
        return config
