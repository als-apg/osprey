"""ARIEL configuration classes.

This module defines typed configuration classes for ARIEL components.
Configuration is loaded from the `ariel:` section of config.yml.

See 04_OSPREY_INTEGRATION.md Sections 3.1-3.8 for full specification.
"""

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
class SearchModuleConfig:
    """Configuration for a single search module (keyword, semantic, rag).

    Attributes:
        enabled: Whether module is active
        model: Model identifier for semantic/rag modules - which model's table to query
        settings: Module-specific settings
    """

    enabled: bool
    model: str | None = None  # Required for semantic/rag, ignored for keyword
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchModuleConfig":
        """Create SearchModuleConfig from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            model=data.get("model"),
            settings=data.get("settings", {}),
        )


@dataclass
class EnhancementModuleConfig:
    """Configuration for a single enhancement module (text_embedding, semantic_processor).

    Attributes:
        enabled: Whether module is active
        models: List of model configurations (for text_embedding)
        settings: Module-specific settings
    """

    enabled: bool
    models: list[ModelConfig] | None = None  # For text_embedding
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnhancementModuleConfig":
        """Create EnhancementModuleConfig from dictionary."""
        models = None
        if "models" in data:
            models = [ModelConfig.from_dict(m) for m in data["models"]]
        return cls(
            enabled=data.get("enabled", False),
            models=models,
            settings=data.get("settings", {}),
        )


@dataclass
class IngestionConfig:
    """Configuration for logbook ingestion.

    Attributes:
        adapter: Adapter name (e.g., "als_logbook", "generic_json")
        source_url: URL for source system API (optional)
        poll_interval_seconds: Polling interval for incremental ingestion
    """

    adapter: str
    source_url: str | None = None
    poll_interval_seconds: int = 3600

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngestionConfig":
        """Create IngestionConfig from dictionary."""
        return cls(
            adapter=data.get("adapter", "generic"),
            source_url=data.get("source_url"),
            poll_interval_seconds=data.get("poll_interval_seconds", 3600),
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

    Attributes:
        llm_model_id: LLM model identifier (default: "gpt-4o-mini")
        llm_provider: LLM provider name (default: "openai")
        max_iterations: Maximum ReAct cycles (default: 5)
        temperature: LLM temperature (default: 0.1)
        tool_timeout_seconds: Per-tool call timeout (default: 30) - V2
        total_timeout_seconds: Total agent execution timeout (default: 120)
    """

    llm_model_id: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    max_iterations: int = 5
    temperature: float = 0.1
    tool_timeout_seconds: int = 30
    total_timeout_seconds: int = 120

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReasoningConfig":
        """Create ReasoningConfig from dictionary."""
        return cls(
            llm_model_id=data.get("llm_model_id", "gpt-4o-mini"),
            llm_provider=data.get("llm_provider", "openai"),
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
        # Database is required
        database = DatabaseConfig.from_dict(config_dict["database"])

        # Parse search modules
        search_modules: dict[str, SearchModuleConfig] = {}
        for name, data in config_dict.get("search_modules", {}).items():
            search_modules[name] = SearchModuleConfig.from_dict(data)

        # Parse enhancement modules
        enhancement_modules: dict[str, EnhancementModuleConfig] = {}
        for name, data in config_dict.get("enhancement_modules", {}).items():
            enhancement_modules[name] = EnhancementModuleConfig.from_dict(data)

        # Parse ingestion
        ingestion = None
        if "ingestion" in config_dict:
            ingestion = IngestionConfig.from_dict(config_dict["ingestion"])

        # Parse reasoning
        reasoning = ReasoningConfig()
        if "reasoning" in config_dict:
            reasoning = ReasoningConfig.from_dict(config_dict["reasoning"])

        # Parse embedding
        embedding = EmbeddingConfig()
        if "embedding" in config_dict:
            embedding = EmbeddingConfig.from_dict(config_dict["embedding"])

        return cls(
            database=database,
            search_modules=search_modules,
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
            name: Module name (keyword, semantic, rag, vision)

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
                errors.append("search_modules.semantic.model is required when semantic search is enabled")

        # Validate RAG search requires model
        if self.is_search_module_enabled("rag"):
            rag_config = self.search_modules.get("rag")
            if rag_config and not rag_config.model:
                errors.append("search_modules.rag.model is required when RAG search is enabled")

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
