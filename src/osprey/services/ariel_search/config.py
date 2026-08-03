"""ARIEL configuration classes.

This module defines typed configuration classes for ARIEL components.
Configuration is loaded from the `ariel:` section of config.yml.

"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .exceptions import ConfigurationError

logger = logging.getLogger("osprey.services.ariel_search.config")

#: Enhancement modules that drive a chat/completion endpoint rather than an
#: embedding one.  ``ariel.embedding.provider`` is never substituted for these.
LLM_ENHANCEMENT_MODULES = frozenset({"semantic_processor"})

#: Defaults for the derived DSN, matching what the postgresql compose service
#: is rendered with when ``services.postgresql`` leaves a field unset.
_DEFAULT_DB_USERNAME = "ariel"
_DEFAULT_DB_NAME = "ariel"
_DEFAULT_DB_PORT = 5432
#: Matches the ``${ARIEL_DB_PASSWORD:-ariel}`` fallback the compose service
#: uses, so the agent stays launchable before the first `osprey deploy up`
#: mints a real password into the project ``.env``.
_DEFAULT_DB_PASSWORD = "ariel"

#: Tripped the first time a config is parsed that still carries the retired
#: MCP-only ``ariel.database.connection_string`` alias.  Warning once per
#: process keeps a CLI that parses the config repeatedly from repeating itself.
_connection_string_warned = False


def resolve_ariel_dsn(
    ariel_section: dict[str, Any],
    services: dict[str, Any] | None = None,
) -> str:
    """Resolve the effective ARIEL database DSN.

    ``ariel.database.uri`` is optional: with it unset, the DSN is derived from
    the ``services.postgresql`` block that defines the Postgres the agent talks
    to, so a project that moves its database port or renames its database stays
    connectable without editing a second, duplicated copy of the same facts.

    Precedence:
      1. An explicit ``uri`` wins, verbatim — including a DSN pointing at an
         external database that has no ``services.postgresql`` counterpart.
      2. A legacy ``connection_string`` (the retired MCP-only alias) is honored
         as the effective ``uri``, with a once-per-process warning naming the
         key that replaced it.  It is never discarded in favor of the derived
         DSN — a project pointed at a real database keeps reaching it.
      3. Otherwise: ``postgresql://{username}:{password}@localhost:{port_host}/
         {database_name}``, where password reads ``ARIEL_DB_PASSWORD`` from the
         environment (the same value the compose service reads as
         ``POSTGRES_PASSWORD``) and falls back to ``ariel``.

    Args:
        ariel_section: The ``ariel`` section from config.yml.
        services: The ``services.postgresql`` mapping from config.yml, or None
            when the caller has no services block — the derived DSN then falls
            back to the shipped Postgres defaults.

    Returns:
        The DSN to connect with.
    """
    database = ariel_section.get("database") or {}

    uri = database.get("uri")
    if uri is not None:
        return str(uri)

    legacy_uri = database.get("connection_string")
    if legacy_uri is not None:
        _warn_once_connection_string_retired()
        return str(legacy_uri)

    postgresql = services or {}
    username = postgresql.get("username", _DEFAULT_DB_USERNAME)
    database_name = postgresql.get("database_name", _DEFAULT_DB_NAME)
    port = postgresql.get("port_host", _DEFAULT_DB_PORT)
    password = os.environ.get("ARIEL_DB_PASSWORD", _DEFAULT_DB_PASSWORD)

    return f"postgresql://{username}:{password}@localhost:{port}/{database_name}"


def _warn_once_connection_string_retired() -> None:
    """Warn once per process that ``connection_string`` is a retired alias."""
    global _connection_string_warned
    if _connection_string_warned:
        return

    _connection_string_warned = True
    logger.warning(
        "config.yml sets ariel.database.connection_string, a retired alias. Its "
        "value is still being honored as the ARIEL DSN, but rename the key to "
        "ariel.database.uri — or delete it entirely and let the DSN derive from "
        "services.postgresql (username, database_name, port_host)."
    )


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
class WriteConfig:
    """Configuration for facility logbook write operations.

    Attributes:
        enabled: Whether writing to the facility logbook is enabled
        write_url: URL for the facility write API (if different from source_url)
        auth_user: Username for write authentication
        auth_password: Password for write authentication
    """

    enabled: bool = False
    write_url: str | None = None
    auth_user: str | None = None
    auth_password: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WriteConfig":
        """Create WriteConfig from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            write_url=data.get("write_url"),
            auth_user=data.get("auth_user") or os.environ.get("ARIEL_WRITE_USER"),
            auth_password=data.get("auth_password") or os.environ.get("ARIEL_WRITE_PASSWORD"),
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
        write: Write operation configuration
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
    write: WriteConfig = field(default_factory=WriteConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngestionConfig":
        """Create IngestionConfig from dictionary."""
        proxy_url = data.get("proxy_url") or os.environ.get("ARIEL_SOCKS_PROXY")

        watch = WatchConfig()
        if "watch" in data:
            watch = WatchConfig.from_dict(data["watch"])

        write = WriteConfig()
        if "write" in data:
            write = WriteConfig.from_dict(data["write"])

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
            write=write,
        )


@dataclass
class DatabaseConfig:
    """Configuration for ARIEL database connection.

    Attributes:
        uri: PostgreSQL connection URI (e.g., "postgresql://localhost:5432/ariel")
    """

    uri: str

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], services: dict[str, Any] | None = None
    ) -> "DatabaseConfig":
        """Create DatabaseConfig from the ``ariel.database`` mapping.

        Args:
            data: The ``ariel.database`` mapping from config.yml.
            services: The ``services.postgresql`` mapping, used to derive the
                DSN when ``uri`` is unset. See :func:`resolve_ariel_dsn`.
        """
        return cls(uri=resolve_ariel_dsn({"database": data}, services))


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
class ARIELConfig:
    """Root configuration for ARIEL service.

    Attributes:
        database: Database connection configuration
        search_modules: Search module configurations by name
        enhancement_modules: Enhancement module configurations by name
        ingestion: Ingestion configuration
        embedding: Embedding provider configuration

    Documented top-level config keys read at runtime (not dataclass fields):
        entry_url_template: Optional ``str`` template for the canonical logbook
            entry URL, e.g. ``"https://logbook.example/olog.php?id={entry_id}"``.
            Read at egress time via ``get_config_value("ariel.entry_url_template")``
            (see ``mcp_server.ariel.server.build_entry_url``), NOT parsed by
            ``from_dict`` — the ARIEL read tools render it with the URL-encoded
            ``entry_id`` so the agent links entries verbatim. Facility-neutral by
            default (unset -> no ``entry_url`` emitted); the facility supplies the
            value in its own config (for ALS, ``als-base.yml``).
    """

    database: DatabaseConfig
    search_modules: dict[str, SearchModuleConfig] = field(default_factory=dict)
    enhancement_modules: dict[str, EnhancementModuleConfig] = field(default_factory=dict)
    ingestion: IngestionConfig | None = None
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], services: dict[str, Any] | None = None
    ) -> "ARIELConfig":
        """Create ARIELConfig from config.yml dictionary.

        Args:
            config_dict: The 'ariel' section from config.yml
            services: The ``services.postgresql`` mapping from config.yml. With
                ``ariel.database.uri`` unset, the DSN is derived from it — see
                :func:`resolve_ariel_dsn`.

        Returns:
            ARIELConfig instance

        Raises:
            ConfigurationError: If the deprecated 'pipelines' section is present.
        """
        if "pipelines" in config_dict:
            raise ConfigurationError(
                "The ariel.pipelines section is no longer supported. "
                "Remove 'pipelines:' from your config — the RAG and ReAct "
                "pipelines have been removed in favor of the Osprey agent layer. "
                "See CHANGELOG for migration details.",
                config_key="pipelines",
            )

        database = DatabaseConfig.from_dict(config_dict.get("database") or {}, services)

        search_modules: dict[str, SearchModuleConfig] = {}
        for name, data in config_dict.get("search_modules", {}).items():
            search_modules[name] = SearchModuleConfig.from_dict(data)

        enhancement_modules: dict[str, EnhancementModuleConfig] = {}
        for name, data in config_dict.get("enhancement_modules", {}).items():
            enhancement_modules[name] = EnhancementModuleConfig.from_dict(data)

        ingestion = None
        if "ingestion" in config_dict:
            ingestion = IngestionConfig.from_dict(config_dict["ingestion"])

        embedding = EmbeddingConfig()
        if "embedding" in config_dict:
            embedding = EmbeddingConfig.from_dict(config_dict["embedding"])

        return cls(
            database=database,
            search_modules=search_modules,
            enhancement_modules=enhancement_modules,
            ingestion=ingestion,
            embedding=embedding,
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

        return errors

    def get_search_model(self) -> str | None:
        """Get the configured search model name.

        Returns the model configured for semantic search.

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

        # `embedding.provider` is the default *embedding* endpoint; substituting it
        # for a module that calls an LLM would silently pick the wrong service, so
        # those modules carry their own provider or fail in configure().
        provider = module_config.provider
        if provider is None and name not in LLM_ENHANCEMENT_MODULES:
            provider = self.embedding.provider
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
