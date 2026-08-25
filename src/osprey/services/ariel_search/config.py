"""ARIEL configuration classes.

This module defines typed configuration classes for ARIEL components.
Configuration is loaded from the `ariel:` section of config.yml.

"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.utils.config_paths import resolve_config_relative_path

from .exceptions import ConfigurationError
from .models import normalize_search_mode, resolve_implicit_search_mode
from .vocabulary.loader import load_vocabulary
from .vocabulary.model import Vocabulary

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
#: uses, so the agent stays launchable before the first `osprey up`
#: mints a real password into the project ``.env``.
_DEFAULT_DB_PASSWORD = "ariel"

#: Dotted prefix of the facility-vocabulary block, so every message this module
#: raises or reports names the key an operator would edit rather than a bare
#: field name. Mirrors ``_SETTINGS_PREFIX`` in ``search/qmd.py``.
_VOCABULARY_PREFIX = "ariel.vocabulary"

#: Tripped the first time a config is parsed that still carries the retired
#: MCP-only ``ariel.database.connection_string`` alias.  Warning once per
#: process keeps a CLI that parses the config repeatedly from repeating itself.
_connection_string_warned = False


def resolve_ariel_dsn(
    ariel_section: dict[str, Any],
    services: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
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
        env: Where to read ``ARIEL_DB_PASSWORD`` from. Defaults to the process
            environment, which is what every in-container consumer wants. A
            caller acting ON a project rather than IN it — the deploy, which
            migrates a store it is bringing up — passes that project's own
            ``.env`` instead: run from another directory, the ambient value
            belongs to some other deployment, and using it would either fail
            confusingly or reach a database this call was never pointed at.

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
    password = (env if env is not None else os.environ).get(
        "ARIEL_DB_PASSWORD", _DEFAULT_DB_PASSWORD
    )

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


def _vocab_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    """Read one boolean knob from the ``ariel.vocabulary`` block.

    A *present* key of the wrong type is refused rather than defaulted, for the
    same reason the hybrid module refuses ``rerank: "false"``: silently ignoring
    it would leave the deployment on the behavior it explicitly asked to leave.

    Args:
        data: The ``ariel.vocabulary`` mapping.
        key: Leaf key to read.
        default: Value to use when the key is absent.

    Returns:
        The boolean value.

    Raises:
        ValueError: If the key is present but not a boolean.
    """
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{_VOCABULARY_PREFIX}.{key} must be a boolean, got {value!r}")
    return value


@dataclass(frozen=True)
class VocabularyConfig:
    """The ``ariel.vocabulary`` block: facility shorthand, query-time expansion.

    Attributes:
        enabled: Whether a vocabulary is configured at all. With it false
            nothing is loaded and no query is expanded, even when ``path`` names
            a file that does not exist.
        path: The ``vocabulary.yml`` to load. A relative value resolves against
            the directory holding the config file the caller loaded (see
            :func:`~osprey.utils.config_paths.resolve_config_relative_path`).
        expand_by_default: Whether a request that names no ``expand_query``
            preference gets expansion. The per-request argument overrides it.
        canonical_to_acronym: Whether a canonical phrase also expands to the
            forms of its ``acronym`` concepts.
        canonical_to_shorthand: The same gate for ``shorthand`` concepts. Off by
            default: expanding prose to operator shorthand rarely helps recall.
        expand_modes: Search modules expansion applies to. ``None`` — the
            default — means every enabled search module. An explicit list is
            normalized at parse; :meth:`ARIELConfig.validate` refuses an entry
            naming a module that is unknown or not enabled.
    """

    enabled: bool = False
    path: str | None = None
    expand_by_default: bool = True
    canonical_to_acronym: bool = True
    canonical_to_shorthand: bool = False
    expand_modes: tuple[str, ...] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VocabularyConfig":
        """Create VocabularyConfig from the ``ariel.vocabulary`` mapping.

        An absent block is the normal case and yields the defaults. Every
        present key is type-checked and a bad one raises, naming the full dotted
        key — the callers that must survive a broken config (the web lifespan,
        ``osprey ariel status``) wrap the whole parse in their own handler and
        report the message, so refusing here loses nothing and defaulting here
        would hide a knob that never took effect.

        Args:
            data: The ``ariel.vocabulary`` mapping from config.yml.

        Returns:
            VocabularyConfig instance.

        Raises:
            ValueError: If any present key has the wrong type or shape.
        """
        enabled = _vocab_bool(data, "enabled", cls.enabled)
        expand_by_default = _vocab_bool(data, "expand_by_default", cls.expand_by_default)
        to_acronym = _vocab_bool(data, "canonical_to_acronym", cls.canonical_to_acronym)
        to_shorthand = _vocab_bool(data, "canonical_to_shorthand", cls.canonical_to_shorthand)

        path = data.get("path", cls.path)
        if path is not None and (not isinstance(path, str) or not path.strip()):
            raise ValueError(
                f"{_VOCABULARY_PREFIX}.path must be a non-empty string naming a "
                f"vocabulary file, got {path!r}"
            )

        expand_modes: tuple[str, ...] | None = None
        raw_modes = data.get("expand_modes")
        if raw_modes is not None:
            if not isinstance(raw_modes, list):
                raise ValueError(
                    f"{_VOCABULARY_PREFIX}.expand_modes must be a list of search "
                    f"mode names, got {raw_modes!r}"
                )
            normalized: list[str] = []
            for entry in raw_modes:
                try:
                    normalized.append(normalize_search_mode(entry))
                except ValueError as exc:
                    raise ValueError(
                        f"{_VOCABULARY_PREFIX}.expand_modes entries must be non-empty "
                        f"search mode names, got {entry!r}"
                    ) from exc
            expand_modes = tuple(normalized)

        return cls(
            enabled=enabled,
            path=path,
            expand_by_default=expand_by_default,
            canonical_to_acronym=to_acronym,
            canonical_to_shorthand=to_shorthand,
            expand_modes=expand_modes,
        )


@dataclass
class ARIELConfig:
    """Root configuration for ARIEL service.

    Attributes:
        database: Database connection configuration
        search_modules: Search module configurations by name
        enhancement_modules: Enhancement module configurations by name
        ingestion: Ingestion configuration
        embedding: Embedding provider configuration
        default_search_mode: Search module a search runs in when the caller
            names no mode. Unset resolves to ``hybrid`` where that module is
            enabled and ``keyword`` otherwise; a name that is set but not
            enabled is a config error rather than a silent fallback.
        vocabulary: The ``ariel.vocabulary`` block.
        loaded_vocabulary: The vocabulary read at parse time, or None when the
            block is disabled, names no path, or failed to load.
        vocabulary_errors: Loader errors, each naming ``ariel.vocabulary.path``
            and the resolved file. Kept as its OWN field rather than folded into
            :meth:`validate`'s pre-existing errors because the two mean
            different things to a surface: a vocabulary error means search
            cannot honor what the deployment configured, while the pre-existing
            classes leave a working service.
        vocabulary_warnings: Loader warnings — a vocabulary that loads but may
            not behave as its author expects. Never blocks anything.

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
    default_search_mode: str | None = None
    vocabulary: VocabularyConfig = field(default_factory=VocabularyConfig)
    loaded_vocabulary: Vocabulary | None = None
    vocabulary_errors: list[str] = field(default_factory=list)
    vocabulary_warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        config_dict: dict[str, Any],
        services: dict[str, Any] | None = None,
        *,
        config_dir: Path | None = None,
    ) -> "ARIELConfig":
        """Create ARIELConfig from config.yml dictionary.

        Args:
            config_dict: The 'ariel' section from config.yml
            services: The ``services.postgresql`` mapping from config.yml. With
                ``ariel.database.uri`` unset, the DSN is derived from it — see
                :func:`resolve_ariel_dsn`.
            config_dir: Directory holding the config.yml this dictionary came
                from. A relative ``ariel.vocabulary.path`` resolves against the
                project root derived from it, so every process reading the
                same config finds the same file.
                Keyword-only with a None default: callers that never name a
                config-relative path are unaffected, and None falls back to the
                shared helper's own rule.

        Returns:
            ARIELConfig instance

        Raises:
            ConfigurationError: If the deprecated 'pipelines' section is present.
            ValueError: If the ``vocabulary`` block is malformed.
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

        default_search_mode = config_dict.get("default_search_mode")
        if default_search_mode is not None:
            default_search_mode = normalize_search_mode(default_search_mode)

        vocabulary_data = config_dict.get("vocabulary")
        if vocabulary_data is None:
            vocabulary = VocabularyConfig()
        elif isinstance(vocabulary_data, dict):
            vocabulary = VocabularyConfig.from_dict(vocabulary_data)
        else:
            raise ValueError(f"{_VOCABULARY_PREFIX} must be a mapping, got {vocabulary_data!r}")

        # The file is read HERE, once, at config parse — not lazily on the first
        # search. Every surface that holds a config therefore already knows
        # whether the vocabulary is usable, and none of them can start searching
        # unexpanded because a read failed later. A disabled block reads nothing
        # at all, so a stale `path` left behind by an operator is inert.
        loaded_vocabulary: Vocabulary | None = None
        vocabulary_errors: list[str] = []
        vocabulary_warnings: list[str] = []
        if vocabulary.enabled and vocabulary.path:
            resolved = resolve_config_relative_path(vocabulary.path, config_dir)
            loaded_vocabulary, load_errors, vocabulary_warnings = load_vocabulary(
                resolved,
                canonical_to_acronym=vocabulary.canonical_to_acronym,
                canonical_to_shorthand=vocabulary.canonical_to_shorthand,
            )
            # Each stored error names the key AND the file the key resolved to:
            # a relative path that resolved somewhere unexpected is the most
            # common cause, and the bare loader message would not show it.
            vocabulary_errors = [
                f"{_VOCABULARY_PREFIX}.path: {resolved} — {error}" for error in load_errors
            ]

        return cls(
            database=database,
            search_modules=search_modules,
            enhancement_modules=enhancement_modules,
            ingestion=ingestion,
            embedding=embedding,
            default_search_mode=default_search_mode,
            vocabulary=vocabulary,
            loaded_vocabulary=loaded_vocabulary,
            vocabulary_errors=vocabulary_errors,
            vocabulary_warnings=vocabulary_warnings,
        )

    @property
    def vocabulary_active(self) -> bool:
        """Whether a usable vocabulary is loaded and expansion can run.

        Returns:
            True when the block is enabled, a vocabulary was loaded, and the
            loader reported no errors.
        """
        return (
            self.vocabulary.enabled
            and self.loaded_vocabulary is not None
            and not self.vocabulary_errors
        )

    def resolve_expand_modes(self) -> tuple[str, ...]:
        """Name the search modules query expansion applies to.

        Returns:
            The configured ``expand_modes`` when the deployment set one, else
            every enabled search module — the documented default of "all enabled
            modes". :meth:`validate` has already refused a configured entry that
            names no enabled module.
        """
        if self.vocabulary.expand_modes is not None:
            return self.vocabulary.expand_modes
        return tuple(self.get_enabled_search_modules())

    def resolve_default_search_mode(self) -> str:
        """Name the search module a search runs in when the caller names none.

        Returns:
            ``default_search_mode`` when the deployment sets it, else the
            implicit answer from
            :func:`~osprey.services.ariel_search.models.resolve_implicit_search_mode`.
            :meth:`validate` has already refused a configured mode that names
            no enabled module, so the returned name is routable.
        """
        if self.default_search_mode:
            return self.default_search_mode
        return resolve_implicit_search_mode(self.is_search_module_enabled)

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

        # A configured default that names no enabled module is refused here
        # rather than falling back: a deployment that asked for a mode and
        # silently got a different one has no way to notice.
        if self.default_search_mode and not self.is_search_module_enabled(self.default_search_mode):
            enabled = ", ".join(self.get_enabled_search_modules()) or "(none)"
            errors.append(
                f"default_search_mode '{self.default_search_mode}' names no enabled "
                f"search module. Enabled modules: {enabled}"
            )

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

        # Vocabulary. Enabling expansion without naming a file is a deployment
        # that believes it has expansion and does not; it is refused rather than
        # treated as "disabled".
        if self.vocabulary.enabled and not self.vocabulary.path:
            errors.append(
                f"{_VOCABULARY_PREFIX}.path is required when {_VOCABULARY_PREFIX}.enabled is true"
            )

        errors.extend(self.vocabulary_errors)

        if self.vocabulary.expand_modes:
            enabled_modules = self.get_enabled_search_modules()
            enabled_text = ", ".join(enabled_modules) or "(none)"
            for mode in self.vocabulary.expand_modes:
                if mode not in enabled_modules:
                    errors.append(
                        f"{_VOCABULARY_PREFIX}.expand_modes names no enabled search "
                        f"module: {mode}. Enabled modules: {enabled_text}"
                    )

        # The keyword module's own knobs. The resolver that the search path uses
        # is the one consulted here, so a malformed knob surfaces at startup and
        # in `ariel vocab-check` / `ariel status` rather than on the first search
        # that happens to carry a pattern. Imported lazily: the search package
        # reaches back into this module, and a module-level import would close
        # that loop. A disabled module is not consulted at all — its settings
        # block reaches no reader, so refusing it would be refusing dead config.
        if self.is_search_module_enabled("keyword"):
            from osprey.services.ariel_search.search.keyword import KeywordSearchSettings

            try:
                KeywordSearchSettings.from_ariel_config(self)
            except ValueError as exc:
                errors.append(str(exc))

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
