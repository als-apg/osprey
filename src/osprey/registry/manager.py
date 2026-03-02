"""Registry manager: lazy-loading, dependency-ordered component registry.

Provides :class:`RegistryManager` plus the global singleton helpers
:func:`get_registry`, :func:`initialize_registry`, and :func:`reset_registry`.

.. seealso:: :doc:`/developer-guides/registry-system`
"""

import logging
from pathlib import Path
from typing import Any

from osprey.errors import ConfigurationError, RegistryError  # noqa: F401 (re-exported)
from osprey.utils.config import get_agent_dir, get_config_value
from osprey.utils.logger import get_logger

from .base import RegistryConfig, RegistryConfigProvider  # noqa: F401 (re-exported)
from .export import export_registry_to_json as _export_registry_to_json
from .initializers import INITIALIZER_DISPATCH
from .loader import build_merged_configuration

logger = get_logger(name="registry", color="sky_blue2")


class RegistryManager:
    """Centralized registry for all Osprey Agentic Framework components.

    This class provides the single point of access for capabilities, context classes,
    services, providers, and connectors throughout the framework. It replaces the
    fragmented registry system with a unified approach that eliminates circular imports
    through lazy loading and provides dependency-ordered initialization.

    The registry system follows a strict initialization order to handle dependencies:
    1. Context classes (required by capabilities)
    2. Providers (AI model backends)
    3. Services (shared infrastructure)
    4. Capabilities (domain-specific functionality)
    5. Connectors (control system adapters)

    All components are loaded lazily using module path and class name metadata,
    preventing circular import issues while maintaining full introspection capabilities.

    .. note::
       The registry is typically accessed through the global functions get_registry()
       and initialize_registry() rather than instantiated directly.

    .. warning::
       Registry initialization must complete successfully before any components
       can be accessed. Failed initialization will raise RegistryError.
    """

    def __init__(self, registry_path: str | None = None):
        """Create a registry manager and build merged configuration.

        Mode is auto-detected: ``ExtendedRegistryConfig`` triggers extend mode
        (merge with framework), plain ``RegistryConfig`` triggers standalone mode.

        :param registry_path: Path to application ``registry.py``, or *None*
            for framework-only mode.
        :raises RegistryError: If registry cannot be loaded or is invalid.
        """
        self.registry_path = registry_path
        self._initialized = False

        self._registries = {
            "services": {},
            "providers": {},
            "connectors": {},
            "ariel_search_modules": {},
            "ariel_enhancement_modules": {},
            "ariel_pipelines": {},
            "ariel_ingestion_adapters": {},
        }

        self.config, self._excluded_provider_names = build_merged_configuration(
            registry_path
        )

    def initialize(self, silent: bool = False) -> None:
        """Load all registered components in dependency order.

        Idempotent -- returns immediately if already initialized.

        :param silent: Suppress INFO/DEBUG logging during init.
        :raises RegistryError: If any component fails to load.
        """
        if self._initialized:
            logger.debug("Registry already initialized")
            return

        original_levels = {}
        if silent:
            loggers_to_silence = [
                "registry",
                "registry.loader",
                "registry.init",
                "registry.export",
                "connector_factory",
            ]
            for logger_name in loggers_to_silence:
                log = logging.getLogger(logger_name)
                original_levels[logger_name] = log.level
                log.setLevel(logging.WARNING)

        try:
            logger.info("Initializing registry system...")

            for component_type in self.config.initialization_order:
                self._initialize_component_type(component_type)

            self._initialized = True
            logger.info(self._get_initialization_summary())

        except (ImportError, AttributeError, ConfigurationError) as e:
            logger.error(f"Registry initialization failed: {e}")
            raise RegistryError(f"Failed to initialize registry: {e}") from e
        except Exception as e:
            logger.error(f"Registry initialization failed with unexpected error: {e}")
            raise RegistryError(
                f"Unexpected error during registry initialization: {e}"
            ) from e
        finally:
            if silent:
                for logger_name, level in original_levels.items():
                    logging.getLogger(logger_name).setLevel(level)

    def _initialize_component_type(self, component_type: str) -> None:
        """Initialize components of a specific type via dispatch.

        :param component_type: Type of components to initialize.
        :raises ValueError: If *component_type* is not recognised.
        """
        initializer = INITIALIZER_DISPATCH.get(component_type)
        if initializer is None:
            raise ValueError(f"Unknown component type: {component_type}")
        initializer(
            config=self.config,
            registries=self._registries,
            excluded_provider_names=self._excluded_provider_names,
        )

    # ------------------------------------------------------------------
    # Accessor methods
    # ------------------------------------------------------------------

    def get_provider(self, name: str) -> type[Any] | None:
        """Retrieve registered provider class by name.

        Falls back to the lightweight ``ProviderRegistry`` when the full
        ``RegistryManager`` hasn't been initialized, so callers like
        ``get_chat_completion()`` work without full registry infrastructure.

        :param name: Unique provider name from registration
        :type name: str
        :return: Provider class if registered, None otherwise
        :rtype: Type[BaseProvider] or None
        """
        if not self._initialized:
            from osprey.models.provider_registry import get_provider_registry

            return get_provider_registry().get_provider(name)

        return self._registries["providers"].get(name)

    def list_providers(self) -> list[str]:
        """Get list of all registered provider names.

        :return: List of provider names
        :rtype: list[str]
        """
        return list(self._registries["providers"].keys())

    def get_connector(self, name: str) -> type[Any] | None:
        """Retrieve registered connector class by name.

        :param name: Unique connector name from registration
        :type name: str
        :return: Connector class if registered, None otherwise
        :rtype: Type[ControlSystemConnector] or Type[ArchiverConnector] or None
        """
        if not self._initialized:
            raise RegistryError("Registry not initialized. Call initialize_registry() first.")

        return self._registries["connectors"].get(name)

    def list_connectors(self) -> list[str]:
        """Get list of all registered connector names.

        :return: List of connector names
        :rtype: list[str]
        """
        return list(self._registries["connectors"].keys())

    @property
    def connectors(self) -> dict[str, type[Any]]:
        """Get all registered connectors as a dictionary.

        :return: Dictionary mapping connector names to connector classes
        :rtype: dict[str, Type]
        """
        return self._registries["connectors"].copy()

    def get_ariel_search_module(self, name: str) -> Any | None:
        """Retrieve an ARIEL search module by registry name.

        :param name: Registry name, e.g., ``"keyword"``
        :return: Imported module if registered, None otherwise
        """
        return self._registries["ariel_search_modules"].get(name)

    def list_ariel_search_modules(self) -> list[str]:
        """List registered ARIEL search module names.

        :return: List of search module names
        """
        return list(self._registries["ariel_search_modules"].keys())

    def get_ariel_search_module_registry(self) -> dict[str, str]:
        """Get search module name → module_path mapping for ARIEL consumers.

        :return: Dict mapping names to module paths
        """
        result = {}
        for reg in self.config.ariel_search_modules:
            if reg.name in self._registries["ariel_search_modules"]:
                result[reg.name] = reg.module_path
        return result

    def get_ariel_enhancement_module(self, name: str) -> tuple[type, Any] | None:
        """Retrieve an ARIEL enhancement module class and registration.

        :param name: Registry name, e.g., ``"text_embedding"``
        :return: Tuple of (class, registration) if registered, None otherwise
        """
        return self._registries["ariel_enhancement_modules"].get(name)

    def list_ariel_enhancement_modules(self) -> list[str]:
        """List registered ARIEL enhancement module names sorted by execution_order.

        :return: List of enhancement module names in execution order
        """
        entries = []
        for name, (_cls, reg) in self._registries["ariel_enhancement_modules"].items():
            entries.append((reg.execution_order, name))
        entries.sort()
        return [name for _, name in entries]

    def get_ariel_pipeline(self, name: str) -> Any | None:
        """Retrieve an ARIEL pipeline module by registry name.

        :param name: Registry name, e.g., ``"rag"``
        :return: Imported module if registered, None otherwise
        """
        return self._registries["ariel_pipelines"].get(name)

    def list_ariel_pipelines(self) -> list[str]:
        """List registered ARIEL pipeline names.

        :return: List of pipeline names
        """
        return list(self._registries["ariel_pipelines"].keys())

    def get_ariel_ingestion_adapter(self, name: str) -> tuple[type, Any] | None:
        """Retrieve an ARIEL ingestion adapter class and registration.

        :param name: Registry name, e.g., ``"als_logbook"``
        :return: Tuple of (class, registration) if registered, None otherwise
        """
        return self._registries["ariel_ingestion_adapters"].get(name)

    def list_ariel_ingestion_adapters(self) -> list[str]:
        """List registered ARIEL ingestion adapter names.

        :return: List of ingestion adapter names
        """
        return list(self._registries["ariel_ingestion_adapters"].keys())

    def get_service(self, name: str) -> Any | None:
        """Retrieve registered service graph by name.

        :param name: Unique service name from registration
        :type name: str
        :return: Compiled service instance if registered, None otherwise
        :rtype: Any, optional
        """
        return self._registries["services"].get(name)

    def get_all_services(self) -> list[Any]:
        """Retrieve all registered service graph instances.

        :return: List of all registered service graph instances
        :rtype: list[Any]
        """
        return list(self._registries["services"].values())

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_registry_to_json(self, output_dir: str = None) -> dict[str, Any]:
        """Export registry metadata for external tools and plan editors.

        :param output_dir: Directory path for saving JSON files; *None* = data only.
        :return: Complete registry metadata dict.
        """
        return _export_registry_to_json(self.config, self._registries, output_dir)

    # ------------------------------------------------------------------
    # Stats / display
    # ------------------------------------------------------------------

    def _get_initialization_summary(self) -> str:
        """Generate user-friendly initialization summary.

        :return: Formatted initialization summary
        :rtype: str
        """
        stats = self.get_stats()

        summary_lines = [
            "Registry initialization complete!",
            "   Components loaded:",
            f"      • {stats['services']} services: {', '.join(stats['service_names'])}",
        ]

        return "\n".join(summary_lines)

    def get_stats(self) -> dict[str, Any]:
        """Retrieve comprehensive registry statistics for debugging.

        :return: Dictionary containing counts and lists of registered components
        :rtype: dict[str, Any]
        """
        return {
            "initialized": self._initialized,
            "services": len(self._registries["services"]),
            "service_names": list(self._registries["services"].keys()),
        }

    def clear(self) -> None:
        """Clear all registry data and reset initialization state.

        .. warning::
           Clears all registered components. Only use for testing
           or complete registry reset scenarios.
        """
        logger.debug("Clearing registry")
        for registry in self._registries.values():
            registry.clear()
        self._initialized = False


# ======================================================================
# Module-level singleton
# ======================================================================

_registry: RegistryManager | None = None
_registry_config_path: str | None = None


def get_registry(config_path: str | None = None) -> RegistryManager:
    """Return the global registry singleton, creating it on first access.

    :param config_path: Optional config path used on first creation only.
    :return: The global :class:`RegistryManager` (may not yet be initialized).
    :raises RuntimeError: If registry creation fails.
    """
    global _registry, _registry_config_path

    if _registry is None:
        logger.debug("Creating new registry instance...")
        _registry_config_path = config_path
        _registry = _create_registry_from_config(config_path)
    else:
        logger.debug("Using existing registry instance...")

    return _registry


def _create_registry_from_config(config_path: str | None = None) -> RegistryManager:
    """Create registry manager from global configuration.

    Supports multiple configuration formats for registry path specification:

    1. Environment variable (highest priority, for container overrides):
       REGISTRY_PATH=/jupyter/repo_src/my_app/registry.py

    2. Top-level format (simple, for single-app projects):
       registry_path: ./src/my_app/registry.py

    3. Nested format (standard, recommended):
       application:
         registry_path: ./src/my_app/registry.py

    :param config_path: Optional explicit path to configuration file
    :return: Configured registry manager with registry paths
    :rtype: RegistryManager
    :raises ConfigurationError: If configuration format is invalid
    """
    import os

    logger.debug("Creating registry from config...")
    try:
        registry_path = None

        env_registry_path = os.environ.get("REGISTRY_PATH")
        if env_registry_path:
            registry_path = env_registry_path
            logger.info(
                f"Using registry path from REGISTRY_PATH environment variable: "
                f"{registry_path}"
            )
            return RegistryManager(registry_path=registry_path)

        if config_path:
            from osprey.utils.config import get_config_builder

            get_config_builder(config_path=config_path, set_as_default=True)
            logger.debug(f"Set {config_path} as default configuration")

        base_path = None
        if config_path:
            project_root = get_config_value("project_root", None)
            if project_root:
                base_path = Path(project_root)
                logger.debug(f"Using project_root from config as base path: {base_path}")
            else:
                base_path = Path(config_path).resolve().parent
                logger.debug(f"Using config file directory as base path: {base_path}")

        def resolve_registry_path(path: str) -> str:
            """Resolve registry path, handling relative paths correctly."""
            if base_path and not Path(path).is_absolute():
                resolved = (base_path / path).resolve()
                logger.debug(f"Resolved registry path '{path}' -> '{resolved}'")
                return str(resolved)
            return path

        registry_path = get_config_value("registry_path", None)

        if not registry_path:
            application = get_config_value("application", None)
            if application and isinstance(application, dict):
                registry_path = application.get("registry_path")

        if registry_path:
            registry_path = resolve_registry_path(registry_path)
            logger.info(f"Using application registry: {registry_path}")
        else:
            logger.info("No application registry configured - using framework-only registry")

        return RegistryManager(registry_path=registry_path)

    except Exception as e:
        logger.error(f"Failed to create registry from config: {e}")
        raise RuntimeError(f"Registry creation failed: {e}") from e


def initialize_registry(
    auto_export: bool = True, config_path: str | None = None, silent: bool = False
) -> None:
    """Initialize the global registry and load all components.

    Idempotent -- subsequent calls are no-ops once initialization succeeds.

    :param auto_export: Export registry metadata to JSON after init.
    :param config_path: Optional config file path for registry creation.
    :param silent: Suppress INFO/DEBUG logging during init.
    :raises RegistryError: If component loading fails.
    """
    registry = get_registry(config_path=config_path)
    registry.initialize(silent=silent)

    try:
        from osprey.connectors.control_system.limits_validator import LimitsValidator

        limits_validator = LimitsValidator.from_config()
        if limits_validator:
            logger.info(
                f"✅ Channel limits database loaded: "
                f"{len(limits_validator.limits)} channels configured"
            )
    except Exception as e:
        logger.debug(f"Channel limits database not loaded: {e}")
        logger.debug("Runtime limits validation will use safe defaults")

    if auto_export:
        try:
            export_dir = Path(get_agent_dir("registry_exports_dir"))
            export_dir.mkdir(parents=True, exist_ok=True)
            registry.export_registry_to_json(str(export_dir))
        except Exception as e:
            logger.warning(f"Failed to auto-export registry data: {e}")


def reset_registry() -> None:
    """Clear the global registry singleton so the next access creates a fresh one.

    Primarily used for test isolation.
    """
    global _registry, _registry_config_path
    if _registry:
        _registry.clear()
    _registry = None
    _registry_config_path = None
