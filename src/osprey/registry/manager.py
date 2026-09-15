"""Registry manager: lazy-loading, dependency-ordered component registry.

Provides :class:`RegistryManager` plus the global singleton helpers
:func:`get_registry`, :func:`initialize_registry`, and :func:`reset_registry`.

.. seealso:: :doc:`/developer-guides/registry-system`
"""

import logging
import os
from collections.abc import Mapping
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
    services, providers, and connectors throughout the framework. One unified
    registry eliminates circular imports through lazy loading and provides
    dependency-ordered initialization.

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
            "ariel_search_modules": {},
            "ariel_enhancement_modules": {},
            "ariel_ingestion_adapters": {},
        }

        self.config, self._excluded_provider_names = build_merged_configuration(registry_path)

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
            raise RegistryError(f"Unexpected error during registry initialization: {e}") from e
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


#: Environment variable that names the application registry file, outranking
#: every config spelling. Containers set it to point at a mounted registry.
REGISTRY_PATH_ENV = "REGISTRY_PATH"


def resolve_registry_path(
    config: Mapping[str, Any] | None = None,
    *,
    base_path: Path | str | None = None,
) -> str | None:
    """Resolve the application registry file path — the one resolver for it.

    Three spellings reach the same file, and every reader must agree on which
    one is in effect: the registry loader, the health row that reports whether
    that file is there, and anything else that asks whether a deployment has an
    application registry at all. A reader that knows only one spelling reports
    on a different deployment than the one that loaded.

    Lookup order, first hit wins:

    1. the ``REGISTRY_PATH`` environment variable — the container override;
    2. top-level ``registry_path`` — the canonical spelling;
    3. ``application.registry_path`` — an accepted alias.

    ``${VAR}`` in the value is expanded against the environment, and a relative
    path is resolved against *base_path*, defaulting to the config's own
    ``project_root``.

    A relative spelling is the documented one, and the anchor it means is the
    deployment — the same anchor ``data/``, ``plans/`` and ``health.plugins``
    use. Falling back to ``project_root`` is what makes that true for a caller
    with no ``base_path`` to offer: every runtime reader reaches the registry
    through ``get_registry()`` with no config path, and anchoring on the
    working directory instead loaded a deployment's own registry from the repo
    root and nowhere else.

    ``project_root`` is used only when it names a directory that exists here. A
    rendered config carries the ``project_root`` of the environment it was
    rendered for, so one read on another machine — a service's config
    bind-mounted into a container, say — names a path that is not this one's,
    and anchoring on it would be confidently wrong rather than merely
    unanchored.

    Args:
        config: Config mapping to read. ``None`` reads through the global
            config singleton, which is what the registry factory has when it
            was handed only a config path.
        base_path: Directory relative paths resolve against. ``None`` falls
            back to the config's ``project_root``, and leaves the path relative
            when there is none.

    Returns:
        The resolved path, or ``None`` when no spelling names one.
    """
    raw: Any = os.environ.get(REGISTRY_PATH_ENV)
    if not raw:
        if config is None:
            raw = get_config_value("registry_path", None)
            if not raw:
                application = get_config_value("application", None)
                if isinstance(application, Mapping):
                    raw = application.get("registry_path")
        else:
            raw = config.get("registry_path")
            if not raw:
                application = config.get("application")
                if isinstance(application, Mapping):
                    raw = application.get("registry_path")

    if not raw or not isinstance(raw, str):
        return None

    expanded: str = os.path.expandvars(str(raw))
    if base_path is None:
        base_path = _project_root_anchor(config)
    if base_path is not None and not Path(expanded).is_absolute():
        return str((Path(base_path) / expanded).resolve())
    return expanded


def _project_root_anchor(config: Mapping[str, Any] | None) -> Path | None:
    """Return the config's ``project_root``, when it names a directory here.

    Args:
        config: Config mapping to read, or ``None`` to read through the global
            config singleton.

    Returns:
        The project root as a path, or ``None`` when the config names none or
        names one that does not exist on this machine.
    """
    root = get_config_value("project_root", None) if config is None else config.get("project_root")
    if not root or not isinstance(root, (str, Path)):
        return None
    path = Path(os.path.expandvars(str(root)))
    return path if path.is_dir() else None


def _create_registry_from_config(config_path: str | None = None) -> RegistryManager:
    """Create registry manager from global configuration.

    The path itself comes from :func:`resolve_registry_path`, which every
    reader of it shares (REGISTRY_PATH, then top-level ``registry_path``, then
    ``application.registry_path``).

    :param config_path: Optional explicit path to configuration file
    :return: Configured registry manager with registry paths
    :rtype: RegistryManager
    :raises ConfigurationError: If configuration format is invalid
    """
    logger.debug("Creating registry from config...")
    try:
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

        registry_path = resolve_registry_path(base_path=base_path)

        if registry_path:
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
