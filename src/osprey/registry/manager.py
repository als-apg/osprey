"""Registry manager: lazy-loading, dependency-ordered component registry.

Provides :class:`RegistryManager` plus the global singleton helpers
:func:`get_registry`, :func:`initialize_registry`, and :func:`reset_registry`.

.. seealso:: :doc:`/developer-guides/registry-system`
"""

import importlib
import inspect
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from osprey.errors import ConfigurationError, RegistryError
from osprey.utils.config import get_agent_dir, get_config_value
from osprey.utils.logger import get_logger

from .base import RegistryConfig, RegistryConfigProvider

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

        # Provider exclusions are deferred: names are only known after class
        # introspection at init time, not at merge time.
        self._excluded_provider_names = []
        self.config = self._build_merged_configuration()

    def _build_merged_configuration(self) -> RegistryConfig:
        """Build configuration from framework and/or application registry.

        Supports two registry modes based on type detection:

        **Standalone Mode** (RegistryConfig):
            Application provides complete registry with ALL components.
            Framework registry is NOT loaded. Application is responsible for
            providing all framework components (nodes, capabilities, etc.).

        **Extend Mode** (ExtendedRegistryConfig):
            Application extends framework defaults via extend_framework_registry().
            Framework registry is loaded first, then application components are
            merged, with applications able to override framework components.

        The mode is detected automatically based on the type returned by the
        application's get_registry_config() method.

        :return: Complete registry configuration
        :rtype: RegistryConfig
        :raises RegistryError: If registry loading fails
        """
        from pathlib import Path

        from .base import ExtendedRegistryConfig

        if not self.registry_path:
            logger.info("Built framework-only registry (no application)")
            return self._load_registry_from_module("osprey.registry.registry")

        try:
            app_config = self._load_registry_from_path(self.registry_path)
            app_name = Path(self.registry_path).resolve().parent.name

            if app_config is None:
                raise RegistryError(
                    f"Registry provider in '{app_name}' returned None. "
                    f"get_registry_config() must return a RegistryConfig instance."
                )

            if isinstance(app_config, ExtendedRegistryConfig):
                logger.info(f"Extending framework registry with application '{app_name}'")

                framework_config = self._load_registry_from_module("osprey.registry.registry")
                merged = RegistryConfig(
                    services=framework_config.services.copy(),
                    providers=framework_config.providers.copy(),
                    connectors=framework_config.connectors.copy(),
                    ariel_search_modules=framework_config.ariel_search_modules.copy(),
                    ariel_enhancement_modules=framework_config.ariel_enhancement_modules.copy(),
                    ariel_pipelines=framework_config.ariel_pipelines.copy(),
                    ariel_ingestion_adapters=framework_config.ariel_ingestion_adapters.copy(),
                    initialization_order=framework_config.initialization_order.copy(),
                )

                self._merge_application_with_override(merged, app_config, app_name)
                logger.info(
                    f"Loaded application registry from: {self.registry_path} (app: {app_name})"
                )
                return merged

            else:
                logger.info(
                    f"Using standalone registry from application '{app_name}' (framework registry skipped)"
                )
                return app_config

        except Exception as e:
            logger.error(f"Failed to load registry from {self.registry_path}: {e}")
            raise RegistryError(f"Failed to load registry from {self.registry_path}: {e}") from e

    def _load_registry_from_module(self, module_path: str) -> RegistryConfig:
        """Generic registry loader using interface pattern.

        Convention: Module must contain exactly one class implementing
        RegistryConfigProvider interface. Used by both framework and applications.

        :param module_path: Python module path (e.g., 'framework.registry.registry')
        :type module_path: str
        :return: Registry configuration
        :rtype: RegistryConfig
        :raises RegistryError: If registry cannot be loaded or interface not implemented
        """
        if module_path.startswith("applications."):
            component_name = f"{module_path.split('.')[1]} application"
        elif module_path == "osprey.registry.registry":
            component_name = "framework"
        else:
            component_name = f"module {module_path}"

        try:
            registry_module = importlib.import_module(module_path)

            provider_classes = []
            for name in dir(registry_module):
                obj = getattr(registry_module, name)
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, RegistryConfigProvider)
                    and obj != RegistryConfigProvider
                ):
                    provider_classes.append(obj)

            if len(provider_classes) == 0:
                raise RegistryError(
                    f"No RegistryConfigProvider implementation found in {module_path}. "
                    f"{component_name} must define exactly one class implementing RegistryConfigProvider. "
                    f"Import: from osprey.registry import RegistryConfigProvider"
                )
            elif len(provider_classes) > 1:
                class_names = [cls.__name__ for cls in provider_classes]
                raise RegistryError(
                    f"Multiple RegistryConfigProvider implementations found in {module_path}: {class_names}. "
                    f"{component_name} must define exactly one provider class."
                )

            provider_class = provider_classes[0]
            provider_instance = provider_class()
            config = provider_instance.get_registry_config()

            logger.debug(
                f"Loaded {component_name} registry via {provider_class.__name__} from {module_path}"
            )
            return config

        except ImportError as e:
            raise RegistryError(
                f"Failed to import {component_name} registry module {module_path}: {e}"
            ) from e
        except Exception as e:
            raise RegistryError(
                f"Failed to load {component_name} registry from {module_path}: {e}"
            ) from e

    def _load_registry_from_path(self, registry_path: str) -> RegistryConfig:
        """Load registry from a filesystem path using ``importlib.util``.

        Adds the appropriate parent directory to ``sys.path`` so that registry
        module references (e.g. ``app_name.context_classes``) resolve correctly.

        :param registry_path: Absolute or relative path to ``registry.py``.
        :return: Registry configuration from the file.
        :raises RegistryError: If file not found, invalid, or no provider found.
        """
        import importlib.util
        import sys
        from pathlib import Path

        # Normalize path (handles absolute/relative, resolves .., etc.)
        path = Path(registry_path).resolve()

        # Validate file exists
        if not path.exists():
            raise RegistryError(
                f"Registry file not found: {registry_path}\n"
                f"Resolved path: {path}\n"
                f"Current directory: {Path.cwd()}"
            )

        if not path.is_file():
            raise RegistryError(
                f"Registry path is not a file: {registry_path}\nResolved path: {path}"
            )

        # Add parent directory to sys.path so registry module references resolve.
        app_dir = path.parent
        project_root = app_dir.parent  # One level up (e.g., ./src/ or project root)

        search_dir = None
        detection_reason = None

        if project_root.name == "src":
            search_dir = project_root
            detection_reason = "registry is in src/ directory structure"
        elif (project_root / "src").exists() and (project_root / "src").is_dir():
            search_dir = project_root / "src"
            detection_reason = "src/ directory exists in project root"
        else:
            search_dir = app_dir
            detection_reason = "flat project structure (no src/ directory)"

        search_dir_str = str(search_dir.resolve())
        if search_dir_str not in sys.path:
            sys.path.insert(0, search_dir_str)
            logger.info(f"Registry: Added {search_dir_str} to sys.path ({detection_reason})")
            logger.debug(
                f"Registry: Path detection details:\n"
                f"  - Registry file: {path}\n"
                f"  - App directory: {app_dir}\n"
                f"  - Project root: {project_root}\n"
                f"  - Added to sys.path: {search_dir_str}"
            )
        else:
            logger.debug(f"Registry: {search_dir_str} already in sys.path ({detection_reason})")

        try:
            spec = importlib.util.spec_from_file_location(
                "_dynamic_registry",
                path,
            )

            if spec is None or spec.loader is None:
                raise RegistryError(
                    f"Could not create module spec from registry file: {registry_path}\n"
                    f"This usually indicates a corrupted or invalid Python file."
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules["_dynamic_registry"] = module
            spec.loader.exec_module(module)

        except Exception as e:
            raise RegistryError(
                f"Failed to load Python module from {registry_path}: {e}\n"
                f"Ensure the file contains valid Python code and no syntax errors."
            ) from e

        provider_classes = []
        for name in dir(module):
            obj = getattr(module, name)
            if (
                inspect.isclass(obj)
                and issubclass(obj, RegistryConfigProvider)
                and obj != RegistryConfigProvider
            ):
                provider_classes.append(obj)

        if len(provider_classes) == 0:
            raise RegistryError(
                f"No RegistryConfigProvider implementation found in {registry_path}.\n"
                f"Registry files must define exactly one class implementing RegistryConfigProvider.\n"
                f"Example:\n"
                f"  from osprey.registry import RegistryConfigProvider, RegistryConfig\n"
                f"  \n"
                f"  class MyRegistryProvider(RegistryConfigProvider):\n"
                f"      def get_registry_config(self) -> RegistryConfig:\n"
                f"          return RegistryConfig(...)"
            )
        elif len(provider_classes) > 1:
            class_names = [cls.__name__ for cls in provider_classes]
            raise RegistryError(
                f"Multiple RegistryConfigProvider implementations found in {registry_path}: {class_names}.\n"
                f"Registry files must define exactly one provider class.\n"
                f"Found {len(provider_classes)} classes: {', '.join(class_names)}"
            )

        try:
            provider_class = provider_classes[0]
            provider_instance = provider_class()
            config = provider_instance.get_registry_config()

            app_name = path.parent.name
            logger.debug(
                f"Loaded registry via {provider_class.__name__} from {registry_path} "
                f"(application: {app_name})"
            )
            return config

        except Exception as e:
            raise RegistryError(
                f"Failed to instantiate or get config from {provider_classes[0].__name__} "
                f"in {registry_path}: {e}"
            ) from e

    def _apply_framework_exclusions(
        self, merged: RegistryConfig, exclusions: dict[str, list[str]], app_name: str
    ) -> None:
        """Apply framework component exclusions to the merged registry configuration.

        Removes specified framework components from the merged configuration based on
        exclusion rules defined by the application. This allows applications to disable
        framework components they don't need or want to replace with custom implementations.

        :param merged: Merged registry configuration to modify
        :type merged: RegistryConfig
        :param exclusions: Component exclusions by type (e.g., {'capabilities': ['python']})
        :type exclusions: Dict[str, List[str]]
        :param app_name: Application name for logging purposes
        :type app_name: str
        """
        for component_type, excluded_names in exclusions.items():
            if not excluded_names:
                continue

            # Handle provider exclusions specially (names are introspected after loading)
            if component_type == "providers":
                self._excluded_provider_names.extend(excluded_names)
                logger.info(
                    f"Application {app_name} will exclude framework providers: {excluded_names}"
                )
                continue

            # Get the component collection from merged config
            component_collection = getattr(merged, component_type, None)
            if component_collection is None:
                logger.warning(
                    f"Application {app_name} tried to exclude unknown component type: {component_type}"
                )
                continue

            original_count = len(component_collection)
            filtered_components = [
                comp for comp in component_collection if comp.name not in excluded_names
            ]
            setattr(merged, component_type, filtered_components)

            excluded_count = original_count - len(filtered_components)
            if excluded_count > 0:
                actually_excluded = [
                    name
                    for name in excluded_names
                    if name in {comp.name for comp in component_collection}
                ]
                if actually_excluded:
                    logger.info(
                        f"Application {app_name} excluded framework {component_type}: {actually_excluded}"
                    )

    def _merge_application_with_override(
        self, merged: RegistryConfig, app_config: RegistryConfig, app_name: str
    ) -> None:
        """Merge application configuration with framework, allowing overrides.

        Applications can override framework components by providing components
        with the same name. This supports customization and extension patterns.
        Enhanced with robust attribute checking for missing components.
        """
        framework_exclusions = getattr(app_config, "framework_exclusions", {})
        if framework_exclusions:
            self._apply_framework_exclusions(merged, framework_exclusions, app_name)

        framework_service_names = {service.name for service in merged.services}
        service_overrides = []

        app_services = getattr(app_config, "services", [])
        for app_service in app_services:
            if app_service.name in framework_service_names:
                merged.services = [
                    service for service in merged.services if service.name != app_service.name
                ]
                service_overrides.append(app_service.name)
            merged.services.append(app_service)

        if service_overrides:
            logger.info(f"Application {app_name} overrode framework services: {service_overrides}")

        framework_provider_keys = {(p.module_path, p.class_name) for p in merged.providers}
        provider_overrides = []
        providers_added = []

        app_providers = getattr(app_config, "providers", [])
        for app_provider in app_providers:
            provider_key = (app_provider.module_path, app_provider.class_name)
            if provider_key in framework_provider_keys:
                merged.providers = [
                    p for p in merged.providers if (p.module_path, p.class_name) != provider_key
                ]
                provider_overrides.append(f"{app_provider.module_path}.{app_provider.class_name}")
                merged.providers.append(app_provider)
            else:
                providers_added.append(f"{app_provider.module_path}.{app_provider.class_name}")
                merged.providers.append(app_provider)

        if provider_overrides:
            logger.info(
                f"Application {app_name} overrode framework providers: {provider_overrides}"
            )
        if providers_added:
            logger.info(f"Application {app_name} added {len(providers_added)} new provider(s)")

        framework_connector_names = {conn.name for conn in merged.connectors}
        connector_overrides = []
        connectors_added = []

        app_connectors = getattr(app_config, "connectors", [])
        for app_connector in app_connectors:
            if app_connector.name in framework_connector_names:
                merged.connectors = [
                    conn for conn in merged.connectors if conn.name != app_connector.name
                ]
                connector_overrides.append(app_connector.name)
                merged.connectors.append(app_connector)
            else:
                connectors_added.append(app_connector.name)
                merged.connectors.append(app_connector)

        if connector_overrides:
            logger.info(
                f"Application {app_name} overrode framework connectors: {connector_overrides}"
            )
        if connectors_added:
            logger.info(
                f"Application {app_name} added {len(connectors_added)} new connector(s): {connectors_added}"
            )

        self._merge_named_registrations(
            merged.ariel_search_modules,
            getattr(app_config, "ariel_search_modules", []),
            "ARIEL search module",
            app_name,
        )
        self._merge_named_registrations(
            merged.ariel_enhancement_modules,
            getattr(app_config, "ariel_enhancement_modules", []),
            "ARIEL enhancement module",
            app_name,
        )
        self._merge_named_registrations(
            merged.ariel_pipelines,
            getattr(app_config, "ariel_pipelines", []),
            "ARIEL pipeline",
            app_name,
        )
        self._merge_named_registrations(
            merged.ariel_ingestion_adapters,
            getattr(app_config, "ariel_ingestion_adapters", []),
            "ARIEL ingestion adapter",
            app_name,
        )

    @staticmethod
    def _merge_named_registrations(
        merged_list: list,
        app_list: list,
        type_label: str,
        app_name: str,
    ) -> None:
        """Merge named registrations with override support.

        For each item in ``app_list``, if a registration with the same ``.name``
        exists in ``merged_list`` it is replaced (override); otherwise the item
        is appended (addition).  ``merged_list`` is mutated in place.

        Args:
            merged_list: Framework registrations (mutated in place).
            app_list: Application registrations to merge in.
            type_label: Human-readable type label for log messages.
            app_name: Application name for log messages.
        """
        framework_names = {item.name for item in merged_list}
        overrides: list[str] = []
        additions: list[str] = []

        for app_item in app_list:
            if app_item.name in framework_names:
                merged_list[:] = [m for m in merged_list if m.name != app_item.name]
                overrides.append(app_item.name)
            else:
                additions.append(app_item.name)
            merged_list.append(app_item)

        if overrides:
            logger.info(f"Application {app_name} overrode framework {type_label}s: {overrides}")
        if additions:
            logger.info(
                f"Application {app_name} added {len(additions)} new {type_label}(s): {additions}"
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
            # Restore original logging levels
            if silent:
                for logger_name, level in original_levels.items():
                    logging.getLogger(logger_name).setLevel(level)

    def _initialize_component_type(self, component_type: str) -> None:
        """Initialize components of a specific type.

        :param component_type: Type of components to initialize (context_classes, capabilities, etc.)
        :type component_type: str
        :raises ValueError: If component_type is not recognized
        """
        if component_type == "providers":
            self._initialize_providers()
        elif component_type == "services":
            self._initialize_services()
        elif component_type == "connectors":
            self._initialize_connectors()
        elif component_type == "ariel_search_modules":
            self._initialize_ariel_search_modules()
        elif component_type == "ariel_enhancement_modules":
            self._initialize_ariel_enhancement_modules()
        elif component_type == "ariel_pipelines":
            self._initialize_ariel_pipelines()
        elif component_type == "ariel_ingestion_adapters":
            self._initialize_ariel_ingestion_adapters()
        else:
            raise ValueError(f"Unknown component type: {component_type}")

    def _get_configured_provider_names(self) -> set[str] | None:
        """Extract provider names from config.yml models section.

        Returns None if config unavailable (load all as fallback).
        """
        try:
            model_configs = get_config_value("models", None)
            if not model_configs or not isinstance(model_configs, dict):
                return None
            providers = set()
            for role_config in model_configs.values():
                if isinstance(role_config, dict) and "provider" in role_config:
                    providers.add(role_config["provider"])
            return providers if providers else None
        except Exception:
            return None  # Config not available (framework-only mode)

    def _initialize_providers(self) -> None:
        """Initialize AI model providers via the lightweight ProviderRegistry.

        Delegates to ``ProviderRegistry.load_providers()`` for the actual import
        loop, then stores the results in the RegistryManager's internal dict for
        backward compatibility. Custom providers from ``self.config.providers``
        (non-built-in) are registered first so they participate in the bulk load.

        Uses config-driven filtering to skip imports for unconfigured providers,
        avoiding costly module-level network calls on air-gapped machines.
        """
        from osprey.models.provider_registry import get_provider_registry

        pr = get_provider_registry()

        for registration in self.config.providers:
            if registration.name and registration.name not in pr.list_providers():
                pr.register_provider(
                    registration.name,
                    registration.module_path,
                    registration.class_name,
                )

        configured_providers = self._get_configured_provider_names()

        if configured_providers is not None:
            logger.info(
                f"Initializing providers (config-driven: {sorted(configured_providers)})..."
            )
        else:
            logger.info("Initializing providers (all available)...")

        loaded = pr.load_providers(
            configured_names=configured_providers,
            excluded_names=self._excluded_provider_names or None,
        )

        for name, provider_class in loaded.items():
            self._registries["providers"][name] = provider_class
            logger.info(f"  ✓ Registered provider: {name}")

        logger.info(
            f"Provider initialization complete: "
            f"{len(self._registries['providers'])} providers loaded"
        )

    def _initialize_connectors(self) -> None:
        """Initialize control system and archiver connectors from registry configuration.

        Loads connector classes and registers them with ConnectorFactory for runtime use.
        This integrates the connector system with the registry, providing unified management
        of all framework components while maintaining the factory pattern for runtime connector creation.

        :raises RegistryError: If connector class cannot be imported or registered
        """
        logger.info(f"Initializing {len(self.config.connectors)} connector(s)...")

        # Import ConnectorFactory for registration
        try:
            from osprey.connectors.factory import ConnectorFactory
        except ImportError as e:
            logger.error(f"Failed to import ConnectorFactory: {e}")
            raise RegistryError(
                "ConnectorFactory not available - connector system may not be installed"
            ) from e

        for registration in self.config.connectors:
            try:
                module = importlib.import_module(registration.module_path)
                connector_class = getattr(module, registration.class_name)

                if registration.connector_type == "control_system":
                    ConnectorFactory.register_control_system(registration.name, connector_class)
                elif registration.connector_type == "archiver":
                    ConnectorFactory.register_archiver(registration.name, connector_class)
                else:
                    raise RegistryError(
                        f"Unknown connector type: {registration.connector_type}. "
                        f"Must be 'control_system' or 'archiver'"
                    )

                self._registries["connectors"][registration.name] = connector_class

                logger.info(
                    f"  ✓ Registered {registration.connector_type} connector: {registration.name}"
                )
                logger.debug(f"    - Description: {registration.description}")
                logger.debug(f"    - Module: {registration.module_path}")
                logger.debug(f"    - Class: {registration.class_name}")

            except ImportError as e:
                # Some connectors may require optional dependencies (e.g., pyepics)
                # Log as warning but don't fail initialization
                logger.warning(f"  ⊘ Skipping connector '{registration.name}' (import failed): {e}")
                logger.debug(f"    Connector {registration.name} may require optional dependencies")
            except Exception as e:
                logger.error(f"  ✗ Failed to register connector '{registration.name}': {e}")
                raise RegistryError(f"Connector registration failed for {registration.name}") from e

        logger.info(
            f"Connector initialization complete: {len(self._registries['connectors'])} connectors loaded"
        )

    def _initialize_ariel_search_modules(self) -> None:
        """Initialize ARIEL search modules from registry configuration.

        Imports each search module and validates it exports a ``get_tool_descriptor`` callable.
        Stores the imported module in the ``ariel_search_modules`` registry.
        """
        if not self.config.ariel_search_modules:
            return

        logger.info(
            f"Initializing {len(self.config.ariel_search_modules)} ARIEL search module(s)..."
        )

        for registration in self.config.ariel_search_modules:
            try:
                module = importlib.import_module(registration.module_path)

                if not hasattr(module, "get_tool_descriptor") or not callable(
                    module.get_tool_descriptor
                ):
                    raise RegistryError(
                        f"ARIEL search module '{registration.name}' at {registration.module_path} "
                        f"must export a callable get_tool_descriptor()"
                    )

                self._registries["ariel_search_modules"][registration.name] = module
                logger.debug(f"  ✓ Registered ARIEL search module: {registration.name}")

            except ImportError as e:
                logger.warning(
                    f"  ⊘ Skipping ARIEL search module '{registration.name}' (import failed): {e}"
                )
            except Exception as e:
                logger.error(
                    f"  ✗ Failed to register ARIEL search module '{registration.name}': {e}"
                )
                raise RegistryError(
                    f"ARIEL search module registration failed for {registration.name}"
                ) from e

        logger.info(
            f"ARIEL search module initialization complete: "
            f"{len(self._registries['ariel_search_modules'])} modules loaded"
        )

    def _initialize_ariel_enhancement_modules(self) -> None:
        """Initialize ARIEL enhancement modules from registry configuration.

        Imports each enhancement module class and stores (class, registration) tuples
        in the ``ariel_enhancement_modules`` registry.
        """
        if not self.config.ariel_enhancement_modules:
            return

        logger.info(
            f"Initializing {len(self.config.ariel_enhancement_modules)} "
            f"ARIEL enhancement module(s)..."
        )

        for registration in self.config.ariel_enhancement_modules:
            try:
                module = importlib.import_module(registration.module_path)
                cls = getattr(module, registration.class_name)

                self._registries["ariel_enhancement_modules"][registration.name] = (
                    cls,
                    registration,
                )
                logger.debug(f"  ✓ Registered ARIEL enhancement module: {registration.name}")

            except ImportError as e:
                logger.warning(
                    f"  ⊘ Skipping ARIEL enhancement module '{registration.name}' "
                    f"(import failed): {e}"
                )
            except Exception as e:
                logger.error(
                    f"  ✗ Failed to register ARIEL enhancement module '{registration.name}': {e}"
                )
                raise RegistryError(
                    f"ARIEL enhancement module registration failed for {registration.name}"
                ) from e

        logger.info(
            f"ARIEL enhancement module initialization complete: "
            f"{len(self._registries['ariel_enhancement_modules'])} modules loaded"
        )

    def _initialize_ariel_pipelines(self) -> None:
        """Initialize ARIEL pipelines from registry configuration.

        Imports each pipeline module and validates it exports a ``get_pipeline_descriptor``
        callable. Stores the imported module in the ``ariel_pipelines`` registry.
        """
        if not self.config.ariel_pipelines:
            return

        logger.info(f"Initializing {len(self.config.ariel_pipelines)} ARIEL pipeline(s)...")

        for registration in self.config.ariel_pipelines:
            try:
                module = importlib.import_module(registration.module_path)

                if not hasattr(module, "get_pipeline_descriptor") or not callable(
                    module.get_pipeline_descriptor
                ):
                    raise RegistryError(
                        f"ARIEL pipeline '{registration.name}' at {registration.module_path} "
                        f"must export a callable get_pipeline_descriptor()"
                    )

                self._registries["ariel_pipelines"][registration.name] = module
                logger.debug(f"  ✓ Registered ARIEL pipeline: {registration.name}")

            except ImportError as e:
                logger.warning(
                    f"  ⊘ Skipping ARIEL pipeline '{registration.name}' (import failed): {e}"
                )
            except Exception as e:
                logger.error(f"  ✗ Failed to register ARIEL pipeline '{registration.name}': {e}")
                raise RegistryError(
                    f"ARIEL pipeline registration failed for {registration.name}"
                ) from e

        logger.info(
            f"ARIEL pipeline initialization complete: "
            f"{len(self._registries['ariel_pipelines'])} pipelines loaded"
        )

    def _initialize_ariel_ingestion_adapters(self) -> None:
        """Initialize ARIEL ingestion adapters from registry configuration.

        Imports each ingestion adapter class and stores (class, registration) tuples
        in the ``ariel_ingestion_adapters`` registry.
        """
        if not self.config.ariel_ingestion_adapters:
            return

        logger.info(
            f"Initializing {len(self.config.ariel_ingestion_adapters)} "
            f"ARIEL ingestion adapter(s)..."
        )

        for registration in self.config.ariel_ingestion_adapters:
            try:
                module = importlib.import_module(registration.module_path)
                cls = getattr(module, registration.class_name)

                self._registries["ariel_ingestion_adapters"][registration.name] = (
                    cls,
                    registration,
                )
                logger.debug(f"  ✓ Registered ARIEL ingestion adapter: {registration.name}")

            except ImportError as e:
                logger.warning(
                    f"  ⊘ Skipping ARIEL ingestion adapter '{registration.name}' "
                    f"(import failed): {e}"
                )
            except Exception as e:
                logger.error(
                    f"  ✗ Failed to register ARIEL ingestion adapter '{registration.name}': {e}"
                )
                raise RegistryError(
                    f"ARIEL ingestion adapter registration failed for {registration.name}"
                ) from e

        logger.info(
            f"ARIEL ingestion adapter initialization complete: "
            f"{len(self._registries['ariel_ingestion_adapters'])} adapters loaded"
        )

    def _initialize_services(self) -> None:
        """Initialize service registry.

        Services provide specialized functionality that can be invoked by capabilities.
        Each service is instantiated and registered for runtime access.

        :raises Exception: Service initialization failures are logged but don't fail registry
        """
        logger.debug("Initializing services...")
        for reg in self.config.services:
            try:
                module = __import__(reg.module_path, fromlist=[reg.class_name])
                service_class = getattr(module, reg.class_name)
                service_instance = service_class()
                self._registries["services"][reg.name] = service_instance
                logger.debug(f"Registered service: {reg.name}")

            except Exception as e:
                logger.warning(f"Failed to initialize service {reg.name}: {e}")

        logger.info(f"Registered {len(self._registries['services'])} services")

    def export_registry_to_json(self, output_dir: str = None) -> dict[str, Any]:
        """Export registry metadata for external tools and plan editors.

        Creates comprehensive JSON export of all registered components including
        capabilities, context types, and workflow templates. This data is used by
        execution plan editors and other external tools to understand system capabilities.

        :param output_dir: Directory path for saving JSON files, if None returns data only
        :type output_dir: str, optional
        :return: Complete registry metadata with capabilities, context types, and templates
        :rtype: dict[str, Any]
        :raises Exception: If file writing fails when output_dir is specified

        Examples:
            Export to directory::

                >>> registry = get_registry()
                >>> data = registry.export_registry_to_json("/tmp/registry")
                >>> print(f"Exported {data['metadata']['total_capabilities']} capabilities")

            Get data without saving::

                >>> data = registry.export_registry_to_json()
                >>> capabilities = data['capabilities']
        """
        export_data = {
            "connectors": self._export_connectors(),
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "registry_version": "1.0",
                "total_connectors": len(self.config.connectors),
            },
        }

        if output_dir:
            self._save_export_data(export_data, output_dir)

        return export_data

    def _export_connectors(self) -> list[dict[str, Any]]:
        """Export connector metadata for external consumption.

        Transforms internal connector registrations into standardized format
        suitable for documentation tools and system introspection. Exports
        all registered connectors (control system and archiver types).

        :return: List of connector metadata dictionaries
        :rtype: list[dict[str, Any]]
        """
        connectors = []

        for conn_reg in self.config.connectors:
            connector_data = {
                "name": conn_reg.name,
                "connector_type": conn_reg.connector_type,
                "description": conn_reg.description,
                "module_path": conn_reg.module_path,
                "class_name": conn_reg.class_name,
            }
            connectors.append(connector_data)

        return connectors

    def _save_export_data(self, export_data: dict[str, Any], output_dir: str) -> None:
        """Save registry export data to JSON files.

        Creates directory if needed and saves both complete export and individual
        component files for easier access by external tools.

        :param export_data: Complete registry export data
        :type export_data: dict[str, Any]
        :param output_dir: Target directory for JSON files
        :type output_dir: str
        :raises OSError: If directory creation or file writing fails
        :raises Exception: If JSON serialization fails
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            export_file = Path(output_dir) / "registry_export.json"
            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Registry export saved to: {export_file}")
            logger.info(
                f"Export contains: {export_data['metadata']['total_connectors']} connectors"
            )

        except Exception as e:
            logger.error(f"Failed to save export data: {e}")
            raise

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

        Connectors are registered with the ConnectorFactory during registry initialization
        and can also be accessed through the registry for introspection purposes.

        :param name: Unique connector name from registration (e.g., 'epics', 'mock', 'tango')
        :type name: str
        :return: Connector class if registered, None otherwise
        :rtype: Type[ControlSystemConnector] or Type[ArchiverConnector] or None

        Examples:
            >>> registry = get_registry()
            >>> epics_class = registry.get_connector('epics')
            >>> mock_class = registry.get_connector('mock')
        """
        if not self._initialized:
            raise RegistryError("Registry not initialized. Call initialize_registry() first.")

        return self._registries["connectors"].get(name)

    def list_connectors(self) -> list[str]:
        """Get list of all registered connector names.

        :return: List of connector names (includes both control system and archiver connectors)
        :rtype: list[str]

        Examples:
            >>> registry = get_registry()
            >>> connectors = registry.list_connectors()
            >>> print(connectors)  # ['mock', 'epics', 'mock_archiver', 'epics_archiver', ...]
        """
        return list(self._registries["connectors"].keys())

    @property
    def connectors(self) -> dict[str, type[Any]]:
        """Get all registered connectors as a dictionary.

        :return: Dictionary mapping connector names to connector classes
        :rtype: dict[str, Type]

        Examples:
            >>> registry = get_registry()
            >>> all_connectors = registry.connectors
            >>> for name, connector_class in all_connectors.items():
            ...     print(f"{name}: {connector_class}")
        """
        return self._registries["connectors"].copy()

    def get_ariel_search_module(self, name: str) -> Any | None:
        """Retrieve an ARIEL search module by registry name.

        :param name: Registry name, e.g., "keyword"
        :return: Imported module if registered, None otherwise
        """
        return self._registries["ariel_search_modules"].get(name)

    def list_ariel_search_modules(self) -> list[str]:
        """List registered ARIEL search module names.

        :return: List of search module names
        """
        return list(self._registries["ariel_search_modules"].keys())

    def get_ariel_search_module_registry(self) -> dict[str, str]:
        """Get search module name→module_path mapping for ARIEL consumers.

        :return: Dict mapping names to module paths
        """
        result = {}
        for reg in self.config.ariel_search_modules:
            if reg.name in self._registries["ariel_search_modules"]:
                result[reg.name] = reg.module_path
        return result

    def get_ariel_enhancement_module(self, name: str) -> tuple[type, Any] | None:
        """Retrieve an ARIEL enhancement module class and registration.

        :param name: Registry name, e.g., "text_embedding"
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

        :param name: Registry name, e.g., "rag"
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

        :param name: Registry name, e.g., "als_logbook"
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

    def _get_initialization_summary(self) -> str:
        """Generate user-friendly initialization summary.

        Creates a multi-line, readable summary of registry initialization
        that's much more user-friendly than the raw stats dictionary.

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

        Examples:
            Check registry status::

                >>> registry = get_registry()
                >>> stats = registry.get_stats()
                >>> print(f"Loaded {stats['capabilities']} capabilities")
                >>> print(f"Available: {stats['capability_names']}")
        """
        return {
            "initialized": self._initialized,
            "services": len(self._registries["services"]),
            "service_names": list(self._registries["services"].keys()),
        }

    def clear(self) -> None:
        """Clear all registry data and reset initialization state.

        Removes all registered components and marks the registry as uninitialized.
        This method is primarily used for testing to ensure clean state between tests.

        .. warning::
           This method clears all registered components. Only use for testing
           or complete registry reset scenarios.
        """
        logger.debug("Clearing registry")
        for registry in self._registries.values():
            registry.clear()
        self._initialized = False


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

    4. Multiple applications format (advanced):
       applications:
         app1:
           registry_path: ./src/app1/registry.py
         app2:
           registry_path: ./src/app2/registry.py

    5. Legacy list format (deprecated):
       applications:
         - app1
         - app2

    :param config_path: Optional explicit path to configuration file
    :return: Configured registry manager with registry paths
    :rtype: RegistryManager
    :raises ConfigurationError: If configuration format is invalid
    """
    import os
    from pathlib import Path

    logger.debug("Creating registry from config...")
    try:
        registry_path = None

        env_registry_path = os.environ.get("REGISTRY_PATH")
        if env_registry_path:
            registry_path = env_registry_path
            logger.info(
                f"Using registry path from REGISTRY_PATH environment variable: {registry_path}"
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
                # Resolve relative path against base_path
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
        from osprey.services.python_executor.execution.limits_validator import LimitsValidator

        limits_validator = LimitsValidator.from_config()
        if limits_validator:
            logger.info(
                f"✅ Channel limits database loaded: {len(limits_validator.limits)} channels configured"
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
