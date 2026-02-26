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
from typing import TYPE_CHECKING, Any, Optional

from osprey.base.errors import ConfigurationError, RegistryError
from osprey.utils.config import get_agent_dir, get_config_value
from osprey.utils.logger import get_logger

from .base import RegistryConfig, RegistryConfigProvider

if TYPE_CHECKING:
    from osprey.base import BaseCapability
    from osprey.context import CapabilityContext

logger = get_logger(name="registry", color="sky_blue2")


class RegistryManager:
    """Centralized registry for all Osprey Agentic Framework components.

    This class provides the single point of access for capabilities, nodes, context classes,
    and data sources throughout the framework. It replaces the fragmented registry system
    with a unified approach that eliminates circular imports through lazy loading and
    provides dependency-ordered initialization.

    The registry system follows a strict initialization order to handle dependencies:
    1. Context classes (required by capabilities)
    2. Data sources (required by capabilities)
    3. Core nodes (infrastructure components)
    4. Capabilities (domain-specific functionality)
    5. Framework prompt providers (application-specific prompts)
    6. Workflow templates (predefined execution patterns)

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
            "capabilities": {},
            "nodes": {},
            "contexts": {},
            "data_sources": {},
            "services": {},
            "domain_analyzers": {},
            "execution_policy_analyzers": {},
            "framework_prompt_providers": {},
            "providers": {},
            "connectors": {},
            "code_generators": {},
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
                    core_nodes=framework_config.core_nodes.copy(),
                    capabilities=framework_config.capabilities.copy(),
                    context_classes=framework_config.context_classes.copy(),
                    data_sources=framework_config.data_sources.copy(),
                    services=framework_config.services.copy(),
                    framework_prompt_providers=framework_config.framework_prompt_providers.copy(),
                    providers=framework_config.providers.copy(),
                    connectors=framework_config.connectors.copy(),
                    code_generators=framework_config.code_generators.copy(),
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
            if component_type == "context_classes":
                filtered_components = [
                    comp for comp in component_collection if comp.context_type not in excluded_names
                ]
            else:
                filtered_components = [
                    comp for comp in component_collection if comp.name not in excluded_names
                ]
            setattr(merged, component_type, filtered_components)

            # Log exclusions that actually occurred
            excluded_count = original_count - len(filtered_components)
            if excluded_count > 0:
                if component_type == "context_classes":
                    actually_excluded = [
                        name
                        for name in excluded_names
                        if name in {comp.context_type for comp in component_collection}
                    ]
                else:
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
        context_overrides = []
        existing_context_types = {cls.context_type for cls in merged.context_classes}

        app_context_classes = getattr(app_config, "context_classes", [])
        for app_context in app_context_classes:
            if app_context.context_type in existing_context_types:
                merged.context_classes = [
                    cls
                    for cls in merged.context_classes
                    if cls.context_type != app_context.context_type
                ]
                context_overrides.append(app_context.context_type)
            merged.context_classes.append(app_context)

        if context_overrides:
            logger.info(
                f"Application {app_name} overrode framework context classes: {context_overrides}"
            )

        framework_exclusions = getattr(app_config, "framework_exclusions", {})
        if framework_exclusions:
            self._apply_framework_exclusions(merged, framework_exclusions, app_name)

        capability_overrides = []
        existing_capability_names = {cap.name for cap in merged.capabilities}

        app_capabilities = getattr(app_config, "capabilities", [])
        for app_capability in app_capabilities:
            if app_capability.name in existing_capability_names:
                merged.capabilities = [
                    cap for cap in merged.capabilities if cap.name != app_capability.name
                ]
                capability_overrides.append(app_capability.name)
            merged.capabilities.append(app_capability)

        if capability_overrides:
            # Check for shadow warnings: non-explicit overrides of native capabilities
            native_control_capabilities = {
                "channel_finding",
                "channel_read",
                "channel_write",
                "archiver_retrieval",
            }
            shadow_caps = []
            explicit_caps = []
            for app_capability in app_capabilities:
                if app_capability.name in capability_overrides:
                    is_explicit = getattr(app_capability, "_is_explicit_override", False)
                    if is_explicit:
                        explicit_caps.append(app_capability.name)
                    elif app_capability.name in native_control_capabilities:
                        shadow_caps.append(app_capability.name)

            if shadow_caps:
                logger.warning(
                    f"Application '{app_name}' shadows native framework capabilities: "
                    f"{shadow_caps}. These capabilities are now built into OSPREY. "
                    f"Run 'osprey migrate check' to update your application, or use "
                    f"override_capabilities=[] in your registry to explicitly override."
                )

            if explicit_caps:
                logger.info(
                    f"Application {app_name} explicitly overrode framework capabilities: "
                    f"{explicit_caps}"
                )

            # Log remaining non-native overrides at INFO level
            other_overrides = [
                c for c in capability_overrides if c not in shadow_caps and c not in explicit_caps
            ]
            if other_overrides:
                logger.info(
                    f"Application {app_name} overrode framework capabilities: {other_overrides}"
                )

        framework_ds_names = {ds.name for ds in merged.data_sources}
        ds_overrides = []

        app_data_sources = getattr(app_config, "data_sources", [])
        for app_ds in app_data_sources:
            if app_ds.name in framework_ds_names:
                merged.data_sources = [ds for ds in merged.data_sources if ds.name != app_ds.name]
                ds_overrides.append(app_ds.name)
            merged.data_sources.append(app_ds)

        if ds_overrides:
            logger.info(f"Application {app_name} overrode framework data sources: {ds_overrides}")

        framework_node_names = {node.name for node in merged.core_nodes}
        node_overrides = []

        app_core_nodes = getattr(app_config, "core_nodes", [])
        for app_node in app_core_nodes:
            if app_node.name in framework_node_names:
                merged.core_nodes = [
                    node for node in merged.core_nodes if node.name != app_node.name
                ]
                node_overrides.append(app_node.name)
            merged.core_nodes.append(app_node)

        if node_overrides:
            logger.info(f"Application {app_name} overrode framework nodes: {node_overrides}")

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

        app_prompt_providers = getattr(app_config, "framework_prompt_providers", [])
        merged.framework_prompt_providers.extend(app_prompt_providers)

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

        framework_generator_names = {gen.name for gen in merged.code_generators}
        generator_overrides = []
        generators_added = []

        app_generators = getattr(app_config, "code_generators", [])
        for app_generator in app_generators:
            if app_generator.name in framework_generator_names:
                merged.code_generators = [
                    gen for gen in merged.code_generators if gen.name != app_generator.name
                ]
                generator_overrides.append(app_generator.name)
                merged.code_generators.append(app_generator)
            else:
                generators_added.append(app_generator.name)
                merged.code_generators.append(app_generator)

        if generator_overrides:
            logger.info(
                f"Application {app_name} overrode framework code generators: {generator_overrides}"
            )
        if generators_added:
            logger.info(
                f"Application {app_name} added {len(generators_added)} new code generator(s): {generators_added}"
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

        :param component_type: Type of components to initialize (context_classes, data_sources, etc.)
        :type component_type: str
        :raises ValueError: If component_type is not recognized
        """
        if component_type == "context_classes":
            self._initialize_context_classes()
        elif component_type == "data_sources":
            self._initialize_data_sources()
        elif component_type == "providers":
            self._initialize_providers()
        elif component_type == "services":
            self._initialize_services()
        elif component_type == "capabilities":
            self._initialize_capabilities()
        elif component_type == "framework_prompt_providers":
            self._initialize_framework_prompt_providers()
        elif component_type == "domain_analyzers":
            self._initialize_domain_analyzers()
        elif component_type == "execution_policy_analyzers":
            self._initialize_execution_policy_analyzers()
        elif component_type == "connectors":
            self._initialize_connectors()
        elif component_type == "code_generators":
            self._initialize_code_generators()
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

    def _initialize_context_classes(self) -> None:
        """Initialize context class registry with lazy loading.

        Dynamically imports and registers all context classes defined in the configuration.
        Context classes define the data structures used for inter-capability communication
        and must be available before capability initialization.

        :raises RegistryError: If any context class module cannot be imported
        :raises AttributeError: If specified class name is not found in module
        """
        logger.debug("Initializing context classes...")
        for reg in self.config.context_classes:
            try:
                module = __import__(reg.module_path, fromlist=[reg.class_name])
                context_class = getattr(module, reg.class_name)
                self._registries["contexts"][reg.context_type] = context_class
                logger.debug(
                    f"Registered context class: {reg.context_type} -> {context_class.__name__}"
                )
            except ImportError as e:
                logger.error(
                    f"Failed to import module for context class {reg.context_type}: {reg.module_path}"
                )
                raise RegistryError(
                    f"Cannot import module {reg.module_path} for context class {reg.context_type}: {e}"
                ) from e
            except AttributeError as e:
                logger.error(
                    f"Context class {reg.class_name} not found in module {reg.module_path}"
                )
                raise RegistryError(
                    f"Class {reg.class_name} not found in module {reg.module_path} for context {reg.context_type}: {e}"
                ) from e

        logger.info(f"Registered {len(self.config.context_classes)} context classes")

    def _initialize_data_sources(self) -> None:
        """Initialize data source provider registry with instantiation.

        Dynamically imports and instantiates all data source providers defined in
        the configuration. Data sources provide external data access and must be
        available before capabilities that depend on them.

        Failed data source initialization is logged as warning but does not fail
        the entire registry initialization, allowing partial functionality.

        :raises Exception: Individual data source failures are caught and logged
        """
        logger.debug("Initializing data sources...")
        for reg in self.config.data_sources:
            try:
                module = __import__(reg.module_path, fromlist=[reg.class_name])
                provider_class = getattr(module, reg.class_name)
                provider_instance = provider_class()
                self._registries["data_sources"][reg.name] = provider_instance
                logger.debug(f"Registered data source: {reg.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize data source {reg.name}: {e}")

        logger.info(f"Registered {len(self._registries['data_sources'])} data sources")

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
            return None  # Config not available (TUI init, framework-only mode)

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

    def _initialize_code_generators(self) -> None:
        """Initialize code generators from registry configuration.

        Loads code generator classes and registers them with the generator factory for runtime use.
        This integrates the code generator system with the registry, providing unified management
        of all framework components while maintaining the factory pattern for runtime generator creation.

        Code generators with optional dependencies are handled gracefully - if dependencies are
        missing, the generator is skipped with a warning but initialization continues.

        :raises RegistryError: If code generator class cannot be imported (non-optional deps)
        """
        logger.info(f"Initializing {len(self.config.code_generators)} code generator(s)...")

        for registration in self.config.code_generators:
            try:
                module = importlib.import_module(registration.module_path)
                generator_class = getattr(module, registration.class_name)

                self._registries["code_generators"][registration.name] = {
                    "class": generator_class,
                    "registration": registration,
                }

                logger.info(f"  ✓ Registered code generator: {registration.name}")
                logger.debug(f"    - Description: {registration.description}")
                logger.debug(f"    - Module: {registration.module_path}")
                logger.debug(f"    - Class: {registration.class_name}")
                if registration.optional_dependencies:
                    logger.debug(
                        f"    - Optional dependencies: {registration.optional_dependencies}"
                    )

            except ImportError as e:
                if registration.optional_dependencies:
                    logger.warning(
                        f"  ⊘ Skipping code generator '{registration.name}' "
                        f"(optional dependencies {registration.optional_dependencies} not installed): {e}"
                    )
                    logger.debug(
                        f"    To use '{registration.name}', install: pip install {' '.join(registration.optional_dependencies)}"
                    )
                else:
                    logger.error(f"  ✗ Failed to import code generator '{registration.name}': {e}")
                    raise RegistryError(
                        f"Code generator registration failed for {registration.name}"
                    ) from e
            except Exception as e:
                logger.error(f"  ✗ Failed to register code generator '{registration.name}': {e}")
                raise RegistryError(
                    f"Code generator registration failed for {registration.name}"
                ) from e

        logger.info(
            f"Code generator initialization complete: {len(self._registries['code_generators'])} generators loaded"
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

    def _initialize_execution_policy_analyzers(self) -> None:
        """Initialize execution policy analyzer registry with instantiation.

        Dynamically imports and instantiates all execution policy analyzer classes defined
        in the configuration. These analyzers make execution mode and approval decisions
        based on code analysis results.

        Failed analyzer initialization is logged as warning but does not fail
        the entire registry initialization, allowing fallback to default analyzer.

        :raises Exception: Individual analyzer failures are caught and logged
        """
        logger.debug("Initializing execution policy analyzers...")
        for reg in self.config.execution_policy_analyzers:
            try:
                module = __import__(reg.module_path, fromlist=[reg.class_name])
                analyzer_class = getattr(module, reg.class_name)
                self._registries["execution_policy_analyzers"][reg.name] = {
                    "class": analyzer_class,
                    "registration": reg,
                }
                logger.debug(f"Registered execution policy analyzer: {reg.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize execution policy analyzer {reg.name}: {e}")

        logger.info(
            f"Registered {len(self._registries['execution_policy_analyzers'])} execution policy analyzers"
        )

    def _initialize_domain_analyzers(self) -> None:
        """Initialize domain analyzer registry with instantiation.

        Dynamically imports and instantiates all domain analyzer classes defined
        in the configuration. These analyzers analyze code for domain-specific
        patterns and operations.

        Failed analyzer initialization is logged as warning but does not fail
        the entire registry initialization, allowing fallback to default analyzer.

        :raises Exception: Individual analyzer failures are caught and logged
        """
        logger.debug("Initializing domain analyzers...")
        for reg in self.config.domain_analyzers:
            try:
                module = __import__(reg.module_path, fromlist=[reg.class_name])
                analyzer_class = getattr(module, reg.class_name)
                self._registries["domain_analyzers"][reg.name] = {
                    "class": analyzer_class,
                    "registration": reg,
                }
                logger.debug(f"Registered domain analyzer: {reg.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize domain analyzer {reg.name}: {e}")

        logger.info(f"Registered {len(self._registries['domain_analyzers'])} domain analyzers")

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

    def _initialize_capabilities(self) -> None:
        """Initialize capability registry.

        Dynamically imports and registers all domain-specific capabilities.

        Failed capability initialization is logged as warning but does not fail
        the entire registry, allowing partial system functionality.
        """
        logger.debug("Initializing capabilities...")
        successful_count = 0
        failed_caps = []

        for reg in self.config.capabilities:
            try:
                module = __import__(reg.module_path, fromlist=[reg.class_name])
                capability_class = getattr(module, reg.class_name)
                capability_instance = capability_class()
                self._registries["capabilities"][reg.name] = capability_instance

                logger.debug(f"Registered capability: {reg.name}")
                successful_count += 1

            except Exception as e:
                failed_caps.append(reg.name)
                logger.warning(f"Failed to initialize capability {reg.name}: {e}")
                import traceback

                logger.debug(
                    f"Capability {reg.name} initialization traceback: {traceback.format_exc()}"
                )

        if failed_caps:
            logger.error(f"❌ Failed to initialize {len(failed_caps)} capabilities: {failed_caps}")
        logger.info(
            f"Registered {len(self._registries['capabilities'])} capabilities ({successful_count} successful, {len(failed_caps)} failed)"
        )

    def _initialize_framework_prompt_providers(self) -> None:
        """Initialize framework prompt providers with explicit mapping.

        Creates prompt providers using explicit builder class mapping. This provides
        clear, maintainable prompt
        customization for different applications while maintaining compatibility
        with the default prompt system.

        The first registered provider is automatically set as the default for
        the framework prompt system.

        :raises Exception: Individual provider failures are caught and logged
        """
        logger.debug("Initializing framework prompt providers...")
        for reg in self.config.framework_prompt_providers:
            try:
                provider = self._create_explicit_provider(reg)

                from osprey.prompts.loader import _prompt_loader

                provider_key = reg.module_path
                _prompt_loader._providers[provider_key] = provider

                self._registries["framework_prompt_providers"][provider_key] = reg
                logger.debug(
                    f"Registered prompt provider from {reg.module_path} with {len(reg.prompt_builders)} custom builders"
                )

            except Exception as e:
                logger.warning(f"Failed to initialize prompt provider from {reg.module_path}: {e}")

        if self.config.framework_prompt_providers:
            default_provider_key = self.config.framework_prompt_providers[-1].module_path

            from osprey.prompts.loader import set_default_framework_prompt_provider

            set_default_framework_prompt_provider(default_provider_key)
            logger.info(f"Set default prompt provider: {default_provider_key}")

        logger.info(
            f"Registered {len(self._registries['framework_prompt_providers'])} framework prompt providers"
        )

    def _create_explicit_provider(self, reg):
        """Create prompt provider by overriding defaults with application-specific builders.

        Overrides framework default builders with application-specific builders.
        Applications only declare what they customize; everything else uses
        framework defaults as fallbacks.

        :param reg: Framework prompt provider registration configuration
        :type reg: FrameworkPromptProviderRegistration
        :return: Configured prompt provider with application overrides
        :rtype: framework.prompts.defaults.DefaultPromptProvider
        :raises ImportError: If builder class modules cannot be imported
        :raises AttributeError: If builder classes are not found in modules
        """
        from osprey.prompts.defaults import DefaultPromptProvider

        provider = DefaultPromptProvider()

        successful_overrides = []
        failed_overrides = []

        for prompt_type, class_name in reg.prompt_builders.items():
            try:
                module = __import__(reg.module_path, fromlist=[class_name])
                builder_class = getattr(module, class_name)
                builder_instance = builder_class()

                attr_name = f"_{prompt_type}_builder"
                setattr(provider, attr_name, builder_instance)

                successful_overrides.append(prompt_type)
                logger.debug(f"  -> Overrode {prompt_type} with {class_name}")

            except Exception as e:
                failed_overrides.append((prompt_type, class_name, str(e)))
                logger.warning(f"Failed to load prompt builder {class_name}: {e}")

        total_builders = len(reg.prompt_builders)
        if successful_overrides:
            logger.info(
                f"Successfully loaded {len(successful_overrides)}/{total_builders} custom prompt builders from {reg.module_path}"
            )
        if failed_overrides:
            logger.warning(
                f"Failed to load {len(failed_overrides)} prompt builders from {reg.module_path} - using framework defaults"
            )
            for prompt_type, class_name, error in failed_overrides:
                logger.debug(f"  -> {prompt_type}({class_name}): {error}")

        self._validate_prompt_provider(provider, reg.module_path)

        return provider

    def _validate_prompt_provider(self, provider, module_path):
        """Validate prompt provider implements required interface methods.

        Ensures the provider implements all required methods for framework operation.
        Missing methods are logged as errors but do not prevent provider registration,
        allowing partial functionality with fallback to defaults.

        :param provider: Prompt provider instance to validate
        :type provider: framework.prompts.defaults.DefaultPromptProvider
        :param module_path: Module path for error reporting
        :type module_path: str
        """
        required_methods = [
            "get_orchestrator_prompt_builder",
            "get_task_extraction_prompt_builder",
            "get_response_generation_prompt_builder",
            "get_classification_prompt_builder",
            "get_error_analysis_prompt_builder",
            "get_clarification_prompt_builder",
            "get_memory_extraction_prompt_builder",
            "get_time_range_parsing_prompt_builder",
        ]

        missing_methods = []
        for method_name in required_methods:
            if not hasattr(provider, method_name) or not callable(getattr(provider, method_name)):
                missing_methods.append(method_name)

        if missing_methods:
            logger.error(
                f"Prompt provider from {module_path} is missing required methods: {missing_methods}"
            )
        else:
            logger.debug(f"Prompt provider from {module_path} passed interface validation")

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
            "capabilities": self._export_capabilities(),
            "context_types": self._export_context_types(),
            "connectors": self._export_connectors(),
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "registry_version": "1.0",
                "total_capabilities": len(self.config.capabilities),
                "total_context_types": len(self.config.context_classes),
                "total_connectors": len(self.config.connectors),
            },
        }

        if output_dir:
            self._save_export_data(export_data, output_dir)

        return export_data

    def _export_capabilities(self) -> list[dict[str, Any]]:
        """Export capability metadata for external consumption.

        Transforms internal capability registrations into standardized format
        suitable for execution plan editors and documentation tools. Exports
        all registered capabilities without filtering.

        :return: List of capability metadata dictionaries
        :rtype: list[dict[str, Any]]
        """
        capabilities = []

        for cap_reg in self.config.capabilities:
            capability_data = {
                "name": cap_reg.name,
                "description": cap_reg.description,
                "provides": cap_reg.provides,
                "requires": cap_reg.requires,
                "module_path": cap_reg.module_path,
                "class_name": cap_reg.class_name,
            }
            capabilities.append(capability_data)

        return capabilities

    def _export_context_types(self) -> list[dict[str, Any]]:
        """Export context type metadata for external consumption.

        Transforms internal context class registrations into standardized format
        suitable for execution plan editors and documentation tools. Exports
        all registered context types without filtering.

        :return: List of context type metadata dictionaries
        :rtype: list[dict[str, Any]]
        """
        context_types = []

        for ctx_reg in self.config.context_classes:
            context_data = {
                "context_type": ctx_reg.context_type,
                "class_name": ctx_reg.class_name,
                "module_path": ctx_reg.module_path,
                "description": getattr(
                    ctx_reg, "description", f"Context class {ctx_reg.class_name}"
                ),
            }
            context_types.append(context_data)

        return context_types

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

            components = ["capabilities", "context_types"]
            for component in components:
                if component in export_data:
                    component_file = Path(output_dir) / f"{component}.json"
                    with open(component_file, "w", encoding="utf-8") as f:
                        json.dump(export_data[component], f, indent=2, ensure_ascii=False)
                    logger.debug(f"Saved {component} to: {component_file}")

            logger.info(f"Registry export saved to: {export_file}")
            logger.info(
                f"Export contains: {export_data['metadata']['total_capabilities']} capabilities, "
                f"{export_data['metadata']['total_context_types']} context types"
            )

        except Exception as e:
            logger.error(f"Failed to save export data: {e}")
            raise

    def get_capability(self, name: str) -> Optional["BaseCapability"]:
        """Retrieve registered capability instance by name.

        :param name: Unique capability name from registration
        :type name: str
        :return: Capability instance if registered, None otherwise
        :rtype: framework.base.BaseCapability, optional
        """
        return self._registries["capabilities"].get(name)

    def get_all_capabilities(self) -> list["BaseCapability"]:
        """Retrieve all registered capability instances.

        :return: List of all registered capability instances
        :rtype: list[framework.base.BaseCapability]
        """
        return list(self._registries["capabilities"].values())

    def get_all_nodes(self) -> dict[str, Any]:
        """Retrieve all registered nodes as (name, callable) pairs.

        :return: Dictionary mapping node names to their callable instances
        :rtype: Dict[str, Any]
        """
        return dict(self._registries["nodes"].items())

    def get_context_class(self, context_type: str) -> type["CapabilityContext"] | None:
        """Retrieve context class by type identifier.

        :param context_type: Context type identifier (e.g., 'PV_ADDRESSES')
        :type context_type: str
        :return: Context class if registered, None otherwise
        :rtype: Type[framework.base.CapabilityContext], optional
        """
        return self._registries["contexts"].get(context_type)

    def get_all_context_classes(self) -> dict[str, type["CapabilityContext"]]:
        """Get dictionary of all registered context classes by context type.

        This method provides access to all registered context classes indexed by their
        context type identifiers. It enables introspection of the complete context
        system and supports dynamic context handling patterns.

        :return: Dictionary mapping context types to their corresponding context classes
        :rtype: Dict[str, Type[CapabilityContext]]

        Examples:
            Access all context classes::

                >>> registry = get_registry()
                >>> context_classes = registry.get_all_context_classes()
                >>> pv_class = context_classes.get("PV_ADDRESSES")
                >>> if pv_class:
                ...     instance = pv_class(pvs=["test:pv"])
        """
        return dict(self._registries["contexts"])

    def get_data_source(self, name: str) -> Any | None:
        """Retrieve data source provider instance by name.

        :param name: Unique data source name from registration
        :type name: str
        :return: Data source provider instance if registered, None otherwise
        :rtype: Any, optional
        """
        return self._registries["data_sources"].get(name)

    def get_all_data_sources(self) -> list[Any]:
        """Retrieve all registered data source provider instances.

        :return: List of all registered data source provider instances
        :rtype: list[Any]
        """
        return list(self._registries["data_sources"].values())

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

    def get_execution_policy_analyzers(self) -> list[Any]:
        """Retrieve all registered execution policy analyzer instances.

        Creates instances of execution policy analyzers with empty configurable.
        The actual configurable will be provided when the analyzers are used.

        :return: List of execution policy analyzer instances
        :rtype: list[Any]
        """
        analyzers = []
        for name, registry_entry in self._registries["execution_policy_analyzers"].items():
            try:
                analyzer_class = registry_entry["class"]
                analyzer_instance = analyzer_class({})
                analyzers.append(analyzer_instance)
            except Exception as e:
                logger.warning(f"Failed to instantiate execution policy analyzer {name}: {e}")
        return analyzers

    def get_domain_analyzers(self) -> list[Any]:
        """Retrieve all registered domain analyzer instances.

        Creates instances of domain analyzers with empty configurable.
        The actual configurable will be provided when the analyzers are used.

        :return: List of domain analyzer instances
        :rtype: list[Any]
        """
        analyzers = []
        for name, registry_entry in self._registries["domain_analyzers"].items():
            try:
                analyzer_class = registry_entry["class"]
                analyzer_instance = analyzer_class({})
                analyzers.append(analyzer_instance)
            except Exception as e:
                logger.warning(f"Failed to instantiate domain analyzer {name}: {e}")
        return analyzers

    @property
    def context_types(self):
        """Dynamic object providing context type constants as attributes.

        Creates a dynamic object where each registered context type is accessible
        as an attribute with its string value.

        :return: Object with context types as attributes
        :rtype: object

        Examples:
            Access context types::

                >>> registry = get_registry()
                >>> pv_type = registry.context_types.PV_ADDRESSES
                >>> print(pv_type)  # "PV_ADDRESSES"
        """
        if not hasattr(self, "_context_types"):
            self._context_types = type(
                "ContextTypes",
                (),
                {
                    ctx_reg.context_type: ctx_reg.context_type
                    for ctx_reg in self.config.context_classes
                },
            )()
        return self._context_types

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
            f"      • {stats['capabilities']} capabilities: {', '.join(stats['capability_names'])}",
            f"      • {stats['context_classes']} context types: {', '.join(stats['context_types'])}",
            f"      • {stats['data_sources']} data sources: {', '.join(stats['data_source_names'])}",
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
            "capabilities": len(self._registries["capabilities"]),
            "nodes": len(self._registries["nodes"]),
            "context_classes": len(self._registries["contexts"]),
            "data_sources": len(self._registries["data_sources"]),
            "services": len(self._registries["services"]),
            "capability_names": list(self._registries["capabilities"].keys()),
            "node_names": list(self._registries["nodes"].keys()),
            "context_types": list(self._registries["contexts"].keys()),
            "data_source_names": list(self._registries["data_sources"].keys()),
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

    from osprey.approval.approval_manager import get_approval_manager

    get_approval_manager()

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
