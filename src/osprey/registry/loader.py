"""Registry loading and merging.

Discovers :class:`RegistryConfigProvider` implementations from Python modules
(both importable packages and filesystem paths), then merges framework and
application configurations into a single :class:`RegistryConfig`.

Extracted from :mod:`osprey.registry.manager` (RF-010).
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from osprey.errors import RegistryError
from osprey.utils.logger import get_logger

from .base import ExtendedRegistryConfig, RegistryConfig, RegistryConfigProvider

logger = get_logger(name="registry.loader", color="sky_blue2")


def build_merged_configuration(
    registry_path: str | None,
) -> tuple[RegistryConfig, list[str]]:
    """Build configuration from framework and/or application registry.

    Supports two registry modes based on type detection:

    **Standalone Mode** (RegistryConfig):
        Application provides complete registry with ALL components.
        Framework registry is NOT loaded.

    **Extend Mode** (ExtendedRegistryConfig):
        Application extends framework defaults via extend_framework_registry().
        Framework registry is loaded first, then application components are
        merged, with applications able to override framework components.

    :param registry_path: Path to application ``registry.py``, or *None*
        for framework-only mode.
    :return: Tuple of (merged config, excluded provider names).
    :raises RegistryError: If registry loading fails.
    """
    excluded_provider_names: list[str] = []

    if not registry_path:
        logger.info("Built framework-only registry (no application)")
        return load_registry_from_module("osprey.registry.registry"), excluded_provider_names

    try:
        app_config = load_registry_from_path(registry_path)
        app_name = Path(registry_path).resolve().parent.name

        if app_config is None:
            raise RegistryError(
                f"Registry provider in '{app_name}' returned None. "
                f"get_registry_config() must return a RegistryConfig instance."
            )

        if isinstance(app_config, ExtendedRegistryConfig):
            logger.info(f"Extending framework registry with application '{app_name}'")

            framework_config = load_registry_from_module("osprey.registry.registry")
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

            merge_application_with_override(merged, app_config, app_name, excluded_provider_names)
            logger.info(
                f"Loaded application registry from: {registry_path} (app: {app_name})"
            )
            return merged, excluded_provider_names

        else:
            logger.info(
                f"Using standalone registry from application '{app_name}' "
                f"(framework registry skipped)"
            )
            return app_config, excluded_provider_names

    except Exception as e:
        logger.error(f"Failed to load registry from {registry_path}: {e}")
        raise RegistryError(
            f"Failed to load registry from {registry_path}: {e}"
        ) from e


def load_registry_from_module(module_path: str) -> RegistryConfig:
    """Load registry from an importable Python module.

    Convention: the module must contain exactly one class implementing
    :class:`RegistryConfigProvider`.

    :param module_path: Dotted Python module path (e.g., ``'osprey.registry.registry'``).
    :return: Registry configuration from the provider.
    :raises RegistryError: If module cannot be imported or no single provider found.
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
                f"{component_name} must define exactly one class implementing "
                f"RegistryConfigProvider. "
                f"Import: from osprey.registry import RegistryConfigProvider"
            )
        elif len(provider_classes) > 1:
            class_names = [cls.__name__ for cls in provider_classes]
            raise RegistryError(
                f"Multiple RegistryConfigProvider implementations found in "
                f"{module_path}: {class_names}. "
                f"{component_name} must define exactly one provider class."
            )

        provider_class = provider_classes[0]
        provider_instance = provider_class()
        config = provider_instance.get_registry_config()

        logger.debug(
            f"Loaded {component_name} registry via {provider_class.__name__} "
            f"from {module_path}"
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


def load_registry_from_path(registry_path: str) -> RegistryConfig:
    """Load registry from a filesystem path using ``importlib.util``.

    Adds the appropriate parent directory to ``sys.path`` so that registry
    module references (e.g. ``app_name.context_classes``) resolve correctly.

    :param registry_path: Absolute or relative path to ``registry.py``.
    :return: Registry configuration from the file.
    :raises RegistryError: If file not found, invalid, or no provider found.
    """
    path = Path(registry_path).resolve()

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
    project_root = app_dir.parent

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
        spec = importlib.util.spec_from_file_location("_dynamic_registry", path)

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
            f"Registry files must define exactly one class implementing "
            f"RegistryConfigProvider.\n"
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
            f"Multiple RegistryConfigProvider implementations found in "
            f"{registry_path}: {class_names}.\n"
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


def apply_framework_exclusions(
    merged: RegistryConfig,
    exclusions: dict[str, list[str]],
    app_name: str,
    excluded_provider_names: list[str],
) -> None:
    """Apply framework component exclusions to the merged registry configuration.

    Removes specified framework components from the merged configuration based on
    exclusion rules defined by the application.

    :param merged: Merged registry configuration to modify.
    :param exclusions: Component exclusions by type
        (e.g., ``{'capabilities': ['python']}``).
    :param app_name: Application name for logging.
    :param excluded_provider_names: Accumulator list for deferred provider exclusions
        (mutated in place).
    """
    for component_type, excluded_names in exclusions.items():
        if not excluded_names:
            continue

        # Handle provider exclusions specially (names are introspected after loading)
        if component_type == "providers":
            excluded_provider_names.extend(excluded_names)
            logger.info(
                f"Application {app_name} will exclude framework providers: {excluded_names}"
            )
            continue

        component_collection = getattr(merged, component_type, None)
        if component_collection is None:
            logger.warning(
                f"Application {app_name} tried to exclude unknown component type: "
                f"{component_type}"
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
                    f"Application {app_name} excluded framework "
                    f"{component_type}: {actually_excluded}"
                )


def merge_application_with_override(
    merged: RegistryConfig,
    app_config: RegistryConfig,
    app_name: str,
    excluded_provider_names: list[str],
) -> None:
    """Merge application configuration with framework, allowing overrides.

    Applications can override framework components by providing components
    with the same name.

    :param merged: Framework config to merge into (mutated in place).
    :param app_config: Application config to merge from.
    :param app_name: Application name for logging.
    :param excluded_provider_names: Accumulator for deferred provider exclusions.
    """
    framework_exclusions = getattr(app_config, "framework_exclusions", {})
    if framework_exclusions:
        apply_framework_exclusions(merged, framework_exclusions, app_name, excluded_provider_names)

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
        logger.info(
            f"Application {app_name} overrode framework services: {service_overrides}"
        )

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
        logger.info(
            f"Application {app_name} added {len(providers_added)} new provider(s)"
        )

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
            f"Application {app_name} added {len(connectors_added)} new connector(s): "
            f"{connectors_added}"
        )

    merge_named_registrations(
        merged.ariel_search_modules,
        getattr(app_config, "ariel_search_modules", []),
        "ARIEL search module",
        app_name,
    )
    merge_named_registrations(
        merged.ariel_enhancement_modules,
        getattr(app_config, "ariel_enhancement_modules", []),
        "ARIEL enhancement module",
        app_name,
    )
    merge_named_registrations(
        merged.ariel_pipelines,
        getattr(app_config, "ariel_pipelines", []),
        "ARIEL pipeline",
        app_name,
    )
    merge_named_registrations(
        merged.ariel_ingestion_adapters,
        getattr(app_config, "ariel_ingestion_adapters", []),
        "ARIEL ingestion adapter",
        app_name,
    )


def merge_named_registrations(
    merged_list: list,
    app_list: list,
    type_label: str,
    app_name: str,
) -> None:
    """Merge named registrations with override support.

    For each item in *app_list*, if a registration with the same ``.name``
    exists in *merged_list* it is replaced; otherwise the item is appended.
    *merged_list* is mutated in place.

    :param merged_list: Framework registrations (mutated in place).
    :param app_list: Application registrations to merge in.
    :param type_label: Human-readable type label for log messages.
    :param app_name: Application name for log messages.
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
        logger.info(
            f"Application {app_name} overrode framework {type_label}s: {overrides}"
        )
    if additions:
        logger.info(
            f"Application {app_name} added {len(additions)} new "
            f"{type_label}(s): {additions}"
        )
