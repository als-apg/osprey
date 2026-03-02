"""Per-component-type initialization functions.

Each function loads registered components of a single type (providers,
connectors, services, ARIEL modules) into the shared *registries* dict.

.. note::

   **LAYERING NOTE** — This module contains upward imports from L4/L5 layers
   (``osprey.connectors``, ``osprey.models``).  These are isolated here so the
   coupling is explicit and easy to refactor later via a registration-callback
   pattern.  See RF-010 for the long-term plan.

Extracted from :mod:`osprey.registry.manager` (RF-010).
"""

import importlib
from typing import Any

from osprey.errors import RegistryError
from osprey.utils.config import get_config_value
from osprey.utils.logger import get_logger

from .base import RegistryConfig

logger = get_logger(name="registry.init", color="sky_blue2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_configured_provider_names() -> set[str] | None:
    """Extract provider names from config.yml ``models`` section.

    Returns ``None`` if config is unavailable (load all as fallback).
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
        return None


# ---------------------------------------------------------------------------
# Initializer functions — uniform signature for dispatch
# ---------------------------------------------------------------------------


def initialize_providers(
    *,
    config: RegistryConfig,
    registries: dict[str, dict[str, Any]],
    excluded_provider_names: list[str],
) -> None:
    """Initialize AI model providers via the lightweight ProviderRegistry.

    Delegates to ``ProviderRegistry.load_providers()`` for the actual import
    loop, then stores the results in *registries* for backward compatibility.
    Custom providers from ``config.providers`` (non-built-in) are registered
    first so they participate in the bulk load.

    Uses config-driven filtering to skip imports for unconfigured providers,
    avoiding costly module-level network calls on air-gapped machines.
    """
    # LAYERING NOTE: upward import from models (L5)
    from osprey.models.provider_registry import get_provider_registry

    pr = get_provider_registry()

    for registration in config.providers:
        if registration.name and registration.name not in pr.list_providers():
            pr.register_provider(
                registration.name,
                registration.module_path,
                registration.class_name,
            )

    configured_providers = _get_configured_provider_names()

    if configured_providers is not None:
        logger.info(
            f"Initializing providers (config-driven: {sorted(configured_providers)})..."
        )
    else:
        logger.info("Initializing providers (all available)...")

    loaded = pr.load_providers(
        configured_names=configured_providers,
        excluded_names=excluded_provider_names or None,
    )

    for name, provider_class in loaded.items():
        registries["providers"][name] = provider_class
        logger.info(f"  ✓ Registered provider: {name}")

    logger.info(
        f"Provider initialization complete: "
        f"{len(registries['providers'])} providers loaded"
    )


def initialize_connectors(
    *,
    config: RegistryConfig,
    registries: dict[str, dict[str, Any]],
    excluded_provider_names: list[str],
) -> None:
    """Initialize control system and archiver connectors from registry config.

    Loads connector classes and registers them with ``ConnectorFactory`` for
    runtime use.
    """
    logger.info(f"Initializing {len(config.connectors)} connector(s)...")

    # LAYERING NOTE: upward import from connectors (L4)
    try:
        from osprey.connectors.factory import ConnectorFactory
    except ImportError as e:
        logger.error(f"Failed to import ConnectorFactory: {e}")
        raise RegistryError(
            "ConnectorFactory not available - connector system may not be installed"
        ) from e

    for registration in config.connectors:
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

            registries["connectors"][registration.name] = connector_class

            logger.info(
                f"  ✓ Registered {registration.connector_type} connector: "
                f"{registration.name}"
            )
            logger.debug(f"    - Description: {registration.description}")
            logger.debug(f"    - Module: {registration.module_path}")
            logger.debug(f"    - Class: {registration.class_name}")

        except ImportError as e:
            logger.warning(
                f"  ⊘ Skipping connector '{registration.name}' (import failed): {e}"
            )
            logger.debug(
                f"    Connector {registration.name} may require optional dependencies"
            )
        except Exception as e:
            logger.error(f"  ✗ Failed to register connector '{registration.name}': {e}")
            raise RegistryError(
                f"Connector registration failed for {registration.name}"
            ) from e

    logger.info(
        f"Connector initialization complete: "
        f"{len(registries['connectors'])} connectors loaded"
    )


def initialize_ariel_search_modules(
    *,
    config: RegistryConfig,
    registries: dict[str, dict[str, Any]],
    excluded_provider_names: list[str],
) -> None:
    """Initialize ARIEL search modules from registry configuration.

    Imports each search module and validates it exports a
    ``get_tool_descriptor`` callable.
    """
    if not config.ariel_search_modules:
        return

    logger.info(
        f"Initializing {len(config.ariel_search_modules)} ARIEL search module(s)..."
    )

    for registration in config.ariel_search_modules:
        try:
            module = importlib.import_module(registration.module_path)

            if not hasattr(module, "get_tool_descriptor") or not callable(
                module.get_tool_descriptor
            ):
                raise RegistryError(
                    f"ARIEL search module '{registration.name}' at "
                    f"{registration.module_path} "
                    f"must export a callable get_tool_descriptor()"
                )

            registries["ariel_search_modules"][registration.name] = module
            logger.debug(f"  ✓ Registered ARIEL search module: {registration.name}")

        except ImportError as e:
            logger.warning(
                f"  ⊘ Skipping ARIEL search module '{registration.name}' "
                f"(import failed): {e}"
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
        f"{len(registries['ariel_search_modules'])} modules loaded"
    )


def initialize_ariel_enhancement_modules(
    *,
    config: RegistryConfig,
    registries: dict[str, dict[str, Any]],
    excluded_provider_names: list[str],
) -> None:
    """Initialize ARIEL enhancement modules from registry configuration.

    Imports each enhancement module class and stores ``(class, registration)``
    tuples in the registry.
    """
    if not config.ariel_enhancement_modules:
        return

    logger.info(
        f"Initializing {len(config.ariel_enhancement_modules)} "
        f"ARIEL enhancement module(s)..."
    )

    for registration in config.ariel_enhancement_modules:
        try:
            module = importlib.import_module(registration.module_path)
            cls = getattr(module, registration.class_name)

            registries["ariel_enhancement_modules"][registration.name] = (
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
                f"  ✗ Failed to register ARIEL enhancement module "
                f"'{registration.name}': {e}"
            )
            raise RegistryError(
                f"ARIEL enhancement module registration failed for {registration.name}"
            ) from e

    logger.info(
        f"ARIEL enhancement module initialization complete: "
        f"{len(registries['ariel_enhancement_modules'])} modules loaded"
    )


def initialize_ariel_pipelines(
    *,
    config: RegistryConfig,
    registries: dict[str, dict[str, Any]],
    excluded_provider_names: list[str],
) -> None:
    """Initialize ARIEL pipelines from registry configuration.

    Imports each pipeline module and validates it exports a
    ``get_pipeline_descriptor`` callable.
    """
    if not config.ariel_pipelines:
        return

    logger.info(f"Initializing {len(config.ariel_pipelines)} ARIEL pipeline(s)...")

    for registration in config.ariel_pipelines:
        try:
            module = importlib.import_module(registration.module_path)

            if not hasattr(module, "get_pipeline_descriptor") or not callable(
                module.get_pipeline_descriptor
            ):
                raise RegistryError(
                    f"ARIEL pipeline '{registration.name}' at "
                    f"{registration.module_path} "
                    f"must export a callable get_pipeline_descriptor()"
                )

            registries["ariel_pipelines"][registration.name] = module
            logger.debug(f"  ✓ Registered ARIEL pipeline: {registration.name}")

        except ImportError as e:
            logger.warning(
                f"  ⊘ Skipping ARIEL pipeline '{registration.name}' (import failed): {e}"
            )
        except Exception as e:
            logger.error(
                f"  ✗ Failed to register ARIEL pipeline '{registration.name}': {e}"
            )
            raise RegistryError(
                f"ARIEL pipeline registration failed for {registration.name}"
            ) from e

    logger.info(
        f"ARIEL pipeline initialization complete: "
        f"{len(registries['ariel_pipelines'])} pipelines loaded"
    )


def initialize_ariel_ingestion_adapters(
    *,
    config: RegistryConfig,
    registries: dict[str, dict[str, Any]],
    excluded_provider_names: list[str],
) -> None:
    """Initialize ARIEL ingestion adapters from registry configuration.

    Imports each ingestion adapter class and stores ``(class, registration)``
    tuples in the registry.
    """
    if not config.ariel_ingestion_adapters:
        return

    logger.info(
        f"Initializing {len(config.ariel_ingestion_adapters)} "
        f"ARIEL ingestion adapter(s)..."
    )

    for registration in config.ariel_ingestion_adapters:
        try:
            module = importlib.import_module(registration.module_path)
            cls = getattr(module, registration.class_name)

            registries["ariel_ingestion_adapters"][registration.name] = (
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
                f"  ✗ Failed to register ARIEL ingestion adapter "
                f"'{registration.name}': {e}"
            )
            raise RegistryError(
                f"ARIEL ingestion adapter registration failed for {registration.name}"
            ) from e

    logger.info(
        f"ARIEL ingestion adapter initialization complete: "
        f"{len(registries['ariel_ingestion_adapters'])} adapters loaded"
    )


def initialize_services(
    *,
    config: RegistryConfig,
    registries: dict[str, dict[str, Any]],
    excluded_provider_names: list[str],
) -> None:
    """Initialize service registry.

    Services provide specialized functionality that can be invoked by
    capabilities. Each service is instantiated and registered for runtime
    access. Failures are logged but do not fail initialization.
    """
    logger.debug("Initializing services...")
    for reg in config.services:
        try:
            module = __import__(reg.module_path, fromlist=[reg.class_name])
            service_class = getattr(module, reg.class_name)
            service_instance = service_class()
            registries["services"][reg.name] = service_instance
            logger.debug(f"Registered service: {reg.name}")

        except Exception as e:
            logger.warning(f"Failed to initialize service {reg.name}: {e}")

    logger.info(f"Registered {len(registries['services'])} services")


# ---------------------------------------------------------------------------
# Dispatch table — maps component_type strings to initializer functions
# ---------------------------------------------------------------------------

INITIALIZER_DISPATCH: dict[str, Any] = {
    "providers": initialize_providers,
    "connectors": initialize_connectors,
    "ariel_search_modules": initialize_ariel_search_modules,
    "ariel_enhancement_modules": initialize_ariel_enhancement_modules,
    "ariel_pipelines": initialize_ariel_pipelines,
    "ariel_ingestion_adapters": initialize_ariel_ingestion_adapters,
    "services": initialize_services,
}
