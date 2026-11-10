"""Registry helper functions for application development.

This module provides utilities that simplify application registry creation
by handling the common pattern of extending the framework registry with
application-specific components.
"""

from .base import (
    ArielEnhancementModuleRegistration,
    ArielIngestionAdapterRegistration,
    ArielSearchModuleRegistration,
    ConnectorRegistration,
    ExtendedRegistryConfig,
    ProviderRegistration,
    ServiceRegistration,
)


def extend_framework_registry(
    services: list[ServiceRegistration] | None = None,
    providers: list[ProviderRegistration] | None = None,
    connectors: list[ConnectorRegistration] | None = None,
    exclude_providers: list[str] | None = None,
    exclude_connectors: list[str] | None = None,
    ariel_search_modules: list[ArielSearchModuleRegistration] | None = None,
    ariel_enhancement_modules: list[ArielEnhancementModuleRegistration] | None = None,
    ariel_ingestion_adapters: list[ArielIngestionAdapterRegistration] | None = None,
    exclude_ariel_search_modules: list[str] | None = None,
    exclude_ariel_enhancement_modules: list[str] | None = None,
    exclude_ariel_ingestion_adapters: list[str] | None = None,
    override_providers: list[ProviderRegistration] | None = None,
    override_connectors: list[ConnectorRegistration] | None = None,
) -> ExtendedRegistryConfig:
    """Create application registry configuration that extends the framework.

    This is the recommended way to create application registries. It simplifies
    registry creation by automatically handling framework component exclusions
    and overrides through clean, declarative parameters.

    Args:
        services: Application services to add to framework defaults
        providers: Application AI model providers to add to framework defaults
        connectors: Application control system/archiver connectors to add
        exclude_providers: Names of built-in providers to remove from the provider
            registry; a name also registered through `providers` or
            `override_providers` keeps that registration.
        exclude_connectors: Names of framework connectors to exclude
        override_providers: Providers that replace framework versions (by name)
        override_connectors: Connectors that replace framework versions (by name)

    Returns:
        ExtendedRegistryConfig that signals extend mode to registry manager

    Examples:
        Simple application (most common)::

            def get_registry_config(self) -> ExtendedRegistryConfig:
                return extend_framework_registry()
    """
    framework_exclusions = {}

    if exclude_providers:
        framework_exclusions["providers"] = exclude_providers

    if exclude_connectors:
        framework_exclusions["connectors"] = exclude_connectors

    if exclude_ariel_search_modules:
        framework_exclusions["ariel_search_modules"] = exclude_ariel_search_modules

    if exclude_ariel_enhancement_modules:
        framework_exclusions["ariel_enhancement_modules"] = exclude_ariel_enhancement_modules

    if exclude_ariel_ingestion_adapters:
        framework_exclusions["ariel_ingestion_adapters"] = exclude_ariel_ingestion_adapters

    all_providers = list(providers or [])
    if override_providers:
        all_providers.extend(override_providers)

    all_connectors = list(connectors or [])
    if override_connectors:
        all_connectors.extend(override_connectors)

    return ExtendedRegistryConfig(
        services=list(services or []),
        providers=all_providers,
        connectors=all_connectors,
        ariel_search_modules=list(ariel_search_modules or []),
        ariel_enhancement_modules=list(ariel_enhancement_modules or []),
        ariel_ingestion_adapters=list(ariel_ingestion_adapters or []),
        framework_exclusions=framework_exclusions,
    )
