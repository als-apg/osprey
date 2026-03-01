"""Registry helper functions for application development.

This module provides utilities that simplify application registry creation
by handling the common pattern of extending the framework registry with
application-specific components.
"""

from .base import (
    ArielEnhancementModuleRegistration,
    ArielIngestionAdapterRegistration,
    ArielPipelineRegistration,
    ArielSearchModuleRegistration,
    ConnectorRegistration,
    ExtendedRegistryConfig,
    ProviderRegistration,
    RegistryConfig,
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
    ariel_pipelines: list[ArielPipelineRegistration] | None = None,
    ariel_ingestion_adapters: list[ArielIngestionAdapterRegistration] | None = None,
    exclude_ariel_search_modules: list[str] | None = None,
    exclude_ariel_enhancement_modules: list[str] | None = None,
    exclude_ariel_pipelines: list[str] | None = None,
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
        exclude_providers: Names of framework providers to exclude
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

    if exclude_ariel_pipelines:
        framework_exclusions["ariel_pipelines"] = exclude_ariel_pipelines

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
        ariel_pipelines=list(ariel_pipelines or []),
        ariel_ingestion_adapters=list(ariel_ingestion_adapters or []),
        framework_exclusions=framework_exclusions if framework_exclusions else None,
    )


def get_framework_defaults() -> RegistryConfig:
    """Get the default framework registry configuration.

    Returns:
        Complete framework RegistryConfig with all core components
    """
    from .registry import FrameworkRegistryProvider

    provider = FrameworkRegistryProvider()
    return provider.get_registry_config()


def generate_explicit_registry_code(
    app_class_name: str,
    app_display_name: str,
    package_name: str,
    services: list[ServiceRegistration] | None = None,
) -> str:
    """Generate explicit registry Python code with all framework + app components.

    Args:
        app_class_name: Python class name for the registry provider
        app_display_name: Human-readable application name
        package_name: Python package name
        services: Application-specific services to add (optional)

    Returns:
        Complete Python source code for the explicit registry
    """
    framework = get_framework_defaults()

    def format_service_registration(
        reg: ServiceRegistration, indent: str = "                "
    ) -> str:
        return f"""{indent}ServiceRegistration(
{indent}    name="{reg.name}",
{indent}    module_path="{reg.module_path}",
{indent}    class_name="{reg.class_name}",
{indent}    description="{reg.description}",
{indent}    provides={reg.provides},
{indent}    requires={reg.requires}
{indent})"""

    code_lines = [
        '"""',
        f"Component registry for {app_display_name}.",
        "",
        "This registry uses the EXPLICIT style, listing all framework components",
        "alongside application-specific components for full visibility and control.",
        '"""',
        "",
        "from osprey.registry import (",
        "    RegistryConfigProvider,",
        "    RegistryConfig,",
        "    ServiceRegistration,",
        "    ProviderRegistration",
        ")",
        "",
        "",
        f"class {app_class_name}(RegistryConfigProvider):",
        f'    """Registry provider for {app_display_name}."""',
        "    ",
        "    def get_registry_config(self):",
        f'        """Return registry configuration for {app_display_name}."""',
        "        return RegistryConfig(",
        "            # ================================================================",
        "            # SERVICES (Framework + Application)",
        "            # ================================================================",
        "            services=[",
        "                # ---- Framework Services ----",
    ]

    # Add framework services
    for i, svc in enumerate(framework.services):
        code_lines.append(format_service_registration(svc))
        if services or i < len(framework.services) - 1:
            code_lines[-1] += ","

    # Add application services
    if services:
        code_lines.extend(
            [
                "",
                "                # ---- Application Services ----",
            ]
        )
        for i, svc in enumerate(services):
            code_lines.append(format_service_registration(svc))
            if i < len(services) - 1:
                code_lines[-1] += ","

    code_lines.extend(
        [
            "            ],",
            "",
            "            # ================================================================",
            "            # AI MODEL PROVIDERS",
            "            # ================================================================",
            "            providers=[",
        ]
    )

    # Add framework AI model providers
    for prov in framework.providers:
        code_lines.append("                ProviderRegistration(")
        code_lines.append(f'                    module_path="{prov.module_path}",')
        code_lines.append(f'                    class_name="{prov.class_name}"')
        code_lines.append("                ),")

    code_lines.extend(["            ],", "        )", ""])

    return "\n".join(code_lines)
