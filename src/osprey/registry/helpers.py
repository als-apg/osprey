"""Registry helper functions for application development.

This module provides utilities that simplify application registry creation
by handling the common pattern of extending the framework registry with
application-specific components.

The helper functions eliminate boilerplate code and provide a clean,
intuitive API for application developers to define their registries.
"""


from .base import (
    CapabilityRegistration,
    ConnectorRegistration,
    ContextClassRegistration,
    DataSourceRegistration,
    ExtendedRegistryConfig,
    FrameworkPromptProviderRegistration,
    NodeRegistration,
    ProviderRegistration,
    RegistryConfig,
    ServiceRegistration,
)


def extend_framework_registry(
    capabilities: list[CapabilityRegistration] | None = None,
    context_classes: list[ContextClassRegistration] | None = None,
    data_sources: list[DataSourceRegistration] | None = None,
    services: list[ServiceRegistration] | None = None,
    framework_prompt_providers: list[FrameworkPromptProviderRegistration] | None = None,
    providers: list[ProviderRegistration] | None = None,
    connectors: list[ConnectorRegistration] | None = None,
    core_nodes: list[NodeRegistration] | None = None,
    exclude_capabilities: list[str] | None = None,
    exclude_nodes: list[str] | None = None,
    exclude_context_classes: list[str] | None = None,
    exclude_data_sources: list[str] | None = None,
    exclude_providers: list[str] | None = None,
    exclude_connectors: list[str] | None = None,
    override_capabilities: list[CapabilityRegistration] | None = None,
    override_nodes: list[NodeRegistration] | None = None,
    override_providers: list[ProviderRegistration] | None = None,
    override_connectors: list[ConnectorRegistration] | None = None,
) -> ExtendedRegistryConfig:
    """Create application registry configuration that extends the framework.

    This is the recommended way to create application registries. It simplifies
    registry creation by automatically handling framework component exclusions
    and overrides through clean, declarative parameters.

    The function returns an application registry configuration that will be
    merged with the framework registry by the RegistryManager. You only need
    to specify your application-specific components and any framework components
    you want to exclude or replace.

    Most applications will only need to specify capabilities and context_classes.

    Args:
        capabilities: Application capabilities to add to framework defaults
        context_classes: Application context classes to add to framework defaults
        data_sources: Application data sources to add to framework defaults
        services: Application services to add to framework defaults
        framework_prompt_providers: Application prompt providers to add
        providers: Application AI model providers to add to framework defaults
        connectors: Application control system/archiver connectors to add
        exclude_capabilities: Names of framework capabilities to exclude
        exclude_nodes: Names of framework nodes to exclude
        exclude_context_classes: Context types to exclude from framework
        exclude_data_sources: Names of framework data sources to exclude
        exclude_providers: Names of framework providers to exclude
        exclude_connectors: Names of framework connectors to exclude
        override_capabilities: Capabilities that replace framework versions (by name)
        override_nodes: Nodes that replace framework versions (by name)
        override_providers: Providers that replace framework versions (by name)
        override_connectors: Connectors that replace framework versions (by name)

    Returns:
        ExtendedRegistryConfig that signals extend mode to registry manager

    Examples:
        Simple application (most common)::

            def get_registry_config(self) -> ExtendedRegistryConfig:
                return extend_framework_registry(
                    capabilities=[
                        CapabilityRegistration(
                            name="weather",
                            module_path="my_app.capabilities.weather",
                            class_name="WeatherCapability",
                            description="Get weather information",
                            provides=["WEATHER_DATA"],
                            requires=[]
                        ),
                    ],
                    context_classes=[
                        ContextClassRegistration(
                            context_type="WEATHER_DATA",
                            module_path="my_app.context_classes",
                            class_name="WeatherContext"
                        ),
                    ]
                )

        Exclude framework component::

            def get_registry_config(self) -> ExtendedRegistryConfig:
                return extend_framework_registry(
                    capabilities=[...],
                    exclude_capabilities=["python"],  # Don't need framework Python
                )

        Override framework component::

            def get_registry_config(self) -> ExtendedRegistryConfig:
                return extend_framework_registry(
                    capabilities=[...],
                    override_capabilities=[
                        CapabilityRegistration(
                            name="memory",  # Replace framework memory
                            module_path="my_app.capabilities.custom_memory",
                            class_name="CustomMemoryCapability",
                            description="Custom memory implementation",
                            provides=["MEMORY_CONTEXT"],
                            requires=[]
                        ),
                    ]
                )

        Add custom AI model providers::

            def get_registry_config(self) -> RegistryConfig:
                return extend_framework_registry(
                    capabilities=[...],
                    context_classes=[...],
                    providers=[
                        ProviderRegistration(
                            module_path="my_app.providers.custom_ai",
                            class_name="CustomAIProviderAdapter"
                        ),
                        ProviderRegistration(
                            module_path="my_app.providers.institutional_ai",
                            class_name="InstitutionalAIProviderAdapter"
                        ),
                    ]
                )

    .. note::
       The returned configuration contains only application components. The
       framework registry system automatically merges this with framework defaults
       during initialization. Exclusions are handled internally via the
       framework_exclusions field.

    .. seealso::
       :func:`get_framework_defaults` : Inspect framework components
       :class:`RegistryConfig` : The returned configuration structure
    """
    # Build framework exclusions dict for the merge process
    framework_exclusions = {}

    if exclude_capabilities:
        framework_exclusions["capabilities"] = exclude_capabilities

    if exclude_nodes:
        framework_exclusions["nodes"] = exclude_nodes

    if exclude_context_classes:
        framework_exclusions["context_classes"] = exclude_context_classes

    if exclude_data_sources:
        framework_exclusions["data_sources"] = exclude_data_sources

    if exclude_providers:
        framework_exclusions["providers"] = exclude_providers

    if exclude_connectors:
        framework_exclusions["connectors"] = exclude_connectors

    # Combine override and regular components
    all_capabilities = list(capabilities or [])
    if override_capabilities:
        all_capabilities.extend(override_capabilities)

    all_nodes = list(core_nodes or [])
    if override_nodes:
        all_nodes.extend(override_nodes)

    all_providers = list(providers or [])
    if override_providers:
        all_providers.extend(override_providers)

    all_connectors = list(connectors or [])
    if override_connectors:
        all_connectors.extend(override_connectors)

    # Return ExtendedRegistryConfig to signal extend mode (framework will be merged by RegistryManager)
    return ExtendedRegistryConfig(
        core_nodes=all_nodes,
        capabilities=all_capabilities,
        context_classes=list(context_classes or []),
        data_sources=list(data_sources or []),
        services=list(services or []),
        framework_prompt_providers=list(framework_prompt_providers or []),
        providers=all_providers,
        connectors=all_connectors,
        framework_exclusions=framework_exclusions if framework_exclusions else None
    )


def get_framework_defaults() -> RegistryConfig:
    """Get the default framework registry configuration.

    This function returns the complete framework registry without any
    application modifications. Useful for inspecting what components
    the framework provides or for manual registry merging.

    Returns:
        Complete framework RegistryConfig with all core components

    Examples:
        Inspect framework components::

            >>> framework = get_framework_defaults()
            >>> print(f"Framework provides {len(framework.capabilities)} capabilities")
            >>> for cap in framework.capabilities:
            ...     print(f"  - {cap.name}: {cap.description}")

        Manual merging (advanced)::

            >>> framework = get_framework_defaults()
            >>> my_config = RegistryConfig(
            ...     core_nodes=framework.core_nodes,
            ...     capabilities=framework.capabilities + my_capabilities,
            ...     context_classes=framework.context_classes + my_context_classes,
            ...     data_sources=framework.data_sources,
            ...     services=framework.services,
            ...     framework_prompt_providers=framework.framework_prompt_providers,
            ...     initialization_order=framework.initialization_order
            ... )

    .. note::
       Most applications should use :func:`extend_framework_registry` instead
       of manually merging. This function is provided for inspection and
       advanced use cases.

    .. seealso::
       :func:`extend_framework_registry` : Recommended way to extend framework
       :class:`FrameworkRegistryProvider` : The provider that generates this config
    """
    from .registry import FrameworkRegistryProvider
    provider = FrameworkRegistryProvider()
    return provider.get_registry_config()


def generate_explicit_registry_code(
    app_class_name: str,
    app_display_name: str,
    package_name: str,
    capabilities: list[CapabilityRegistration] | None = None,
    context_classes: list[ContextClassRegistration] | None = None,
    data_sources: list[DataSourceRegistration] | None = None,
    services: list[ServiceRegistration] | None = None,
    framework_prompt_providers: list[FrameworkPromptProviderRegistration] | None = None,
) -> str:
    """Generate explicit registry Python code with all framework + app components.

    This function creates a complete explicit registry as Python source code,
    combining framework defaults with application-specific components. This is
    useful for template generation where you want the full registry visible.

    Args:
        app_class_name: Python class name for the registry provider (e.g., "WeatherAgentRegistryProvider")
        app_display_name: Human-readable application name (e.g., "Weather Agent")
        package_name: Python package name (e.g., "weather_agent")
        capabilities: Application-specific capabilities to add
        context_classes: Application-specific context classes to add
        data_sources: Application-specific data sources to add (optional)
        services: Application-specific services to add (optional)
        framework_prompt_providers: Application-specific prompt providers (optional)

    Returns:
        Complete Python source code for the explicit registry

    Examples:
        Generate registry for a simple app::

            >>> code = generate_explicit_registry_code(
            ...     app_class_name="WeatherAgentRegistryProvider",
            ...     app_display_name="Weather Agent",
            ...     package_name="weather_agent",
            ...     capabilities=[
            ...         CapabilityRegistration(
            ...             name="current_weather",
            ...             module_path="weather_agent.capabilities.current_weather",
            ...             class_name="CurrentWeatherCapability",
            ...             description="Get current weather",
            ...             provides=["CURRENT_WEATHER"],
            ...             requires=[]
            ...         )
            ...     ],
            ...     context_classes=[
            ...         ContextClassRegistration(
            ...             context_type="CURRENT_WEATHER",
            ...             module_path="weather_agent.context_classes",
            ...             class_name="CurrentWeatherContext"
            ...         )
            ...     ]
            ... )
            >>> print(code[:100])
            '''
            Component registry for Weather Agent.
            ...

    """
    # Get framework defaults
    framework = get_framework_defaults()

    # Helper function to format a registration as code
    def format_node_registration(reg: NodeRegistration, indent: str = "                ") -> str:
        return f'''{indent}NodeRegistration(
{indent}    name="{reg.name}",
{indent}    module_path="{reg.module_path}",
{indent}    function_name="{reg.function_name}",
{indent}    description="{reg.description}"
{indent})'''

    def format_capability_registration(reg: CapabilityRegistration, indent: str = "                ") -> str:
        lines = [
            f'{indent}CapabilityRegistration(',
            f'{indent}    name="{reg.name}",',
            f'{indent}    module_path="{reg.module_path}",',
            f'{indent}    class_name="{reg.class_name}",',
            f'{indent}    description="{reg.description}",',
            f'{indent}    provides={reg.provides},',
            f'{indent}    requires={reg.requires},',
        ]
        if hasattr(reg, 'always_active') and reg.always_active:
            lines.append(f'{indent}    always_active=True,')
        if hasattr(reg, 'functional_node') and reg.functional_node:
            lines.append(f'{indent}    functional_node="{reg.functional_node}",')
        # Remove trailing comma from last line before closing paren
        if lines[-1].endswith(','):
            lines[-1] = lines[-1][:-1]
        lines.append(f'{indent})')
        return '\n'.join(lines)

    def format_context_class_registration(reg: ContextClassRegistration, indent: str = "                ") -> str:
        return f'''{indent}ContextClassRegistration(
{indent}    context_type="{reg.context_type}",
{indent}    module_path="{reg.module_path}",
{indent}    class_name="{reg.class_name}"
{indent})'''

    def format_data_source_registration(reg: DataSourceRegistration, indent: str = "                ") -> str:
        return f'''{indent}DataSourceRegistration(
{indent}    name="{reg.name}",
{indent}    module_path="{reg.module_path}",
{indent}    class_name="{reg.class_name}",
{indent}    description="{reg.description}",
{indent}    health_check_required={reg.health_check_required}
{indent})'''

    def format_service_registration(reg: ServiceRegistration, indent: str = "                ") -> str:
        return f'''{indent}ServiceRegistration(
{indent}    name="{reg.name}",
{indent}    module_path="{reg.module_path}",
{indent}    class_name="{reg.class_name}",
{indent}    description="{reg.description}",
{indent}    provides={reg.provides},
{indent}    requires={reg.requires},
{indent}    internal_nodes={reg.internal_nodes}
{indent})'''

    # Build the code sections
    code_lines = [
        '"""',
        f'Component registry for {app_display_name}.',
        '',
        'This registry uses the EXPLICIT style, listing all framework components',
        'alongside application-specific components for full visibility and control.',
        '"""',
        '',
        'from osprey.registry import (',
        '    RegistryConfigProvider,',
        '    RegistryConfig,',
        '    NodeRegistration,',
        '    CapabilityRegistration,',
        '    ContextClassRegistration,',
        '    DataSourceRegistration,',
        '    ServiceRegistration,',
        '    FrameworkPromptProviderRegistration,',
        '    ProviderRegistration',
        ')',
        '',
        '',
        f'class {app_class_name}(RegistryConfigProvider):',
        f'    """Registry provider for {app_display_name}."""',
        '    ',
        '    def get_registry_config(self):',
        f'        """Return registry configuration for {app_display_name}."""',
        '        # EXPLICIT REGISTRY: All framework + application components listed',
        '        return RegistryConfig(',
        '            # ================================================================',
        '            # FRAMEWORK CORE NODES',
        '            # ================================================================',
        '            core_nodes=[',
    ]

    # Add framework nodes
    for i, node in enumerate(framework.core_nodes):
        code_lines.append(format_node_registration(node))
        if i < len(framework.core_nodes) - 1:
            code_lines[-1] += ','

    code_lines.extend([
        '            ],',
        '',
        '            # ================================================================',
        '            # ALL CAPABILITIES (Framework + Application)',
        '            # ================================================================',
        '            capabilities=[',
        '                # ---- Framework Capabilities ----',
    ])

    # Add framework capabilities
    for i, cap in enumerate(framework.capabilities):
        code_lines.append(format_capability_registration(cap))
        code_lines[-1] += ','

    # Add application capabilities
    if capabilities:
        code_lines.extend([
            '',
            '                # ---- Application Capabilities ----',
        ])
        for i, cap in enumerate(capabilities):
            code_lines.append(format_capability_registration(cap))
            if i < len(capabilities) - 1:
                code_lines[-1] += ','

    code_lines.extend([
        '            ],',
        '',
        '            # ================================================================',
        '            # ALL CONTEXT CLASSES (Framework + Application)',
        '            # ================================================================',
        '            context_classes=[',
        '                # ---- Framework Context Classes ----',
    ])

    # Add framework context classes
    for i, ctx in enumerate(framework.context_classes):
        code_lines.append(format_context_class_registration(ctx))
        code_lines[-1] += ','

    # Add application context classes
    if context_classes:
        code_lines.extend([
            '',
            '                # ---- Application Context Classes ----',
        ])
        for i, ctx in enumerate(context_classes):
            code_lines.append(format_context_class_registration(ctx))
            if i < len(context_classes) - 1:
                code_lines[-1] += ','

    code_lines.extend([
        '            ],',
        '',
        '            # ================================================================',
        '            # DATA SOURCES (Framework + Application)',
        '            # ================================================================',
        '            data_sources=[',
        '                # ---- Framework Data Sources ----',
    ])

    # Add framework data sources
    for i, ds in enumerate(framework.data_sources):
        code_lines.append(format_data_source_registration(ds))
        if data_sources or i < len(framework.data_sources) - 1:
            code_lines[-1] += ','

    # Add application data sources
    if data_sources:
        code_lines.extend([
            '',
            '                # ---- Application Data Sources ----',
        ])
        for i, ds in enumerate(data_sources):
            code_lines.append(format_data_source_registration(ds))
            if i < len(data_sources) - 1:
                code_lines[-1] += ','

    code_lines.extend([
        '            ],',
        '',
        '            # ================================================================',
        '            # SERVICES (Framework + Application)',
        '            # ================================================================',
        '            services=[',
        '                # ---- Framework Services ----',
    ])

    # Add framework services
    for i, svc in enumerate(framework.services):
        code_lines.append(format_service_registration(svc))
        if services or i < len(framework.services) - 1:
            code_lines[-1] += ','

    # Add application services
    if services:
        code_lines.extend([
            '',
            '                # ---- Application Services ----',
        ])
        for i, svc in enumerate(services):
            code_lines.append(format_service_registration(svc))
            if i < len(services) - 1:
                code_lines[-1] += ','

    code_lines.extend([
        '            ],',
        '',
        '            # ================================================================',
        '            # AI MODEL PROVIDERS',
        '            # ================================================================',
        '            providers=[',
    ])

    # Add framework AI model providers
    for prov in framework.providers:
        code_lines.append('                ProviderRegistration(')
        code_lines.append(f'                    module_path="{prov.module_path}",')
        code_lines.append(f'                    class_name="{prov.class_name}"')
        code_lines.append('                ),')

    code_lines.extend([
        '            ],',
        '',
        '            # ================================================================',
        '            # FRAMEWORK PROMPT PROVIDERS',
        '            # ================================================================',
        '            framework_prompt_providers=[',
    ])

    # Add framework prompt providers
    for prov in framework.framework_prompt_providers:
        code_lines.append('                FrameworkPromptProviderRegistration(')
        code_lines.append(f'                    module_path="{prov.module_path}",')
        code_lines.append('                    prompt_builders={')
        for key, value in prov.prompt_builders.items():
            code_lines.append(f'                        "{key}": "{value}",')
        code_lines.append('                    }')
        code_lines.append('                ),')

    # Add application prompt providers if any
    if framework_prompt_providers:
        code_lines.append('')
        for prov in framework_prompt_providers:
            code_lines.append('                FrameworkPromptProviderRegistration(')
            code_lines.append(f'                    module_path="{prov.module_path}",')
            code_lines.append('                    prompt_builders={')
            for key, value in prov.prompt_builders.items():
                code_lines.append(f'                        "{key}": "{value}",')
            code_lines.append('                    }')
            code_lines.append('                ),')

    code_lines.extend([
        '            ],',
        '        )',
        ''
    ])

    return '\n'.join(code_lines)

