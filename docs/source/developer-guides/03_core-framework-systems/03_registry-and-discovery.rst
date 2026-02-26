========================
Registry and Discovery
========================

.. currentmodule:: osprey.registry

The Osprey Framework implements a centralized component registration and discovery system that enables clean separation between framework infrastructure and application-specific functionality.

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Understanding :class:`RegistryManager` and component access patterns
   - Choosing between **Extend Mode** (recommended) and **Standalone Mode** registry patterns
   - Implementing application registries with :class:`RegistryConfigProvider`
   - Using ``@capability_node`` decorator for LangGraph integration
   - Registry configuration patterns and best practices
   - Component loading order and dependency management

   **Prerequisites:** Understanding of :doc:`01_state-management-architecture` (AgentState) and Python decorators

   **Time Investment:** 25-35 minutes for complete understanding

System Overview
===============

The registry system uses a two-tier architecture:

**Framework Registry**
  Core infrastructure components (nodes, base capabilities, services)

**Application Registries**
  Domain-specific components (capabilities, context classes, data sources)

Applications register components by implementing ``RegistryConfigProvider`` in their ``registry.py`` module. The framework loads registries from the path specified in configuration:

.. code-block:: yaml

   # config.yml
   registry_path: ./src/my_app/registry.py

Registry Modes
==============

The registry system supports two modes for application registries, detected automatically based on the registry type. **Most applications should use Extend Mode** (recommended default).

.. tab-set::

   .. tab-item:: Extend Mode (Recommended)

      **Extend Mode** allows applications to build on top of framework defaults by adding domain-specific components while automatically inheriting all framework infrastructure.

      **When to use:**
         - Most applications (95% of use cases)
         - When you want automatic framework features
         - When you need to add domain-specific components
         - When you want easier framework upgrades

      **How it works:**
         Framework components load first, then application components merge in. Applications can :ref:`add, exclude, or override specific framework components <excluding-overriding-components>`.

      **Type marker:**
         Returns :class:`ExtendedRegistryConfig` (via :func:`extend_framework_registry` helper)

      **Example:**

      .. code-block:: python

         from osprey.registry import (
             RegistryConfigProvider,
             extend_framework_registry,
             CapabilityRegistration,
             ContextClassRegistration,
             ExtendedRegistryConfig
         )

         class MyAppRegistryProvider(RegistryConfigProvider):
             def get_registry_config(self) -> ExtendedRegistryConfig:
                 return extend_framework_registry(
                     capabilities=[
                         CapabilityRegistration(
                             name="weather",
                             module_path="my_app.capabilities.weather",
                             class_name="WeatherCapability",
                             description="Get weather data",
                             provides=["WEATHER_DATA"],
                             requires=[]
                         )
                     ],
                     context_classes=[
                         ContextClassRegistration(
                             context_type="WEATHER_DATA",
                             module_path="my_app.context_classes",
                             class_name="WeatherContext"
                         )
                     ]
                 )

      **Benefits:**
         - Less boilerplate code (only specify your components)
         - Easier framework upgrades (new features automatically included)
         - Full framework capability access without explicit registration
         - Flexible (can exclude or override framework components as needed)

   .. tab-item:: Standalone Mode (Advanced)

      **Standalone Mode** gives applications complete control by requiring them to define ALL components, including framework infrastructure. The framework registry is NOT loaded.

      **When to use:**
         - Special cases requiring full control over all components
         - Minimal deployments that don't need full framework
         - Custom framework variations

      **How it works:**
         Application provides complete registry including ALL framework components. Framework registry is skipped entirely.

      **Type marker:**
         Returns :class:`RegistryConfig` directly (not via helper)

      **Example:**

      .. code-block:: python

         from osprey.registry import (
             RegistryConfigProvider,
             RegistryConfig,
             NodeRegistration,
             CapabilityRegistration,
             # ... all other registration types
         )

         class StandaloneRegistryProvider(RegistryConfigProvider):
             def get_registry_config(self) -> RegistryConfig:
                 return RegistryConfig(
                     # Must provide ALL framework nodes
                     core_nodes=[
                         NodeRegistration(
                             name="orchestrator",
                             module_path="osprey.langgraph.nodes.orchestrator",
                             function_name="orchestrator_node",
                             description="Core orchestration"
                         ),
                         # ... ALL other framework nodes required
                     ],
                     # Must provide ALL framework capabilities
                     capabilities=[
                         CapabilityRegistration(
                             name="memory",
                             module_path="osprey.capabilities.memory",
                             # ... complete framework memory capability
                         ),
                         # ... ALL other framework capabilities + app capabilities
                     ],
                     # ... ALL other component types
                 )

      .. warning::
         Standalone mode requires maintaining a complete list of framework components.
         This is significantly more maintenance overhead and is only recommended for
         advanced use cases. Framework upgrades may require registry updates.

**Mode Detection:**

The :class:`RegistryManager` automatically detects the mode by checking the type returned by :meth:`RegistryConfigProvider.get_registry_config`:

- ``isinstance(config, ExtendedRegistryConfig)`` → **Extend Mode** (load framework, then merge)
- ``isinstance(config, RegistryConfig)`` but not ``ExtendedRegistryConfig`` → **Standalone Mode** (skip framework)

.. seealso::
   :class:`ExtendedRegistryConfig`
      API reference for the Extend Mode marker class

   :func:`extend_framework_registry`
      Helper function that returns ExtendedRegistryConfig

   :class:`RegistryConfig`
      Base configuration class for Standalone Mode

RegistryManager
===============

The ``RegistryManager`` provides centralized access to all framework components:

.. code-block:: python

   from osprey.registry import initialize_registry, get_registry

   # Initialize the registry system
   initialize_registry()

   # Access the singleton registry instance
   registry = get_registry()

   # Access components
   capability = registry.get_capability("weather_data_retrieval")
   context_class = registry.get_context_class("WEATHER_DATA")
   data_source = registry.get_data_source("knowledge_base")

**Key Methods:**

- ``get_capability(name)`` - Get capability instance by name
- ``get_context_class(context_type)`` - Get context class by type identifier
- ``get_data_source(name)`` - Get data source provider instance
- ``get_node(name)`` - Get LangGraph node function

Advanced Registry Patterns
==========================

This section covers advanced patterns for customizing your application registry beyond the basic examples shown in **Registry Modes** above.

.. _excluding-overriding-components:

Excluding and Overriding Framework Components
---------------------------------------------

You can selectively exclude or replace framework components to customize behavior:

.. code-block:: python

   return extend_framework_registry(

       # Exclude generic Python capability (replaced with specialized turbine_analysis)
       exclude_capabilities=["python"],

       capabilities=[
           # Specialized analysis replaces generic Python capability
           CapabilityRegistration(
               name="turbine_analysis",
               module_path="my_app.capabilities.turbine_analysis",
               class_name="TurbineAnalysisCapability",
               description="Analyze turbine performance with domain-specific logic",
               provides=["ANALYSIS_RESULTS"],
               requires=["TURBINE_DATA", "WEATHER_DATA"]
           )
       ],
       context_classes=[...],

       # Add data sources
       data_sources=[
           DataSourceRegistration(
               name="knowledge_base",
               module_path="my_app.data_sources.knowledge",
               class_name="KnowledgeProvider",
               description="Domain knowledge retrieval"
           )
       ],

       # Add custom framework prompt providers
       framework_prompt_providers=[
           FrameworkPromptProviderRegistration(
               module_path="my_app.framework_prompts",
               prompt_builders={"response_generation": "CustomResponseBuilder"}
           )
       ]
   )

Component Registration
======================

**Capability Registration:**

.. code-block:: python

   # In registry.py
   CapabilityRegistration(
       name="weather_data_retrieval",
       module_path="my_app.capabilities.weather_data_retrieval",
       class_name="WeatherDataRetrievalCapability",
       description="Retrieve weather data for analysis",
       provides=["WEATHER_DATA"],
       requires=["TIME_RANGE"]
   )

   # Implementation in src/my_app/capabilities/weather_data_retrieval.py
   from osprey.base import BaseCapability, capability_node

   @capability_node
   class WeatherDataRetrievalCapability(BaseCapability):
       name = "weather_data_retrieval"
       description = "Retrieve weather data for analysis"
       provides = ["WEATHER_DATA"]
       requires = ["TIME_RANGE"]

       async def execute(self) -> Dict[str, Any]:
           # Get required contexts automatically
           time_range, = self.get_required_contexts()
           # Implementation here
           return self.store_output_context(weather_data)

**Context Class Registration:**

.. code-block:: python

   # In registry.py
   ContextClassRegistration(
       context_type="WEATHER_DATA",
       module_path="my_app.context_classes",
       class_name="WeatherDataContext"
   )

   # Implementation in src/my_app/context_classes.py
   from osprey.context.base import CapabilityContext

   class WeatherDataContext(CapabilityContext):
       CONTEXT_TYPE: ClassVar[str] = "WEATHER_DATA"
       CONTEXT_CATEGORY: ClassVar[str] = "LIVE_DATA"

       location: str
       temperature: float
       conditions: str
       timestamp: datetime

**Data Source Registration:**

.. code-block:: python

   # In registry.py
   DataSourceRegistration(
       name="knowledge_base",
       module_path="my_app.data_sources.knowledge",
       class_name="KnowledgeProvider",
       description="Domain knowledge retrieval"
   )

.. _ariel-service-registration:

ARIEL Service Components
------------------------

Applications that use the :doc:`ARIEL logbook search service </developer-guides/05_production-systems/07_logbook-search-service/index>` can register custom :ref:`ingestion adapters <facility adapters>`, :ref:`enhancement modules <Enhancement Pipeline>`, :doc:`search modules </developer-guides/05_production-systems/07_logbook-search-service/search-modes>`, and :doc:`pipelines </developer-guides/05_production-systems/07_logbook-search-service/search-modes>` through the same ``extend_framework_registry`` mechanism. The framework provides built-in defaults for all ARIEL components, and applications can add custom implementations or exclude built-in ones.

**Ingestion Adapter Registration:**

.. code-block:: python

   from osprey.registry.helpers import extend_framework_registry
   from osprey.registry.base import ArielIngestionAdapterRegistration

   return extend_framework_registry(
       ariel_ingestion_adapters=[
           ArielIngestionAdapterRegistration(
               name="my_facility",
               module_path="my_app.adapters.my_facility",
               class_name="MyFacilityAdapter",
               description="Adapter for My Facility's logbook system",
           ),
       ],
   )

**Enhancement Module Registration:**

.. code-block:: python

   from osprey.registry.helpers import extend_framework_registry
   from osprey.registry.base import ArielEnhancementModuleRegistration

   return extend_framework_registry(
       ariel_enhancement_modules=[
           ArielEnhancementModuleRegistration(
               name="my_enhancer",
               module_path="my_app.enhancement.my_enhancer",
               class_name="MyEnhancerModule",
               description="Custom enhancement module",
               execution_order=30,  # Runs after built-in modules (10, 20)
           ),
       ],
   )

You can also exclude built-in ARIEL components with ``exclude_ariel_ingestion_adapters``, ``exclude_ariel_enhancement_modules``, ``exclude_ariel_search_modules``, and ``exclude_ariel_pipelines``.

.. _custom-ai-provider-registration:

AI Provider Registration
------------------------

Applications can register custom AI providers for institutional services or commercial providers not included in the framework.

**Basic Provider Registration:**

.. code-block:: python

   # In src/my_app/registry.py
   from osprey.registry import RegistryConfigProvider, ProviderRegistration
   from osprey.registry.helpers import extend_framework_registry

   class MyAppRegistryProvider(RegistryConfigProvider):
       def get_registry_config(self):
           return extend_framework_registry(
               capabilities=[...],
               context_classes=[...],
               providers=[
                   ProviderRegistration(
                       module_path="my_app.providers.azure",
                       class_name="AzureOpenAIProviderAdapter"
                   ),
                   ProviderRegistration(
                       module_path="my_app.providers.institutional",
                       class_name="InstitutionalAIProvider"
                   )
               ]
           )

**Excluding Framework Providers:**

You can exclude framework providers if you want to use only custom providers:

.. code-block:: python

   return extend_framework_registry(
       capabilities=[...],
       providers=[
           ProviderRegistration(
               module_path="my_app.providers.custom",
               class_name="CustomProvider"
           )
       ],
       exclude_providers=["anthropic", "google"]  # Exclude specific framework providers
   )

**Replacing Framework Providers:**

To replace a framework provider with a custom implementation:

.. code-block:: python

   return extend_framework_registry(
       capabilities=[...],
       override_providers=[
           ProviderRegistration(
               module_path="my_app.providers.custom_openai",
               class_name="CustomOpenAIProvider"
           )
       ],
       exclude_providers=["openai"]  # Remove framework version
   )

**Provider Implementation:**

Custom providers are useful for integrating institutional AI services (e.g., Stanford AI Playground, LBNL CBorg), commercial providers not yet in the framework (e.g., Cohere, Mistral AI), or self-hosted models with custom endpoints.

All custom providers must inherit from :class:`BaseProvider` and implement two core methods:

- ``execute_completion()`` - Execute completions via LiteLLM (used by infrastructure nodes)
- ``check_health()`` - Test connectivity and authentication (used by ``osprey health`` CLI)

.. code-block:: python

   # Implementation in src/my_app/providers/institutional.py
   from typing import Any
   from osprey.models.providers.base import BaseProvider
   from osprey.models.providers.litellm_adapter import (
       execute_litellm_completion,
       check_litellm_health,
   )

   class InstitutionalAIProvider(BaseProvider):
       """Custom provider for institutional AI service."""

       # Provider metadata - displayed in CLI and used by framework
       name = "institutional_ai"
       description = "Institutional AI Service (Custom Models)"
       requires_api_key = True
       requires_base_url = True
       requires_model_id = True
       supports_proxy = True
       default_base_url = "https://ai.institution.edu/v1"
       default_model_id = "gpt-4"
       health_check_model_id = "gpt-3.5-turbo"  # Cheapest for health checks
       available_models = ["gpt-4", "gpt-3.5-turbo", "custom-model"]

       # API key acquisition info (shown in CLI help)
       api_key_url = "https://ai.institution.edu/api-keys"
       api_key_instructions = [
           "Log in with institutional credentials",
           "Navigate to API Keys section",
           "Generate new API key",
           "Copy and save the key securely"
       ]
       api_key_note = "Requires active institutional affiliation"

       def execute_completion(
           self,
           message: str,
           model_id: str,
           api_key: str | None,
           base_url: str | None,
           max_tokens: int = 1024,
           temperature: float = 0.0,
           **kwargs
       ) -> str | Any:
           """Execute completion via LiteLLM adapter."""
           return execute_litellm_completion(
               provider=self.name,
               message=message,
               model_id=model_id,
               api_key=api_key,
               base_url=base_url,
               max_tokens=max_tokens,
               temperature=temperature,
               **kwargs,
           )

       def check_health(
           self,
           api_key: str | None,
           base_url: str | None,
           timeout: float = 5.0,
           model_id: str | None = None
       ) -> tuple[bool, str]:
           """Test provider connectivity via LiteLLM."""
           return check_litellm_health(
               provider=self.name,
               api_key=api_key,
               base_url=base_url,
               timeout=timeout,
               model_id=model_id or self.health_check_model_id,
           )

Once registered, custom providers integrate seamlessly with the framework:

- Available in ``osprey init`` interactive provider selection
- Accessible via :func:`osprey.models.get_chat_completion`
- Tested automatically by ``osprey health`` command
- Configuration managed in ``config.yml`` like framework providers

.. dropdown:: 📋 Complete Provider Metadata Reference
   :color: info
   :icon: info

   **Required Metadata Attributes:**

   .. list-table::
      :widths: 30 70
      :header-rows: 1

      * - Attribute
        - Description
      * - ``name``
        - Provider identifier (e.g., ``"azure_openai"``, ``"institutional_ai"``)
      * - ``description``
        - User-friendly description shown in TUI (e.g., ``"Azure OpenAI (Enterprise)"``)
      * - ``requires_api_key``
        - ``True`` if provider needs API key for authentication
      * - ``requires_base_url``
        - ``True`` if provider needs custom base URL
      * - ``requires_model_id``
        - ``True`` if provider requires model ID specification
      * - ``supports_proxy``
        - ``True`` if provider supports HTTP proxy configuration

   **Optional Metadata Attributes:**

   .. list-table::
      :widths: 30 70
      :header-rows: 1

      * - Attribute
        - Description
      * - ``default_base_url``
        - Default API endpoint (e.g., ``"https://api.openai.com/v1"``)
      * - ``default_model_id``
        - Recommended model for general use (used in project templates)
      * - ``health_check_model_id``
        - Cheapest/fastest model for ``osprey health`` checks
      * - ``available_models``
        - List of model IDs for CLI selection (e.g., ``["gpt-4", "gpt-3.5-turbo"]``)
      * - ``api_key_url``
        - URL where users obtain API keys (e.g., ``"https://console.anthropic.com/"``)
      * - ``api_key_instructions``
        - Step-by-step list of strings for obtaining the key
      * - ``api_key_note``
        - Additional requirements (e.g., ``"Requires institutional affiliation"``)

   **Implementation Notes:**

   - Use ``execute_litellm_completion()`` and ``check_litellm_health()`` from the adapter
   - For OpenAI-compatible APIs, LiteLLM handles them via ``openai/`` prefix with custom ``api_base``
   - ``check_health()`` should use minimal tokens (typically 5-10 tokens, ~$0.0001 cost)
   - All metadata is introspected from class attributes (single source of truth)

Registry Initialization and Usage
=================================

.. code-block:: python

   from osprey.registry import initialize_registry, get_registry

   # Initialize registry (loads framework + application components)
   initialize_registry()

   # Access registry throughout application
   registry = get_registry()

   # Get capabilities
   capability = registry.get_capability("weather_data_retrieval")
   all_capabilities = registry.get_all_capabilities()

   # Get context classes
   weather_context_class = registry.get_context_class("WEATHER_DATA")

   # Get data sources
   knowledge_provider = registry.get_data_source("knowledge_base")

**Registry Export for External Tools:**

The registry system automatically exports metadata during initialization for use by external tools and debugging:

.. code-block:: python

   # Export happens automatically during initialization
   initialize_registry(auto_export=True)  # Default behavior

   # Manual export for debugging or integration
   registry = get_registry()
   export_data = registry.export_registry_to_json("/path/to/export")

   # Export creates standardized JSON files:
   # - registry_export.json (complete metadata)
   # - capabilities.json (capability definitions)
   # - context_types.json (context type definitions)

**Export Configuration:**

The default export directory is configured in ``config.yml``:

.. code-block:: yaml

   file_paths:
     agent_data_dir: _agent_data
     registry_exports_dir: registry_exports

This creates exports in ``_agent_data/registry_exports/`` relative to the project root. The path can be customized through the configuration system.

Component Loading Order
=======================

Components are loaded lazily during registry initialization:

1. **Context classes** - Required by capabilities
2. **Data sources** - Required by capabilities
3. **Providers** - AI model providers
4. **Core nodes** - Infrastructure components
5. **Services** - Internal LangGraph service graphs
6. **Capabilities** - Domain-specific functionality
7. **Framework prompt providers** - Application-specific prompts

Best Practices and Troubleshooting
===================================

.. tab-set::

   .. tab-item:: Best Practices

      **Registry Configuration:**
         - Keep registrations simple and focused
         - Use clear, descriptive names and descriptions
         - Define ``provides`` and ``requires`` accurately for dependency tracking

      **Capability Implementation:**
         - Always use ``@capability_node`` decorator
         - Implement required attributes: ``name``, ``description``
         - Make ``execute()`` method static and async
         - Return dictionary of state updates

      **Application Structure:**
         - Place registry in ``src/{app_name}/registry.py``
         - Implement exactly one ``RegistryConfigProvider`` class per application
         - Organize capabilities in ``src/{app_name}/capabilities/`` directory
         - Configure registry location in ``config.yml`` via ``registry_path``

   .. tab-item:: Common Issues

      **Component Not Found:**
         1. Verify component is registered in ``RegistryConfigProvider``
         2. Check module path and class name are correct
         3. Ensure ``initialize_registry()`` was called

      **Missing @capability_node:**
         1. Ensure ``@capability_node`` decorator is applied
         2. Verify ``name`` and ``description`` class attributes exist
         3. Check that ``execute()`` method is implemented as static method

      **Registry Export Issues:**
         1. Check that export directory is writable and accessible
         2. Verify ``auto_export=True`` during initialization for automatic exports
         3. Use manual ``export_registry_to_json()`` for debugging specific states

.. seealso::
   :doc:`05_message-and-execution-flow`
       Complete message processing pipeline using registered components
   :doc:`01_state-management-architecture`
       State management integration with registry system
