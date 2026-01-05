Convention over Configuration: Configuration-Driven Registry Patterns
======================================================================

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Configuration-driven component loading and explicit registry patterns
   - Using ``@capability_node`` and ``@infrastructure_node`` decorators
   - Application registry implementation with :class:`RegistryConfigProvider`
   - Component requirements and streaming integration
   - Convention-based module loading and dependency management

   **Prerequisites:** Understanding of Python decorators and class inheritance

   **Time Investment:** 15-20 minutes for complete understanding

Overview
========

The Osprey Framework eliminates boilerplate through convention-based configuration loading and explicit registry patterns. Components are declared in registry configurations and loaded using standardized naming conventions.

Project Creation and Structure
==============================

Creating a New Project
~~~~~~~~~~~~~~~~~~~~~~

Projects are created using the ``osprey init`` command with predefined templates:

.. code-block:: bash

   # Install the framework
   pip install osprey-framework

   # Create a new project from a template
   osprey init my-agent --template hello_world_weather
   cd my-agent

Available templates include:

- ``minimal`` - Basic skeleton for starting from scratch
- ``hello_world_weather`` - Simple weather agent (recommended for learning)
- ``control_assistant`` - Production control system integration template

Standard Project Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generated projects follow a consistent structure:

.. code-block::

   my-agent/
   ├── src/
   │   └── my_agent/              # Package name derived from project name
   │       ├── __init__.py
   │       ├── registry.py         # Component registration
   │       ├── context_classes.py  # Data models
   │       ├── capabilities/       # Business logic
   │       └── ...
   ├── services/                   # Container configurations
   ├── config.yml                  # Application settings
   └── .env.example                # Environment variables template

This structure provides:

- **Clear separation**: Application code in ``src/``, configuration at root level
- **Module consistency**: Package name matches project directory (``my-agent`` → ``my_agent``)
- **Self-contained**: Each project includes complete configuration and service definitions

Core Architecture
=================

Configuration-Driven Loading System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework uses three key patterns:

1. **Explicit Component Registration**: Components are declared in registry configurations with full metadata
2. **Configuration-Driven Discovery**: Registry location specified in ``config.yml``, components use simple module paths
3. **Interface-Based Registry Pattern**: `RegistryConfigProvider` ensures type-safe component declarations

This approach reduces boilerplate by ~80% while ensuring consistency and avoiding hidden dependencies.

Component Decorators
====================

@capability_node Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Transforms capability classes into LangGraph-compatible nodes with complete infrastructure:

.. code-block:: python

   from osprey.base import BaseCapability, capability_node
   from osprey.state import AgentState
   from typing import Dict, Any

   @capability_node
   class WeatherCapability(BaseCapability):
       name = "weather_data"
       description = "Retrieve current weather conditions"
       provides = ["WEATHER_DATA"]
       requires = ["LOCATION"]

       async def execute(self) -> Dict[str, Any]:
           # Extract the location context
           location, = self.get_required_contexts()

           # Use context attributes directly (e.g., city, country)
           weather_data = await fetch_weather(location.city, location.country)

           # Create context object with the data
           weather_context = WeatherDataContext(
               current_conditions=weather_data,
               last_updated=datetime.now().isoformat()
           )

           # Use helper method to store - automatically handles state updates
           return self.store_output_context(weather_context)

**Infrastructure Features Provided:**
- LangGraph node creation (`langgraph_node` attribute)
- Error handling and classification
- State management and step progression
- Streaming support via LangGraph
- Performance monitoring
- Validation of required components

@infrastructure_node Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates infrastructure components for system operations:

.. code-block:: python

   from osprey.base import BaseInfrastructureNode, infrastructure_node

   @infrastructure_node
   class TaskExtractionNode(BaseInfrastructureNode):
       name = "task_extraction"
       description = "Extract actionable tasks from conversation"

       @staticmethod
       async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
           # Extract task from conversation
           task = await extract_task_from_messages(state["messages"])
           return {"task_current_task": task}

**Infrastructure vs Capability:**
- **Infrastructure**: System components (orchestration, routing, classification)
- **Capabilities**: Business logic components (data analysis, PV finding, etc.)
- **Same patterns**: Identical decorator and validation patterns
- **Different defaults**: Infrastructure has more conservative error handling

Registry System
===============

Application Registry Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each application provides a registry configuration using the ``extend_framework_registry()`` helper:

.. code-block:: python

   # File: src/my_agent/registry.py
   from osprey.registry import (
       extend_framework_registry,
       CapabilityRegistration,
       ContextClassRegistration,
       RegistryConfig,
       RegistryConfigProvider
   )

   class MyAgentRegistryProvider(RegistryConfigProvider):
       def get_registry_config(self) -> RegistryConfig:
           return extend_framework_registry(
               capabilities=[
                   CapabilityRegistration(
                       name="weather_data_retrieval",
                       module_path="my_agent.capabilities.weather_data_retrieval",
                       class_name="WeatherDataRetrievalCapability",
                       description="Retrieve weather data for analysis",
                       provides=["WEATHER_DATA"],
                       requires=["TIME_RANGE"]
                   ),
                   CapabilityRegistration(
                       name="turbine_analysis",
                       module_path="my_agent.capabilities.turbine_analysis",
                       class_name="TurbineAnalysisCapability",
                       description="Analyze turbine performance data",
                       provides=["ANALYSIS_RESULTS"],
                       requires=["TURBINE_DATA", "WEATHER_DATA"]
                   )
               ],
               context_classes=[
                   ContextClassRegistration(
                       context_type="WEATHER_DATA",
                       module_path="my_agent.context_classes",
                       class_name="WeatherDataContext"
                   ),
                   ContextClassRegistration(
                       context_type="ANALYSIS_RESULTS",
                       module_path="my_agent.context_classes",
                       class_name="AnalysisResultsContext"
                   )
               ]
           )

The ``extend_framework_registry()`` helper automatically includes all framework capabilities (memory, Python execution, time parsing, etc.) while adding your application-specific components.

.. dropdown:: **Advanced Registry Patterns**
   :color: info
   :icon: tools

   For specialized use cases, the framework provides advanced registry configuration patterns:

   .. tab-set::

      .. tab-item:: Explicit Registration

         For complete control over all registered components, you can explicitly list everything:

         .. code-block:: python

            class MyAgentRegistryProvider(RegistryConfigProvider):
                def get_registry_config(self) -> RegistryConfig:
                    return RegistryConfig(
                        capabilities=[
                            # Must explicitly list all capabilities including framework ones
                            CapabilityRegistration(
                                name="orchestrator",
                                module_path="osprey.infrastructure.orchestrator_node",
                                class_name="OrchestratorCapability",
                                # ... full configuration
                            ),
                            # ... all other framework capabilities
                            CapabilityRegistration(
                                name="my_capability",
                                module_path="my_agent.capabilities.my_capability",
                                class_name="MyCapability",
                                provides=["MY_DATA"],
                                requires=[]
                            )
                        ],
                        context_classes=[...],
                        # ... all other registry sections
                    )

         **When to use:**

         - Learning how the registry system works internally
         - Debugging registry issues
         - Needing complete control over initialization order

         **Recommendation:** Use ``extend_framework_registry()`` for standard workflows - it's cleaner and less error-prone.

      .. tab-item:: Excluding Framework Components

         Replace framework capabilities with specialized versions:

         .. code-block:: python

            class MyAgentRegistryProvider(RegistryConfigProvider):
                def get_registry_config(self) -> RegistryConfig:
                    return extend_framework_registry(
                        capabilities=[
                            CapabilityRegistration(
                                name="specialized_python",
                                module_path="my_agent.capabilities.specialized_python",
                                class_name="SpecializedPythonCapability",
                                description="Domain-specific Python execution",
                                provides=["ANALYSIS_RESULTS"],
                                requires=["DOMAIN_DATA"]
                            )
                        ],
                        # Exclude generic Python capability to avoid conflicts
                        exclude_capabilities=["python"]
                    )

         **Common use cases:**

         - Building domain-specific versions of framework capabilities
         - Implementing custom approval workflows
         - Adding specialized error handling for specific domains
         - Requiring different dependency resolution

Registry Initialization
~~~~~~~~~~~~~~~~~~~~~~~

The framework uses configuration-driven registry discovery:

**Configuration Setup:**

.. code-block:: yaml

   # In your project's config.yml
   registry_path: src/my_agent/registry.py  # Relative or absolute path

**Initialization Process:**

The framework systematically:

1. Reads ``registry_path`` from configuration (supports relative and absolute paths)
2. Dynamically imports the registry provider
3. Calls ``get_registry_config()`` to obtain component registrations
4. Imports components using their module paths
5. Validates dependencies and initialization order
6. Creates component instances ready for use

.. code-block:: python

   from osprey.registry import initialize_registry, get_registry

   # Initialize the registry system (automatically uses registry_path from config)
   initialize_registry()

   # Access components
   registry = get_registry()
   capability = registry.get_capability("weather_data_retrieval")

Component Requirements
======================

Registry Declaration Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All components must be explicitly declared in registry configurations and implement required patterns:

.. code-block:: python

   @capability_node  # or @infrastructure_node
   class MyComponent(BaseCapability):  # or BaseInfrastructureNode
       # REQUIRED: Validated at decoration time
       name: str = "my_component"
       description: str = "Component description"

       # REQUIRED: Main execution logic (instance method for capabilities)
       async def execute(self) -> Dict[str, Any]:
           # Create context object
           result_context = MyResultContext(status="success")
           # Use helper method to store
           return self.store_output_context(result_context)

       # OPTIONAL: Custom error handling (inherits defaults)
       @staticmethod
       def classify_error(exc: Exception, context: dict) -> ErrorClassification:
           if isinstance(exc, ConnectionError):
               return ErrorClassification(
                   severity=ErrorSeverity.RETRIABLE,
                   user_message="Connection lost, retrying...",
                   metadata={"technical_details": str(exc)}
               )
           return ErrorClassification(
               severity=ErrorSeverity.CRITICAL,
               user_message=f"Error: {exc}",
               metadata={"technical_details": str(exc)}
           )

Error Classification Levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides sophisticated error handling:

- **CRITICAL**: End execution immediately
- **RETRIABLE**: Retry execution with same parameters
- **REPLANNING**: Create new execution plan
- **FATAL**: System-level failure requiring immediate termination

Always-Active Capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

Some capabilities are always included in execution:

.. code-block:: python

   # In registry configuration:
   CapabilityRegistration(
       name="respond",
       module_path="framework.infrastructure.respond_node",
       class_name="RespondCapability",
       always_active=True  # Always included in active capabilities
   )

Streaming Integration
=====================

Framework components use LangGraph's native streaming via the unified logger:

.. code-block:: python

   @capability_node
   class MyCapability(BaseCapability):
       async def execute(self) -> Dict[str, Any]:
           # Get unified logger with automatic streaming support
           logger = self.get_logger()

           logger.status("Processing data...")
           result = await process_data()
           logger.success("Processing complete")

           # Create context and use helper method to store
           result_context = ProcessedDataContext(data=result)
           return self.store_output_context(result_context)

Benefits
========

Reduced Boilerplate
~~~~~~~~~~~~~~~~~~~

**Configuration-driven approach** (Component: 5 lines + Registry: 8 lines):

.. code-block:: python

   # Component implementation (src/my_agent/capabilities/my_capability.py)
   @capability_node
   class MyCapability(BaseCapability):
       name = "my_capability"
       description = "What it does"
       provides = ["MY_DATA"]
       requires = []
       # Implementation handles infrastructure

   # Registry declaration (in src/my_agent/registry.py)
   extend_framework_registry(
       capabilities=[
           CapabilityRegistration(
               name="my_capability",
               module_path="my_agent.capabilities.my_capability",
               class_name="MyCapability",
               description="What it does",
               provides=["MY_DATA"],
               requires=[]
           )
       ]
   )

Consistency Guarantee
~~~~~~~~~~~~~~~~~~~~~

- All components have identical infrastructure integration via decorators
- Error handling follows same patterns across components
- State management is consistent through framework patterns
- Performance monitoring is standardized
- Registry declarations ensure complete metadata

Easy Testing
~~~~~~~~~~~~

.. code-block:: python

   # Test individual capability without framework overhead
   capability = MyCapability()
   result = await capability.execute(mock_state)

   # Test with full framework integration (requires registry declaration)
   @capability_node
   class TestCapability(BaseCapability):
       # Gets framework integration via decorator
       # Must still be declared in registry for framework use

Troubleshooting
===============

Common Issues
~~~~~~~~~~~~~

**Missing required attributes:**

.. code-block:: python

   # Problem: Missing required convention
   @capability_node
   class MyCapability(BaseCapability):
       # Missing 'name' attribute - will fail at decoration time
       description = "Does something"

   # Solution: Add required attributes
   @capability_node
   class MyCapability(BaseCapability):
       name = "my_capability"
       description = "Does something"

**Module path mismatch:**

.. code-block:: python

   # Problem: Module path doesn't match file location
   # File: src/my_agent/capabilities/my_capability.py
   CapabilityRegistration(
       module_path="my_agent.capabilities.missing",  # Wrong - file doesn't exist
       class_name="MyCapability"
   )

   # Solution: Use correct module path matching file structure
   # File: src/my_agent/capabilities/my_capability.py
   CapabilityRegistration(
       module_path="my_agent.capabilities.my_capability",  # Correct
       class_name="MyCapability"
   )

Development Utilities Integration
=================================

The framework's development utilities follow the same convention-over-configuration patterns, providing consistent interfaces that reduce boilerplate and integrate seamlessly with the configuration system.

.. _framework-logging-streaming:

Framework Logging and Streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides a unified logging system that automatically handles both CLI output and web UI streaming through a single, consistent API.

.. versionadded:: 0.9.4
   Unified logging with automatic streaming support via ``logger.status()``

.. admonition:: Evolving Interface Design
   :class: note

   The logging and streaming system is designed to grow with our interfaces (OpenWebUI, CLI, and future additions). Currently, "streaming" refers to real-time updates in OpenWebUI while logging appears in the CLI. As our interfaces mature, different interfaces will be able to subscribe to different message types and granularities based on their needs.

   We're actively refining the balance between detailed logging and high-level status updates. **Your feedback is valuable** - if you find the current message granularity doesn't match your interface needs, please share your use case.

**Capability Logging Pattern** (Recommended):

.. code-block:: python

   @capability_node
   class MyCapability(BaseCapability):
       name = "my_capability"
       description = "Example capability"

       async def execute(self) -> Dict[str, Any]:
           # Get unified logger - automatically uses capability name and state
           logger = self.get_logger()

           # High-level status updates - automatically stream to web UI
           logger.status("Processing data...")

           # Detailed information - CLI only by default
           logger.info(f"Retrieved {len(data)} records")
           logger.debug("Detailed trace information")

           # Explicit streaming when needed
           logger.info("Step 1 of 3 complete", stream=True, progress=0.33)

           # Errors and warnings automatically stream
           logger.warning("Retrying connection...")
           logger.error("Processing failed", exc_info=True)

           # Success messages automatically stream
           logger.success("Capability completed successfully")

           return self.store_output_context(result)

**Module-Level Logging Pattern** (Utilities, Tests):

.. code-block:: python

   from osprey.utils.logger import get_logger

   # Simple component name - no streaming (no state available)
   logger = get_logger("data_processor")

   logger.key_info("Starting data processing")
   logger.info("Processing user request")
   logger.debug("Detailed trace information")
   logger.warning("Configuration fallback used")
   logger.timing("Execution completed in 2.3 seconds")

**Streaming Behavior**:

The logger intelligently determines what to stream based on message type:

- **Always streams**: ``status()``, ``error()``, ``success()`` (default), ``warning()`` (default)
- **Never streams by default**: ``info()``, ``debug()``, ``key_info()``, ``timing()``
- **Override with** ``stream=True`` parameter to explicitly stream any message

**Configuration Integration**: Color schemes are automatically loaded from the configuration using the same paths as component registration:

.. code-block:: yaml

   # Framework component colors (in framework's internal config.yml)
   logging:
     framework:
       logging_colors:
         orchestrator: "cyan"
         task_extraction: "thistle1"

   # Application component colors (in your project's config.yml)
   logging:
     logging_colors:
       current_weather: "blue"
       data_analysis: "magenta"

**Automatic Step Tracking**: When streaming is enabled (via ``self.get_logger()`` in capabilities), the logger automatically includes execution context:

- **Task Preparation Phase**: Automatic step numbering for infrastructure components (1/3, 2/3, 3/3)
- **Execution Phase**: Dynamic step extraction from execution plan (e.g., "Step 2 of 5")
- **Custom Metadata**: Pass additional data via ``**kwargs`` (e.g., ``progress=0.5``, ``batch_num=2``)

**Graceful Degradation**: The logger works seamlessly in all contexts - when LangGraph streaming is unavailable (tests, utilities, CLI-only execution), streaming is automatically disabled while logging continues normally.


.. _performance-configuration-section:
Performance Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

The framework supports performance optimization through bypass configuration:

.. code-block:: yaml

   # Performance bypass settings (in main config.yml)
   execution_control:
     agent_control:
       task_extraction_bypass_enabled: false      # Skip LLM-based task extraction
       capability_selection_bypass_enabled: false # Skip LLM-based capability selection

Both settings default to ``false`` and can be overridden at runtime using :ref:`slash commands <slash-commands-section>` (``/task:off``, ``/caps:off``).


LLM Completion Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LLM completion interface integrates with the configuration system following the same provider configuration patterns:

.. code-block:: python

   from osprey.models import get_chat_completion

   # Configuration-driven completion (provider config auto-loaded)
   response = get_chat_completion(
       message="Explain quantum computing",
       provider="anthropic",
       model_id="claude-sonnet-4",
   )

   # With explicit provider config for development/testing
   response = get_chat_completion(
       message="Explain quantum computing",
       provider="anthropic",
       model_id="claude-sonnet-4",
       provider_config={"api_key": "explicit-key-for-testing"}
   )

**Provider Conventions**: All providers follow the same configuration structure with provider-specific requirements automatically validated:

.. code-block:: yaml

   # Provider configuration (in main config.yml)
   api:
     providers:
       cborg:
         api_key: "${CBORG_API_KEY}"
         base_url: "https://api.cborg.lbl.gov/v1"     # LBNL internal service
       stanford:
         api_key: "${STANFORD_API_KEY}"
         base_url: "https://aiapi-prod.stanford.edu/v1"  # Stanford AI Playground
       argo:
         api_key: "${ARGO_API_KEY}"
         base_url: "https://argo-bridge.cels.anl.gov"     # ANL Argo proxy
       anthropic:
         api_key: "${ANTHROPIC_API_KEY}"
         base_url: "https://api.anthropic.com"
       openai:
         api_key: "${OPENAI_API_KEY}"
         base_url: "https://api.openai.com/v1"
       ollama:
         base_url: "http://localhost:11434"     # Required for Ollama
         # No api_key needed for local models

.. dropdown:: Need Support for Additional Providers?
    :color: info
    :icon: people

    The framework's provider system is designed for extensibility. Many research institutions and national laboratories now operate their own AI/LM services similar to LBNL's CBorg system. We're happy to work with you to implement native support for your institution's internal AI services or other providers you need. Contact us to discuss integration requirements.

**Enterprise Integration**: HTTP proxy configuration follows environment variable conventions with automatic detection and validation.

Consistency Benefits
~~~~~~~~~~~~~~~~~~~~

Development utilities provide the same benefits as component registration:

- **Standardized Interfaces**: All utilities use the same source/component naming pattern
- **Configuration Integration**: Automatic loading from configuration system
- **Graceful Degradation**: Continue functioning when configuration is unavailable
- **Type Safety**: Full type hints and validation for development-time error detection
- **Performance Optimization**: Caching and lazy loading reduce overhead

.. seealso::

   :doc:`../../getting-started/hello-world-tutorial`
       Step-by-step tutorial showing registry configuration in a working weather agent

   :doc:`../../getting-started/control-assistant`
       Advanced patterns including framework capability exclusion and custom prompts

   :doc:`../05_production-systems/06_control-system-integration`
       Complete guide to control system connector patterns

   :doc:`../05_production-systems/02_data-source-integration`
       Data source provider implementation patterns

   :doc:`../../api_reference/01_core_framework/03_registry_system`
       API reference for registry management and component discovery

   :doc:`../03_core-framework-systems/03_registry-and-discovery`
       Registry patterns and component registration workflows

   :doc:`../02_quick-start-patterns/01_building-your-first-capability`
       Hands-on guide to implementing components with decorators