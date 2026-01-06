====================
Prompt Customization
====================

This guide covers customizing framework prompts for domain-specific applications. The framework's prompt management system enables clean separation between generic framework functionality and domain-specific prompt customization through sophisticated dependency injection patterns.

Architecture Overview
=====================

The prompt system uses a provider architecture where applications register custom prompt implementations that the framework components request through dependency injection. This enables applications to provide domain-specific prompts while the framework remains generic.

Applications can override any prompt builder with domain-specific implementations while maintaining full compatibility with all framework components.

**Key Benefits:**

- **Domain Agnostic**: Framework remains generic while supporting specialized prompts
- **No Circular Dependencies**: Clean separation through dependency injection
- **Flexible Composition**: Modular prompt building with optional components
- **Development Support**: Integrated debugging and prompt inspection tools

Quick Start: Custom Prompt Provider
===================================

Here's a minimal example of creating a custom prompt provider:

.. code-block:: python

   from osprey.prompts import FrameworkPromptBuilder, FrameworkPromptProvider
   from osprey.prompts.defaults import DefaultPromptProvider

   class MyDomainPromptBuilder(FrameworkPromptBuilder):
       def get_role_definition(self) -> str:
           return "You are a domain-specific expert system."

       def get_instructions(self) -> str:
           return "Provide analysis using domain-specific terminology."

   class MyAppPromptProvider(FrameworkPromptProvider):
       def __init__(self):
           # Use custom builders for key prompts
           self._orchestrator = MyDomainPromptBuilder()

           # Use framework defaults for others
           self._defaults = DefaultPromptProvider()

       def get_orchestrator_prompt_builder(self):
           return self._orchestrator

       def get_classification_prompt_builder(self):
           # Delegate to framework default
           return self._defaults.get_classification_prompt_builder()

       # ... implement other required methods

Development and Debugging
=========================

All prompts automatically integrate with the framework's debug system for development visibility. The system provides comprehensive debugging capabilities through both console output and file persistence, making it invaluable for prompt development, troubleshooting, and optimization.

Configuration Options
---------------------

The debug system is controlled through the ``development.prompts`` configuration section in your ``config.yml`` file:

.. code-block:: yaml

   development:
     prompts:
       # Console output with detailed formatting and separators
       show_all: true

       # File output to prompts directory for inspection
       print_all: true

       # File naming strategy
       latest_only: false  # true: latest.md files, false: timestamped files

Console Output
--------------

When ``show_all: true`` is set, all generated prompts are displayed in the console with clear visual separators and metadata:

.. code-block:: text

   ================================================================================
   🔍 DEBUG PROMPT: orchestrator_system (DefaultOrchestratorPromptBuilder)
   ================================================================================
   You are an intelligent orchestration agent for the ALS Assistant system...
   ================================================================================

File Persistence
----------------

When ``print_all: true`` is enabled, prompts are automatically saved to the configured ``prompts_dir`` with rich metadata headers:

- **Timestamped files** (``latest_only: false``): Each prompt generation creates a new file with timestamp

  - Format: ``{name}_{YYYYMMDD_HHMMSS}.md``
  - Use case: Track prompt evolution over time, compare versions, debug prompt changes
  - Example: ``orchestrator_system_20241215_143022.md``

- **Latest files** (``latest_only: true``): Overwrites the previous version, keeping only current state

  - Format: ``{name}_latest.md``
  - Use case: Always see current prompt without file clutter
  - Example: ``orchestrator_system_latest.md``

Metadata Headers
----------------

All saved prompt files include comprehensive metadata for traceability:

.. code-block:: markdown

   # PROMPT METADATA
   # Generated: 2024-12-15 14:30:22
   # Name: orchestrator_system
   # Builder: DefaultOrchestratorPromptBuilder
   # File: /path/to/prompts/orchestrator_system_latest.md
   # Latest Only: true

API Call Logging
================

Complementing prompt debugging, the framework can log complete LLM API interactions including raw inputs, outputs, and caller metadata. This provides full transparency into what's sent to and received from AI providers.

Configuration
-------------

Enable API call logging in your ``config.yml``:

.. code-block:: yaml

   development:
     # LLM API call logging
     api_calls:
       save_all: true             # Enable logging of all API calls
       latest_only: true          # Keep only latest call per function
       include_stack_trace: false # Include full Python stack trace

Log File Organization
---------------------

API calls are by default saved to ``_agent_data/api_calls/`` with descriptive filenames based on the calling function:

.. code-block:: text

   _agent_data/api_calls/
   ├── task_extraction_node__extract_task_latest.txt
   ├── classification_node_CapabilityClassifier__perform_classification_python_latest.txt
   ├── classification_node_CapabilityClassifier__perform_classification_memory_latest.txt
   ├── orchestration_node_OrchestrationNode__create_execution_plan_latest.txt
   └── generator_node_LLMCodeGenerator_generate_code_latest.txt

Each log file contains:

- **Caller metadata**: Function, module, class, file path, line number
- **Model configuration**: Provider, model ID, tokens, temperature
- **Complete input**: Full message sent to LLM including all context
- **Complete output**: Raw response from LLM

.. note::
   - **Prompts directory** contains curated prompt templates
   - **API calls directory** contains complete API request/response pairs with full context
   - Use both together for comprehensive debugging of LLM interactions

Provider Interface Implementation
=================================

Applications implement the FrameworkPromptProvider interface to provide domain-specific prompts to framework infrastructure. All methods are required and must return FrameworkPromptBuilder instances.

.. note::
   Applications typically inherit from DefaultPromptProvider and override only the prompt builders they want to customize, using framework defaults for the rest.

Complete Provider Interface
---------------------------

.. tab-set::
   :class: natural-width

   .. tab-item:: Orchestrator

      Controls execution planning and coordination:

      .. code-block:: python

         def get_orchestrator_prompt_builder(self) -> FrameworkPromptBuilder:
             """
             Return prompt builder for orchestration operations.

             Used by the orchestrator node to create execution plans
             and coordinate capability execution sequences.
             """

   .. tab-item:: Task Extraction

      Handles task parsing and structuring:

      .. code-block:: python

         def get_task_extraction_prompt_builder(self) -> FrameworkPromptBuilder:
             """
             Return prompt builder for task extraction operations.

             Used by task extraction node to parse user requests
             into structured, actionable tasks.
             """

   .. tab-item:: Classification

      Manages request classification and routing:

      .. code-block:: python

         def get_classification_prompt_builder(self) -> FrameworkPromptBuilder:
             """
             Return prompt builder for classification operations.

             Used by classification node to determine which capabilities
             should handle specific user requests.
             """

   .. tab-item:: Response Generation

      Controls final response formatting:

      .. code-block:: python

         def get_response_generation_prompt_builder(self) -> FrameworkPromptBuilder:
             """
             Return prompt builder for response generation.

             Used by response generation to format final answers
             using capability results and conversation context.
             """

   .. tab-item:: Error Analysis

      Handles error classification and recovery:

      .. code-block:: python

         def get_error_analysis_prompt_builder(self) -> FrameworkPromptBuilder:
             """
             Return prompt builder for error analysis operations.

             Used by error handling system to classify errors
             and determine recovery strategies.
             """

   .. tab-item:: Clarification

      Manages clarification requests:

      .. code-block:: python

         def get_clarification_prompt_builder(self) -> FrameworkPromptBuilder:
             """
             Return prompt builder for clarification requests.

             Used when the system needs additional information
             from users to complete tasks.
             """

   .. tab-item:: Memory Extraction

      Controls memory operations:

      .. code-block:: python

         def get_memory_extraction_prompt_builder(self) -> FrameworkPromptBuilder:
             """
             Return prompt builder for memory extraction operations.

             Used by memory capability to extract and store
             relevant information from conversations.
             """

   .. tab-item:: Time Range Parsing

      Handles temporal query parsing:

      .. code-block:: python

         def get_time_range_parsing_prompt_builder(self) -> FrameworkPromptBuilder:
             """
             Return prompt builder for time range parsing.

             Used by time parsing capability to understand
             temporal references in user queries.
             """

   .. tab-item:: Python

      Controls code generation and execution:

      .. code-block:: python

         def get_python_prompt_builder(self) -> FrameworkPromptBuilder:
             """
             Return prompt builder for Python operations.

             Used by Python capability for code generation,
             analysis, and execution guidance.
             """

Default Builder Reference
=========================

The framework provides individual default prompt builder implementations organized by framework node. Each node has its own specialized prompt builder that applications can use directly or extend.

.. dropdown:: View Default Implementation Examples
   :animate: fade-in-slide-down

   .. tab-set::
      :class: natural-width

      .. tab-item:: Orchestrator

         .. literalinclude:: ../../../../src/osprey/prompts/defaults/orchestrator.py
            :language: python

      .. tab-item:: Task Extraction

         .. literalinclude:: ../../../../src/osprey/prompts/defaults/task_extraction.py
            :language: python

      .. tab-item:: Classification

         .. literalinclude:: ../../../../src/osprey/prompts/defaults/classification.py
            :language: python

      .. tab-item:: Response Generation

         .. literalinclude:: ../../../../src/osprey/prompts/defaults/response_generation.py
            :language: python

      .. tab-item:: Error Analysis

         .. literalinclude:: ../../../../src/osprey/prompts/defaults/error_analysis.py
            :language: python

      .. tab-item:: Clarification

         .. literalinclude:: ../../../../src/osprey/prompts/defaults/clarification.py
            :language: python

      .. tab-item:: Memory Extraction

         .. literalinclude:: ../../../../src/osprey/prompts/defaults/memory_extraction.py
            :language: python

      .. tab-item:: Time Range Parsing

         .. literalinclude:: ../../../../src/osprey/prompts/defaults/time_range_parsing.py
            :language: python

      .. tab-item:: Python

         .. literalinclude:: ../../../../src/osprey/prompts/defaults/python.py
            :language: python

Registration Patterns
=====================

Applications register their prompt providers during initialization using the registry system:

Basic Registration
------------------

.. code-block:: python

   from osprey.prompts.loader import register_framework_prompt_provider
   from applications.myapp.framework_prompts import MyAppPromptProvider

   # During application initialization
   register_framework_prompt_provider("myapp", MyAppPromptProvider())

Registry-Based Registration
---------------------------

For automatic discovery, include prompt providers in your application registry:

.. code-block:: python

   # In applications/myapp/registry.py
   from osprey.registry import RegistryConfig, FrameworkPromptProviderRegistration

   class MyAppRegistryProvider(RegistryConfigProvider):
       def get_registry_config(self) -> RegistryConfig:
           return RegistryConfig(
               # ... other registrations
               framework_prompt_providers=[
                   FrameworkPromptProviderRegistration(
                       module_path="applications.myapp.framework_prompts",
                       prompt_builders={
                           "orchestrator": "MyOrchestratorPromptBuilder",
                           "classification": "MyClassificationPromptBuilder"
                           # Others use framework defaults
                       }
                   )
               ]
           )

Advanced Patterns
=================

Multi-Application Deployments
-----------------------------

For deployments with multiple applications, you can access specific providers:

.. code-block:: python

   from osprey.prompts import get_framework_prompts

   # Access specific application's prompts
   als_provider = get_framework_prompts("als_assistant")
   assistant_provider = get_framework_prompts("control_assistant")

   # Use default provider (first registered)
   default_provider = get_framework_prompts()

Selective Override Pattern
--------------------------

Override only specific builders while inheriting others:

.. code-block:: python

   from osprey.prompts.defaults import DefaultPromptProvider

   class MyAppPromptProvider(DefaultPromptProvider):
       def __init__(self):
           super().__init__()
           # Override specific builders
           self._custom_orchestrator = MyOrchestratorPromptBuilder()

       def get_orchestrator_prompt_builder(self):
           return self._custom_orchestrator

       # All other methods inherited from DefaultPromptProvider

Testing Strategies
------------------

Test your custom prompts in isolation:

.. code-block:: python

   def test_custom_orchestrator_prompt():
       builder = MyOrchestratorPromptBuilder()

       # Test role definition
       role = builder.get_role_definition()
       assert "domain-specific" in role.lower()

       # Test full prompt generation
       system_prompt = builder.get_system_instructions(
           capabilities=["test_capability"],
           context_manager=mock_context
       )
       assert len(system_prompt) > 0

.. seealso::

   :doc:`../../api_reference/01_core_framework/05_prompt_management`
       API reference for prompt system classes and functions

   :doc:`03_registry-and-discovery`
       Component registration and discovery patterns

   :doc:`../01_understanding-the-framework/02_convention-over-configuration`
       Framework conventions and patterns