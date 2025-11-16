=============================================
Part 4: Customization & Extension
=============================================

**You've deployed your assistant - now make it yours!**

Part 3 got your control assistant running in production with real hardware. Part 4 shows you how to customize it for your facility's specific needs and extend it with advanced features.

**What You'll Learn:**

- Add facility-specific domain knowledge and terminology
- Configure models for optimal cost/performance
- Customize the CLI appearance
- Use advanced debugging and optimization features
- Build custom capabilities for facility-specific operations

Step 10: Prompt Customization
=============================

The template works out of the box, but customizing prompts with facility-specific knowledge dramatically improves accuracy, relevance, and user trust. OSPREY provides two levels of prompt customization: **service-level prompts** (like channel finder) and **framework-level prompts** (orchestrator, classification, response generation).

Channel Finder Prompt Customization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An important customization you can make is adding facility-specific context to the channel finder prompts. This was covered in detail in :ref:`Part 2: Building Your Channel Finder <channel-finder-benchmarking>`, where you learned how to customize the ``facility_description`` variable to include:

- Physical system descriptions and hierarchy
- Naming conventions and operational terminology
- Disambiguation rules for ambiguous queries

See :doc:`control-assistant-part2-channel-finder` for comprehensive channel finder customization guidance.

.. _part4-framework-prompt-customization:

Framework Prompt Customization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework prompt provider system allows you to customize how the agent thinks, plans, and responds—unlike service-level prompts (specific to one service), framework prompts control core agent behaviors across all capabilities. See :doc:`../developer-guides/03_core-framework-systems/04_prompt-customization` for comprehensive prompt customization patterns and advanced techniques.

**What Framework Prompts Control:**

- **Orchestrator**: How the agent creates execution plans and sequences capabilities
- **Classifier**: How the agent decides which capabilities to invoke for a given task
- **Task Extraction**: How conversations are compressed into concrete tasks
- **Response Generation**: How final answers are formatted and presented to users

**The Prompt Provider Architecture**

OSPREY uses a **provider registration system** that allows you to override default framework prompts without modifying framework code. Each prompt type has a builder class that you can subclass and customize.

**Step 1: Create Your Custom Prompt Builder**

Create a new module in your agent project (e.g., ``src/my_control_assistant/framework_prompts/``) and subclass the appropriate default builder:

.. code-block:: python

   # src/my_control_assistant/framework_prompts/orchestrator.py
   import textwrap
   from osprey.prompts.defaults.orchestrator import DefaultOrchestratorPromptBuilder
   from osprey.registry import get_registry

   class MyFacilityOrchestratorPromptBuilder(DefaultOrchestratorPromptBuilder):
       """Facility-specific orchestrator prompt customization."""

       def get_role_definition(self) -> str:
           """Override the agent's role description."""
           return "You are an expert execution planner for the MyFacility control system assistant."

       def get_instructions(self) -> str:
           """Extend base instructions with facility-specific guidance."""
           registry = get_registry()
           base_instructions = super().get_instructions()

           facility_guidance = textwrap.dedent("""
               MyFacility-Specific Planning Rules:

               1. SAFETY PRIORITIES:
                  - Always verify beam status before executing magnet changes
                  - For vacuum operations, check interlocks before valve commands
                  - Never plan writes to critical systems without explicit user confirmation

               2. STANDARD WORKFLOWS:
                  - Beam current queries: Use MAIN_DCCT (not backup DCCTs unless specified)
                  - Magnet tuning: Always read current values before planning changes
                  - Vacuum readbacks: Prefer ION-PUMP channels over GAUGE channels for routine monitoring

               3. OPERATIONAL CONTEXT:
                  - Morning startup procedures require sequential system checks
                  - Magnet ramping needs 2-second settling time between steps
                  - RF cavity adjustments affect beam stability—plan conservatively

               Focus on being practical and efficient while ensuring robust execution.
               Never plan for simulated or fictional data - only real MyFacility operations.
           """).strip()

           return f"{base_instructions}\n\n{facility_guidance}"

**Step 2: Create the Module __init__.py**

Create an ``__init__.py`` file in your framework_prompts module to export your builders:

.. code-block:: python

   # src/my_control_assistant/framework_prompts/__init__.py
   from .orchestrator import MyFacilityOrchestratorPromptBuilder
   # Add other builders as you create them

   __all__ = [
       "MyFacilityOrchestratorPromptBuilder",
   ]

**Step 3: Register Your Custom Prompt Provider**

In your agent's ``registry.py``, extend the existing registry configuration to include your custom prompt builders. This builds on the ``RegistryConfigProvider`` pattern you already have from Parts 1-3:

.. code-block:: python

   # src/my_control_assistant/registry.py
   from osprey.registry import (
       RegistryConfigProvider,
       extend_framework_registry,
       CapabilityRegistration,
       ContextClassRegistration,
       FrameworkPromptProviderRegistration,
       RegistryConfig
   )


   class MyControlAssistantRegistryProvider(RegistryConfigProvider):
       """Registry provider for My Control Assistant."""

       def get_registry_config(self) -> RegistryConfig:
           """Return registry configuration with custom framework prompts."""
           return extend_framework_registry(
               # Your existing capabilities
               capabilities=[
                   CapabilityRegistration(
                       name="channel_finding",
                       module_path="my_control_assistant.capabilities.channel_finding",
                       class_name="ChannelFindingCapability",
                       description="Find control system channels using semantic search",
                       provides=["CHANNEL_ADDRESSES"],
                       requires=[]
                   ),
                   # ... other capabilities ...
               ],

               # Your existing context classes
               context_classes=[
                   ContextClassRegistration(
                       context_type="CHANNEL_ADDRESSES",
                       module_path="my_control_assistant.context_classes",
                       class_name="ChannelAddressesContext"
                   ),
                   # ... other context classes ...
               ],

               # Add custom framework prompts
               framework_prompt_providers=[
                   FrameworkPromptProviderRegistration(
                       module_path="my_control_assistant.framework_prompts",
                       prompt_builders={
                           "orchestrator": "MyFacilityOrchestratorPromptBuilder",
                           # Add other builders as needed:
                           # "task_extraction": "MyFacilityTaskExtractionPromptBuilder",
                           # "response_generation": "MyFacilityResponseGenerationPromptBuilder",
                           # "classification": "MyFacilityClassificationPromptBuilder",
                           # "error_analysis": "MyFacilityErrorAnalysisPromptBuilder",
                           # "clarification": "MyFacilityClarificationPromptBuilder",
                           # "memory_extraction": "MyFacilityMemoryExtractionPromptBuilder",
                       }
                   )
               ]
           )

The framework automatically discovers and uses your custom builders. You can override as many or as few prompt types as needed—any not specified will use the framework defaults.

**Available Prompt Builder Types:**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Builder Type
     - Base Class
     - Purpose
   * - ``orchestrator``
     - ``DefaultOrchestratorPromptBuilder``
     - Controls execution planning and capability sequencing
   * - ``task_extraction``
     - ``DefaultTaskExtractionPromptBuilder``
     - Extracts actionable tasks from conversations
   * - ``response_generation``
     - ``DefaultResponseGenerationPromptBuilder``
     - Formats final responses to users
   * - ``classification``
     - ``DefaultClassificationPromptBuilder``
     - Determines which capabilities match user tasks
   * - ``error_analysis``
     - ``DefaultErrorAnalysisPromptBuilder``
     - Generates explanations for execution errors
   * - ``clarification``
     - ``DefaultClarificationPromptBuilder``
     - Creates targeted questions for ambiguous queries
   * - ``memory_extraction``
     - ``DefaultMemoryExtractionPromptBuilder``
     - Extracts and stores user preferences and context

**Step 4: Test and Debug Your Custom Prompts**

Run your agent and verify the custom prompts are being used. The framework includes prompt debugging tools to inspect the actual prompts sent to the LLM:

.. code-block:: yaml

   # config.yml
   development:
     # Prompt debugging configuration
     prompts:
       show_all: false      # Print prompts to console (verbose, useful for live debugging)
       print_all: true      # Save prompts to files in _agent_data/prompts/
       latest_only: false   # Keep all versions to compare prompt changes (set to true to keep only latest)

With ``print_all: true``, prompts are saved to ``_agent_data/prompts/`` with filenames like:

- ``orchestrator_latest.md`` - Current orchestrator prompt
- ``task_extraction_latest.md`` - Current task extraction prompt
- ``response_generation_latest.md`` - Current response generation prompt

Set ``latest_only: false`` to preserve multiple versions (timestamped) when iterating on prompt changes, making it easy to compare different prompt versions and track what changed.

**Advanced Customization Patterns**

.. dropdown:: Example: Custom Classification Prompt
   :color: info

   Customize how the agent classifies tasks:

   .. code-block:: python

      # src/my_control_assistant/framework_prompts/classification.py
      import textwrap
      from osprey.prompts.defaults.classification import DefaultClassificationPromptBuilder

      class MyFacilityClassificationPromptBuilder(DefaultClassificationPromptBuilder):
          """Custom classification for facility-specific task routing."""

          def get_role_definition(self) -> str:
              return "You are an expert task classification assistant for MyFacility."

          def get_instructions(self) -> str:
              return textwrap.dedent("""
                  Based on the instructions and examples, you must output a JSON object
                  with a key "is_match": A boolean (true or false) indicating if the
                  user's request matches the capability.

                  Consider MyFacility-specific terminology and synonyms when matching.

                  Respond ONLY with the JSON object. Do not provide any explanation.
              """).strip()

.. dropdown:: Example: Custom Response Generation Prompt
   :color: info

   Customize response formatting to match facility communication standards:

   .. code-block:: python

      # src/my_control_assistant/framework_prompts/response_generation.py
      from typing import Optional
      from osprey.prompts.defaults.response_generation import DefaultResponseGenerationPromptBuilder

      class MyFacilityResponseGenerationPromptBuilder(DefaultResponseGenerationPromptBuilder):
          """Custom response formatting for facility standards."""

          def get_role_definition(self) -> str:
              return "You are an expert assistant for the MyFacility accelerator."

          def _get_conversational_guidelines(self) -> list[str]:
              """Override conversational guidelines with facility-specific standards."""
              return [
                  "Be professional and focused on MyFacility operations",
                  "Always include units in parentheses (e.g., 'current is 500.2 mA')",
                  "Mention system mode when relevant (e.g., 'Storage ring in USER mode')",
                  "Highlight any out-of-range or alarm conditions prominently",
                  "Provide helpful context about accelerator physics when relevant"
              ]

.. dropdown:: Example: Custom Task Extraction Prompt
   :color: info

   Customize how conversations are extracted into actionable tasks:

   .. code-block:: python

      # src/my_control_assistant/framework_prompts/task_extraction.py
      import textwrap
      from osprey.prompts.defaults.task_extraction import DefaultTaskExtractionPromptBuilder

      class MyFacilityTaskExtractionPromptBuilder(DefaultTaskExtractionPromptBuilder):
          """Custom task extraction for MyFacility operations."""

          def get_role_definition(self) -> str:
              return "You are a MyFacility control system task extraction specialist."

          def get_instructions(self) -> str:
              return textwrap.dedent("""
                  Extract clear, actionable tasks related to MyFacility control systems.

                  Guidelines:
                  - Create self-contained task descriptions executable without conversation context
                  - Resolve temporal references to specific times using facility timestamps
                  - Extract specific measurements, device names, and parameters from previous responses
                  - Understand MyFacility device naming conventions
                  - Set depends_on_chat_history=True if task references previous messages
                  - Be specific and actionable using MyFacility terminology
              """).strip()

Step 11: System Configuration
==============================

The framework uses **8 specialized models** for different roles. You can optimize each for performance, cost, or latency.

Model Configuration
^^^^^^^^^^^^^^^^^^^

**Configuration:** ``config.yml``

.. code-block:: yaml

   models:
     orchestrator:              # Plans execution (most critical)
       provider: cborg
       model_id: anthropic/claude-haiku
       max_tokens: 4096

     response:                  # Generates final responses
       provider: cborg
       model_id: anthropic/claude-haiku

     classifier:                # Selects capabilities (fast, simple)
       provider: ollama
       model_id: mistral:7b

     approval:                  # Analyzes code for safety
       provider: cborg
       model_id: anthropic/claude-haiku

     task_extraction:           # Compresses conversations
       provider: cborg
       model_id: anthropic/claude-haiku

     memory:                    # Memory extraction
       provider: cborg
       model_id: google/gemini-flash

     python_code_generator:     # Generates analysis code
       provider: cborg
       model_id: anthropic/claude-sonnet  # Use stronger model here

     time_parsing:              # Parses time expressions
       provider: cborg
       model_id: anthropic/claude-haiku

**Optimization Strategy:**

1. **Start with Haiku everywhere** - Reliable baseline with good cost/performance
2. **Identify bottlenecks** - Watch for poor quality outputs in specific areas
3. **Upgrade selectively** - Use Sonnet for ``python_code_generator`` if code quality matters
4. **Consider cost** - Haiku is ~10x cheaper than Sonnet for similar tasks
5. **Consider local models** - Ollama for small tasks like classification can increase speed when GPU is available

.. important::
   **Structured Output Support Variability**

   The framework relies extensively on `structured outputs <https://platform.openai.com/docs/guides/structured-outputs>`_ to make LLM responses predictable
   in downstream pipelines. This ensures that the output from one model can be reliably consumed
   by subsequent components and models in the system. Model support for structured outputs varies
   significantly and is not always well-documented. What works today may change with the next
   model release, particularly from providers like Ollama.

   **Recommended Models with Reliable Structured Output Support:**

   - **Claude Haiku/Sonnet** - Excellent structured output support, well-tested
   - **Mistral 7B (via Ollama)** - Cost-effective and reliable for classification tasks
   - **Test before deploying** - Always validate structured output quality with your specific use cases

   If you experience issues with function calling or malformed outputs, the model's structured
   output support is the first thing to investigate.

**Application-Specific Models:**

You can also define custom model roles for your own capabilities:

.. code-block:: yaml

   models:
     # Framework models (above)...

     # Your custom capability models
     machine_operations:
       provider: cborg
       model_id: anthropic/claude-sonnet  # Higher stakes = stronger model

     data_visualization:
       provider: cborg
       model_id: anthropic/claude-haiku   # Simpler task = lighter model

CLI Theme Customization
^^^^^^^^^^^^^^^^^^^^^^^

Customize the command-line interface appearance for your facility branding:

**Configuration:** ``config.yml``

.. code-block:: yaml

   cli:
     theme: "custom"     # Options: default, vulcan, custom

     # Custom theme colors (only used when theme: custom)
     custom_theme:
       primary: "#1E90FF"      # Brand color
       success: "#32CD32"      # Success messages
       accent: "#FF6347"       # Interactive elements
       command: "#9370DB"      # Shell commands
       path: "#20B2AA"         # File paths
       info: "#4682B4"         # Info messages

     # Optional: Custom ASCII banner
     banner: |
       ╔═══════════════════════════════════════╗
       ║   MyFacility Control Assistant        ║
       ║   Version 1.0.0                       ║
       ╚═══════════════════════════════════════╝

**Built-in Themes:**

- ``default`` / ``vulcan``: Purple-teal theme (both are identical)
- ``custom``: Define your own facility colors using the ``custom_theme`` section above

.. admonition:: Collaboration Welcome
   :class: outreach

   We welcome contributions of new built-in themes! If you've designed a theme for your facility that you'd like to share with the community, please open a GitHub issue. We're happy to include additional themes that showcase different color palettes and facility branding styles.

Step 12: Advanced Features
===========================

For experienced users, the framework provides several advanced features for optimization and debugging.

Slash Commands
^^^^^^^^^^^^^^

The framework supports runtime commands prefixed with ``/`` for dynamic control (see implementation details in the command processing code).

**Planning and Execution:**

.. code-block:: text

   /planning          # Enable planning mode for current query
   /planning:on       # Enable planning mode (same as /planning)
   /planning:off      # Disable planning mode

**Performance Optimization:**

.. code-block:: text

   /task:off          # Bypass task extraction (use full chat history)
   /task:on           # Enable task extraction (default)

   /caps:off          # Bypass capability selection (activate all capabilities)
   /caps:on           # Enable capability selection (default)

.. _bypass-task-extraction-section:

.. tab-set::

   .. tab-item:: Task Extraction Bypass

      The ``/task:off`` command skips LLM-based task extraction and passes the full conversation history directly to downstream processing.

      **When to Use:**

      - R&D scenarios where full conversational context aids development
      - Short conversations where task extraction overhead exceeds benefits
      - Debugging to see how orchestrator handles raw conversation history

      **Trade-offs:**

      - ✅ Faster preprocessing (skips one LLM call)
      - ✅ Preserves all conversational nuance
      - ⚠️ Slower orchestration (more tokens to process)
      - ⚠️ Potential information overload in long conversations

      **Example Usage:**

      .. code-block:: text

         # Skip task extraction for complex conversations
         You: /task:off continue with previous analysis

      **Further Reading:** :ref:`Task Extraction Bypass <bypass-task-extraction-section>`

   .. tab-item:: Capability Selection Bypass
      :name: bypass-capability-selection-section

      The ``/caps:off`` command skips capability classification and activates all registered capabilities.

      **When to Use:**

      - Debugging when unsure which capabilities should be active
      - Ensuring all capabilities are considered in execution planning
      - Testing orchestrator behavior with full capability access

      **Trade-offs:**

      - ✅ Faster preprocessing (skips parallel classification LLM calls)
      - ✅ Ensures no capability is missed
      - ⚠️ Longer orchestration prompts (more capability descriptions)
      - ⚠️ May activate unnecessary capabilities

      **Example Usage:**

      .. code-block:: text

         # Force all capabilities active for debugging
         You: /caps:off find channels for beam position

**Configuration:**

You can also set bypass modes as system defaults in ``config.yml``:

.. code-block:: yaml

   agent_control:
     task_extraction_bypass_enabled: false  # default
     capability_selection_bypass_enabled: false  # default

.. admonition:: Collaboration Welcome
   :class: outreach

   We welcome contributions of new slash commands that improve workflow efficiency! If you've implemented custom slash commands that would benefit the community (e.g., session management, debugging helpers, or facility-specific controls), please open a GitHub issue. We're happy to consider adding useful slash commands to the framework.

Step 13: Extending Framework Capabilities
=========================================

Sometimes a framework capability is too generic for your needs. A real-world example: the ALS Assistant replaced the generic Python capability with specialized capabilities for different task types.

**Why Replace a Framework Capability?**

The framework includes a generic ``python`` capability for running arbitrary Python code. While functional, using one capability for everything creates problems:

- **Data analysis** needs structured result templates and multi-phase planning
- **Data visualization** needs different prompts and figure management
- **Machine operations** needs safety checks and approval workflows

One Python capability trying to handle all three becomes a complicated mess. Better to split into specialized capabilities, each doing one thing well.

**Real-World Example: Data Analysis Capability**

At ALS, we created a specialized ``data_analysis`` capability that:

1. Replaces the generic Python capability (via ``exclude_capabilities=["python"]``)
2. Generates structured analysis plans using LLMs
3. Prepares task-specific prompts for the Python executor service
4. Returns standardized ``ANALYSIS_RESULTS`` context

.. dropdown:: Simplified Data Analysis Capability
   :color: info

   **Step 1:** Create the specialized capability:

   .. code-block:: python

      # src/my_control_assistant/capabilities/data_analysis.py
      from osprey.base import BaseCapability, capability_node
      from osprey.state import AgentState, StateManager
      from osprey.registry import get_registry
      from osprey.services.python_executor import PythonExecutionRequest

      registry = get_registry()

      @capability_node
      class DataAnalysisCapability(BaseCapability):
          """Specialized data analysis for control system data."""

          name = "data_analysis"
          description = "Analyze accelerator control system data with domain-specific prompts"
          provides = ["ANALYSIS_RESULTS"]
          requires = []  # Flexible - works with any available context

          @staticmethod
          async def execute(state: AgentState, **kwargs):
              """Execute data analysis with specialized prompts."""
              step = StateManager.get_current_step(state)

              # 1. Generate analysis plan (simplified)
              analysis_plan = await create_analysis_plan(
                  task_objective=step.get('task_objective'),
                  state=state
              )

              # 2. Create domain-specific prompts
              prompts = [
                  f"**ANALYSIS PLAN:** {format_plan(analysis_plan)}",
                  f"**EXPECTED OUTPUT:** {create_results_template(analysis_plan)}",
                  "**DOMAIN CONTEXT:** ALS accelerator physics analysis..."
              ]

              # 3. Call Python executor with specialized prompts
              python_service = registry.get_service("python_executor")
              request = PythonExecutionRequest(
                  task_objective=step.get('task_objective'),
                  capability_prompts=prompts,  # Domain-specific guidance
                  execution_folder_name="data_analysis",
                  capability_context_data=state.get('capability_context_data', {})
              )

              result = await python_service.ainvoke(request, config=kwargs.get("config"))

              # 4. Store results in standardized context
              context_updates = StateManager.store_context(
                  state,
                  registry.context_types.ANALYSIS_RESULTS,
                  step.get("context_key"),
                  result.execution_result
              )

              return context_updates

          def _create_classifier_guide(self):
              """Teach the classifier when to use this capability."""
              return TaskClassifierGuide(
                  instructions="Classify as True for data analysis requests",
                  examples=[
                      ClassifierExample(
                          query="Analyze the beam lifetime trends from yesterday",
                          result=True,
                          reason="Requires analysis of historical data"
                      ),
                      ClassifierExample(
                          query="Show me a plot of beam current",
                          result=False,
                          reason="Visualization request, not analysis"
                      )
                  ]
              )

          def _create_orchestrator_guide(self):
              """Teach the orchestrator how to plan with this capability."""
              return OrchestratorGuide(
                  instructions=f"""
                  Use data_analysis for numerical analysis of control system data.

                  **Input Requirements:**
                  - Works with any available context data
                  - Specify inputs via context keys

                  **Output:** {registry.context_types.ANALYSIS_RESULTS}
                  - Structured analysis results
                  - Available to downstream steps
                  """,
                  examples=[
                      OrchestratorExample(
                          step=PlannedStep(
                              context_key="trend_analysis",
                              capability="data_analysis",
                              task_objective="Analyze beam current trends and identify anomalies",
                              expected_output=registry.context_types.ANALYSIS_RESULTS,
                              inputs=[{"ARCHIVER_DATA": "historical_beam_data"}]
                          ),
                          scenario_description="Trend analysis of time-series data"
                      )
                  ]
              )

   **Step 2:** Register and exclude the framework capability:

   .. code-block:: python

      # In registry.py
      from osprey.registry import (
          RegistryConfigProvider,
          extend_framework_registry,
          CapabilityRegistration,
          RegistryConfig
      )

      class MyControlAssistantRegistryProvider(RegistryConfigProvider):
          """Registry provider for My Control Assistant."""

          def get_registry_config(self) -> RegistryConfig:
              """Return registry configuration with specialized capabilities."""
              return extend_framework_registry(
                  capabilities=[
                      CapabilityRegistration(
                          name="data_analysis",
                          module_path="my_control_assistant.capabilities.data_analysis",
                          class_name="DataAnalysisCapability",
                          description="Domain-specific data analysis for control system data",
                          provides=["ANALYSIS_RESULTS"],
                          requires=[]
                      ),
                      # ... other custom capabilities
                  ],

                  # Exclude the framework's generic Python capability
                  exclude_capabilities=["python"]  # Use specialized data_analysis instead
              )

   **Step 3:** The framework automatically:

   - Uses your specialized capability instead of the generic one
   - Includes it in classification and orchestration
   - Handles all state management and error recovery
   - Provides your domain-specific prompts to the Python executor


Navigation
==========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: ← Part 3: Integration & Deployment
      :link: control-assistant-part3-production
      :link-type: doc

      Return to integration guide

   .. grid-item-card:: Tutorial Home
      :link: control-assistant
      :link-type: doc

      Back to tutorial overview
