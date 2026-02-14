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

.. _part4-channel-finder-prompts:

Channel Finder Prompt Customization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The channel finder uses facility-specific prompts to dramatically improve semantic matching accuracy. Each pipeline (in_context, hierarchical, middle_layer) has its own prompts directory with a clear separation of concerns:

**Prompt File Structure:**

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - File
     - Purpose
     - Edit Required?
   * - ``in_context.py``
     - Prompt builder for in-context pipeline (facility description, matching rules)
     - **REQUIRED** (if using in-context pipeline)
   * - ``hierarchical.py``
     - Prompt builder for hierarchical pipeline (facility description, matching rules)
     - **REQUIRED** (if using hierarchical pipeline)
   * - ``middle_layer.py``
     - Prompt builder for middle layer pipeline (facility description, matching rules)
     - **REQUIRED** (if using middle layer pipeline)

**Directory Structure:**

Channel finder prompts are customized through the **framework prompt provider** system. Your project's ``framework_prompts/channel_finder/`` directory contains prompt builders that override the framework's generic defaults:

.. code-block:: text

   src/my_control_assistant/framework_prompts/channel_finder/
   ├── in_context.py               # REQUIRED: Facility description for in-context pipeline
   ├── hierarchical.py             # REQUIRED: Facility description for hierarchical pipeline
   └── middle_layer.py             # REQUIRED: Facility description for middle layer pipeline

Each file contains a prompt builder class that provides ``facility_description`` and ``matching_rules`` for its pipeline. Edit the facility description and matching rules within these builders to customize channel finding for your facility.

.. dropdown:: **Step 1: Edit your pipeline prompt builder (Required)**
   :color: success
   :open:

   Each pipeline prompt builder file defines your facility's identity and structure. The LLM uses this context to understand your control system and make accurate semantic matches.

   **What to include:**

   - Physical system descriptions (accelerator sections, subsystems)
   - Channel naming patterns and their meanings
   - Disambiguation rules for ambiguous queries

   **Example (UCSB FEL Accelerator):**

   .. code-block:: python

      # framework_prompts/channel_finder/in_context.py
      import textwrap

      facility_description = textwrap.dedent(
          """
          The University of California, Santa Barbara (UCSB) Free Electron Laser (FEL)
          uses relativistic electrons to generate a powerful terahertz (THz) laser beam.

          1. Electron Source (Thermionic Gun):
             - Electrons are emitted from a thermionic cathode in short pulses
             - Control parameters include gun voltage, beam pulse timing

          2. Acceleration Section:
             - Electrons accelerated by high terminal voltage
             - Control parameters: accelerator voltage stability

          3. Beam Transport and Steering:
             - Steering coils and dipole magnets control beam trajectory
             - Quadrupole magnets focus/defocus the beam

          IMPORTANT TERMINOLOGY AND CONVENTIONS:

          Channel Naming Patterns:
          - "Motor" channels = Control/command channels (for setting positions)
          - "MotorReadBack" or "ReadBack" channels = Status/measurement channels
          - "SetPoint" or "Set" channels = Control values to be commanded

          Disambiguation Rules:
          - When query asks for "control" or "motor control" → return ONLY Motor/Set channels
          - When query asks for "status" or "readback" → return ONLY ReadBack channels
          - When query is ambiguous (e.g., "check") → include both Set and ReadBack
          """
      )

.. dropdown:: **Step 2: Edit matching_rules.py (Optional)**
   :color: info

   If your facility uses terminology that differs from the defaults (or you want more detailed matching rules), customize this file. This is especially useful for:

   - Custom setpoint/readback naming conventions
   - Device synonyms operators commonly use
   - Operational context that affects channel selection

   **Example:**

   .. code-block:: python

      # framework_prompts/channel_finder/in_context.py (matching_rules section)
      import textwrap

      matching_rules = textwrap.dedent(
          """
          MATCHING TERMINOLOGY:

          Setpoint vs Readback:
          - "SP" (Setpoint) = Control/command value to be written
          - "RB" (Readback) = Actual measured value (read-only)
          - "GOLDEN" = Reference value for known good operation
          - When user asks to "set", "control", "adjust" → return SP channels
          - When user asks to "read", "monitor", "measure" → return RB channels
          - When ambiguous ("show me", "what is") → include both SP and RB

          Common Device Synonyms:
          - "bending magnet" = dipole magnet
          - "focusing magnet" or "quad" = quadrupole magnet
          - "corrector" or "steering" = corrector magnet
          - "vacuum level" or "vacuum pressure" = pressure measurement
          """
      )

   **Note:** If you don't need custom matching rules, you can leave this file with minimal content or use the defaults.

.. dropdown:: **How Prompt Builders Work**
   :color: secondary

   Each pipeline prompt builder file (e.g., ``in_context.py``) contains a class that provides ``get_facility_description()`` and ``get_matching_rules()`` methods. The framework's prompt loading system calls these methods to build the complete system prompt for the pipeline.

   Override a method in your prompt builder class to customize that part of the prompt. Leave it unoverridden to use the framework's generic defaults.

**Best Practices:**

1. **Start with facility description**: Get the basic structure working first
2. **Run benchmarks early**: Test with a few queries before writing all rules
3. **Add matching_rules.py incrementally**: Only add rules when benchmarks reveal terminology gaps
4. **Use the CLI for rapid iteration**: ``osprey channel-finder``
5. **Document for your team**: Comments in these files help future maintainers

.. _part4-framework-prompt-customization:

Framework Prompt Customization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework prompt provider system allows you to customize how the agent thinks, plans, and responds—unlike service-level prompts (specific to one service), framework prompts control core agent behaviors across all capabilities. See :doc:`../developer-guides/03_core-framework-systems/04_prompt-customization` for comprehensive prompt customization patterns and advanced techniques.

**What Framework Prompts Control:**

- **Orchestrator**: How the agent creates execution plans and sequences capabilities
- **Classifier**: How the agent decides which capabilities to invoke for a given task
- **Task Extraction**: How conversations are compressed into concrete tasks
- **Response Generation**: How final answers are formatted and presented to users
- **Python Code Generation**: How Python code is generated for data analysis and control system operations

**The Prompt Provider Architecture**

OSPREY uses a **provider registration system** that allows you to override default framework prompts without modifying framework code. Each prompt type has a builder class that you can subclass and customize.

**Step 1: Create Your Custom Prompt Builder**

Create a new module in your agent project (e.g., ``src/my_control_assistant/framework_prompts/``) and subclass the appropriate default builder.

.. dropdown:: **Example: Python Prompt Builder (Already in Your Template!)**
   :color: success

   The control assistant template already includes a custom Python prompt builder that teaches the LLM to use ``osprey.runtime`` utilities for control system operations. This is located at ``src/my_control_assistant/framework_prompts/python.py``:

   .. code-block:: python

      # src/my_control_assistant/framework_prompts/python.py
      import textwrap
      from osprey.prompts.defaults.python import DefaultPythonPromptBuilder

      class ControlSystemPythonPromptBuilder(DefaultPythonPromptBuilder):
          """Python prompt builder with control system runtime utilities guidance.

          Extends the framework's default Python prompts to teach LLMs how to
          interact with control systems using osprey.runtime utilities.
          """

          def get_instructions(self) -> str:
              """Get Python instructions with control system operations guidance."""
              # Get base framework instructions
              base_instructions = super().get_instructions()

              # Add control system-specific guidance
              control_system_guidance = textwrap.dedent("""

                  === CONTROL SYSTEM OPERATIONS ===
                  For reading/writing to control systems, use osprey.runtime utilities:

                  from osprey.runtime import write_channel, read_channel, write_channels

                  Examples:
                      # Write a calculated value
                      from osprey.runtime import write_channel
                      import math
                      voltage = math.sqrt(4150)
                      write_channel("TerminalVoltageSetPoint", voltage)
                      results = {"voltage_set": voltage}

                      # Read current value
                      from osprey.runtime import read_channel
                      current = read_channel("BeamCurrent")
                      print(f"Current: {current}")
                      results = {"beam_current": current}

                  These utilities work with ANY control system (EPICS, Mock, etc.) - you don't
                  need to know which one is configured. All safety checks (limits validation,
                  approval workflows) happen automatically.

                  IMPORTANT:
                  - Never use epics.caput() or epics.caget() directly - use osprey.runtime utilities
                  """).strip()

              return base_instructions + "\n\n" + control_system_guidance

   **What This Does:**

   - Extends ``DefaultPythonPromptBuilder`` with control system-specific instructions
   - Teaches the LLM to use ``osprey.runtime`` utilities instead of direct EPICS calls
   - Provides concrete examples of read/write operations
   - Ensures generated code works with any configured control system (EPICS, Mock, LabVIEW, etc.)

   **Result:** When the Python capability generates code, the LLM automatically uses ``osprey.runtime`` utilities for control system operations, ensuring consistency, safety, and control-system-agnostic code.

.. dropdown:: **Example: Task Extraction Prompt (Already in Your Template!)**
   :color: success

   The control assistant template already includes a custom task extraction prompt builder optimized for control system operations. This is located at ``src/my_control_assistant/framework_prompts/task_extraction.py``.

   **Why Task Extraction Customization is Critical:** Task extraction sits at the beginning of the pipeline and converts conversational input into structured tasks. If it misinterprets the user's intent, the entire downstream pipeline executes the wrong task. This is especially important for domain-specific terminology—for example, "BPM" averaged over the english language on the internet means "beats per minute," but in accelerator physics it means "Beam Position Monitor." The custom prompt teaches domain-specific terminology and provides control system examples to ensure correct interpretation.

   .. code-block:: python

      # src/my_control_assistant/framework_prompts/task_extraction.py
      import textwrap
      from osprey.prompts.defaults import (
          DefaultTaskExtractionPromptBuilder,
          ExtractedTask,
          TaskExtractionExample,
      )
      from osprey.state import MessageUtils, UserMemories

      class ControlSystemTaskExtractionPromptBuilder(DefaultTaskExtractionPromptBuilder):
          """Control-system-specific task extraction prompt builder.

          Provides comprehensive task extraction examples tailored for control system
          operations. These examples replace framework defaults with domain-specific
          patterns for channels, devices, and system monitoring workflows.
          """

          def __init__(self):
              """Initialize with ONLY control system examples.

              The control system examples are comprehensive enough to cover all
              necessary task extraction patterns while being domain-specific and
              relevant to control system operations. This reduces prompt latency
              compared to including framework defaults.
              """
              super().__init__(include_default_examples=False)  # Use only control system examples
              self._add_control_system_examples()

          def get_role(self) -> str:
              """Get the control-system-specific role definition."""
              return "You are a control system assistant task extraction specialist that analyzes conversations to extract actionable tasks related to control system operations."

          def get_instructions(self) -> str:
              """Get the control-system-specific task extraction instructions."""
              return textwrap.dedent("""
              Your job is to:
              1. Understand what the user is asking for in the context of control system operations
              2. Extract a clear, actionable task related to channels, devices, or system monitoring
              3. Determine if the task depends on chat history context
              4. Determine if the task depends on user memory

              ## Control System Guidelines:
              - Create self-contained task descriptions executable without conversation context
              - Resolve channel references from previous messages ("that channel", "those magnets")
              - Resolve temporal references precisely ("an hour ago" → specific timestamp)
              - Extract device families and system names from conversation context
              - Carry forward channel addresses found in previous responses
              - Set depends_on_chat_history=True if task references previous messages
              - Set depends_on_user_memory=True only when task needs specific information from user memory
              - Be specific about channels, time ranges, and operations in task descriptions

              ## Control System Terminology:
              - BPM = Beam Position Monitor (NOT beats per minute - this is accelerator/beam diagnostics)
              - SP = Setpoint (desired value to write to a device)
              - RB/RBV = Readback/Readback Value (actual measured value from a device)
              - Common devices: quadrupoles (focusing magnets), dipoles (bending magnets), RF cavities, vacuum gauges

              ## Common Patterns:
              - Channel reference: "What about that magnet?" → resolve "magnet" to specific channel from history
              - Temporal: "Show me the last hour" → calculate exact start/end times
              - Comparative: "Compare with yesterday" → extract both current and historical requirements
              - Device families: "All quadrupoles in sector 2" → be explicit about the device pattern

              ## Write Operations:
              - Extract the target (channel/device) and value clearly
              - "Set X to Y" → task should specify both X and Y
              - For contextual values, extract the value from conversation history

              ## Computational Requests:
              - State the computational goal, not the implementation steps
              - "Plot X over time" → goal is to create the plot (orchestrator handles data retrieval)
              - "Calculate average" → goal is the calculation (orchestrator handles data gathering)
              """).strip()

   **What This Does:**

   - Replaces framework defaults with control-system-specific examples
   - Uses domain terminology throughout (channels, BPMs, magnets, setpoints, readbacks)
   - Teaches the LLM to extract the goal (WHAT to do) while leaving the workflow (HOW to do it) to the orchestrator
   - Reduces prompt size for better latency compared to including framework defaults

   **Result:** When users have multi-turn conversations about control systems, the LLM extracts clear, actionable tasks with proper context resolution while using domain-specific terminology.

**Step 2: Create the Module __init__.py**

Create an ``__init__.py`` file in your framework_prompts module to export your builders. The control assistant template already has this set up:

.. code-block:: python

   # src/my_control_assistant/framework_prompts/__init__.py
   from .python import ControlSystemPythonPromptBuilder  # Already in template!
   from .task_extraction import ControlSystemTaskExtractionPromptBuilder  # Already in template!
   # from .orchestrator import MyFacilityOrchestratorPromptBuilder  # Add your own
   # Add other builders as you create them

   __all__ = [
       "ControlSystemPythonPromptBuilder",  # Already exported
       "ControlSystemTaskExtractionPromptBuilder",  # Already exported
       # "MyFacilityOrchestratorPromptBuilder",  # Add your own
   ]

**Step 3: Register Your Custom Prompt Provider**

In your agent's ``registry.py``, extend the existing registry configuration to include your custom prompt builders. **The template already registers the Python and Task Extraction prompt builders** - you can add more as needed.

The framework automatically discovers and uses your custom builders. You can override as many or as few prompt types as needed—any not specified will use the framework defaults.

.. tab-set::

   .. tab-item:: Registry Configuration

      .. code-block:: python

         # src/my_control_assistant/registry.py
         from osprey.registry import (
             RegistryConfigProvider,
             extend_framework_registry,
             FrameworkPromptProviderRegistration,
             RegistryConfig
         )


         class MyControlAssistantRegistryProvider(RegistryConfigProvider):
             """Registry provider for My Control Assistant.

             Control system capabilities (channel finding, reading, writing,
             archiver) are provided natively by the framework — no application
             registration needed. This registry only registers facility-specific
             prompt customizations.
             """

             def get_registry_config(self) -> RegistryConfig:
                 """Return registry configuration with custom framework prompts."""
                 return extend_framework_registry(
                     # Custom framework prompts only — capabilities are framework-native
                     framework_prompt_providers=[
                         FrameworkPromptProviderRegistration(
                             module_path="my_control_assistant.framework_prompts",
                             prompt_builders={
                                 "python": "ControlSystemPythonPromptBuilder",
                                 "task_extraction": "ControlSystemTaskExtractionPromptBuilder",
                                 "channel_finder_in_context": "FacilityInContextPromptBuilder",
                                 "channel_finder_hierarchical": "FacilityHierarchicalPromptBuilder",
                                 "channel_finder_middle_layer": "FacilityMiddleLayerPromptBuilder",
                                 # Add your own custom builders:
                                 # "orchestrator": "MyFacilityOrchestratorPromptBuilder",
                                 # "response_generation": "MyFacilityResponseGenerationPromptBuilder",
                                 # "classification": "MyFacilityClassificationPromptBuilder",
                             }
                         )
                     ]
                 )

      .. tip::

         Need to customize framework capabilities or services beyond prompt overrides?
         Use ``osprey eject`` to copy framework source to your project:

         .. code-block:: bash

            osprey eject capability channel_finding    # Copy capability for customization
            osprey eject service channel_finder        # Copy entire service for customization

         After ejecting, register the local version using ``override_capabilities`` in your registry.

   .. tab-item:: Available Builder Types

      .. list-table::
         :header-rows: 1
         :widths: 25 35 40

         * - Builder Type
           - Base Class
           - Purpose
         * - ``python``
           - ``DefaultPythonPromptBuilder``
           - Controls Python code generation (**already customized in template**)
         * - ``task_extraction``
           - ``DefaultTaskExtractionPromptBuilder``
           - Extracts actionable tasks from conversations (**already customized in template**)
         * - ``orchestrator``
           - ``DefaultOrchestratorPromptBuilder``
           - Controls execution planning and capability sequencing
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
         * - ``logbook_search``
           - ``DefaultLogbookSearchPromptBuilder``
           - Customizes logbook search routing and orchestrator guidance for facility-specific terminology

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

.. dropdown:: Example: Orchestrator Prompt Builder (Custom Facility Rules)
   :color: info

   You can create additional custom prompt builders for other framework components. Here's an example of customizing the orchestrator with facility-specific planning rules:

   .. code-block:: python

      # src/my_control_assistant/framework_prompts/orchestrator.py
      import textwrap
      from osprey.prompts.defaults.orchestrator import DefaultOrchestratorPromptBuilder
      from osprey.registry import get_registry

      class MyFacilityOrchestratorPromptBuilder(DefaultOrchestratorPromptBuilder):
          """Facility-specific orchestrator prompt customization."""

          def get_role(self) -> str:
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

.. dropdown:: Example: Custom Classification Prompt
   :color: info

   Customize how the agent classifies tasks:

   .. code-block:: python

      # src/my_control_assistant/framework_prompts/classification.py
      import textwrap
      from osprey.prompts.defaults.classification import DefaultClassificationPromptBuilder

      class MyFacilityClassificationPromptBuilder(DefaultClassificationPromptBuilder):
          """Custom classification for facility-specific task routing."""

          def get_role(self) -> str:
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

          def get_role(self) -> str:
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

.. admonition:: Need a Custom Provider?
   :class: tip

   The framework includes built-in providers for Anthropic, OpenAI, Google, Ollama, CBorg (LBNL), and Stanford AI Playground. If you need to integrate with your institution's AI service or another commercial provider, you can register custom providers. See :ref:`custom-ai-provider-registration` for complete implementation guidance.

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

          async def execute(self, **kwargs):
              """Execute data analysis with specialized prompts."""
              task_objective = self.get_task_objective()

              # 1. Generate analysis plan (simplified)
              analysis_plan = await create_analysis_plan(
                  task_objective=task_objective,
                  state=self._state
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
                  task_objective=task_objective,
                  capability_prompts=prompts,  # Domain-specific guidance
                  execution_folder_name="data_analysis",
                  capability_context_data=self._state.get('capability_context_data', {})
              )

              result = await python_service.ainvoke(request, config=kwargs.get("config"))

              # 4. Store results
              return self.store_output_context(result.execution_result)

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
