:orphan:

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
- Build custom MCP tools for facility-specific operations

Step 10: Prompt Customization
=============================

The template works out of the box, but customizing prompts with facility-specific knowledge dramatically improves accuracy, relevance, and user trust. In the Claude Code architecture, prompt customization is done through **Claude rules files** and **MCP server configuration**.

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
          - When query asks for "control" or "motor control" -> return ONLY Motor/Set channels
          - When query asks for "status" or "readback" -> return ONLY ReadBack channels
          - When query is ambiguous (e.g., "check") -> include both Set and ReadBack
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
          - When user asks to "set", "control", "adjust" -> return SP channels
          - When user asks to "read", "monitor", "measure" -> return RB channels
          - When ambiguous ("show me", "what is") -> include both SP and RB

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

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 166):** "The framework prompt provider system allows you to customize how the agent thinks, plans, and responds"
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old orchestrator/classifier/response-generation prompt provider architecture has been replaced by Claude Code's rules files, CLAUDE.md, and MCP tool descriptions. The customization surface is fundamentally different.
   **Action needed:** Document how users customize agent behavior in the Claude Code architecture (e.g., editing ``.claude/rules/``, ``CLAUDE.md``, MCP server tool descriptions).

The old framework used a prompt provider registration system with builder classes for orchestrator, classifier, task extraction, and response generation prompts. In the Claude Code architecture, agent behavior is customized through different mechanisms:

- **Claude rules files** (``.claude/rules/``): Control agent behavior, planning, and response style
- **CLAUDE.md**: Project-level instructions that shape how Claude approaches tasks
- **MCP tool descriptions**: How tools present themselves affects how Claude selects and uses them
- **System prompt configuration**: Via ``osprey claude chat`` settings

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 187):** "Example: Python Prompt Builder" and "Example: Task Extraction Prompt Builder" showing subclassing of ``DefaultPythonPromptBuilder`` and ``DefaultTaskExtractionPromptBuilder`` from ``osprey.prompts``
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** These builder classes imported from ``osprey.prompts`` (DELETED) and ``osprey.state`` (DELETED). The Python executor is now an MCP server, and task extraction is handled by Claude Code natively.
   **Action needed:** Provide equivalent examples showing how to customize Python code generation guidance and task interpretation via Claude rules files or MCP tool configuration.

Step 11: System Configuration
==============================

The ``config.yml`` file controls key aspects of your deployment.

.. admonition:: PLACEHOLDER: CONFIG-ARCHITECTURE
   :class: warning

   **Old content (line 592):** "The framework uses 8 specialized models for different roles" (orchestrator, response, classifier, approval, task_extraction, memory, python_code_generator, time_parsing)
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** Claude Code uses a single model (Claude) for all reasoning. The multi-model architecture with separate orchestrator/classifier/response models no longer applies. However, MCP servers may still use models for specific tasks (e.g., channel finder pipelines).
   **Action needed:** Document what model configuration still applies in the Claude Code architecture (provider env vars, MCP server model settings) and what has been removed.

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
       +===========================================+
       |   MyFacility Control Assistant            |
       |   Version 1.0.0                           |
       +===========================================+

**Built-in Themes:**

- ``default`` / ``vulcan``: Purple-teal theme (both are identical)
- ``custom``: Define your own facility colors using the ``custom_theme`` section above

.. admonition:: Collaboration Welcome
   :class: outreach

   We welcome contributions of new built-in themes! If you've designed a theme for your facility that you'd like to share with the community, please open a GitHub issue. We're happy to include additional themes that showcase different color palettes and facility branding styles.

Step 12: Advanced Features
===========================

For experienced users, the framework provides several advanced features for optimization and debugging.

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 727):** Slash commands ``/planning``, ``/task:off``, ``/task:on``, ``/caps:off``, ``/caps:on`` for runtime control of planning mode, task extraction bypass, and capability selection bypass.
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** Claude Code has its own slash command system. The old ``/task:off`` and ``/caps:off`` bypasses controlled the LangGraph pipeline stages (task extraction, capability classification) which no longer exist. Claude Code's tool selection is native.
   **Action needed:** Document which runtime controls are available in the Claude Code architecture (e.g., ``osprey claude chat`` flags, Claude Code native commands) and remove references to deleted pipeline stages.

Step 13: Extending with Custom MCP Tools
=========================================

Sometimes the built-in MCP tools are too generic for your needs. You can create custom MCP tools to handle facility-specific operations.

.. admonition:: PLACEHOLDER: CONCEPTUAL-MAPPING
   :class: warning

   **Old content (line 820):** "Extending Framework Capabilities" showing ``BaseCapability`` subclassing with ``@capability_node``, ``CapabilityContext``, classifier/orchestrator guides, and registry-based capability replacement via ``exclude_capabilities``.
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** ``BaseCapability``, ``@capability_node``, and ``CapabilityContext`` are all DELETED. Capabilities are now MCP tools served by MCP servers. The extension pattern is fundamentally different: instead of subclassing a base class and registering in a Python registry, you create a new MCP server (or add tools to an existing one) and register it in Claude Code's MCP configuration.
   **Action needed:** Provide a concrete example of creating a custom MCP tool (e.g., a specialized data analysis tool) and registering it with Claude Code, replacing the old ``DataAnalysisCapability`` example.

**Why Create Custom MCP Tools?**

The framework includes built-in MCP tools for channel finding, reading, writing, and archiver access. While functional, you may need specialized tools for your facility:

- **Data analysis** that needs structured result templates and domain-specific prompts
- **Data visualization** with facility-specific plotting conventions
- **Machine operations** with custom safety checks and approval workflows

.. admonition:: PLACEHOLDER: MISSING-EXAMPLE
   :class: warning

   **Old content (line 844):** Complete ``DataAnalysisCapability`` example with classifier guide, orchestrator guide, and registry configuration showing ``exclude_capabilities=["python"]``.
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** Need a concrete example of a custom MCP tool that replaces this pattern. The MCP server creation pattern (using FastMCP) exists in the codebase but no tutorial example has been written yet.
   **Action needed:** Write an equivalent example showing: (1) creating a custom MCP tool with FastMCP, (2) registering it in the Claude Code MCP configuration, (3) how Claude discovers and uses it.


Navigation
==========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Part 3: Integration & Deployment
      :link: control-assistant-part3-production
      :link-type: doc

      Return to integration guide

   .. grid-item-card:: Tutorial Home
      :link: control-assistant
      :link-type: doc

      Back to tutorial overview
