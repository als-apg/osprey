=====================
Claude Code Generator
=====================

The Claude Code Generator provides advanced code generation by leveraging `Claude Code <https://www.claude.com/product/claude-code>`_, Anthropic's enterprise-level agentic code generator. Through the Claude Code SDK, this generator enables multi-turn agentic reasoning, codebase learning, and configurable quality profiles.

Overview
========

**What This Generator Provides**

This Osprey generator leverages the Claude Code SDK to provide capabilities beyond traditional single-pass LLM generators:

- **Read your codebase** to learn from successful examples
- **Execute multi-phase workflows** (scan → plan → implement)
- **Iterate intelligently** with multi-turn reasoning
- **Design custom workflows** through pure configuration - no code changes needed
- **Balance quality and speed** through configurable profiles tailored to your use cases

**Two Pre-configured Approaches:**

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ⚡ Fast Profile (Single-Phase)
      :class-header: sd-bg-primary sd-text-white

      **~20 seconds**

      .. code-block:: text

         User Request
              ↓
         Claude Code (optional lookup)
              ↓
         Python Code

      Single-phase generation with optional codebase learning. Claude autonomously decides whether to reference examples based on the task.

   .. grid-item-card:: 🔬 Robust Profile (Multi-Phase)
      :class-header: sd-bg-info sd-text-white

      **~60 seconds**

      .. code-block:: text

         User Request
              ↓
         Phase 1: SCAN
         → Find examples & patterns
              ↓
         Phase 2: PLAN
         → Create implementation plan
              ↓
         Phase 3: IMPLEMENT
         → Write Python code
              ↓
         Python Code

      Structured 3-phase workflow with comprehensive analysis. Conversation history is maintained across all phases, allowing Claude to build on insights from previous steps.

.. dropdown:: Installation & Setup
   :color: secondary
   :icon: tools

   .. tab-set::

      .. tab-item:: Dependencies

        **Included in Core**

        The Claude Agent SDK is included as a core dependency in Osprey v0.9.6+. No additional installation needed:

        .. code-block:: bash

            pip install osprey-framework

      .. tab-item:: API Keys

         Choose ONE of the following based on your setup:

         **Direct Anthropic API** (default):

         .. code-block:: bash

            export ANTHROPIC_API_KEY='your-api-key-here'

         **CBORG (Lawrence Berkeley Lab)**:

         .. code-block:: bash

            export CBORG_API_KEY='your-cborg-key-here'

         Then configure API provider in ``claude_generator_config.yml`` (see Configuration Reference below).

Quick Start
===========

Minimal Configuration
---------------------

**Step 1: Generate the Claude Code configuration file**

.. code-block:: bash

   osprey generate claude-config

This creates ``claude_generator_config.yml`` with sensible defaults based on your project's LLM provider.

**Step 2: Enable in your main config**

.. code-block:: yaml

   # config.yml
   execution:
     code_generator: "claude_code"
     generators:
       claude_code:
         profile: "fast"  # fast (DEFAULT) | robust | your custom profiles
         claude_config_path: "claude_generator_config.yml"

That's it! This uses the fast profile with sensible defaults. For advanced configuration, see Configuration Reference below.


Configuration-Driven Workflows
==============================

This generator implements a fully configuration-based architecture where you design custom agentic workflows by composing phases and profiles.

**The Power:** Create unlimited custom workflows simply by defining new phases and profiles in your configuration file. No code changes needed!

.. tab-set::

   .. tab-item:: Phases

      **Reusable building blocks** - individual steps in code generation. Each phase configures:

      - ``prompt``: Instructions for what to do
      - ``tools``: Available tools (Read/Grep/Glob or none)
      - ``max_turns``: Iteration limit
      - ``agent_name``: Identity for this phase

      .. tab-set::

         .. tab-item:: generate

            Fast, direct code generation with optional example lookup.

            **Configuration:**

            - **Tools:** Read, Grep, Glob (for optional codebase reading)
            - **Typical max_turns:** 3
            - **Use in:** Single-phase workflows

            **What it does:** Claude can check examples if relevant, then generates code. All in one phase for speed.

            .. dropdown:: Full YAML Configuration
               :icon: code

               .. code-block:: yaml

                  generate:
                    agent_name: "code-generator"
                    prompt: |
                      Generate high-quality Python code for this task.

                      **Your approach:**
                      1. If relevant examples exist in example_scripts/, quickly check them for patterns
                      2. Apply any useful patterns or best practices you find
                      3. Generate the code directly

                      **Code Requirements:**
                      - Include ALL necessary imports at the top
                      - Store results in a dictionary named 'results'
                      - Add comments explaining key steps
                      - Use clear, descriptive variable names
                      - Handle errors appropriately
                      - Output ONLY Python code in a ```python code block

                      **Available Tools:**
                      - Glob: Find example files (e.g., `example_scripts/**/*.py`)
                      - Read: Read relevant examples if they exist
                      - Grep: Search for specific patterns

                      **Strategy:** Check examples ONLY if they seem relevant. Focus on generating good code quickly.

                    tools: ["Read", "Grep", "Glob"]
                    max_turns: 3

         .. tab-item:: scan

            Search codebase for relevant examples and identify patterns.

            **Configuration:**

            - **Tools:** Read, Grep, Glob
            - **Typical max_turns:** 3
            - **Use in:** Multi-phase workflows as first step

            **What it does:** Finds similar implementations, identifies patterns and best practices, notes libraries and approaches.

            .. dropdown:: Full YAML Configuration
               :icon: code

               .. code-block:: yaml

                  scan:
                    agent_name: "code-scanner"
                    prompt: |
                      Analyze the user's request and search the codebase for relevant examples.

                      Your goal is to:
                      1. Identify similar implementations or patterns
                      2. Find relevant libraries, functions, or approaches
                      3. Note any best practices or conventions to follow
                      4. Highlight useful code snippets to reference

                      Output a concise analysis (2-3 paragraphs) covering:
                      - Relevant files and patterns found
                      - Key libraries or approaches to use
                      - Important conventions or best practices observed

                    tools: ["Read", "Grep", "Glob"]
                    max_turns: 3

         .. tab-item:: plan

            Create detailed implementation plan based on requirements.

            **Configuration:**

            - **Tools:** Read (can reference specific files)
            - **Typical max_turns:** 2
            - **Use in:** Multi-phase workflows as middle step

            **What it does:** Defines data structures, functions, and approach based on discovered patterns.

            .. dropdown:: Full YAML Configuration
               :icon: code

               .. code-block:: yaml

                  plan:
                    agent_name: "code-planner"
                    prompt: |
                      Now create a detailed implementation plan for this task.

                      Your plan should include:
                      1. **Imports**: All required libraries and modules
                      2. **Approach**: Step-by-step implementation strategy
                      3. **Data Structures**: Variables and their purposes
                      4. **Key Functions**: Main operations to perform
                      5. **Results**: What to store in the 'results' dictionary
                      6. **Error Handling**: How to handle edge cases

                      Output a structured plan with numbered steps (not code yet, just the plan).
                      Be specific but concise (1-2 pages maximum).

                      Make sure to explicitly present this plan - I will use it in the next step to generate code.

                    tools: ["Read"]
                    max_turns: 2

         .. tab-item:: implement

            Write Python code following a plan from previous phases.

            **Configuration:**

            - **Tools:** None (focuses on implementation)
            - **Typical max_turns:** 2
            - **Use in:** Multi-phase workflows as final step

            **What it does:** Generates code following the plan, includes all imports and error handling.

            .. dropdown:: Full YAML Configuration
               :icon: code

               .. code-block:: yaml

                  implement:
                    agent_name: "code-implementer"
                    prompt: |
                      Generate high-quality Python code following the implementation plan.

                      Requirements:
                      1. Include ALL necessary imports at the top
                      2. Follow the plan's approach precisely (or implement the capability's structured plan)
                      3. Store results in a dictionary named 'results'
                      4. Add comments explaining key steps
                      5. Use clear, descriptive variable names
                      6. Handle errors appropriately
                      7. Output ONLY Python code in a code block

                      The code will be executed in a Jupyter environment with:
                      - Common scientific libraries (numpy, pandas, matplotlib, scipy)
                      - EPICS channel access (pyepics)
                      - Archiver access (configured)

                    tools: []
                    max_turns: 2

         .. tab-item:: custom

            **Create your own phases for specialized workflows**

            You can define unlimited custom phases tailored to your specific needs:

            **Example: Code Review Phase**

            .. code-block:: yaml

               phases:
                 review:
                   agent_name: "code-reviewer"
                   prompt: |
                     Review the generated code for potential issues:
                     - Check for missing imports
                     - Verify error handling
                     - Suggest optimizations
                     - Ensure best practices
                   tools: ["Read"]
                   max_turns: 2

            **Example: Documentation Phase**

            .. code-block:: yaml

               phases:
                 document:
                   agent_name: "code-documenter"
                   prompt: |
                     Generate comprehensive documentation for the code:
                     - Docstrings for all functions
                     - Usage examples
                     - Parameter descriptions
                     - Return value documentation
                   tools: []
                   max_turns: 1

            **Example: Optimization Phase**

            .. code-block:: yaml

               phases:
                 optimize:
                   agent_name: "code-optimizer"
                   prompt: |
                     Analyze and optimize the generated code:
                     - Identify performance bottlenecks
                     - Suggest vectorization opportunities
                     - Recommend caching strategies
                     - Improve algorithm efficiency
                   tools: ["Read", "Grep"]
                   max_turns: 2

            **Use custom phases in profiles:**

            .. code-block:: yaml

               profiles:
                 generate_and_review:
                   phases: [generate, review]
                   model: "claude-haiku-4-5-20251001"
                   max_turns: 5
                   max_budget_usd: 0.15

                 full_workflow_with_docs:
                   phases: [scan, plan, implement, document]
                   model: "claude-haiku-4-5-20251001"
                   max_turns: 12
                   max_budget_usd: 0.30



   .. tab-item:: Profiles

      **Complete workflows** - specify which phases to run and how. Each profile configures:

      - ``phases``: List of phases to execute in order
      - ``model``: Which Claude model to use
      - ``max_turns``: Total iteration budget
      - ``max_budget_usd``: Cost limit
      - ``save_prompts``: Save conversation history (default: true)
      - ``description``: Optional description

      .. tab-set::

         .. tab-item:: fast (DEFAULT)

            **Single-phase generation with optional example lookup**

            **Configuration:**

            - **Phases:** [generate]
            - **Model:** Claude Haiku 4.5
            - **Speed:** Fast (~20s)
            - **Workflow:** Single-phase (generate only)
            - **Codebase Reading:** Enabled (but optional within the phase)
            - **Cost:** ~$0.03 per generation

            **Best For:** Development, simple tasks, rapid iteration, most use cases

            **Tradeoffs:** Less structured than robust profile, but still high quality

            .. dropdown:: Full YAML Configuration
               :icon: code

               .. code-block:: yaml

                  fast:
                    phases: [generate]  # Single-phase generation
                    model: "claude-haiku-4-5-20251001"
                    max_turns: 3
                    max_budget_usd: 0.10
                    save_prompts: true
                    description: "Single-phase generation with optional example lookup (fast, ~20s)"

         .. tab-item:: robust

            **Structured workflow with codebase learning**

            **Configuration:**

            - **Phases:** [scan, plan, implement]
            - **Model:** Claude Haiku 4.5
            - **Speed:** Moderate (~60s)
            - **Workflow:** Multi-phase (scan → plan → implement)
            - **Codebase Reading:** Enabled
            - **Cost:** ~$0.05 per generation

            **Best For:** Complex tasks, learning from examples, structured code generation

            **Tradeoffs:** Slower than fast profile due to 3-phase workflow

            .. dropdown:: Full YAML Configuration
               :icon: code

               .. code-block:: yaml

                  robust:
                    phases: [scan, plan, implement]  # Multi-phase workflow
                    model: "claude-haiku-4-5-20251001"
                    max_turns: 10
                    max_budget_usd: 0.25
                    save_prompts: true
                    description: "Multi-phase workflow with thorough analysis (slower, ~60s)"

         .. tab-item:: Custom Profiles

            **Create your own workflows**

            You can create unlimited custom profiles:

            .. code-block:: yaml

               profiles:
                 # Quick scan then direct generation
                 scan_and_generate:
                   phases: [scan, generate]
                   model: "claude-haiku-4-5-20251001"
                   max_turns: 5
                   max_budget_usd: 0.15

                 # Plan-focused without scanning
                 plan_first:
                   phases: [plan, implement]
                   model: "claude-haiku-4-5-20251001"
                   max_turns: 4
                   max_budget_usd: 0.12

                 # High-quality with Sonnet
                 thorough:
                   phases: [scan, plan, implement]
                   model: "claude-sonnet-4-5-20250929"
                   max_turns: 15
                   max_budget_usd: 0.50



Codebase Reading
================

This generator enables Claude to read your codebase and learn from successful examples. You configure libraries of example scripts, and Claude finds similar implementations, identifies patterns and conventions, uses the same libraries and approaches, and matches your code style. The result is generated code that fits naturally into your codebase, not generic solutions.

Example Workflow
----------------

.. code-block:: text

  User: "Retrieve EPICS PV data and create time series plot"

  SCAN:  Finds read_beam_data.py → Uses pyepics, handles timeouts
          Finds time_series.py → Uses matplotlib, saves with dpi=300

  PLAN:  Use pyepics like examples, handle timeouts, create plot

  IMPLEMENT: Generated code follows discovered patterns!

How It Works
------------

Define example libraries with directories and guidance in ``claude_generator_config.yml``. The guidance field tells Claude what scenarios each directory covers (when to look there). The actual best practices are in the example code itself. All libraries are provided to Claude - it determines what's relevant for each task.

Create a directory structure for your examples:

.. code-block:: bash

  mkdir -p _agent_data/example_scripts/{epics,plotting,analysis}

Add well-documented example scripts:

.. code-block:: python

  # _agent_data/example_scripts/epics/read_channel_example.py
  """
  Example: Reading EPICS channel values with error handling.
  Standard pattern for EPICS operations.
  """
  from epics import caget

  def read_channel_with_timeout(channel_name, timeout=5.0):
      """Read channel value with timeout handling."""
      try:
          value = caget(channel_name, timeout=timeout)
          if value is None:
              raise ValueError(f"Failed to read channel: {channel_name}")
          return value
      except Exception as e:
          print(f"Error reading {channel_name}: {e}")
          return None

  beam_current = read_channel_with_timeout('BEAM:CURRENT')
  results = {'beam_current': beam_current}

Claude will find and learn from examples like this when generating code. Adding a README file in each directory (e.g., ``_agent_data/example_scripts/epics/README.md``) is also helpful - Claude naturally discovers these files and can learn generic patterns and conventions from them.

.. code-block:: markdown

  # _agent_data/example_scripts/epics/README.md

  This directory contains examples for EPICS channel access operations. All examples use
  pyepics and follow standard timeout/error handling patterns. PV names follow the convention
  `SYSTEM:SUBSYSTEM:PARAMETER`.

Security
--------

Codebase reading is read-only with multiple protection layers:

- **Layer 0:** Directory isolation - Claude runs in ``/tmp/osprey_claude_code_restricted/`` with only copied examples
- **Layer 1:** SDK ``allowed_tools`` only includes Read/Grep/Glob
- **Layer 2:** SDK ``disallowed_tools`` blocks Write/Edit/Delete/Bash/Python
- **Layer 3:** PreToolUse safety hook actively blocks dangerous operations

See the Advanced Reference section below for complete safety model details.

Usage Examples
==============

**Basic Usage**

.. code-block:: python

   from osprey.services.python_executor.generation import ClaudeCodeGenerator
   from osprey.services.python_executor.models import PythonExecutionRequest

   generator = ClaudeCodeGenerator()

   request = PythonExecutionRequest(
       user_query="Calculate mean and standard deviation",
       task_objective="Compute basic statistics",
       execution_folder_name="stats",
       expected_results={"mean": "float", "std": "float"}
   )

   code = await generator.generate_code(request, [])

**With Custom Profile**

.. code-block:: python

   # Fast profile for development
   generator = ClaudeCodeGenerator({"profile": "fast"})

   # Robust profile for critical tasks
   generator = ClaudeCodeGenerator({"profile": "robust"})

**With Context and Guidance**

.. code-block:: python

   request = PythonExecutionRequest(
       user_query="Process EPICS PV data and create time series plot",
       task_objective="Visualize accelerator data",
       execution_folder_name="epics_viz",

       capability_context_data={
           "pv_names": ["BEAM:CURRENT", "BEAM:ENERGY"],
           "time_range": "last 1 hour"
       },

       capability_prompts=[
           "Use pyepics for channel access",
           "Handle connection timeouts gracefully",
           "Create matplotlib plot with labels",
           "Save plot to execution folder"
       ],

       expected_results={
           "plot_path": "str",
           "statistics": "dict",
           "pv_values": "list"
       }
   )

   code = await generator.generate_code(request, [])

**With Error Feedback**

.. code-block:: python

   # First attempt
   code_v1 = await generator.generate_code(request, [])

   # Retry with error feedback
   error_chain = ["NameError: name 'pd' is not defined"]
   code_v2 = await generator.generate_code(request, error_chain)
   # Should now include 'import pandas as pd'

.. dropdown:: Advanced Reference & Customization
   :color: secondary
   :icon: tools

   .. tab-set::
      :class: sd-tab-set-wrap

      .. tab-item:: API Configuration

         The Claude Code Generator supports multiple API providers through explicit configuration.

         **Direct Anthropic API (Default)**

         1. Obtain an API key from https://console.anthropic.com/

         2. Set environment variable:

            .. code-block:: bash

               export ANTHROPIC_API_KEY='your-api-key-here'

         3. Configure in ``claude_generator_config.yml``:

            .. code-block:: yaml

               api_config:
                 provider: "anthropic"

         **Model Names:**

         - ``claude-haiku-4-5-20251001`` for Claude Haiku 4.5
         - ``claude-sonnet-4-5-20250929`` for Claude Sonnet 4.5

         **CBORG (Lawrence Berkeley Lab)**

         1. Obtain a CBORG API key from Science IT

         2. Set environment variable:

            .. code-block:: bash

               export CBORG_API_KEY='your-cborg-key-here'

         3. Configure in ``claude_generator_config.yml``:

            .. code-block:: yaml

               api_config:
                 provider: "cborg"
                 base_url: "https://api.cborg.lbl.gov"
                 disable_non_essential_model_calls: true
                 disable_telemetry: true
                 max_output_tokens: 8192  # Reduces throttling

         **Model Names:**

         - ``anthropic/claude-haiku`` for Claude Haiku
         - ``anthropic/claude-sonnet`` for Claude Sonnet
         - ``anthropic/claude-opus`` for Claude Opus

         **Provider Comparison:**

         .. list-table::
            :header-rows: 1
            :widths: 30 35 35

            * - Aspect
              - Direct Anthropic
              - CBORG
            * - **API Key**
              - ``ANTHROPIC_API_KEY``
              - ``CBORG_API_KEY``
            * - **Base URL**
              - Default (api.anthropic.com)
              - https://api.cborg.lbl.gov
            * - **Model Names**
              - Full model IDs (e.g., claude-haiku-4-5-20251001)
              - anthropic/* prefix format (e.g., anthropic/claude-haiku)
            * - **Billing**
              - Direct Anthropic billing
              - LBL project recharge

      .. tab-item:: Codebase Guidance

         Configure example libraries that Claude can read to learn patterns:

         .. code-block:: yaml

            codebase_guidance:
              epics:
                directories:
                  - "_agent_data/example_scripts/epics/"
                guidance: |
                  Use for EPICS channel access tasks: reading PVs, monitoring values,
                  control operations. Examples show pyepics usage patterns.

              plotting:
                directories:
                  - "_agent_data/example_scripts/plotting/"
                guidance: |
                  Use for plotting and visualization: time series plots, multi-parameter
                  comparisons, correlation matrices, publication-quality figures.

              data_analysis:
                directories:
                  - "_agent_data/example_scripts/analysis/"
                guidance: |
                  Use for data analysis tasks: statistical calculations, data manipulation,
                  pandas and numpy operations.

         **How it works:**

         1. **Directories** → Claude can search these paths (via Read/Grep/Glob tools)
         2. **Guidance** → Tells Claude WHEN to use these examples (what scenarios/tasks they cover)
         3. **Always active** → ALL libraries are included, Claude picks what's relevant

      .. tab-item:: Troubleshooting

         **Installation**

         .. code-block:: bash

            # ImportError (upgrade osprey to v0.9.6+)
            pip install --upgrade osprey-framework

            # API key not found
            export ANTHROPIC_API_KEY='your-key'

         **Configuration**

         - **Profile not found:** Check profile name matches exactly in ``claude_generator_config.yml``
         - **Multi-phase workflow not working:** Ensure profile specifies phases correctly: ``phases: [scan, plan, implement]``

         **Generation Issues**

         - **Timeout:** Increase timeout, use fast profile, limit codebase directories
         - **High API usage:** Use fast profile, reduce max_turns, set stricter budgets
         - **Code doesn't follow examples:** Verify directories exist, check that codebase_guidance is configured

         **Quality Issues**

         - **Low quality:** Use robust profile (multi-phase), add better examples, provide more context
         - **Missing imports:** Add examples with imports, include guidance in capability_prompts

         **Performance Issues**

         - **Too slow:** Use fast profile (single-phase), reduce max_turns, limit example directories
         - **Too many API calls:** Reduce max_turns, use fast profile, set stricter budgets

See Also
========

:doc:`service-overview`
    Complete service documentation

:doc:`generator-basic`
    Basic LLM generator for simple setups

:doc:`generator-mock`
    Testing with mock generator

:doc:`index`
    Python Execution Service documentation

`Claude Agent SDK <https://github.com/anthropics/claude-agent-sdk>`_
    Official SDK documentation
