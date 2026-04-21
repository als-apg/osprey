:orphan:

========================================
Part 1: Getting Started
========================================

Create your control assistant project, explore the generated architecture, and configure models, providers, and safety controls.

.. dropdown:: **Prerequisites**
   :color: info
   :icon: list-unordered

   **Required:** :doc:`Installation of the framework <installation>` and a working development environment.
   **Recommended:** Complete the :doc:`Hello World Tutorial <hello-world-tutorial>` first.

Step 1: Create the Project
==========================

.. tab-set::

   .. tab-item:: Interactive Mode (Recommended)

      Launch the interactive menu:

      .. code-block:: bash

         osprey

      The menu will guide you through:

      1. **Template Selection** → Choose ``control_assistant``
      2. **Project Name** → e.g., ``my-control-assistant``
      3. **Channel Finder Mode** → Select pipeline approach (in_context, hierarchical, middle_layer, or all)
      4. **Provider & Model** → Configure AI provider and model
      5. **API Key** → Automatic detection or secure input

      Projects start in **Mock mode** by default for safe learning and development.
      See :ref:`Migrate to Production <migrate-to-production>` in Part 3 for details.

   .. tab-item:: Direct CLI Command

      .. code-block:: bash

         osprey init my-control-assistant --template control_assistant
         cd my-control-assistant

**Generated Project Structure:**

.. note::

   Since v0.11, control system capabilities (channel finding, channel read/write, archiver
   retrieval) and the channel finder service are provided **natively by the framework** — they
   are not generated into your project. Your project only contains prompt customizations, data,
   and a slim registry. Use ``osprey eject`` if you need to customize
   framework components beyond prompt overrides.

.. code-block:: text

   my-control-assistant/
   ├── src/my_control_assistant/
   │   ├── framework_prompts/              # Prompt customizations (override framework defaults)
   │   │   ├── python.py                   # Python code generation prompts
   │   │   ├── task_extraction.py          # Task extraction prompts
   │   │   └── channel_finder/             # Channel finder prompt overrides
   │   │       ├── in_context.py           # Facility description for in-context pipeline
   │   │       ├── hierarchical.py         # Facility description for hierarchical pipeline
   │   │       └── middle_layer.py         # Facility description for middle layer pipeline
   │   ├── data/                           # Your data goes here
   │   │   ├── channel_databases/          # Generated databases (in_context.json, hierarchical.json, etc.)
   │   │   ├── benchmarks/                 # Test query datasets and results
   │   │   └── raw/                        # Raw address data (CSV files)
   │   └── registry.py                     # Registry (prompt provider registrations only)
   ├── config.yml                          # Main configuration
   └── pyproject.toml

**Framework-Provided MCP Tools** (ready to use, no code to write):

.. tab-set::

   .. tab-item:: Control System

      - **Channel Finding** — Resolves natural language to channel addresses using three pipeline modes (see :doc:`Part 2 <control-assistant-part2-channel-finder>`)
      - **Channel Read** — Reads live ``ChannelValue`` objects via connector abstraction (mock/EPICS/Tango/LabVIEW)
      - **Channel Write** — Sets values and returns ``ChannelWriteResult`` with safety layers (master switch, approval, limits, verification)
      - **Archiver Retrieval** — Queries historical time-series data from facility archivers

   .. tab-item:: Analysis & Execution

      - **Python Execution** — Generates and executes analysis code in sandboxed environments
      - **Time Range Parsing** — Converts natural language time expressions (*"last 24 hours"*) to precise ranges

   .. tab-item:: Knowledge Retrieval

      - **Memory** — Stores and recalls information across conversations
      - **Logbook Search (ARIEL)** — Searches facility electronic logbooks. See :doc:`ARIEL Logbook Search </how-to/ariel/index>`.

See :doc:`MCP Servers </architecture/mcp-servers>` for the full list of tools available.


Step 2: Understanding Configuration
=====================================

The generated project includes a complete configuration. Let's examine the key sections you'll customize for your facility.

Configuration File (config.yml)
--------------------------------

The framework uses a **single configuration file** approach - all settings in one place (``my-control-assistant/config.yml``). See :doc:`Architecture Overview </architecture/index>` for the complete philosophy.

.. admonition:: PLACEHOLDER: CONFIG-ARCHITECTURE
   :class: warning

   **Old content (line 158):** "The framework uses **10 specialized AI models** for different roles. Each can use a different provider and model..."
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old architecture used 10 model slots (orchestrator, classifier, response, approval, etc.) for LangGraph node routing. Claude Code uses a single model with MCP tools. The config.yml model section likely has a different shape now.
   **Action needed:** Document the current model configuration section of config.yml for Claude Code deployments.

API Provider Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure your AI/LLM providers with API keys from environment variables:

.. code-block:: yaml

   api:
     providers:
       cborg:                   # LBNL's internal service
         api_key: ${CBORG_API_KEY}
         base_url: https://api.cborg.lbl.gov/v1
       anthropic:
         api_key: ${ANTHROPIC_API_KEY}
         base_url: https://api.anthropic.com
       openai:
         api_key: ${OPENAI_API_KEY}
         base_url: https://api.openai.com/v1
       ollama:                  # Local models
         api_key: ollama
         base_url: http://localhost:11434

Update the providers to match your environment.

.. admonition:: Custom Providers
   :class: tip

   Need to integrate your institution's AI service? See :doc:`Configure Providers </how-to/configure-providers>` for complete implementation guidance.

Semantic Channel Finding Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Control which pipeline mode is active and configure pipeline-specific settings:

.. code-block:: yaml

   channel_finder:
     pipeline_mode: hierarchical    # Options: "in_context", "hierarchical", "middle_layer"

     pipelines:
       in_context:
         database:
           type: template
           path: src/my_control_assistant/data/channel_databases/in_context.json
           presentation_mode: template
         processing:
           chunk_dictionary: false
           max_correction_iterations: 2

       hierarchical:
         database:
           type: hierarchical
           path: src/my_control_assistant/data/channel_databases/hierarchical.json

**Pipeline Selection:** Start with ``in_context`` for systems with few hundred channels, or ``hierarchical`` for larger systems. You'll explore both :doc:`in Part 2 <control-assistant-part2-channel-finder>`.

Control System & Archiver Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides a **connector abstraction** that enables development with mock connectors and seamless migration to production by changing a single configuration line.

.. tab-set::

   .. tab-item:: Tutorial Mode (Recommended)

      The template starts with **mock connectors** that simulate control system behavior:

      .. code-block:: yaml

         control_system:
           type: mock                   # Mock connector (no hardware needed)

         archiver:
           type: mock_archiver          # Mock archiver (synthetic data)

      Works with any channel names, no EPICS installation needed, no risk to hardware.

   .. tab-item:: Production Mode

      Switch to real control systems by changing the ``type`` field. For complete production configuration details, see :ref:`Part 3: Production Deployment <deploy-containerized-services>`.

      .. code-block:: yaml

         control_system:
           type: epics
           connector:
             epics:
               gateways:
                 read_only:
                   address: cagw.facility.edu
                   port: 5064
                 read_write:
                   address: cagw-rw.facility.edu
                   port: 5065
               timeout: 5.0

         archiver:
           type: epics_archiver
           epics_archiver:
             url: https://archiver.facility.edu:8443
             timeout: 60

      **Production Requirements:**

      - Install ``pyepics``: ``uv add pyepics``
      - Configure gateway addresses for your facility

Your MCP tools use the ``ConnectorFactory`` API, which means the same tools work in both modes. No changes needed when migrating from tutorial to production -- just update the config. See :doc:`Add a Connector </how-to/add-connector>` for implementing custom connectors.

**Pattern Detection (Security Layer):** The framework automatically detects ALL control system operations in generated Python code -- both approved API usage AND circumvention attempts:

- **Approved API**: ``write_channel()``, ``read_channel()`` (has limits, verification)
- **Circumvention**: Direct library calls like ``epics.caput()``, ``tango.DeviceProxy().write_attribute()``

.. note::
   The pattern detection includes both the unified ``osprey.runtime`` API (``write_channel``,
   ``read_channel``) and legacy EPICS functions (``caput``, ``caget``) for backward compatibility.

.. seealso::
   For more details about pattern detection and how it integrates with the approval system,
   see :doc:`Use the Python Executor </how-to/use-python-executor>`.

Safety Controls
~~~~~~~~~~~~~~~~

.. admonition:: PLACEHOLDER: APPROVAL-WORKFLOW
   :class: warning

   **Old content (line 331):** "approval: global_mode: selective ... capabilities: python_execution: enabled: true mode: control_writes ... execution_control: limits: graph_recursion_limit: 100"
   **New equivalent:** Needs human judgment
   **Why this is fuzzy:** The old approval system used LangGraph-based modes (``global_mode``, ``all_capabilities``, ``graph_recursion_limit``). Claude Code uses per-tool approval policies defined in Claude Code settings, not config.yml. The safety controls config section needs to reflect the new MCP-based approval model.
   **Action needed:** Document the current safety controls config section, including how per-tool approval policies are configured and how execution limits work under Claude Code orchestration.

Environment Variables (.env)
------------------------------

Create a ``.env`` file in your project root for secrets and dynamic values:

.. code-block:: bash

   cp .env.example .env

**Required Variables:**

.. code-block:: bash

   # API Keys (configure for your chosen provider)
   CBORG_API_KEY=your-cborg-key           # If using CBorg
   ANTHROPIC_API_KEY=sk-ant-...           # If using Anthropic
   OPENAI_API_KEY=sk-...                  # If using OpenAI
   GOOGLE_API_KEY=...                     # If using Google

   # System configuration
   TZ=America/Los_Angeles                 # Timezone for containers

.. dropdown:: **Where do I get an API key?**
   :color: info
   :icon: key

   - **Anthropic**: https://console.anthropic.com/ -- sign up, navigate to API Keys, create key
   - **OpenAI**: https://platform.openai.com/api-keys -- sign up, add billing, create secret key
   - **Google**: https://aistudio.google.com/app/apikey -- sign in, create API key for a Cloud project
   - **LBNL CBorg**: https://cborg.lbl.gov -- request API key ($50/month per user allocation)
   - **Ollama**: Runs locally, no API key required

**Security:** The ``.env`` file should be in ``.gitignore`` (already configured). Never commit API keys to version control. The framework automatically resolves ``${VARIABLE_NAME}`` syntax in ``config.yml`` from your ``.env`` file.

Next Steps
==========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Tutorial Home
      :link: control-assistant
      :link-type: doc

      Return to tutorial overview

   .. grid-item-card:: Part 2: Channel Finder
      :link: control-assistant-part2-channel-finder
      :link-type: doc

      Build and test your channel database
