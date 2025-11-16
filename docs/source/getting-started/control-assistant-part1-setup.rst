========================================
Part 1: Getting Started
========================================

In this first part, you'll create your control assistant project and explore the generated architecture. The template includes two alternative channel finding pipelines (in-context and hierarchical), service layer patterns, database utilities, and comprehensive testing tools. You'll understand the configuration system that orchestrates all components, including model selection, safety controls, and service deployment. By the end of this section, you'll have a complete project structure ready for customization.

**What You'll Accomplish:**

- Create a control assistant project using the interactive CLI
- Understand the complete project structure and architecture
- Configure AI models, providers, and safety controls
- Set up environment variables for your deployment
- Learn configuration best practices for production deployment

.. dropdown:: **Prerequisites**
   :color: info
   :icon: list-unordered

   **Required:** :doc:`Installation of the framework <installation>` and a working development environment.

   **Recommended:** Complete the :doc:`Hello World Tutorial <hello-world-tutorial>` first.

Step 1: Create the Project
==========================

The interactive menu provides the best onboarding experience with channel finder mode selection:

.. tab-set::

   .. tab-item:: Interactive Mode (Recommended)

      Launch the interactive menu:

      .. code-block:: bash

         osprey

      The menu will guide you through:

      1. **Main Menu** → Select ``[+] Create new project (interactive)``
      2. **Template Selection** → Choose ``control_assistant``
      3. **Project Name** → e.g., ``my-control-assistant``
      4. **Channel Finder Mode** → Select pipeline approach:

         .. code-block:: text

            ○ in_context   - Semantic search (best for few hundred channels, faster)
            ○ hierarchical - Structured navigation (best for >1,000 channels, scalable)
            ● both         - Include both pipelines (maximum flexibility, comparison)

      5. **Registry Style** → Choose ``extend`` (recommended)
      6. **Provider & Model** → Configure AI provider and model (recommended: Claude Haiku)
      7. **API Key** → Automatic detection or secure input

      **Result:** Complete project ready to run with your chosen pipeline configuration.

   .. tab-item:: Direct CLI Command

      For automation or when you know what you want:

      .. code-block:: bash

         # Create with both pipelines enabled (default)
         osprey init my-control-assistant --template control_assistant
         cd my-control-assistant

         # The channel finder mode can be changed later in config.yml

**Generated Project Structure:**

.. code-block:: text

   my-control-assistant/
   ├── src/my_control_assistant/
   │   ├── capabilities/                   # ← Agent capabilities (Osprey integration)
   │   │   ├── channel_finding.py          # Wraps channel_finder service
   │   │   ├── channel_value_retrieval.py  # Live value reads via ConnectorFactory
   │   │   └── archiver_retrieval.py       # Historical data via ConnectorFactory
   │   ├── services/                       # ← Service Layer (key pattern!)
   │   │   └── channel_finder/             # Standalone, testable business logic
   │   │       ├── pipelines/              # Two pipeline architectures:
   │   │       │   ├── in_context/         #   - Semantic search (small systems)
   │   │       │   └── hierarchical/       #   - Hierarchical nav (large systems)
   │   │       ├── databases/              # Database adapters (template, hierarchical, legacy)
   │   │       ├── prompts/                # Pipeline-specific prompts
   │   │       │   ├── in_context/         #   (channel_matcher, query_splitter, etc.)
   │   │       │   └── hierarchical/
   │   │       ├── benchmarks/             # Evaluation system (runner, models)
   │   │       ├── core/                   # Base classes & models
   │   │       ├── llm/                    # LLM completion utilities
   │   │       ├── utils/                  # Config & prompt loading
   │   │       ├── service.py              # Main service API
   │   │       └── cli.py                  # Service CLI
   │   ├── data/                           # ← Your data goes here
   │   │   ├── channel_databases/          # Generated databases
   │   │   │   ├── in_context.json
   │   │   │   ├── hierarchical.json
   │   │   │   └── TEMPLATE_EXAMPLE.json
   │   │   ├── benchmarks/
   │   │   │   ├── datasets/               # Test query datasets
   │   │   │   │   ├── in_context_main.json
   │   │   │   │   └── hierarchical_benchmark.json
   │   │   │   └── results/                # Benchmark output
   │   │   ├── raw/                        # Raw address data (CSV files)
   │   │   │   ├── CSV_EXAMPLE.csv         # Format reference with examples
   │   │   │   └── address_list.csv        # Real UCSB FEL channel data
   │   │   ├── tools/                      # Database utilities
   │   │   │   ├── build_channel_database.py
   │   │   │   ├── validate_database.py
   │   │   │   ├── preview_database.py
   │   │   │   └── llm_channel_namer.py
   │   │   └── README.md                   # Data directory documentation
   │   ├── context_classes.py              # Context type definitions
   │   └── registry.py                     # Capability registry
   ├── services/                           # Docker services
   │   ├── jupyter/                        # JupyterLab + EPICS kernels
   │   ├── open-webui/                     # Chat interface + custom functions
   │   └── pipelines/                      # Osprey backend API
   ├── _agent_data/                        # Runtime data (auto-generated)
   ├── config.yml                          # Main configuration
   └── requirements.txt


Step 2: Understanding Configuration
=====================================

The generated project includes a complete, self-contained configuration that orchestrates all components. Let's examine the key sections you'll customize for your facility.

Configuration File (config.yml)
--------------------------------

The framework uses a **single configuration file** approach - all settings in one place. See :doc:`Configuration Architecture <../developer-guides/03_core-framework-systems/06_configuration-architecture>` for the complete philosophy.

**Location:** ``my-control-assistant/config.yml``

Model Configuration
~~~~~~~~~~~~~~~~~~~~

The framework uses **8 specialized AI models** for different roles. Each can use a different provider and model for optimal performance and cost:

.. code-block:: yaml

   models:
     orchestrator:              # Creates execution plans
       provider: cborg
       model_id: anthropic/claude-haiku
       max_tokens: 4096
     response:                  # Generates final user responses
       provider: cborg
       model_id: anthropic/claude-haiku
     classifier:                # Classifies tasks and selects capabilities
       provider: cborg
       model_id: anthropic/claude-haiku
     # ... 5 more models (approval, task_extraction, memory,
     #     python_code_generator, time_parsing)

**Recommended Starting Configuration:** Use **Claude Haiku for all 8 models**. It provides an excellent trade-off between capabilities and cost, and works exceptionally well with structured outputs - which the framework relies on heavily for task extraction, classification, and orchestration. While you can use different models for different roles as you optimize, Haiku is the best starting point for reliability and consistency. See :doc:`API Reference <../api_reference/01_core_framework/04_configuration_system>` for complete model configuration options.

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

The template includes CBorg (LBNL's service) by default. Simply update the providers to match your environment. See :doc:`Convention over Configuration <../developer-guides/01_understanding-the-framework/02_convention-over-configuration>` for provider patterns.

Semantic Channel Finding Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Control which pipeline mode is active and configure pipeline-specific settings:

.. code-block:: yaml

   channel_finder:
     # Active pipeline mode - Options: "in_context" or "hierarchical"
     pipeline_mode: hierarchical

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

**Key Innovation:** The framework provides a **connector abstraction** that enables development with mock connectors and seamless migration to production. This is a critical feature that lets you develop without hardware access, then deploy to real control systems by changing a single configuration line.

.. tab-set::

   .. tab-item:: Tutorial Mode (Recommended)

      The template starts with **mock connectors** that simulate control system behavior:

      .. code-block:: yaml

         control_system:
           type: mock                   # ← Mock connector (no hardware needed!)

         archiver:
           type: mock_archiver          # ← Mock archiver (synthetic data)

      **Tutorial Mode Benefits:**

      - Works with **any channel names** - no real PVs required
      - Instant setup - no EPICS installation needed
      - Safe experimentation - no risk to hardware
      - Perfect for learning, demos, and development

   .. tab-item:: Production Mode

      Switch to real control systems by changing the ``type`` field. This is a simplified example to show the basic structure - for complete production configuration details including gateway options, SSH tunnels, and troubleshooting, see :ref:`Part 3: Production Deployment <deploy-containerized-services>`.

      .. code-block:: yaml

         control_system:
           type: epics                  # ← Change to 'epics' for production!
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
           type: epics_archiver         # ← EPICS Archiver Appliance
           epics_archiver:
             url: https://archiver.facility.edu:8443
             timeout: 60

      **Production Requirements:**

      - Install ``pyepics``: ``pip install pyepics``
      - Install ``archivertools``: ``pip install archivertools``
      - Configure gateway addresses for your facility
      - Real channel names must exist in your control system

**The Power of Connectors:** Your capabilities use the ``ConnectorFactory`` API, which means the same code works in both modes. No capability changes needed when migrating from tutorial to production - just update the config! See :doc:`Control System Integration Guide <../developer-guides/05_production-systems/06_control-system-integration>` for implementing custom connectors.

**Pattern Detection:** The framework automatically detects control system operations in generated Python code using configurable regex patterns. This enables the approval system to identify when code will read or write to control systems:

.. code-block:: yaml

   control_system:
     type: epics
     patterns:                     # Used by approval system
       epics:
         write:
           - 'epics\.caput\('       # Matches: epics.caput(...)
           - '\.put\('              # Matches: pv.put(...)
         read:
           - 'epics\.caget\('       # Matches: epics.caget(...)
           - '\.get\('              # Matches: pv.get(...)

You'll see this pattern detection in action when you use the Python execution capability in :doc:`Part 3 <control-assistant-part3-production>`.

.. seealso::
   For more details about pattern detection and how it integrates with the approval system,
   see :doc:`../developer-guides/05_production-systems/03_python-execution-service`.

Safety Controls
~~~~~~~~~~~~~~~~

Critical for production deployments - control what code can execute:

.. code-block:: yaml

   # Approval workflow configuration
   approval:
     global_mode: "selective"     # disabled | selective | all_capabilities
     capabilities:
       python_execution:
         enabled: true
         mode: "epics_writes"     # disabled | all_code | epics_writes
       memory:
         enabled: true

   # Execution limits and master safety switches
   execution_control:
     epics:
       writes_enabled: false      # ⚠️ Set true only for production hardware

     limits:
       max_step_retries: 3
       max_execution_time_seconds: 3000
       graph_recursion_limit: 100

**Safety Philosophy:** Fail-secure defaults. EPICS writes are disabled by default - only enable when you're ready to control hardware. See :doc:`Human Approval Workflows <../developer-guides/05_production-systems/01_human-approval-workflows>` for complete security patterns.

Services Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Define which containerized services to deploy:

.. code-block:: yaml

   services:
     jupyter:                     # Python execution environment
       path: ./services/jupyter
       containers:
         read:                    # Read-only kernel
           name: jupyter-read
           port_host: 8088
         write:                   # Write-enabled kernel
           name: jupyter-write
           port_host: 8089
       copy_src: true

     open_webui:                  # Chat interface
       path: ./services/open-webui
       port_host: 8080

     pipelines:                   # Osprey backend
       path: ./services/pipelines
       port_host: 9099
       copy_src: true

   deployed_services:             # Which services to start
     - jupyter
     - open_webui
     - pipelines

The framework provides three core services. Add application-specific services (MongoDB, Redis, etc.) as needed. See :doc:`Container Deployment <../developer-guides/05_production-systems/05_container-and-deployment>` for advanced patterns.

Environment Variables (.env)
------------------------------

Create a ``.env`` file in your project root for secrets and dynamic values:

.. code-block:: bash

   # Copy the example template
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

   Choose your provider for instructions on obtaining an API key:

   **Anthropic (Claude)**

   1. Visit: https://console.anthropic.com/
   2. Sign up or log in with your account
   3. Navigate to 'API Keys' in the settings
   4. Click 'Create Key' and name your key
   5. Copy the key (shown only once!)

   **OpenAI (GPT)**

   1. Visit: https://platform.openai.com/api-keys
   2. Sign up or log in to your OpenAI account
   3. Add billing information if not already set up
   4. Click '+ Create new secret key'
   5. Name your key and copy it (shown only once!)

   **Google (Gemini)**

   1. Visit: https://aistudio.google.com/app/apikey
   2. Sign in with your Google account
   3. Click 'Create API key'
   4. Select a Google Cloud project or create a new one
   5. Copy the generated API key

   **LBNL CBorg**

   1. Visit: https://cborg.lbl.gov
   2. As a Berkeley Lab employee, click 'Request API Key'
   3. Create an API key ($50/month per user allocation)
   4. Copy the key provided

   **Ollama (Local Models)**

   Ollama runs locally and does not require an API key. Simply install Ollama and ensure it's running.

**Optional Variables** (for advanced use cases):

.. code-block:: bash

   # Override project root from config.yml (for multi-environment deployments)
   PROJECT_ROOT=/path/to/my-control-assistant

   # Override Python environment path
   LOCAL_PYTHON_VENV=/path/to/venv/bin/python

**Security:** The ``.env`` file should be in ``.gitignore`` (already configured). Never commit API keys to version control.

**Environment Variable Resolution:** The framework automatically resolves ``${VARIABLE_NAME}`` syntax in ``config.yml`` from your ``.env`` file. See :doc:`Configuration System API <../api_reference/01_core_framework/04_configuration_system>` for advanced patterns.

Next Steps
==========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: ← Tutorial Home
      :link: control-assistant
      :link-type: doc

      Return to tutorial overview

   .. grid-item-card:: Part 2: Channel Finder →
      :link: control-assistant-part2-channel-finder
      :link-type: doc

      Build and test your channel database
