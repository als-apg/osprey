====================
Configuration System
====================

.. admonition:: 📖 Configuration Architecture Guide
   :class: tip

   For a comprehensive guide to the YAML configuration system, self-contained architecture, and best practices, see :doc:`../../developer-guides/03_core-framework-systems/06_configuration-architecture`.

   This page provides the complete reference for all configuration sections and the Python API for accessing configuration at runtime.

Configuration system with YAML loading, environment resolution, and seamless LangGraph integration.

.. currentmodule:: osprey.utils.config

Core Classes
============

ConfigBuilder
-------------

.. autoclass:: ConfigBuilder
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Main configuration builder with YAML loading, environment resolution, and LangGraph integration.

Primary Access Functions
========================

.. admonition:: New in v0.7.7: Multi-Project Support
   :class: version-07plus-change

   All configuration functions now accept an optional ``config_path`` parameter to explicitly specify which project's configuration to load. This enables multi-project workflows where you can work with multiple projects simultaneously.

   **Common pattern:**

   .. code-block:: python

      from osprey.utils.config import get_model_config

      # Use specific project's config
      model = get_model_config("orchestrator", config_path="~/project/config.yml")

      # Default behavior (searches current directory)
      model = get_model_config("orchestrator")

   See :doc:`../../developer-guides/03_core-framework-systems/06_configuration-architecture` for complete multi-project workflow documentation.

.. autofunction:: get_config_value

**Multi-Project Usage:**

.. code-block:: python

   # Get value from specific project
   value = get_config_value(
       "models.orchestrator.provider",
       config_path="/path/to/project/config.yml"
   )

.. autofunction:: get_full_configuration

**Multi-Project Usage:**

.. code-block:: python

   # Load complete config from specific project
   config = get_full_configuration(config_path="~/my-agent/config.yml")

.. autofunction:: get_agent_dir

**Multi-Project Usage:**

.. code-block:: python

   # Get agent directory for specific project
   agent_dir = get_agent_dir(config_path="~/my-agent/config.yml")

Specialized Configuration Functions
===================================

Model and Provider Access
-------------------------

.. autofunction:: get_model_config

.. autofunction:: get_provider_config

.. dropdown:: Need Support for Additional Providers?
    :color: info
    :icon: people

    The framework's provider system is designed for extensibility. Applications can register custom AI providers directly through the registry system - see :doc:`../../developer-guides/03_core-framework-systems/03_registry-and-discovery` for examples of registering Azure OpenAI, institutional AI services (Stanford AI Playground, LBNL CBorg, etc.), or other providers.

    For providers that would benefit the broader community, we're happy to work with you to add native framework support. Please create an issue on the GitHub repository to discuss integration requirements.

Service Configuration
---------------------

.. autofunction:: get_framework_service_config

.. autofunction:: get_application_service_config

Runtime Information
-------------------

.. autofunction:: get_session_info

.. autofunction:: get_interface_context

.. autofunction:: get_current_application

.. autofunction:: get_execution_limits

.. autofunction:: get_agent_control_defaults

.. autofunction:: get_classification_config

Development Utilities
---------------------

.. autofunction:: get_pipeline_config

Internal Implementation
=======================

.. autofunction:: _get_config

.. autofunction:: _get_configurable

Configuration Sections Reference
=================================

This section provides a complete reference for all configuration sections available in the Osprey Framework.

.. admonition:: 📖 Understanding Configuration Architecture
   :class: note

   Before diving into specific sections, review :doc:`../../developer-guides/03_core-framework-systems/06_configuration-architecture` to understand:

   - The self-contained configuration approach
   - Configuration templates and project initialization
   - Environment variable integration
   - Best practices for configuration organization

Configuration Sections
======================

All configuration is contained in a single ``config.yml`` file at the project root.

build_dir
---------

**Type:** String

**Location:** Root ``config.yml``

**Default:** ``./build``

**Purpose:** Specifies the directory where container build files are generated.

.. code-block:: yaml

   build_dir: ./build

**Details:**

- Used by container management system
- Contains generated Docker Compose files
- Should be in ``.gitignore``
- Can use environment variables: ``${BUILD_DIR}``

project_name
------------

**Type:** String

**Location:** Root ``config.yml``

**Default:** Derived from ``project_root`` path if not specified

**Purpose:** Canonical identifier for the project/agent.

.. code-block:: yaml

   project_name: "weather-agent"

**Details:**

- First-class attribute for project identification
- Used for container labels (Docker/Podman metadata)
- Enables multi-project container management
- Shown in ``osprey deploy status`` output
- Separate from ``pipeline.name`` (which is for OpenWebUI display)
- Falls back to extracting from ``project_root`` path if not specified

.. admonition:: New in v0.8.2: Container Project Tracking
   :class: version-08plus-change

   The ``project_name`` attribute enables tracking which project owns deployed containers using Docker labels. All containers are automatically labeled with ``osprey.project.name`` for easy identification and filtering.

project_root
------------

**Type:** String

**Location:** Root ``config.yml``

**Default:** None (must be specified)

**Purpose:** Absolute path to project root directory.

.. code-block:: yaml

   project_root: ${PROJECT_ROOT}
   # Or hard-coded:
   project_root: /home/user/my-project

**Details:**

- Required for all path resolution
- Use environment variable for portability across machines
- Must be absolute path
- Used by container management and execution systems

registry_path
-------------

**Type:** String

**Location:** Root ``config.yml``

**Default:** None (must be specified)

**Purpose:** Path to the application's registry module.

.. code-block:: yaml

   registry_path: ./src/my_app/registry.py

**Details:**

- Points to the Python module that registers capabilities and context types
- Path is relative to project root
- Must contain a ``registry.py`` file with capability and context registrations
- Required for framework to discover application capabilities

Approval Configuration
======================

Controls human approval workflows and safety policies.

approval.global_mode
--------------------

**Type:** String (enum)

**Location:** Root ``config.yml`` (operator control)

**Default:** ``"selective"``

**Options:** ``"disabled"`` | ``"selective"`` | ``"all_capabilities"``

**Purpose:** System-wide approval policy.

.. code-block:: yaml

   approval:
     global_mode: "selective"

**Values:**

- ``disabled`` - No approval required for any operations
- ``selective`` - Approval based on capability-specific settings
- ``all_capabilities`` - All capability executions require approval

approval.capabilities.python_execution
--------------------------------------

**Type:** Object

**Location:** Root ``config.yml``

**Purpose:** Controls approval for Python code generation and execution.

.. code-block:: yaml

   approval:
     capabilities:
       python_execution:
         enabled: true
         mode: "epics_writes"

**Fields:**

``enabled`` (boolean)
   Whether Python execution capability is available

   - ``true`` - Python capability can be used
   - ``false`` - Python capability disabled

``mode`` (string)
   When to require approval:

   - ``"disabled"`` - No approval required
   - ``"epics_writes"`` - Approve only code that writes to EPICS
   - ``"all_code"`` - Approve all Python code execution

**Example:**

.. code-block:: yaml

   # In config.yml (root)
   approval:
     capabilities:
       python_execution:
         enabled: true
         mode: "epics_writes"  # Approve only EPICS write operations

approval.capabilities.memory
----------------------------

**Type:** Object

**Location:** Root ``config.yml``

**Purpose:** Controls approval for memory operations.

.. code-block:: yaml

   approval:
     capabilities:
       memory:
         enabled: true

**Fields:**

``enabled`` (boolean)
   - ``true`` - Memory storage and retrieval allowed
   - ``false`` - Memory operations disabled

Execution Control
=================

Runtime behavior configuration and safety limits.

execution_control.epics.writes_enabled
--------------------------------------

**Type:** Boolean

**Location:** Root ``config.yml`` (operator control)

**Default:** ``false``

**Purpose:** Master switch for EPICS hardware write operations.

.. code-block:: yaml

   execution_control:
     epics:
       writes_enabled: false

**Details:**

- ``true`` - EPICS write operations can execute (production mode)
- ``false`` - All EPICS writes blocked (safe default for development)
- Operator-level safety control
- Independent of approval settings

execution_control.agent_control
-------------------------------

**Type:** Object

**Location:** Root ``config.yml``

**Purpose:** Performance bypass settings for agent processing.

.. code-block:: yaml

   execution_control:
     agent_control:
       task_extraction_bypass_enabled: false
       capability_selection_bypass_enabled: false

**Fields:**

``task_extraction_bypass_enabled`` (boolean)
   Skip LLM-based task extraction

   - ``false`` (default) - Use LLM to extract structured tasks
   - ``true`` - Pass full conversation history (faster but less precise)
   - Can be overridden at runtime with ``/task:off`` command

``capability_selection_bypass_enabled`` (boolean)
   Skip LLM-based capability classification

   - ``false`` (default) - Use LLM to select relevant capabilities
   - ``true`` - Activate all registered capabilities (faster but less efficient)
   - Can be overridden at runtime with ``/caps:off`` command

**Use Cases:**

- Development: Enable bypasses for faster iteration
- Production: Keep disabled for optimal performance
- R&D: Enable when exploring with small capability sets

execution_control.limits
------------------------

**Type:** Object

**Location:** Root ``config.yml``

**Purpose:** Safety limits and execution constraints.

.. code-block:: yaml

   execution_control:
     limits:
       max_reclassifications: 1
       max_planning_attempts: 2
       max_step_retries: 3
       max_execution_time_seconds: 3000
       graph_recursion_limit: 100
       max_concurrent_classifications: 5

**Fields:**

``max_reclassifications`` (integer)
   Maximum number of reclassifications allowed

   - Default: ``1`` (allows 1 reclassification, meaning 2 total classification attempts)
   - Prevents infinite reclassification loops
   - Counter only increments when reclassification actually occurs, not on initial classification

``max_planning_attempts`` (integer)
   Maximum orchestrator planning attempts

   - Default: ``2``
   - Fails task if planning repeatedly fails

``max_step_retries`` (integer)
   Maximum retries per execution step

   - Default: ``0``
   - Applies to retriable errors only

``max_execution_time_seconds`` (integer)
   Maximum total execution time

   - Default: ``300`` (5 minutes)
   - Prevents runaway executions

``graph_recursion_limit`` (integer)
   LangGraph recursion limit

   - Default: ``100``
   - Prevents infinite state graph loops

``max_concurrent_classifications`` (integer)
   Maximum concurrent LLM classification requests

   - Default: ``5``
   - Controls parallel capability classification to prevent API flooding
   - Balances performance with API rate limits
   - Higher values = faster classification but more API load

System Configuration
====================

Infrastructure-wide settings.

system.timezone
---------------

**Type:** String

**Location:** Root or framework ``config.yml``

**Default:** ``America/Los_Angeles``

**Purpose:** Timezone for all framework services and containers.

.. code-block:: yaml

   system:
     timezone: ${TZ:-America/Los_Angeles}

**Details:**

- Uses standard timezone names (e.g., ``America/New_York``, ``Europe/London``)
- Ensures consistent timestamps across all components
- Propagated to all containers
- Can use environment variable for host timezone: ``${TZ}``

File Paths Configuration
========================

Controls agent data directory structure.

file_paths.agent_data_dir
-------------------------

**Type:** String

**Location:** Root ``config.yml``

**Default:** ``_agent_data``

**Purpose:** Parent directory for all agent-related data.

.. code-block:: yaml

   file_paths:
     agent_data_dir: _agent_data

**Details:**

- All agent data subdirectories are relative to this
- Created automatically if doesn't exist
- Should be in ``.gitignore`` for development

file_paths Subdirectories
--------------------------

**Type:** Strings

**Location:** Root ``config.yml``

**Purpose:** Subdirectories within ``agent_data_dir``.

.. code-block:: yaml

   file_paths:
     agent_data_dir: _agent_data
     executed_python_scripts_dir: executed_scripts
     execution_plans_dir: execution_plans
     user_memory_dir: user_memory
     registry_exports_dir: registry_exports
     prompts_dir: prompts
     checkpoints: checkpoints

**Subdirectories:**

``executed_python_scripts_dir``
   Stores Python code executed by the framework

``execution_plans_dir``
   Stores orchestrator execution plans (JSON)

``user_memory_dir``
   Stores user memory data

``registry_exports_dir``
   Stores exported registry information

``prompts_dir``
   Stores generated prompts when debug enabled

``checkpoints``
   Stores LangGraph checkpoints for conversation state

**Application-Specific Paths:**

Applications can add their own paths:

.. code-block:: yaml

   # In src/applications/als_assistant/config.yml
   file_paths:
     launcher_outputs_dir: launcher_outputs

Deployed Services
=================

Controls which containers are started.

deployed_services
-----------------

**Type:** List of strings

**Location:** Root, framework, or application ``config.yml``

**Default:** ``[]``

**Purpose:** Specifies which services to deploy when running container management.

.. code-block:: yaml

   deployed_services:
     # Framework infrastructure services
     - jupyter
     - open_webui
     - pipelines

     # Application-specific services (optional)
     - mongo          # E.g., MongoDB for ALS Assistant
     - pv_finder      # E.g., PV Finder MCP for ALS Assistant
     - langfuse       # E.g., Observability for ALS Assistant

**Service Naming:**

Services use simple names that match their configuration keys:
   - Infrastructure services: ``jupyter``, ``open_webui``, ``pipelines``
   - Application services: ``mongo``, ``pv_finder``, etc.
   - Names correspond to keys defined in the ``services:`` configuration section

**Configuration Structure:**

.. code-block:: yaml

   # User project config.yml
   deployed_services:
     - jupyter
     - open_webui
     - pipelines
     - mongo       # Application-specific service

   # All services defined in services: section
   services:
     jupyter:
       path: ./services/jupyter
     open_webui:
       path: ./services/open-webui
     pipelines:
       path: ./services/pipelines
     mongo:
       path: ./services/mongo

**Details:**

- Services listed in ``deployed_services`` must be defined in the ``services:`` section
- Only listed services will be deployed when running ``osprey deploy up``
- Service names are simple strings matching configuration keys
- See :doc:`../../developer-guides/05_production-systems/05_container-and-deployment` for complete deployment guide

API Provider Configuration
==========================

Configuration for AI/ML model providers.

api.providers
-------------

**Type:** Object (nested)

**Location:** Root ``config.yml``

**Purpose:** Configuration for external API providers.

.. code-block:: yaml

   api:
     providers:
       cborg:
         api_key: ${CBORG_API_KEY}
         base_url: https://api.cborg.lbl.gov/v1
         timeout: 30

       stanford:
         api_key: ${STANFORD_API_KEY}
         base_url: https://aiapi-prod.stanford.edu/v1

       argo:
         api_key: ${ARGO_API_KEY}
         base_url: https://argo-bridge.cels.anl.gov

       anthropic:
         api_key: ${ANTHROPIC_API_KEY}
         base_url: https://api.anthropic.com

      openai:
        api_key: ${OPENAI_API_KEY}
        base_url: https://api.openai.com/v1

      google:
        api_key: ${GOOGLE_API_KEY}
        base_url: https://generativelanguage.googleapis.com/v1beta

       ollama:
         api_key: ollama
         base_url: http://doudna:11434
         host: doudna
         port: 11434

**Common Fields:**

``api_key`` (string, required)
   API authentication key

   - Use environment variables: ``${API_KEY_NAME}``
   - Never hard-code actual keys
   - For Ollama: use literal string ``"ollama"``

``base_url`` (string, required)
   Base URL for API endpoint

   - Must include protocol (http/https)
   - No trailing slash
   - Can use environment variables

``timeout`` (integer, optional)
   Request timeout in seconds

   - Default varies by provider
   - Increase for slow connections

**Provider-Specific Fields:**

Ollama:
   Additional fields for container networking:

   .. code-block:: yaml

      ollama:
        api_key: ollama
        base_url: http://doudna:11434
        host: doudna  # For container access
        port: 11434   # For container access

**Security:**

- Always use environment variables for API keys
- Store keys in ``.env`` file (add to ``.gitignore``)
- Never commit actual keys to version control

Model Configuration
===================

LLM model assignments for framework and application components.

Framework Models
----------------

**Type:** Object (nested)

**Location:** ``src/osprey/config.yml`` (defaults), can override in root or application

**Purpose:** Model configurations for framework infrastructure components.

.. code-block:: yaml

   # In src/osprey/config.yml
   framework:
     models:
       orchestrator:
         provider: cborg
         model_id: anthropic/claude-sonnet

       response:
         provider: cborg
         model_id: google/gemini-flash
         max_tokens: 5000

       classifier:
         provider: ollama
         model_id: mistral:7b

       approval:
         provider: ollama
         model_id: mistral:7b

       task_extraction:
         provider: cborg
         model_id: google/gemini-flash
         max_tokens: 1024

       memory:
         provider: cborg
         model_id: google/gemini-flash
         max_tokens: 256

       python_code_generator:
         provider: cborg
         model_id: anthropic/claude-haiku
         max_tokens: 4096

       time_parsing:
         provider: ollama
         model_id: mistral:7b
         max_tokens: 512

**Framework Model Roles:**

``orchestrator``
   Creates execution plans

``response``
   Generates final user responses

``classifier``
   Classifies tasks and selects capabilities

``approval``
   Analyzes code for approval decisions

``task_extraction``
   Extracts structured tasks from conversations

``memory``
   Processes memory storage/retrieval

``python_code_generator``
   Generates Python code

``time_parsing``
   Parses temporal references

**Model Configuration Fields:**

``provider`` (string, required)
   Provider name (must match ``api.providers`` key)

``model_id`` (string, required)
   Model identifier

   - Format varies by provider
   - Anthropic: ``anthropic/claude-sonnet``
   - OpenAI: ``openai/gpt-4``
   - Ollama: ``mistral:7b``

``max_tokens`` (integer, optional)
   Maximum output tokens

   - Not supported by all providers
   - Default varies by model

Application Models
------------------

**Type:** Object (nested)

**Location:** ``src/applications/{app}/config.yml``

**Purpose:** Application-specific model configurations.

.. code-block:: yaml

   # In src/applications/als_assistant/config.yml
   models:
     data_analysis:
       provider: cborg
       model_id: anthropic/claude-sonnet

     machine_operations:
       provider: cborg
       model_id: anthropic/claude-sonnet
       max_tokens: 4096

     data_visualization:
       provider: cborg
       model_id: anthropic/claude-sonnet
       max_tokens: 4096

     pv_finder:
       keyword:
         provider: cborg
         model_id: google/gemini-flash
         max_tokens: 4096
       query_splitter:
         provider: cborg
         model_id: google/gemini-flash
         max_tokens: 4096

**Details:**

- Application defines its own model names
- Same configuration format as framework models
- Can be deeply nested (e.g., ``pv_finder.keyword``)
- Merged with framework models (no conflicts)

Development Configuration
=========================

Development and debugging settings.

development.raise_raw_errors
----------------------------

**Type:** Boolean

**Location:** Root ``config.yml``

**Default:** ``false``

**Purpose:** Controls error handling behavior.

.. code-block:: yaml

   development:
     raise_raw_errors: false

**Values:**

- ``false`` (production) - Errors wrapped in ``ExecutionError`` with user-friendly messages
- ``true`` (development) - Raw exceptions raised with full stack traces

development.prompts
-------------------

**Type:** Object

**Location:** Root ``config.yml``

**Purpose:** Prompt debugging configuration.

.. code-block:: yaml

   development:
     prompts:
       show_all: false
       print_all: true
       latest_only: true

**Fields:**

``show_all`` (boolean)
   Print all prompts to console

   - ``true`` - Display prompts with formatting and separators
   - ``false`` - No console output

``print_all`` (boolean)
   Save all prompts to files

   - ``true`` - Save to ``file_paths.prompts_dir``
   - ``false`` - No file output

``latest_only`` (boolean)
   File naming strategy

   - ``true`` - Overwrite with latest (``{name}_latest.md``)
   - ``false`` - Timestamp each file (``{name}_YYYYMMDD_HHMMSS.md``)

**Use Cases:**

.. code-block:: yaml

   # Development: See everything
   development:
     prompts:
       show_all: true
       print_all: true
       latest_only: false

   # Production: Silent
   development:
     prompts:
       show_all: false
       print_all: false

Logging Configuration
=====================

Logging colors and output control.

logging.rich_tracebacks
-----------------------

**Type:** Boolean

**Location:** Root ``config.yml``

**Default:** ``false``

**Purpose:** Enable rich-formatted error tracebacks.

.. code-block:: yaml

   logging:
     rich_tracebacks: false
     show_traceback_locals: false
     show_full_paths: false

**Fields:**

``rich_tracebacks`` (boolean)
   Enable rich formatting

   - ``true`` - Colorized, formatted tracebacks
   - ``false`` - Standard Python tracebacks

``show_traceback_locals`` (boolean)
   Show local variables in tracebacks

   - ``true`` - Display all local variables
   - ``false`` - Variables hidden

``show_full_paths`` (boolean)
   Path display in tracebacks

   - ``true`` - Full absolute paths
   - ``false`` - Relative paths

logging Colors
--------------

**Type:** Object (nested)

**Location:** Framework or application ``config.yml``

**Purpose:** Customize component logging colors.

.. code-block:: yaml

   # Framework colors (src/osprey/config.yml)
   logging:
     framework:
       logging_colors:
         orchestrator: "cyan"
         classifier: "light_salmon1"
         task_extraction: "thistle1"
         python: "light_salmon1"
         respond: "thistle1"

     interface:
       logging_colors:
         cli: "deep_sky_blue1"
         pipeline: "deep_sky_blue1"

   # Application colors (src/applications/{app}/config.yml)
   logging:
     logging_colors:
       data_analysis: "deep_sky_blue1"
       data_visualization: "dark_turquoise"
       pv_address_finding: "dodger_blue2"

**Details:**

- Uses Rich library color names
- Framework component colors under ``logging.framework.logging_colors``
- Interface component colors under ``logging.interface.logging_colors``
- Application colors under ``logging.logging_colors``
- Colors used in console output and logs
- Falls back to white if not configured

**Available Colors:**

See `Rich color documentation <https://rich.readthedocs.io/en/stable/appendix/colors.html>`_ for full color list.

Service Configuration
=====================

Container service definitions.

Framework Services
------------------

**Location:** ``src/osprey/config.yml``

**Purpose:** Define framework infrastructure services.

.. code-block:: yaml

   framework:
     services:
       jupyter:
         path: ./services/osprey/jupyter
         containers:
           read:
             name: jupyter-read
             hostname: jupyter-read
             port_host: 8088
             port_container: 8088
             execution_modes: ["read_only"]
           write:
             name: jupyter-write
             hostname: jupyter-write
             port_host: 8089
             port_container: 8088
             execution_modes: ["write_access"]
         copy_src: true
         render_kernel_templates: true

       open_webui:
         path: ./services/osprey/open-webui
         hostname: appsdev2
         port_host: 8080
         port_container: 8080

       pipelines:
         path: ./services/osprey/pipelines
         port_host: 9099
         port_container: 9099
         copy_src: true
         additional_dirs:
           - interfaces

**Service Configuration Fields:**

``path`` (string, required)
   Directory containing Docker Compose template

``name`` (string, optional)
   Container name (defaults to service key)

``hostname`` (string, optional)
   Container hostname

``port_host`` (integer, optional)
   Host port mapping

``port_container`` (integer, optional)
   Container port

``copy_src`` (boolean, optional)
   Copy ``src/`` to container

   - Default: ``false``

``additional_dirs`` (list, optional)
   Extra directories to copy

``render_kernel_templates`` (boolean, optional)
   Process Jupyter kernel templates

   - Default: ``false``
   - Jupyter-specific

``containers`` (object, optional)
   Multiple container definitions

   - For services with multiple containers
   - Each container has same configuration fields

Application Services
--------------------

**Location:** ``src/applications/{app}/config.yml``

**Purpose:** Define application-specific services.

.. code-block:: yaml

   # In src/applications/als_assistant/config.yml
   services:
     mongo:
       name: mongo
       path: ./services/applications/als_assistant/mongo
       port_host: 27017
       port_container: 27017
       copy_src: true

     pv_finder:
       path: ./services/applications/als_assistant/pv_finder
       name: pv-finder
       port_host: 8051
       port_container: 8051
       copy_src: true

     langfuse:
       path: ./services/applications/als_assistant/langfuse
       name: langfuse
       copy_src: false

**Same configuration fields as framework services.**

Pipeline Configuration
======================

OpenWebUI pipeline settings.

pipeline.name
-------------

**Type:** String

**Location:** Framework or application ``config.yml``

**Purpose:** Display name for the pipeline in OpenWebUI.

.. code-block:: yaml

   # In src/applications/als_assistant/config.yml
   pipeline:
     name: "ALS Assistant"

**Details:**

- Shown in OpenWebUI interface
- Identifies the application/pipeline
- Framework default: ``"Generic AI Agent Framework"``

pipeline.startup_hooks
----------------------

**Type:** List of strings

**Location:** Application ``config.yml``

**Purpose:** Python functions to run at pipeline startup.

.. code-block:: yaml

   pipeline:
     startup_hooks:
       - "initialization.setup_nltk_resources"
       - "initialization.setup_system_packages"
       - "initialization.setup_pv_finder_resources"

**Details:**

- Functions are called during pipeline initialization
- Format: ``"module_path.function_name"``
- Module path is relative to application directory
- Used for downloading resources, initializing services, etc.

Configuration Examples
======================

Complete Configuration Example
------------------------------

.. code-block:: yaml

   # config.yml - Complete, self-contained configuration
   project_name: "my-agent"
   build_dir: ./build
   project_root: ${PROJECT_ROOT}
   registry_path: ./src/my_app/registry.py

   # Model configuration (8 specialized models)
   models:
     orchestrator:
       provider: cborg
       model_id: anthropic/claude-sonnet
     response:
       provider: cborg
       model_id: google/gemini-flash
     # ... other 6 models ...

   # Service deployment
   deployed_services:
     - jupyter
     - open_webui
     - pipelines

   # Safety controls
   approval:
     global_mode: "selective"
     capabilities:
       python_execution:
         enabled: true
         mode: "epics_writes"
       memory:
         enabled: true

   execution_control:
     epics:
       writes_enabled: false
     agent_control:
       task_extraction_bypass_enabled: false
       capability_selection_bypass_enabled: false
     limits:
       max_reclassifications: 1
       max_planning_attempts: 2
       max_step_retries: 3
       max_execution_time_seconds: 3000
       graph_recursion_limit: 100
       max_concurrent_classifications: 5

   system:
     timezone: ${TZ:-America/Los_Angeles}

   file_paths:
     agent_data_dir: _agent_data
     executed_python_scripts_dir: executed_scripts
     execution_plans_dir: execution_plans
     user_memory_dir: user_memory
     registry_exports_dir: registry_exports
     prompts_dir: prompts
     checkpoints: checkpoints

   deployed_services:
     - jupyter
     - open_webui
     - pipelines
     - mongo
     - pv_finder

   development:
     raise_raw_errors: false
     prompts:
       show_all: false
       print_all: true
       latest_only: true

   logging:
     rich_tracebacks: false
     show_traceback_locals: false
     show_full_paths: false

   api:
     providers:
       cborg:
         api_key: ${CBORG_API_KEY}
         base_url: https://api.cborg.lbl.gov/v1
         timeout: 30
       stanford:
         api_key: ${STANFORD_API_KEY}
         base_url: https://aiapi-prod.stanford.edu/v1
       anthropic:
         api_key: ${ANTHROPIC_API_KEY}
         base_url: https://api.anthropic.com
       openai:
         api_key: ${OPENAI_API_KEY}
         base_url: https://api.openai.com/v1
       ollama:
         api_key: ollama
         base_url: http://host.containers.internal:11434
         host: host.containers.internal
         port: 11434

Application Configuration Example
---------------------------------

.. code-block:: yaml

   # src/applications/als_assistant/config.yml

   file_paths:
     launcher_outputs_dir: launcher_outputs

   services:
     pv_finder:
       path: ./services/applications/als_assistant/pv_finder
       name: pv-finder
       port_host: 8051
       port_container: 8051
       copy_src: true

     mongo:
       name: mongo
       path: ./services/applications/als_assistant/mongo
       port_host: 27017
       port_container: 27017
       copy_src: true

   models:
     data_analysis:
       provider: cborg
       model_id: anthropic/claude-sonnet

     machine_operations:
       provider: cborg
       model_id: anthropic/claude-sonnet
       max_tokens: 4096

     pv_finder:
       keyword:
         provider: cborg
         model_id: google/gemini-flash
         max_tokens: 4096

   pipeline:
     name: "ALS Assistant"
     startup_hooks:
       - "initialization.setup_nltk_resources"
       - "initialization.setup_pv_finder_resources"

   logging:
     logging_colors:
       data_analysis: "deep_sky_blue1"
       data_visualization: "dark_turquoise"
       pv_address_finding: "dodger_blue2"

.. seealso::

   :doc:`../../developer-guides/03_core-framework-systems/06_configuration-architecture`
       Complete guide to configuration architecture, flat structure, and best practices

   :class:`osprey.state.StateManager`
       State management utilities that use configuration

   :doc:`02_state_and_context`
       State and context systems that depend on configuration

   :doc:`../../developer-guides/05_production-systems/05_container-and-deployment`
       Container deployment and service configuration